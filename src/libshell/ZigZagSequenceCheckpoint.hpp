#ifndef ZIGZAG_SEQUENCE_CHECKPOINT_HPP
#define ZIGZAG_SEQUENCE_CHECKPOINT_HPP

#include "ZigZagSequenceGrowth.hpp"
#include <fcntl.h>
#include <unistd.h>
#include <fstream>
#include <Eigen/SVD>
#include <array>
#include <cmath>
#include <limits>

// Accepted-repeat checkpoints only, not optimizer/continuation checkpoints.
// Reinitialize the identical mesh/recipe before loading; immutable data is
// compared before any live state is changed. Deliberately rejects edge clamps.
namespace zigzag_checkpoint
{
using Json = zigzag_sequence::json;

template<class Matrix> Json matrix(const Matrix& a)
{
    Json rows = Json::array();
    for(int i=0;i<a.rows();++i) {
        Json row=Json::array();
        for(int j=0;j<a.cols();++j) row.push_back(a(i,j));
        rows.push_back(row);
    }
    return rows;
}
inline Json forms(const tVecMat2d& a)
{
    Json result=Json::array();
    for(const auto& f:a) result.push_back(matrix(f));
    return result;
}
inline std::string checksum(const Json& a)
{
    // Accidental corruption check only. Runner supplies SHA-256 build/recipe identity.
    uint64_t h=14695981039346656037ull;
    for(unsigned char c:a.dump()){h^=c;h*=1099511628211ull;}
    return std::to_string(h);
}
inline void publish(const std::string& path,const Json& payload)
{
    const std::string bytes=Json({{"payload",payload},{"checksum",checksum(payload)}}).dump()+"\n";
    const std::string tmp=path+".tmp";
    int fd=::open(tmp.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
    if(fd<0) throw std::runtime_error("checkpoint temp exists or cannot be created");
    size_t done=0;
    while(done<bytes.size()) {
        const ssize_t n=::write(fd,bytes.data()+done,bytes.size()-done);
        if(n<=0){::close(fd);throw std::runtime_error("checkpoint write failed");}
        done+=n;
    }
    if(::fsync(fd)!=0){::close(fd);throw std::runtime_error("checkpoint fsync failed");}
    ::close(fd);
    if(::link(tmp.c_str(),path.c_str())!=0) throw std::runtime_error("checkpoint publication refused");
    const auto slash=path.find_last_of('/');
    const std::string dir=slash==std::string::npos?".":(slash==0?"/":path.substr(0,slash));
    fd=::open(dir.c_str(),O_RDONLY|O_DIRECTORY);
    if(fd<0) throw std::runtime_error("checkpoint directory open failed");
    const int rc=::fsync(fd);::close(fd);
    if(rc!=0) throw std::runtime_error("checkpoint directory fsync failed");
    ::unlink(tmp.c_str());
}
inline Json read(const std::string& path)
{
    std::ifstream f(path); if(!f) throw std::runtime_error("checkpoint missing");
    Json envelope; f>>envelope;
    f>>std::ws;
    if(!f.eof() || envelope.at("checksum")!=checksum(envelope.at("payload")))
        throw std::runtime_error("checkpoint corrupt or trailing data");
    return envelope.at("payload");
}
template<class Mesh> void requireVertexClampsOnly(const Mesh& mesh)
{
    const auto& bc=mesh.getBoundaryConditions();
    if(bc.getEdgeBoundaryConditions().count()!=0 ||
       bc.getClampedEdgeNormals().size()!=0 || !bc.getClampedEdgeNormalsIndices().empty())
        throw std::runtime_error("sequence checkpoint does not support edge/normal clamps");
}
template<class Mesh> Json immutable(const Mesh& mesh,const Eigen::MatrixXd& uv,
                                    const Json& recipe,const std::string& identity)
{
    requireVertexClampsOnly(mesh);
    if(identity.size()!=64) throw std::runtime_error("checkpoint requires runner SHA-256 identity");
    return {{"runner_identity",identity},{"recipe",recipe},
        {"rest",matrix(mesh.getRestConfiguration().getPositions())},
        {"faces",matrix(mesh.getTopology().getFace2Vertices())},
        {"edges",matrix(mesh.getTopology().getEdge2Vertices())},
        {"uv",matrix(uv)}, {"bbar",forms(mesh.getRestConfiguration().getSecondFundamentalForms())},
        {"vertex_mask",matrix(mesh.getBoundaryConditions().getVertexBoundaryConditions())},
        {"edge_mask",matrix(mesh.getBoundaryConditions().getEdgeBoundaryConditions())}};
}
template<class Mesh> Json capture(const Mesh& mesh,const Eigen::VectorXi& counts,
                                  int cycle,const Json& proof,const Json& identity)
{
    requireVertexClampsOnly(mesh);
    if(!proof.at("accepted").get<bool>() || proof.at("termination")!="gradient_target" ||
       proof.at("curvature")!="positive_pivots")
        throw std::runtime_error("cannot checkpoint unaccepted repeat");
    if(matrix(mesh.getBoundaryConditions().getVertexBoundaryConditions())!=identity.at("vertex_mask") ||
       !mesh.getCurrentConfiguration().getPositions().allFinite() || counts.minCoeff()<0)
        throw std::runtime_error("checkpoint capture requires finite clamped state/history");
    return {{"schema",1},{"identity",identity},{"cycle",cycle},{"proof",proof},
        {"state",matrix(mesh.getCurrentConfiguration().getPositions())},
        {"top",forms(mesh.getRestConfiguration().template getFirstFundamentalForms<top>())},
        {"bottom",forms(mesh.getRestConfiguration().template getFirstFundamentalForms<bottom>())},
        {"counts",matrix(counts)}};
}
inline Json gaugeGeometry(const Eigen::MatrixXd& x,const std::array<int,3>& ids)
{
    Eigen::Vector3d b=(x.row(ids[1])-x.row(ids[0])).transpose();
    Eigen::Vector3d c=(x.row(ids[2])-x.row(ids[0])).transpose();
    const double scale=std::max(b.norm(),c.norm());
    if(!std::isfinite(scale) || scale<=0) throw std::runtime_error("degenerate gauge geometry");
    b/=scale;c/=scale;
    Eigen::Matrix3d m;
    m << -b.z(),0,b.x(), b.y(),-b.x(),0, c.y(),-c.x(),0;
    const Eigen::Vector3d singular=m.jacobiSvd().singularValues();
    const double ratio=singular[0]>0?singular[2]/singular[0]:0;
    return {{"normalized_rotation_constraint_singular_values",matrix(singular)},
        {"minimum_maximum_ratio",ratio},{"length_scale_m",scale},
        {"full_rank_at_relative_1e_10",ratio>1e-10}};
}
inline Eigen::MatrixXd decode(const Json& j,int rows,int cols)
{
    if(!j.is_array() || j.size()!=size_t(rows)) throw std::runtime_error("checkpoint row mismatch");
    Eigen::MatrixXd a(rows,cols);
    for(int i=0;i<rows;++i) {
        if(!j[i].is_array() || j[i].size()!=size_t(cols)) throw std::runtime_error("checkpoint column mismatch");
        for(int k=0;k<cols;++k) {
            if(!j[i][k].is_number()) throw std::runtime_error("checkpoint nonnumeric value");
            a(i,k)=j[i][k].get<double>();
        }
    }
    if(!a.allFinite()) throw std::runtime_error("checkpoint nonfinite value");
    return a;
}
inline tVecMat2d decodeForms(const Json& j,int n)
{
    if(!j.is_array() || j.size()!=size_t(n)) throw std::runtime_error("checkpoint face mismatch");
    tVecMat2d result(n);
    for(int i=0;i<n;++i) {
        result[i]=decode(j[i],2,2);
        const auto& a=result[i];
        if(std::abs(a(0,1)-a(1,0))>1e-12*a.norm() || a(0,0)<=0 || a.determinant()<=0)
            throw std::runtime_error("checkpoint invalid metric");
    }
    return result;
}
template<class Mesh> int restore(Mesh& mesh,Eigen::VectorXi& counts,const Json& saved,
                                 const Json& identity,int totalCycles,double tolerance)
{
    requireVertexClampsOnly(mesh);
    if(saved.at("schema")!=1 || saved.at("identity")!=identity)
        throw std::runtime_error("checkpoint identity mismatch");
    if(!saved.at("cycle").is_number_integer()) throw std::runtime_error("checkpoint noninteger cycle");
    const int cycle=saved.at("cycle").get<int>();
    const auto& proof=saved.at("proof");
    const double g=proof.at("gradient_norm").get<double>();
    const double target=proof.at("gradient_tolerance").get<double>();
    if(cycle<1 || cycle>totalCycles || !proof.at("accepted").get<bool>() ||
       proof.at("termination")!="gradient_target" || proof.at("curvature")!="positive_pivots" ||
       !std::isfinite(g) || !std::isfinite(target) || g<0 || target<=0 || target>tolerance || g>target)
        throw std::runtime_error("checkpoint acceptance/cursor mismatch");
    const int nv=mesh.getNumberOfVertices(), nf=mesh.getNumberOfFaces();
    const Eigen::VectorXd state=decode(saved.at("state"),3*nv+mesh.getNumberOfEdges(),1);
    const auto topForms=decodeForms(saved.at("top"),nf), botForms=decodeForms(saved.at("bottom"),nf);
    const auto rawCounts=decode(saved.at("counts"),nf,1);
    Eigen::VectorXi restoredCounts(nf);
    for(int i=0;i<nf;++i) {
        if(rawCounts(i,0)<0 || rawCounts(i,0)>std::numeric_limits<int>::max() ||
           rawCounts(i,0)!=std::floor(rawCounts(i,0))) throw std::runtime_error("checkpoint invalid history");
        restoredCounts[i]=int(rawCounts(i,0));
    }
    // Compare static geometry to this invocation before mutating targets/state.
    if(matrix(mesh.getRestConfiguration().getPositions())!=identity.at("rest") ||
       matrix(mesh.getTopology().getFace2Vertices())!=identity.at("faces") ||
       matrix(mesh.getTopology().getEdge2Vertices())!=identity.at("edges") ||
       forms(mesh.getRestConfiguration().getSecondFundamentalForms())!=identity.at("bbar"))
        throw std::runtime_error("checkpoint immutable mesh mismatch");
    Eigen::MatrixXb mask(nv,3);
    for(int i=0;i<nv;++i) for(int k=0;k<3;++k)
        mask(i,k)=identity.at("vertex_mask").at(i).at(k).get<bool>();
    // Fixed gripper coordinates must equal the original reference coordinates.
    for(int i=0;i<nv;++i) for(int k=0;k<3;++k)
        if(mask(i,k) && state[k*nv+i]!=mesh.getRestConfiguration().getVertices()(i,k))
            throw std::runtime_error("checkpoint changed clamped coordinate");
    mesh.getRestConfiguration().template getFirstFundamentalForms<top>()=topForms;
    mesh.getRestConfiguration().template getFirstFundamentalForms<bottom>()=botForms;
    mesh.getCurrentConfiguration().getPositions()=state;
    mesh.getBoundaryConditions().getVertexBoundaryConditions()=mask;
    counts=restoredCounts;
    mesh.updateDeformedConfiguration();
    return cycle;
}
}
#endif

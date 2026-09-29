#ifndef SHELL_BENCHMARK_PROBLEM_HPP
#define SHELL_BENCHMARK_PROBLEM_HPP
#include "NonconvexBenchmark.hpp"
#include "TinyADHessian_Bilayer.hpp"
#include <Eigen/QR>
#include <Eigen/SVD>
#include <memory>

namespace NonconvexBenchmark
{
// Caller owns a fixed material/target/geometry bundle. Its independently verified
// content identity must be supplied; an arbitrary label is insufficient for a
// production restart. Benchmark engines do not mutate the rest targets.
template<class Mesh,class Energy>
class ShellProblem final : public Problem
{
    Mesh& mesh;
    Energy& energy;
    TinyADHessian_Bilayer<Mesh> ad;
    std::string id;
    Vec scale, fixed, restArea;
    std::vector<char> mask;
    double reference;
public:
    ShellProblem(Mesh& m,Energy& e,double young,double nu,double h,std::string identity,
                 std::string assembly="triplets",int assemblyThreads=1)
        :mesh(m),energy(e),ad(young,nu,h,assemblyThreads),id(std::move(identity))
    {
        if(assemblyThreads<1 || (assemblyThreads!=1 && assembly!="local_dihedral_csc"))
            throw std::invalid_argument("parallel shell assembly requires local-dihedral Hessian and positive threads");
        if(assembly!="triplets" && assembly!="cached_csc" && assembly!="local_dihedral_csc")
            throw std::invalid_argument("unknown shell Hessian assembly");
        const int n=size(),nv=mesh.getNumberOfVertices(),nf=mesh.getNumberOfFaces();
        fixed=Eigen::Map<const Vec>(mesh.getDataPointer(),n);
        mask=ShellEquilibrium::constrainedDofMask(mesh);
        const auto X=mesh.getRestConfiguration().getVertices();const auto F=mesh.getTopology().getFace2Vertices();
        restArea.resize(nf);double area=0;
        for(int f=0;f<nf;++f)
        {
            const Eigen::Vector3d a=(X.row(F(f,1))-X.row(F(f,0))).transpose();
            const Eigen::Vector3d b=(X.row(F(f,2))-X.row(F(f,0))).transpose();
            restArea[f]=0.5*a.cross(b).norm();area+=restArea[f];
        }
        if(!restArea.allFinite() || restArea.minCoeff()<=0 || young<=0 || h<=0 || !std::isfinite(young*h*area))
            throw std::invalid_argument("invalid shell reference geometry/material");
        scale=Vec::Ones(n);scale.head(3*nv).setConstant(std::sqrt(area));reference=young*h*area;
    }
    int size() const override{return 3*mesh.getNumberOfVertices()+mesh.getNumberOfEdges();}
    std::string identity() const override{return id;}
    Vec scales() const override{return scale;}
    double energyReference() const override{return reference;}
    void restore(const Vec& x) override
    {
        if(x.size()!=size())throw std::invalid_argument("shell state size mismatch");
        Eigen::Map<Vec>(mesh.getDataPointer(),size())=x;mesh.updateDeformedConfiguration();
    }
    Value evaluate(const Vec& x) override
    {
        if(!x.allFinite())throw std::runtime_error("nonfinite shell state");
        for(int i=0;i<size();++i)if(mask[i] && x[i]!=fixed[i])throw std::runtime_error("physical constraint violation");
        restore(x);const auto V=mesh.getCurrentConfiguration().getVertices();const auto F=mesh.getTopology().getFace2Vertices();
        for(int f=0;f<mesh.getNumberOfFaces();++f)
        {
            const Eigen::Vector3d a=(V.row(F(f,1))-V.row(F(f,0))).transpose();
            const Eigen::Vector3d b=(V.row(F(f,2))-V.row(F(f,0))).transpose();
            const double ratio=0.5*a.cross(b).norm()/restArea[f];
            if(!std::isfinite(ratio) || ratio<=1e-8)throw std::runtime_error("degenerate shell trial");
        }
        Vec g=Vec::Zero(size());const double f=energy.compute(mesh,g);return {f,g};
    }
    Sparse hessian(const Vec& x) override
    {
        restore(x);
        Sparse H = ad.assembleHessian(mesh);
        if(ad.lastAssemblyThreads() != ad.assemblyThreads())
            throw std::runtime_error("requested Hessian OpenMP team unavailable");
        return H;
    }
    int lastHessianThreads() const
    {
        return ad.lastAssemblyThreads();
    }
    Vec adGradient(const Vec& x){restore(x);return ad.gradient(mesh);}
    int hessianCacheBuilds() const
    {
        return ad.cacheBuilds();
    }
    using Problem::freeDofs;
    std::vector<int> freeDofs(const Vec& x,const Vec& S) const override
    {
        return freeDofs(x,S,{});
    }
    std::vector<int> freeDofs(const Vec& x,const Vec& S,const std::vector<int>& hint) const override
    {
        // Construct the gauge from the explicit base state, never a lingering trial.
        if(x.size()!=size() || S.size()!=size() || !S.allFinite() || S.minCoeff()<=0)
            throw std::invalid_argument("invalid gauge scales");
        const ShellEquilibrium::RigidModeProjector P(
            Eigen::Map<const Eigen::MatrixXd>(x.data(),mesh.getNumberOfVertices(),3),
            mesh.getNumberOfEdges(),mask);
        std::vector<char> fixedMask=mask;
        const int k=P.numberOfRigidModes();
        if(k)
        {
            const Eigen::MatrixXd R=S.cwiseInverse().asDiagonal()*P.basis();
            bool reusedHint=false;
            int nUnconstrained=0;for(int i=0;i<size();++i)if(!mask[i])++nUnconstrained;
            if(!hint.empty() && static_cast<int>(hint.size())==nUnconstrained-k)
            {
                std::vector<char> isFree(size(),0);
                for(int id:hint)if(id>=0 && id<size())isFree[id]=1;
                std::vector<int> candidateGauge;candidateGauge.reserve(k);
                for(int i=0;i<size();++i)if(!mask[i] && !isFree[i])candidateGauge.push_back(i);
                if(static_cast<int>(candidateGauge.size())==k)
                {
                    Eigen::MatrixXd sub(k,k);
                    for(int j=0;j<k;++j)sub.row(j)=R.row(candidateGauge[j]);
                    Eigen::JacobiSVD<Eigen::MatrixXd> svd(sub);
                    const auto& sv=svd.singularValues();
                    if(sv(k-1)>1e-10)
                    {
                        const double cond=sv(0)/sv(k-1);
                        if(cond<100.0)
                        {
                            for(int g:candidateGauge)fixedMask[g]=1;
                            reusedHint=true;
                        }
                    }
                }
            }
            if(!reusedHint)
            {
                Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(R.transpose());qr.setThreshold(1e-10);
                if(qr.rank()!=k)throw std::runtime_error("rank deficient rigid gauge");
                for(int j=0;j<k;++j)
                {
                    const int i=qr.colsPermutation().indices()[j];
                    if(fixedMask[i])throw std::runtime_error("gauge selected physical constraint");
                    fixedMask[i]=1;
                }
            }
        }
        std::vector<int> ids;for(int i=0;i<size();++i)if(!fixedMask[i])ids.push_back(i);return ids;
    }
};
}
#endif

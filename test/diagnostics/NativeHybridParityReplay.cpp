// Fixed-case qualification driver: numerical work calls the production API.
// Bundle loading and artifact framing are independent of the solver implementation.
#include "gtest/gtest.h"
#include "NativeHybridShellSolver.hpp"
#include "Geometry.hpp"
#include "Mesh.hpp"
#include "MaterialProperties.hpp"
#include "CombinedOperator_Parametric.hpp"
#include "EnergyOperatorList.hpp"
#include <filesystem>
#include <fstream>
#include <cstring>
#include <cstdlib>
// This qualification build reuses a project archive compiled with AVX-512.
static_assert(EIGEN_DEFAULT_ALIGN_BYTES==64,"retained project archive requires 64-byte Eigen alignment");
#include <fcntl.h>
#include <unistd.h>

namespace NativeHybridParity {
using namespace ShellNativeHybrid;
namespace fs=std::filesystem;
void need(bool b,const char* message){if(!b)throw std::runtime_error(message);}
Json read(const fs::path& file){std::ifstream f(file);need(bool(f),"JSON input");Json j;f>>j;return j;}
void atomic(const fs::path& file,const Json& value){
    const std::string tmp=file.string()+".pending",bytes=value.dump()+"\n";
    const int fd=::open(tmp.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);need(fd>=0,"output temporary exists");
    size_t n=0;while(n<bytes.size()){const auto k=::write(fd,bytes.data()+n,bytes.size()-n);
        if(k<=0){::close(fd);throw std::runtime_error("artifact write");}n+=size_t(k);}
    const int sync=::fsync(fd),closed=::close(fd);need(sync==0 && closed==0,"artifact sync");
    need(::rename(tmp.c_str(),file.c_str())==0,"artifact rename");
    const int dir=::open(file.parent_path().c_str(),O_RDONLY|O_DIRECTORY);need(dir>=0,"artifact directory");
    const int ds=::fsync(dir),dc=::close(dir);need(ds==0 && dc==0,"artifact directory sync");
}
Vec vector(const fs::path& file,int n){
    std::ifstream f(file,std::ios::binary|std::ios::ate);
    need(bool(f) && f.tellg()==std::streamoff(n*sizeof(double)),"bundle vector size");
    f.seekg(0);Vec x(n);f.read(reinterpret_cast<char*>(x.data()),n*sizeof(double));
    need(bool(f) && x.allFinite(),"bundle vector values");return x;
}
template<class T>void equalFile(const fs::path& file,const T* p,size_t n){
    std::ifstream f(file,std::ios::binary|std::ios::ate);
    need(bool(f) && f.tellg()==std::streamoff(n*sizeof(T)),"bundle reconstruction size");
    f.seekg(0);std::vector<char> bytes(n*sizeof(T));f.read(bytes.data(),bytes.size());
    need(bool(f) && std::memcmp(bytes.data(),p,bytes.size())==0,"bundle reconstruction bytes");
}
void raw(const fs::path& file,const Vec& x){std::ofstream f(file,std::ios::binary);
    f.write(reinterpret_cast<const char*>(x.data()),x.size()*sizeof(double));need(bool(f),"raw output");}
void forms(const fs::path& file,tVecMat2d& a){const auto x=vector(file,int(4*a.size()));
    for(size_t f=0;f<a.size();++f)for(int i=0;i<2;++i)for(int j=0;j<2;++j)a[f](i,j)=x[4*f+2*i+j];}

void worker(const Json& request){
    const double entry=Engine::wallClock();auto elapsed=[&]{return Engine::wallClock()-entry;};
    need(request.at("recipe")=="hybrid" && request.at("start")=="cylinder_y_plus" &&
         request.at("toy")==false && request.at("resume")==false &&
         request.at("switch_attempts")==2000 && request.at("hessian_threads")==4 &&
         request.at("guarded_hessian")==false && request.at("reuse_symbolic")==false,"fixed parity controls");
    const fs::path bundle=request.at("bundle").get<std::string>(),out=request.at("out").get<std::string>();
    need(fs::is_directory(out) && fs::is_empty(out),"fresh parity output required");
    BilayerMesh mesh;RectangularPlate_RightAngle geo(.127,.1524,2*.127*.015,false,false);geo.setQuiet();mesh.init(geo);
    need(mesh.getNumberOfVertices()==5427 && mesh.getNumberOfFaces()==10560,"fixed parity mesh");
    const auto X=mesh.getRestConfiguration().getVertices();const auto F=mesh.getTopology().getFace2Vertices();
    const auto E=mesh.getTopology().getEdge2Vertices();const auto mask=ShellEquilibrium::constrainedDofMask(mesh);
    equalFile(bundle/"reference.f64",X.data(),X.size());equalFile(bundle/"faces.i32",F.data(),F.size());
    equalFile(bundle/"edges.i32",E.data(),E.size());equalFile(bundle/"constraints.u8",mask.data(),mask.size());
    forms(bundle/"top.f64",mesh.getRestConfiguration().getFirstFundamentalForms<top>());
    forms(bundle/"bottom.f64",mesh.getRestConfiguration().getFirstFundamentalForms<bottom>());
    forms(bundle/"bbar.f64",mesh.getRestConfiguration().getSecondFundamentalForms());
    MaterialProperties_Iso_Constant material(1,.33,.0006);
    CombinedOperator_Parametric<BilayerMesh,Material_Isotropic,bottom> lower(material);
    CombinedOperator_Parametric<BilayerMesh,Material_Isotropic,top> upper(material);
    EnergyOperatorList<BilayerMesh> energy({&lower,&upper});
    HybridOptions options;options.budgetStart=entry;options.hessianThreads=4;options.problemIdentity=request.at("identity");
    options.implementationIdentity=request.at("implementation");options.dependencyIdentity=request.at("dependency_identity");
    const auto configs=hybridConfigs(options);const Config& lb=configs.first;const Config& nt=configs.second;
    // This independent observer supplies immutable checkpoint metadata only;
    // evaluate/step/hessian/endpoint verification are performed by the public API.
    ShellProblem<BilayerMesh,decltype(energy)> observer(mesh,energy,1,.33,.0006,options.problemIdentity,"local_dihedral_csc",4);
    const Vec initial=vector(bundle/"cylinder_y_plus.f64",observer.size());observer.restore(initial);
    Json policy={{"schema",1},{"recipe","hybrid"},{"switch_attempts",2000},{"toy",false},
        {"start","cylinder_y_plus"},{"identity",options.problemIdentity},{"lbfgs",lb.json()},{"newton",nt.json()}};
    int generation=0;double lastSave=0;bool switchPublished=false;
    auto journal=[&](const std::string& name,const std::string& phase){atomic(out/"journal.json",
        {{"policy",policy},{"problem",observer.identity()},{"phase",phase},{"active",name},
         {"generation",generation},{"charged_elapsed",elapsed()}});};
    auto persist=[&](State s,const Config& c,const std::string& phase){
        s.elapsed=std::max(s.elapsed,elapsed());const std::string name="checkpoint_"+std::to_string(generation++)+".json";
        saveCheckpoint((out/name).string(),s,c,observer);journal(name,phase);lastSave=elapsed();return name;};
    auto boundary=[&](HybridSession& session){
        if(session.transitioned() && !switchPublished){
            // Publish the exact pre-transform source and fresh target, not a
            // reconstruction from a resumed/cleared quasi-Newton state.
            const std::string source="checkpoint_"+std::to_string(generation++)+".json";
            saveCheckpoint((out/source).string(),session.source(),lb,observer);
            saveCheckpoint((out/"switch_target.json").string(),session.target(),nt,observer);
            switchPublished=true;journal("switch_target.json","newton");lastSave=elapsed();
        }else if(generation==0 || elapsed()-lastSave>=30){persist(session.snapshot(),session.config(),session.phase());}
    };
    const auto report=solveNativeHybridShell(mesh,energy,1,.33,.0006,options,boundary);
    const std::string checkpoint=persist(report.terminal,report.phase=="newton"?nt:lb,report.phase);
    const auto& s=report.terminal;raw(out/"final_x.f64",s.x);raw(out/"final_g.f64",s.g);raw(out/"selected_x.f64",report.selected);
    Json result={{"recipe","hybrid"},{"phase",report.phase},{"status",s.status},{"energy",s.energy},
        {"gradient_norm",s.g.norm()},{"attempts",s.attempts},{"iterations",s.iterations},
        {"evaluations",s.evaluations},{"hessians",s.hessians},{"hvps",s.hvps},{"checkpoint",(out/checkpoint).string()},
        {"optimizer_elapsed",s.elapsed},{"elapsed",elapsed()},{"accepted",report.accepted},{"curvature",report.curvature},
        {"verification_evaluations",report.verificationEvaluations},{"verification_hessians",report.verificationHessians},
        {"implementation_path","ShellNativeHybrid::solveNativeHybridShell"},{"handoff_demonstrated",report.transitioned}};
    if(report.verificationEvaluations){result["selected_energy"]=report.energy;result["selected_gradient_norm"]=report.gradientNorm;
        result["native_gate_met"]=report.gradientNorm<=options.physicalGate;}
    result["best_residual_candidate_energy"]=s.bestValid?Json(s.bestEnergy):Json(nullptr);
    atomic(out/"result.json",result);atomic(out/"completed.json",{{"completed",true},{"status",s.status},{"elapsed",elapsed()}});
}
}
// Keep the established runner's selected-test framing; the implementation path
// above, source identity and binary identity distinguish this production replay.
TEST(ConvergenceBenchmark,Dispatch){
    const char* name=std::getenv("CONVERGENCE_REQUEST");ASSERT_NE(name,nullptr);
    NativeHybridParity::worker(NativeHybridParity::read(name));
}

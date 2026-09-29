#ifndef NONCONVEX_BENCHMARK_HPP
#define NONCONVEX_BENCHMARK_HPP

// Isolated benchmark engines. Production HLBFGS and shell solve paths are untouched.
// Native recurrence follows external/hlbfgs/HLBFGS.cpp (Yang Liu, 2009-2010);
// upstream kernels and their non-commercial license/attribution are retained.
// Sequential use only: upstream MCSRCH uses static scratch variables.
#include "ShellEquilibriumSolver.hpp"
#include "BenchmarkNewtonFactorization.hpp"
#include "HLBFGS.h"
#include "LineSearch.h"
#include <Eigen/SparseCholesky>
#include <nlohmann/json.hpp>
#include <chrono>
#include <functional>
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

namespace NonconvexBenchmark
{
using Vec=Eigen::VectorXd;
using Sparse=Eigen::SparseMatrix<double>;
using Json=nlohmann::json;

struct Value { double energy; Vec gradient; };
struct Problem
{
    virtual ~Problem()=default;
    virtual int size() const=0;
    // Must identify immutable geometry, targets, constraints, material and oracle version.
    virtual std::string identity() const=0;
    // A successful evaluation leaves any mutable working geometry at x.
    virtual Value evaluate(const Vec& x)=0;
    virtual Sparse hessian(const Vec& x)=0;
    virtual Vec scales() const { return Vec::Ones(size()); }
    virtual double energyReference() const { return 1; }
    virtual std::vector<int> freeDofs(const Vec&,const Vec&) const
    { std::vector<int> ids(size()); std::iota(ids.begin(),ids.end(),0); return ids; }
    virtual std::vector<int> freeDofs(const Vec& x,const Vec& S,const std::vector<int>& /*hint*/) const
    { return freeDofs(x,S); }
    // Restore physical state/cache WITHOUT another counted energy evaluation.
    virtual void restore(const Vec&) {}
};

struct Config
{
    std::string method="native_lbfgs", implementationId="nonconvex-core-v8-local-dihedral-optin";
    std::string newtonBackend="eigen",dependencyIdentity,hessianAssembly="triplets";
    bool reuseNewtonSymbolic=false, guardedHessianReuse=false;
    int hessianThreads=1; // Explicit benchmark-only local-dihedral assembly team.
    int linearSolverThreads=1; // Explicit benchmark-only Newton/CHOLMOD linear solver team.
    int memory=10, maxAttempts=10000, maxEvaluations=100000, maxHessians=10000;
    int maxLineEvaluations=30, maxShiftAttempts=12, maxCG=250;
    double seconds=270, tolerance=5e-14, physicalGate=1e-13;
    double c1=1e-4, cgC2=0.1, pivotFloor=1e-12, linearTolerance=1e-8;
    double initialRadius=0.1, minRadius=1e-12, maxRadius=10, trAcceptance=0.1;
    double cgForcing=0.1, cgAbsolute=1e-12;
    Json json() const
    { Json j={{"method",method},{"implementationId",implementationId},{"newtonBackend",newtonBackend},{"dependencyIdentity",dependencyIdentity},{"hessianAssembly",hessianAssembly},{"memory",memory},{"maxAttempts",maxAttempts},
        {"maxEvaluations",maxEvaluations},{"maxHessians",maxHessians},
        {"maxLineEvaluations",maxLineEvaluations},{"maxShiftAttempts",maxShiftAttempts},{"maxCG",maxCG},
        {"seconds",seconds},{"tolerance",tolerance},{"physicalGate",physicalGate},{"c1",c1},{"cgC2",cgC2},
        {"pivotFloor",pivotFloor},{"linearTolerance",linearTolerance},{"initialRadius",initialRadius},
        {"minRadius",minRadius},{"maxRadius",maxRadius},{"trAcceptance",trAcceptance},
        {"cgForcing",cgForcing},{"cgAbsolute",cgAbsolute}};
      if(reuseNewtonSymbolic)j["reuseNewtonSymbolic"]=true;
      if(guardedHessianReuse)j["guardedHessianReuse"]=true;
      if(hessianThreads!=1)j["hessianThreads"]=hessianThreads;
      if(linearSolverThreads!=1)j["linearSolverThreads"]=linearSolverThreads;
      return j; }
    void validate() const
    {
        if(implementationId.empty())throw std::invalid_argument("missing implementation identity");
        if(linearSolverThreads<1)throw std::invalid_argument("linearSolverThreads must be positive");
        if(hessianThreads<1 || (hessianThreads!=1 && hessianAssembly!="local_dihedral_csc"))
            throw std::invalid_argument("parallel assembly requires local-dihedral Hessian and positive threads");
#ifndef _OPENMP
        if(hessianThreads!=1)throw std::invalid_argument("parallel Hessian requires OpenMP");
#endif
        if((reuseNewtonSymbolic || guardedHessianReuse) && method!="sparse_newton")throw std::invalid_argument("reuse requires Newton");
        if(hessianAssembly!="triplets" && hessianAssembly!="cached_csc" && hessianAssembly!="local_dihedral_csc")
            throw std::invalid_argument("unknown Hessian assembly");
#ifndef NONCONVEX_TEST_CACHED_TINYAD
        if(hessianAssembly=="cached_csc")throw std::invalid_argument("cached Hessian assembly not compiled into benchmark");
#endif
#ifndef NONCONVEX_TEST_LOCAL_DIHEDRAL
        if(hessianAssembly=="local_dihedral_csc")throw std::invalid_argument("local-dihedral Hessian not compiled into benchmark");
#endif
        if(newtonBackend!="eigen" && newtonBackend!="cholmod")throw std::invalid_argument("unknown Newton backend");
        if(newtonBackend=="cholmod"){
#ifndef NONCONVEX_WITH_CHOLMOD
            throw std::invalid_argument("CHOLMOD not compiled into benchmark");
#endif
            if(dependencyIdentity.empty())throw std::invalid_argument("missing runtime dependency identity");
        }
        if(method!="native_lbfgs" && method!="nonlinear_cg" && method!="sparse_newton" && method!="trust_region_cg")
            throw std::invalid_argument("unknown benchmark method");
        if(memory<1 || maxAttempts<1 || maxEvaluations<1 || maxHessians<1 || maxLineEvaluations<1 ||
           maxShiftAttempts<1 || maxCG<1) throw std::invalid_argument("invalid integer budget");
        for(double v:{seconds,tolerance,physicalGate,c1,cgC2,pivotFloor,linearTolerance,
                      initialRadius,minRadius,maxRadius,trAcceptance,cgForcing,cgAbsolute})
            if(!std::isfinite(v) || v<=0) throw std::invalid_argument("invalid positive control");
        if(tolerance>physicalGate || c1>=cgC2 || cgC2>=0.5 || initialRadius<minRadius ||
           initialRadius>maxRadius || trAcceptance>=0.25 || cgForcing>=1)
            throw std::invalid_argument("inconsistent solver controls");
        if(method=="native_lbfgs" && (memory!=10 || c1!=1e-4))
            throw std::invalid_argument("native reference profile is frozen");
    }
};

inline Json pack(const Vec& v)
{ std::vector<double> a(v.size()); for(int i=0;i<v.size();++i)a[i]=v[i]; return a; }
inline Vec unpack(const Json& j)
{
    const auto a=j.get<std::vector<double>>(); Vec v(a.size());
    for(size_t i=0;i<a.size();++i)v[i]=a[i];
    if(!v.allFinite()) throw std::invalid_argument("nonfinite checkpoint vector");
    return v;
}
inline Json packSparse(Sparse A)
{
    A.makeCompressed();
    std::vector<double> v(A.nonZeros()); std::vector<int> inner(A.nonZeros()),outer(A.cols()+1);
    for(int i=0;i<A.nonZeros();++i){v[i]=A.valuePtr()[i];inner[i]=A.innerIndexPtr()[i];}
    for(int i=0;i<=A.cols();++i)outer[i]=A.outerIndexPtr()?A.outerIndexPtr()[i]:0;
    return {{"rows",A.rows()},{"cols",A.cols()},{"values",v},{"inner",inner},{"outer",outer}};
}
inline Sparse unpackSparse(const Json& j)
{
    const int n=j.at("rows"),m=j.at("cols");
    const auto v=j.at("values").get<std::vector<double>>();
    const auto inner=j.at("inner").get<std::vector<int>>(),outer=j.at("outer").get<std::vector<int>>();
    if(n<0 || m<0 || v.size()!=inner.size() || outer.size()!=size_t(m+1) ||
       outer.front()!=0 || outer.back()!=int(v.size())) throw std::invalid_argument("invalid CSC checkpoint");
    std::vector<Eigen::Triplet<double>> t; t.reserve(v.size());
    for(int c=0;c<m;++c)
    {
        if(outer[c]<0 || outer[c]>outer[c+1] || outer[c+1]>int(v.size())) throw std::invalid_argument("bad CSC offsets");
        int previous=-1;
        for(int k=outer[c];k<outer[c+1];++k)
        {
            if(inner[k]<=previous || inner[k]>=n || !std::isfinite(v[k])) throw std::invalid_argument("bad CSC entry");
            previous=inner[k]; t.emplace_back(inner[k],c,v[k]);
        }
    }
    Sparse A(n,m); A.setFromTriplets(t.begin(),t.end()); return A;
}

struct State
{
    std::string status="ready", error, problemId, recipeId, scaleId;
    bool initialized=false, bestValid=false, cacheValid=false, boundary=true;
    int attempts=0, iterations=0, evaluations=0, hessians=0, hvps=0, ring=0;
    int hessianReuseAge=0; // At a durable boundary: 1 means one remaining stale update.
    double elapsed=0, energy=0, previousEnergy=0, bestEnergy=0, radius=0.1, lastAlpha=1, lastShift=0;
    Vec x,g,previousX,previousG,cgGradient,cgDirection,bestX;
    std::vector<double> s,y,rho,diagonal;
    std::vector<int> free;
    Sparse A;
    Json trace=Json::array();
    Json json() const
    {
        Json j={{"status",status},{"error",error},{"problemId",problemId},{"recipeId",recipeId},{"scaleId",scaleId},
            {"initialized",initialized},{"bestValid",bestValid},{"boundary",boundary},
            {"cacheValid",cacheValid},{"attempts",attempts},{"iterations",iterations},{"evaluations",evaluations},
            {"hessians",hessians},{"hvps",hvps},{"ring",ring},{"elapsed",elapsed},{"energy",energy},
            {"previousEnergy",previousEnergy},{"bestEnergy",bestEnergy},{"radius",radius},
            {"lastAlpha",lastAlpha},{"lastShift",lastShift},{"x",pack(x)},{"g",pack(g)},
            {"previousX",pack(previousX)},{"previousG",pack(previousG)},
            {"cgGradient",pack(cgGradient)},{"cgDirection",pack(cgDirection)},{"bestX",pack(bestX)},
            {"s",s},{"y",y},{"rho",rho},{"diagonal",diagonal},{"free",free},
            {"A",packSparse(A)},{"trace",trace}};
        if(hessianReuseAge)j["hessianReuseAge"]=hessianReuseAge;
        return j;
    }
    static State fromJson(const Json& j,int n,const Config& c)
    {
        State s;
        s.status=j.at("status"); s.error=j.at("error");
        s.problemId=j.at("problemId");s.recipeId=j.at("recipeId");s.scaleId=j.at("scaleId");
        if(s.problemId.empty() || s.recipeId!=c.json().dump() || s.scaleId.empty())
            throw std::invalid_argument("state identity/recipe mismatch");
        s.initialized=j.at("initialized"); s.bestValid=j.at("bestValid"); s.cacheValid=j.at("cacheValid");
        s.hessianReuseAge=j.value("hessianReuseAge",0);
        if((c.guardedHessianReuse && s.hessianReuseAge!=(s.cacheValid?1:0)) ||
           (!c.guardedHessianReuse && s.hessianReuseAge!=0))
            throw std::invalid_argument("invalid Hessian reuse age/cache state");
        s.boundary=j.at("boundary");if(!s.boundary)throw std::invalid_argument("checkpoint is inside an active step");
        s.attempts=j.at("attempts");s.iterations=j.at("iterations");s.evaluations=j.at("evaluations");
        s.hessians=j.at("hessians");s.hvps=j.at("hvps");s.ring=j.at("ring");
        s.elapsed=j.at("elapsed");s.energy=j.at("energy");s.previousEnergy=j.at("previousEnergy");
        s.bestEnergy=j.at("bestEnergy");s.radius=j.at("radius");s.lastAlpha=j.at("lastAlpha");s.lastShift=j.at("lastShift");
        s.x=unpack(j.at("x"));s.g=unpack(j.at("g"));s.previousX=unpack(j.at("previousX"));
        s.previousG=unpack(j.at("previousG"));s.cgGradient=unpack(j.at("cgGradient"));
        s.cgDirection=unpack(j.at("cgDirection"));s.bestX=unpack(j.at("bestX"));
        s.s=j.at("s").get<std::vector<double>>();s.y=j.at("y").get<std::vector<double>>();
        s.rho=j.at("rho").get<std::vector<double>>();s.diagonal=j.at("diagonal").get<std::vector<double>>();
        s.free=j.at("free").get<std::vector<int>>();s.A=unpackSparse(j.at("A"));s.trace=j.at("trace");
        for(const Vec* v:{&s.x,&s.g,&s.previousX,&s.previousG,&s.cgGradient,&s.cgDirection,&s.bestX})
            if(v->size()!=n) throw std::invalid_argument("checkpoint vector dimension mismatch");
        if(s.attempts<0 || s.iterations<0 || s.iterations>s.attempts || s.evaluations<0 || s.hessians<0 ||
           s.hvps<0 || s.ring<0 || s.ring>=c.memory || !s.trace.is_array()) throw std::invalid_argument("bad counters");
        for(double v:{s.elapsed,s.energy,s.previousEnergy,s.bestEnergy,s.radius,s.lastAlpha,s.lastShift})
            if(!std::isfinite(v))throw std::invalid_argument("nonfinite checkpoint scalar");
        if(s.elapsed<0 || s.radius<=0 || s.lastAlpha<=0 || s.lastShift<0)throw std::invalid_argument("bad checkpoint control");
        const bool lb=c.method=="native_lbfgs";
        if(s.s.size()!=size_t(lb?n*c.memory:0) || s.y.size()!=s.s.size() ||
           s.rho.size()!=size_t(lb?c.memory:0) || s.diagonal.size()!=size_t(lb?n:0))
            throw std::invalid_argument("bad L-BFGS memory dimensions");
        for(const auto* v:{&s.s,&s.y,&s.rho,&s.diagonal})
            for(double x:*v)if(!std::isfinite(x))throw std::invalid_argument("bad L-BFGS memory values");
        for(double d:s.diagonal)if(d<=0)throw std::invalid_argument("nonpositive L-BFGS diagonal");
        std::vector<char> seen(n,0);
        for(int i:s.free){if(i<0 || i>=n || seen[i])throw std::invalid_argument("bad free-DOF map");seen[i]=1;}
        if(s.cacheValid && (s.A.rows()!=int(s.free.size()) || s.A.cols()!=s.A.rows() || s.free.empty()))
            throw std::invalid_argument("bad Hessian cache");
        return s;
    }
};

// FNV-1a detects accidental corruption, NOT malicious modification. Source/problem
// identity strings must be independently content-hashed by the benchmark runner.
inline std::string checksum(const std::string& bytes)
{
    uint64_t h=14695981039346656037ull;
    for(unsigned char c:bytes){h^=c;h*=1099511628211ull;}
    return std::to_string(h);
}
inline void saveCheckpoint(const std::string& path,const State& s,const Config& c,const Problem& p)
{
    c.validate(); State::fromJson(s.json(),p.size(),c);
    if(s.problemId!=p.identity() || s.scaleId!=checksum(Json({{"scales",pack(p.scales())},{"energyReference",p.energyReference()}}).dump()))
        throw std::invalid_argument("checkpoint problem mismatch");
    const Json payload={{"schema",2},{"identity",p.identity()},{"config",c.json()},
        {"scales",pack(p.scales())},{"energyReference",p.energyReference()},{"state",s.json()}};
    const std::string bytes=Json({{"payload",payload},{"checksum",checksum(payload.dump())}}).dump()+"\n";
    const std::string temp=path+".tmp";
    int fd=::open(temp.c_str(),O_WRONLY|O_CREAT|O_EXCL,0600);
    if(fd<0)throw std::runtime_error("checkpoint temporary exists or cannot be created");
    size_t done=0;
    while(done<bytes.size())
    {
        const ssize_t wrote=::write(fd,bytes.data()+done,bytes.size()-done);
        if(wrote<=0){::close(fd);throw std::runtime_error("checkpoint write failed; partial temp preserved");}
        done+=wrote;
    }
    if(::fsync(fd)!=0){::close(fd);throw std::runtime_error("checkpoint fsync failed");}
    ::close(fd);
    // Atomic no-clobber publication in the SAME filesystem. Never replace old generations.
    if(::link(temp.c_str(),path.c_str())!=0)throw std::runtime_error("checkpoint publication failed or target exists");
    const auto slash=path.find_last_of('/');
    const std::string parent=slash==std::string::npos?".":(slash==0?"/":path.substr(0,slash));
    int dir=::open(parent.c_str(),O_RDONLY|O_DIRECTORY);
    if(dir<0)throw std::runtime_error("cannot open checkpoint directory for fsync");
    const int rc=::fsync(dir);::close(dir);
    if(rc!=0)throw std::runtime_error("checkpoint directory fsync failed; names preserved");
    ::unlink(temp.c_str());
}
struct ResourceFloor
{
    double elapsed=0;
    int evaluations=0, hessians=0, attempts=0, hvps=0;
    bool cellCompleted=false;
};
inline State loadCheckpoint(const std::string& path,const Config& c,const Problem& p,const ResourceFloor& floor)
{
    c.validate(); std::ifstream in(path); if(!in)throw std::runtime_error("checkpoint not found");
    Json j;in>>j;const Json& v=j.at("payload");
    if(j.at("checksum")!=checksum(v.dump()) || v.at("schema")!=2 || v.at("identity")!=p.identity() ||
       v.at("config")!=c.json() || v.at("scales")!=pack(p.scales()) || v.at("energyReference")!=p.energyReference())
        throw std::invalid_argument("checkpoint integrity/identity/recipe mismatch");
    State s=State::fromJson(v.at("state"),p.size(),c);
    if(s.problemId!=p.identity() || s.scaleId!=checksum(Json({{"scales",pack(p.scales())},{"energyReference",p.energyReference()}}).dump()))
        throw std::invalid_argument("checkpoint state identity mismatch");
    if(!std::isfinite(floor.elapsed) || floor.elapsed<s.elapsed || floor.evaluations<s.evaluations ||
       floor.hessians<s.hessians || floor.attempts<s.attempts || floor.hvps<s.hvps ||
       (floor.cellCompleted && s.status=="ready")) throw std::invalid_argument("resource ledger reset or completed cell");
    s.elapsed=floor.elapsed;s.evaluations=floor.evaluations;s.hessians=floor.hessians;
    s.attempts=floor.attempts;s.hvps=floor.hvps;
    return s;
}
inline ResourceFloor resources(const State& s)
{return {s.elapsed,s.evaluations,s.hessians,s.attempts,s.hvps,s.status!="ready"};}

class Engine
{
    struct Stop { std::string reason; };
    Problem& problem;
    Config config;
    State s;
    Vec S;
    double E0,lastClock;
    std::function<double()> clock;
    // Deliberately absent from checkpoints: restoration performs a fresh analysis.
    std::unique_ptr<BenchmarkNewtonSymbolicCache> symbolicCache;
    void sync()
    {
        const double now=clock();
        if(!std::isfinite(now) || now<lastClock)throw std::runtime_error("nonmonotonic clock");
        s.elapsed+=now-lastClock;lastClock=now;
    }
    void timeCheck(){sync();if(s.elapsed>=config.seconds)throw Stop{"time_cap"};}
    Value evaluate(const Vec& x,Json& row)
    {
        timeCheck(); if(s.evaluations>=config.maxEvaluations)throw Stop{"evaluation_cap"};
        ++s.evaluations;
        if(!x.allFinite())throw Stop{"nonfinite_trial"};
        const double start=clock();Value v=problem.evaluate(x);
        row["energy_gradient_seconds"]=row.value("energy_gradient_seconds",0.)+(clock()-start);
        timeCheck();
        if(!std::isfinite(v.energy) || v.gradient.size()!=problem.size() || !v.gradient.allFinite() ||
           !std::isfinite(v.gradient.norm()))throw Stop{"invalid_oracle"};
        return v;
    }
    void best()
    {
        if(s.g.norm()<=config.physicalGate && (!s.bestValid || s.energy<s.bestEnergy))
        {s.bestValid=true;s.bestEnergy=s.energy;s.bestX=s.x;}
    }
    void verifyStop(Json& row)
    {
        if(s.g.norm()>config.tolerance)return;
        const Value v=evaluate(s.x,row);
        s.energy=v.energy;s.g=v.gradient;
        s.status=s.g.norm()<=config.tolerance?"gradient_target":"verification_failed";
        if(s.status=="verification_failed")s.bestValid=false;else best();
    }
    void invalidateModel(){s.cacheValid=false;s.A.resize(0,0);s.hessianReuseAge=0;}
    void model(Json& row)
    {
        if(config.guardedHessianReuse && s.cacheValid){
            const double begin=clock();const auto currentFree=problem.freeDofs(s.x,S,s.free);timeCheck();
            row["reuse_guard_seconds"]=clock()-begin;
            if(s.hessianReuseAge!=1 || currentFree!=s.free){
                row["hessian_refresh_reason"]="age_or_restricted_dofs_changed";invalidateModel();
            }
        }
        row["model_cache_hit"]=s.cacheValid;
        if(s.cacheValid)return;
        timeCheck(); if(s.hessians>=config.maxHessians)throw Stop{"hessian_cap"};
        ++s.hessians;const double oracleStart=clock();const Sparse H=problem.hessian(s.x);
        row["hessian_oracle_seconds"]=clock()-oracleStart;
        if(config.hessianThreads!=1)row["hessian_threads"]=config.hessianThreads;
        timeCheck();
        const int n=problem.size();
        if(H.rows()!=n || H.cols()!=n)throw Stop{"invalid_hessian"};
        const double gaugeStart=clock();s.free=problem.freeDofs(s.x,S,s.free);
        row["gauge_seconds"]=clock()-gaugeStart;
        const double assemblyStart=clock();
        if(s.free.empty())throw Stop{"no_free_dofs"};
        std::vector<int> map(n,-1);
        for(size_t j=0;j<s.free.size();++j)
        {const int i=s.free[j];if(i<0 || i>=n || map[i]>=0)throw Stop{"invalid_gauge"};map[i]=j;}
        std::vector<Eigen::Triplet<double>> t;t.reserve(H.nonZeros());
        for(int j=0;j<H.outerSize();++j)for(Sparse::InnerIterator it(H,j);it;++it)
        {
            if(!std::isfinite(it.value()))throw Stop{"invalid_hessian"};
            if(map[it.row()]>=0 && map[it.col()]>=0)
            {
                const double a=it.value()*S[it.row()]*S[it.col()]/E0;
                if(!std::isfinite(a))throw Stop{"invalid_hessian"};
                t.emplace_back(map[it.row()],map[it.col()],a);
            }
        }
        s.A.resize(s.free.size(),s.free.size());s.A.setFromTriplets(t.begin(),t.end());
        row["reduced_assembly_seconds"]=clock()-assemblyStart;
        const double symmetryStart=clock();
        const Sparse trans=s.A.transpose(); const Sparse diff=s.A-trans;
        if(diff.norm()>1e-10*s.A.norm())throw Stop{"asymmetric_hessian"};
        s.A=0.5*(s.A+trans);s.cacheValid=true;
        row["symmetry_seconds"]=clock()-symmetryStart;row["model_nnz"]=s.A.nonZeros();
    }
    double curvatureScale() const
    {
        double scale=1e-12;
        for(int j=0;j<s.A.outerSize();++j)for(Sparse::InnerIterator it(s.A,j);it;++it)
            scale=std::max(scale,std::abs(it.value()));
        return scale; // an indefinite Hessian can have a zero diagonal but nonzero coupling
    }
    Vec scaledGradient() const { return S.cwiseProduct(s.g)/E0; }
    Vec reducedGradient() const
    {Vec b(s.free.size());for(size_t j=0;j<s.free.size();++j)b[j]=S[s.free[j]]*s.g[s.free[j]]/E0;return b;}
    Vec expand(const Vec& reduced) const
    {Vec p=Vec::Zero(problem.size());for(size_t j=0;j<s.free.size();++j)p[s.free[j]]=reduced[j];return p;}
    Vec lbfgsDirection()
    {
        const int n=problem.size(),m=config.memory,iter=s.iterations,cur=s.ring;
        Vec p=-s.g; std::vector<double> alpha(m,0);
        if(iter>0)
        {
            const int start=cur*n;
            for(int i=0;i<n;++i){s.s[start+i]=s.x[i]-s.previousX[i];s.y[start+i]=s.g[i]-s.previousG[i];}
            const double ys=HLBFGS_DDOT(n,s.y.data()+start,s.s.data()+start);
            if(!(ys>0) || !std::isfinite(ys))throw Stop{"lbfgs_curvature_failure"};
            s.rho[cur]=1.0/ys;
            double a=1.0/(1+s.rho[cur]*(6*(s.previousEnergy-s.energy)+3*(
                HLBFGS_DDOT(n,s.g.data(),s.s.data()+start)+HLBFGS_DDOT(n,s.previousG.data(),s.s.data()+start))));
            if(a<0.01)a=0.01;else if(a>100)a=100;
            if(!std::isfinite(a))throw Stop{"lbfgs_curvature_failure"};
            s.rho[cur]*=a;
            const int bound=iter>m?m-1:iter-1;
            HLBFGS_UPDATE_First_Step(n,m,p.data(),s.s.data(),s.y.data(),s.rho.data(),alpha.data(),bound,cur,iter);
            int info[20];double params[20];INIT_HLBFGS(params,info);info[2]=iter;info[3]=1;info[12]=1;
            HLBFGS_UPDATE_Hessian(nullptr,n,m,p.data(),s.s.data(),s.y.data(),cur,s.diagonal.data(),info);
            for(double d:s.diagonal)if(!std::isfinite(d) || d<=0)throw Stop{"lbfgs_curvature_failure"};
            HLBFGS_UPDATE_Second_Step(n,m,p.data(),s.s.data(),s.y.data(),s.rho.data(),alpha.data(),bound,cur,iter);
            s.ring=(cur+1)%m;
        }
        return p;
    }
    void commit(const Vec& x,const Value& v,const bool retainHessian=false)
    {
        s.previousX=s.x;s.previousG=s.g;s.previousEnergy=s.energy;
        s.x=x;s.g=v.gradient;s.energy=v.energy;++s.iterations;
        if(retainHessian)s.hessianReuseAge=1;else invalidateModel();
        best(); // the successful oracle already left its working geometry at x
    }
    void wolfe(const Vec& p,const bool native,Json& row)
    {
        if(!p.allFinite() || !std::isfinite(p.norm()))throw Stop{"invalid_direction"};
        Vec u=native?s.x:Vec::Zero(problem.size()); Vec gh=native?s.g:scaledGradient();
        const double slope=gh.dot(p);
        if(!std::isfinite(slope) || slope>=0)throw Stop{"non_descent"};
        const Vec base=s.x; const Vec baseGH=gh; Value accepted;
        double f=s.energy/E0,alpha=native?(s.iterations==0?1.0/HLBFGS_DNRM2(problem.size(),s.g.data()):1.0):s.lastAlpha;
        if(!native && s.iterations==0)alpha=std::min(1.0,1.0/gh.norm());
        int n=problem.size(),info=0,nfev=0,keep[20]={};double rkeep[40]={};
        double c1=config.c1,c2=native?0.9:config.cgC2,xtol=1e-16,stpmin=1e-20,stpmax=1e20;
        int maxfev=native?20:config.maxLineEvaluations;
        Vec wa=Vec::Zero(n),direction=p;
        for(int guard=0;guard<2*maxfev+4;++guard)
        {
            timeCheck();
            MCSRCH(&n,u.data(),&f,gh.data(),direction.data(),&alpha,&c1,&c2,&xtol,&stpmin,&stpmax,
                   &maxfev,&info,&nfev,wa.data(),keep,rkeep,nullptr);
            if(info==-1)
            {
                const Vec x=native?u:(base+S.cwiseProduct(u)).eval();
                accepted=evaluate(x,row);f=accepted.energy/E0;gh=S.cwiseProduct(accepted.gradient)/E0;
                row["trials"].push_back({{"alpha",alpha},{"energy",accepted.energy},{"gradient_norm",accepted.gradient.norm()}});
            }
            else
            {
                row["line_search_info"]=info;row["alpha"]=alpha;
                if(info!=1 || row["trials"].empty())throw Stop{"line_search_failed"};
                const Vec x=native?u:(base+S.cwiseProduct(u)).eval();
                if(!native)
                {s.cgGradient=baseGH;s.cgDirection=p;s.lastAlpha=std::max(1e-20,std::min(1e20,alpha));}
                commit(x,accepted);return;
            }
        }
        throw Stop{"line_search_failed"};
    }
    void newton(Json& row)
    {
        const double modelStart=clock();model(row);row["model_seconds"]=clock()-modelStart;
        const bool stale=config.guardedHessianReuse && s.hessianReuseAge==1;
        if(config.guardedHessianReuse)row["hessian_reused"]=stale;
        const Vec b=reducedGradient();const double scale=curvatureScale();
        // All shifts at this base state share a pattern. Explicit zero diagonals
        // are essential when the original Hessian has off-diagonal-only curvature.
        Sparse B=s.A;const Vec diagonal=s.A.diagonal();
        for(int i=0;i<B.rows();++i)B.coeffRef(i,i)=diagonal[i];
        B.makeCompressed();
        std::unique_ptr<BenchmarkNewtonFactorization> freshFactor;
        BenchmarkNewtonFactorization* selected=nullptr;
        row["factorization_backend"]=config.newtonBackend=="cholmod"?"cholmod_supernodal_external_eigen_amd":"llt_early_nonpositive_pivot";
        timeCheck();const double symbolicStart=clock();bool analyzed=true;
        if(config.reuseNewtonSymbolic){
            if(!symbolicCache)symbolicCache=std::make_unique<BenchmarkNewtonSymbolicCache>(config.newtonBackend, config.linearSolverThreads);
            analyzed=symbolicCache->prepare(B,s.free);selected=&symbolicCache->get();
        }else{
            freshFactor=std::make_unique<BenchmarkNewtonFactorization>(config.newtonBackend, config.linearSolverThreads);
            freshFactor->analyzePattern(B);selected=freshFactor.get();
        }
        BenchmarkNewtonFactorization& factor=*selected;
        row["symbolic_seconds"]=clock()-symbolicStart;row["symbolic_analyses"]=analyzed?1:0;
        if(config.reuseNewtonSymbolic)row["symbolic_cache_hit"]=!analyzed;
        row["numerical_factorizations"]=0;row["factorization_seconds"]=0.;timeCheck();
        double shift=0;Vec y;bool solved=false;
        for(int attempt=0;attempt<config.maxShiftAttempts;++attempt)
        {
            timeCheck();
            for(int i=0;i<B.rows();++i)B.coeffRef(i,i)=diagonal[i]+shift;
            const double factorStart=clock();factor.factorize(B);
            const double factorSeconds=clock()-factorStart;
            const bool acceptable=factor.acceptable(config.pivotFloor*scale);
            row["numerical_factorizations"]=attempt+1;
            row["factorization_seconds"]=row["factorization_seconds"].get<double>()+factorSeconds;
            row["factorization_trials"].push_back({{"shift",shift},{"seconds",factorSeconds},
                {"completed",factor.successful()},{"pivot_gate_passed",acceptable}});
            timeCheck();
            if(acceptable)
            {
                const double solveStart=clock();y=factor.solve(-b);
                row["solve_seconds"]=clock()-solveStart;timeCheck();
                const double residual=(B*y+b).norm()/std::max(b.norm(),1e-300);
                row["linear_residual"]=residual;
                if(!factor.successful() || !y.allFinite() || !std::isfinite(residual) || residual>config.linearTolerance)
                    throw Stop{"linear_solve_failed"};
                row[config.newtonBackend=="cholmod"?"factor_stored_values":"factor_nnz"]=factor.storedValues();
                row["factor_column_square_proxy"]=factor.columnSquareWorkProxy();
                solved=true;break;
            }
            shift=shift==0?1e-6*scale:10*shift;
        }
        if(!solved)throw Stop{"regularization_failed"};
        s.lastShift=shift;row["shift"]=shift;
        const Vec p=S.cwiseProduct(expand(y)); const double slope=s.g.dot(p);
        if(!std::isfinite(slope) || slope>=0 || !p.allFinite())throw Stop{"non_descent"};
        for(int trial=0;trial<config.maxLineEvaluations;++trial)
        {
            const double alpha=std::ldexp(1.0,-trial);const Vec x=s.x+alpha*p;
            const Value v=evaluate(x,row);
            row["trials"].push_back({{"alpha",alpha},{"energy",v.energy},{"gradient_norm",v.gradient.norm()}});
            if(v.energy<s.energy && v.energy<=s.energy+config.c1*alpha*slope)
            {
                row["alpha"]=alpha;
                // Only a fresh, unshifted, full accepted step can seed one reuse.
                const bool retain=config.guardedHessianReuse && !stale && shift==0 && alpha==1;
                commit(x,v,retain);return;
            }
            if(stale){
                // A stale model gets ONE full-step trial, not a stale line search.
                // Charge this rejected attempt/evaluation, restore x, then let the
                // next budgeted step build a fresh model. Numerical failures remain terminal.
                problem.restore(s.x);invalidateModel();row["accepted"]=false;
                row["hessian_refresh_reason"]="stale_full_step_rejected";return;
            }
        }
        throw Stop{"line_search_failed"};
    }
    void trustRegion(Json& row)
    {
        const double modelStart=clock();model(row);row["model_seconds"]=clock()-modelStart;
        const Vec b=reducedGradient();
        if(s.radius<config.minRadius)throw Stop{"trust_radius_collapsed"};
        const double diagScale=curvatureScale();
        const Vec rawMetric=Vec(s.A.diagonal()).cwiseAbs();
        const Vec metric=rawMetric.cwiseMax(1e-8*diagScale);
        row["metric_min"]=metric.minCoeff();row["metric_max"]=metric.maxCoeff();
        row["metric_floor_count"]=(rawMetric.array()<1e-8*diagScale).count();
        row["reduced_gradient_norm"]=b.norm();
        row["dual_metric_gradient_norm"]=b.cwiseQuotient(metric.cwiseSqrt()).norm();
        ShellEquilibrium::SteihaugOptions opts;opts.maxIterations=config.maxCG;
        opts.forcingFactor=config.cgForcing;opts.superlinearForcing=false;opts.absoluteTolerance=config.cgAbsolute;
        const auto hv=[&](const Vec& v)->Vec {timeCheck();++s.hvps;Vec w=s.A*v;if(!w.allFinite())throw Stop{"invalid_hvp"};return w;};
        const double innerStart=clock();
        const auto cg=ShellEquilibrium::steihaugTruncatedCG(b,hv,ShellEquilibrium::NoProjection(),metric,s.radius,opts);
        row["inner_seconds"]=clock()-innerStart;
        row["step_metric_norm"]=cg.stepMetricNorm;row["scaled_step_norm"]=cg.step.norm();
        row["model_linear_term"]=b.dot(cg.step);
        row["inner_iterations"]=cg.iterations;row["negative_curvature"]=cg.negativeCurvatureDetected;
        row["cg_termination"]=ShellEquilibrium::toString(cg.termination);row["radius_before"]=s.radius;
        if(!cg.step.allFinite() || !std::isfinite(cg.predictedReduction) || cg.predictedReduction<=0)
            throw Stop{"invalid_trust_model"};
        const Vec x=s.x+S.cwiseProduct(expand(cg.step));const Value v=evaluate(x,row);
        const double ratio=(s.energy-v.energy)/(E0*cg.predictedReduction);
        if(!std::isfinite(ratio))throw Stop{"invalid_trust_ratio"};
        const bool accepted=ratio>config.trAcceptance && v.energy<s.energy;
        row["ratio"]=ratio;row["predicted_drop"]=E0*cg.predictedReduction;
        row["trials"].push_back({{"energy",v.energy},{"gradient_norm",v.gradient.norm()},{"accepted",accepted}});
        if(ratio<0.25)s.radius*=0.25;
        else if(ratio>0.75 && cg.boundaryReached)s.radius=std::min(config.maxRadius,2*s.radius);
        row["radius_after"]=s.radius;row["accepted"]=accepted;
        if(accepted)commit(x,v);else problem.restore(s.x); // cache survives rejection and checkpoint
    }
public:
    static double wallClock()
    {return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
    Engine(Problem& p,const Config& c,const Vec& initial,std::function<double()> timer=wallClock)
      :problem(p),config(c),clock(std::move(timer))
    {
        config.validate();if(initial.size()!=p.size() || p.size()<1 || !initial.allFinite() || p.identity().empty())
            throw std::invalid_argument("invalid initial state or identity");
        S=c.method=="native_lbfgs"?Vec::Ones(p.size()):p.scales();E0=c.method=="native_lbfgs"?1:p.energyReference();
        if(S.size()!=p.size() || !S.allFinite() || S.minCoeff()<=0 || !std::isfinite(E0) || E0<=0)
            throw std::invalid_argument("invalid oracle scales");
        s.x=initial;s.g=s.previousX=s.previousG=s.cgGradient=s.cgDirection=s.bestX=Vec::Zero(p.size());
        s.radius=c.initialRadius;lastClock=clock();
        s.problemId=p.identity();s.recipeId=c.json().dump();
        s.scaleId=checksum(Json({{"scales",pack(p.scales())},{"energyReference",p.energyReference()}}).dump());
        if(c.method=="native_lbfgs")
        {s.s.assign(p.size()*c.memory,0);s.y=s.s;s.rho.assign(c.memory,0);s.diagonal.assign(p.size(),1);}
    }
    Engine(Problem& p,const Config& c,const State& saved,std::function<double()> timer=wallClock)
      :Engine(p,c,saved.x,std::move(timer))
    {
        if(saved.problemId!=s.problemId || saved.recipeId!=s.recipeId || saved.scaleId!=s.scaleId)
            throw std::invalid_argument("direct resume identity/recipe mismatch");
        s=State::fromJson(saved.json(),p.size(),c);problem.restore(s.x);
    }
    const State& state() const{return s;}
    void accountCurvatureProbe()
    {
        ++s.hessians;
        if(s.hessians>=config.maxHessians)s.status="hessian_cap";
    }
    State snapshot()
    {if(!s.boundary)throw std::logic_error("snapshot requested inside active step");sync();return s;}
    bool step()
    {
        if(s.status!="ready")return false;
        if(!s.boundary)throw std::logic_error("reentrant optimizer step");
        s.boundary=false;
        Json row={{"attempt",s.attempts+1},{"trials",Json::array()}};
        try
        {
            timeCheck();
            if(!s.initialized)
            {const Value v=evaluate(s.x,row);s.energy=v.energy;s.g=v.gradient;s.initialized=true;best();}
            verifyStop(row);
            if(s.status!="ready")
            {
                sync();s.boundary=true;
                if(config.guardedHessianReuse)invalidateModel();
                row["initial_stop"]=true;row["status"]=s.status;row["energy"]=s.energy;
                row["gradient_norm"]=s.g.norm();row["evaluations"]=s.evaluations;row["elapsed"]=s.elapsed;
                s.trace.push_back(row);return false;
            }
            if(s.attempts>=config.maxAttempts)throw Stop{"iteration_cap"};
            ++s.attempts;row["base_energy"]=s.energy;row["base_gradient_norm"]=s.g.norm();
            if(config.method=="native_lbfgs")
            {
                const double start=clock();const Vec direction=lbfgsDirection();
                row["lbfgs_direction_seconds"]=clock()-start;wolfe(direction,true,row);
            }
            else if(config.method=="nonlinear_cg")
            {
                const Vec g=scaledGradient();
                if(!g.allFinite() || !std::isfinite(g.norm()) || g.norm()==0)throw Stop{"invalid_scaled_gradient"};
                double beta=0;
                if(s.iterations>0 && s.cgGradient.squaredNorm()>0)
                    beta=std::max(0.0,g.dot(g-s.cgGradient)/s.cgGradient.squaredNorm());
                Vec p=-g+beta*s.cgDirection; bool restart=false;
                if(!std::isfinite(beta) || !p.allFinite() || g.dot(p)>=-1e-3*g.squaredNorm())
                {p=-g;beta=0;restart=true;s.lastAlpha=1;}
                row["beta"]=beta;row["descent_restart"]=restart;wolfe(p,false,row);
            }
            else if(config.method=="sparse_newton")newton(row);
            else trustRegion(row);
            row["energy"]=s.energy;row["gradient_norm"]=s.g.norm();row["iterations"]=s.iterations;
            if(!row.contains("accepted"))row["accepted"]=true;
            verifyStop(row);timeCheck();
        }
        catch(const Stop& e){s.status=e.reason;row["failure"]=e.reason;}
        catch(const std::exception& e){s.status="oracle_exception";s.error=e.what();row["failure"]=s.error;}
        if(s.status!="ready" && s.status!="gradient_target")problem.restore(s.x);
        if(config.guardedHessianReuse && s.status!="ready")invalidateModel();
        sync();s.boundary=true;row["status"]=s.status;row["evaluations"]=s.evaluations;row["elapsed"]=s.elapsed;
        s.trace.push_back(row);return s.status=="ready";
    }
};
}
#endif

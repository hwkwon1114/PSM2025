#include "gtest/gtest.h"
#include "../diagnostics/NonconvexBenchmark.hpp"
#include "../diagnostics/ShellBenchmarkProblem.hpp"
#include "../diagnostics/GeometricNestedDissection.hpp"
#include "Geometry.hpp"
#include "Mesh.hpp"
#include "MaterialProperties.hpp"
#include "CombinedOperator_Parametric.hpp"
#include "EnergyOperatorList.hpp"
#include <tbb/global_control.h>
#include <filesystem>
#include <cstdlib>

using namespace NonconvexBenchmark;
namespace
{
const std::vector<std::string> methods={"native_lbfgs","nonlinear_cg","sparse_newton","trust_region_cg"};
struct Toy : Problem
{
    std::string kind="rosen",tag="v1";
    Vec current, scaling=Vec::Ones(2);
    double reference=1;
    int calls=0,hcalls=0;
    double* ticks=nullptr;
    explicit Toy(std::string k="rosen"):kind(std::move(k)),current(Vec::Zero(2)){}
    int size() const override{return 2;}
    std::string identity() const override{return "toy:"+kind+":"+tag;}
    Vec scales() const override{return scaling;}
    double energyReference() const override{return reference;}
    void restore(const Vec& x) override{current=x;}
    Value evaluate(const Vec& x) override
    {
        ++calls;if(ticks)*ticks+=0.1;current=x;Vec g(2);double f=0;
        if(kind=="rosen")
        {const double a=1-x[0],b=x[1]-x[0]*x[0];f=a*a+100*b*b;g<<-2*a-400*x[0]*b,200*b;}
        else if(kind=="well")
        {const double a=x[0]*x[0]-1;f=0.25*a*a+1.5*x[1]*x[1];g<<x[0]*a,3*x[1];}
        else if(kind=="quartic")
        {f=(0.5*x.array().square()-0.25*x.array().pow(4)).sum();g=x-x.array().cube().matrix();}
        else if(kind=="plateau") {f=1;g.setOnes();}
        else if(kind=="broken")
        {g.setOnes();f=(x-Vec::Ones(2)).norm()>0?std::numeric_limits<double>::quiet_NaN():1;}
        else if(kind=="overflow") {f=1;g.setConstant(1e308);}
        else {f=0.5*x[0]*x[0]+2*x[1]*x[1];g<<x[0],4*x[1];}
        return {reference*f,reference*g};
    }
    Sparse hessian(const Vec& x) override
    {
        ++hcalls;if(ticks)*ticks+=0.2;current=x;Eigen::Matrix2d H=Eigen::Matrix2d::Zero();
        if(kind=="rosen")H<<2-400*x[1]+1200*x[0]*x[0],-400*x[0],-400*x[0],200;
        else if(kind=="well")H.diagonal()<<3*x[0]*x[0]-1,3;
        else if(kind=="quartic")H.diagonal()=(1-3*x.array().square()).matrix();
        else H.diagonal()<<1,4;
        return (reference*H).sparseView();
    }
};
Config controls(const std::string& method)
{
    Config c;c.method=method;c.tolerance=1e-7;c.physicalGate=1e-6;c.maxAttempts=3000;
    c.maxEvaluations=100000;c.seconds=60;return c;
}
void run(Engine& engine,int max=5000)
{for(int i=0;i<max && engine.state().status=="ready";++i)engine.step();}
static std::vector<std::string>& tempDirs()
{
    static auto* dirs = new std::vector<std::string>();
    return *dirs;
}

std::string directory()
{
    char name[]="checkpoint_test_XXXXXX";
    const char* p=::mkdtemp(name);if(!p)throw std::runtime_error("mkdtemp failed");
    tempDirs().push_back(p);
    return p;
}

struct TempDirCleanup : public ::testing::Environment
{
    void TearDown() override
    {
        for(const auto& d : tempDirs())
        {
            std::error_code ec;
            std::filesystem::remove_all(d, ec);
        }
        tempDirs().clear();
    }
};

static ::testing::Environment* const temp_env = ::testing::AddGlobalTestEnvironment(new TempDirCleanup);
struct NativeStop{};
struct NativeTrace
{
    Toy problem;int limit=25;std::vector<Vec> x,g;std::vector<double> f;std::vector<int> evaluations;
    static void eval(void* ptr,int n,double* x,double*,double* f,double* g)
    {
        auto& self=*static_cast<NativeTrace*>(ptr);const auto v=self.problem.evaluate(Eigen::Map<Vec>(x,n));
        *f=v.energy;Eigen::Map<Vec>(g,n)=v.gradient;
    }
    static void iteration(void* ptr,int,int evaluations,double* x,double* f,double* g,double*)
    {
        auto& self=*static_cast<NativeTrace*>(ptr);self.x.push_back(Eigen::Map<Vec>(x,2));
        self.g.push_back(Eigen::Map<Vec>(g,2));self.f.push_back(*f);self.evaluations.push_back(evaluations);
        if(int(self.x.size())==self.limit)throw NativeStop{};
    }
};
}

TEST(BenchmarkSolvers, KnownConvexAndNonconvexMinima)
{
    for(const auto& method:methods)for(const std::string kind:{"quadratic","well"})
    {
        SCOPED_TRACE(method+":"+kind);Toy p(kind);Config c=controls(method);
        Vec x(2);x<<0.2,0.3;Engine e(p,c,x,[]{return 0.;});run(e);
        ASSERT_EQ(e.state().status,"gradient_target")<<e.state().trace.dump();
        EXPECT_LE(e.state().g.norm(),c.tolerance);EXPECT_TRUE(e.state().bestValid);
        EXPECT_LT(e.state().energy,1e-12);EXPECT_LT((e.state().x-p.current).norm(),1e-15);
        if(kind=="well")EXPECT_NEAR(std::abs(e.state().x[0]),1,1e-5);
        else EXPECT_LT(e.state().x.norm(),1e-5);
    }
}

TEST(BenchmarkNativeLBFGS, MatchesOriginalTrajectoryAcrossMemoryWrap)
{
    NativeTrace native;Vec x(2);x<<-1.2,1;
    double params[20];int info[20];INIT_HLBFGS(params,info);
    params[0]=1e-4;params[1]=1e-16;params[2]=0.9;params[5]=0;params[6]=1e-30;
    info[3]=1;info[4]=100000;info[5]=0;info[6]=0;info[7]=0;info[10]=0;info[11]=1;info[13]=3;
    try { HLBFGS(2,10,x.data(),NativeTrace::eval,nullptr,HLBFGS_UPDATE_Hessian,NativeTrace::iteration,&native,params,info); }
    catch(const NativeStop&){}
    ASSERT_EQ(native.x.size(),size_t(native.limit));
    Toy p;Config c=controls("native_lbfgs");c.tolerance=1e-30;c.physicalGate=1e-20;
    Vec initial(2);initial<<-1.2,1;Engine e(p,c,initial,[]{return 0.;});
    for(int i=0;i<native.limit;++i)
    {
        SCOPED_TRACE(i);e.step();ASSERT_EQ(e.state().status,"ready");
        EXPECT_LT((e.state().x-native.x[i]).norm(),1e-11);
        EXPECT_LT((e.state().g-native.g[i]).norm(),1e-9);
        EXPECT_NEAR(e.state().energy,native.f[i],1e-11);
        EXPECT_EQ(e.state().evaluations,native.evaluations[i]);
    }
    EXPECT_GT(e.state().iterations,2*c.memory);
}

TEST(BenchmarkSolvers, StrongWolfeAndPRDescent)
{
    for(const std::string method:{"native_lbfgs","nonlinear_cg"})
    {
        Toy p;Config c=controls(method);Vec x(2);x<<-1.2,1;Engine e(p,c,x,[]{return 0.;});
        for(int i=0;i<12 && e.state().status=="ready";++i)
        {
            const Vec base=e.state().x;const auto vb=p.evaluate(base);e.step();
            ASSERT_TRUE(e.state().status=="ready" || e.state().status=="gradient_target");
            const auto& row=e.state().trace.back();double a=row.at("alpha");const Vec direction=(e.state().x-base)/a;
            const double slope=vb.gradient.dot(direction),end=e.state().g.dot(direction);
            EXPECT_LT(slope,0);EXPECT_LE(e.state().energy,vb.energy+c.c1*a*slope+1e-12);
            EXPECT_LE(std::abs(end),(method=="native_lbfgs"?0.9:c.cgC2)*(-slope)+1e-10);
        }
    }
}

TEST(BenchmarkSolvers, RegularizationAndNegativeCurvatureUseOriginalEnergy)
{
    for(const std::string method:{"sparse_newton","trust_region_cg"})
    {
        Toy p("well");Config c=controls(method);Vec x(2);x<<0.2,0;
        const auto before=p.evaluate(x);Engine e(p,c,x,[]{return 0.;});e.step();
        ASSERT_TRUE(e.state().status=="ready" || e.state().status=="gradient_target");
        EXPECT_LT(e.state().energy,before.energy);
        EXPECT_DOUBLE_EQ(e.state().energy,p.evaluate(e.state().x).energy);
        const auto& row=e.state().trace.back();
        if(method=="sparse_newton")EXPECT_GT(row.at("shift").get<double>(),0);
        else EXPECT_TRUE(row.at("negative_curvature").get<bool>());
        EXPECT_LT(p.hessian(x).coeff(0,0),0); // original Hessian was not convexified
    }
}

TEST(BenchmarkSolvers, RejectedTrustStepRollsBackAndCachedHessianSurvivesResume)
{
    Toy p("quartic");Config c=controls("trust_region_cg");c.initialRadius=10;
    Vec x=Vec::Constant(2,0.5);Engine e(p,c,x,[]{return 0.;});e.step();
    ASSERT_EQ(e.state().status,"ready");ASSERT_EQ(e.state().iterations,0);
    EXPECT_EQ((p.current-x).norm(),0);EXPECT_EQ((e.state().x-x).norm(),0);
    EXPECT_TRUE(e.state().cacheValid);EXPECT_EQ(e.state().hessians,1);EXPECT_LT(e.state().radius,10);
    const State snapshot=e.snapshot();const auto path=directory()+"/resume.json";saveCheckpoint(path,snapshot,c,p);
    Toy fresh("quartic");State restored=loadCheckpoint(path,c,fresh,resources(snapshot));
    Engine resumed(fresh,c,restored,[]{return 0.;});e.step();resumed.step();
    EXPECT_EQ(fresh.hcalls,0);EXPECT_EQ(resumed.state().hessians,1);
    EXPECT_TRUE(resumed.state().trace.back().at("model_cache_hit").get<bool>());
    EXPECT_FALSE(resumed.state().trace.back().contains("hessian_oracle_seconds"));
    EXPECT_EQ((e.state().x-resumed.state().x).norm(),0);EXPECT_DOUBLE_EQ(e.state().radius,resumed.state().radius);
    EXPECT_EQ((fresh.current-resumed.state().x).norm(),0);
}

TEST(BenchmarkSolvers, LineSearchFailureAndNonfiniteOracleNeverBecomeSuccess)
{
    for(const auto& method:methods)
    {
        Toy bad("broken");auto c=controls(method);Vec x=Vec::Ones(2);Engine broken(bad,c,x,[]{return 0.;});broken.step();
        EXPECT_EQ(broken.state().status,"invalid_oracle")<<method;
        EXPECT_EQ(broken.state().evaluations,2);EXPECT_EQ((broken.state().x-x).norm(),0);
        EXPECT_EQ((bad.current-x).norm(),0);int calls=bad.calls;broken.step();EXPECT_EQ(calls,bad.calls);
        if(method!="trust_region_cg")
        {
            Toy p("plateau");Engine e(p,c,x,[]{return 0.;});e.step();
            EXPECT_EQ(e.state().status,"line_search_failed")<<method;
            EXPECT_EQ(e.state().iterations,0);EXPECT_EQ((p.current-x).norm(),0);
        }
    }
    Toy overflow("overflow");auto c=controls("nonlinear_cg");Engine e(overflow,c,Vec::Ones(2),[]{return 0.;});e.step();
    EXPECT_EQ(e.state().status,"invalid_oracle");EXPECT_FALSE(e.state().initialized);
}

TEST(BenchmarkSolvers, NativeStoppingIsNotAScaledNormOrMinimumCertificate)
{
    for(const std::string method:{"nonlinear_cg","sparse_newton","trust_region_cg"})
    {
        Toy p("quadratic");p.scaling.setConstant(1e-12);p.reference=1e-6;
        auto c=controls(method);c.tolerance=1e-8;c.physicalGate=1e-7;
        Engine e(p,c,Vec::Ones(2),[]{return 0.;});e.step();
        EXPECT_NE(e.state().status,"gradient_target")<<method;
        EXPECT_FALSE(e.state().bestValid);
    }
    Toy saddle("well");auto c=controls("sparse_newton");Engine e(saddle,c,Vec::Zero(2),[]{return 0.;});e.step();
    EXPECT_EQ(e.state().status,"gradient_target");EXPECT_EQ(e.state().hessians,0);
    EXPECT_GT(e.state().energy,0);EXPECT_LT(saddle.hessian(Vec::Zero(2)).coeff(0,0),0);
    // "gradient_target" means verified stationarity, never a stability/global certificate.
}

TEST(BenchmarkCheckpoint, AllMethodsResumeWithoutResettingHistoryOrCounters)
{
    for(const auto& method:methods)
    {
        SCOPED_TRACE(method);Toy fullProblem,splitProblem,fresh;
        auto c=controls(method);c.tolerance=1e-12;c.physicalGate=1e-10;
        Vec x(2);x<<-1.2,1;
        Engine full(fullProblem,c,x,[]{return 0.;});for(int i=0;i<20;++i)full.step();
        Engine split(splitProblem,c,x,[]{return 0.;});for(int i=0;i<5;++i)split.step();
        const State snap=split.snapshot();const std::string path=directory()+"/state.json";
        saveCheckpoint(path,snap,c,splitProblem);
        const auto restored=loadCheckpoint(path,c,fresh,resources(snap));Engine resumed(fresh,c,restored,[]{return 0.;});
        for(int i=5;i<20;++i)resumed.step();
        EXPECT_EQ(full.state().status,resumed.state().status);
        EXPECT_EQ(full.state().attempts,resumed.state().attempts);EXPECT_EQ(full.state().iterations,resumed.state().iterations);
        EXPECT_EQ(full.state().evaluations,resumed.state().evaluations);EXPECT_EQ(full.state().hessians,resumed.state().hessians);
        EXPECT_EQ(full.state().hvps,resumed.state().hvps);EXPECT_EQ(full.state().ring,resumed.state().ring);
        EXPECT_EQ(full.state().trace,resumed.state().trace);
        EXPECT_EQ((full.state().x-resumed.state().x).norm(),0);
        EXPECT_EQ(full.state().s,resumed.state().s);EXPECT_EQ(full.state().diagonal,resumed.state().diagonal);
        EXPECT_EQ((full.state().cgDirection-resumed.state().cgDirection).norm(),0);
        EXPECT_DOUBLE_EQ(full.state().radius,resumed.state().radius);
    }
}

TEST(BenchmarkCheckpoint, IdentityIntegrityNoClobberAndCompletedCellBarriers)
{
    Toy p;auto c=controls("native_lbfgs");Vec x(2);x<<-1.2,1;Engine e(p,c,x,[]{return 0.;});e.step();
    const auto snap=e.snapshot();const auto path=directory()+"/state.json";saveCheckpoint(path,snap,c,p);
    std::ifstream stream(path);Json original;stream>>original;
    EXPECT_THROW(saveCheckpoint(path,snap,c,p),std::runtime_error);
    std::ifstream unchanged(path);Json still;unchanged>>still;EXPECT_EQ(original,still);
    auto changed=c;changed.maxAttempts++;
    EXPECT_THROW(loadCheckpoint(path,changed,p,resources(snap)),std::invalid_argument);
    EXPECT_THROW((Engine(p,changed,snap,[]{return 0.;})),std::invalid_argument);
    changed=c;changed.implementationId="different_source_build";
    EXPECT_THROW(loadCheckpoint(path,changed,p,resources(snap)),std::invalid_argument);
    Toy other;other.tag="different";
    EXPECT_THROW(loadCheckpoint(path,c,other,resources(snap)),std::invalid_argument);
    EXPECT_THROW((Engine(other,c,snap,[]{return 0.;})),std::invalid_argument);
    other.tag=p.tag;other.scaling[0]=2;
    EXPECT_THROW(loadCheckpoint(path,c,other,resources(snap)),std::invalid_argument);
    ResourceFloor completed=resources(snap);completed.cellCompleted=true;
    EXPECT_THROW(loadCheckpoint(path,c,p,completed),std::invalid_argument);
    auto reset=resources(snap);reset.evaluations=0;
    EXPECT_THROW(loadCheckpoint(path,c,p,reset),std::invalid_argument);
    Json bad=original;bad["payload"]["state"]["x"][0]=42;
    const auto corrupt=directory()+"/corrupt.json";std::ofstream(corrupt)<<bad.dump();
    EXPECT_THROW(loadCheckpoint(corrupt,c,p,resources(snap)),std::invalid_argument);
}

TEST(BenchmarkBudgets, CapsAndExternalResourceFloorCannotBeReset)
{
    for(const auto& method:methods)
    {
        Toy p;auto c=controls(method);c.maxEvaluations=1;Vec x(2);x<<-1.2,1;
        Engine e(p,c,x,[]{return 0.;});e.step();
        EXPECT_EQ(e.state().status,"evaluation_cap");EXPECT_EQ(e.state().evaluations,1);
        EXPECT_EQ((e.state().x-x).norm(),0);EXPECT_EQ((p.current-x).norm(),0);
        const auto path=directory()+"/capped.json";saveCheckpoint(path,e.snapshot(),c,p);
        const auto snap=e.snapshot();const auto restored=loadCheckpoint(path,c,p,resources(snap));
        int calls=p.calls;Engine again(p,c,restored,[]{return 0.;});again.step();EXPECT_EQ(p.calls,calls);
    }
    Toy p;auto c=controls("native_lbfgs");c.seconds=2;Vec x(2);x<<-1.2,1;
    Engine e(p,c,x,[]{return 0.;});e.step();const auto snap=e.snapshot();const auto path=directory()+"/elapsed.json";
    saveCheckpoint(path,snap,c,p);auto ledger=resources(snap);ledger.elapsed=2;ledger.evaluations+=3;
    const auto restored=loadCheckpoint(path,c,p,ledger);int calls=p.calls;
    Engine resumed(p,c,restored,[]{return 0.;});resumed.step();
    EXPECT_EQ(resumed.state().status,"time_cap");EXPECT_EQ(p.calls,calls);
    EXPECT_EQ(resumed.state().evaluations,ledger.evaluations);
    Toy slow;double time=0;slow.ticks=&time;c.seconds=0.15;
    Engine slowEngine(slow,c,x,[&]{return time;});slowEngine.step();
    EXPECT_EQ(slowEngine.state().status,"time_cap");EXPECT_GE(slowEngine.state().elapsed,c.seconds);
}

TEST(BenchmarkBudgets, AttemptCapAndIndependentVerificationAreCounted)
{
    Toy p;auto c=controls("nonlinear_cg");c.maxAttempts=1;Vec x(2);x<<-1.2,1;
    Engine e(p,c,x,[]{return 0.;});e.step();ASSERT_EQ(e.state().status,"ready");
    const int calls=e.state().evaluations;e.step();EXPECT_EQ(e.state().status,"iteration_cap");
    EXPECT_EQ(calls,e.state().evaluations);EXPECT_EQ(e.state().attempts,1);
    Toy q("quadratic");auto zero=controls("sparse_newton");Engine stationary(q,zero,Vec::Zero(2),[]{return 0.;});stationary.step();
    EXPECT_EQ(stationary.state().status,"gradient_target");EXPECT_EQ(stationary.state().evaluations,2);
    EXPECT_EQ(stationary.state().hessians,0);
}

TEST(BenchmarkBudgets, CappedInnerCGIsNotConvergenceAndHessianBudgetIsEnforced)
{
    Toy p;auto c=controls("trust_region_cg");c.maxCG=1;c.maxHessians=1;c.initialRadius=10;
    Vec x=Vec::Constant(2,0.5);Engine e(p,c,x,[]{return 0.;});e.step();
    ASSERT_EQ(e.state().status,"ready");ASSERT_EQ(e.state().iterations,1);
    const auto& row=e.state().trace.back();EXPECT_EQ(row.at("cg_termination"),"max-iterations");
    EXPECT_EQ(row.at("inner_iterations"),1);EXPECT_GT(e.state().g.norm(),c.tolerance);
    const Vec accepted=e.state().x;const int calls=e.state().evaluations;e.step();
    EXPECT_EQ(e.state().status,"hessian_cap");EXPECT_EQ(e.state().hessians,1);
    EXPECT_EQ(e.state().evaluations,calls);EXPECT_EQ((e.state().x-accepted).norm(),0);
}

TEST(BenchmarkSolvers, InvalidHessiansAndGaugeMapsAreRejected)
{
    struct Bad : Toy
    {
        int mode;
        explicit Bad(int m):Toy("quadratic"),mode(m){}
        Sparse hessian(const Vec& x) override
        {
            auto H=Toy::hessian(x);
            if(mode==0)H.coeffRef(0,1)=1;
            if(mode==1)H.coeffRef(0,0)=std::numeric_limits<double>::quiet_NaN();
            if(mode==2)return Sparse(1,1);
            return H;
        }
        std::vector<int> freeDofs(const Vec&,const Vec&) const override
        {return mode==3?std::vector<int>{0,0}:std::vector<int>{0,1};}
    };
    for(const std::string method:{"sparse_newton","trust_region_cg"})for(int mode=0;mode<4;++mode)
    {
        Bad p(mode);const auto c=controls(method);Engine e(p,c,Vec::Ones(2),[]{return 0.;});e.step();
        EXPECT_EQ(e.state().status,mode==0?"asymmetric_hessian":(mode==3?"invalid_gauge":"invalid_hessian"));
        EXPECT_EQ(e.state().iterations,0);EXPECT_EQ(e.state().evaluations,1);
        EXPECT_EQ((p.current-Vec::Ones(2)).norm(),0);
    }
}

TEST(BenchmarkSolvers, OffDiagonalCurvatureSetsTheRegularizationScale)
{
    struct Coupled : Problem
    {
        int size() const override{return 2;}
        std::string identity() const override{return "zero_diagonal_indefinite_v1";}
        Value evaluate(const Vec& x) override
        {
            Vec g=x.array().cube().matrix();g[0]+=x[1]-1;g[1]+=x[0]+1;
            return {0.25*x.array().pow(4).sum()+x[0]*x[1]-x[0]+x[1],g};
        }
        Sparse hessian(const Vec& x) override
        {Eigen::Matrix2d H;H<<3*x[0]*x[0],1,1,3*x[1]*x[1];return H.sparseView();}
    } p;
    auto c=controls("sparse_newton");Engine e(p,c,Vec::Zero(2),[]{return 0.;});e.step();
    ASSERT_EQ(e.state().status,"ready");EXPECT_GT(e.state().lastShift,1);
    EXPECT_LT(e.state().energy,0);EXPECT_EQ(e.state().iterations,1);
    EXPECT_EQ(e.state().trace.back().at("symbolic_analyses"),1);
    EXPECT_GT(e.state().trace.back().at("numerical_factorizations").get<int>(),1);
    EXPECT_EQ(p.hessian(Vec::Zero(2)).nonZeros(),2); // original pattern is untouched
    auto tr=controls("trust_region_cg");Engine t(p,tr,Vec::Zero(2),[]{return 0.;});t.step();
    ASSERT_EQ(t.state().status,"ready");const auto& row=t.state().trace.back();
    EXPECT_EQ(row.at("metric_floor_count"),2);
    EXPECT_DOUBLE_EQ(row.at("metric_min").get<double>(),1e-8);
    EXPECT_DOUBLE_EQ(row.at("metric_max").get<double>(),1e-8);
    EXPECT_EQ(row.at("cg_termination"),"negative-curvature");
    EXPECT_NEAR(row.at("step_metric_norm").get<double>(),tr.initialRadius,1e-12);
}

TEST(BenchmarkNewton, EarlyCholeskyRejectsBadPivotsAndRecoversWithSamePattern)
{
    Sparse A(2,2);A.coeffRef(0,0)=0;A.coeffRef(1,1)=0;
    A.coeffRef(0,1)=1;A.coeffRef(1,0)=1;A.makeCompressed();
    ShiftedNewtonFactorization fast,reference(false);
    EXPECT_FALSE(fast.acceptable(1e-12));EXPECT_THROW(fast.solve(Vec::Ones(2)),std::logic_error);
    fast.analyzePattern(A);reference.analyzePattern(A);
    for(double shift:{0.,1.,2.,0.,3.})
    {
        A.coeffRef(0,0)=shift;A.coeffRef(1,1)=shift;
        fast.factorize(A);reference.factorize(A);
        EXPECT_EQ(fast.acceptable(1e-12),shift>1);
        EXPECT_EQ(fast.acceptable(1e-12),reference.acceptable(1e-12));
        if(shift>1)
        {
            Vec b(2);b<<1,-2;const Vec x=fast.solve(b);
            EXPECT_LT((A*x-b).norm(),1e-13);
            EXPECT_LT((x-reference.solve(b)).norm(),1e-13);
        }
        else EXPECT_THROW(fast.solve(Vec::Ones(2)),std::logic_error);
    }
}

TEST(BenchmarkNewton, EarlyCholeskySuccessDoesNotBypassPivotFloor)
{
    Sparse A(2,2);A.coeffRef(0,0)=1e-16;A.coeffRef(1,1)=1;A.makeCompressed();
    ShiftedNewtonFactorization fast;fast.analyzePattern(A);fast.factorize(A);
    ASSERT_TRUE(fast.successful());EXPECT_FALSE(fast.acceptable(1e-12));
    EXPECT_TRUE(fast.acceptable(1e-18));
    EXPECT_FALSE(fast.acceptable(std::numeric_limits<double>::quiet_NaN()));
    A.coeffRef(0,0)=std::numeric_limits<double>::infinity();fast.factorize(A);
    EXPECT_FALSE(fast.acceptable(1e-12));
    A.coeffRef(0,0)=std::numeric_limits<double>::quiet_NaN();fast.factorize(A);
    EXPECT_FALSE(fast.acceptable(1e-12));
}

TEST(BenchmarkNewton, ReusedSymbolicAnalysisMatchesFreshFactorizationSearch)
{
    for(const std::string kind:{"quadratic","well","rosen"})
    {
        SCOPED_TRACE(kind);Toy p(kind);auto c=controls("sparse_newton");
        Vec x(2);x<<0.2,0.3;const Value initial=p.evaluate(x);const Sparse A=p.hessian(x);
        double scale=1e-12;
        for(int j=0;j<A.outerSize();++j)for(Sparse::InnerIterator it(A,j);it;++it)
            scale=std::max(scale,std::abs(it.value()));
        double shift=0;Vec direction;int factors=0;
        for(int i=0;i<c.maxShiftAttempts;++i)
        {
            Sparse B=A;if(shift>0)for(int j=0;j<B.rows();++j)B.coeffRef(j,j)+=shift;
            Eigen::SimplicialLDLT<Sparse> solver;solver.compute(B);++factors;
            if(solver.info()==Eigen::Success && solver.vectorD().minCoeff()>c.pivotFloor*scale)
            {direction=solver.solve(-initial.gradient);break;}
            shift=shift==0?1e-6*scale:10*shift;
        }
        ASSERT_EQ(direction.size(),2);double alpha=1;bool accepted=false;
        for(int i=0;i<c.maxLineEvaluations;++i)
        {
            alpha=std::ldexp(1.,-i);const auto v=p.evaluate(x+alpha*direction);
            if(v.energy<initial.energy && v.energy<=initial.energy+c.c1*alpha*initial.gradient.dot(direction))
            {accepted=true;break;}
        }
        ASSERT_TRUE(accepted);Engine e(p,c,x,[]{return 0.;});e.step();
        ASSERT_EQ(e.state().iterations,1);const auto& row=e.state().trace.back();
        EXPECT_DOUBLE_EQ(e.state().lastShift,shift);EXPECT_DOUBLE_EQ(row.at("alpha").get<double>(),alpha);
        EXPECT_LT((e.state().x-x-alpha*direction).norm(),1e-13);
        EXPECT_EQ(row.at("symbolic_analyses"),1);EXPECT_EQ(row.at("numerical_factorizations"),factors);
        EXPECT_EQ(row.at("factorization_backend"),"llt_early_nonpositive_pivot");
        ASSERT_EQ(row.at("factorization_trials").size(),size_t(factors));
        EXPECT_TRUE(row.at("factorization_trials").back().at("pivot_gate_passed").get<bool>());
        EXPECT_GT(row.at("factor_nnz").get<long long>(),0);
        EXPECT_GT(row.at("factor_column_square_proxy").get<double>(),0);
        for(const char* key:{"model_seconds","symbolic_seconds","factorization_seconds","solve_seconds"})
            EXPECT_DOUBLE_EQ(row.at(key).get<double>(),0.);
    }
}

TEST(BenchmarkNewton, ShiftCapPreservesBaseAndDoesNotReanalyzePattern)
{
    Toy p("well");auto c=controls("sparse_newton");c.maxShiftAttempts=2;
    Vec x(2);x<<0.2,0;Engine e(p,c,x,[]{return 0.;});e.step();
    EXPECT_EQ(e.state().status,"regularization_failed");EXPECT_EQ(e.state().iterations,0);
    EXPECT_EQ((p.current-x).norm(),0);EXPECT_EQ((e.state().x-x).norm(),0);
    EXPECT_EQ(e.state().trace.back().at("symbolic_analyses"),1);
    EXPECT_EQ(e.state().trace.back().at("numerical_factorizations"),2);
}

TEST(BenchmarkTrustMetric, DiagnosticsUseScaledModelWithoutExtraHVP)
{
    Toy p("quadratic");p.scaling<<2,0.5;p.reference=7;
    auto c=controls("trust_region_cg");c.initialRadius=0.01;
    Vec x(2);x<<0.2,0.3;const Vec g=p.scaling.cwiseProduct(p.evaluate(x).gradient)/p.reference;
    Vec metric(2);metric<<4,1;double ticks=0;p.ticks=&ticks;
    Engine e(p,c,x,[&]{return ticks;});e.step();ASSERT_EQ(e.state().iterations,1);
    const auto& row=e.state().trace.back();const Vec step=(e.state().x-x).cwiseQuotient(p.scaling);
    EXPECT_EQ(row.at("cg_termination"),"trust-region-boundary");
    EXPECT_EQ(row.at("inner_iterations"),1);EXPECT_EQ(e.state().hvps,1);
    EXPECT_DOUBLE_EQ(row.at("metric_min").get<double>(),1);
    EXPECT_DOUBLE_EQ(row.at("metric_max").get<double>(),4);
    EXPECT_EQ(row.at("metric_floor_count"),0);
    EXPECT_NEAR(row.at("dual_metric_gradient_norm").get<double>(),g.cwiseQuotient(metric.cwiseSqrt()).norm(),1e-14);
    EXPECT_NEAR(row.at("step_metric_norm").get<double>(),std::sqrt(step.dot(metric.cwiseProduct(step))),1e-14);
    EXPECT_NEAR(row.at("scaled_step_norm").get<double>(),step.norm(),1e-14);
    EXPECT_NEAR(row.at("model_linear_term").get<double>(),g.dot(step),1e-14);
    EXPECT_NEAR(row.at("model_seconds").get<double>(),0.2,1e-14);
    EXPECT_NEAR(row.at("hessian_oracle_seconds").get<double>(),0.2,1e-14);
    for(const char* key:{"gauge_seconds","reduced_assembly_seconds","symmetry_seconds"})
        EXPECT_DOUBLE_EQ(row.at(key).get<double>(),0.);
    EXPECT_EQ(row.at("model_nnz"),2);
    EXPECT_NEAR(row.at("energy_gradient_seconds").get<double>(),0.2,1e-14);
    EXPECT_DOUBLE_EQ(row.at("inner_seconds").get<double>(),0);
    EXPECT_DOUBLE_EQ(row.at("radius_before").get<double>(),c.initialRadius);
}

TEST(BenchmarkOrdering, SeparatorPermutationAndSolveAgreeWithAMD)
{
    const int nx=9,ny=7,n=nx*ny;Eigen::MatrixXd xy(n,2);
    std::vector<Eigen::Triplet<double>> t;
    for(int j=0;j<ny;++j)for(int i=0;i<nx;++i)
    {
        int v=j*nx+i;xy.row(v)<<i,j;t.emplace_back(v,v,5);
        if(i+1<nx){t.emplace_back(v,v+1,-1);t.emplace_back(v+1,v,-1);}
        if(j+1<ny){t.emplace_back(v,v+nx,-1);t.emplace_back(v+nx,v,-1);}
    }
    Sparse A(n,n);A.setFromTriplets(t.begin(),t.end());
    const auto nd=geometricNestedDissection(A,xy,4);
    EXPECT_GT(nd.splits,0);EXPECT_GT(nd.maxSeparator,0);
    EXPECT_EQ(nd.order,geometricNestedDissection(A,xy,4).order);
    std::vector<int> inverse(n);for(int i=0;i<n;++i)inverse[nd.order[i]]=i;
    std::vector<Eigen::Triplet<double>> pt;
    for(int j=0;j<n;++j)for(Sparse::InnerIterator it(A,j);it;++it)pt.emplace_back(inverse[it.row()],inverse[it.col()],it.value());
    Sparse B(n,n);B.setFromTriplets(pt.begin(),pt.end());
    Vec rhs=Vec::LinSpaced(n,-1,1),b(n);for(int i=0;i<n;++i)b[i]=rhs[nd.order[i]];
    Eigen::SimplicialLLT<Sparse,Eigen::Lower,Eigen::NaturalOrdering<int>> solver;solver.compute(B);
    ASSERT_EQ(solver.info(),Eigen::Success);const Vec y=solver.solve(b);Vec x(n);
    for(int i=0;i<n;++i)x[nd.order[i]]=y[i];
    ShiftedNewtonFactorization amd;amd.analyzePattern(A);amd.factorize(A);ASSERT_TRUE(amd.acceptable(1e-12));
    EXPECT_LT((A*x-rhs).norm()/rhs.norm(),1e-13);EXPECT_LT((x-amd.solve(rhs)).norm(),1e-13);
}

TEST(BenchmarkOrdering, DegenerateCoordinatesDisconnectedGraphAndInvalidInputs)
{
    Sparse A(8,8);A.setIdentity();Eigen::MatrixXd xy=Eigen::MatrixXd::Zero(8,2);
    const auto flat=geometricNestedDissection(A,xy,2);EXPECT_EQ(flat.splits,0);EXPECT_EQ(flat.order.size(),8);
    for(int i=0;i<8;++i)xy(i,0)=i;
    const auto split=geometricNestedDissection(A,xy,2);EXPECT_GT(split.splits,0);
    EXPECT_EQ(split.maxSeparator,0);EXPECT_EQ(split.order.size(),8);
    EXPECT_THROW(geometricNestedDissection(A,xy,1),std::invalid_argument);
    xy(0,0)=std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(geometricNestedDissection(A,xy,2),std::invalid_argument);
}

#ifdef NONCONVEX_WITH_CHOLMOD
TEST(BenchmarkCholmod, RemappingAndPatternBarrier)
{
    Eigen::Matrix3d M;M<<4,1,.5,1,3,.25,.5,.25,2;Sparse A=M.sparseView();
    BenchmarkNewtonFactorization c("cholmod"),e("eigen");c.analyzePattern(A);e.analyzePattern(A);
    for(int k=0;k<3;++k){A.coeffRef(0,0)+=.2;A.coeffRef(1,2)+=.03;A.coeffRef(2,1)+=.03;
        c.factorize(A);e.factorize(A);ASSERT_TRUE(c.acceptable(1e-12));ASSERT_TRUE(e.acceptable(1e-12));
        Vec b=Vec::LinSpaced(3,1,3),x=c.solve(b);EXPECT_LT((A*x-b).norm()/b.norm(),1e-13);EXPECT_LT((x-e.solve(b)).norm(),1e-13);}
    Sparse different(3,3);different.setIdentity();EXPECT_THROW(c.factorize(different),std::invalid_argument);
}
TEST(BenchmarkCholmod, IntegratedTrajectoryAndBoundaryRecovery)
{
    Config c=controls("sparse_newton");c.newtonBackend="cholmod";c.dependencyIdentity="unit-runtime";
    Config reference=c;reference.newtonBackend="eigen";
    Toy p("rosen"),q("rosen");Vec x(2);x<<-1.2,1;
    Engine e(p,c,x,[]{return 0.;}),r(q,reference,x,[]{return 0.;});
    for(int i=0;i<3;++i){e.step();r.step();ASSERT_EQ(e.state().status,"ready");EXPECT_LT((e.state().x-r.state().x).norm(),1e-9);}
    auto saved=e.snapshot();std::string file=directory()+"/cholmod.json";saveCheckpoint(file,saved,c,p);
    Toy fresh("rosen");auto state=loadCheckpoint(file,c,fresh,resources(saved));Engine resumed(fresh,c,state,[]{return 0.;});
    for(int i=0;i<3;++i){e.step();resumed.step();}
    EXPECT_EQ(e.state().trace,resumed.state().trace);EXPECT_DOUBLE_EQ((e.state().x-resumed.state().x).norm(),0);
    EXPECT_EQ(e.state().trace.back().at("factorization_backend"),"cholmod_supernodal_external_eigen_amd");
    EXPECT_TRUE(e.state().trace.back().contains("factor_stored_values"));
    Config wrong=c;wrong.dependencyIdentity="changed-library";EXPECT_THROW(loadCheckpoint(file,wrong,fresh,resources(saved)),std::invalid_argument);
    wrong=c;wrong.newtonBackend="eigen";EXPECT_THROW(loadCheckpoint(file,wrong,fresh,resources(saved)),std::invalid_argument);
}
TEST(BenchmarkCholmod, BackendDefaultsAndDependencyIdentity)
{
    Config c;EXPECT_EQ(c.newtonBackend,"eigen");c.newtonBackend="cholmod";
    EXPECT_THROW(c.validate(),std::invalid_argument);c.dependencyIdentity="test-runtime";EXPECT_NO_THROW(c.validate());
    c.newtonBackend="unknown";EXPECT_THROW(c.validate(),std::invalid_argument);
}
TEST(BenchmarkCholmod, FrozenNearGate)
{
    const char* path=std::getenv("NONCONVEX_NEAR_FIXTURE");if(!path)GTEST_SKIP()<<"Explicit frozen-fixture gate only";
    Json j;std::ifstream(path)>>j;const auto& p=j.at("payload");ASSERT_TRUE(p.at("fixture_only").get<bool>());
    const auto& s=p.at("state");Sparse A=unpackSparse(s.at("A"));const Sparse transpose=A.transpose();A=.5*(A+transpose);
    for(int i=0;i<A.rows();++i) A.coeffRef(i,i)+=0.;
    A.makeCompressed();
    const Vec scales=unpack(p.at("scales")),g=unpack(s.at("g")),archived=unpack(p.at("reference_correction"));
    const auto free=s.at("free").get<std::vector<int>>();Vec b(A.rows());
    for(int i=0;i<b.size();++i)b[i]=scales[free[i]]*g[free[i]]/p.at("energyReference").get<double>();
    double scale=1e-12;for(int k=0;k<A.nonZeros();++k)scale=std::max(scale,std::abs(A.valuePtr()[k]));
    Vec reference;Json rows=Json::array();
    for(const std::string backend:{"eigen","cholmod"}){
        BenchmarkNewtonFactorization f(backend);f.analyzePattern(A);f.factorize(A);ASSERT_TRUE(f.acceptable(1e-12*scale));
        Vec y=f.solve(-b),physical=Vec::Zero(g.size());for(int i=0;i<y.size();++i)physical[free[i]]=scales[free[i]]*y[i];
        if(reference.size()==0)reference=y;
        double residual=(A*y+b).norm()/b.norm(),difference=(y-reference).norm()/reference.norm(),archiveDifference=(physical-archived).norm()/archived.norm();
        Json row={{"backend",backend},{"shift",0},{"linear_residual",residual},{"relative_correction_difference",difference},{"relative_archived_correction_difference",archiveDifference}};
        std::cout<<"NEAR_BACKEND_JSON "<<row.dump()<<std::endl;rows.push_back(row);
        ASSERT_TRUE(std::isfinite(residual) && residual<=1e-8);ASSERT_TRUE(std::isfinite(difference) && difference<=1e-5);
        ASSERT_TRUE(std::isfinite(archiveDifference) && archiveDifference<=1e-5);
    }
    if(const char* output=std::getenv("NONCONVEX_NEAR_OUTPUT")){
        ASSERT_FALSE(std::filesystem::exists(output));std::ofstream(output)<<Json({{"passed",true},{"rows",rows},{"scope","frozen unshifted linear check, not a native-gradient convergence certificate"}}).dump(2);}
}

TEST(BenchmarkCholmod, MultithreadParityAndScaling)
{
    // 1. Thread settings validation
    CholmodFrozenFactorization f_default;
    EXPECT_EQ(f_default.threads(), 1);
    CholmodFrozenFactorization f_multi(4);
    EXPECT_EQ(f_multi.threads(), 4);
    f_multi.setThreads(2);
    EXPECT_EQ(f_multi.threads(), 2);
    f_multi.setThreads(0);
    EXPECT_EQ(f_multi.threads(), 1);

    BenchmarkNewtonFactorization bnf("cholmod", 4);
    EXPECT_EQ(bnf.threads(), 4);
    bnf.setThreads(1);
    EXPECT_EQ(bnf.threads(), 1);

    BenchmarkNewtonSymbolicCache cache("cholmod", 4);
    EXPECT_EQ(cache.threads(), 4);
    cache.setThreads(2);
    EXPECT_EQ(cache.threads(), 2);

    Config cfg;
    cfg.linearSolverThreads = 4;
    cfg.newtonBackend = "cholmod";
    cfg.dependencyIdentity = "test-runtime";
    EXPECT_NO_THROW(cfg.validate());
    cfg.linearSolverThreads = 0;
    EXPECT_THROW(cfg.validate(), std::invalid_argument);

    // 2. Synthetic 2D Laplacian matrix (64x64 grid = 4096 DOFs, multi-supernode)
    const int N = 64, total = N * N;
    Eigen::SparseMatrix<double> A(total, total);
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(total * 5);
    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < N; ++j) {
            int row = i * N + j;
            triplets.emplace_back(row, row, 4.0 + 1e-4);
            if(i > 0) triplets.emplace_back(row, (i - 1) * N + j, -1.0);
            if(i < N - 1) triplets.emplace_back(row, (i + 1) * N + j, -1.0);
            if(j > 0) triplets.emplace_back(row, i * N + (j - 1), -1.0);
            if(j < N - 1) triplets.emplace_back(row, i * N + (j + 1), -1.0);
        }
    }
    A.setFromTriplets(triplets.begin(), triplets.end());
    A.makeCompressed();

    Eigen::VectorXd b(total);
    for(int i = 0; i < total; ++i) b[i] = std::sin(i + 1.0);

    BenchmarkNewtonFactorization f1("cholmod", 1);
    f1.analyzePattern(A);
    f1.factorize(A);
    ASSERT_TRUE(f1.successful());
    ASSERT_TRUE(f1.acceptable(1e-12));
    Eigen::VectorXd y1 = f1.solve(-b);
    double res1 = (A * y1 + b).norm() / b.norm();
    EXPECT_LT(res1, 1e-10);

    BenchmarkNewtonFactorization f2("cholmod", 2);
    f2.analyzePattern(A);
    f2.factorize(A);
    ASSERT_TRUE(f2.successful());
    ASSERT_TRUE(f2.acceptable(1e-12));
    Eigen::VectorXd y2 = f2.solve(-b);
    double res2 = (A * y2 + b).norm() / b.norm();
    EXPECT_LT(res2, 1e-10);

    BenchmarkNewtonFactorization f4("cholmod", 4);
    f4.analyzePattern(A);
    f4.factorize(A);
    ASSERT_TRUE(f4.successful());
    ASSERT_TRUE(f4.acceptable(1e-12));
    Eigen::VectorXd y4 = f4.solve(-b);
    double res4 = (A * y4 + b).norm() / b.norm();
    EXPECT_LT(res4, 1e-10);

    // Parity across thread counts
    double diff12 = (y1 - y2).norm() / y1.norm();
    double diff14 = (y1 - y4).norm() / y1.norm();
    EXPECT_LT(diff12, 1e-12);
    EXPECT_LT(diff14, 1e-12);

    // 3. If near-equilibrium fixture exists, test full 32k DOF shell stiffness matrix
    const char* envPath = std::getenv("NONCONVEX_NEAR_FIXTURE");
    std::string fixturePath = envPath ? std::string(envPath) :
        "run/nonconvex_solver_benchmark/newton_pilot_7386943/near_equilibrium_matrix.json";
    if (std::filesystem::exists(fixturePath)) {
        Json j;
        std::ifstream(fixturePath) >> j;
        const auto& p = j.at("payload");
        const auto& s = p.at("state");
        Sparse Ashell = unpackSparse(s.at("A"));
        Sparse AshellT = Ashell.transpose();
        Ashell = 0.5 * (Ashell + AshellT);
        for(int i = 0; i < Ashell.rows(); ++i) Ashell.coeffRef(i, i) += 0.;
        Ashell.makeCompressed();

        const Vec scales = unpack(p.at("scales")), g = unpack(s.at("g"));
        const auto free = s.at("free").get<std::vector<int>>();
        Vec bshell(Ashell.rows());
        for(int i = 0; i < bshell.size(); ++i)
            bshell[i] = scales[free[i]] * g[free[i]] / p.at("energyReference").get<double>();
        double scale = 1e-12;
        for(int k = 0; k < Ashell.nonZeros(); ++k)
            scale = std::max(scale, std::abs(Ashell.valuePtr()[k]));

        // Measure timings for 1T, 2T, 4T
        BenchmarkNewtonFactorization fs1("cholmod", 1);
        fs1.analyzePattern(Ashell);
        auto t0 = std::chrono::high_resolution_clock::now();
        fs1.factorize(Ashell);
        auto t1 = std::chrono::high_resolution_clock::now();
        Vec ys1 = fs1.solve(-bshell);
        auto t2 = std::chrono::high_resolution_clock::now();
        double time_factor_1T = std::chrono::duration<double>(t1 - t0).count();
        double time_solve_1T = std::chrono::duration<double>(t2 - t1).count();

        BenchmarkNewtonFactorization fs4("cholmod", 4);
        fs4.analyzePattern(Ashell);
        t0 = std::chrono::high_resolution_clock::now();
        fs4.factorize(Ashell);
        t1 = std::chrono::high_resolution_clock::now();
        Vec ys4 = fs4.solve(-bshell);
        t2 = std::chrono::high_resolution_clock::now();
        double time_factor_4T = std::chrono::duration<double>(t1 - t0).count();
        double time_solve_4T = std::chrono::duration<double>(t2 - t1).count();

        ASSERT_TRUE(fs1.acceptable(1e-12 * scale));
        ASSERT_TRUE(fs4.acceptable(1e-12 * scale));

        double res_s1 = (Ashell * ys1 + bshell).norm() / bshell.norm();
        double res_s4 = (Ashell * ys4 + bshell).norm() / bshell.norm();
        double diff_s14 = (ys1 - ys4).norm() / ys1.norm();

        EXPECT_LT(res_s1, 1e-8);
        EXPECT_LT(res_s4, 1e-8);
        EXPECT_LT(diff_s14, 1e-10);

        std::cout << "CHOLMOD_32K_BENCHMARK:"
                  << " 1T factor=" << time_factor_1T << "s solve=" << time_solve_1T << "s"
                  << " | 4T factor=" << time_factor_4T << "s solve=" << time_solve_4T << "s"
                  << " | factor speedup=" << time_factor_1T / std::max(1e-6, time_factor_4T) << "x"
                  << " | rel_diff=" << diff_s14 << std::endl;
    }
}
#endif

TEST(BenchmarkSelection, LowerEnergyWithoutPhysicalResidualGateIsNotBest)
{
    Toy p("well");auto c=controls("sparse_newton");c.physicalGate=0.2;
    Vec x(2);x<<0.2,0;const auto original=p.evaluate(x);
    ASSERT_LE(original.gradient.norm(),c.physicalGate);
    Engine e(p,c,x,[]{return 0.;});e.step();ASSERT_EQ(e.state().status,"ready");
    EXPECT_LT(e.state().energy,original.energy);EXPECT_GT(e.state().g.norm(),c.physicalGate);
    EXPECT_TRUE(e.state().bestValid);EXPECT_DOUBLE_EQ(e.state().bestEnergy,original.energy);
    EXPECT_EQ((e.state().bestX-x).norm(),0);
}

TEST(BenchmarkSelection, FailedIndependentVerificationInvalidatesCandidate)
{
    struct Inconsistent : Toy
    {
        Value evaluate(const Vec& x) override
        {auto v=Toy::evaluate(x);v.gradient.setConstant(calls==1?0:1);return v;}
    } p;
    auto c=controls("nonlinear_cg");Engine e(p,c,Vec::Ones(2),[]{return 0.;});e.step();
    EXPECT_EQ(e.state().status,"verification_failed");EXPECT_FALSE(e.state().bestValid);
    EXPECT_EQ(e.state().evaluations,2);EXPECT_TRUE(e.state().boundary);
}

TEST(BenchmarkCheckpoint, ActiveLineSearchCannotBeCheckpointedOrReentered)
{
    struct Callback : Toy
    {
        Engine* engine=nullptr;Config config;std::string path;int checked=0;
        Value evaluate(const Vec& x) override
        {
            EXPECT_THROW(engine->snapshot(),std::logic_error);
            EXPECT_THROW(engine->step(),std::logic_error);
            EXPECT_THROW(saveCheckpoint(path,engine->state(),config,*this),std::invalid_argument);
            ++checked;return Toy::evaluate(x);
        }
    } p;
    p.config=controls("native_lbfgs");p.path=directory()+"/unsafe.json";
    Vec x(2);x<<-1.2,1;Engine e(p,p.config,x,[]{return 0.;});p.engine=&e;e.step();
    EXPECT_GT(p.checked,0);EXPECT_TRUE(e.state().boundary);EXPECT_FALSE(std::filesystem::exists(p.path));
    EXPECT_NO_THROW(saveCheckpoint(p.path,e.snapshot(),p.config,p));
}

TEST(BenchmarkCheckpoint, FullCoarseDimensionLBFGSStateRoundtrip)
{
    struct Large : Problem
    {
        Vec diagonal;
        Large():diagonal(32267){for(int i=0;i<diagonal.size();++i)diagonal[i]=1+double(i%97)/10;}
        int size() const override{return diagonal.size();}
        std::string identity() const override{return "32267_dof_diagonal_fixture_v1";}
        Value evaluate(const Vec& x) override
        {Vec g=diagonal.cwiseProduct(x);return {0.5*x.dot(g),g};}
        Sparse hessian(const Vec&) override
        {Sparse H(size(),size());for(int i=0;i<size();++i)H.insert(i,i)=diagonal[i];H.makeCompressed();return H;}
    } p;
    Vec x(p.size());for(int i=0;i<x.size();++i)x[i]=1e-3*std::sin(i+1.0);
    auto c=controls("native_lbfgs");Engine e(p,c,x,[]{return 0.;});for(int i=0;i<12;++i)e.step();
    ASSERT_EQ(e.state().status,"ready");const auto saved=e.snapshot();const auto path=directory()+"/large.json";
    const double begin=Engine::wallClock();saveCheckpoint(path,saved,c,p);
    const auto loaded=loadCheckpoint(path,c,p,resources(saved));
    const double seconds=Engine::wallClock()-begin;
    EXPECT_EQ((saved.x-loaded.x).norm(),0);EXPECT_EQ(saved.s,loaded.s);EXPECT_EQ(saved.y,loaded.y);
    EXPECT_EQ(saved.rho,loaded.rho);EXPECT_EQ(saved.diagonal,loaded.diagonal);EXPECT_EQ(saved.trace,loaded.trace);
    std::cout<<"LARGE_CHECKPOINT_ROUNDTRIP bytes="<<std::filesystem::file_size(path)<<" seconds="<<seconds<<"\n";
    Engine resumed(p,c,loaded,[]{return 0.;});e.step();resumed.step();
    // Large Eigen reductions may round differently with a different allocation alignment.
    EXPECT_LT((e.state().x-resumed.state().x).norm(),1e-13);EXPECT_EQ(e.state().evaluations,resumed.state().evaluations);
}

TEST(BenchmarkShell, DerivativeParityConstraintsGaugeAndAllFourAdapters)
{
    tbb::global_control threads(tbb::global_control::max_allowed_parallelism,1);
    for(bool clamped:{false,true})
    {
        RectangularPlate geometry(0.0635,0.0762,0.5,{clamped,false},{false,false});geometry.setQuiet();
        BilayerMesh mesh;mesh.init(geometry,false);const int nv=mesh.getNumberOfVertices(),nd=3*nv+mesh.getNumberOfEdges();
        MaterialProperties_Iso_Constant mat(1,0.33,0.0006);
        CombinedOperator_Parametric<BilayerMesh,Material_Isotropic,bottom> bot(mat);
        CombinedOperator_Parametric<BilayerMesh,Material_Isotropic,top> topOp(mat);
        EnergyOperatorList<BilayerMesh> op({&bot,&topOp});
        auto& abar=mesh.getRestConfiguration().getFirstFundamentalForms<top>();for(auto& a:abar)a*=1.005*1.005;
        ShellProblem<BilayerMesh,decltype(op)> p(mesh,op,1,0.33,0.0006,clamped?"small_clamped_v1":"small_free_v1");
        Vec initial=Eigen::Map<Vec>(mesh.getDataPointer(),nd);
        const auto mask=ShellEquilibrium::constrainedDofMask(mesh);
        for(int i=0;i<nv;++i)if(!mask[2*nv+i])initial[2*nv+i]+=0.001*std::sin(i+1.0);
        const auto value=p.evaluate(initial);const Vec ga=p.adGradient(initial);
        EXPECT_LE((ga-value.gradient).norm(),1e-18+1e-8*value.gradient.norm());
        Vec v=Vec::Zero(nd);for(int i=0;i<nd;++i)if(!mask[i])v[i]=p.scales()[i]*std::sin(i+1.0);
        const Sparse H=p.hessian(initial);const double eps=1e-6;
        const auto plus=p.evaluate(initial+eps*v),minus=p.evaluate(initial-eps*v);
        EXPECT_LE(((plus.gradient-minus.gradient)/(2*eps)-H*v).norm(),1e-16+1e-5*(H*v).norm());
        p.restore(initial);const auto ids=p.freeDofs(initial,p.scales());
        const auto P=ShellEquilibrium::RigidModeProjector::fromMesh(mesh);
        EXPECT_EQ(ids.size(),size_t(nd-P.numberOfConstrainedDofs()-P.numberOfRigidModes()));
        p.restore(initial+1e-6*v);EXPECT_EQ(ids,p.freeDofs(initial,p.scales())); // explicit base, not lingering trial
        for(const auto& method:methods)
        {
            SCOPED_TRACE(method);p.restore(initial);auto c=controls(method);c.tolerance=5e-14;c.physicalGate=1e-13;
            Engine e(p,c,initial,[]{return 0.;});e.step();
            ASSERT_TRUE(e.state().status=="ready" || e.state().status=="gradient_target")<<e.state().trace.dump();
            EXPECT_LE(e.state().energy,value.energy);EXPECT_TRUE(e.state().x.allFinite());
            for(int i=0;i<nd;++i){if(mask[i]){EXPECT_DOUBLE_EQ(e.state().x[i],initial[i]);}}
            EXPECT_EQ((Eigen::Map<Vec>(mesh.getDataPointer(),nd)-e.state().x).norm(),0);
        }
    }
}

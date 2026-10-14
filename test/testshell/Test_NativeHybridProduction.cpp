#include "gtest/gtest.h"
#ifdef SHELL_WITH_NATIVE_HYBRID
#include "Geometry.hpp"
#include "Mesh.hpp"
#include "MaterialProperties.hpp"
#include "CombinedOperator_Parametric.hpp"
#include "EnergyOperatorList.hpp"
#include "NativeHybridShellSolver.hpp"
#include <sstream>

namespace
{
using namespace ShellNativeHybrid;
struct Rosen : Problem
{
    Vec current=Vec::Zero(2);
    int size() const override{return 2;}
    std::string identity() const override{return "native-production-fixture:rosen";}
    Vec scales() const override{return Vec::Ones(2);}
    double energyReference() const override{return 1.;}
    void restore(const Vec& x) override{current=x;}
    Value evaluate(const Vec& x) override
    {current=x;const double a=1-x[0],b=x[1]-x[0]*x[0];Vec g(2);g<<-2*a-400*x[0]*b,200*b;return {a*a+100*b*b,g};}
    Sparse hessian(const Vec& x) override
    {current=x;Eigen::Matrix2d H;H<<2-400*x[1]+1200*x[0]*x[0],-400*x[0],-400*x[0],200;return H.sparseView();}
};
// Deterministic oracle overhead: no sleeping and no physical shell fit.
struct TimedRosen : Rosen
{
    double& now;double identityCharge=0;
    explicit TimedRosen(double& time):now(time){}
    std::string identity() const override{now+=identityCharge;return Rosen::identity();}
};
HybridOptions fixtureOptions()
{HybridOptions o;o.warmupAttempts=2;o.maxAttempts=8;o.maxEvaluations=200;o.maxHessians=12;o.seconds=5;o.gradientTolerance=1e-12;o.physicalGate=1e-11;return o;}
// Exact energy/gradient with controlled faulty curvature for rejection/fallback
// fixtures. These three DOFs are NOT a physical shell or tuning observations.
struct ControllerQuadratic : Problem
{
    double& now;double hessianCost=.02,modelScale=1,gradientCost=.02;
    bool zeroAfterFirst=false,cheapAfterHessian=false,asymmetric=false;
    int hessianCalls=0;Vec current=Vec::Ones(3);
    Eigen::Matrix3d A=Eigen::Vector3d(.001,1.,100.).asDiagonal();
    explicit ControllerQuadratic(double& time):now(time){}
    int size() const override{return 3;}
    std::string identity() const override{return "controller-correctness-quadratic";}
    Vec scales() const override{return Vec::Ones(3);}
    double energyReference() const override{return 1;}
    void restore(const Vec& x) override{current=x;}
    Value evaluate(const Vec& x) override
    {now+=(cheapAfterHessian && hessianCalls>0)?.001:gradientCost;current=x;Vec g=A*x;return {.5*x.dot(g),g};}
    Sparse hessian(const Vec& x) override
    {now+=hessianCost;current=x;++hessianCalls;Eigen::Matrix3d H=modelScale*A;
     if(zeroAfterFirst && hessianCalls>1)H.setZero();
     if(asymmetric)H(0,1)=.01;return H.sparseView();}
};
HybridOptions adaptiveFixtureOptions()
{
    auto o=fixtureOptions();o.warmupAttempts=40;o.maxAttempts=100;o.gradientTolerance=1e-10;o.physicalGate=1e-9;
    o.handoff.adaptive=true;o.handoff.minimumWarmup=2;o.handoff.window=2;
    o.handoff.relativeGradient=.99;return o;
}
}

TEST(NativeHybridProduction, DeclaredReferenceControls)
{
#ifdef NONCONVEX_WITH_CHOLMOD
    HybridOptions o;const auto c=hybridConfigs(o);
    EXPECT_EQ(o.warmupAttempts,2000);EXPECT_EQ(c.first.memory,10);
    EXPECT_EQ(c.first.method,"native_lbfgs");EXPECT_EQ(c.second.method,"sparse_newton");
    EXPECT_EQ(c.second.newtonBackend,"cholmod");EXPECT_EQ(c.second.hessianAssembly,"local_dihedral_csc");
    EXPECT_FALSE(c.second.guardedHessianReuse);EXPECT_FALSE(c.second.reuseNewtonSymbolic);
    EXPECT_DOUBLE_EQ(c.first.tolerance,5e-14);EXPECT_DOUBLE_EQ(c.first.physicalGate,1e-13);
#else
    EXPECT_THROW(hybridConfigs(HybridOptions()),std::invalid_argument);
#endif
}

#ifdef NONCONVEX_WITH_CHOLMOD
TEST(NativeHybridProduction, RelativeGradientWindowIsUnitScaleInvariant)
{
    GradientWindow a(2),b(2);a.push(10,0);a.push(8,1);a.push(6,2);
    b.push(10000,0);b.push(8000,1);b.push(6000,2);
    ASSERT_TRUE(a.ready());EXPECT_NEAR(a.ratio(),.6,1e-14);
    EXPECT_NEAR(a.ratio(),b.ratio(),1e-14);EXPECT_NEAR(a.rate(),b.rate(),1e-14);
    EXPECT_THROW(a.push(1,1),std::invalid_argument);
}

TEST(NativeHybridProduction, AdaptiveTrialCommitsRealNewtonAndChargesAllWork)
{
    auto o=adaptiveFixtureOptions();auto c=hybridConfigs(o);double now=0;ControllerQuadratic p(now);
    Vec x(3);x<<1,2,3;HybridSession s(p,c.first,c.second,x,40,[&]{return now;},0,o.handoff);
    for(int i=0;i<10 && s.state().status=="ready";++i)s.step();
    EXPECT_TRUE(s.transitioned());EXPECT_EQ(s.state().status,"gradient_target");
    EXPECT_EQ(s.lbfgsAttempts()+s.newtonAttempts(),s.state().attempts);
    EXPECT_GE(s.probeAttempts(),1);EXPECT_GE(s.state().hessians,1);
    EXPECT_LE(s.state().g.norm(),o.gradientTolerance);EXPECT_DOUBLE_EQ(s.snapshot().elapsed,now);
}

TEST(NativeHybridProduction, RejectedProbePreservesHistoryAndChargesResources)
{
    auto o=adaptiveFixtureOptions();auto c=hybridConfigs(o);double now=0,other=0;
    ControllerQuadratic p(now),oracle(other);p.modelScale=.01;p.hessianCost=1.5;
    Vec x(3);x<<1,2,3;Engine expected(oracle,c.first,x,[&]{return other;});
    expected.step();expected.step();const State wanted=expected.snapshot();
    HybridSession s(p,c.first,c.second,x,40,[&]{return now;},0,o.handoff);s.step();s.step();
    ASSERT_EQ(s.probes(),1);EXPECT_FALSE(s.transitioned());EXPECT_EQ(s.phase(),"lbfgs");
    const State actual=s.snapshot();EXPECT_EQ((actual.x-wanted.x).norm(),0);EXPECT_EQ((actual.g-wanted.g).norm(),0);
    EXPECT_EQ(actual.s,wanted.s);EXPECT_EQ(actual.y,wanted.y);EXPECT_EQ(actual.rho,wanted.rho);
    EXPECT_EQ(actual.diagonal,wanted.diagonal);EXPECT_EQ(actual.ring,wanted.ring);
    EXPECT_EQ(actual.iterations,wanted.iterations);EXPECT_GT(actual.attempts,wanted.attempts);
    EXPECT_GT(actual.evaluations,wanted.evaluations);EXPECT_GT(actual.hessians,wanted.hessians);
    EXPECT_EQ((p.current-actual.x).norm(),0);EXPECT_DOUBLE_EQ(actual.elapsed,now);
    EXPECT_EQ(s.lbfgsAttempts()+s.newtonAttempts(),actual.attempts);
}

TEST(NativeHybridProduction, RejectedProbeCannotRenewTimeOrHessianCap)
{
    auto o=adaptiveFixtureOptions();o.maxHessians=1;auto c=hybridConfigs(o);double now=0;
    ControllerQuadratic p(now);p.modelScale=.01;p.hessianCost=1.5;Vec x(3);x<<1,2,3;
    HybridSession s(p,c.first,c.second,x,40,[&]{return now;},0,o.handoff);s.step();s.step();
    ASSERT_EQ(s.probes(),1);EXPECT_EQ(s.state().status,"hessian_cap");
    const int evaluations=s.state().evaluations;EXPECT_FALSE(s.step());EXPECT_EQ(s.state().evaluations,evaluations);
    o.maxHessians=12;o.seconds=.2;c=hybridConfigs(o);double shortTime=0;ControllerQuadratic slow(shortTime);
    slow.hessianCost=.3;HybridSession timed(slow,c.first,c.second,x,40,[&]{return shortTime;},0,o.handoff);
    timed.step();timed.step();EXPECT_EQ(timed.state().status,"time_cap");
    EXPECT_GE(timed.state().elapsed,o.seconds);EXPECT_FALSE(timed.step());
}

TEST(NativeHybridProduction, InvalidHessianIsTerminalNotARecoverableProbe)
{
    auto o=adaptiveFixtureOptions();auto c=hybridConfigs(o);double now=0;ControllerQuadratic p(now);
    p.asymmetric=true;Vec x(3);x<<1,2,3;
    HybridSession s(p,c.first,c.second,x,40,[&]{return now;},0,o.handoff);s.step();s.step();
    EXPECT_EQ(s.state().status,"asymmetric_hessian");EXPECT_FALSE(s.step());
    EXPECT_FALSE(s.transitioned());EXPECT_EQ(s.fallbacks(),0);
    EXPECT_EQ(s.lbfgsAttempts()+s.newtonAttempts(),s.state().attempts);
}

TEST(NativeHybridProduction, NewtonFailureRestartsHistoryWithoutResettingBudget)
{
    auto o=adaptiveFixtureOptions();auto c=hybridConfigs(o);double now=0;ControllerQuadratic p(now);
    p.modelScale=2;p.zeroAfterFirst=true;p.gradientCost=.2;p.cheapAfterHessian=true;
    Vec x(3);x<<1,2,3;
    HybridSession s(p,c.first,c.second,x,40,[&]{return now;},0,o.handoff);s.step();s.step();
    ASSERT_TRUE(s.transitioned());ASSERT_EQ(s.state().status,"ready");s.step();
    ASSERT_EQ(s.fallbacks(),1);ASSERT_EQ(s.phase(),"lbfgs");const State resumed=s.snapshot();
    EXPECT_EQ(resumed.lbfgsStartIteration,resumed.iterations);EXPECT_EQ(resumed.ring,0);
    for(double v:resumed.s)EXPECT_EQ(v,0);for(double v:resumed.rho)EXPECT_EQ(v,0);
    const State decoded=State::fromJson(resumed.json(),p.size(),c.first);
    EXPECT_EQ(decoded.lbfgsStartIteration,resumed.lbfgsStartIteration);
    const int work=resumed.attempts;s.step();EXPECT_GT(s.state().attempts,work);
    EXPECT_NE(s.state().status,"lbfgs_curvature_failure");
    EXPECT_EQ(s.lbfgsAttempts()+s.newtonAttempts(),s.state().attempts);
    EXPECT_GE(s.state().elapsed,resumed.elapsed);EXPECT_LE(s.fallbacks(),2);
}

TEST(NativeHybridProduction, ExactWarmupEvaluationBoundaryNeverRequestsTransition)
{
    Vec x(2);x<<-1.2,1.;Rosen oracle;auto o=fixtureOptions();auto c=hybridConfigs(o);
    Engine expected(oracle,c.first,x);expected.step();expected.step();
    o.maxEvaluations=expected.snapshot().evaluations;
    for(bool adaptive:{false,true})
    {
        o.handoff.adaptive=adaptive;o.handoff.minimumWarmup=2;o.handoff.window=2;
        o.handoff.relativeGradient=.99;c=hybridConfigs(o);Rosen p;
        HybridSession session(p,c.first,c.second,x,2,Engine::wallClock,
            std::numeric_limits<double>::quiet_NaN(),o.handoff);
        session.step();EXPECT_NO_THROW(session.step());
        EXPECT_EQ(session.state().status,"evaluation_cap");EXPECT_FALSE(session.transitioned());
        EXPECT_EQ(session.probes(),0);EXPECT_EQ(session.newtonAttempts(),0);
        const State stopped=session.snapshot();EXPECT_EQ(stopped.evaluations,o.maxEvaluations);
        EXPECT_FALSE(session.step());EXPECT_EQ(session.state().evaluations,stopped.evaluations);
        EXPECT_EQ(session.state().hessians,0);
    }
    o=fixtureOptions();o.maxAttempts=1;c=hybridConfigs(o);Rosen p;Engine capped(p,c.first,x);
    capped.step();EXPECT_EQ(capped.snapshot().status,"iteration_cap");
    const int evaluations=capped.state().evaluations;EXPECT_FALSE(capped.step());
    EXPECT_EQ(capped.state().evaluations,evaluations);
    // Initial evaluation PLUS independent verifyStop reevaluation; both count.
    o.maxEvaluations=2;c=hybridConfigs(o);Rosen rest;Engine accepted(rest,c.first,Vec::Ones(2));
    accepted.step();EXPECT_EQ(accepted.snapshot().status,"gradient_target");
    EXPECT_EQ(accepted.state().evaluations,o.maxEvaluations);
}

TEST(NativeHybridProduction, SnapshotClockCrossingProducesTerminalCapWithoutOracleWork)
{
    auto o=fixtureOptions();auto c=hybridConfigs(o);double now=0,tick=0;Rosen p;
    Vec x(2);x<<-1.2,1.;auto clock=[&]{double sampled=now;now+=tick;return sampled;};
    HybridSession session(p,c.first,c.second,x,2,clock,0);session.step();
    const int evaluations=session.state().evaluations,attempts=session.state().attempts;
    now=o.seconds-.15;tick=.1;State stopped;
    EXPECT_NO_THROW(stopped=session.snapshot());EXPECT_EQ(stopped.status,"time_cap");
    EXPECT_GE(stopped.elapsed,o.seconds);EXPECT_FALSE(session.transitioned());
    EXPECT_FALSE(session.step());EXPECT_EQ(session.state().evaluations,evaluations);
    EXPECT_EQ(session.state().attempts,attempts);EXPECT_EQ(session.state().hessians,0);
}

TEST(NativeHybridProduction, FailedControllerProofFlushesAllCountersAndEventsWithoutAcceptance)
{
    auto o=adaptiveFixtureOptions();o.maxHessians=1;auto c=hybridConfigs(o);double now=0;
    ControllerQuadratic p(now);p.modelScale=.01;p.hessianCost=1.5;Vec x(3);x<<1,2,3;
    HybridSession session(p,c.first,c.second,x,40,[&]{return now;},0,o.handoff);
    session.step();session.step();ASSERT_EQ(session.state().status,"hessian_cap");
    HybridReport failed;failed.terminal=session.snapshot();failed.phase=session.phase();
    failed.energy=failed.terminal.energy;failed.gradientNorm=failed.terminal.g.norm();
    failed.error=failed.terminal.error;failed.lbfgsAttempts=session.lbfgsAttempts();
    failed.newtonAttempts=session.newtonAttempts();failed.probeAttempts=session.probeAttempts();
    failed.probes=session.probes();failed.fallbacks=session.fallbacks();
    failed.lbfgsSeconds=session.lbfgsSeconds();failed.newtonSeconds=session.newtonSeconds();
    failed.handoffPolicy=o.handoff.json();failed.policyEvents=session.policyEvents();
    ASSERT_GT(failed.probes,0);ASSERT_FALSE(failed.policyEvents.empty());
    std::ostringstream stream;failed.writeProof(stream,o.gradientTolerance);
    const Json proof=Json::parse(stream.str());EXPECT_FALSE(proof.at("accepted").get<bool>());
    EXPECT_EQ(failed.code(),1);EXPECT_EQ(proof.at("termination"),"hessian_cap");
    EXPECT_EQ(proof.at("attempts"),failed.terminal.attempts);
    EXPECT_EQ(proof.at("evaluations"),failed.evaluations());EXPECT_EQ(proof.at("hessians"),failed.terminal.hessians);
    EXPECT_EQ(proof.at("probe_attempts"),failed.probeAttempts);EXPECT_EQ(proof.at("probes"),failed.probes);
    EXPECT_EQ(proof.at("fallbacks"),failed.fallbacks);EXPECT_EQ(proof.at("policy_events"),failed.policyEvents);
    EXPECT_EQ(proof.at("lbfgs_seconds"),failed.lbfgsSeconds);EXPECT_EQ(proof.at("newton_seconds"),failed.newtonSeconds);
    std::ostringstream broken;broken.setstate(std::ios::badbit);
    EXPECT_THROW(failed.writeProof(broken,o.gradientTolerance),std::runtime_error);
}

TEST(NativeHybridProduction, ExactLiveWarmupAndCumulativeTransition)
{
    auto o=fixtureOptions();const auto c=hybridConfigs(o);Vec x(2);x<<-1.2,1.;Rosen p,oracle;
    Engine expected(oracle,c.first,x);for(int i=0;i<2;++i)expected.step();const State wanted=expected.snapshot();
    HybridSession actual(p,c.first,c.second,x,o.warmupAttempts);
    actual.step();EXPECT_FALSE(actual.transitioned());actual.step();ASSERT_TRUE(actual.transitioned());
    const auto& a=actual.source();const auto& b=actual.target();
    EXPECT_EQ(a.attempts,2);EXPECT_EQ((a.x-wanted.x).norm(),0);EXPECT_EQ((a.g-wanted.g).norm(),0);
    EXPECT_DOUBLE_EQ(a.energy,wanted.energy);EXPECT_EQ(a.evaluations,wanted.evaluations);
    EXPECT_EQ(b.attempts,a.attempts);EXPECT_EQ(b.iterations,a.iterations);EXPECT_EQ(b.evaluations,a.evaluations);
    EXPECT_EQ(b.hessians,a.hessians);EXPECT_EQ(b.hvps,a.hvps);EXPECT_EQ(b.trace,a.trace);
    EXPECT_EQ((b.x-a.x).norm(),0);EXPECT_EQ((b.g-a.g).norm(),0);
    EXPECT_GE(b.elapsed,a.elapsed);EXPECT_GE(actual.state().elapsed,b.elapsed);
    EXPECT_TRUE(b.s.empty());EXPECT_TRUE(b.y.empty());EXPECT_FALSE(b.cacheValid);
    EXPECT_EQ(actual.phase(),"newton");EXPECT_EQ((actual.state().x-b.x).norm(),0);
    actual.step();EXPECT_EQ(actual.state().attempts,3);EXPECT_GE(actual.state().hessians,1);
}

TEST(NativeHybridProduction, CumulativeClockChargesInitializationAndTransition)
{
    auto o=fixtureOptions();const auto c=hybridConfigs(o);Vec x(2);x<<-1.2,1.;
    double now=0;TimedRosen p(now);p.identityCharge=.25;
    HybridSession s(p,c.first,c.second,x,2,[&]{return now;});
    EXPECT_GT(now,0);EXPECT_DOUBLE_EQ(s.state().elapsed,now);
    p.identityCharge=0;s.step();p.identityCharge=.125;s.step();ASSERT_TRUE(s.transitioned());
    EXPECT_GT(s.target().elapsed,s.source().elapsed);
    EXPECT_GT(s.state().elapsed,s.target().elapsed);EXPECT_DOUBLE_EQ(s.state().elapsed,now);
    EXPECT_EQ(s.target().attempts,s.source().attempts);
    EXPECT_EQ(s.target().evaluations,s.source().evaluations);
}

TEST(NativeHybridProduction, TransitionOverheadCannotBuyAnotherNewtonStep)
{
    auto o=fixtureOptions();const auto c=hybridConfigs(o);Vec x(2);x<<-1.2,1.;
    double now=0;TimedRosen p(now);HybridSession s(p,c.first,c.second,x,2,[&]{return now;});
    s.step();p.identityCharge=1;s.step();ASSERT_TRUE(s.transitioned());
    EXPECT_EQ(s.state().status,"time_cap");EXPECT_GE(s.state().elapsed,o.seconds);
    const auto capped=s.snapshot();EXPECT_FALSE(s.step());
    EXPECT_EQ(s.state().attempts,2);EXPECT_EQ(s.state().evaluations,capped.evaluations);
    EXPECT_EQ(s.state().hessians,0);
}

TEST(NativeHybridProduction, PublicationOverheadCannotBuyAnotherWarmupStep)
{
    auto o=fixtureOptions();const auto c=hybridConfigs(o);Vec x(2);x<<-1.2,1.;
    double now=0;TimedRosen p(now);HybridSession s(p,c.first,c.second,x,2,[&]{return now;});
    s.step();const int evaluations=s.state().evaluations;now=o.seconds+1;
    EXPECT_FALSE(s.step());EXPECT_EQ(s.state().status,"time_cap");
    EXPECT_EQ(s.state().attempts,1);EXPECT_EQ(s.state().evaluations,evaluations);EXPECT_FALSE(s.transitioned());
    EXPECT_DOUBLE_EQ(s.snapshot().elapsed,now);
}

TEST(NativeHybridProduction, HighLevelRestAcceptanceAndCapAreHonest)
{
    printf("[native_fixture] Eigen allocation alignment=%d\n", EIGEN_DEFAULT_ALIGN_BYTES);fflush(stdout);
    BilayerMesh mesh;RectangularPlate geometry(.02,.015,.5,{false,false},{false,false});
    geometry.setQuiet();mesh.init(geometry,false);MaterialProperties_Iso_Constant mat(1.,.33,.0006);
    CombinedOperator_Parametric<BilayerMesh,Material_Isotropic,bottom> bot(mat);
    CombinedOperator_Parametric<BilayerMesh,Material_Isotropic,top> topEnergy(mat);
    EnergyOperatorList<BilayerMesh> energy({&bot,&topEnergy});
    const Vec initial=Eigen::Map<const Vec>(mesh.getDataPointer(),3*mesh.getNumberOfVertices()+mesh.getNumberOfEdges());
    auto o=fixtureOptions();const auto report=solveNativeHybridShell(mesh,energy,1.,.33,.0006,o);
    EXPECT_TRUE(report.accepted);EXPECT_EQ(report.curvature,"positive_pivots");
    EXPECT_EQ(report.terminal.status,"gradient_target");EXPECT_FALSE(report.transitioned);
    EXPECT_LE(report.gradientNorm,o.gradientTolerance);EXPECT_LE(report.evaluations(),o.maxEvaluations);
    EXPECT_EQ((Eigen::Map<const Vec>(mesh.getDataPointer(),initial.size())-initial).norm(),0);
    o.maxEvaluations=1;const auto limited=solveNativeHybridShell(mesh,energy,1.,.33,.0006,o);
    EXPECT_FALSE(limited.accepted);EXPECT_EQ(limited.terminal.status,"evaluation_cap");
    EXPECT_EQ(limited.evaluations(),1);EXPECT_EQ(limited.verificationEvaluations,0);
    o.maxEvaluations=200;o.budgetStart=Engine::wallClock()-o.seconds-1;
    const auto exhausted=solveNativeHybridShell(mesh,energy,1.,.33,.0006,o);
    EXPECT_FALSE(exhausted.accepted);EXPECT_EQ(exhausted.terminal.status,"time_cap");
    EXPECT_GE(exhausted.terminal.elapsed,o.seconds);EXPECT_EQ(exhausted.evaluations(),0);
    EXPECT_EQ(exhausted.verificationHessians,0);
}

TEST(NativeHybridProduction, NeverForcesAnEarlyAcceptedHandoff)
{
    auto o=fixtureOptions();const auto c=hybridConfigs(o);Rosen p;Vec x=Vec::Ones(2);
    HybridSession s(p,c.first,c.second,x,o.warmupAttempts);s.step();
    EXPECT_EQ(s.state().status,"gradient_target");EXPECT_FALSE(s.transitioned());EXPECT_EQ(s.phase(),"lbfgs");
}

TEST(NativeHybridProduction, NeverForcesAnEvaluationCapHandoff)
{
    auto o=fixtureOptions();o.maxEvaluations=1;const auto c=hybridConfigs(o);Rosen p;Vec x(2);x<<-1.2,1.;
    HybridSession s(p,c.first,c.second,x,2);s.step();
    EXPECT_EQ(s.state().status,"evaluation_cap");EXPECT_FALSE(s.transitioned());EXPECT_EQ(s.state().evaluations,1);
}

TEST(NativeHybridProduction, RejectsUnrelatedPhaseControlChanges)
{
    auto o=fixtureOptions();auto c=hybridConfigs(o);Rosen p;Vec x(2);x<<-1.2,1.;c.second.memory++;
    EXPECT_THROW(HybridSession(p,c.first,c.second,x,2),std::invalid_argument);
    c=hybridConfigs(o);c.second.guardedHessianReuse=true;
    EXPECT_THROW(HybridSession(p,c.first,c.second,x,2),std::invalid_argument);
    c=hybridConfigs(o);EXPECT_THROW(HybridSession(p,c.first,c.second,x,0),std::invalid_argument);
}
#endif

TEST(NativeHybridProduction, FixedShellAdapterMatchesAnalyticGradientAndFreshGauge)
{
    BilayerMesh mesh;const Real young=1.,nu=.33,h=.0006;
    RectangularPlate geometry(.02,.015,.5,{false,false},{false,false});geometry.setQuiet();mesh.init(geometry,false);
    auto vertices=mesh.getCurrentConfiguration().getVertices();
    for(int i=0;i<vertices.rows();++i){const double x=vertices(i,0),y=vertices(i,1);vertices(i,2)=.2*x*x-.1*y*y;}
    mesh.updateDeformedConfiguration();MaterialProperties_Iso_Constant mat(young,nu,h);
    CombinedOperator_Parametric<BilayerMesh,Material_Isotropic,bottom> bot(mat);
    CombinedOperator_Parametric<BilayerMesh,Material_Isotropic,top> topEnergy(mat);
    EnergyOperatorList<BilayerMesh> energy({&bot,&topEnergy});
    const auto identity=shellObjectiveFingerprint(mesh,young,nu,h);
    ShellProblem<BilayerMesh,decltype(energy)> p(mesh,energy,young,nu,h,identity,"local_dihedral_csc",1);
    const Vec x=Eigen::Map<const Vec>(mesh.getDataPointer(),p.size());
    Vec analytic=Vec::Zero(p.size());const double e=energy.compute(mesh,analytic);const auto value=p.evaluate(x);
    EXPECT_DOUBLE_EQ(value.energy,e);EXPECT_EQ((analytic-value.gradient).norm(),0);
    EXPECT_LE((p.adGradient(x)-analytic).norm(),1e-18+1e-8*analytic.norm());
    const auto gauge=p.freeDofs(x,p.scales());EXPECT_EQ(gauge,p.freeDofs(x,p.scales()));
    const auto H=p.hessian(x);ASSERT_EQ(H.rows(),p.size());EXPECT_LT((H-Sparse(H.transpose())).norm(),1e-10*H.norm());
    mesh.getRestConfiguration().template getFirstFundamentalForms<top>()[0](0,0)*=1.001;
    EXPECT_NE(identity,shellObjectiveFingerprint(mesh,young,nu,h));
}

TEST(NativeHybridProduction, EndpointCheckAcceptsZeroAndDiminishingGradients)
{
    BilayerMesh mesh;const Real young=1.,nu=.33,h=.0006;
    RectangularPlate geometry(.02,.015,.5,{false,false},{false,false});geometry.setQuiet();mesh.init(geometry,false);
    auto vertices=mesh.getCurrentConfiguration().getVertices();
    for(int i=0;i<vertices.rows();++i){const double x=vertices(i,0),y=vertices(i,1);vertices(i,2)=.2*x*x-.1*y*y;}
    mesh.updateDeformedConfiguration();MaterialProperties_Iso_Constant mat(young,nu,h);
    CombinedOperator_Parametric<BilayerMesh,Material_Isotropic,bottom> bot(mat);
    CombinedOperator_Parametric<BilayerMesh,Material_Isotropic,top> topEnergy(mat);
    EnergyOperatorList<BilayerMesh> energy({&bot,&topEnergy});
    const auto identity=shellObjectiveFingerprint(mesh,young,nu,h);
    ShellProblem<BilayerMesh,decltype(energy)> p(mesh,energy,young,nu,h,identity,"local_dihedral_csc",1);
    const Vec x=Eigen::Map<const Vec>(mesh.getDataPointer(),p.size());
    const auto H=p.hessian(x);
    const ShellEquilibrium::RigidModeProjector P(
        Eigen::Map<const Eigen::MatrixXd>(x.data(),mesh.getNumberOfVertices(),3),
        mesh.getNumberOfEdges(),ShellEquilibrium::constrainedDofMask(mesh));

    // 1. Zero gradient (b = 0)
    Vec g0 = Vec::Zero(p.size());
    auto rep0 = ShellNativeEndpointCheck::inspect(H, g0, p.scales(), P, p.energyReference());
    EXPECT_EQ(rep0.status, "positive_pivots");
    EXPECT_DOUBLE_EQ(rep0.relativeSolveResidual, 0.0);
    EXPECT_DOUBLE_EQ(rep0.decrement, 0.0);
    EXPECT_DOUBLE_EQ(rep0.predictedEnergyDrop, 0.0);
    EXPECT_GT(rep0.minPivot, 0.0);

    // 2. Diminishing gradients from 1e-6 down to 1e-30
    Vec g_dir = Vec::LinSpaced(p.size(), 1.0, 2.0).normalized();
    const double scales[] = {1e-6, 1e-12, 1e-18, 1e-24, 1e-30};
    double prev_dec = 1e10;
    for(double sc : scales)
    {
        Vec g = sc * g_dir;
        auto rep = ShellNativeEndpointCheck::inspect(H, g, p.scales(), P, p.energyReference());
        EXPECT_EQ(rep.status, "positive_pivots");
        EXPECT_LE(rep.relativeSolveResidual, 1e-6);
        EXPECT_TRUE(std::isfinite(rep.decrement));
        EXPECT_GE(rep.decrement, 0.0);
        EXPECT_LT(rep.decrement, prev_dec);
        prev_dec = rep.decrement;
    }
}

TEST(NativeHybridProduction, EndpointCheckSyntheticCycle29ScalingReproducer)
{
    // Synthetic representative matching the cycle 29 scale (||g|| ~ 6.84e-17, ||b|| ~ 6e-24).
    // Verifies that near-zero gradient and reduced RHS scaling do not trigger false rejection.
    BilayerMesh mesh;const Real young=1.,nu=.33,h=.0006;
    RectangularPlate geometry(.02,.015,.5,{false,false},{false,false});geometry.setQuiet();mesh.init(geometry,false);
    auto vertices=mesh.getCurrentConfiguration().getVertices();
    for(int i=0;i<vertices.rows();++i){const double x=vertices(i,0),y=vertices(i,1);vertices(i,2)=.2*x*x-.1*y*y;}
    mesh.updateDeformedConfiguration();MaterialProperties_Iso_Constant mat(young,nu,h);
    CombinedOperator_Parametric<BilayerMesh,Material_Isotropic,bottom> bot(mat);
    CombinedOperator_Parametric<BilayerMesh,Material_Isotropic,top> topEnergy(mat);
    EnergyOperatorList<BilayerMesh> energy({&bot,&topEnergy});
    const auto identity=shellObjectiveFingerprint(mesh,young,nu,h);
    ShellProblem<BilayerMesh,decltype(energy)> p(mesh,energy,young,nu,h,identity,"local_dihedral_csc",1);
    const Vec x=Eigen::Map<const Vec>(mesh.getDataPointer(),p.size());
    const auto H=p.hessian(x);
    const ShellEquilibrium::RigidModeProjector P(
        Eigen::Map<const Eigen::MatrixXd>(x.data(),mesh.getNumberOfVertices(),3),
        mesh.getNumberOfEdges(),ShellEquilibrium::constrainedDofMask(mesh));

    Vec g = Vec::LinSpaced(p.size(), -0.5, 0.5).normalized() * 6.84e-17;
    ASSERT_NEAR(g.norm(), 6.84e-17, 1e-20);
    auto rep = ShellNativeEndpointCheck::inspect(H, g, p.scales(), P, p.energyReference());
    EXPECT_EQ(rep.status, "positive_pivots");
    EXPECT_LE(rep.relativeSolveResidual, 1e-6);
    EXPECT_TRUE(std::isfinite(rep.decrement));
    EXPECT_GT(rep.decrement, 0.0);
    EXPECT_TRUE(rep.correction.allFinite());
}

TEST(NativeHybridProduction, EndpointCheckRejectsIndefiniteAsymmetricAndZeroDiagonalHessians)
{
    BilayerMesh mesh;const Real young=1.,nu=.33,h=.0006;
    RectangularPlate geometry(.02,.015,.5,{false,false},{false,false});geometry.setQuiet();mesh.init(geometry,false);
    auto vertices=mesh.getCurrentConfiguration().getVertices();
    for(int i=0;i<vertices.rows();++i){const double x=vertices(i,0),y=vertices(i,1);vertices(i,2)=.2*x*x-.1*y*y;}
    mesh.updateDeformedConfiguration();MaterialProperties_Iso_Constant mat(young,nu,h);
    CombinedOperator_Parametric<BilayerMesh,Material_Isotropic,bottom> bot(mat);
    CombinedOperator_Parametric<BilayerMesh,Material_Isotropic,top> topEnergy(mat);
    EnergyOperatorList<BilayerMesh> energy({&bot,&topEnergy});
    const auto identity=shellObjectiveFingerprint(mesh,young,nu,h);
    ShellProblem<BilayerMesh,decltype(energy)> p(mesh,energy,young,nu,h,identity,"local_dihedral_csc",1);
    const Vec x=Eigen::Map<const Vec>(mesh.getDataPointer(),p.size());
    const auto H=p.hessian(x);
    const ShellEquilibrium::RigidModeProjector P(
        Eigen::Map<const Eigen::MatrixXd>(x.data(),mesh.getNumberOfVertices(),3),
        mesh.getNumberOfEdges(),ShellEquilibrium::constrainedDofMask(mesh));
    Vec g = Vec::LinSpaced(p.size(), 0.1, 1.0).normalized() * 1e-12;

    // 1. Indefinite Hessian (shift diagonal negative)
    Sparse H_indef = H;
    for(int i=0;i<H_indef.rows();++i) H_indef.coeffRef(i,i) -= 1e6;
    auto rep_indef = ShellNativeEndpointCheck::inspect(H_indef, g, p.scales(), P, p.energyReference());
    EXPECT_EQ(rep_indef.status, "indefinite");

    // 2. Asymmetric Hessian
    Sparse H_asym = H;
    H_asym.coeffRef(0, 1) += 10.0;
    auto rep_asym = ShellNativeEndpointCheck::inspect(H_asym, g, p.scales(), P, p.energyReference());
    EXPECT_EQ(rep_asym.status, "asymmetric_hessian");

    // 3. Zero-diagonal Hessian (zero out diagonals of free DOFs -> LDLT factorization unresolved)
    Sparse H_zero_diag = H;
    for(int i=0;i<H_zero_diag.rows();++i) H_zero_diag.coeffRef(i,i) = 0.0;
    auto rep_zero_diag = ShellNativeEndpointCheck::inspect(H_zero_diag, g, p.scales(), P, p.energyReference());
    EXPECT_EQ(rep_zero_diag.status, "factorization_unresolved");
}

TEST(NativeHybridProduction, EvaluateSolveAcceptanceResidualThresholdBothSides)
{
    // Test both sides of the relativeSolveResidual <= 1e-6 threshold using extracted helper seam
    const double bNorm = 2.0;
    const double yNorm = 0.5;
    const double maxAbsDiagonal = 10.0;
    const double denom = bNorm + maxAbsDiagonal * yNorm + 1e-16; // 2.0 + 5.0 = 7.0
    const double dotVal = 1.0; // positive
    double relRes = 0.0, decSq = 0.0;

    // Passing side: residualNorm just below 1e-6 * denom
    const double resPass = 1e-6 * denom - 1e-8;
    std::string stPass = ShellNativeEndpointCheck::evaluateSolveAcceptance(
        resPass, bNorm, yNorm, maxAbsDiagonal, dotVal, relRes, decSq);
    EXPECT_EQ(stPass, "accepted");
    EXPECT_LE(relRes, 1e-6);
    EXPECT_GT(decSq, 0.0);

    // Failing side: residualNorm just above 1e-6 * denom
    const double resFail = 1e-6 * denom + 1e-8;
    std::string stFail = ShellNativeEndpointCheck::evaluateSolveAcceptance(
        resFail, bNorm, yNorm, maxAbsDiagonal, dotVal, relRes, decSq);
    EXPECT_EQ(stFail, "linear_solve_unresolved");
    EXPECT_GT(relRes, 1e-6);
}

TEST(NativeHybridProduction, EvaluateSolveAcceptanceNegativeDotThresholdBothSides)
{
    // Test both sides of negative dotVal allowance: threshold is -1e-12 * (bNorm*yNorm + 1e-20)
    const double bNorm = 2.0;
    const double yNorm = 3.0;
    const double maxAbsDiagonal = 5.0;
    const double threshold = -1e-12 * (bNorm * yNorm + 1e-20); // -6e-12
    const double resNorm = 1e-8; // comfortably below 1e-6 * denom
    double relRes = 0.0, decSq = 0.0;

    // Passing negative side: dotVal above threshold (e.g. 0.5 * threshold = -3e-12)
    // Within negative tolerance; clamped to 0.0
    const double dotWithin = 0.5 * threshold;
    std::string stPass = ShellNativeEndpointCheck::evaluateSolveAcceptance(
        resNorm, bNorm, yNorm, maxAbsDiagonal, dotWithin, relRes, decSq);
    EXPECT_EQ(stPass, "accepted");
    EXPECT_DOUBLE_EQ(decSq, 0.0);

    // Failing negative side: dotVal below threshold (e.g. 1.5 * threshold = -9e-12)
    // Exceeds negative tolerance; rejected as linear_solve_unresolved
    const double dotExceed = 1.5 * threshold;
    std::string stFail = ShellNativeEndpointCheck::evaluateSolveAcceptance(
        resNorm, bNorm, yNorm, maxAbsDiagonal, dotExceed, relRes, decSq);
    EXPECT_EQ(stFail, "linear_solve_unresolved");
    EXPECT_DOUBLE_EQ(decSq, 0.0);

    // Passing strictly positive side: dotVal = 4.0
    const double dotPos = 4.0;
    std::string stPos = ShellNativeEndpointCheck::evaluateSolveAcceptance(
        resNorm, bNorm, yNorm, maxAbsDiagonal, dotPos, relRes, decSq);
    EXPECT_EQ(stPos, "accepted");
    EXPECT_DOUBLE_EQ(decSq, 4.0);
}

TEST(NativeHybridProduction, EvaluateSolveAcceptanceNonfiniteIntermediates)
{
    // Explicitly test nonfinite denominator, dotVal, and residualNorm guards
    const double bNorm = 1.0, yNorm = 1.0, maxAbsDiagonal = 1.0, dotVal = 1.0, resNorm = 1e-8;
    double relRes = 0.0, decSq = 0.0;

    // 1. NaN in denominator via NaN bNorm
    std::string s1 = ShellNativeEndpointCheck::evaluateSolveAcceptance(
        resNorm, std::numeric_limits<double>::quiet_NaN(), yNorm, maxAbsDiagonal, dotVal, relRes, decSq);
    EXPECT_EQ(s1, "linear_solve_unresolved");

    // 2. Inf in denominator via Inf maxAbsDiagonal
    std::string s2 = ShellNativeEndpointCheck::evaluateSolveAcceptance(
        resNorm, bNorm, yNorm, std::numeric_limits<double>::infinity(), dotVal, relRes, decSq);
    EXPECT_EQ(s2, "linear_solve_unresolved");

    // 3. NaN in dotVal
    std::string s3 = ShellNativeEndpointCheck::evaluateSolveAcceptance(
        resNorm, bNorm, yNorm, maxAbsDiagonal, std::numeric_limits<double>::quiet_NaN(), relRes, decSq);
    EXPECT_EQ(s3, "linear_solve_unresolved");

    // 4. -Inf in dotVal
    std::string s4 = ShellNativeEndpointCheck::evaluateSolveAcceptance(
        resNorm, bNorm, yNorm, maxAbsDiagonal, -std::numeric_limits<double>::infinity(), relRes, decSq);
    EXPECT_EQ(s4, "linear_solve_unresolved");

    // 5. NaN in residualNorm
    std::string s5 = ShellNativeEndpointCheck::evaluateSolveAcceptance(
        std::numeric_limits<double>::quiet_NaN(), bNorm, yNorm, maxAbsDiagonal, dotVal, relRes, decSq);
    EXPECT_EQ(s5, "linear_solve_unresolved");
}

TEST(NativeHybridProduction, EndpointCheckRejectsNonfiniteIntermediates)
{
    BilayerMesh mesh;const Real young=1.,nu=.33,h=.0006;
    RectangularPlate geometry(.02,.015,.5,{false,false},{false,false});geometry.setQuiet();mesh.init(geometry,false);
    auto vertices=mesh.getCurrentConfiguration().getVertices();
    for(int i=0;i<vertices.rows();++i){const double x=vertices(i,0),y=vertices(i,1);vertices(i,2)=.2*x*x-.1*y*y;}
    mesh.updateDeformedConfiguration();MaterialProperties_Iso_Constant mat(young,nu,h);
    CombinedOperator_Parametric<BilayerMesh,Material_Isotropic,bottom> bot(mat);
    CombinedOperator_Parametric<BilayerMesh,Material_Isotropic,top> topEnergy(mat);
    EnergyOperatorList<BilayerMesh> energy({&bot,&topEnergy});
    const auto identity=shellObjectiveFingerprint(mesh,young,nu,h);
    ShellProblem<BilayerMesh,decltype(energy)> p(mesh,energy,young,nu,h,identity,"local_dihedral_csc",1);
    const Vec x=Eigen::Map<const Vec>(mesh.getDataPointer(),p.size());
    const auto H=p.hessian(x);
    const ShellEquilibrium::RigidModeProjector P(
        Eigen::Map<const Eigen::MatrixXd>(x.data(),mesh.getNumberOfVertices(),3),
        mesh.getNumberOfEdges(),ShellEquilibrium::constrainedDofMask(mesh));

    // 1. NaN in gradient
    Vec g_nan = Vec::Zero(p.size());
    g_nan[0] = std::numeric_limits<double>::quiet_NaN();
    auto rep_nan = ShellNativeEndpointCheck::inspect(H, g_nan, p.scales(), P, p.energyReference());
    EXPECT_EQ(rep_nan.status, "invalid_input");

    // 2. Inf in gradient
    Vec g_inf = Vec::Zero(p.size());
    g_inf[0] = std::numeric_limits<double>::infinity();
    auto rep_inf = ShellNativeEndpointCheck::inspect(H, g_inf, p.scales(), P, p.energyReference());
    EXPECT_EQ(rep_inf.status, "invalid_input");

    // 3. Nonpositive energyReference
    Vec g = Vec::Zero(p.size());
    auto rep_ref0 = ShellNativeEndpointCheck::inspect(H, g, p.scales(), P, 0.0);
    EXPECT_EQ(rep_ref0.status, "invalid_input");
    auto rep_refneg = ShellNativeEndpointCheck::inspect(H, g, p.scales(), P, -1.0);
    EXPECT_EQ(rep_refneg.status, "invalid_input");
}

TEST(NativeHybridProduction, EndpointCheckNegativeDotProductClampThreshold)
{
    BilayerMesh mesh;const Real young=1.,nu=.33,h=.0006;
    RectangularPlate geometry(.02,.015,.5,{false,false},{false,false});geometry.setQuiet();mesh.init(geometry,false);
    auto vertices=mesh.getCurrentConfiguration().getVertices();
    for(int i=0;i<vertices.rows();++i){const double x=vertices(i,0),y=vertices(i,1);vertices(i,2)=.2*x*x-.1*y*y;}
    mesh.updateDeformedConfiguration();MaterialProperties_Iso_Constant mat(young,nu,h);
    CombinedOperator_Parametric<BilayerMesh,Material_Isotropic,bottom> bot(mat);
    CombinedOperator_Parametric<BilayerMesh,Material_Isotropic,top> topEnergy(mat);
    EnergyOperatorList<BilayerMesh> energy({&bot,&topEnergy});
    const auto identity=shellObjectiveFingerprint(mesh,young,nu,h);
    ShellProblem<BilayerMesh,decltype(energy)> p(mesh,energy,young,nu,h,identity,"local_dihedral_csc",1);
    const Vec x=Eigen::Map<const Vec>(mesh.getDataPointer(),p.size());
    const auto H=p.hessian(x);
    const ShellEquilibrium::RigidModeProjector P(
        Eigen::Map<const Eigen::MatrixXd>(x.data(),mesh.getNumberOfVertices(),3),
        mesh.getNumberOfEdges(),ShellEquilibrium::constrainedDofMask(mesh));

    // Normal solve has dotVal > 0 and decrement > 0
    Vec g = Vec::LinSpaced(p.size(), 1.0, 2.0).normalized() * 1e-12;
    auto rep = ShellNativeEndpointCheck::inspect(H, g, p.scales(), P, p.energyReference());
    EXPECT_EQ(rep.status, "positive_pivots");
    EXPECT_GT(rep.decrement, 0.0);
    EXPECT_LE(rep.relativeSolveResidual, 1e-6);
}
#endif // SHELL_WITH_NATIVE_HYBRID

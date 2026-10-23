#ifndef NATIVE_HYBRID_SHELL_SOLVER_HPP
#define NATIVE_HYBRID_SHELL_SOLVER_HPP

#include "ReferenceHybrid/ShellProblem.hpp"
#include "ReferenceHybrid/Transition.hpp"
#include "ReferenceHybrid/EndpointCheck.hpp"
#include "NativeHybridHandoff.hpp"
#include <tbb/global_control.h>
#include <ostream>

namespace ShellNativeHybrid
{
// Fixed-cut mode preserves the cleared recipe. Opt-in adaptation requests
// counted Newton trials and never resets the live cumulative resource ledger.
// This is an in-process controller, NOT a disk-restart policy.
class HybridSession
{
    Problem& problem;
    Config lbfgs, newton;
    int cut;
    std::unique_ptr<Engine> engine;
    std::string phase_="lbfgs";
    bool transitioned_=false;
    State source_, target_;
    std::function<double()> clock;
    double origin;
    HandoffPolicy policy;
    GradientWindow lbWindow, ntWindow;
    int lbAttempts_=0, ntAttempts_=0, probeAttempts_=0, probes_=0, fallbacks_=0;
    int epochLbSteps=0, nextCheck=0, nextProbe=0, retryInterval=0, ntSteps=0, ntStalls=0, ntIndefinite=0;
    double lbSeconds_=0, ntSeconds_=0, initialNorm=0;
    Json events_=Json::array();
    void chargeBudget(){engine->chargeElapsed(elapsed());}
    static bool isInnerFailure(const std::string& status)
    {return status=="line_search_failed" || status=="regularization_failed" ||
            status=="linear_solve_failed" || status=="non_descent";}
    static bool recoverable(const std::string& status)
    {return isInnerFailure(status) || status=="newton_stagnation" || status=="indefinite_curvature";}
    static bool resourceStop(const std::string& status)
    {return status=="time_cap" || status=="iteration_cap" || status=="evaluation_cap" || status=="hessian_cap" || status=="newton_stagnation" || status=="indefinite_curvature";}
    void restartWarmup(const std::string& reason)
    {
        if(fallbacks_>=HandoffPolicy::maximumFallbacks)return;
        State restart=snapshot();
        events_.push_back({{"event","newton_to_lbfgs"},{"reason",reason},{"status",restart.status},
            {"error",restart.error},{"gradient_norm",restart.g.norm()},{"attempts",restart.attempts},
            {"evaluations",restart.evaluations},{"hessians",restart.hessians},{"elapsed",elapsed()}});
        restart.status="ready";restart.error.clear();restart.recipeId=lbfgs.json().dump();
        restart.lbfgsStartIteration=restart.iterations;restart.ring=0;
        restart.s.assign(problem.size()*lbfgs.memory,0);restart.y=restart.s;
        restart.rho.assign(lbfgs.memory,0);restart.diagonal.assign(problem.size(),1);
        restart.cacheValid=false;restart.hessianReuseAge=0;restart.A.resize(0,0);restart.free.clear();
        restart.lastAlpha=1;restart.lastShift=0;restart.elapsed=std::max(restart.elapsed,elapsed());
        engine=std::make_unique<Engine>(problem,lbfgs,restart,clock);
        auto floor=resources(restart);floor.cellCompleted=false;engine->chargeResources(floor);
        phase_="lbfgs";++fallbacks_;epochLbSteps=0;nextCheck=policy.minimumWarmup;
        nextProbe=policy.minimumWarmup;retryInterval=policy.window;lbWindow.clear();
        initialNorm=0;chargeBudget();
    }
    bool requestProbe(std::string& reason)
    {
        if(probes_>=HandoffPolicy::maximumProbes || epochLbSteps<policy.minimumWarmup || epochLbSteps<nextProbe)
            return false;
        const bool guard=lbAttempts_>=cut;
        if(!guard && epochLbSteps<nextCheck)return false;
        nextCheck=epochLbSteps+policy.window;
        const double g=engine->state().g.norm();
        if(guard)reason="warmup_guard";
        else if(initialNorm>0 && g<=policy.relativeGradient*initialNorm)reason="relative_gradient";
        else if(lbWindow.ready() && lbWindow.ratio()>=policy.plateauRatio)reason="gradient_plateau";
        else return false;
        return true;
    }
    void probeNewton(const std::string& reason)
    {
        const double begin=elapsed();const State saved=snapshot();
        if(saved.status!="ready")return;
        const State transformed=transitionToNewton(problem,lbfgs,newton,saved,saved.attempts);
        State live=transformed;live.elapsed=std::max(live.elapsed,elapsed());
        auto trial=std::make_unique<Engine>(problem,newton,live,clock);
        trial->chargeElapsed(elapsed());
        trial->step();trial->chargeElapsed(elapsed());
        const State result=trial->snapshot();
        const int attempts=result.attempts-saved.attempts;
        ntAttempts_+=attempts;probeAttempts_+=attempts;++probes_;
        const double cost=std::max(1e-12,elapsed()-begin);ntSeconds_+=cost;
        const double before=saved.g.norm(),after=result.g.norm();
        const double gain=GradientWindow::logReduction(before,after);
        const double lbRate=lbWindow.rate(),ntRate=std::max(0.,gain)/cost;
        const bool moved=result.iterations>saved.iterations;
        const bool numerical=(result.status=="ready" || result.status=="gradient_target") &&
            result.g.allFinite() && std::isfinite(result.energy) && std::isfinite(after) &&
            result.energy<saved.energy && moved;
        const bool qualified=numerical && (result.status=="gradient_target" ||
            (after<=policy.trialGradientRatio*before && ntRate>=policy.rateAdvantage*lbRate));
        events_.push_back({{"event","newton_probe"},{"reason",reason},{"qualified",qualified},
            {"status",result.status},{"error",result.error},{"lbfgs_attempts",lbAttempts_},
            {"gradient_before",before},{"gradient_after",after},{"energy_before",saved.energy},
            {"energy_after",result.energy},{"lbfgs_window_ratio",lbWindow.ratio()},
            {"lbfgs_rate",lbRate},{"newton_rate",ntRate},{"probe_seconds",cost},
            {"attempts",result.attempts},{"evaluations",result.evaluations},{"hessians",result.hessians}});
        if(qualified)
        {
            source_=saved;target_=transformed;engine=std::move(trial);
            phase_="newton";transitioned_=true;ntSteps=1;ntStalls=0;ntIndefinite=0;
            ntWindow.clear();ntWindow.push(before,ntSeconds_-cost);ntWindow.push(after,ntSeconds_);
            chargeBudget();return;
        }
        // The retained L-BFGS x/g/previous iterates/ring/history stay untouched.
        // Its oracle geometry may have been moved by speculative evaluations.
        problem.restore(saved.x);auto floor=resources(result);floor.cellCompleted=false;
        engine->chargeResources(floor);chargeBudget();
        if(engine->state().status=="ready" && result.status!="ready" &&
           !recoverable(result.status) && result.status!="gradient_target" && !resourceStop(result.status))
            engine->boundaryFailure(result.status,result.error);
        nextProbe=epochLbSteps+retryInterval;
        retryInterval=std::min(16*policy.window,2*retryInterval);
    }
    void handleNewtonStep(bool moved)
    {
        if(phase_!="newton")return;
        if(moved && engine->state().status=="ready")
        {
            if(engine->state().lastShift>0)
                ++ntIndefinite;
            else
                ntIndefinite=0;
            if(ntIndefinite>=policy.maxNewtonIndefinite)
            {
                if(policy.onNewtonStall=="cutback")
                    engine->boundaryFailure("indefinite_curvature","persistent indefinite curvature detected (consecutive_shifts="+std::to_string(ntIndefinite)+")");
                else if(policy.onNewtonStall=="lbfgs" && fallbacks_<HandoffPolicy::maximumFallbacks)
                {
                    restartWarmup("indefinite_curvature");
                    return;
                }
            }
            if(engine->state().status=="ready")
            {
                ++ntSteps;ntWindow.push(engine->state().g.norm(),ntSeconds_);
                if(ntSteps>=policy.newtonStallGrace && ntSteps%policy.newtonStallWindow==0 && ntWindow.ready())
                {
                    ntStalls=ntWindow.ratio()>=policy.newtonStallRatio?ntStalls+1:0;
                    if(ntStalls>=policy.maxNewtonStalls)
                    {
                        if(policy.onNewtonStall=="cutback")
                            engine->boundaryFailure("newton_stagnation","newton plateau detected (ratio="+std::to_string(ntWindow.ratio())+")");
                        else if(policy.onNewtonStall=="lbfgs" && fallbacks_<HandoffPolicy::maximumFallbacks)
                        {
                            restartWarmup("gradient_plateau");
                            return;
                        }
                    }
                }
            }
        }
        else if(isInnerFailure(engine->state().status))
        {
            if(policy.onNewtonStall=="cutback")
            {
                engine->boundaryFailure(engine->state().status=="regularization_failed"?"indefinite_curvature":"newton_stagnation",
                                       "newton failure ("+engine->state().status+"): "+engine->state().error);
            }
            else if(fallbacks_<HandoffPolicy::maximumFallbacks && policy.onNewtonStall=="lbfgs")
                restartWarmup(engine->state().status);
        }
    }
public:
    HybridSession(Problem& p,const Config& lb,const Config& nt,const Vec& initial,int switchAttempts,
                  std::function<double()> timer=Engine::wallClock,
                  double budgetStart=std::numeric_limits<double>::quiet_NaN(),
                  const HandoffPolicy& handoff=HandoffPolicy())
        :problem(p),lbfgs(lb),newton(nt),cut(switchAttempts),clock(std::move(timer)),
         policy(handoff),lbWindow(handoff.window),ntWindow(handoff.newtonStallWindow)
    {
        const double entry=clock();origin=std::isnan(budgetStart)?entry:budgetStart;
        if(!std::isfinite(entry) || !std::isfinite(origin) || origin>entry)
            throw std::invalid_argument("invalid cumulative budget origin");
        lbfgs.validate();newton.validate();policy.validate();
        if(policy.adaptive && policy.minimumWarmup>=lbfgs.maxAttempts)
            throw std::invalid_argument("adaptive warmup exceeds cumulative attempt budget");
        nextCheck=nextProbe=policy.minimumWarmup;retryInterval=policy.window;
        if(lbfgs.method!="native_lbfgs" || newton.method!="sparse_newton" ||
           lbfgs.newtonBackend!="cholmod" || newton.newtonBackend!="cholmod" ||
           newton.hessianAssembly!="local_dihedral_csc" ||
           lbfgs.reuseNewtonSymbolic || newton.reuseNewtonSymbolic ||
           lbfgs.guardedHessianReuse || newton.guardedHessianReuse ||
           cut<1 || cut>=lbfgs.maxAttempts)
            throw std::invalid_argument("native hybrid requires the declared live warmup/fresh CHOLMOD recipe");
        Json left=lbfgs.json(),right=newton.json();
        for(const char* k:{"method","hessianAssembly","hessianThreads"}){left.erase(k);right.erase(k);}
        if(left!=right)throw std::invalid_argument("hybrid phase budgets/controls differ");
        engine=std::make_unique<Engine>(problem,lbfgs,initial,clock);
        chargeBudget();
    }
    double elapsed() const
    {
        const double now=clock();
        if(!std::isfinite(now) || now<origin)throw std::runtime_error("invalid cumulative clock");
        return now-origin;
    }
    const std::string& phase() const{return phase_;}
    const Config& config() const{return phase_=="lbfgs"?lbfgs:newton;}
    const State& state() const{return engine->state();}
    State snapshot(){chargeBudget();return engine->snapshot();}
    bool transitioned() const{return transitioned_;}
    int lbfgsAttempts() const{return lbAttempts_;}
    int newtonAttempts() const{return ntAttempts_;}
    int probeAttempts() const{return probeAttempts_;}
    int probes() const{return probes_;}
    int fallbacks() const{return fallbacks_;}
    double lbfgsSeconds() const{return lbSeconds_;}
    double newtonSeconds() const{return ntSeconds_;}
    const Json& policyEvents() const{return events_;}
    const State& source() const
    {if(!transitioned_)throw std::logic_error("no native handoff");return source_;}
    const State& target() const
    {if(!transitioned_)throw std::logic_error("no native handoff");return target_;}
    bool step()
    {
        chargeBudget();
        if(engine->state().status!="ready")return false;
        const int beforeAttempts=engine->state().attempts,beforeIterations=engine->state().iterations;
        const double begin=elapsed();engine->step();chargeBudget();
        const int work=engine->state().attempts-beforeAttempts;
        const bool moved=engine->state().iterations>beforeIterations;
        const double seconds=elapsed()-begin;
        if(phase_=="lbfgs"){lbAttempts_+=work;lbSeconds_+=seconds;}
        else {ntAttempts_+=work;ntSeconds_+=seconds;}
        if(policy.adaptive)
        {
            if(phase_=="lbfgs" && moved && engine->state().status=="ready")
            {
                ++epochLbSteps;const double g=engine->state().g.norm();
                if(initialNorm==0)
                {
                    initialNorm=g;
                    // Reuse the already evaluated starting gradient; no extra
                    // oracle call, and window costs include that initialization.
                    const auto& row=engine->state().trace.back();
                    lbWindow.push(row.at("base_gradient_norm").get<double>(),lbSeconds_-seconds);
                }
                lbWindow.push(g,lbSeconds_);
                std::string reason;if(requestProbe(reason))probeNewton(reason);
            }
            else if(phase_=="newton")
            {
                handleNewtonStep(moved);
            }
            chargeBudget();return engine->state().status=="ready";
        }
        // Fixed mode never overwrites a genuine acceptance/failure/resource stop.
        if(phase_=="lbfgs" && engine->state().status=="ready" && engine->state().attempts==cut)
        {
            source_=snapshot();
            if(engine->state().status!="ready")return false;
            target_=transitionToNewton(problem,lbfgs,newton,source_,cut);
            target_.elapsed=std::max(target_.elapsed,elapsed());
            engine=std::make_unique<Engine>(problem,newton,target_,clock);
            chargeBudget();phase_="newton";transitioned_=true;
            ntSteps=0;ntStalls=0;ntIndefinite=0;ntWindow.clear();
        }
        else if(phase_=="newton")
        {
            handleNewtonStep(moved);
        }
        chargeBudget();
        return engine->state().status=="ready";
    }
};

struct HybridOptions
{
    int warmupAttempts=2000, hessianThreads=1;
    bool refineLinearSolve=false;
    int maxAttempts=100000, maxEvaluations=200000, maxHessians=40;
    double seconds=1800, gradientTolerance=5e-14, physicalGate=1e-13;
    // Same steady-clock origin as the calling worker, or API entry by default.
    double budgetStart=std::numeric_limits<double>::quiet_NaN();
    std::string implementationIdentity="native-production-reference-v1";
    std::string dependencyIdentity="linked-cholmod-runtime";
    std::string problemIdentity; // Optional externally verified fixed-objective identity.
    HandoffPolicy handoff;
};

inline std::pair<Config,Config> hybridConfigs(const HybridOptions& o)
{
    Config lb;lb.method="native_lbfgs";lb.newtonBackend="cholmod";
    lb.implementationId=o.implementationIdentity;lb.dependencyIdentity=o.dependencyIdentity;
    lb.maxAttempts=o.maxAttempts;lb.maxEvaluations=o.maxEvaluations;lb.maxHessians=o.maxHessians;
    lb.seconds=o.seconds;lb.tolerance=o.gradientTolerance;lb.physicalGate=o.physicalGate;
    lb.refineLinearSolve=o.refineLinearSolve; // Shared identity; only Newton uses this control.
    Config nt=lb;nt.method="sparse_newton";nt.hessianAssembly="local_dihedral_csc";
    nt.hessianThreads=o.hessianThreads;nt.refineLinearSolve=o.refineLinearSolve;
    lb.validate();nt.validate();return {lb,nt};
}

// A content fingerprint for an in-process solve, NOT a cryptographic build
// identity or authorization to resume a different objective. External campaigns
// supply their independently verified fixed-objective identity instead.
template<class Mesh>
std::string shellObjectiveFingerprint(const Mesh& mesh,double young,double nu,double h)
{
    std::uint64_t hash=1469598103934665603ULL;
    auto append=[&](const void* ptr,std::size_t n){const auto* p=static_cast<const unsigned char*>(ptr);
        for(std::size_t i=0;i<n;++i){hash^=p[i];hash*=1099511628211ULL;}};
    const double material[]={young,nu,h};append(material,sizeof(material));
    const auto X=mesh.getRestConfiguration().getVertices();const auto F=mesh.getTopology().getFace2Vertices();
    const auto E=mesh.getTopology().getEdge2Vertices();
    const int dims[]={int(X.rows()),int(F.rows()),int(E.rows())};append(dims,sizeof(dims));
    append(X.data(),X.size()*sizeof(double));append(F.data(),F.size()*sizeof(int));append(E.data(),E.size()*sizeof(int));
    const auto mask=ShellEquilibrium::constrainedDofMask(mesh);append(mask.data(),mask.size());
    for(const auto* forms:{&mesh.getRestConfiguration().template getFirstFundamentalForms<bottom>(),
                          &mesh.getRestConfiguration().template getFirstFundamentalForms<top>(),
                          &mesh.getRestConfiguration().getSecondFundamentalForms()})
        for(const auto& form:*forms)append(form.data(),4*sizeof(double));
    return "in-process-fixed-objective:"+std::to_string(hash);
}

struct HybridReport
{
    State terminal, handoffSource, handoffTarget;
    bool transitioned=false, accepted=false;
    std::string phase, curvature="not_checked", error;
    Vec selected;
    double energy=std::numeric_limits<double>::quiet_NaN();
    double gradientNorm=std::numeric_limits<double>::infinity();
    int verificationEvaluations=0, verificationHessians=0;
    int lbfgsAttempts=0, newtonAttempts=0, probeAttempts=0, probes=0, fallbacks=0;
    double lbfgsSeconds=0, newtonSeconds=0;
    Json handoffPolicy, policyEvents=Json::array();
    int code() const{return accepted?0:1;}
    int iterations() const{return terminal.iterations;}
    int evaluations() const{return terminal.evaluations+verificationEvaluations;}
    Json proof(double tolerance) const
    {
        return {{"accepted",accepted},{"termination",terminal.status},{"error",error},
            {"phase",phase},{"transitioned",transitioned},{"curvature",curvature},
            {"gradient_norm",gradientNorm},{"gradient_tolerance",tolerance},{"energy",energy},
            {"attempts",terminal.attempts},{"evaluations",evaluations()},
            {"hessians",terminal.hessians+verificationHessians},
            {"verification_evaluations",verificationEvaluations},{"verification_hessians",verificationHessians},
            {"warmup_attempts",lbfgsAttempts},{"newton_attempts",newtonAttempts},
            {"probe_attempts",probeAttempts},{"probes",probes},{"fallbacks",fallbacks},
            {"lbfgs_seconds",lbfgsSeconds},{"newton_seconds",newtonSeconds},
            {"handoff_policy",handoffPolicy},{"policy_events",policyEvents}};
    }
    // Flush every returned solve BEFORE acceptance-dependent sequence handling.
    // Failure evidence is not a successful sequence manifest.
    void writeProof(std::ostream& output,double tolerance) const
    {
        output<<proof(tolerance).dump()<<'\n';output.flush();
        if(!output)throw std::runtime_error("native hybrid proof write failed");
    }
};

// Fixed isotropic bilayer material only. The caller keeps targets and physical
// constraints immutable during this call. Existing HLBFGS/TR APIs are untouched.
template<class Mesh,class Energy>
HybridReport solveNativeHybridShell(Mesh& mesh,Energy& energy,double young,double nu,double h,
                                    const HybridOptions& options=HybridOptions(),
                                    const std::function<void(HybridSession&)>& boundary={})
{
    const double entry=Engine::wallClock();
    const double start=std::isnan(options.budgetStart)?entry:options.budgetStart;
    if(!std::isfinite(start) || start>entry)throw std::invalid_argument("invalid cumulative budget origin");
    tbb::global_control one(tbb::global_control::max_allowed_parallelism,1);
    HybridOptions innerOptions = options;
    if(options.maxHessians > 1)
        innerOptions.maxHessians = options.maxHessians - 1;
    const auto configs=hybridConfigs(innerOptions);
    const std::string id=options.problemIdentity.empty()?shellObjectiveFingerprint(mesh,young,nu,h):options.problemIdentity;
    ShellProblem<Mesh,Energy> problem(mesh,energy,young,nu,h,id,"local_dihedral_csc",options.hessianThreads);
    const Vec initial=Eigen::Map<const Vec>(mesh.getDataPointer(),problem.size());
    HybridSession session(problem,configs.first,configs.second,initial,options.warmupAttempts,
                          Engine::wallClock,start,options.handoff);
    if(boundary)boundary(session);
    while(session.state().status=="ready" && Engine::wallClock()-start<options.seconds)
    {
        session.step();
        if(boundary)boundary(session); // durable snapshots may be emitted here only.
    }
    HybridReport report;report.terminal=session.snapshot();report.phase=session.phase();
    report.transitioned=session.transitioned();
    report.lbfgsAttempts=session.lbfgsAttempts();report.newtonAttempts=session.newtonAttempts();
    report.probeAttempts=session.probeAttempts();report.probes=session.probes();report.fallbacks=session.fallbacks();
    report.lbfgsSeconds=session.lbfgsSeconds();report.newtonSeconds=session.newtonSeconds();
    report.handoffPolicy=options.handoff.json();report.policyEvents=session.policyEvents();
    if(report.transitioned){report.handoffSource=session.source();report.handoffTarget=session.target();}
    report.selected=(report.terminal.status=="gradient_target" || !report.terminal.bestValid)?report.terminal.x:report.terminal.bestX;
    problem.restore(report.selected);
    // Charge independent verification separately. Never turn a failed/capped
    // optimizer into acceptance merely because a retained candidate is good.
    if(report.terminal.initialized && report.terminal.evaluations<options.maxEvaluations &&
       Engine::wallClock()-start<options.seconds+30)
    {
        ++report.verificationEvaluations;
        const auto v=problem.evaluate(report.selected);report.energy=v.energy;report.gradientNorm=v.gradient.norm();
        if((report.terminal.hessians + 1)<=options.maxHessians && Engine::wallClock()-start<options.seconds+30)
        {
            ++report.verificationHessians;
            const auto H=problem.hessian(report.selected);
            const auto projector=ShellEquilibrium::RigidModeProjector::fromMesh(mesh);
            report.curvature=ShellNativeEndpointCheck::inspect(H,v.gradient,problem.scales(),projector,problem.energyReference()).status;
            report.accepted=report.terminal.status=="gradient_target" && std::isfinite(v.energy) &&
                v.gradient.allFinite() && report.gradientNorm<=options.gradientTolerance && report.curvature=="positive_pivots";
        }
    }
    if(Engine::wallClock()-start>=options.seconds+50)report.accepted=false;
    report.error=report.terminal.error;return report;
}
}
#endif

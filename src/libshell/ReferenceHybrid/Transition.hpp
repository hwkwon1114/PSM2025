#ifndef SHELL_NATIVE_TRANSITION_HPP
#define SHELL_NATIVE_TRANSITION_HPP
#include "Core.hpp"
namespace ShellNativeHybrid {
// Explicit NEW recipe transition, not checkpoint identity rewriting. Preserve
// the source checkpoint separately; target has fresh Newton working memory.
inline State transitionToNewton(Problem& p,const Config& from,const Config& to,
                                const State& source,int switchAttempts)
{
    from.validate();to.validate();
    if(from.method!="native_lbfgs" || to.method!="sparse_newton" || switchAttempts<1)
        throw std::invalid_argument("invalid hybrid phase policy");
    if(source.status!="ready" || !source.boundary || !source.initialized ||
       source.attempts!=switchAttempts)
        throw std::invalid_argument("switch only at declared live boundary");
    if(source.elapsed>=to.seconds || source.attempts>=to.maxAttempts ||
       source.evaluations>=to.maxEvaluations || source.hessians>=to.maxHessians)
        throw std::invalid_argument("hybrid budget exhausted");
    Json left=from.json(),right=to.json();
    // The declared Newton phase may opt into guarded Hessian reuse; native
    // L-BFGS cannot carry this reuse option (Config::validate forbids it).
    // The Newton-only assembly thread count may also differ. Both phase
    // configurations remain explicit in the durable handoff policy.
    for(const char* key:{"method","hessianAssembly","guardedHessianReuse","hessianThreads"}){left.erase(key);right.erase(key);}
    if(left!=right)throw std::invalid_argument("hybrid phase controls or dependency identity differ");
    // Full source identity validation through existing direct resume contract.
    Engine checked(p,from,source,[]{return 0.;});
    Engine fresh(p,to,source.x,[]{return 0.;});State target=fresh.snapshot();
    target.initialized=source.initialized;target.x=source.x;target.g=source.g;
    target.energy=source.energy;target.previousEnergy=source.energy;
    target.attempts=source.attempts;target.iterations=source.iterations;
    target.evaluations=source.evaluations;target.hessians=source.hessians;
    target.hvps=source.hvps;target.elapsed=source.elapsed;
    target.bestValid=source.bestValid;target.bestX=source.bestX;target.bestEnergy=source.bestEnergy;
    target.trace=source.trace;
    State::fromJson(target.json(),p.size(),to);
    return target;
}
}
#endif

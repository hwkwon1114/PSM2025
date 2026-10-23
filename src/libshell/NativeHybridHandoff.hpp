#ifndef NATIVE_HYBRID_HANDOFF_HPP
#define NATIVE_HYBRID_HANDOFF_HPP

#include "ReferenceHybrid/Core.hpp"
#include <deque>

namespace ShellNativeHybrid
{
// Request a safeguarded, counted Newton trial; never interpret a plateau as
// convergence. The final physical gradient/curvature checks are independent.
struct HandoffPolicy
{
    bool adaptive=false;
    int minimumWarmup=50, window=20;
    double plateauRatio=.90, relativeGradient=.01;
    double trialGradientRatio=.95, rateAdvantage=1.1;
    static constexpr int maximumProbes=8, maximumFallbacks=2, newtonWindow=4;
    static constexpr double newtonPlateauRatio=.90;
    std::string onNewtonStall="lbfgs"; // "lbfgs", "cutback", or "none"
    int newtonStallWindow=5;
    double newtonStallRatio=0.85;
    int newtonStallGrace=10;
    int maxNewtonStalls=2;
    int maxNewtonIndefinite=3;
    void validate() const
    {
        if(window<2 || window>1000 || minimumWarmup<window || minimumWarmup>100000 ||
           !std::isfinite(plateauRatio) || plateauRatio<=0 || plateauRatio>1 ||
           !std::isfinite(relativeGradient) || relativeGradient<=0 || relativeGradient>=1 ||
           !std::isfinite(trialGradientRatio) || trialGradientRatio<=0 || trialGradientRatio>=1 ||
           !std::isfinite(rateAdvantage) || rateAdvantage<1 ||
           (onNewtonStall!="cutback" && onNewtonStall!="lbfgs" && onNewtonStall!="none") ||
           newtonStallWindow<2 || newtonStallWindow>100 ||
           !std::isfinite(newtonStallRatio) || newtonStallRatio<=0 || newtonStallRatio>1 ||
           newtonStallGrace<1 || newtonStallGrace>1000 || maxNewtonStalls<1 || maxNewtonStalls>50 ||
           maxNewtonIndefinite<1 || maxNewtonIndefinite>50)
            throw std::invalid_argument("invalid gradient handoff controls");
    }
    Json json() const
    {
        return {{"mode",adaptive?"adaptive":"fixed"},{"version","gradient-cost-v1"},
            {"minimum_warmup",minimumWarmup},{"window",window},{"plateau_ratio",plateauRatio},
            {"relative_gradient",relativeGradient},{"trial_gradient_ratio",trialGradientRatio},
            {"rate_advantage",rateAdvantage},{"maximum_probes",maximumProbes},
            {"maximum_fallbacks",maximumFallbacks},{"newton_window",newtonWindow},
            {"newton_plateau_ratio",newtonPlateauRatio},
            {"on_newton_stall",onNewtonStall},{"newton_stall_window",newtonStallWindow},
            {"newton_stall_ratio",newtonStallRatio},{"newton_stall_grace",newtonStallGrace},
            {"max_newton_stalls",maxNewtonStalls},{"max_newton_indefinite",maxNewtonIndefinite}};
    }
};

class GradientWindow
{
    struct Sample {double norm, seconds;};
    int width;
    std::deque<Sample> samples;
public:
    explicit GradientWindow(int w):width(w)
    {if(w<1)throw std::invalid_argument("invalid gradient window");}
    void clear(){samples.clear();}
    void push(double norm,double cumulativeMethodSeconds)
    {
        if(!std::isfinite(norm) || norm<0 || !std::isfinite(cumulativeMethodSeconds) ||
           cumulativeMethodSeconds<0 || (!samples.empty() && cumulativeMethodSeconds<samples.back().seconds))
            throw std::invalid_argument("invalid gradient progress sample");
        samples.push_back({norm,cumulativeMethodSeconds});
        while(samples.size()>std::size_t(width+1))samples.pop_front();
    }
    bool ready() const{return samples.size()==std::size_t(width+1);}
    // Log differences avoid overflowing a ratio of two individually finite norms.
    static double logReduction(double before,double after)
    {return std::log(std::max(before,1e-300))-std::log(std::max(after,1e-300));}
    double ratio() const
    {
        if(!ready())return 1;
        return std::exp(std::max(-700.,std::min(700.,-logReduction(samples.front().norm,samples.back().norm))));
    }
    double rate() const
    {
        if(!ready())return 0;
        return std::max(0.,logReduction(samples.front().norm,samples.back().norm))/
            std::max(1e-12,samples.back().seconds-samples.front().seconds);
    }
};
}
#endif

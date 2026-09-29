#include "gtest/gtest.h"
#include "common.hpp"
#include "HLBFGS_Wrapper.hpp"
#include <limits>

#ifdef USEHLBFGS
namespace {
class StopProbe : public HLBFGS_Methods::HLBFGS_Wrapper {
public:
    using HLBFGS_Wrapper::default_setup;
    double norm = -1.0;
    void evaluate(int, double* x, double*, double* f, double* g) override {
        const double a=x[0]-100.0, b=x[1]-100.0;
        *f=0.5*(a*a+10.0*b*b);
        g[0]=a; g[1]=10.0*b;
    }
    void newiteration(int, int, double*, double*, double*, double* gnorm) override {
        norm=*gnorm;
    }
};
}
TEST(hlbfgs_absolute_stop, legacy_parameters_unchanged) {
    StopProbe p; double params[20]; int info[20];
    p.default_setup(params,info,1e-12,false);
    EXPECT_DOUBLE_EQ(params[5],1e-12);
    EXPECT_DOUBLE_EQ(params[6],1e-16);
}
TEST(hlbfgs_absolute_stop, opt_in_disables_relative_stop) {
    StopProbe p; double params[20]; int info[20];
    p.set_absolute_gradient_tolerance(5e-14);
    p.default_setup(params,info,1e-12,false);
    EXPECT_DOUBLE_EQ(params[5],0.0);
    EXPECT_DOUBLE_EQ(params[6],5e-14);
    EXPECT_THROW(p.set_absolute_gradient_tolerance(0.0),std::invalid_argument);
    EXPECT_THROW(p.set_absolute_gradient_tolerance(-1.0),std::invalid_argument);
    EXPECT_THROW(p.set_absolute_gradient_tolerance(std::numeric_limits<double>::quiet_NaN()),std::invalid_argument);
    EXPECT_THROW(p.set_absolute_gradient_tolerance(std::numeric_limits<double>::infinity()),std::invalid_argument);
}
TEST(hlbfgs_absolute_stop, translated_quadratic_meets_absolute_gate) {
    StopProbe p; double params[20]; int info[20];
    p.set_absolute_gradient_tolerance(5e-10);
    p.default_setup(params,info,1e-6,false);
    info[4]=1000;
    double x[2]={102.0,101.0};
    const int code=HLBFGS(2,10,x,HLBFGS_Methods::evaluate,nullptr,
        HLBFGS_UPDATE_Hessian,HLBFGS_Methods::newiteration,&p,params,info);
    EXPECT_TRUE(code==2 || code==3);
    EXPECT_GE(p.norm,0.0);
    EXPECT_LT(p.norm,5e-10);
    const double a=x[0]-100.0,b=10.0*(x[1]-100.0);
    EXPECT_LT(std::hypot(a,b),5e-10);
}
#endif

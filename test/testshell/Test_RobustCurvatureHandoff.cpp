#include "gtest/gtest.h"
#include "../diagnostics/RobustCurvatureHandoff.hpp"
#include "../diagnostics/NonconvexBenchmark.hpp"
#include <memory>

using namespace NonconvexBenchmark;

namespace {

struct ToyRosenbrock : Problem {
    Vec current;
    ToyRosenbrock() : current(Vec::Zero(2)) {}
    int size() const override { return 2; }
    std::string identity() const override { return "toy:rosenbrock"; }
    Vec scales() const override { return Vec::Ones(2); }
    double energyReference() const override { return 1.0; }
    void restore(const Vec& x) override { current = x; }
    Value evaluate(const Vec& x) override {
        current = x;
        const double a = 1.0 - x[0];
        const double b = x[1] - x[0] * x[0];
        const double f = a * a + 100.0 * b * b;
        Vec g(2);
        g << -2.0 * a - 400.0 * x[0] * b, 200.0 * b;
        return {f, g};
    }
    Sparse hessian(const Vec& x) override {
        Eigen::Matrix2d H;
        H << 2.0 - 400.0 * x[1] + 1200.0 * x[0] * x[0], -400.0 * x[0],
             -400.0 * x[0], 200.0;
        return H.sparseView();
    }
};

struct ToyQuarticSaddle : Problem {
    Vec current;
    ToyQuarticSaddle() : current(Vec::Zero(2)) {}
    int size() const override { return 2; }
    std::string identity() const override { return "toy:quartic_saddle"; }
    Vec scales() const override { return Vec::Ones(2); }
    double energyReference() const override { return 1.0; }
    void restore(const Vec& x) override { current = x; }
    Value evaluate(const Vec& x) override {
        current = x;
        const double f = (0.5 * x.array().square() - 0.25 * x.array().pow(4)).sum();
        Vec g = x - x.array().cube().matrix();
        return {f, g};
    }
    Sparse hessian(const Vec& x) override {
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();
        H.diagonal() = (1.0 - 3.0 * x.array().square()).matrix();
        return H.sparseView();
    }
};

} // namespace

TEST(RobustCurvatureHandoff, WarmupActiveRejection) {
    ToyRosenbrock p;
    RobustCurvatureOptions options;
    options.minWarmupAttempts = 200;
    options.gradientToleranceGate = 1.0;
    RobustCurvatureGate gate(options);

    Vec x(2);
    x << 1.0, 1.0;
    Value v = p.evaluate(x);

    // Below warmup limit
    auto res = gate.probeDirect(p, x, v.gradient, 50);
    EXPECT_FALSE(res.probed);
    EXPECT_FALSE(res.certifiedConvex);
    EXPECT_EQ(res.reason, "warmup_active");
}

TEST(RobustCurvatureHandoff, GradientAboveGateRejection) {
    ToyRosenbrock p;
    RobustCurvatureOptions options;
    options.minWarmupAttempts = 0;
    options.gradientToleranceGate = 1e-4;
    RobustCurvatureGate gate(options);

    Vec x(2);
    x << -1.2, 1.0; // Gradient norm ~ 5.6
    Value v = p.evaluate(x);

    auto res = gate.probeDirect(p, x, v.gradient, 10);
    EXPECT_FALSE(res.probed);
    EXPECT_FALSE(res.certifiedConvex);
    EXPECT_EQ(res.reason, "gradient_above_gate");
}

TEST(RobustCurvatureHandoff, IntervalSkip) {
    ToyRosenbrock p;
    RobustCurvatureOptions options;
    options.minWarmupAttempts = 100;
    options.checkInterval = 25;
    options.gradientToleranceGate = 10.0;
    RobustCurvatureGate gate(options);

    Vec x(2);
    x << 1.0, 1.0;
    Value v = p.evaluate(x);

    // Attempt 110 is not on interval (100 + 25k)
    auto res = gate.probeDirect(p, x, v.gradient, 110);
    EXPECT_FALSE(res.probed);
    EXPECT_FALSE(res.certifiedConvex);
    EXPECT_EQ(res.reason, "interval_skip");

    // Attempt 125 is on interval
    auto res2 = gate.probeDirect(p, x, v.gradient, 125);
    EXPECT_TRUE(res2.probed);
    EXPECT_TRUE(res2.certifiedConvex);
    EXPECT_EQ(res2.reason, "certified_convex");
}

TEST(RobustCurvatureHandoff, IndefiniteSaddleRejection) {
    ToyRosenbrock p;
    RobustCurvatureOptions options;
    options.minWarmupAttempts = 0;
    options.gradientToleranceGate = 1000.0;
    options.checkInterval = 1;
    RobustCurvatureGate gate(options);

    // At (0, 0.5), H_00 = 2 - 400(0.5) = -198 < 0 -> Indefinite!
    Vec x(2);
    x << 0.0, 0.5;
    Value v = p.evaluate(x);

    auto res = gate.probeDirect(p, x, v.gradient, 1);
    EXPECT_TRUE(res.probed);
    EXPECT_FALSE(res.certifiedConvex);
    EXPECT_EQ(res.reason, "indefinite_requires_shift");
}

TEST(RobustCurvatureHandoff, QuarticSaddleRejection) {
    ToyQuarticSaddle p;
    RobustCurvatureOptions options;
    options.minWarmupAttempts = 0;
    options.gradientToleranceGate = 1000.0;
    options.checkInterval = 1;
    RobustCurvatureGate gate(options);

    // At (1.5, 0.0), H_00 = 1 - 3*(1.5)^2 = 1 - 6.75 = -5.75 < 0 -> Indefinite!
    Vec x(2);
    x << 1.5, 0.0;
    Value v = p.evaluate(x);

    auto res = gate.probeDirect(p, x, v.gradient, 1);
    EXPECT_TRUE(res.probed);
    EXPECT_FALSE(res.certifiedConvex);
    EXPECT_EQ(res.reason, "indefinite_requires_shift");
}

TEST(RobustCurvatureHandoff, StrictlyConvexBasinCertification) {
    ToyRosenbrock p;
    RobustCurvatureOptions options;
    options.minWarmupAttempts = 0;
    options.gradientToleranceGate = 1000.0;
    options.checkInterval = 1;
    RobustCurvatureGate gate(options);

    // Near minimum (1, 1), H is strictly positive definite
    Vec x(2);
    x << 1.0, 1.0;
    Value v = p.evaluate(x);

    auto res = gate.probeDirect(p, x, v.gradient, 1);
    EXPECT_TRUE(res.probed);
    EXPECT_TRUE(res.certifiedConvex);
    EXPECT_EQ(res.reason, "certified_convex");
    EXPECT_GT(res.curvatureScale, 0.0);
}

TEST(RobustCurvatureHandoff, EndToEndAdaptiveSwitchConvergence) {
    ToyRosenbrock p;
    Config lb;
    lb.method = "native_lbfgs";
    lb.seconds = 30;
    lb.maxAttempts = 500;
    lb.tolerance = 1e-14;
    lb.physicalGate = 1e-13;

    Config nt = lb;
    nt.method = "sparse_newton";

    Vec x(2);
    x << -1.2, 1.0;

    RobustCurvatureOptions gateOpt;
    gateOpt.minWarmupAttempts = 5;
    gateOpt.gradientToleranceGate = 1.0;
    gateOpt.checkInterval = 2;
    RobustCurvatureGate gate(gateOpt);

    Engine engine(p, lb, x);
    bool switched = false;
    int switchAttempt = -1;

    while (engine.state().status == "ready" && engine.state().attempts < 100) {
        engine.step();
        auto probeRes = gate.probe(p, engine.state());
        if (probeRes.certifiedConvex) {
            switched = true;
            switchAttempt = engine.state().attempts;
            break;
        }
    }

    ASSERT_TRUE(switched);
    EXPECT_GE(switchAttempt, 5);

    // Transfer to Newton at the certified convex state
    Engine newtonEngine(p, nt, engine.state().x);
    while (newtonEngine.state().status == "ready" && newtonEngine.state().attempts < 30) {
        newtonEngine.step();
        // Since state is in certified convex basin, all Newton steps should be unshifted
        EXPECT_DOUBLE_EQ(newtonEngine.state().lastShift, 0.0);
    }

    EXPECT_EQ(newtonEngine.state().status, "gradient_target");
    EXPECT_LE(newtonEngine.state().g.norm(), 1e-12);
    EXPECT_NEAR(newtonEngine.state().x[0], 1.0, 1e-7);
    EXPECT_NEAR(newtonEngine.state().x[1], 1.0, 1e-7);
}

namespace {
struct ToyCorruptedProblem : Problem {
    Vec current;
    bool badGradient = false;
    bool badHessian = false;
    bool badScale = false;
    bool badEnergy = false;

    ToyCorruptedProblem() : current(Vec::Zero(2)) {}
    int size() const override { return 2; }
    std::string identity() const override { return "toy:corrupted"; }
    Vec scales() const override {
        if (badScale) return Vec::Constant(2, std::numeric_limits<double>::quiet_NaN());
        return Vec::Ones(2);
    }
    double energyReference() const override {
        if (badEnergy) return -1.0;
        return 1.0;
    }
    void restore(const Vec& x) override { current = x; }
    Value evaluate(const Vec& x) override {
        current = x;
        Vec g = Vec::Zero(2);
        if (badGradient) g[0] = std::numeric_limits<double>::quiet_NaN();
        return {0.0, g};
    }
    Sparse hessian(const Vec&) override {
        Eigen::Matrix2d H = Eigen::Matrix2d::Identity();
        if (badHessian) {
            H(0, 1) = std::numeric_limits<double>::quiet_NaN();
            H(1, 0) = std::numeric_limits<double>::quiet_NaN();
        }
        return H.sparseView();
    }
};
} // namespace

TEST(RobustCurvatureHandoff, InvalidOracleNonfiniteGradient) {
    ToyCorruptedProblem p;
    p.badGradient = true;
    RobustCurvatureOptions options;
    options.minWarmupAttempts = 0;
    options.gradientToleranceGate = 1000.0;
    RobustCurvatureGate gate(options);

    Vec x = Vec::Zero(2);
    Value v = p.evaluate(x);
    auto res = gate.probeDirect(p, x, v.gradient, 0);
    EXPECT_FALSE(res.probed);
    EXPECT_FALSE(res.certifiedConvex);
    EXPECT_EQ(res.reason, "nonfinite_gradient");
}

TEST(RobustCurvatureHandoff, InvalidOracleNonfiniteHessian) {
    ToyCorruptedProblem p;
    p.badHessian = true;
    RobustCurvatureOptions options;
    options.minWarmupAttempts = 0;
    options.gradientToleranceGate = 1000.0;
    RobustCurvatureGate gate(options);

    Vec x = Vec::Zero(2);
    Value v = p.evaluate(x);
    auto res = gate.probeDirect(p, x, v.gradient, 0);
    EXPECT_TRUE(res.probed);
    EXPECT_FALSE(res.certifiedConvex);
    EXPECT_EQ(res.reason, "nonfinite_hessian_entry");
}

TEST(RobustCurvatureHandoff, InvalidOracleInvalidScalesAndEnergy) {
    RobustCurvatureOptions options;
    options.minWarmupAttempts = 0;
    options.gradientToleranceGate = 1000.0;
    RobustCurvatureGate gate(options);
    Vec x = Vec::Zero(2);

    {
        ToyCorruptedProblem p;
        p.badEnergy = true;
        Value v = p.evaluate(x);
        auto res = gate.probeDirect(p, x, v.gradient, 0);
        EXPECT_TRUE(res.probed);
        EXPECT_FALSE(res.certifiedConvex);
        EXPECT_EQ(res.reason, "invalid_energy_reference");
    }
    {
        ToyCorruptedProblem p;
        p.badScale = true;
        Value v = p.evaluate(x);
        auto res = gate.probeDirect(p, x, v.gradient, 0);
        EXPECT_TRUE(res.probed);
        EXPECT_FALSE(res.certifiedConvex);
        EXPECT_EQ(res.reason, "invalid_scales");
    }
}


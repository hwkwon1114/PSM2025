//
//  Test_MinimizationReport.cpp
//
//  Contracts of Sim<>::MinimizationReport: strict equilibrium acceptance
//  (accepted) versus the purely descriptive raw classification (termination),
//  and the legacy converged() that callers are still allowed to use.
//

#include "gtest/gtest.h"

#include <cmath>
#include <limits>
#include <type_traits>

#include "common.hpp"
#include "Mesh.hpp"
#include "Sim.hpp"

namespace
{
    typedef Sim<BilayerMesh>::MinimizationReport Report;
    typedef Report::Termination Termination;

    constexpr Real tol = 1e-6;
    constexpr Real nan_norm = std::numeric_limits<Real>::quiet_NaN();
    constexpr Real inf_norm = std::numeric_limits<Real>::infinity();

    /// A report that only fails or passes because of the fields under test.
    Report makeReport(const int code, const Real gradientNorm)
    {
        Report report;
        report.code = code;
        report.iterations = 42;
        report.evaluations = 137;
        report.gradientNorm = gradientNorm;
        return report;
    }

    // The threshold is mandatory: accepted() must not be callable without one,
    // otherwise "no tolerance given" silently becomes "anything goes".
    template<typename R, typename = void>
    struct HasThresholdlessAccepted : std::false_type {};

    template<typename R>
    struct HasThresholdlessAccepted<
        R, decltype(void(std::declval<const R&>().accepted()))> : std::true_type {};
}

TEST(test_minimization_report, threshold_is_a_mandatory_argument)
{
    static_assert(not HasThresholdlessAccepted<Report>::value,
        "MinimizationReport::accepted must require an explicit gradient tolerance");
    static_assert(std::is_invocable_r_v<bool, decltype(&Report::accepted), const Report&, Real>,
        "MinimizationReport::accepted(Real) must be callable on a const report");
}

TEST(test_minimization_report, codes_one_to_four_accept_when_the_norm_is_met)
{
    for(const int code : {1, 2, 3, 4})
    {
        const Report report = makeReport(code, 0.1*tol);
        EXPECT_TRUE(report.accepted(tol)) << "code " << code;
    }
}

TEST(test_minimization_report, code_five_never_accepts_however_small_the_norm)
{
    // Iteration cap: the state is wherever the budget ran out, so it is not an
    // equilibrium even if the norm happens to look good.
    EXPECT_FALSE(makeReport(5, 0.0).accepted(tol));
    EXPECT_FALSE(makeReport(5, 0.1*tol).accepted(tol));
    EXPECT_FALSE(makeReport(5, 1e-300).accepted(tol));
}

TEST(test_minimization_report, codes_outside_one_to_four_are_rejected)
{
    for(const int code : {-2, -1, 0, 6, 7, 1000})
    {
        const Report report = makeReport(code, 0.1*tol);
        EXPECT_FALSE(report.accepted(tol)) << "code " << code;
    }
}

TEST(test_minimization_report, default_report_is_never_accepted)
{
    const Report report;
    EXPECT_EQ(report.code, -1);
    EXPECT_EQ(report.iterations, 0);
    EXPECT_EQ(report.evaluations, 0);
    EXPECT_FALSE(report.accepted(tol));
    EXPECT_FALSE(report.accepted(std::numeric_limits<Real>::max()));
    EXPECT_EQ(report.termination(), Termination::NeverRan);
}

TEST(test_minimization_report, every_accepted_code_requires_a_finite_norm)
{
    for(const int code : {1, 2, 3, 4})
    {
        EXPECT_FALSE(makeReport(code, nan_norm).accepted(tol)) << "NaN, code " << code;
        EXPECT_FALSE(makeReport(code, inf_norm).accepted(tol)) << "+inf, code " << code;
        EXPECT_FALSE(makeReport(code, -inf_norm).accepted(tol)) << "-inf, code " << code;
    }
}

TEST(test_minimization_report, negative_norms_are_rejected)
{
    for(const int code : {1, 2, 3, 4})
    {
        // -1 is the "wrapper never reported a norm" sentinel.
        EXPECT_FALSE(makeReport(code, -1.0).accepted(tol)) << "sentinel, code " << code;
        EXPECT_FALSE(makeReport(code, -1e-30).accepted(tol)) << "tiny negative, code " << code;
        EXPECT_FALSE(makeReport(code, -tol).accepted(tol)) << "negative, code " << code;
    }
    // An exactly zero norm is a legitimate, fully converged solve.
    EXPECT_TRUE(makeReport(1, 0.0).accepted(tol));
}

TEST(test_minimization_report, invalid_thresholds_reject_even_a_perfect_solve)
{
    const Report perfect = makeReport(2, 0.0);
    EXPECT_FALSE(perfect.accepted(0.0));
    EXPECT_FALSE(perfect.accepted(-0.0));
    EXPECT_FALSE(perfect.accepted(-tol));
    EXPECT_FALSE(perfect.accepted(-1.0));
    EXPECT_FALSE(perfect.accepted(nan_norm));
    EXPECT_FALSE(perfect.accepted(inf_norm));
    EXPECT_FALSE(perfect.accepted(-inf_norm));
    // ... while the smallest positive threshold it actually meets is honoured.
    EXPECT_TRUE(perfect.accepted(std::numeric_limits<Real>::denorm_min()));
}

TEST(test_minimization_report, threshold_comparison_is_inclusive_at_the_boundary)
{
    const Report atBoundary = makeReport(4, tol);
    EXPECT_TRUE(atBoundary.accepted(tol));

    const Report justAbove = makeReport(4, std::nextafter(tol, 1.0));
    EXPECT_FALSE(justAbove.accepted(tol));

    const Report justBelow = makeReport(4, std::nextafter(tol, 0.0));
    EXPECT_TRUE(justBelow.accepted(tol));
}

TEST(test_minimization_report, acceptance_tightens_monotonically_with_the_threshold)
{
    const Report report = makeReport(1, tol);
    EXPECT_TRUE(report.accepted(10.0*tol));
    EXPECT_TRUE(report.accepted(tol));
    EXPECT_FALSE(report.accepted(0.1*tol));
}

TEST(test_minimization_report, termination_classifies_the_raw_code_only)
{
    EXPECT_EQ(makeReport(-1, 0.0).termination(), Termination::NeverRan);
    EXPECT_EQ(makeReport(-7, 0.0).termination(), Termination::NeverRan);
    EXPECT_EQ(makeReport(1, 0.0).termination(), Termination::LineSearchStalled);
    EXPECT_EQ(makeReport(4, 0.0).termination(), Termination::LineSearchStalled);
    EXPECT_EQ(makeReport(2, 0.0).termination(), Termination::GradientTolerance);
    EXPECT_EQ(makeReport(3, 0.0).termination(), Termination::GradientTolerance);
    EXPECT_EQ(makeReport(5, 0.0).termination(), Termination::IterationCap);
    EXPECT_EQ(makeReport(0, 0.0).termination(), Termination::Unknown);
    EXPECT_EQ(makeReport(6, 0.0).termination(), Termination::Unknown);

    // The classification is deliberately blind to the achieved norm: a stalled
    // line search stays "stalled" whether or not it is acceptable.
    const Report stalledBad = makeReport(1, 1e3);
    EXPECT_EQ(stalledBad.termination(), Termination::LineSearchStalled);
    EXPECT_FALSE(stalledBad.accepted(tol));
    EXPECT_EQ(makeReport(1, nan_norm).termination(), Termination::LineSearchStalled);
}

TEST(test_minimization_report, acceptance_is_strictly_stronger_than_legacy_converged)
{
    // Legacy behaviour, kept source-compatible until every caller migrates:
    // codes 2 and 3 pass unconditionally and 1/4 pass without a tolerance.
    const Report toleranceHitWithHugeNorm = makeReport(2, 1e3);
    EXPECT_TRUE(toleranceHitWithHugeNorm.converged());
    EXPECT_TRUE(toleranceHitWithHugeNorm.converged(tol));
    EXPECT_FALSE(toleranceHitWithHugeNorm.accepted(tol));

    const Report stalledNoTolerance = makeReport(1, 1e3);
    EXPECT_TRUE(stalledNoTolerance.converged());
    EXPECT_FALSE(stalledNoTolerance.converged(tol));
    EXPECT_FALSE(stalledNoTolerance.accepted(tol));

    const Report unreportedNorm = makeReport(4, -1.0);
    EXPECT_TRUE(unreportedNorm.converged());
    EXPECT_FALSE(unreportedNorm.accepted(tol));

    // Both agree on the iteration cap and on a genuinely converged solve.
    EXPECT_FALSE(makeReport(5, 0.0).converged());
    EXPECT_FALSE(makeReport(5, 0.0).accepted(tol));
    EXPECT_TRUE(makeReport(1, 0.1*tol).converged(tol));
    EXPECT_TRUE(makeReport(1, 0.1*tol).accepted(tol));
}

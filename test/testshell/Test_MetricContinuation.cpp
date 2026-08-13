#include "gtest/gtest.h"

#include "common.hpp"
#include "GrowthContinuation.hpp"

TEST(MetricContinuation, PreservesInitialAndTargetMetricsAtEndpoints)
{
    Eigen::MatrixXd referenceBasis(3, 2);
    referenceBasis << 1.0, 0.0,
                      0.0, 1.0,
                      0.0, 0.0;

    Eigen::Matrix2d initialMetric;
    initialMetric << 1.0, 0.0,
                     0.0, 1.0;
    Eigen::Matrix2d targetMetric;
    targetMetric << 4.0, 0.0,
                    0.0, 9.0;

    MetricContinuation continuation(referenceBasis, initialMetric, targetMetric);

    EXPECT_TRUE(continuation.interpolate(0.0).isApprox(initialMetric));
    EXPECT_TRUE(continuation.interpolate(1.0).isApprox(targetMetric));
    EXPECT_TRUE(continuation.interpolateLogEucl(0.0).isApprox(initialMetric));
    EXPECT_TRUE(continuation.interpolateLogEucl(1.0).isApprox(targetMetric));
}

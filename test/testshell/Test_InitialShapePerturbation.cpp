#include "gtest/gtest.h"

#include "InitialShapePerturbation.hpp"

TEST(InitialShapePerturbation, FlatProfileIsZero)
{
    EXPECT_DOUBLE_EQ(initial_shape::height("flat", 0.2, -0.4, 1.0, 0.3, 4), 0.0);
}

TEST(InitialShapePerturbation, DomeOrientationsAreMirrored)
{
    const Real up = initial_shape::height("dome_up", 0.25, 0.0, 1.0, 0.2, 4);
    const Real down = initial_shape::height("dome_down", 0.25, 0.0, 1.0, 0.2, 4);

    EXPECT_GT(up, 0.0);
    EXPECT_DOUBLE_EQ(up, -down);
    EXPECT_DOUBLE_EQ(initial_shape::height("dome_up", 1.0, 0.0, 1.0, 0.2, 4), 0.0);
}

TEST(InitialShapePerturbation, SaddleChangesSignAcrossAxes)
{
    const Real alongX = initial_shape::height("saddle", 0.5, 0.0, 1.0, 0.2, 4);
    const Real alongY = initial_shape::height("saddle", 0.0, 0.5, 1.0, 0.2, 4);

    EXPECT_DOUBLE_EQ(alongX, -alongY);
}

TEST(InitialShapePerturbation, RandomProfileUsesSeededSampleAndEnvelope)
{
    EXPECT_DOUBLE_EQ(initial_shape::height("random", 0.0, 0.0, 1.0, 0.2, 4, 0.5), 0.1);
    EXPECT_DOUBLE_EQ(initial_shape::height("random", 1.0, 0.0, 1.0, 0.2, 4, 0.5), 0.0);
}

TEST(InitialShapePerturbation, RejectsInvalidModesAndWaveCounts)
{
    EXPECT_THROW(initial_shape::height("wrinkle", 0.2, 0.1, 1.0, 0.2, 0), std::invalid_argument);
    EXPECT_THROW(initial_shape::height("unknown", 0.2, 0.1, 1.0, 0.2, 4), std::invalid_argument);
}

#include "gtest/gtest.h"

#include <cmath>
#include <numbers>

#include "GrowthHelper.hpp"
#include "Mesh.hpp"

namespace
{

using TestMesh = BilayerMesh;

Eigen::Matrix2d materialEdgeBasis(
    const Eigen::MatrixXd& materialCoordinates,
    const Eigen::MatrixXi& faces)
{
    const int i0 = faces(0, 0);
    const int i1 = faces(0, 1);
    const int i2 = faces(0, 2);
    Eigen::Matrix2d Dm;
    Dm.col(0) = materialCoordinates.row(i2).transpose() -
                materialCoordinates.row(i1).transpose();
    Dm.col(1) = materialCoordinates.row(i0).transpose() -
                materialCoordinates.row(i2).transpose();
    return Dm;
}

Eigen::Matrix2d materialGrowth(
    const double angle,
    const double growth1,
    const double growth2)
{
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    Eigen::Matrix2d R;
    R << c, -s,
         s,  c;
    Eigen::Matrix2d L = Eigen::Matrix2d::Zero();
    L(0, 0) = 1.0 + growth1;
    L(1, 1) = 1.0 + growth2;
    return R * L * R.transpose();
}

struct MetricFixture
{
    Eigen::MatrixXd materialCoordinates;
    Eigen::MatrixXi faces;
    Eigen::Matrix2d Dm;
    Eigen::Matrix2d initialMetric;

    MetricFixture()
    {
        materialCoordinates.resize(3, 2);
        materialCoordinates <<
            -0.4,  0.1,
             0.8, -0.2,
             0.2,  0.9;
        faces.resize(1, 3);
        faces << 0, 1, 2;
        Dm = materialEdgeBasis(materialCoordinates, faces);
        initialMetric = Dm.transpose() * Dm;
    }
};

} // namespace

TEST(ZigZagSequenceMetric, OneIncrementMatchesMaterialTensorPullback)
{
    MetricFixture fixture;
    const double angle = 0.37;
    const double growth1 = 0.0015;
    const double growth2 = 0.0045;
    Eigen::Matrix2d metric = fixture.initialMetric;

    GrowthHelper<TestMesh>::updateAbarWithMaterialGrowthIncrement(
        fixture.materialCoordinates,
        fixture.faces,
        0,
        angle,
        growth1,
        growth2,
        metric);

    const Eigen::Matrix2d G = materialGrowth(angle, growth1, growth2);
    const Eigen::Matrix2d expected =
        fixture.Dm.transpose() * G.transpose() * G * fixture.Dm;
    EXPECT_TRUE(metric.isApprox(expected, 1e-13));
}

TEST(ZigZagSequenceMetric, EqualOrthogonalPairProducesIsotropicMaterialMetric)
{
    MetricFixture fixture;
    const double angle = 0.37;
    const double growth1 = 0.0015;
    const double growth2 = 0.0045;
    Eigen::Matrix2d metric = fixture.initialMetric;

    GrowthHelper<TestMesh>::updateAbarWithMaterialGrowthIncrement(
        fixture.materialCoordinates,
        fixture.faces,
        0,
        angle,
        growth1,
        growth2,
        metric);
    GrowthHelper<TestMesh>::updateAbarWithMaterialGrowthIncrement(
        fixture.materialCoordinates,
        fixture.faces,
        0,
        angle + 0.5 * std::numbers::pi,
        growth1,
        growth2,
        metric);

    const double stretch = (1.0 + growth1) * (1.0 + growth2);
    const Eigen::Matrix2d expected =
        (stretch * stretch) * fixture.initialMetric;
    EXPECT_TRUE(metric.isApprox(expected, 1e-12));
}

TEST(ZigZagSequenceMetric, OrthogonalPairIsOrderIndependent)
{
    MetricFixture fixture;
    const double angle = -0.81;
    const double growth1 = 0.0015;
    const double growth2 = 0.0045;
    Eigen::Matrix2d forward = fixture.initialMetric;
    Eigen::Matrix2d reverse = fixture.initialMetric;

    GrowthHelper<TestMesh>::updateAbarWithMaterialGrowthIncrement(
        fixture.materialCoordinates,
        fixture.faces,
        0,
        angle,
        growth1,
        growth2,
        forward);
    GrowthHelper<TestMesh>::updateAbarWithMaterialGrowthIncrement(
        fixture.materialCoordinates,
        fixture.faces,
        0,
        angle + 0.5 * std::numbers::pi,
        growth1,
        growth2,
        forward);

    GrowthHelper<TestMesh>::updateAbarWithMaterialGrowthIncrement(
        fixture.materialCoordinates,
        fixture.faces,
        0,
        angle + 0.5 * std::numbers::pi,
        growth1,
        growth2,
        reverse);
    GrowthHelper<TestMesh>::updateAbarWithMaterialGrowthIncrement(
        fixture.materialCoordinates,
        fixture.faces,
        0,
        angle,
        growth1,
        growth2,
        reverse);

    EXPECT_TRUE(forward.isApprox(reverse, 1e-13));
}

TEST(ZigZagSequenceMetric, IsotropicRepeatedUpdatesMatchClosedForms)
{
    MetricFixture fixture;
    constexpr double growth = 0.002;
    constexpr int increments = 7;
    Eigen::Matrix2d multiplicative = fixture.initialMetric;
    Eigen::Matrix2d recursive = fixture.initialMetric;
    Eigen::Matrix2d referenceAdditive = fixture.initialMetric;

    for(int i = 0; i < increments; ++i)
    {
        GrowthHelper<TestMesh>::updateAbarWithMaterialGrowthIncrement(
            fixture.materialCoordinates, fixture.faces, 0, 0.0,
            growth, growth, multiplicative,
            GrowthMetricUpdate::Multiplicative);
        GrowthHelper<TestMesh>::updateAbarWithMaterialGrowthIncrement(
            fixture.materialCoordinates, fixture.faces, 0, 0.0,
            growth, growth, recursive,
            GrowthMetricUpdate::RecursiveLinearized);
        GrowthHelper<TestMesh>::updateAbarWithMaterialGrowthIncrement(
            fixture.materialCoordinates, fixture.faces, 0, 0.0,
            growth, growth, referenceAdditive,
            GrowthMetricUpdate::ReferenceAdditiveLinearized);
    }

    const double multiplicativeScale =
        std::pow(1.0 + growth, 2 * increments);
    const double recursiveScale = std::pow(1.0 + 2.0 * growth, increments);
    const double referenceAdditiveScale =
        1.0 + 2.0 * increments * growth;
    EXPECT_TRUE(multiplicative.isApprox(
        multiplicativeScale * fixture.initialMetric, 1e-12));
    EXPECT_TRUE(recursive.isApprox(
        recursiveScale * fixture.initialMetric, 1e-12));
    EXPECT_TRUE(referenceAdditive.isApprox(
        referenceAdditiveScale * fixture.initialMetric, 1e-12));
}

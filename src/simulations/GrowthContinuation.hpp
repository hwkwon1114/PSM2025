//
//  GrowthContinuation.hpp
//  Elasticity
//
//  Simulation policy for ramping a prescribed target metric.
//

#ifndef GrowthContinuation_hpp
#define GrowthContinuation_hpp

#include "GrowthHelper.hpp"

/**
 * Represents a numerical continuation path between two prescribed metrics.
 *
 * This is deliberately a simulation concern: the non-Euclidean plate model
 * consumes the prescribed metric, independent of how a simulation ramps to it.
 */
class MetricContinuation
{
protected:
    const Eigen::MatrixXd referenceBasis;
    const Eigen::Matrix2d targetMetric;
    const DecomposedGrowthState targetDecomposition;
    Eigen::Matrix2d initialMetric;
    DecomposedGrowthState initialDecomposition;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MetricContinuation(const Eigen::MatrixXd & referenceBasisIn, const Eigen::Matrix2d & targetMetricIn):
    referenceBasis(referenceBasisIn),
    targetMetric(targetMetricIn),
    targetDecomposition(referenceBasis, targetMetric),
    initialMetric(referenceBasis.transpose() * referenceBasis),
    initialDecomposition(referenceBasis, initialMetric)
    {}

    MetricContinuation(const Eigen::MatrixXd & referenceBasisIn, const Eigen::Matrix2d & initialMetricIn, const Eigen::Matrix2d & targetMetricIn):
    referenceBasis(referenceBasisIn),
    targetMetric(targetMetricIn),
    targetDecomposition(referenceBasis, targetMetric),
    initialMetric(initialMetricIn),
    initialDecomposition(referenceBasis, initialMetric)
    {}

    void changeInitialMetric(const Eigen::Matrix2d & initialMetricIn)
    {
        initialMetric = initialMetricIn;
        initialDecomposition.changeMetric(referenceBasis, initialMetric);
    }

    Eigen::Matrix2d interpolateFromIsotropic(const Real t) const
    {
        const Real s1 = (1.0 - t) * initialDecomposition.get_s1() + t * targetDecomposition.get_s1();
        const Real s2 = (1.0 - t) * initialDecomposition.get_s2() + t * targetDecomposition.get_s2();
        const DecomposedGrowthState intermediate(s1, s2, targetDecomposition.get_v1(), targetDecomposition.get_v2());
        return intermediate.computeMetric(referenceBasis);
    }

    Eigen::Matrix2d interpolateFromIsotropicLogEucl(const Real t) const
    {
        const Real s1 = std::exp((1.0 - t) * std::log(initialDecomposition.get_s1()) + t * std::log(targetDecomposition.get_s1()));
        const Real s2 = std::exp((1.0 - t) * std::log(initialDecomposition.get_s2()) + t * std::log(targetDecomposition.get_s2()));
        const DecomposedGrowthState intermediate(s1, s2, targetDecomposition.get_v1(), targetDecomposition.get_v2());
        return intermediate.computeMetric(referenceBasis);
    }

    Eigen::Matrix2d interpolate(const Real t) const
    {
        if(std::abs(initialDecomposition.get_s1() - initialDecomposition.get_s2()) < 1e-12)
            return interpolateFromIsotropic(t);
        return (1.0 - t) * initialMetric + t * targetMetric;
    }

    Eigen::Matrix2d interpolateLogEucl(const Real t) const
    {
        if(std::abs(initialDecomposition.get_s1() - initialDecomposition.get_s2()) < 1e-12)
            return interpolateFromIsotropicLogEucl(t);
        return ((1.0 - t) * initialMetric.log() + t * targetMetric.log()).exp();
    }

    const DecomposedGrowthState & getInitialDecomposition() const
    {
        return initialDecomposition;
    }

    const DecomposedGrowthState & getTargetDecomposition() const
    {
        return targetDecomposition;
    }
};

#endif /* GrowthContinuation_hpp */

#ifndef CONTINUATION_STEPPING_HPP
#define CONTINUATION_STEPPING_HPP

#include <cmath>
#include <algorithm>
#include <vector>
#include <Eigen/Dense>

namespace ShellContinuation
{

inline double snapTrialLoad(const double acceptedLambda, const double stepSize)
{
    const double remainingLambda = 1.0 - acceptedLambda;
    if(remainingLambda <= 1e-12) return acceptedLambda;
    double effectiveStep = stepSize;
    if(effectiveStep >= remainingLambda || (remainingLambda - effectiveStep) < 1e-6)
        effectiveStep = remainingLambda;
    return acceptedLambda + effectiveStep;
}

inline double computeNextStepOnFailure(const double trialLambda, const double acceptedLambda, const double minStep)
{
    const double actualAttempted = trialLambda - acceptedLambda;
    return std::max(minStep, 0.5 * actualAttempted);
}

inline double compute2DMetricStrainJump(const Eigen::Matrix2d& A_acc, const Eigen::Matrix2d& dA)
{
    if(dA.squaredNorm() <= 1e-28) return 0.0;
    const Eigen::Matrix2d M = A_acc.inverse() * dA;
    const double tr = M.trace();
    const double det = M.determinant();
    const double disc = std::max(0.0, tr * tr - 4.0 * det);
    const double sqrtDisc = std::sqrt(disc);
    const double s1 = std::abs(0.5 * (tr + sqrtDisc));
    const double s2 = std::abs(0.5 * (tr - sqrtDisc));
    return 0.5 * std::max(s1, s2);
}

template<typename VecMat2d>
inline double computeMaxMetricStrainJump(
    const VecMat2d& startTop,
    const VecMat2d& endTop,
    const VecMat2d& startBot,
    const VecMat2d& endBot,
    const double lamStart,
    const double lamEnd)
{
    const double dLam = lamEnd - lamStart;
    if(std::abs(dLam) <= 1e-15) return 0.0;
    double maxStrain = 0.0;
    const size_t nFaces = startTop.size();
    for(size_t f = 0; f < nFaces; ++f)
    {
        for(int layer = 0; layer < 2; ++layer)
        {
            const auto& startA = (layer == 0) ? startTop[f] : startBot[f];
            const auto& endA = (layer == 0) ? endTop[f] : endBot[f];
            const auto dA_total = endA - startA;
            if(dA_total.squaredNorm() <= 1e-28) continue;
            const auto A_acc = startA + lamStart * dA_total;
            const auto dA = dLam * dA_total;
            maxStrain = std::max(maxStrain, compute2DMetricStrainJump(A_acc, dA));
        }
    }
    return maxStrain;
}

} // namespace ShellContinuation

#endif // CONTINUATION_STEPPING_HPP

#ifndef ROBUST_CURVATURE_HANDOFF_HPP
#define ROBUST_CURVATURE_HANDOFF_HPP

#include "NonconvexBenchmark.hpp"
#include "BenchmarkNewtonFactorization.hpp"
#include <chrono>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>

namespace NonconvexBenchmark {

/**
 * @brief Configuration parameters for adaptive curvature-aware handoff from L-BFGS to Newton.
 */
struct RobustCurvatureOptions {
    double gradientToleranceGate = 1e-7;   //!< Only probe Hessian curvature once ||g|| <= this value
    int minWarmupAttempts = 200;           //!< Minimum L-BFGS attempts before any curvature probe is allowed
    int checkInterval = 50;                //!< Evaluate curvature probe every checkInterval attempts
    double pivotFloor = 1e-12;             //!< Relative pivot tolerance for positive-definiteness certification
    std::string newtonBackend = "eigen";   //!< Linear algebra backend: "eigen" or "cholmod"
    int linearSolverThreads = 1;           //!< Threads for linear solver / factorization
};

/**
 * @brief Detailed diagnostic output from a curvature probe evaluation.
 */
struct CurvatureProbeResult {
    bool probed = false;                   //!< True if the probe actually assembled and factored the Hessian
    bool certifiedConvex = false;          //!< True if unshifted Hessian is strictly positive-definite
    int attempts = 0;                      //!< Attempt count when probed
    double gradientNorm = 0.0;             //!< Unscaled gradient norm ||g||
    double curvatureScale = 0.0;           //!< Max absolute entry of reduced Hessian
    double minPivot = 0.0;                 //!< Smallest pivot from factorization (if available)
    double probeSeconds = 0.0;             //!< Wall-clock time spent in probe
    std::string reason;                    //!< Diagnostic reason string
};

/**
 * @brief Curvature-aware adaptive switch gate for L-BFGS -> Newton transition.
 *
 * Verifies whether the objective has physically entered the positive-definite basin
 * of attraction by performing a trial zero-shift factorization of the reduced Hessian.
 * This prevents prematurely triggering Newton on buckling saddles or indefinite ridges
 * where Levenberg-Marquardt shifting would destroy quadratic convergence.
 */
class RobustCurvatureGate {
    RobustCurvatureOptions options;
    std::unique_ptr<BenchmarkNewtonFactorization> factor;

public:
    explicit RobustCurvatureGate(const RobustCurvatureOptions& opt = RobustCurvatureOptions())
        : options(opt)
    {
#ifdef NONCONVEX_WITH_CHOLMOD
        if (options.newtonBackend == "cholmod") {
            factor = std::make_unique<BenchmarkNewtonFactorization>("cholmod", options.linearSolverThreads);
        } else {
            factor = std::make_unique<BenchmarkNewtonFactorization>("eigen", options.linearSolverThreads);
        }
#else
        factor = std::make_unique<BenchmarkNewtonFactorization>("eigen", options.linearSolverThreads);
#endif
    }

    const RobustCurvatureOptions& getOptions() const { return options; }

    /**
     * @brief Probe curvature from an engine State.
     */
    CurvatureProbeResult probe(Problem& problem, const State& state, Json* auditRow = nullptr) {
        return probeDirect(problem, state.x, state.g, state.attempts, auditRow);
    }

    /**
     * @brief Direct probe given position, gradient, and attempt counter.
     */
    CurvatureProbeResult probeDirect(Problem& problem, const Vec& x, const Vec& g, int attempts = 0, Json* auditRow = nullptr) {
        CurvatureProbeResult result;
        result.attempts = attempts;
        result.gradientNorm = g.norm();

        if (!std::isfinite(result.gradientNorm)) {
            result.reason = "nonfinite_gradient";
            if (auditRow) (*auditRow)["curvature_probe"] = {{"status", result.reason}, {"attempts", attempts}};
            return result;
        }

        // 1. Warmup guard: do not waste compute early in the simulation when the shell is far from equilibrium
        if (attempts < options.minWarmupAttempts) {
            result.reason = "warmup_active";
            if (auditRow) (*auditRow)["curvature_probe"] = {{"status", result.reason}, {"attempts", attempts}};
            return result;
        }

        // 2. Coarse gradient gate: only inspect when the first-order residual is reasonably small
        if (result.gradientNorm > options.gradientToleranceGate) {
            result.reason = "gradient_above_gate";
            if (auditRow) (*auditRow)["curvature_probe"] = {{"status", result.reason}, {"gradient_norm", result.gradientNorm}};
            return result;
        }

        // 3. Periodic interval: do not assemble Hessian on every single step
        if (options.checkInterval > 1 && ((attempts - options.minWarmupAttempts) % options.checkInterval != 0)) {
            result.reason = "interval_skip";
            if (auditRow) (*auditRow)["curvature_probe"] = {{"status", result.reason}, {"attempts", attempts}};
            return result;
        }

        // 4. Physical / Numerical curvature evaluation
        result.probed = true;
        const auto start = std::chrono::steady_clock::now();

        // Check scaling and energy reference
        const int n = problem.size();
        const Vec S = problem.scales();
        const double E0 = problem.energyReference();
        if (!std::isfinite(E0) || E0 <= 0.0) {
            result.reason = "invalid_energy_reference";
            return result;
        }
        if (S.size() != n || (S.array() <= 0.0).any() || !S.allFinite()) {
            result.reason = "invalid_scales";
            return result;
        }

        // Assemble full Hessian
        const Sparse H = problem.hessian(x);
        if (H.rows() != n || H.cols() != n) {
            result.reason = "invalid_hessian_dimensions";
            return result;
        }

        // Determine free DOFs (gauge elimination)
        std::vector<int> free = problem.freeDofs(x, S, {});
        if (free.empty()) {
            result.reason = "no_free_dofs";
            return result;
        }

        // Build reduced Hessian matrix in coordinate space of free DOFs
        std::vector<int> map(n, -1);
        for (size_t j = 0; j < free.size(); ++j) {
            const int idx = free[j];
            if (idx >= 0 && idx < n) map[idx] = static_cast<int>(j);
        }

        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(H.nonZeros());
        for (int j = 0; j < H.outerSize(); ++j) {
            for (Sparse::InnerIterator it(H, j); it; ++it) {
                if (!std::isfinite(it.value())) {
                    result.reason = "nonfinite_hessian_entry";
                    return result;
                }
                if (map[it.row()] >= 0 && map[it.col()] >= 0) {
                    const double a = it.value() * S[it.row()] * S[it.col()] / E0;
                    if (!std::isfinite(a)) {
                        result.reason = "nonfinite_scaled_entry";
                        return result;
                    }
                    triplets.emplace_back(map[it.row()], map[it.col()], a);
                }
            }
        }

        Sparse A(free.size(), free.size());
        A.setFromTriplets(triplets.begin(), triplets.end());
        Sparse trans = A.transpose();
        A = 0.5 * (A + trans);

        // Curvature scale
        double scale = 1e-12;
        for (int j = 0; j < A.outerSize(); ++j) {
            for (Sparse::InnerIterator it(A, j); it; ++it) {
                if (!std::isfinite(it.value())) {
                    result.reason = "nonfinite_symmetrized_entry";
                    return result;
                }
                scale = std::max(scale, std::abs(it.value()));
            }
        }
        if (!std::isfinite(scale) || scale <= 0.0) {
            result.reason = "invalid_curvature_scale";
            return result;
        }
        result.curvatureScale = scale;

        // Perform zero-shift factorization
        factor->analyzePattern(A);
        factor->factorize(A);

        const double pivotThreshold = options.pivotFloor * scale;
        const bool acceptable = factor->acceptable(pivotThreshold);

        const auto finish = std::chrono::steady_clock::now();
        result.probeSeconds = std::chrono::duration<double>(finish - start).count();

        if (acceptable) {
            result.certifiedConvex = true;
            result.reason = "certified_convex";
        } else {
            result.certifiedConvex = false;
            result.reason = "indefinite_requires_shift";
        }

        if (auditRow) {
            (*auditRow)["curvature_probe"] = {
                {"probed", true},
                {"certified_convex", result.certifiedConvex},
                {"reason", result.reason},
                {"scale", result.curvatureScale},
                {"seconds", result.probeSeconds},
                {"attempts", attempts},
                {"gradient_norm", result.gradientNorm}
            };
        }

        return result;
    }
};

} // namespace NonconvexBenchmark

#endif // ROBUST_CURVATURE_HANDOFF_HPP

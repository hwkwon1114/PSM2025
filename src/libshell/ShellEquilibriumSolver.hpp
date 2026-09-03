#ifndef ShellEquilibriumSolver_hpp
#define ShellEquilibriumSolver_hpp

/**
 * ShellEquilibriumSolver : solver primitives for equilibrium of the DCS shell discretization.
 *
 * Three independent, matrix-free primitives, all operating on the full mesh DOF layout
 * (global index of vertex k coordinate d is d*nV + k, edge director e is 3*nV + e -- the
 * layout of mesh.getDataPointer(), CombinedOperator_Parametric gradients and
 * TinyADHessian_Bilayer):
 *
 *   1. RigidModeProjector  -- boundary-condition-aware projector that removes the fixed DOFs
 *      and the rigid-body modes that the boundary conditions actually leave as exact energy
 *      symmetries. For a free shell that is the full 6-dimensional rigid space; for a
 *      constrained shell it is the subspace of rigid motions that keep every constrained DOF
 *      at zero (and only those: a rigid mode with a nonzero entry on a constrained DOF is not
 *      a null direction of the constrained Hessian and is therefore never projected out).
 *      MeshRigidModeProjection wraps it for the solver and rebuilds the basis whenever the
 *      configuration moves (the rotation modes depend on the current vertex positions).
 *
 *   2. smallestProjectedRitzPair -- deterministic matrix-free Lanczos with full (twice-is-
 *      enough) reorthogonalization and thick restarts, returning the smallest Ritz value of
 *      the projected operator together with the explicitly recomputed absolute and relative
 *      residual ||P(A u) - theta u||. Convergence is reported only when that residual meets
 *      the requested tolerance -- a budget-exhausted run returns its best Ritz pair with
 *      residualConverged == false.
 *
 *   3. steihaugTruncatedCG / trustRegionNewtonCG -- Steihaug-Toint truncated-CG trust-region
 *      Newton corrector. The trust region is measured in a diagonal metric M (separate vertex
 *      and edge-director scales, or a Jacobi metric), which is also the CG preconditioner, so
 *      the O(1/(h^2 l_e)) stiffness ratio between stretching and bending DOFs does not
 *      distort the step or stall CG. The predicted reduction is evaluated with the Hessian of
 *      the *base* state (H s is accumulated inside CG, never re-evaluated at the trial state)
 *      and a rejected trial restores the base state exactly.
 *
 * This is a minimizer: it descends to a nearby equilibrium. It deliberately contains no
 * branch tracing / branch switching / mode injection -- the smallest Ritz pair is reported as
 * a diagnostic only, and no claim about which physical branch a state belongs to is made or
 * implied here.
 *
 * The primitives are templates over callables/operators, so they can be driven by
 * TinyADHessian_Bilayer + CombinedOperator_Parametric (see MeshEquilibriumProblem and
 * solveShellEquilibrium) or by any other symmetric operator.
 */

#include "common.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ShellEquilibrium
{
    // -----------------------------------------------------------------------------------
    // DOF layout / boundary conditions
    // -----------------------------------------------------------------------------------

    /*! total number of DOFs of the DCS layout [x|y|z|directors] */
    inline int numberOfDofs(const int nVertices, const int nEdges)
    {
        return 3 * nVertices + nEdges;
    }

    /*! mask of DOFs held fixed by the boundary conditions (1 == constrained), in the global
     *  DOF layout. Identical convention to TinyADHessian_Bilayer's internal mask. */
    template<typename tMesh>
    std::vector<char> constrainedDofMask(const tMesh & mesh)
    {
        const int nVertices = mesh.getNumberOfVertices();
        const int nEdges = mesh.getNumberOfEdges();
        std::vector<char> mask(numberOfDofs(nVertices, nEdges), 0);

        const auto & bc = mesh.getBoundaryConditions();
        const auto vertices_bc = bc.getVertexBoundaryConditions();
        const auto edges_bc = bc.getEdgeBoundaryConditions();

        if(vertices_bc.rows() == nVertices && vertices_bc.cols() == 3)
            for(int i = 0; i < nVertices; ++i)
                for(int d = 0; d < 3; ++d)
                    if(vertices_bc(i, d)) mask[d * nVertices + i] = 1;

        if(edges_bc.size() == nEdges)
            for(int e = 0; e < nEdges; ++e)
                if(edges_bc(e)) mask[3 * nVertices + e] = 1;

        return mask;
    }

    // -----------------------------------------------------------------------------------
    // Trust-region / preconditioner metrics
    // -----------------------------------------------------------------------------------

    /*! diagonal metric with separate vertex and edge-director scales:
     *      ||s||_M^2 = sum_v (s_v / vertexScale)^2 + sum_e (s_e / directorScale)^2
     *  Vertex DOFs carry a length, director DOFs an angle -- an unscaled l2 trust region
     *  therefore equates metres with radians and, with the O(E h) vs O(E h^3 l_e) diagonal
     *  stiffness ratio, either chokes bending or lets in-plane motion invert elements.
     *  vertexScale is a characteristic length (e.g. the mean edge length or the shell
     *  thickness) and directorScale a characteristic angle (e.g. 1 rad). */
    inline Eigen::VectorXd blockDiagonalMetric(const int nVertices, const int nEdges,
                                               const Real vertexScale, const Real directorScale)
    {
        if(!(vertexScale > 0.0) || !(directorScale > 0.0))
            throw std::invalid_argument("ShellEquilibrium::blockDiagonalMetric: scales must be positive");

        Eigen::VectorXd metric(numberOfDofs(nVertices, nEdges));
        metric.head(3 * nVertices).setConstant(1.0 / (vertexScale * vertexScale));
        metric.tail(nEdges).setConstant(1.0 / (directorScale * directorScale));
        return metric;
    }

    /*! Jacobi metric M_ii = max(|H_ii|, relativeFloor * max_j |H_jj|), constrained DOFs set to
     *  one. Use with the diagonal of TinyADHessian_Bilayer::assembleHessian() when an assembled
     *  Hessian is affordable; blockDiagonalMetric is the matrix-free alternative. */
    inline Eigen::VectorXd jacobiMetric(const Eigen::Ref<const Eigen::VectorXd> & hessianDiagonal,
                                        const std::vector<char> & mask,
                                        const Real relativeFloor = 1e-8)
    {
        const int nDofs = static_cast<int>(hessianDiagonal.size());
        if(!mask.empty() && static_cast<int>(mask.size()) != nDofs)
            throw std::invalid_argument("ShellEquilibrium::jacobiMetric: mask length mismatch");

        Eigen::VectorXd metric = hessianDiagonal.cwiseAbs();
        const Real largest = (nDofs > 0 ? metric.maxCoeff() : 0.0);
        const Real floorValue = (largest > 0.0 ? relativeFloor * largest : 1.0);
        for(int i = 0; i < nDofs; ++i)
        {
            if(!mask.empty() && mask[i]) metric(i) = 1.0;
            else if(!(metric(i) > floorValue)) metric(i) = floorValue;
        }
        return metric;
    }

    // -----------------------------------------------------------------------------------
    // Rigid-mode projection
    // -----------------------------------------------------------------------------------

    /*! Projection onto the subspace of admissible variations: constrained DOFs are zeroed and
     *  the boundary-condition-compatible rigid-body modes are removed.
     *
     *  Rigid motions act on the vertices only: the edge directors measure the angle between
     *  the mid-edge normal and the average face normal, which is invariant under global
     *  translations and rotations, so all six raw modes have zero director components.
     *
     *  Boundary-condition awareness: the six raw modes are restricted to the subspace whose
     *  constrained-DOF entries vanish (nullspace of the constrained rows). Only those motions
     *  are exact symmetries of the constrained energy and hence exact null directions of the
     *  constrained Hessian. A clamped shell typically yields zero rigid modes, a shell pinned
     *  at two vertices yields the single rotation about the line joining them, and a free
     *  shell yields all six. */
    class RigidModeProjector
    {
        int nVertices_ = 0;
        int nEdges_ = 0;
        int nDofs_ = 0;
        std::vector<char> mask_;
        Eigen::MatrixXd basis_;   // nDofs x nRigidModes, orthonormal, zero on constrained DOFs

        /*! modified Gram-Schmidt with two passes; columns whose norm collapses are dropped */
        static Eigen::MatrixXd orthonormalizeColumns(const Eigen::MatrixXd & columns, const Real dropTolerance)
        {
            const int nRows = static_cast<int>(columns.rows());
            const int nCols = static_cast<int>(columns.cols());
            Eigen::MatrixXd basis(nRows, nCols);
            int kept = 0;
            for(int c = 0; c < nCols; ++c)
            {
                Eigen::VectorXd v = columns.col(c);
                const Real initialNorm = v.norm();
                if(!(initialNorm > 0.0)) continue;
                for(int pass = 0; pass < 2; ++pass)
                    for(int k = 0; k < kept; ++k)
                        v.noalias() -= basis.col(k) * basis.col(k).dot(v);
                const Real norm = v.norm();
                if(!(norm > dropTolerance * initialNorm)) continue;
                basis.col(kept) = v / norm;
                ++kept;
            }
            return basis.leftCols(kept).eval();
        }

    public:

        RigidModeProjector() = default;

        /*! \param vertices           nVertices x 3 current vertex positions
         *  \param nEdges             number of edge directors
         *  \param mask               constrained-DOF mask (empty == nothing constrained)
         *  \param nullspaceTolerance fraction of a (unit-normalized) rigid mode's squared norm
         *                            that may live on constrained DOFs and still count as zero
         *  \param dropTolerance      relative Gram-Schmidt drop threshold */
        RigidModeProjector(const Eigen::Ref<const Eigen::MatrixXd> & vertices,
                           const int nEdges,
                           const std::vector<char> & mask,
                           const Real nullspaceTolerance = 1e-12,
                           const Real dropTolerance = 1e-8)
        : nVertices_(static_cast<int>(vertices.rows())), nEdges_(nEdges),
          nDofs_(numberOfDofs(static_cast<int>(vertices.rows()), nEdges)), mask_(mask)
        {
            if(vertices.cols() != 3)
                throw std::invalid_argument("ShellEquilibrium::RigidModeProjector: vertices must be nVertices x 3");
            if(nEdges_ < 0)
                throw std::invalid_argument("ShellEquilibrium::RigidModeProjector: negative number of edges");
            if(mask_.empty()) mask_.assign(nDofs_, 0);
            if(static_cast<int>(mask_.size()) != nDofs_)
                throw std::invalid_argument("ShellEquilibrium::RigidModeProjector: mask length mismatch");

            if(nVertices_ == 0)
            {
                basis_.resize(nDofs_, 0);
                return;
            }

            // ---- the six raw rigid modes about the vertex centroid ----
            const Eigen::Vector3d centroid = vertices.colwise().mean();
            Eigen::MatrixXd raw = Eigen::MatrixXd::Zero(nDofs_, 6);
            for(int i = 0; i < nVertices_; ++i)
            {
                const Real dx = vertices(i, 0) - centroid(0);
                const Real dy = vertices(i, 1) - centroid(1);
                const Real dz = vertices(i, 2) - centroid(2);

                raw(0 * nVertices_ + i, 0) = 1.0;                                     // translation x
                raw(1 * nVertices_ + i, 1) = 1.0;                                     // translation y
                raw(2 * nVertices_ + i, 2) = 1.0;                                     // translation z
                raw(1 * nVertices_ + i, 3) = -dz; raw(2 * nVertices_ + i, 3) =  dy;   // rotation about x
                raw(0 * nVertices_ + i, 4) =  dz; raw(2 * nVertices_ + i, 4) = -dx;   // rotation about y
                raw(0 * nVertices_ + i, 5) = -dy; raw(1 * nVertices_ + i, 5) =  dx;   // rotation about z
            }

            // normalize so the nullspace tolerance below is a pure energy fraction, and drop
            // modes that are identically zero (degenerate vertex clouds)
            Eigen::MatrixXd modes(nDofs_, 6);
            int nModes = 0;
            for(int c = 0; c < 6; ++c)
            {
                const Real norm = raw.col(c).norm();
                if(!(norm > 0.0)) continue;
                modes.col(nModes) = raw.col(c) / norm;
                ++nModes;
            }
            modes.conservativeResize(nDofs_, nModes);

            // ---- restrict to the rigid motions compatible with the boundary conditions ----
            int nConstrained = 0;
            for(int i = 0; i < nDofs_; ++i) if(mask_[i]) ++nConstrained;

            Eigen::MatrixXd admissible;
            if(nConstrained == 0 || nModes == 0)
            {
                admissible = modes;
            }
            else
            {
                // Gram matrix of the constrained rows: c^T G c is the fraction of the squared
                // norm of the rigid motion "modes * c" that sits on constrained DOFs.
                Eigen::MatrixXd constrainedRows(nConstrained, nModes);
                int row = 0;
                for(int i = 0; i < nDofs_; ++i)
                    if(mask_[i]) constrainedRows.row(row++) = modes.row(i);

                const Eigen::MatrixXd gram = constrainedRows.transpose() * constrainedRows;
                const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(gram);
                if(eigenSolver.info() != Eigen::Success)
                    throw std::runtime_error("ShellEquilibrium::RigidModeProjector: rigid-mode Gram eigensolve failed");

                Eigen::MatrixXd coefficients(nModes, nModes);
                int nAdmissible = 0;
                for(int c = 0; c < nModes; ++c)
                    if(eigenSolver.eigenvalues()(c) <= nullspaceTolerance)
                        coefficients.col(nAdmissible++) = eigenSolver.eigenvectors().col(c);
                coefficients.conservativeResize(nModes, nAdmissible);

                admissible = (nAdmissible > 0 ? Eigen::MatrixXd(modes * coefficients)
                                              : Eigen::MatrixXd(nDofs_, 0));
                // the admissible modes are zero on the constrained DOFs up to the tolerance --
                // make that exact so the projector can never touch a constrained DOF
                for(int i = 0; i < nDofs_; ++i)
                    if(mask_[i]) admissible.row(i).setZero();
            }

            basis_ = orthonormalizeColumns(admissible, dropTolerance);
        }

        template<typename tMesh>
        static RigidModeProjector fromMesh(const tMesh & mesh,
                                           const Real nullspaceTolerance = 1e-12,
                                           const Real dropTolerance = 1e-8)
        {
            return RigidModeProjector(mesh.getCurrentConfiguration().getVertices(),
                                      mesh.getNumberOfEdges(),
                                      constrainedDofMask(mesh),
                                      nullspaceTolerance, dropTolerance);
        }

        /*! in-place projection: zero the constrained DOFs, then remove the rigid components.
         *  Applied twice for numerical idempotence; the second pass is free of round-off
         *  re-entry because the basis vanishes on the constrained DOFs. */
        void applyInPlace(Eigen::VectorXd & v) const
        {
            if(v.size() != nDofs_)
                throw std::invalid_argument("ShellEquilibrium::RigidModeProjector: vector length mismatch");

            for(int i = 0; i < nDofs_; ++i)
                if(mask_[i]) v(i) = 0.0;

            if(basis_.cols() > 0)
            {
                v.noalias() -= basis_ * (basis_.transpose() * v);
                v.noalias() -= basis_ * (basis_.transpose() * v);
            }
        }

        void operator()(Eigen::VectorXd & v) const { applyInPlace(v); }

        Eigen::VectorXd apply(const Eigen::VectorXd & v) const
        {
            Eigen::VectorXd out = v;
            applyInPlace(out);
            return out;
        }

        /*! the same projection expressed in the scaled coordinates y = D x, D = diag(scaling).
         *  Used to run the eigensolver on the congruent operator D^{-1} H D^{-1}, which has the
         *  same inertia (Sylvester) but O(1) conditioning. */
        RigidModeProjector transformedByDiagonalScaling(const Eigen::Ref<const Eigen::VectorXd> & scaling,
                                                        const Real dropTolerance = 1e-8) const
        {
            if(scaling.size() != nDofs_)
                throw std::invalid_argument("ShellEquilibrium::RigidModeProjector: scaling length mismatch");

            RigidModeProjector scaled(*this);
            if(basis_.cols() > 0)
                scaled.basis_ = orthonormalizeColumns(scaling.asDiagonal() * basis_, dropTolerance);
            return scaled;
        }

        int numberOfRigidModes() const { return static_cast<int>(basis_.cols()); }
        int numberOfConstrainedDofs() const
        {
            int count = 0;
            for(int i = 0; i < nDofs_; ++i) if(mask_[i]) ++count;
            return count;
        }
        const Eigen::MatrixXd & basis() const { return basis_; }
        const std::vector<char> & mask() const { return mask_; }
        int nDofs() const { return nDofs_; }
        int nVertices() const { return nVertices_; }
        int nEdges() const { return nEdges_; }
    };

    /*! projection that leaves every vector untouched (plain unconstrained operator) */
    struct NoProjection
    {
        void operator()(Eigen::VectorXd &) const {}
    };

    /*! Mesh-bound projection whose rigid basis follows the deforming configuration: the
     *  rotation modes depend on the current vertex positions, so a basis frozen at the initial
     *  state is only correct to first order in the accumulated displacement. refresh() rebuilds
     *  it; trustRegionNewtonCG calls that hook whenever the base state has changed. */
    template<typename tMesh>
    class MeshRigidModeProjection
    {
        const tMesh & mesh_;
        Real nullspaceTolerance_;
        Real dropTolerance_;
        RigidModeProjector projector_;

    public:

        explicit MeshRigidModeProjection(const tMesh & mesh,
                                         const Real nullspaceTolerance = 1e-12,
                                         const Real dropTolerance = 1e-8)
        : mesh_(mesh), nullspaceTolerance_(nullspaceTolerance), dropTolerance_(dropTolerance),
          projector_(RigidModeProjector::fromMesh(mesh, nullspaceTolerance, dropTolerance))
        {}

        void refresh()
        {
            projector_ = RigidModeProjector::fromMesh(mesh_, nullspaceTolerance_, dropTolerance_);
        }

        void operator()(Eigen::VectorXd & v) const { projector_.applyInPlace(v); }

        const RigidModeProjector & projector() const { return projector_; }
    };

    // -----------------------------------------------------------------------------------
    // Smallest projected Ritz pair (matrix-free Lanczos)
    // -----------------------------------------------------------------------------------

    struct SmallestRitzPairOptions
    {
        int krylovDimension = 40;              //!< Lanczos vectors per restart cycle
        int maxRestarts = 20;                  //!< thick restarts with the current Ritz vector
        Real absoluteResidualTolerance = 1e-10;//!< converged if ||A u - theta u|| <= this ...
        Real relativeResidualTolerance = 1e-6; //!< ... or if that residual / |theta| <= this
        Real breakdownTolerance = 1e-13;       //!< Lanczos beta below this ends the cycle
        std::uint64_t seed = 20250903ull;      //!< deterministic start vector
    };

    struct RitzPairReport
    {
        Real eigenvalue = 0.0;                 //!< smallest Ritz value of the projected operator
        Real residualAbsolute = std::numeric_limits<Real>::infinity();
        Real residualRelative = std::numeric_limits<Real>::infinity();
        Eigen::VectorXd eigenvector;           //!< unit, projected Ritz vector (empty if none)
        bool residualConverged = false;        //!< strictly implied by the residuals above
        bool breakdown = false;                //!< Krylov space collapsed (invariant subspace or empty subspace)
        int subspaceDimension = 0;             //!< Lanczos vectors of the last cycle
        int lanczosIterations = 0;             //!< total Lanczos steps over all restarts
        int restarts = 0;
        int operatorEvaluations = 0;
        Real requestedAbsoluteTolerance = 0.0;
        Real requestedRelativeTolerance = 0.0;
    };

    namespace detail
    {
        /*! splitmix64 -> [-1,1): fully specified by the standard, hence bit-reproducible on
         *  every platform (unlike std::normal_distribution) */
        inline Real deterministicUnit(std::uint64_t & state)
        {
            state += 0x9E3779B97F4A7C15ull;
            std::uint64_t z = state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            z ^= (z >> 31);
            return 2.0 * static_cast<Real>(z >> 11) * (1.0 / 9007199254740992.0) - 1.0;
        }

        inline Real relativeResidual(const Real absolute, const Real eigenvalue)
        {
            const Real scale = std::abs(eigenvalue);
            if(!(scale > 0.0)) return std::numeric_limits<Real>::infinity();
            return absolute / scale;
        }

        /*! opt-in hook: projections that depend on the configuration (MeshRigidModeProjection)
         *  expose refresh(); plain projections are left alone at zero cost */
        template<typename tProjection, typename = void>
        struct HasRefresh : std::false_type {};

        template<typename tProjection>
        struct HasRefresh<tProjection, std::void_t<decltype(std::declval<tProjection&>().refresh())>>
        : std::true_type {};

        template<typename tProjection>
        void refreshProjection(tProjection && project)
        {
            using tPlain = typename std::remove_reference<tProjection>::type;
            if constexpr (HasRefresh<tPlain>::value) project.refresh();
        }
    }

    /*! Smallest eigenpair of the projected symmetric operator P A P, matrix-free.
     *
     *  \param nDofs         length of the state vectors
     *  \param applyOperator v -> A v (symmetric; the projection is applied here, not by A)
     *  \param project       in-place projection P (e.g. RigidModeProjector)
     *
     *  Deterministic: the start vector depends only on options.seed and nDofs, and no
     *  randomness enters afterwards. Full (twice-is-enough) reorthogonalization keeps the
     *  Lanczos basis orthogonal, so the tridiagonal Ritz values are trustworthy; the returned
     *  residual is nevertheless recomputed explicitly from an extra operator application, and
     *  residualConverged is set from that residual alone. */
    template<typename tOperator, typename tProjection>
    RitzPairReport smallestProjectedRitzPair(const int nDofs,
                                             tOperator && applyOperator,
                                             tProjection && project,
                                             const SmallestRitzPairOptions & options = SmallestRitzPairOptions())
    {
        RitzPairReport report;
        report.requestedAbsoluteTolerance = options.absoluteResidualTolerance;
        report.requestedRelativeTolerance = options.relativeResidualTolerance;

        if(nDofs <= 0)
            throw std::invalid_argument("ShellEquilibrium::smallestProjectedRitzPair: nDofs must be positive");
        if(options.krylovDimension < 1)
            throw std::invalid_argument("ShellEquilibrium::smallestProjectedRitzPair: krylovDimension must be >= 1");

        const int krylovDimension = std::min(options.krylovDimension, nDofs);

        // ---- deterministic, projected start vector ----
        std::uint64_t rngState = options.seed;
        Eigen::VectorXd start(nDofs);
        Real startNorm = 0.0;
        for(int attempt = 0; attempt < 8 && !(startNorm > 0.0); ++attempt)
        {
            for(int i = 0; i < nDofs; ++i) start(i) = detail::deterministicUnit(rngState);
            project(start);
            startNorm = start.norm();
        }
        if(!(startNorm > 0.0))
        {
            // the admissible subspace is empty (everything constrained / projected away)
            report.breakdown = true;
            return report;
        }
        start /= startNorm;

        std::vector<Eigen::VectorXd> lanczos;
        lanczos.reserve(static_cast<std::size_t>(krylovDimension) + 1);
        std::vector<Real> alpha, beta;
        alpha.reserve(static_cast<std::size_t>(krylovDimension));
        beta.reserve(static_cast<std::size_t>(krylovDimension));

        Eigen::VectorXd w(nDofs);
        Eigen::VectorXd ritzVector(nDofs);

        for(int restart = 0; restart <= options.maxRestarts; ++restart)
        {
            report.restarts = restart;

            lanczos.clear();
            alpha.clear();
            beta.clear();
            lanczos.push_back(start);

            bool invariant = false;
            int dimension = 0;
            for(int j = 0; j < krylovDimension; ++j)
            {
                w = applyOperator(lanczos[static_cast<std::size_t>(j)]);
                ++report.operatorEvaluations;
                ++report.lanczosIterations;
                project(w);

                const Real a = lanczos[static_cast<std::size_t>(j)].dot(w);
                alpha.push_back(a);
                dimension = j + 1;

                w.noalias() -= a * lanczos[static_cast<std::size_t>(j)];
                if(j > 0) w.noalias() -= beta[static_cast<std::size_t>(j - 1)] * lanczos[static_cast<std::size_t>(j - 1)];

                // full reorthogonalization (twice is enough), staying inside the subspace
                for(int pass = 0; pass < 2; ++pass)
                {
                    for(int k = 0; k <= j; ++k)
                        w.noalias() -= lanczos[static_cast<std::size_t>(k)].dot(w) * lanczos[static_cast<std::size_t>(k)];
                    project(w);
                }

                const Real b = w.norm();
                if(!(b > options.breakdownTolerance))
                {
                    invariant = true;
                    break;
                }
                beta.push_back(b);
                if(j + 1 < krylovDimension) lanczos.push_back(w / b);
            }

            report.subspaceDimension = dimension;

            // ---- smallest Ritz pair of the tridiagonal projection ----
            Eigen::MatrixXd tridiagonal = Eigen::MatrixXd::Zero(dimension, dimension);
            for(int i = 0; i < dimension; ++i)
            {
                tridiagonal(i, i) = alpha[static_cast<std::size_t>(i)];
                if(i + 1 < dimension && static_cast<int>(beta.size()) > i)
                {
                    tridiagonal(i, i + 1) = beta[static_cast<std::size_t>(i)];
                    tridiagonal(i + 1, i) = beta[static_cast<std::size_t>(i)];
                }
            }

            const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(tridiagonal);
            if(eigenSolver.info() != Eigen::Success)
                throw std::runtime_error("ShellEquilibrium::smallestProjectedRitzPair: tridiagonal eigensolve failed");

            // the Ritz vector; its Rayleigh quotient below is used as the eigenvalue, which
            // minimizes the residual for this vector and is never worse than the tridiagonal
            // Ritz value
            const Eigen::VectorXd coefficients = eigenSolver.eigenvectors().col(0);

            ritzVector.setZero();
            for(int i = 0; i < dimension; ++i)
                ritzVector.noalias() += coefficients(i) * lanczos[static_cast<std::size_t>(i)];
            project(ritzVector);
            const Real ritzNorm = ritzVector.norm();
            if(!(ritzNorm > 0.0))
            {
                report.breakdown = true;
                return report;
            }
            ritzVector /= ritzNorm;

            // ---- explicit residual of the returned pair (never inferred from the tridiagonal) ----
            w = applyOperator(ritzVector);
            ++report.operatorEvaluations;
            project(w);
            const Real rayleigh = ritzVector.dot(w);
            w.noalias() -= rayleigh * ritzVector;

            report.eigenvalue = rayleigh;
            report.residualAbsolute = w.norm();
            report.residualRelative = detail::relativeResidual(report.residualAbsolute, rayleigh);
            report.eigenvector = ritzVector;
            report.residualConverged = (report.residualAbsolute <= options.absoluteResidualTolerance)
                                    || (report.residualRelative <= options.relativeResidualTolerance);
            report.breakdown = invariant;

            if(report.residualConverged || invariant) return report;

            start = ritzVector;   // thick restart on the current best Ritz vector
        }

        return report;   // budget exhausted: best pair so far, residualConverged == false
    }

    // -----------------------------------------------------------------------------------
    // Steihaug-Toint truncated CG
    // -----------------------------------------------------------------------------------

    enum class SteihaugTermination
    {
        ZeroGradient,
        ResidualTolerance,
        NegativeCurvature,
        TrustRegionBoundary,
        PreconditionerBreakdown,
        MaxIterations
    };

    inline const char * toString(const SteihaugTermination termination)
    {
        switch(termination)
        {
            case SteihaugTermination::ZeroGradient:            return "zero-gradient";
            case SteihaugTermination::ResidualTolerance:       return "residual-tolerance";
            case SteihaugTermination::NegativeCurvature:       return "negative-curvature";
            case SteihaugTermination::TrustRegionBoundary:     return "trust-region-boundary";
            case SteihaugTermination::PreconditionerBreakdown: return "preconditioner-breakdown";
            case SteihaugTermination::MaxIterations:           return "max-iterations";
        }
        return "unknown";
    }

    struct SteihaugOptions
    {
        int maxIterations = 250;
        Real forcingFactor = 0.1;        //!< ||r|| <= forcing * ||g|| terminates CG
        bool superlinearForcing = true;  //!< forcing = min(forcingFactor, sqrt(||g||))
        Real absoluteTolerance = 1e-14;  //!< floor on the CG residual tolerance
    };

    struct SteihaugStepReport
    {
        Eigen::VectorXd step;                  //!< s, in the admissible subspace, ||s||_M <= radius
        Real predictedReduction = 0.0;         //!< -(g.s + 0.5 s.Hs) with H at the BASE state
        Real stepMetricNorm = 0.0;             //!< ||s||_M
        Real trustRadius = 0.0;
        Real initialGradientNorm = 0.0;
        Real residualNorm = 0.0;               //!< final ||-g - H s||
        Real smallestCurvature = std::numeric_limits<Real>::infinity();  //!< min p.Hp / p.Mp
        bool negativeCurvatureDetected = false;
        bool boundaryReached = false;
        int iterations = 0;
        int operatorEvaluations = 0;
        SteihaugTermination termination = SteihaugTermination::MaxIterations;
    };

    /*! Steihaug-Toint truncated CG for  min_s  g.s + 0.5 s.Hs   s.t.  ||s||_M <= trustRadius,
     *  preconditioned by the very same diagonal metric M (so the preconditioned CG iterates
     *  are monotone in the trust-region norm and the boundary test is the natural one).
     *
     *  \param gradient        g at the base state, already projected (projected again here)
     *  \param applyHessian    v -> H v at the BASE state (projection applied here)
     *  \param project         in-place projection P
     *  \param metricDiagonal  strictly positive diagonal of M
     *
     *  H s is accumulated from the CG curvature products, so predictedReduction is exact for
     *  the base-state model and costs no extra Hessian-vector product -- and in particular is
     *  never contaminated by a Hessian evaluated at the trial state. */
    template<typename tOperator, typename tProjection>
    SteihaugStepReport steihaugTruncatedCG(const Eigen::VectorXd & gradient,
                                           tOperator && applyHessian,
                                           tProjection && project,
                                           const Eigen::VectorXd & metricDiagonal,
                                           const Real trustRadius,
                                           const SteihaugOptions & options = SteihaugOptions())
    {
        const int nDofs = static_cast<int>(gradient.size());
        if(metricDiagonal.size() != nDofs)
            throw std::invalid_argument("ShellEquilibrium::steihaugTruncatedCG: metric length mismatch");
        if(!(trustRadius > 0.0))
            throw std::invalid_argument("ShellEquilibrium::steihaugTruncatedCG: trust radius must be positive");
        if(!(metricDiagonal.minCoeff() > 0.0))
            throw std::invalid_argument("ShellEquilibrium::steihaugTruncatedCG: metric must be strictly positive");

        SteihaugStepReport report;
        report.trustRadius = trustRadius;
        report.step = Eigen::VectorXd::Zero(nDofs);

        Eigen::VectorXd g = gradient;
        project(g);
        const Real gradientNorm = g.norm();
        report.initialGradientNorm = gradientNorm;
        report.residualNorm = gradientNorm;
        if(!(gradientNorm > 0.0))
        {
            report.termination = SteihaugTermination::ZeroGradient;
            return report;
        }

        Eigen::VectorXd s = Eigen::VectorXd::Zero(nDofs);
        Eigen::VectorXd hessianStep = Eigen::VectorXd::Zero(nDofs);   // H s, accumulated exactly
        Eigen::VectorXd residual = -g;
        Eigen::VectorXd preconditioned = residual.cwiseQuotient(metricDiagonal);
        project(preconditioned);
        Eigen::VectorXd direction = preconditioned;
        Eigen::VectorXd w(nDofs);

        Real residualDotPreconditioned = residual.dot(preconditioned);
        if(!(residualDotPreconditioned > 0.0))
        {
            report.termination = SteihaugTermination::PreconditionerBreakdown;
            return report;
        }

        const Real forcing = options.superlinearForcing
                           ? std::min(options.forcingFactor, std::sqrt(gradientNorm))
                           : options.forcingFactor;
        const Real cgTolerance = std::max(options.absoluteTolerance, forcing * gradientNorm);
        const Real radiusSquared = trustRadius * trustRadius;

        Real stepMetricSquared = 0.0;   // s.Ms, maintained incrementally

        for(int iteration = 0; iteration < options.maxIterations; ++iteration)
        {
            w = applyHessian(direction);
            project(w);
            ++report.operatorEvaluations;
            report.iterations = iteration + 1;

            const Real curvature = direction.dot(w);
            const Real directionMetric = direction.cwiseProduct(metricDiagonal).dot(direction);
            const Real stepDirectionMetric = s.cwiseProduct(metricDiagonal).dot(direction);
            if(directionMetric > 0.0)
                report.smallestCurvature = std::min(report.smallestCurvature, curvature / directionMetric);

            // tau > 0 with ||s + tau p||_M = trustRadius
            const auto boundaryStep = [&]() -> Real
            {
                if(!(directionMetric > 0.0)) return 0.0;
                const Real discriminant = std::max(0.0, stepDirectionMetric * stepDirectionMetric
                                                        + directionMetric * (radiusSquared - stepMetricSquared));
                return (-stepDirectionMetric + std::sqrt(discriminant)) / directionMetric;
            };

            if(!(curvature > 0.0))
            {
                const Real tau = boundaryStep();
                s.noalias() += tau * direction;
                hessianStep.noalias() += tau * w;
                stepMetricSquared += 2.0 * tau * stepDirectionMetric + tau * tau * directionMetric;
                report.negativeCurvatureDetected = true;
                report.boundaryReached = true;
                report.termination = SteihaugTermination::NegativeCurvature;
                break;
            }

            const Real alpha = residualDotPreconditioned / curvature;
            const Real trialMetricSquared = stepMetricSquared + 2.0 * alpha * stepDirectionMetric
                                          + alpha * alpha * directionMetric;
            if(trialMetricSquared >= radiusSquared)
            {
                const Real tau = boundaryStep();
                s.noalias() += tau * direction;
                hessianStep.noalias() += tau * w;
                stepMetricSquared += 2.0 * tau * stepDirectionMetric + tau * tau * directionMetric;
                report.boundaryReached = true;
                report.termination = SteihaugTermination::TrustRegionBoundary;
                break;
            }

            s.noalias() += alpha * direction;
            hessianStep.noalias() += alpha * w;
            stepMetricSquared = trialMetricSquared;
            residual.noalias() -= alpha * w;
            report.residualNorm = residual.norm();

            if(report.residualNorm <= cgTolerance)
            {
                report.termination = SteihaugTermination::ResidualTolerance;
                break;
            }

            preconditioned = residual.cwiseQuotient(metricDiagonal);
            project(preconditioned);
            const Real next = residual.dot(preconditioned);
            if(!(next > 0.0))
            {
                report.termination = SteihaugTermination::PreconditionerBreakdown;
                break;
            }
            direction = preconditioned + (next / residualDotPreconditioned) * direction;
            residualDotPreconditioned = next;
        }

        report.step = s;
        report.stepMetricNorm = std::sqrt(std::max(0.0, stepMetricSquared));
        report.predictedReduction = -(g.dot(s) + 0.5 * s.dot(hessianStep));
        return report;
    }

    // -----------------------------------------------------------------------------------
    // Trust-region Newton-CG driver
    // -----------------------------------------------------------------------------------

    enum class TrustRegionStatus
    {
        GradientTolerance,
        TrustRegionCollapsed,
        MaxIterations,
        NonFiniteState
    };

    inline const char * toString(const TrustRegionStatus status)
    {
        switch(status)
        {
            case TrustRegionStatus::GradientTolerance:    return "gradient-tolerance";
            case TrustRegionStatus::TrustRegionCollapsed: return "trust-region-collapsed";
            case TrustRegionStatus::MaxIterations:        return "max-iterations";
            case TrustRegionStatus::NonFiniteState:       return "non-finite-state";
        }
        return "unknown";
    }

    struct TrustRegionNewtonOptions
    {
        Real gradientTolerance = 1e-8;      //!< on ||P grad E||_2
        int maxIterations = 100;
        Real initialTrustRadius = 1e-2;     //!< in the metric norm ||.||_M
        Real maxTrustRadius = 1.0;
        Real minTrustRadius = 1e-12;
        Real acceptanceRatio = 1e-4;        //!< accept if actual/predicted >= this
        Real shrinkThreshold = 0.25;
        Real expandThreshold = 0.75;
        Real shrinkFactor = 0.25;
        Real expandFactor = 2.0;
        SteihaugOptions cg = SteihaugOptions();

        // only used by the mesh entry points, to build the default block metric
        Real vertexLengthScale = 1.0;       //!< characteristic vertex displacement (length)
        Real directorAngleScale = 1.0;      //!< characteristic director change (angle)
    };

    struct TrustRegionNewtonReport
    {
        bool accepted = false;                 //!< finite state that met the gradient tolerance
        TrustRegionStatus status = TrustRegionStatus::MaxIterations;
        Real initialEnergy = 0.0;
        Real energy = 0.0;                     //!< energy of the returned (base) state
        Real initialGradientNorm = std::numeric_limits<Real>::infinity();
        Real gradientNorm = std::numeric_limits<Real>::infinity();   //!< ||P grad E|| at the returned state
        Real gradientNormMetric = std::numeric_limits<Real>::infinity(); //!< sqrt(g.M^{-1}g)
        Real trustRadius = 0.0;
        Real lastStepMetricNorm = 0.0;
        int iterations = 0;
        int acceptedSteps = 0;
        int rejectedSteps = 0;
        int energyEvaluations = 0;
        int gradientEvaluations = 0;
        int hessianVectorProducts = 0;
        int cgIterations = 0;
        int negativeCurvatureSteps = 0;        //!< outer iterations whose CG hit negative curvature
        Real smallestCurvature = std::numeric_limits<Real>::infinity(); //!< min p.Hp / p.Mp seen
        int rigidModes = 0;                    //!< set by the mesh entry points
    };

    /*! Trust-region Newton-CG on a stateful problem. The problem must provide
     *      int              numberOfDofs() const
     *      Eigen::VectorXd  state() const
     *      void             setState(const Eigen::VectorXd &)
     *      Real             energy()                                    // at the current state
     *      Eigen::VectorXd  gradient()                                  // at the current state
     *      Eigen::VectorXd  hessianVectorProduct(const Eigen::VectorXd &) // at the current state
     *
     *  A rejected trial step restores the base state through setState (bit-identical vector),
     *  so the problem is always left at the last accepted state -- and, because the base state
     *  is unchanged by a rejection, its gradient is reused instead of recomputed. */
    template<typename tProblem, typename tProjection>
    TrustRegionNewtonReport trustRegionNewtonCG(tProblem & problem,
                                                tProjection && project,
                                                const Eigen::VectorXd & metricDiagonal,
                                                const TrustRegionNewtonOptions & options = TrustRegionNewtonOptions())
    {
        const int nDofs = problem.numberOfDofs();
        if(metricDiagonal.size() != nDofs)
            throw std::invalid_argument("ShellEquilibrium::trustRegionNewtonCG: metric length mismatch");
        if(!(options.initialTrustRadius > 0.0) || !(options.maxTrustRadius >= options.initialTrustRadius))
            throw std::invalid_argument("ShellEquilibrium::trustRegionNewtonCG: inconsistent trust radii");
        if(!(options.gradientTolerance > 0.0))
            throw std::invalid_argument("ShellEquilibrium::trustRegionNewtonCG: gradient tolerance must be positive");

        TrustRegionNewtonReport report;

        Eigen::VectorXd base = problem.state();
        Real energy = problem.energy();
        ++report.energyEvaluations;
        report.initialEnergy = energy;
        report.energy = energy;

        Real radius = options.initialTrustRadius;
        report.trustRadius = radius;

        Eigen::VectorXd gradient;
        bool gradientIsStale = true;

        const auto applyHessian = [&](const Eigen::VectorXd & v)
        {
            ++report.hessianVectorProducts;
            return problem.hessianVectorProduct(v);
        };

        for(int iteration = 0; iteration < options.maxIterations; ++iteration)
        {
            report.iterations = iteration + 1;

            if(gradientIsStale)
            {
                // the base state changed (or this is the first pass): let configuration-
                // dependent projections rebuild before the gradient is projected
                detail::refreshProjection(project);
                gradient = problem.gradient();
                project(gradient);
                ++report.gradientEvaluations;
                gradientIsStale = false;
            }

            report.gradientNorm = gradient.norm();
            report.gradientNormMetric = std::sqrt(std::max(0.0, gradient.cwiseQuotient(metricDiagonal).dot(gradient)));
            if(iteration == 0) report.initialGradientNorm = report.gradientNorm;

            if(!std::isfinite(report.gradientNorm) || !std::isfinite(energy))
            {
                report.status = TrustRegionStatus::NonFiniteState;
                break;
            }
            if(report.gradientNorm <= options.gradientTolerance)
            {
                report.status = TrustRegionStatus::GradientTolerance;
                break;
            }

            const SteihaugStepReport step = steihaugTruncatedCG(gradient, applyHessian, project,
                                                                metricDiagonal, radius, options.cg);
            report.cgIterations += step.iterations;
            report.lastStepMetricNorm = step.stepMetricNorm;
            if(step.negativeCurvatureDetected) ++report.negativeCurvatureSteps;
            report.smallestCurvature = std::min(report.smallestCurvature, step.smallestCurvature);

            if(!(step.predictedReduction > 0.0) || !(step.stepMetricNorm > 0.0))
            {
                // degenerate model step: nothing to test, tighten the region and retry
                ++report.rejectedSteps;
                radius *= options.shrinkFactor;
                report.trustRadius = radius;
                if(radius < options.minTrustRadius)
                {
                    report.status = TrustRegionStatus::TrustRegionCollapsed;
                    break;
                }
                continue;
            }

            problem.setState(base + step.step);
            const Real trialEnergy = problem.energy();
            ++report.energyEvaluations;

            const Real actualReduction = energy - trialEnergy;
            const Real ratio = actualReduction / step.predictedReduction;
            const bool acceptStep = std::isfinite(trialEnergy) && ratio >= options.acceptanceRatio;

            if(acceptStep)
            {
                base += step.step;
                energy = trialEnergy;
                report.energy = energy;
                ++report.acceptedSteps;
                gradientIsStale = true;
            }
            else
            {
                problem.setState(base);   // full rollback of the trial state
                ++report.rejectedSteps;
            }

            // trust-region update
            if(!std::isfinite(ratio) || ratio < options.shrinkThreshold)
                radius = options.shrinkFactor * step.stepMetricNorm;
            else if(ratio > options.expandThreshold && step.boundaryReached)
                radius = std::min(options.expandFactor * radius, options.maxTrustRadius);
            report.trustRadius = radius;

            if(radius < options.minTrustRadius)
            {
                report.status = TrustRegionStatus::TrustRegionCollapsed;
                break;
            }
        }

        // the reported residual always belongs to the returned state
        if(gradientIsStale && report.status != TrustRegionStatus::NonFiniteState)
        {
            detail::refreshProjection(project);
            gradient = problem.gradient();
            project(gradient);
            ++report.gradientEvaluations;
            report.gradientNorm = gradient.norm();
            report.gradientNormMetric = std::sqrt(std::max(0.0, gradient.cwiseQuotient(metricDiagonal).dot(gradient)));
            if(std::isfinite(report.gradientNorm) && report.gradientNorm <= options.gradientTolerance)
                report.status = TrustRegionStatus::GradientTolerance;
        }

        report.accepted = (report.status == TrustRegionStatus::GradientTolerance)
                       && std::isfinite(report.gradientNorm)
                       && report.gradientNorm <= options.gradientTolerance
                       && std::isfinite(report.energy);
        return report;
    }

    // -----------------------------------------------------------------------------------
    // Mesh binding
    // -----------------------------------------------------------------------------------

    /*! Adapts a mesh + energy operator + exact Hessian operator to the stateful problem
     *  interface of trustRegionNewtonCG.
     *
     *  tEnergyOperator      : compute(const tMesh &) -> Real          (EnergyOperator / EnergyOperatorList)
     *  tHessianOperator     : gradient(tMesh &) -> VectorXd,
     *                         hessianVectorProduct(tMesh &, const VectorXd &) -> VectorXd
     *                         (TinyADHessian_Bilayer)
     *
     *  The energy and the gradient must belong to the same functional, otherwise the
     *  actual/predicted reduction ratio is meaningless. */
    template<typename tMesh, typename tEnergyOperator, typename tHessianOperator>
    class MeshEquilibriumProblem
    {
        tMesh & mesh;
        const tEnergyOperator & energyOperator;
        const tHessianOperator & hessianOperator;
        const int nDofs_;

    public:

        MeshEquilibriumProblem(tMesh & mesh_in, const tEnergyOperator & energyOperator_in,
                               const tHessianOperator & hessianOperator_in)
        : mesh(mesh_in), energyOperator(energyOperator_in), hessianOperator(hessianOperator_in),
          nDofs_(ShellEquilibrium::numberOfDofs(mesh_in.getNumberOfVertices(), mesh_in.getNumberOfEdges()))
        {}

        int numberOfDofs() const { return nDofs_; }

        Eigen::VectorXd state() const
        {
            return Eigen::Map<const Eigen::VectorXd>(mesh.getDataPointer(), nDofs_);
        }

        void setState(const Eigen::VectorXd & x)
        {
            if(x.size() != nDofs_)
                throw std::invalid_argument("ShellEquilibrium::MeshEquilibriumProblem: state length mismatch");
            Eigen::Map<Eigen::VectorXd>(mesh.getDataPointer(), nDofs_) = x;
            mesh.updateDeformedConfiguration();
        }

        Real energy() { return energyOperator.compute(mesh); }

        Eigen::VectorXd gradient() { return hessianOperator.gradient(mesh); }

        Eigen::VectorXd hessianVectorProduct(const Eigen::VectorXd & v)
        {
            return hessianOperator.hessianVectorProduct(mesh, v);
        }
    };

    /*! Trust-region Newton-CG equilibrium correction of a shell, with an explicit projector and
     *  metric. The mesh is left at the last accepted state. */
    template<typename tMesh, typename tEnergyOperator, typename tHessianOperator>
    TrustRegionNewtonReport solveShellEquilibrium(tMesh & mesh,
                                                  const tEnergyOperator & energyOperator,
                                                  const tHessianOperator & hessianOperator,
                                                  const RigidModeProjector & projector,
                                                  const Eigen::VectorXd & metricDiagonal,
                                                  const TrustRegionNewtonOptions & options = TrustRegionNewtonOptions())
    {
        MeshEquilibriumProblem<tMesh, tEnergyOperator, tHessianOperator> problem(mesh, energyOperator, hessianOperator);
        TrustRegionNewtonReport report = trustRegionNewtonCG(problem, projector, metricDiagonal, options);
        report.rigidModes = projector.numberOfRigidModes();
        return report;
    }

    /*! Same, but with the boundary-condition-aware rigid basis rebuilt from the mesh whenever
     *  the base state changes, and the trust-region metric built from
     *  options.vertexLengthScale / options.directorAngleScale. */
    template<typename tMesh, typename tEnergyOperator, typename tHessianOperator>
    TrustRegionNewtonReport solveShellEquilibrium(tMesh & mesh,
                                                  const tEnergyOperator & energyOperator,
                                                  const tHessianOperator & hessianOperator,
                                                  const TrustRegionNewtonOptions & options = TrustRegionNewtonOptions())
    {
        MeshEquilibriumProblem<tMesh, tEnergyOperator, tHessianOperator> problem(mesh, energyOperator, hessianOperator);
        MeshRigidModeProjection<tMesh> projection(mesh);
        const Eigen::VectorXd metric = blockDiagonalMetric(mesh.getNumberOfVertices(), mesh.getNumberOfEdges(),
                                                           options.vertexLengthScale, options.directorAngleScale);
        TrustRegionNewtonReport report = trustRegionNewtonCG(problem, projection, metric, options);
        report.rigidModes = projection.projector().numberOfRigidModes();
        return report;
    }

    /*! Smallest projected eigenpair of the exact shell Hessian at the current state.
     *
     *  With an empty metricDiagonal the operator is P H P and the eigenvalue is the plain
     *  smallest Hessian eigenvalue on the admissible subspace. With a positive metric the
     *  operator is the congruent D^{-1} H D^{-1} (D = sqrt(M)), which has the same inertia --
     *  the sign of the returned eigenvalue is unchanged, its magnitude is metric-dependent --
     *  but O(1) conditioning, so the Lanczos residual actually reaches the tolerance on
     *  vertex/director-mixed problems.
     *
     *  This is a diagnostic of the current state only: no branch identification is performed
     *  or implied. */
    template<typename tMesh, typename tHessianOperator>
    RitzPairReport smallestProjectedRitzPairForMesh(tMesh & mesh,
                                                    const tHessianOperator & hessianOperator,
                                                    const RigidModeProjector & projector,
                                                    const Eigen::VectorXd & metricDiagonal = Eigen::VectorXd(),
                                                    const SmallestRitzPairOptions & options = SmallestRitzPairOptions())
    {
        const int nDofs = numberOfDofs(mesh.getNumberOfVertices(), mesh.getNumberOfEdges());
        if(projector.nDofs() != nDofs)
            throw std::invalid_argument("ShellEquilibrium::smallestProjectedRitzPairForMesh: projector length mismatch");

        if(metricDiagonal.size() == 0)
        {
            return smallestProjectedRitzPair(nDofs,
                                             [&](const Eigen::VectorXd & v) { return hessianOperator.hessianVectorProduct(mesh, v); },
                                             projector, options);
        }

        if(metricDiagonal.size() != nDofs)
            throw std::invalid_argument("ShellEquilibrium::smallestProjectedRitzPairForMesh: metric length mismatch");
        if(!(metricDiagonal.minCoeff() > 0.0))
            throw std::invalid_argument("ShellEquilibrium::smallestProjectedRitzPairForMesh: metric must be strictly positive");

        const Eigen::VectorXd scaling = metricDiagonal.cwiseSqrt();
        const RigidModeProjector scaledProjector = projector.transformedByDiagonalScaling(scaling);
        return smallestProjectedRitzPair(nDofs,
                                         [&](const Eigen::VectorXd & y)
                                         {
                                             const Eigen::VectorXd x = y.cwiseQuotient(scaling);
                                             return Eigen::VectorXd(hessianOperator.hessianVectorProduct(mesh, x).cwiseQuotient(scaling));
                                         },
                                         scaledProjector, options);
    }
}

#endif /* ShellEquilibriumSolver_hpp */

#include "gtest/gtest.h"

#include "common.hpp"
#include "ShellEquilibriumSolver.hpp"

#include "Geometry.hpp"
#include "Mesh.hpp"
#include "MaterialProperties.hpp"
#include "CombinedOperator_Parametric.hpp"
#include "EnergyOperatorList.hpp"
#include "TinyADHessian_Bilayer.hpp"

using namespace ShellEquilibrium;

namespace
{
    /*! deterministic, portable pseudo-random stream (splitmix64 -> [-1,1)) so that every test
     *  vector below is bit-reproducible */
    struct DeterministicStream
    {
        std::uint64_t state;
        explicit DeterministicStream(const std::uint64_t seed) : state(seed) {}
        Real operator()()
        {
            state += 0x9E3779B97F4A7C15ull;
            std::uint64_t z = state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
            z ^= (z >> 31);
            return 2.0 * static_cast<Real>(z >> 11) * (1.0 / 9007199254740992.0) - 1.0;
        }
        Eigen::VectorXd vector(const int n)
        {
            Eigen::VectorXd v(n);
            for(int i = 0; i < n; ++i) v(i) = (*this)();
            return v;
        }
    };

    /*! a vertex cloud in general position (no three vertices collinear) */
    Eigen::MatrixXd vertexCloud(const int nVertices)
    {
        Eigen::MatrixXd vertices(nVertices, 3);
        for(int i = 0; i < nVertices; ++i)
        {
            vertices(i, 0) = 0.31 * i - 0.7;
            vertices(i, 1) = 0.17 * i * i - 0.4 * i;
            vertices(i, 2) = 0.05 * i - 0.9 + 0.01 * i * i;
        }
        return vertices;
    }

    /*! the rigid variation  t + w x (v_i - c)  in the global DOF layout */
    Eigen::VectorXd rigidMode(const Eigen::MatrixXd & vertices, const int nEdges,
                              const Eigen::Vector3d & translation, const Eigen::Vector3d & rotation,
                              const Eigen::Vector3d & center)
    {
        const int nVertices = static_cast<int>(vertices.rows());
        Eigen::VectorXd mode = Eigen::VectorXd::Zero(numberOfDofs(nVertices, nEdges));
        for(int i = 0; i < nVertices; ++i)
        {
            const Eigen::Vector3d displacement = translation + rotation.cross(Eigen::Vector3d(vertices.row(i).transpose()) - center);
            for(int d = 0; d < 3; ++d) mode(d * nVertices + i) = displacement(d);
        }
        return mode;
    }

    /*! symmetric tridiagonal 1D Laplacian minus a shift; eigenvalues 2-2cos(k pi/(n+1)) - shift */
    Eigen::MatrixXd shiftedLaplacian(const int n, const Real shift)
    {
        Eigen::MatrixXd A = Eigen::MatrixXd::Zero(n, n);
        for(int i = 0; i < n; ++i)
        {
            A(i, i) = 2.0 - shift;
            if(i + 1 < n) { A(i, i + 1) = -1.0; A(i + 1, i) = -1.0; }
        }
        return A;
    }

    // ---------------------------------------------------------------------------------------
    // synthetic stateful problems for the trust-region driver
    // ---------------------------------------------------------------------------------------

    /*! separable  f(x) = sum_i (0.5 x_i^2 - 0.25 x_i^4): at x_i = 1/2 the Newton step is -3/2
     *  per coordinate, which raises the energy from 3/32 to 1/4 per coordinate -- a rejected
     *  trial with certainty, independent of round-off. */
    class QuarticProblem
    {
        Eigen::VectorXd x;
    public:
        int setStateCalls = 0;
        std::vector<Eigen::VectorXd> visited;

        explicit QuarticProblem(const Eigen::VectorXd & x0) : x(x0) {}
        int numberOfDofs() const { return static_cast<int>(x.size()); }
        Eigen::VectorXd state() const { return x; }
        void setState(const Eigen::VectorXd & v) { x = v; ++setStateCalls; visited.push_back(v); }
        Real energy() { return (0.5 * x.array().square() - 0.25 * x.array().pow(4)).sum(); }
        Eigen::VectorXd gradient() { return x - x.array().cube().matrix(); }
        Eigen::VectorXd hessianVectorProduct(const Eigen::VectorXd & v)
        { return ((1.0 - 3.0 * x.array().square()) * v.array()).matrix(); }
    };

    class RosenbrockProblem
    {
        Eigen::VectorXd x;
    public:
        explicit RosenbrockProblem(const Eigen::VectorXd & x0) : x(x0) {}
        int numberOfDofs() const { return 2; }
        Eigen::VectorXd state() const { return x; }
        void setState(const Eigen::VectorXd & v) { x = v; }
        Real energy()
        {
            const Real a = 1.0 - x(0);
            const Real b = x(1) - x(0) * x(0);
            return a * a + 100.0 * b * b;
        }
        Eigen::VectorXd gradient()
        {
            Eigen::VectorXd g(2);
            const Real b = x(1) - x(0) * x(0);
            g(0) = -2.0 * (1.0 - x(0)) - 400.0 * x(0) * b;
            g(1) = 200.0 * b;
            return g;
        }
        Eigen::VectorXd hessianVectorProduct(const Eigen::VectorXd & v)
        {
            const Real b = x(1) - x(0) * x(0);
            Eigen::Matrix2d H;
            H(0, 0) = 2.0 - 400.0 * b + 800.0 * x(0) * x(0);
            H(0, 1) = -400.0 * x(0);
            H(1, 0) = H(0, 1);
            H(1, 1) = 200.0;
            return H * v;
        }
    };

    // ---------------------------------------------------------------------------------------
    // stub with the libshell mesh API: a rigid spring truss (translation/rotation invariant,
    // like the shell energy) plus decoupled quadratic edge directors
    // ---------------------------------------------------------------------------------------

    struct StubBoundaryConditions
    {
        Eigen::MatrixXb vertices_bc;
        Eigen::VectorXb edges_bc;
        const Eigen::MatrixXb & getVertexBoundaryConditions() const { return vertices_bc; }
        const Eigen::VectorXb & getEdgeBoundaryConditions() const { return edges_bc; }
    };

    struct StubConfiguration
    {
        Eigen::Map<Eigen::MatrixXd> vertexmap;
        StubConfiguration(Real * data, const int nVertices) : vertexmap(data, nVertices, 3) {}
        const Eigen::Map<Eigen::MatrixXd> & getVertices() const { return vertexmap; }
    };

    class TrussMesh
    {
    public:
        int nVertices, nEdges;
        std::vector<Real> data;
        StubBoundaryConditions bc;
        std::vector<std::pair<int, int>> springs;
        std::vector<Real> restLength;
        Real stiffness = 3.0;
        Real directorStiffness = 0.7;

        TrussMesh(const int nVertices_in, const int nEdges_in)
        : nVertices(nVertices_in), nEdges(nEdges_in), data(3 * nVertices_in + nEdges_in, 0.0)
        {
            bc.vertices_bc = Eigen::MatrixXb::Constant(nVertices, 3, false);
            bc.edges_bc = Eigen::VectorXb::Constant(nEdges, false);
            DeterministicStream stream(4242ull);
            for(int i = 0; i < nVertices; ++i)
                for(int d = 0; d < 3; ++d)
                    data[d * nVertices + i] = stream();
            for(int i = 0; i < nVertices; ++i)
                for(int j = i + 1; j < nVertices; ++j)
                    springs.emplace_back(i, j);   // complete graph: infinitesimally rigid
            restLength.resize(springs.size());
            for(std::size_t s = 0; s < springs.size(); ++s) restLength[s] = length(static_cast<int>(s));
        }

        Eigen::Vector3d vertex(const int i) const
        { return Eigen::Vector3d(data[i], data[nVertices + i], data[2 * nVertices + i]); }
        Real length(const int s) const
        { return (vertex(springs[s].first) - vertex(springs[s].second)).norm(); }

        int getNumberOfVertices() const { return nVertices; }
        int getNumberOfEdges() const { return nEdges; }
        const StubBoundaryConditions & getBoundaryConditions() const { return bc; }
        StubConfiguration getCurrentConfiguration() const
        { return StubConfiguration(const_cast<Real *>(data.data()), nVertices); }
        Real * getDataPointer() { return data.data(); }
        void updateDeformedConfiguration() {}
    };

    struct TrussEnergy
    {
        Real compute(const TrussMesh & mesh) const
        {
            Real energy = 0.0;
            for(std::size_t s = 0; s < mesh.springs.size(); ++s)
            {
                const Real stretch = mesh.length(static_cast<int>(s)) - mesh.restLength[s];
                energy += 0.5 * mesh.stiffness * stretch * stretch;
            }
            for(int e = 0; e < mesh.nEdges; ++e)
            {
                const Real phi = mesh.data[3 * mesh.nVertices + e];
                energy += 0.5 * mesh.directorStiffness * phi * phi;
            }
            return energy;
        }
    };

    struct TrussHessian
    {
        Eigen::VectorXd gradient(TrussMesh & mesh) const
        {
            const int nV = mesh.nVertices;
            Eigen::VectorXd g = Eigen::VectorXd::Zero(numberOfDofs(nV, mesh.nEdges));
            for(std::size_t s = 0; s < mesh.springs.size(); ++s)
            {
                const int i = mesh.springs[s].first;
                const int j = mesh.springs[s].second;
                const Eigen::Vector3d d = mesh.vertex(i) - mesh.vertex(j);
                const Real r = d.norm();
                const Eigen::Vector3d f = mesh.stiffness * (r - mesh.restLength[s]) * d / r;
                for(int k = 0; k < 3; ++k) { g(k * nV + i) += f(k); g(k * nV + j) -= f(k); }
            }
            for(int e = 0; e < mesh.nEdges; ++e)
                g(3 * nV + e) = mesh.directorStiffness * mesh.data[3 * nV + e];
            return g;
        }

        Eigen::VectorXd hessianVectorProduct(TrussMesh & mesh, const Eigen::VectorXd & v) const
        {
            const int nV = mesh.nVertices;
            Eigen::VectorXd out = Eigen::VectorXd::Zero(numberOfDofs(nV, mesh.nEdges));
            for(std::size_t s = 0; s < mesh.springs.size(); ++s)
            {
                const int i = mesh.springs[s].first;
                const int j = mesh.springs[s].second;
                const Eigen::Vector3d d = mesh.vertex(i) - mesh.vertex(j);
                const Real r = d.norm();
                const Eigen::Vector3d u = d / r;
                Eigen::Vector3d dv;
                for(int k = 0; k < 3; ++k) dv(k) = v(k * nV + i) - v(k * nV + j);
                const Real axial = u.dot(dv);
                const Eigen::Vector3d hv = mesh.stiffness * (u * axial + (1.0 - mesh.restLength[s] / r) * (dv - u * axial));
                for(int k = 0; k < 3; ++k) { out(k * nV + i) += hv(k); out(k * nV + j) -= hv(k); }
            }
            for(int e = 0; e < mesh.nEdges; ++e)
                out(3 * nV + e) = mesh.directorStiffness * v(3 * nV + e);
            return out;
        }
    };
}

// ===========================================================================================
// rigid-mode projector
// ===========================================================================================

TEST(ShellEquilibriumProjector, FreeShellRemovesEveryRigidModeAndIsIdempotent)
{
    const int nVertices = 7;
    const int nEdges = 5;
    const Eigen::MatrixXd vertices = vertexCloud(nVertices);
    const RigidModeProjector projector(vertices, nEdges, std::vector<char>());

    ASSERT_EQ(projector.numberOfRigidModes(), 6);
    ASSERT_EQ(projector.nDofs(), numberOfDofs(nVertices, nEdges));
    ASSERT_EQ(projector.numberOfConstrainedDofs(), 0);

    // every rigid variation -- about an arbitrary centre, not just the centroid -- is removed
    const Eigen::Vector3d center(0.3, -0.2, 0.7);
    for(int k = 0; k < 3; ++k)
    {
        Eigen::Vector3d axis = Eigen::Vector3d::Zero();
        axis(k) = 1.0;
        const Eigen::VectorXd translation = rigidMode(vertices, nEdges, axis, Eigen::Vector3d::Zero(), center);
        const Eigen::VectorXd rotation = rigidMode(vertices, nEdges, Eigen::Vector3d::Zero(), axis, center);
        EXPECT_LT(projector.apply(translation).norm(), 1e-12 * translation.norm());
        EXPECT_LT(projector.apply(rotation).norm(), 1e-12 * rotation.norm());
    }

    DeterministicStream stream(17ull);
    const Eigen::VectorXd v = stream.vector(projector.nDofs());
    const Eigen::VectorXd projected = projector.apply(v);
    const Eigen::VectorXd twice = projector.apply(projected);

    EXPECT_GT(projected.norm(), 0.1);                                        // nothing degenerate
    EXPECT_LT((twice - projected).norm(), 1e-13 * projected.norm());         // idempotence
    EXPECT_LT((projector.basis().transpose() * projected).norm(), 1e-12);    // orthogonal to the rigid space

    // rigid motions never touch the edge directors, so the director block passes through exactly
    Eigen::VectorXd directorOnly = Eigen::VectorXd::Zero(projector.nDofs());
    directorOnly.tail(nEdges).setConstant(1.5);
    EXPECT_EQ((projector.apply(directorOnly) - directorOnly).norm(), 0.0);
}

TEST(ShellEquilibriumProjector, BoundaryConditionsRestrictTheRigidSubspace)
{
    const int nVertices = 7;
    const int nEdges = 5;
    const Eigen::MatrixXd vertices = vertexCloud(nVertices);
    const int nDofs = numberOfDofs(nVertices, nEdges);

    // two fully pinned vertices leave exactly the rotation about the line joining them
    std::vector<char> twoPinned(nDofs, 0);
    for(int d = 0; d < 3; ++d) { twoPinned[d * nVertices + 0] = 1; twoPinned[d * nVertices + 3] = 1; }
    const RigidModeProjector pinnedTwo(vertices, nEdges, twoPinned);
    ASSERT_EQ(pinnedTwo.numberOfRigidModes(), 1);

    Eigen::Vector3d axis = (vertices.row(3) - vertices.row(0)).transpose();
    axis.normalize();
    Eigen::VectorXd axisRotation = rigidMode(vertices, nEdges, Eigen::Vector3d::Zero(), axis,
                                             Eigen::Vector3d(vertices.row(0).transpose()));
    axisRotation.normalize();
    EXPECT_NEAR(std::abs(pinnedTwo.basis().col(0).dot(axisRotation)), 1.0, 1e-12);
    EXPECT_LT(pinnedTwo.apply(axisRotation).norm(), 1e-12);

    // a rigid motion that moves a pinned vertex is NOT a symmetry and must survive projection
    const Eigen::VectorXd translation = rigidMode(vertices, nEdges, Eigen::Vector3d(1, 0, 0),
                                                  Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
    EXPECT_GT(pinnedTwo.apply(translation).norm(), 0.5 * translation.norm());

    // three non-collinear pinned vertices leave nothing
    std::vector<char> threePinned = twoPinned;
    for(int d = 0; d < 3; ++d) threePinned[d * nVertices + 5] = 1;
    const RigidModeProjector pinnedThree(vertices, nEdges, threePinned);
    EXPECT_EQ(pinnedThree.numberOfRigidModes(), 0);

    DeterministicStream stream(23ull);
    const Eigen::VectorXd v = stream.vector(nDofs);
    const Eigen::VectorXd projected = pinnedThree.apply(v);
    for(int i = 0; i < nDofs; ++i)
        EXPECT_DOUBLE_EQ(projected(i), threePinned[i] ? 0.0 : v(i));

    // constraining an edge director removes no rigid mode (directors are rigid-invariant)
    std::vector<char> directorFixed(nDofs, 0);
    directorFixed[3 * nVertices + 2] = 1;
    const RigidModeProjector fixedDirector(vertices, nEdges, directorFixed);
    EXPECT_EQ(fixedDirector.numberOfRigidModes(), 6);
    EXPECT_DOUBLE_EQ(fixedDirector.apply(Eigen::VectorXd::Ones(nDofs))(3 * nVertices + 2), 0.0);
}

// ===========================================================================================
// smallest projected Ritz pair
// ===========================================================================================

TEST(ShellEquilibriumRitzPair, SmallestEigenpairOfASymmetricOperator)
{
    const int n = 60;
    const Real shift = 0.5;
    const Eigen::MatrixXd A = shiftedLaplacian(n, shift);
    const auto op = [&](const Eigen::VectorXd & v) { return Eigen::VectorXd(A * v); };

    const RitzPairReport report = smallestProjectedRitzPair(n, op, NoProjection());
    const Real exact = 2.0 - 2.0 * std::cos(M_PI / (n + 1)) - shift;

    ASSERT_TRUE(report.residualConverged);
    EXPECT_LT(report.eigenvalue, 0.0);                       // the smallest eigenvalue is negative here
    EXPECT_NEAR(report.eigenvalue, exact, 1e-9);
    EXPECT_NEAR(report.eigenvector.norm(), 1.0, 1e-12);
    EXPECT_GT(report.operatorEvaluations, 0);

    // the reported residual is the true residual of the returned pair, recomputed independently
    const Eigen::VectorXd residual = A * report.eigenvector - report.eigenvalue * report.eigenvector;
    EXPECT_NEAR(residual.norm(), report.residualAbsolute, 1e-14 + 1e-6 * report.residualAbsolute);
    EXPECT_LE(report.residualRelative, report.requestedRelativeTolerance);

    // deterministic: identical input, bit-identical output
    const RitzPairReport again = smallestProjectedRitzPair(n, op, NoProjection());
    EXPECT_EQ(again.eigenvalue, report.eigenvalue);
    EXPECT_EQ(again.residualAbsolute, report.residualAbsolute);
    EXPECT_EQ(again.operatorEvaluations, report.operatorEvaluations);
}

TEST(ShellEquilibriumRitzPair, ExhaustedBudgetIsNeverReportedAsConverged)
{
    const int n = 60;
    const Eigen::MatrixXd A = shiftedLaplacian(n, 0.5);
    const auto op = [&](const Eigen::VectorXd & v) { return Eigen::VectorXd(A * v); };

    SmallestRitzPairOptions options;
    options.krylovDimension = 2;
    options.maxRestarts = 0;
    options.absoluteResidualTolerance = 1e-14;
    options.relativeResidualTolerance = 1e-14;

    const RitzPairReport report = smallestProjectedRitzPair(n, op, NoProjection(), options);

    EXPECT_FALSE(report.residualConverged);
    EXPECT_GT(report.residualAbsolute, options.absoluteResidualTolerance);
    EXPECT_GT(report.residualRelative, options.relativeResidualTolerance);

    // the flag is exactly the residual test -- nothing else may set it
    EXPECT_EQ(report.residualConverged,
              (report.residualAbsolute <= report.requestedAbsoluteTolerance)
              || (report.residualRelative <= report.requestedRelativeTolerance));

    // and the returned pair is still an honest Ritz pair with the reported residual
    const Eigen::VectorXd residual = A * report.eigenvector - report.eigenvalue * report.eigenvector;
    EXPECT_NEAR(residual.norm(), report.residualAbsolute, 1e-6 * report.residualAbsolute);
}

TEST(ShellEquilibriumRitzPair, ProjectionKeepsTheSolverOffTheProjectedDirection)
{
    const int n = 40;
    DeterministicStream stream(99ull);
    Eigen::VectorXd z = stream.vector(n);
    z.normalize();

    // K is SPD; B agrees with K on z^perp but carries a strongly negative eigenvalue along z.
    // Only a solver that really projects can report the (positive) smallest eigenvalue.
    const Eigen::MatrixXd K = shiftedLaplacian(n, -0.1);
    const Eigen::MatrixXd deflate = Eigen::MatrixXd::Identity(n, n) - z * z.transpose();
    const Eigen::MatrixXd B = deflate * K * deflate - 10.0 * z * z.transpose();

    const auto op = [&](const Eigen::VectorXd & v) { return Eigen::VectorXd(B * v); };
    const auto project = [&](Eigen::VectorXd & v) { v -= z * (z.dot(v)); };

    const RitzPairReport report = smallestProjectedRitzPair(n, op, project);
    ASSERT_TRUE(report.residualConverged);

    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> dense(deflate * B * deflate);
    EXPECT_NEAR(dense.eigenvalues()(0), 0.0, 1e-12);                    // the projected direction
    EXPECT_NEAR(report.eigenvalue, dense.eigenvalues()(1), 1e-8);       // the smallest admissible one
    EXPECT_GT(report.eigenvalue, 0.0);
    EXPECT_LT(std::abs(z.dot(report.eigenvector)), 1e-12);
}

// ===========================================================================================
// Steihaug truncated CG
// ===========================================================================================

namespace
{
    Eigen::MatrixXd smallSpdMatrix(const int n)
    {
        Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
        for(int i = 0; i < n; ++i)
        {
            H(i, i) = 1.0 + i;
            if(i + 1 < n) { H(i, i + 1) = 0.3; H(i + 1, i) = 0.3; }
        }
        return H;
    }

    Eigen::VectorXd smallGradient(const int n)
    {
        Eigen::VectorXd g(n);
        for(int i = 0; i < n; ++i) g(i) = 0.1 * (i + 1) * ((i % 2) ? -1.0 : 1.0);
        return g;
    }

    SteihaugOptions exactCgOptions()
    {
        SteihaugOptions options;
        options.superlinearForcing = false;
        options.forcingFactor = 0.0;   // drive CG to the absolute floor
        return options;
    }
}

TEST(ShellEquilibriumSteihaug, InteriorSolutionIsTheNewtonStep)
{
    const int n = 6;
    const Eigen::MatrixXd H = smallSpdMatrix(n);
    const Eigen::VectorXd g = smallGradient(n);
    const auto op = [&](const Eigen::VectorXd & v) { return Eigen::VectorXd(H * v); };

    const SteihaugStepReport report = steihaugTruncatedCG(g, op, NoProjection(),
                                                          Eigen::VectorXd::Ones(n), 1e6, exactCgOptions());

    const Eigen::VectorXd newton = H.ldlt().solve(-g);
    EXPECT_LT((report.step - newton).norm(), 1e-10 * newton.norm());
    EXPECT_FALSE(report.boundaryReached);
    EXPECT_FALSE(report.negativeCurvatureDetected);
    EXPECT_EQ(report.termination, SteihaugTermination::ResidualTolerance);
    EXPECT_LE(report.iterations, n);

    // predicted reduction of the base-state model: 0.5 g^T H^{-1} g
    EXPECT_NEAR(report.predictedReduction, 0.5 * g.dot(-newton), 1e-12);
    EXPECT_GT(report.predictedReduction, 0.0);
    EXPECT_GT(report.smallestCurvature, 0.0);
}

TEST(ShellEquilibriumSteihaug, TruncatedStepSitsExactlyOnTheTrustRegionBoundary)
{
    const int n = 6;
    const Real radius = 0.05;
    const Eigen::MatrixXd H = smallSpdMatrix(n);
    const Eigen::VectorXd g = smallGradient(n);
    const auto op = [&](const Eigen::VectorXd & v) { return Eigen::VectorXd(H * v); };

    const SteihaugStepReport report = steihaugTruncatedCG(g, op, NoProjection(),
                                                          Eigen::VectorXd::Ones(n), radius, exactCgOptions());

    EXPECT_TRUE(report.boundaryReached);
    EXPECT_EQ(report.termination, SteihaugTermination::TrustRegionBoundary);
    EXPECT_NEAR(report.stepMetricNorm, radius, 1e-12);
    EXPECT_NEAR(report.step.norm(), radius, 1e-12);   // identity metric
    EXPECT_GT(report.predictedReduction, 0.0);
}

TEST(ShellEquilibriumSteihaug, TheTrustRegionIsMeasuredInTheDofMetric)
{
    // one vertex (3 DOFs) with a millimetre length scale, three directors with a radian scale
    const Real vertexScale = 1e-3;
    const Real radius = 1.0;
    const Eigen::VectorXd metric = blockDiagonalMetric(1, 3, vertexScale, 1.0);
    const Eigen::VectorXd g = Eigen::VectorXd::Ones(6);
    const auto op = [&](const Eigen::VectorXd & v) { return Eigen::VectorXd(v); };   // H = I

    const SteihaugStepReport report = steihaugTruncatedCG(g, op, NoProjection(), metric, radius, exactCgOptions());

    EXPECT_TRUE(report.boundaryReached);
    EXPECT_NEAR(report.stepMetricNorm, radius, 1e-12);
    // the vertex block moves by ~vertexScale^2 of the director block: the metric, not the plain
    // l2 norm, is what limits the step
    EXPECT_LT(report.step.head(3).cwiseAbs().maxCoeff(), 1e-4 * report.step.tail(3).cwiseAbs().maxCoeff());
    EXPECT_GT(report.step.tail(3).cwiseAbs().minCoeff(), 0.1);
}

TEST(ShellEquilibriumSteihaug, NegativeCurvatureIsFollowedToTheBoundary)
{
    // immediate negative curvature: H = diag(-1, 2), g = (1,0), so p0 = -g has p0.Hp0 < 0
    const Real radius = 0.5;
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 2);
    H(0, 0) = -1.0;
    H(1, 1) = 2.0;
    Eigen::VectorXd g(2);
    g << 1.0, 0.0;
    const auto op = [&](const Eigen::VectorXd & v) { return Eigen::VectorXd(H * v); };

    const SteihaugStepReport immediate = steihaugTruncatedCG(g, op, NoProjection(),
                                                             Eigen::VectorXd::Ones(2), radius, exactCgOptions());
    EXPECT_TRUE(immediate.negativeCurvatureDetected);
    EXPECT_TRUE(immediate.boundaryReached);
    EXPECT_EQ(immediate.termination, SteihaugTermination::NegativeCurvature);
    EXPECT_EQ(immediate.iterations, 1);
    EXPECT_NEAR(immediate.stepMetricNorm, radius, 1e-14);
    EXPECT_NEAR(immediate.step(0), -radius, 1e-14);
    EXPECT_LT(immediate.smallestCurvature, 0.0);
    // model reduction  -(g.s + 0.5 s.Hs) = radius + 0.5 radius^2
    EXPECT_NEAR(immediate.predictedReduction, radius + 0.5 * radius * radius, 1e-14);

    // negative curvature discovered only after several CG steps on an indefinite operator
    const int n = 6;
    Eigen::MatrixXd indefinite = smallSpdMatrix(n);
    indefinite(2, 2) = -2.0;
    const Eigen::VectorXd gn = smallGradient(n);
    const auto opn = [&](const Eigen::VectorXd & v) { return Eigen::VectorXd(indefinite * v); };

    const SteihaugStepReport delayed = steihaugTruncatedCG(gn, opn, NoProjection(),
                                                           Eigen::VectorXd::Ones(n), 2.0, exactCgOptions());
    EXPECT_TRUE(delayed.negativeCurvatureDetected);
    EXPECT_TRUE(delayed.boundaryReached);
    EXPECT_GT(delayed.iterations, 1);
    EXPECT_NEAR(delayed.stepMetricNorm, 2.0, 1e-12);
    EXPECT_GT(delayed.predictedReduction, 0.0);
    EXPECT_LT(delayed.smallestCurvature, 0.0);
}

// ===========================================================================================
// trust-region Newton-CG driver
// ===========================================================================================

TEST(ShellEquilibriumTrustRegion, RejectedTrialRestoresTheBaseStateExactly)
{
    const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(3, 0.5);
    QuarticProblem problem(x0);
    const Real baseEnergy = 3.0 * (0.5 * 0.25 - 0.25 * 0.0625);   // 3/32 per coordinate

    TrustRegionNewtonOptions options;
    options.maxIterations = 1;
    options.initialTrustRadius = 4.0;   // large enough for the full (bad) Newton step
    options.maxTrustRadius = 4.0;

    const TrustRegionNewtonReport report = trustRegionNewtonCG(problem, NoProjection(),
                                                               Eigen::VectorXd::Ones(3), options);

    EXPECT_EQ(report.acceptedSteps, 0);
    EXPECT_EQ(report.rejectedSteps, 1);
    EXPECT_FALSE(report.accepted);
    EXPECT_EQ(report.status, TrustRegionStatus::MaxIterations);

    // the trial state was really visited, and then undone bit for bit
    ASSERT_GE(problem.visited.size(), 2u);
    EXPECT_GT((problem.visited.front() - x0).norm(), 1.0);
    EXPECT_EQ((problem.state() - x0).norm(), 0.0);
    EXPECT_EQ((problem.visited.back() - x0).norm(), 0.0);

    // the report describes the state the solver is left in, not the discarded trial
    EXPECT_DOUBLE_EQ(report.energy, baseEnergy);
    EXPECT_DOUBLE_EQ(report.initialEnergy, baseEnergy);
    EXPECT_DOUBLE_EQ(report.energy, problem.energy());
    EXPECT_EQ(report.energyEvaluations, 2);       // base + rejected trial
    EXPECT_EQ(report.gradientEvaluations, 1);     // the base gradient is reused, not recomputed
    EXPECT_LT(report.trustRadius, options.initialTrustRadius);
}

TEST(ShellEquilibriumTrustRegion, DescendsToAStationaryPointAndReportsIt)
{
    const Eigen::VectorXd x0 = Eigen::VectorXd::Constant(3, 0.5);
    QuarticProblem problem(x0);

    TrustRegionNewtonOptions options;
    options.maxIterations = 200;
    options.initialTrustRadius = 0.2;
    options.maxTrustRadius = 1.0;
    options.gradientTolerance = 1e-10;

    const TrustRegionNewtonReport report = trustRegionNewtonCG(problem, NoProjection(),
                                                               Eigen::VectorXd::Ones(3), options);

    EXPECT_TRUE(report.accepted);
    EXPECT_EQ(report.status, TrustRegionStatus::GradientTolerance);
    EXPECT_LE(report.gradientNorm, options.gradientTolerance);
    EXPECT_GE(report.acceptedSteps, 1);
    EXPECT_LT(report.energy, report.initialEnergy);
    EXPECT_LT(problem.state().norm(), 1e-8);
    EXPECT_DOUBLE_EQ(report.energy, problem.energy());
}

TEST(ShellEquilibriumTrustRegion, ConvergesOnANonSeparableProblemWithRejectedSteps)
{
    Eigen::VectorXd x0(2);
    x0 << -1.2, 1.0;
    RosenbrockProblem problem(x0);

    TrustRegionNewtonOptions options;
    options.maxIterations = 500;
    options.initialTrustRadius = 0.5;
    options.maxTrustRadius = 10.0;
    options.gradientTolerance = 1e-9;

    const TrustRegionNewtonReport report = trustRegionNewtonCG(problem, NoProjection(),
                                                               Eigen::VectorXd::Ones(2), options);

    EXPECT_TRUE(report.accepted);
    EXPECT_LE(report.gradientNorm, options.gradientTolerance);
    EXPECT_NEAR(problem.state()(0), 1.0, 1e-6);
    EXPECT_NEAR(problem.state()(1), 1.0, 1e-6);
    EXPECT_LT(report.energy, 1e-16);
    EXPECT_GE(report.acceptedSteps, 1);
    EXPECT_GT(report.cgIterations, report.iterations);   // CG really iterates
    EXPECT_EQ(report.acceptedSteps + report.rejectedSteps, report.iterations - 1);
}

// ===========================================================================================
// mesh binding (stub mesh with the libshell API and a rigid-motion-invariant energy)
// ===========================================================================================

TEST(ShellEquilibriumMeshBinding, RigidModesAreNullDirectionsAndAreExcludedFromTheRitzPair)
{
    TrussMesh mesh(6, 4);
    const TrussHessian hessianOperator;
    const int nDofs = numberOfDofs(mesh.nVertices, mesh.nEdges);

    const RigidModeProjector projector = RigidModeProjector::fromMesh(mesh);
    ASSERT_EQ(projector.numberOfRigidModes(), 6);

    DeterministicStream stream(5ull);
    Eigen::VectorXd probe = stream.vector(nDofs);
    projector.applyInPlace(probe);
    probe.normalize();
    const Real reference = hessianOperator.hessianVectorProduct(mesh, probe).norm();
    ASSERT_GT(reference, 1e-3);

    for(int k = 0; k < projector.numberOfRigidModes(); ++k)
    {
        const Eigen::VectorXd mode = projector.basis().col(k);
        EXPECT_LT(hessianOperator.hessianVectorProduct(mesh, mode).norm(), 1e-10 * reference);
    }

    // dense reference spectrum of the projected operator: six zeros, then the physical modes
    Eigen::MatrixXd dense(nDofs, nDofs);
    for(int c = 0; c < nDofs; ++c)
    {
        Eigen::VectorXd basis = Eigen::VectorXd::Zero(nDofs);
        basis(c) = 1.0;
        projector.applyInPlace(basis);
        Eigen::VectorXd column = hessianOperator.hessianVectorProduct(mesh, basis);
        projector.applyInPlace(column);
        dense.col(c) = column;
    }
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> reference_spectrum(0.5 * (dense + dense.transpose()));
    for(int k = 0; k < 6; ++k) EXPECT_NEAR(reference_spectrum.eigenvalues()(k), 0.0, 1e-10);

    const RitzPairReport ritz = smallestProjectedRitzPairForMesh(mesh, hessianOperator, projector);
    ASSERT_TRUE(ritz.residualConverged);
    EXPECT_NEAR(ritz.eigenvalue, reference_spectrum.eigenvalues()(6), 1e-8);
    EXPECT_GT(ritz.eigenvalue, 0.0);
    EXPECT_LT((projector.basis().transpose() * ritz.eigenvector).norm(), 1e-10);

    // the congruent scaled operator preserves the inertia (sign), not the magnitude
    const Eigen::VectorXd metric = blockDiagonalMetric(mesh.nVertices, mesh.nEdges, 0.5, 1.0);
    const RitzPairReport scaled = smallestProjectedRitzPairForMesh(mesh, hessianOperator, projector, metric);
    ASSERT_TRUE(scaled.residualConverged);
    EXPECT_GT(scaled.eigenvalue, 0.0);
}

TEST(ShellEquilibriumMeshBinding, CorrectorDrivesAPerturbedConfigurationBackToEquilibrium)
{
    TrussMesh mesh(6, 4);
    const TrussEnergy energyOperator;
    const TrussHessian hessianOperator;
    const int nDofs = numberOfDofs(mesh.nVertices, mesh.nEdges);

    DeterministicStream stream(11ull);
    for(int i = 0; i < nDofs; ++i) mesh.data[i] += 0.05 * stream();
    const Real perturbedEnergy = energyOperator.compute(mesh);
    ASSERT_GT(perturbedEnergy, 1e-6);

    TrustRegionNewtonOptions options;
    options.gradientTolerance = 1e-10;
    options.maxIterations = 200;
    options.initialTrustRadius = 0.1;
    options.maxTrustRadius = 2.0;
    options.vertexLengthScale = 0.5;
    options.directorAngleScale = 1.0;

    const TrustRegionNewtonReport report = solveShellEquilibrium(mesh, energyOperator, hessianOperator, options);

    EXPECT_TRUE(report.accepted);
    EXPECT_EQ(report.rigidModes, 6);
    EXPECT_LE(report.gradientNorm, options.gradientTolerance);
    EXPECT_LT(report.energy, perturbedEnergy);
    EXPECT_GE(report.acceptedSteps, 1);
    EXPECT_GT(report.hessianVectorProducts, 0);

    // the mesh is left at exactly the state the report describes
    EXPECT_DOUBLE_EQ(energyOperator.compute(mesh), report.energy);
    for(int e = 0; e < mesh.nEdges; ++e)
        EXPECT_NEAR(mesh.data[3 * mesh.nVertices + e], 0.0, 1e-10);
}

// ===========================================================================================
// shell fixture: the exact bilayer Hessian and the real DCS mesh
// ===========================================================================================

TEST(ShellEquilibriumShellFixture, RigidModesAnnihilateTheExactBilayerHessian)
{
    RectangularPlate geometry(0.5, 0.5, 0.05, {false, false}, {false, false});
    geometry.setQuiet();

    BilayerMesh mesh;
    mesh.init(geometry, false);

    const Real Young = 1.0;
    const Real poisson = 0.3;
    const Real thickness = 0.01;
    const TinyADHessian_Bilayer<BilayerMesh> hessianOperator(Young, poisson, thickness);

    const int nDofs = numberOfDofs(mesh.getNumberOfVertices(), mesh.getNumberOfEdges());
    ASSERT_EQ(hessianOperator.nDofs(mesh), nDofs);

    const RigidModeProjector projector = RigidModeProjector::fromMesh(mesh);
    ASSERT_EQ(projector.numberOfRigidModes(), 6);   // the plate is free
    ASSERT_EQ(projector.numberOfConstrainedDofs(), 0);

    DeterministicStream stream(3ull);
    Eigen::VectorXd probe = stream.vector(nDofs);
    projector.applyInPlace(probe);
    probe.normalize();
    const Real reference = hessianOperator.hessianVectorProduct(mesh, probe).norm();
    ASSERT_GT(reference, 0.0);

    for(int k = 0; k < projector.numberOfRigidModes(); ++k)
    {
        const Eigen::VectorXd mode = projector.basis().col(k);
        EXPECT_LT(hessianOperator.hessianVectorProduct(mesh, mode).norm(), 1e-8 * reference);
    }
}

TEST(ShellEquilibriumShellFixture, CorrectorTerminatesImmediatelyAtTheRestState)
{
    RectangularPlate geometry(0.5, 0.5, 0.05, {false, false}, {false, false});
    geometry.setQuiet();

    BilayerMesh mesh;
    mesh.init(geometry, false);

    const Real Young = 1.0;
    const Real poisson = 0.3;
    const Real thickness = 0.01;
    MaterialProperties_Iso_Constant matprop(Young, poisson, thickness);
    CombinedOperator_Parametric<BilayerMesh, Material_Isotropic, bottom> engOp_bot(matprop);
    CombinedOperator_Parametric<BilayerMesh, Material_Isotropic, top> engOp_top(matprop);
    EnergyOperatorList<BilayerMesh> engOps({&engOp_bot, &engOp_top});
    const TinyADHessian_Bilayer<BilayerMesh> hessianOperator(Young, poisson, thickness);

    TrustRegionNewtonOptions options;
    options.gradientTolerance = 1e-10;
    options.maxIterations = 20;
    options.initialTrustRadius = 1e-2;
    options.vertexLengthScale = 0.05;
    options.directorAngleScale = 0.05;

    const TrustRegionNewtonReport report = solveShellEquilibrium(mesh, engOps, hessianOperator, options);

    // the flat plate with rest forms taken from itself is already an exact minimum
    EXPECT_TRUE(report.accepted);
    EXPECT_EQ(report.status, TrustRegionStatus::GradientTolerance);
    EXPECT_EQ(report.iterations, 1);
    EXPECT_EQ(report.acceptedSteps, 0);
    EXPECT_EQ(report.rejectedSteps, 0);
    EXPECT_EQ(report.rigidModes, 6);
    EXPECT_LE(report.gradientNorm, options.gradientTolerance);
    EXPECT_DOUBLE_EQ(report.energy, report.initialEnergy);
}

#include "gtest/gtest.h"
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
    struct DirectToyProblem
    {
        Eigen::VectorXd x = Eigen::VectorXd::Constant(2, 0.2);
        bool indefinite = false;
        bool rejectTrials = false;
        bool throwTrials = false;
        bool nonfiniteGradient = false;
        bool corruptTerminalGradient = false;
        bool nonfiniteHessian = false;
        bool missingDiagonal = false;
        bool asymmetric = false;
        int numberOfDofs() const { return 2; }
        Eigen::VectorXd state() const { return x; }
        void setState(const Eigen::VectorXd &value) { x = value; }
        Real energy()
        {
            if(throwTrials && (x.array() != 0.2).any())
                throw std::runtime_error("injected trial failure");
            if(rejectTrials && (x.array() != 0.2).any())
                return std::numeric_limits<Real>::infinity();
            return indefinite ? 0.25 * std::pow(x[0], 4) - 0.5 * x[0] * x[0] + 0.5 * x[1] * x[1]
                              : 0.5 * x.squaredNorm();
        }
        Eigen::VectorXd gradient()
        {
            Eigen::VectorXd g = x;
            if(indefinite) g[0] = x[0] * x[0] * x[0] - x[0];
            if(nonfiniteGradient || (corruptTerminalGradient && x[1] == 0.0))
                g[0] = std::numeric_limits<Real>::quiet_NaN();
            return g;
        }
        Eigen::SparseMatrix<double> hessianMatrix()
        {
            Eigen::SparseMatrix<double> H(2, 2);
            H.insert(0, 0) = indefinite ? 3 * x[0] * x[0] - 1 : 1;
            if(!missingDiagonal) H.insert(1, 1) = 1;
            if(nonfiniteHessian) H.coeffRef(0, 0) = std::numeric_limits<Real>::infinity();
            if(asymmetric) H.insert(0, 1) = 0.1;
            H.makeCompressed();
            return H;
        }
    };
}

TEST(ShellEquilibriumSparseDirect, IndefiniteHessianIsShiftedAndEnergyDecreases)
{
    DirectToyProblem problem;
    problem.indefinite = true;
    const Real initial = problem.energy();
    TrustRegionNewtonOptions options;
    options.maxIterations = 1;
    options.initialTrustRadius = 0.1;
    const auto identity = [](Eigen::VectorXd &) {};
    const auto report = trustRegionNewtonDirect(problem, identity, Eigen::VectorXd::Ones(2), options);
    EXPECT_GT(report.diagonalShift, 0.0);
    EXPECT_GT(report.directFactorizationAttempts, 1);
    EXPECT_EQ(report.acceptedSteps, 1);
    EXPECT_LT(problem.energy(), initial);
    EXPECT_DOUBLE_EQ(report.energy, problem.energy());
    EXPECT_NEAR(report.gradientNorm, problem.gradient().norm(), 1e-14);
}

TEST(ShellEquilibriumSparseDirect, NonfiniteTrialsRestoreTheExactBaseState)
{
    DirectToyProblem problem;
    problem.rejectTrials = true;
    const Eigen::VectorXd base = problem.state();
    TrustRegionNewtonOptions options;
    options.maxIterations = 3;
    const auto identity = [](Eigen::VectorXd &) {};
    const auto report = trustRegionNewtonDirect(problem, identity, Eigen::VectorXd::Ones(2), options);
    EXPECT_FALSE(report.accepted);
    EXPECT_EQ(report.acceptedSteps, 0);
    EXPECT_EQ(report.rejectedSteps, 3);
    EXPECT_TRUE((problem.state().array() == base.array()).all());
    EXPECT_DOUBLE_EQ(report.energy, problem.energy());
}

TEST(ShellEquilibriumSparseDirect, ShiftBudgetFailureLeavesTheBaseStateUnchanged)
{
    DirectToyProblem problem;
    problem.indefinite = true;
    const Eigen::VectorXd base = problem.state();
    TrustRegionNewtonOptions options;
    options.directMaxShiftAttempts = 1;
    const auto identity = [](Eigen::VectorXd &) {};
    const auto report = trustRegionNewtonDirect(problem, identity, Eigen::VectorXd::Ones(2), options);
    EXPECT_FALSE(report.accepted);
    EXPECT_EQ(report.status, TrustRegionStatus::SparseFactorizationFailed);
    EXPECT_TRUE((problem.state().array() == base.array()).all());
}

TEST(ShellEquilibriumSparseDirect, RejectsInvalidControlsAndOraclesBeforeTrial)
{
    const auto identity = [](Eigen::VectorXd &) {};
    const auto eraseCorruptComponent = [](Eigen::VectorXd &g) { g[0] = 0.0; };
    const Eigen::VectorXd metric = Eigen::VectorXd::Ones(2);
    TrustRegionNewtonOptions options;
    DirectToyProblem problem;
    options.gradientTolerance = std::numeric_limits<Real>::infinity();
    EXPECT_THROW(trustRegionNewtonDirect(problem, identity, metric, options), std::invalid_argument);
    options = TrustRegionNewtonOptions();
    problem.nonfiniteGradient = true;
    auto report = trustRegionNewtonDirect(problem, eraseCorruptComponent, metric, options);
    EXPECT_EQ(report.status, TrustRegionStatus::NonFiniteState);
    EXPECT_EQ(report.acceptedSteps, 0);
    problem.nonfiniteGradient = false;
    problem.nonfiniteHessian = true;
    report = trustRegionNewtonDirect(problem, identity, metric, options);
    EXPECT_EQ(report.status, TrustRegionStatus::NonFiniteState);
    problem.nonfiniteHessian = false;
    problem.missingDiagonal = true;
    EXPECT_THROW(trustRegionNewtonDirect(problem, identity, metric, options), std::invalid_argument);
    problem.missingDiagonal = false;
    problem.asymmetric = true;
    EXPECT_THROW(trustRegionNewtonDirect(problem, identity, metric, options), std::invalid_argument);
}

TEST(ShellEquilibriumSparseDirect, TerminalGradientCorruptionCannotBeHiddenByProjection)
{
    DirectToyProblem problem;
    problem.corruptTerminalGradient = true;
    const auto eraseCorruptComponent = [](Eigen::VectorXd &g) { g[0] = 0.0; };
    TrustRegionNewtonOptions options;
    options.maxIterations = 1;
    options.initialTrustRadius = 1.0;
    const auto report = trustRegionNewtonDirect(problem, eraseCorruptComponent,
                                                Eigen::VectorXd::Ones(2), options);
    ASSERT_EQ(report.acceptedSteps, 1);
    EXPECT_FALSE(report.accepted);
    EXPECT_EQ(report.status, TrustRegionStatus::NonFiniteState);
    EXPECT_FALSE(std::isfinite(report.gradientNorm));
    EXPECT_DOUBLE_EQ(report.energy, problem.energy());
}

TEST(ShellEquilibriumSparseDirect, ZeroIterationTerminalGradientIsValidatedBeforeProjection)
{
    DirectToyProblem problem;
    problem.nonfiniteGradient = true;
    const auto eraseCorruptComponent = [](Eigen::VectorXd &g) { g[0] = 0.0; };
    TrustRegionNewtonOptions options;
    options.maxIterations = 0;
    const auto report = trustRegionNewtonDirect(problem, eraseCorruptComponent,
                                                Eigen::VectorXd::Ones(2), options);
    EXPECT_FALSE(report.accepted);
    EXPECT_EQ(report.status, TrustRegionStatus::NonFiniteState);
    EXPECT_EQ(report.gradientEvaluations, 1);
    EXPECT_EQ(report.hessianAssemblies, 0);
}

TEST(ShellEquilibriumSparseDirect, ThrowingTrialRestoresBaseBeforeRethrowing)
{
    DirectToyProblem problem;
    problem.throwTrials = true;
    const Eigen::VectorXd base = problem.state();
    const auto identity = [](Eigen::VectorXd &) {};
    EXPECT_THROW(trustRegionNewtonDirect(problem, identity, Eigen::VectorXd::Ones(2)), std::runtime_error);
    EXPECT_TRUE((problem.state().array() == base.array()).all());
}

TEST(ShellEquilibriumSparseDirect, DeformedFreeBilayerConvergesWithFreshRigidProjection)
{
    RectangularPlate geometry(0.2, 0.2, 0.12, {false, false}, {false, false});
    geometry.setQuiet();
    BilayerMesh mesh;
    mesh.init(geometry, false);
    const int n = numberOfDofs(mesh.getNumberOfVertices(), mesh.getNumberOfEdges());
    Eigen::Map<Eigen::VectorXd> data(mesh.getDataPointer(), n);
    for(int i = 0; i < n; ++i) data[i] += 1e-5 * std::sin(0.37 * (i + 1));
    mesh.updateDeformedConfiguration();
    MaterialProperties_Iso_Constant material(1.0, 0.3, 0.02);
    CombinedOperator_Parametric<BilayerMesh, Material_Isotropic, bottom> bottomEnergy(material);
    CombinedOperator_Parametric<BilayerMesh, Material_Isotropic, top> topEnergy(material);
    EnergyOperatorList<BilayerMesh> energy({&bottomEnergy, &topEnergy});
    TinyADHessian_Bilayer<BilayerMesh> hessian(1.0, 0.3, 0.02);
    const Real initialEnergy = energy.compute(mesh);
    TrustRegionNewtonOptions options;
    options.backend = TrustRegionNewtonBackend::SparseDirect;
    options.gradientTolerance = 1e-10;
    options.maxIterations = 100;
    options.initialTrustRadius = 0.1;
    options.maxTrustRadius = 1.0;
    options.vertexLengthScale = 0.02;
    options.directorAngleScale = 1.0;
    const auto report = solveShellEquilibrium(mesh, energy, hessian, options);
    EXPECT_TRUE(report.accepted) << toString(report.status) << " g=" << report.gradientNorm
        << " steps=" << report.acceptedSteps << " shift=" << report.diagonalShift;
    EXPECT_EQ(report.rigidModes, 6);
    EXPECT_GT(report.hessianAssemblies, 0);
    EXPECT_EQ(report.hessianVectorProducts, 0);
    EXPECT_LT(report.energy, initialEnergy);
    Eigen::VectorXd finalGradient = hessian.gradient(mesh);
    const auto projector = RigidModeProjector::fromMesh(mesh);
    projector.applyInPlace(finalGradient);
    EXPECT_NEAR(report.gradientNorm, finalGradient.norm(), 1e-14);
    EXPECT_LE(finalGradient.norm(), options.gradientTolerance);
    EXPECT_DOUBLE_EQ(report.energy, energy.compute(mesh));
}

#include "gtest/gtest.h"
#include "ShellEquilibriumSolver.hpp"
#include "Geometry.hpp"
#include "Mesh.hpp"
#include "GrowthHelper.hpp"
#include "MaterialProperties.hpp"
#include "CombinedOperator_Parametric.hpp"
#include "EnergyOperatorList.hpp"
#include "TinyADHessian_Bilayer.hpp"
#include <iostream>

// Controlled operator qualification, NOT a replay of a zigzag toolpath or a
// certification of its final equilibrium. Parameters match the study scale;
// top growth is uniform per cycle so the oracle check is reproducible and bounded.
TEST(NegativeOrthotropyOracle, TwoOrthogonalGrowthCyclesAtStudyScale)
{
    using namespace ShellEquilibrium;
    RectangularPlate geometry(0.127, 0.1524, 0.030, {false, false}, {false, false});
    geometry.setQuiet();
    BilayerMesh mesh;
    mesh.init(geometry, false);
    const int nv = mesh.getNumberOfVertices();
    const int n = numberOfDofs(nv, mesh.getNumberOfEdges());
    const Eigen::MatrixXd rest = mesh.getRestConfiguration().getVertices();
    const Eigen::MatrixXi faces = mesh.getTopology().getFace2Vertices();
    auto &topForms = mesh.getRestConfiguration().getFirstFundamentalForms<top>();
    Eigen::Map<Eigen::VectorXd> state(mesh.getDataPointer(), n);
    for(int i = 0; i < nv; ++i)
    {
        const Real x = rest(i, 0), y = rest(i, 1);
        state[2 * nv + i] += 0.003 * std::cos(12 * x) * std::sin(10 * y) + 0.002 * x;
    }
    for(int i = 3 * nv; i < n; ++i) state[i] += 0.005 * std::cos(0.21 * i);
    mesh.updateDeformedConfiguration();
    MaterialProperties_Iso_Constant material(1.0, 0.33, 0.0006);
    CombinedOperator_Parametric<BilayerMesh, Material_Isotropic, bottom> bottomEnergy(material);
    CombinedOperator_Parametric<BilayerMesh, Material_Isotropic, top> topEnergy(material);
    EnergyOperatorList<BilayerMesh> energy({&bottomEnergy, &topEnergy});
    TinyADHessian_Bilayer<BilayerMesh> hessian(1.0, 0.33, 0.0006, 4);
    MeshEquilibriumProblem<BilayerMesh, decltype(energy), decltype(hessian)> problem(mesh, energy, hessian);

    for(int cycle = 0; cycle < 2; ++cycle)
    {
        const Real angle = cycle == 0 ? 0.0 : 1.5707963267948966;
        // g = 0.003, ortho = -0.5: g1 = 0.0015, g2 = 0.0045.
        for(int f = 0; f < mesh.getNumberOfFaces(); ++f)
            GrowthHelper<BilayerMesh>::updateAbarWithMaterialGrowthIncrement(
                rest.leftCols<2>(), faces, f, angle, 0.0015, 0.0045, topForms[f]);
        const Eigen::VectorXd base = problem.state();
        ASSERT_TRUE(base.allFinite());
        const Real e = energy.compute(mesh);
        ASSERT_TRUE(std::isfinite(e));
        Eigen::VectorXd independentGradient = Eigen::VectorXd::Zero(n);
        const Real gradientEnergy = energy.compute(mesh, independentGradient);
        const Eigen::VectorXd g = hessian.gradient(mesh);
        ASSERT_TRUE(g.allFinite());
        ASSERT_TRUE(independentGradient.allFinite());
        EXPECT_NEAR(e, gradientEnergy, 1e-12 * std::max<Real>(std::abs(e), 1e-20));
        const Real gradientRelative = (g - independentGradient).norm() /
            std::max<Real>(independentGradient.norm(), 1e-30);
        EXPECT_LT(gradientRelative, 1e-10);

        const Eigen::SparseMatrix<double> H = hessian.assembleHessian(mesh);
        Real entryScale = 0.0, symmetryError = 0.0;
        for(int col = 0; col < n; ++col)
        {
            bool diagonal = false;
            for(Eigen::SparseMatrix<double>::InnerIterator it(H, col); it; ++it)
            {
                ASSERT_TRUE(std::isfinite(it.value()));
                entryScale = std::max(entryScale, std::abs(it.value()));
                symmetryError = std::max(symmetryError, std::abs(it.value() - H.coeff(col, it.row())));
                diagonal = diagonal || it.row() == col;
            }
            ASSERT_TRUE(diagonal) << "Missing diagonal " << col;
        }
        EXPECT_LE(symmetryError, 1e-12 * std::max<Real>(entryScale, 1e-300));
        for(int probe = 0; probe < 3; ++probe)
        {
            Eigen::VectorXd v(n);
            for(int i = 0; i < n; ++i)
                v[i] = (i < 3 * nv ? 0.01 : 0.1) * std::sin((0.13 + 0.07 * probe) * (i + 1));
            const Eigen::VectorXd hv = hessian.hessianVectorProduct(mesh, v);
            ASSERT_TRUE(hv.allFinite());
            const Real hvRelative = (H * v - hv).norm() / std::max<Real>(hv.norm(), 1e-30);
            EXPECT_LT(hvRelative, 1e-11);
            const Real slope = independentGradient.dot(v);
            Real bestFiniteDifferenceError = std::numeric_limits<Real>::infinity();
            for(const Real epsilon : {1e-3, 1e-4, 1e-5})
            {
                problem.setState(base + epsilon * v);
                const Real plus = problem.energy();
                problem.setState(base - epsilon * v);
                const Real minus = problem.energy();
                problem.setState(base);
                ASSERT_TRUE(std::isfinite(plus) && std::isfinite(minus));
                const Real numericalSlope = (plus - minus) / (2 * epsilon);
                bestFiniteDifferenceError = std::min(bestFiniteDifferenceError, std::abs(numericalSlope - slope));
            }
            EXPECT_LE(bestFiniteDifferenceError, 1e-5 * std::max<Real>(std::abs(slope), 1e-20) + 1e-18);
            std::cout << "[negative-orthotropy oracle] cycle=" << cycle + 1 << " probe=" << probe
                << " dofs=" << n << " energy=" << e << " gradient_rel=" << gradientRelative
                << " symmetry_rel=" << symmetryError / std::max<Real>(entryScale, 1e-300)
                << " hvp_rel=" << hvRelative << " fd_abs=" << bestFiniteDifferenceError << '\n';
        }
        EXPECT_TRUE((problem.state().array() == base.array()).all());
    }
}

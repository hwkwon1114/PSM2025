//
//  pyshell.cpp
//
//  Python bindings for the libshell energy operators.
//
//  The point of this module is to expose the energy and its gradient over the full
//  degree-of-freedom vector, so that numerical experiments (eigenmode analysis,
//  preconditioners, alternative optimizers) can be prototyped in Python against the
//  same C++ energy the production solver uses.
//
//  DOF layout, matching ConfigurationData::mapArrays :
//      [ x_0 .. x_{nV-1} | y_0 .. y_{nV-1} | z_0 .. z_{nV-1} | theta_0 .. theta_{nE-1} ]
//  so the vertex block is column-major, NOT interleaved. nDofs = 3*nV + nE.
//

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "common.hpp"
#include "Geometry.hpp"
#include "Mesh.hpp"
#include "MaterialProperties.hpp"
#include "EnergyOperator.hpp"
#include "EnergyOperatorList.hpp"
#include "CombinedOperator_Parametric.hpp"
#include "GrowthHelper.hpp"

#include <TinyAD/Scalar.hh>

#include <stdexcept>
#include <cmath>

namespace py = pybind11;

namespace
{

/**
 * The six rigid-body modes, in the DOF layout described above.
 *
 * Translations and infinitesimal rotations act on the vertex block only; the edge
 * directors are angles measured relative to the face frame and so are unchanged by a
 * global rigid motion. The columns are orthonormalized before being returned, which is
 * what a deflation step wants.
 */
template<typename tMesh>
Eigen::MatrixXd rigidModes(const tMesh & mesh, const int nDofs)
{
    const int nV = mesh.getNumberOfVertices();
    const auto verts = mesh.getCurrentConfiguration().getVertices();

    Eigen::MatrixXd modes(nDofs, 6);
    modes.setZero();

    for(int i=0;i<nV;++i)
    {
        const Real x = verts(i,0);
        const Real y = verts(i,1);
        const Real z = verts(i,2);

        modes(0*nV + i, 0) = 1.0; // translate x
        modes(1*nV + i, 1) = 1.0; // translate y
        modes(2*nV + i, 2) = 1.0; // translate z

        modes(1*nV + i, 3) = -z;  // rotate about x
        modes(2*nV + i, 3) =  y;

        modes(2*nV + i, 4) = -x;  // rotate about y
        modes(0*nV + i, 4) =  z;

        modes(0*nV + i, 5) = -y;  // rotate about z
        modes(1*nV + i, 5) =  x;
    }

    const Eigen::HouseholderQR<Eigen::MatrixXd> qr(modes);
    return qr.householderQ() * Eigen::MatrixXd::Identity(nDofs, 6);
}

/// Pack a tVecMat2d of symmetric 2x2 forms into an (nFaces, 3) array of (a11, a12, a22).
Eigen::MatrixXd formsToArray(const tVecMat2d & forms)
{
    const int n = (int)forms.size();
    Eigen::MatrixXd out(n, 3);
    for(int i=0;i<n;++i)
    {
        out(i,0) = forms[i](0,0);
        out(i,1) = forms[i](0,1);
        out(i,2) = forms[i](1,1);
    }
    return out;
}

/// Inverse of formsToArray.
void arrayToForms(const Eigen::MatrixXd & arr, tVecMat2d & forms)
{
    if(arr.cols() != 3)
        throw std::invalid_argument("fundamental forms array must have 3 columns (a11, a12, a22)");
    if((int)forms.size() != (int)arr.rows())
        throw std::invalid_argument("fundamental forms array has the wrong number of rows (expected one per face)");

    for(int i=0;i<arr.rows();++i)
    {
        forms[i](0,0) = arr(i,0);
        forms[i](0,1) = forms[i](1,0) = arr(i,1);
        forms[i](1,1) = arr(i,2);
    }
}

/**
 * Common state and accessors shared by the monolayer and bilayer wrappers.
 */
template<typename tMesh>
class ShellBase
{
protected:
    tMesh mesh;
    Real E, nu, h;
    bool initialized;

    void requireMesh() const
    {
        if(not initialized)
            throw std::runtime_error("no geometry yet -- call one of the init_* methods first");
    }

public:
    ShellBase(): E(1.0), nu(0.3), h(0.01), initialized(false) {}
    virtual ~ShellBase() {}

    virtual int nDofs() const = 0;

    void initDisk(const Real radius, const int res, const bool fixedBoundary)
    {
        CircularPlate geometry(radius, res, fixedBoundary);
        geometry.setQuiet();
        mesh.init(geometry);
        initialized = true;
    }

    void initAnnulus(const Real outerRadius, const Real innerRadius, const Real edgeLength,
                     const bool fixedOuter, const bool fixedInner)
    {
        AnnulusPlate geometry(outerRadius, innerRadius, edgeLength, fixedOuter, fixedInner);
        geometry.setQuiet();
        mesh.init(geometry);
        initialized = true;
    }

    void initRectangle(const Real halfX, const Real halfY, const Real maxRelArea,
                       const std::array<bool,2> fixedX, const std::array<bool,2> fixedY)
    {
        RectangularPlate geometry(halfX, halfY, maxRelArea, fixedX, fixedY);
        geometry.setQuiet();
        mesh.init(geometry);
        initialized = true;
    }

    void setMaterial(const Real E_in, const Real nu_in, const Real h_in)
    {
        E = E_in;
        nu = nu_in;
        h = h_in;
    }

    py::tuple getMaterial() const { return py::make_tuple(E, nu, h); }

    int nVertices() const { requireMesh(); return mesh.getNumberOfVertices(); }
    int nFaces()    const { requireMesh(); return mesh.getNumberOfFaces(); }
    int nEdges()    const { requireMesh(); return mesh.getNumberOfEdges(); }

    Eigen::VectorXd getDofs()
    {
        requireMesh();
        const int n = nDofs();
        return Eigen::Map<const Eigen::VectorXd>(mesh.getDataPointer(), n);
    }

    void setDofs(const Eigen::Ref<const Eigen::VectorXd> & x)
    {
        requireMesh();
        const int n = nDofs();
        if(x.size() != n)
            throw std::invalid_argument("dof vector has the wrong length");
        Eigen::Map<Eigen::VectorXd>(mesh.getDataPointer(), n) = x;
        mesh.updateDeformedConfiguration();
    }

    Eigen::MatrixXd vertices()
    {
        requireMesh();
        return mesh.getCurrentConfiguration().getVertices();
    }

    Eigen::MatrixXi faces()
    {
        requireMesh();
        return mesh.getTopology().getFace2Vertices();
    }

    Eigen::MatrixXd restVertices()
    {
        requireMesh();
        return mesh.getRestConfiguration().getVertices();
    }

    Eigen::MatrixXd rigidBodyModes()
    {
        requireMesh();
        return rigidModes(mesh, nDofs());
    }

    tMesh & getMesh() { return mesh; }
};

/**
 * Single-layer shell: one prescribed first fundamental form per face.
 */
class MonolayerShell : public ShellBase<Mesh>
{
    typedef CombinedOperator_Parametric<Mesh, Material_Isotropic, single> tOperator;

public:
    int nDofs() const override
    {
        requireMesh();
        return 3*mesh.getNumberOfVertices() + mesh.getNumberOfEdges();
    }

    Real energy()
    {
        requireMesh();
        MaterialProperties_Iso_Constant matprop(E, nu, h);
        tOperator op(matprop);
        mesh.updateDeformedConfiguration();
        return op.compute(mesh);
    }

    py::tuple energyAndGradient()
    {
        requireMesh();
        MaterialProperties_Iso_Constant matprop(E, nu, h);
        tOperator op(matprop);
        mesh.updateDeformedConfiguration();

        Eigen::VectorXd grad = Eigen::VectorXd::Zero(nDofs());
        const Real eng = op.compute(mesh, grad);
        return py::make_tuple(eng, grad);
    }

    py::dict energyTerms()
    {
        requireMesh();
        MaterialProperties_Iso_Constant matprop(E, nu, h);
        tOperator op(matprop);
        mesh.updateDeformedConfiguration();
        op.compute(mesh);

        py::dict out;
        out["stretching_aa"] = op.getLastStretchingEnergy();
        out["bending_bb"] = op.getLastBendingEnergy();
        return out;
    }

    Eigen::MatrixXd getAbars()
    {
        requireMesh();
        return formsToArray(mesh.getRestConfiguration().getFirstFundamentalForms());
    }

    /**
     * STAGE 2a: per-face Saint-Venant STRETCHING energy and its gradient, computed via
     * TinyAD forward-mode autodiff over the 9 vertex DOFs of each face.
     *
     * This reproduces the production stretching term (SaintVenantEnergy<...,single>, the
     * `stretching_aa` contribution) independently, purely from the deformed vertex
     * positions and the per-face rest first fundamental form (abar). It does NOT touch
     * bending or the existing gradient code; it exists to seed an exact-Hessian effort.
     *
     * Formula (all verified against the C++ energy):
     *   e1 = v2 - v1,  e2 = v0 - v2                (CreateExtendedTriangleInfos.hpp:87-88)
     *   firstFF = [[e1.e1, e1.e2],[e1.e2, e2.e2]]  (ExtendedTriangleInfo.hpp:221-227)
     *   E    = abar_inv * firstFF - I              (EnergyHelper_Parametric.hpp:75)
     *   norm = (c1*tr(E)^2 + c2*tr(E*E)) * area    (EnergyHelper_Parametric.hpp:148)
     *   E_face = (h/4) * norm                      (SaintVenantEnergy compute_h_aa :429, compute :477)
     * with c1 = getStVenantFactor1() = 0.5*E*nu/(1-nu^2)  (MaterialProperties.hpp:45),
     *      c2 = getStVenantFactor2() = 0.5*E/(1+nu)       (MaterialProperties.hpp:50),
     *      area = 0.5*sqrt(det(abar))                     (EnergyHelper_Parametric.hpp:33).
     *
     * Returns (total_stretching_energy, full_gradient) with the gradient scattered into a
     * length-(3*nV+nE) vector in the column-major vertex layout [x|y|z|theta]; the theta
     * block is left zero because stretching does not depend on the directors.
     */
    py::tuple stretchingEnergyAndGradientTinyAD()
    {
        requireMesh();
        mesh.updateDeformedConfiguration();

        const int nV = mesh.getNumberOfVertices();
        const int nF = mesh.getNumberOfFaces();
        const int nD = nDofs();

        const Eigen::MatrixXi F = mesh.getTopology().getFace2Vertices();
        const Eigen::MatrixXd V = mesh.getCurrentConfiguration().getVertices();
        const tVecMat2d & abars = mesh.getRestConfiguration().getFirstFundamentalForms();

        // Isotropic Saint-Venant material factors, matching Material_Isotropic.
        const Real c1 = 0.5 * E * nu / (1.0 - nu * nu); // getStVenantFactor1()
        const Real c2 = 0.5 * E / (1.0 + nu);           // getStVenantFactor2()
        const Real h_aa = h / 4.0;                       // single-layer compute_h_aa

        Eigen::VectorXd grad = Eigen::VectorXd::Zero(nD);
        Real total = 0.0;

        // 9 active variables per face: [v0(xyz) v1(xyz) v2(xyz)]. Gradient only (no Hessian).
        using ADouble = TinyAD::Double<9, false>;

        for(int f = 0; f < nF; ++f)
        {
            const int i0 = F(f, 0), i1 = F(f, 1), i2 = F(f, 2);

            Eigen::Matrix<double, 9, 1> x0;
            x0 << V(i0, 0), V(i0, 1), V(i0, 2),
                  V(i1, 0), V(i1, 1), V(i1, 2),
                  V(i2, 0), V(i2, 1), V(i2, 2);

            const Eigen::Matrix<ADouble, 9, 1> x = ADouble::make_active(x0);

            const Eigen::Matrix<ADouble, 3, 1> v0 = x.segment<3>(0);
            const Eigen::Matrix<ADouble, 3, 1> v1 = x.segment<3>(3);
            const Eigen::Matrix<ADouble, 3, 1> v2 = x.segment<3>(6);

            const Eigen::Matrix<ADouble, 3, 1> e1 = v2 - v1;
            const Eigen::Matrix<ADouble, 3, 1> e2 = v0 - v2;

            const ADouble F11 = e1.dot(e1);
            const ADouble F12 = e1.dot(e2);
            const ADouble F22 = e2.dot(e2);

            const Eigen::Matrix2d & abar = abars[f];
            const Eigen::Matrix2d abar_inv = abar.inverse();
            const double ai11 = abar_inv(0, 0);
            const double ai12 = abar_inv(0, 1);
            const double ai22 = abar_inv(1, 1);
            const double area = 0.5 * std::sqrt(abar.determinant());

            // strain E = abar_inv * firstFF - I  (row-major 2x2, generally non-symmetric)
            const ADouble E11 = ai11 * F11 + ai12 * F12 - 1.0;
            const ADouble E12 = ai11 * F12 + ai12 * F22;
            const ADouble E21 = ai12 * F11 + ai22 * F12;
            const ADouble E22 = ai12 * F12 + ai22 * F22 - 1.0;

            const ADouble trace = E11 + E22;
            const ADouble trace_sq = E11 * E11 + 2.0 * E12 * E21 + E22 * E22; // tr(E*E)

            const ADouble material_norm = (c1 * trace * trace + c2 * trace_sq) * area;
            const ADouble energy = h_aa * material_norm;

            total += energy.val;

            const Eigen::Matrix<double, 9, 1> & g = energy.grad;
            const int idx[3] = {i0, i1, i2};
            for(int a = 0; a < 3; ++a)
            {
                grad(0 * nV + idx[a]) += g(3 * a + 0);
                grad(1 * nV + idx[a]) += g(3 * a + 1);
                grad(2 * nV + idx[a]) += g(3 * a + 2);
            }
        }

        return py::make_tuple(total, grad);
    }

    void setAbars(const Eigen::MatrixXd & arr)
    {
        requireMesh();
        arrayToForms(arr, mesh.getRestConfiguration().getFirstFundamentalForms());
    }

    /// Orthotropic growth: principal direction at angle `angles`, stretches (1+rate1), (1+rate2).
    void setOrthoGrowth(const Eigen::VectorXd & angles,
                        const Eigen::VectorXd & rate1,
                        const Eigen::VectorXd & rate2)
    {
        requireMesh();
        const int nF = mesh.getNumberOfFaces();
        if(angles.size() != nF or rate1.size() != nF or rate2.size() != nF)
            throw std::invalid_argument("growth arrays must have one entry per face");

        GrowthHelper<Mesh>::computeAbarsOrthoGrowth(
            mesh, angles, rate1, rate2, mesh.getRestConfiguration().getFirstFundamentalForms());
    }
};

/**
 * Bilayer shell: separate prescribed first fundamental forms for the bottom and top layer.
 * The mismatch between them is what produces a spontaneous curvature.
 */
class BilayerShell : public ShellBase<BilayerMesh>
{
    typedef CombinedOperator_Parametric<BilayerMesh, Material_Isotropic, bottom> tOperatorBot;
    typedef CombinedOperator_Parametric<BilayerMesh, Material_Isotropic, top> tOperatorTop;

public:
    int nDofs() const override
    {
        requireMesh();
        return 3*mesh.getNumberOfVertices() + mesh.getNumberOfEdges();
    }

    Real energy()
    {
        requireMesh();
        MaterialProperties_Iso_Constant matprop(E, nu, h);
        tOperatorBot opBot(matprop);
        tOperatorTop opTop(matprop);
        EnergyOperatorList<BilayerMesh> op({&opBot, &opTop});
        mesh.updateDeformedConfiguration();
        return op.compute(mesh);
    }

    py::tuple energyAndGradient()
    {
        requireMesh();
        MaterialProperties_Iso_Constant matprop(E, nu, h);
        tOperatorBot opBot(matprop);
        tOperatorTop opTop(matprop);
        EnergyOperatorList<BilayerMesh> op({&opBot, &opTop});
        mesh.updateDeformedConfiguration();

        Eigen::VectorXd grad = Eigen::VectorXd::Zero(nDofs());
        const Real eng = op.compute(mesh, grad);
        return py::make_tuple(eng, grad);
    }

    Eigen::MatrixXd getAbars(const std::string & layer)
    {
        requireMesh();
        if(layer == "bottom")
            return formsToArray(mesh.getRestConfiguration().getFirstFundamentalForms<bottom>());
        if(layer == "top")
            return formsToArray(mesh.getRestConfiguration().getFirstFundamentalForms<top>());
        throw std::invalid_argument("layer must be 'bottom' or 'top'");
    }

    void setAbars(const std::string & layer, const Eigen::MatrixXd & arr)
    {
        requireMesh();
        if(layer == "bottom")
            arrayToForms(arr, mesh.getRestConfiguration().getFirstFundamentalForms<bottom>());
        else if(layer == "top")
            arrayToForms(arr, mesh.getRestConfiguration().getFirstFundamentalForms<top>());
        else
            throw std::invalid_argument("layer must be 'bottom' or 'top'");
    }

    void setOrthoGrowth(const std::string & layer,
                        const Eigen::VectorXd & angles,
                        const Eigen::VectorXd & rate1,
                        const Eigen::VectorXd & rate2)
    {
        requireMesh();
        const int nF = mesh.getNumberOfFaces();
        if(angles.size() != nF or rate1.size() != nF or rate2.size() != nF)
            throw std::invalid_argument("growth arrays must have one entry per face");

        if(layer == "bottom")
            GrowthHelper<BilayerMesh>::computeAbarsOrthoGrowth(
                mesh, angles, rate1, rate2, mesh.getRestConfiguration().getFirstFundamentalForms<bottom>());
        else if(layer == "top")
            GrowthHelper<BilayerMesh>::computeAbarsOrthoGrowth(
                mesh, angles, rate1, rate2, mesh.getRestConfiguration().getFirstFundamentalForms<top>());
        else
            throw std::invalid_argument("layer must be 'bottom' or 'top'");
    }
};

} // anonymous namespace


PYBIND11_MODULE(pyshell, m)
{
    m.doc() =
        "Python bindings for libshell.\n\n"
        "Exposes the elastic energy and its gradient over the full DOF vector, laid out as\n"
        "    [ x (nV) | y (nV) | z (nV) | edge directors (nE) ]\n"
        "so the vertex block is column-major rather than interleaved.";

    py::class_<MonolayerShell>(m, "MonolayerShell",
        "Single-layer non-Euclidean plate: one prescribed first fundamental form per face.")
        .def(py::init<>())
        .def("init_disk", &MonolayerShell::initDisk,
             py::arg("radius") = 1.0, py::arg("res") = 64, py::arg("fixed_boundary") = false)
        .def("init_annulus", &MonolayerShell::initAnnulus,
             py::arg("outer_radius"), py::arg("inner_radius"), py::arg("edge_length"),
             py::arg("fixed_outer") = false, py::arg("fixed_inner") = false)
        .def("init_rectangle", &MonolayerShell::initRectangle,
             py::arg("half_x"), py::arg("half_y"), py::arg("max_rel_area"),
             py::arg("fixed_x") = std::array<bool,2>{{false,false}},
             py::arg("fixed_y") = std::array<bool,2>{{false,false}})
        .def("set_material", &MonolayerShell::setMaterial,
             py::arg("E"), py::arg("nu"), py::arg("h"))
        .def("get_material", &MonolayerShell::getMaterial, "Returns (E, nu, h).")
        .def_property_readonly("n_dofs", &MonolayerShell::nDofs)
        .def_property_readonly("n_vertices", &MonolayerShell::nVertices)
        .def_property_readonly("n_faces", &MonolayerShell::nFaces)
        .def_property_readonly("n_edges", &MonolayerShell::nEdges)
        .def("get_dofs", &MonolayerShell::getDofs)
        .def("set_dofs", &MonolayerShell::setDofs, py::arg("x"))
        .def("vertices", &MonolayerShell::vertices)
        .def("rest_vertices", &MonolayerShell::restVertices)
        .def("faces", &MonolayerShell::faces)
        .def("energy", &MonolayerShell::energy)
        .def("energy_and_gradient", &MonolayerShell::energyAndGradient,
             "Returns (energy, gradient) with the gradient over the full DOF vector.")
        .def("energy_terms", &MonolayerShell::energyTerms,
             "Returns the stretching and bending contributions separately.")
        .def("stretching_energy_and_gradient_tinyad",
             &MonolayerShell::stretchingEnergyAndGradientTinyAD,
             "Per-face Saint-Venant stretching energy and its full-DOF gradient via TinyAD "
             "autodiff. Returns (energy, gradient); matches energy_terms()['stretching_aa'].")
        .def("get_abars", &MonolayerShell::getAbars,
             "Prescribed first fundamental forms as an (n_faces, 3) array of (a11, a12, a22).")
        .def("set_abars", &MonolayerShell::setAbars, py::arg("abars"))
        .def("set_ortho_growth", &MonolayerShell::setOrthoGrowth,
             py::arg("angles"), py::arg("rate1"), py::arg("rate2"),
             "Orthotropic growth per face: principal direction at `angles`, stretches (1+rate1) and (1+rate2).")
        .def("rigid_body_modes", &MonolayerShell::rigidBodyModes,
             "The 6 orthonormalized rigid-body modes as an (n_dofs, 6) array, for deflation.");

    py::class_<BilayerShell>(m, "BilayerShell",
        "Two bonded layers with separate prescribed metrics; the mismatch gives a spontaneous curvature.")
        .def(py::init<>())
        .def("init_disk", &BilayerShell::initDisk,
             py::arg("radius") = 1.0, py::arg("res") = 64, py::arg("fixed_boundary") = false)
        .def("init_annulus", &BilayerShell::initAnnulus,
             py::arg("outer_radius"), py::arg("inner_radius"), py::arg("edge_length"),
             py::arg("fixed_outer") = false, py::arg("fixed_inner") = false)
        .def("init_rectangle", &BilayerShell::initRectangle,
             py::arg("half_x"), py::arg("half_y"), py::arg("max_rel_area"),
             py::arg("fixed_x") = std::array<bool,2>{{false,false}},
             py::arg("fixed_y") = std::array<bool,2>{{false,false}})
        .def("set_material", &BilayerShell::setMaterial,
             py::arg("E"), py::arg("nu"), py::arg("h"))
        .def("get_material", &BilayerShell::getMaterial, "Returns (E, nu, h).")
        .def_property_readonly("n_dofs", &BilayerShell::nDofs)
        .def_property_readonly("n_vertices", &BilayerShell::nVertices)
        .def_property_readonly("n_faces", &BilayerShell::nFaces)
        .def_property_readonly("n_edges", &BilayerShell::nEdges)
        .def("get_dofs", &BilayerShell::getDofs)
        .def("set_dofs", &BilayerShell::setDofs, py::arg("x"))
        .def("vertices", &BilayerShell::vertices)
        .def("rest_vertices", &BilayerShell::restVertices)
        .def("faces", &BilayerShell::faces)
        .def("energy", &BilayerShell::energy)
        .def("energy_and_gradient", &BilayerShell::energyAndGradient,
             "Returns (energy, gradient) with the gradient over the full DOF vector.")
        .def("get_abars", &BilayerShell::getAbars, py::arg("layer"))
        .def("set_abars", &BilayerShell::setAbars, py::arg("layer"), py::arg("abars"))
        .def("set_ortho_growth", &BilayerShell::setOrthoGrowth,
             py::arg("layer"), py::arg("angles"), py::arg("rate1"), py::arg("rate2"))
        .def("rigid_body_modes", &BilayerShell::rigidBodyModes,
             "The 6 orthonormalized rigid-body modes as an (n_dofs, 6) array, for deflation.");
}

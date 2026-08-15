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

#include <Eigen/Sparse>

#include <stdexcept>
#include <cmath>
#include <vector>

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

    /**
     * STAGE 2b: FULL per-face Saint-Venant energy (STRETCHING + BENDING) and its gradient
     * over ALL degrees of freedom, computed via TinyAD forward-mode autodiff.
     *
     * MONOLAYER only. This reproduces the production energy (SaintVenantEnergy<...,single>:
     * stretching_energy + bending_energy) independently, from the deformed vertex positions,
     * the edge directors, the per-face rest first fundamental form (abar) and rest second
     * fundamental form (bbar, = 0 for a flat plate). It does NOT touch the existing
     * energy/gradient code; it seeds an exact-Hessian effort.
     *
     * STRETCHING (see stretchingEnergyAndGradientTinyAD above for the derivation):
     *   e1 = v2-v1, e2 = v0-v2; firstFF = [[e1.e1,e1.e2],[e1.e2,e2.e2]]
     *   E_st = abar_inv*firstFF - I; E_face_aa = h_aa*(c1*tr^2 + c2*tr(E^2))*area, h_aa = h/4.
     *
     * BENDING (ExtendedTriangleInfo.hpp:229-243, EnergyHelper_Parametric.hpp):
     *   e0 = v1-v0 (CreateExtendedTriangleInfos.hpp:86).
     *   double_face_area = |e2 x e0| (== the un-normalized face-normal norm; equals the
     *       CreateExtendedTriangleInfos.hpp:122-135 cross-product magnitude since
     *       (v0-v2)x(v1-v2) = e2 x e0). face_normal = (e2 x e0)/|e2 x e0| (line 147).
     *   height(i) = double_face_area/|e_i| (line 157).
     *   For interior edge i, theta(i) is the dihedral angle. With n_own the own face normal
     *   and n_nbr the neighbour face normal (built from the neighbour's own vertex winding,
     *   normal = ((nv0-nv2)x(nv1-nv0)).normalized()), and edge-vector e_i (own orientation):
     *       signTheta = sign((n_own x n_nbr).e_i)
     *       theta(i)  = 2*signTheta*atan2(|n_own-n_nbr|, |n_own+n_nbr|).
     *   (This signTheta equals the CreateExtendedTriangleInfos.hpp:228-231 value regardless
     *   of which incident face is edge2faces(e,0), because swapping the two faces flips both
     *   the cross product and the edge orientation.) Boundary edges: theta(i)=0.
     *   alpha(i) = 0.5*theta(i) + sign(i)*phi(i), sign(i)=+1 if edge2faces(e_i,0)==this face
     *       else -1 (line 92); phi(i) the edge director of edge e_i.
     *   secondFF = [[2*(n0.e1 - n2.e1), -2*n0.e1],[-2*n0.e1, 2*(n1.e2 - n0.e2)]] with
     *       n2.e1=+h2*sin(a2), n0.e1=-h0*sin(a0), n0.e2=-n0.e1, n1.e2=-h1*sin(a1).
     *   Sb = abar_inv*(secondFF - bbar); E_face_bb = h_bb*(c1*tr(Sb)^2 + c2*tr(Sb^2))*area,
     *       h_bb = thickness^3/12 (single layer).
     *
     * Stencil: 21 active DOFs -- own v0,v1,v2 (9), the up-to-3 opposite vertices, one per
     * interior edge (9), and the 3 edge directors (3). Boundary-edge opposite-vertex slots
     * stay inactive. Clamped edges are unsupported (asserted off).
     *
     * Returns (total_energy, full_gradient) over the length-(3*nV+nE) DOF vector in the
     * column-major [x|y|z|phi] layout. Definitive test: this must match energy() and the
     * production analytic gradient energy_and_gradient()[1] over ALL dofs.
     */
    py::tuple energyAndGradientTinyAD()
    {
        requireMesh();
        mesh.updateDeformedConfiguration();

        const int nV = mesh.getNumberOfVertices();
        const int nF = mesh.getNumberOfFaces();
        const int nD = nDofs();

        const auto & topo = mesh.getTopology();
        const Eigen::MatrixXi F   = topo.getFace2Vertices();
        const Eigen::MatrixXi F2E = topo.getFace2Edges();
        const Eigen::MatrixXi E2F = topo.getEdge2Faces();

        const Eigen::MatrixXd V   = mesh.getCurrentConfiguration().getVertices();
        const Eigen::VectorXd phi = mesh.getCurrentConfiguration().getEdgeDirectors();
        const tVecMat2d & abars   = mesh.getRestConfiguration().getFirstFundamentalForms();

        const Real c1   = 0.5 * E * nu / (1.0 - nu * nu); // getStVenantFactor1()
        const Real c2   = 0.5 * E / (1.0 + nu);           // getStVenantFactor2()
        const Real h_aa = h / 4.0;                         // single-layer compute_h_aa
        const Real h_bb = h * h * h / 12.0;                // single-layer compute_h_bb

        Eigen::VectorXd grad = Eigen::VectorXd::Zero(nD);
        Real total = 0.0;

        // 21 active variables per face: [v0 v1 v2 | v_opp_e0 v_opp_e1 v_opp_e2 | phi0 phi1 phi2]
        using ADouble = TinyAD::Double<21, false>;
        typedef Eigen::Matrix<ADouble, 3, 1> Vec3A;

        for(int f = 0; f < nF; ++f)
        {
            const int i0 = F(f, 0), i1 = F(f, 1), i2 = F(f, 2);
            const int ownIdx[3] = {i0, i1, i2};

            bool interior[3];
            int  sgn[3];
            int  eIdx[3];
            int  oppGlobal[3];
            int  nbrMap[3][3];      // neighbour (nv0,nv1,nv2) -> 0/1/2 = own v0/v1/v2, 3 = opposite

            for(int i = 0; i < 3; ++i)
            {
                const int e = F2E(f, i);
                eIdx[i] = e;
                const int fA = E2F(e, 0), fB = E2F(e, 1);
                sgn[i] = (fA == f ? +1 : -1);
                oppGlobal[i] = -1;
                if(fA < 0 || fB < 0) { interior[i] = false; continue; }
                interior[i] = true;
                const int nbr = (fA == f ? fB : fA);
                const int nv[3] = {F(nbr, 0), F(nbr, 1), F(nbr, 2)};
                for(int k = 0; k < 3; ++k)
                {
                    const int gidx = nv[k];
                    if(gidx == i0)      nbrMap[i][k] = 0;
                    else if(gidx == i1) nbrMap[i][k] = 1;
                    else if(gidx == i2) nbrMap[i][k] = 2;
                    else { nbrMap[i][k] = 3; oppGlobal[i] = gidx; }
                }
            }

            Eigen::Matrix<double, 21, 1> x0;
            x0.setZero();
            x0.segment<3>(0) = V.row(i0).transpose();
            x0.segment<3>(3) = V.row(i1).transpose();
            x0.segment<3>(6) = V.row(i2).transpose();
            for(int i = 0; i < 3; ++i)
                if(interior[i]) x0.segment<3>(9 + 3 * i) = V.row(oppGlobal[i]).transpose();
            x0(18) = phi(eIdx[0]);
            x0(19) = phi(eIdx[1]);
            x0(20) = phi(eIdx[2]);

            const Eigen::Matrix<ADouble, 21, 1> x = ADouble::make_active(x0);

            const Vec3A v0 = x.segment<3>(0);
            const Vec3A v1 = x.segment<3>(3);
            const Vec3A v2 = x.segment<3>(6);

            const Vec3A e0 = v1 - v0;
            const Vec3A e1 = v2 - v1;
            const Vec3A e2 = v0 - v2;

            // --- geometry (own face) ---
            const Vec3A fn_unnorm = e2.cross(e0);
            const ADouble dbl_area = fn_unnorm.norm();
            const Vec3A n_own = fn_unnorm / dbl_area;

            const ADouble height0 = dbl_area / e0.norm();
            const ADouble height1 = dbl_area / e1.norm();
            const ADouble height2 = dbl_area / e2.norm();

            // --- material (shared abar) ---
            const Eigen::Matrix2d & abar = abars[f];
            const Eigen::Matrix2d abar_inv = abar.inverse();
            const double ai11 = abar_inv(0, 0);
            const double ai12 = abar_inv(0, 1);
            const double ai22 = abar_inv(1, 1);
            const double area = 0.5 * std::sqrt(abar.determinant());

            // --- stretching ---
            const ADouble aF11 = e1.dot(e1);
            const ADouble aF12 = e1.dot(e2);
            const ADouble aF22 = e2.dot(e2);

            const ADouble Est11 = ai11 * aF11 + ai12 * aF12 - 1.0;
            const ADouble Est12 = ai11 * aF12 + ai12 * aF22;
            const ADouble Est21 = ai12 * aF11 + ai22 * aF12;
            const ADouble Est22 = ai12 * aF12 + ai22 * aF22 - 1.0;

            const ADouble tr_st   = Est11 + Est22;
            const ADouble trsq_st = Est11 * Est11 + 2.0 * Est12 * Est21 + Est22 * Est22;
            const ADouble stretch_energy = h_aa * (c1 * tr_st * tr_st + c2 * trsq_st) * area;

            // --- bending: dihedral angles ---
            const Vec3A eLocal[3] = {e0, e1, e2};
            ADouble theta[3];
            for(int i = 0; i < 3; ++i)
            {
                if(!interior[i]) { theta[i] = ADouble(0.0); continue; }

                auto pick = [&](int k) -> Vec3A {
                    const int tag = nbrMap[i][k];
                    if(tag == 0) return v0;
                    if(tag == 1) return v1;
                    if(tag == 2) return v2;
                    return Vec3A(x.segment<3>(9 + 3 * i));
                };
                const Vec3A nb0 = pick(0);
                const Vec3A nb1 = pick(1);
                const Vec3A nb2 = pick(2);
                const Vec3A nbn_unnorm = (nb0 - nb2).cross(nb1 - nb0);
                const Vec3A n_nbr = nbn_unnorm / nbn_unnorm.norm();

                const double s = (n_own.cross(n_nbr)).dot(eLocal[i]).val > 0.0 ? 1.0 : -1.0;
                const ADouble num = (n_own - n_nbr).norm();
                const ADouble den = (n_own + n_nbr).norm();
                theta[i] = 2.0 * s * atan2(num, den);
            }

            const ADouble alpha0 = 0.5 * theta[0] + double(sgn[0]) * x(18);
            const ADouble alpha1 = 0.5 * theta[1] + double(sgn[1]) * x(19);
            const ADouble alpha2 = 0.5 * theta[2] + double(sgn[2]) * x(20);

            const ADouble n2_dot_e1 =  height2 * sin(alpha2);
            const ADouble n0_dot_e1 = -height0 * sin(alpha0);
            const ADouble n0_dot_e2 = -n0_dot_e1;
            const ADouble n1_dot_e2 = -height1 * sin(alpha1);

            const ADouble b11 =  2.0 * (n0_dot_e1 - n2_dot_e1);
            const ADouble b12 = -2.0 * n0_dot_e1;
            const ADouble b22 =  2.0 * (n1_dot_e2 - n0_dot_e2);

            // rest second fundamental form (0 for a flat plate); subtract as production does.
            const Eigen::Matrix2d bbar = mesh.getRestConfiguration().getSecondFundamentalForm(f);
            const double bb11 = bbar(0, 0), bb12 = bbar(0, 1), bb22 = bbar(1, 1);

            const ADouble d11 = b11 - bb11;
            const ADouble d12 = b12 - bb12;
            const ADouble d22 = b22 - bb22;

            // Sb = abar_inv * (secondFF - bbar)  (row-major 2x2, generally non-symmetric)
            const ADouble Sb11 = ai11 * d11 + ai12 * d12;
            const ADouble Sb12 = ai11 * d12 + ai12 * d22;
            const ADouble Sb21 = ai12 * d11 + ai22 * d12;
            const ADouble Sb22 = ai12 * d12 + ai22 * d22;

            const ADouble tr_b   = Sb11 + Sb22;
            const ADouble trsq_b = Sb11 * Sb11 + 2.0 * Sb12 * Sb21 + Sb22 * Sb22;
            const ADouble bend_energy = h_bb * (c1 * tr_b * tr_b + c2 * trsq_b) * area;

            const ADouble energy = stretch_energy + bend_energy;
            total += energy.val;

            const Eigen::Matrix<double, 21, 1> & g = energy.grad;
            for(int a = 0; a < 3; ++a)
            {
                grad(0 * nV + ownIdx[a]) += g(3 * a + 0);
                grad(1 * nV + ownIdx[a]) += g(3 * a + 1);
                grad(2 * nV + ownIdx[a]) += g(3 * a + 2);
            }
            for(int i = 0; i < 3; ++i)
                if(interior[i])
                {
                    grad(0 * nV + oppGlobal[i]) += g(9 + 3 * i + 0);
                    grad(1 * nV + oppGlobal[i]) += g(9 + 3 * i + 1);
                    grad(2 * nV + oppGlobal[i]) += g(9 + 3 * i + 2);
                }
            grad(3 * nV + eIdx[0]) += g(18);
            grad(3 * nV + eIdx[1]) += g(19);
            grad(3 * nV + eIdx[2]) += g(20);
        }

        return py::make_tuple(total, grad);
    }

private:
    /**
     * STAGE 3 helper: build the per-face TinyAD scalar (val + grad + Hessian) for face `f`,
     * using EXACTLY the same 21-DOF stencil, geometry and energy expression as
     * energyAndGradientTinyAD() above. Templated on the TinyAD scalar type so it can be
     * instantiated with the Hessian-carrying TinyAD::Double<21> (default with_hessian=true).
     *
     * On return, gidx[21] holds the global DOF index of each of the 21 local slots, in the
     * order [v0 v1 v2 | v_opp_e0 v_opp_e1 v_opp_e2 | phi0 phi1 phi2] with the global scatter
     *   vertex k  -> (0*nV+k, 1*nV+k, 2*nV+k)
     *   director e-> 3*nV+e
     * Boundary-edge opposite-vertex slots get gidx = -1 (inactive: theta=0 there, so those
     * slots never enter the energy and their grad/Hess rows are exactly zero anyway).
     */
    template<typename ADouble>
    ADouble computeFaceEnergyAD(
        const int f,
        const Eigen::MatrixXi & F, const Eigen::MatrixXi & F2E, const Eigen::MatrixXi & E2F,
        const Eigen::MatrixXd & V, const Eigen::VectorXd & phi, const tVecMat2d & abars,
        const int nV, int gidx[21])
    {
        typedef Eigen::Matrix<ADouble, 3, 1> Vec3A;

        const Real c1   = 0.5 * E * nu / (1.0 - nu * nu); // getStVenantFactor1()
        const Real c2   = 0.5 * E / (1.0 + nu);           // getStVenantFactor2()
        const Real h_aa = h / 4.0;                         // single-layer compute_h_aa
        const Real h_bb = h * h * h / 12.0;                // single-layer compute_h_bb

        const int i0 = F(f, 0), i1 = F(f, 1), i2 = F(f, 2);
        const int ownIdx[3] = {i0, i1, i2};

        bool interior[3];
        int  sgn[3];
        int  eIdx[3];
        int  oppGlobal[3];
        int  nbrMap[3][3];

        for(int i = 0; i < 3; ++i)
        {
            const int e = F2E(f, i);
            eIdx[i] = e;
            const int fA = E2F(e, 0), fB = E2F(e, 1);
            sgn[i] = (fA == f ? +1 : -1);
            oppGlobal[i] = -1;
            if(fA < 0 || fB < 0) { interior[i] = false; continue; }
            interior[i] = true;
            const int nbr = (fA == f ? fB : fA);
            const int nv[3] = {F(nbr, 0), F(nbr, 1), F(nbr, 2)};
            for(int k = 0; k < 3; ++k)
            {
                const int gg = nv[k];
                if(gg == i0)      nbrMap[i][k] = 0;
                else if(gg == i1) nbrMap[i][k] = 1;
                else if(gg == i2) nbrMap[i][k] = 2;
                else { nbrMap[i][k] = 3; oppGlobal[i] = gg; }
            }
        }

        // global scatter indices for the 21 local slots
        for(int a = 0; a < 3; ++a)
        {
            gidx[3 * a + 0] = 0 * nV + ownIdx[a];
            gidx[3 * a + 1] = 1 * nV + ownIdx[a];
            gidx[3 * a + 2] = 2 * nV + ownIdx[a];
        }
        for(int i = 0; i < 3; ++i)
        {
            if(interior[i])
            {
                gidx[9 + 3 * i + 0] = 0 * nV + oppGlobal[i];
                gidx[9 + 3 * i + 1] = 1 * nV + oppGlobal[i];
                gidx[9 + 3 * i + 2] = 2 * nV + oppGlobal[i];
            }
            else
            {
                gidx[9 + 3 * i + 0] = -1;
                gidx[9 + 3 * i + 1] = -1;
                gidx[9 + 3 * i + 2] = -1;
            }
        }
        gidx[18] = 3 * nV + eIdx[0];
        gidx[19] = 3 * nV + eIdx[1];
        gidx[20] = 3 * nV + eIdx[2];

        Eigen::Matrix<double, 21, 1> x0;
        x0.setZero();
        x0.segment<3>(0) = V.row(i0).transpose();
        x0.segment<3>(3) = V.row(i1).transpose();
        x0.segment<3>(6) = V.row(i2).transpose();
        for(int i = 0; i < 3; ++i)
            if(interior[i]) x0.segment<3>(9 + 3 * i) = V.row(oppGlobal[i]).transpose();
        x0(18) = phi(eIdx[0]);
        x0(19) = phi(eIdx[1]);
        x0(20) = phi(eIdx[2]);

        const Eigen::Matrix<ADouble, 21, 1> x = ADouble::make_active(x0);

        const Vec3A v0 = x.template segment<3>(0);
        const Vec3A v1 = x.template segment<3>(3);
        const Vec3A v2 = x.template segment<3>(6);

        const Vec3A e0 = v1 - v0;
        const Vec3A e1 = v2 - v1;
        const Vec3A e2 = v0 - v2;

        const Vec3A fn_unnorm = e2.cross(e0);
        const ADouble dbl_area = fn_unnorm.norm();
        const Vec3A n_own = fn_unnorm / dbl_area;

        const ADouble height0 = dbl_area / e0.norm();
        const ADouble height1 = dbl_area / e1.norm();
        const ADouble height2 = dbl_area / e2.norm();

        const Eigen::Matrix2d & abar = abars[f];
        const Eigen::Matrix2d abar_inv = abar.inverse();
        const double ai11 = abar_inv(0, 0);
        const double ai12 = abar_inv(0, 1);
        const double ai22 = abar_inv(1, 1);
        const double area = 0.5 * std::sqrt(abar.determinant());

        const ADouble aF11 = e1.dot(e1);
        const ADouble aF12 = e1.dot(e2);
        const ADouble aF22 = e2.dot(e2);

        const ADouble Est11 = ai11 * aF11 + ai12 * aF12 - 1.0;
        const ADouble Est12 = ai11 * aF12 + ai12 * aF22;
        const ADouble Est21 = ai12 * aF11 + ai22 * aF12;
        const ADouble Est22 = ai12 * aF12 + ai22 * aF22 - 1.0;

        const ADouble tr_st   = Est11 + Est22;
        const ADouble trsq_st = Est11 * Est11 + 2.0 * Est12 * Est21 + Est22 * Est22;
        const ADouble stretch_energy = h_aa * (c1 * tr_st * tr_st + c2 * trsq_st) * area;

        const Vec3A eLocal[3] = {e0, e1, e2};
        ADouble theta[3];
        for(int i = 0; i < 3; ++i)
        {
            if(!interior[i]) { theta[i] = ADouble(0.0); continue; }

            auto pick = [&](int k) -> Vec3A {
                const int tag = nbrMap[i][k];
                if(tag == 0) return v0;
                if(tag == 1) return v1;
                if(tag == 2) return v2;
                return Vec3A(x.template segment<3>(9 + 3 * i));
            };
            const Vec3A nb0 = pick(0);
            const Vec3A nb1 = pick(1);
            const Vec3A nb2 = pick(2);
            const Vec3A nbn_unnorm = (nb0 - nb2).cross(nb1 - nb0);
            const Vec3A n_nbr = nbn_unnorm / nbn_unnorm.norm();

            // Signed dihedral angle, SMOOTH form:  theta = atan2( (n_own x n_nbr).e_hat , n_own.n_nbr ).
            // This is mathematically identical to the 2b form 2*s*atan2(|n0-n1|,|n0+n1|) with
            // s = sign((n_own x n_nbr).e_i) -- both equal the signed dihedral in (-pi,pi) -- but
            // it avoids the 0/0 in d/dx ||n_own - n_nbr|| at the exactly flat (coplanar) state,
            // where the 2b form's frozen sign reintroduces a non-differentiable kink and TinyAD
            // returns NaN derivatives. At any curved state the two forms give identical value AND
            // gradient (verified: Stage-3 gates 1-3 compare against the production analytic gradient).
            const Vec3A e_hat = eLocal[i] / eLocal[i].norm();
            const ADouble sinComp = (n_own.cross(n_nbr)).dot(e_hat);
            const ADouble cosComp = n_own.dot(n_nbr);
            theta[i] = atan2(sinComp, cosComp);
        }

        const ADouble alpha0 = 0.5 * theta[0] + double(sgn[0]) * x(18);
        const ADouble alpha1 = 0.5 * theta[1] + double(sgn[1]) * x(19);
        const ADouble alpha2 = 0.5 * theta[2] + double(sgn[2]) * x(20);

        const ADouble n2_dot_e1 =  height2 * sin(alpha2);
        const ADouble n0_dot_e1 = -height0 * sin(alpha0);
        const ADouble n0_dot_e2 = -n0_dot_e1;
        const ADouble n1_dot_e2 = -height1 * sin(alpha1);

        const ADouble b11 =  2.0 * (n0_dot_e1 - n2_dot_e1);
        const ADouble b12 = -2.0 * n0_dot_e1;
        const ADouble b22 =  2.0 * (n1_dot_e2 - n0_dot_e2);

        const Eigen::Matrix2d bbar = mesh.getRestConfiguration().getSecondFundamentalForm(f);
        const double bb11 = bbar(0, 0), bb12 = bbar(0, 1), bb22 = bbar(1, 1);

        const ADouble d11 = b11 - bb11;
        const ADouble d12 = b12 - bb12;
        const ADouble d22 = b22 - bb22;

        const ADouble Sb11 = ai11 * d11 + ai12 * d12;
        const ADouble Sb12 = ai11 * d12 + ai12 * d22;
        const ADouble Sb21 = ai12 * d11 + ai22 * d12;
        const ADouble Sb22 = ai12 * d12 + ai22 * d22;

        const ADouble tr_b   = Sb11 + Sb22;
        const ADouble trsq_b = Sb11 * Sb11 + 2.0 * Sb12 * Sb21 + Sb22 * Sb22;
        const ADouble bend_energy = h_bb * (c1 * tr_b * tr_b + c2 * trsq_b) * area;

        return stretch_energy + bend_energy;
    }

public:
    /**
     * STAGE 3: exact Hessian-vector product H*v at the CURRENT dofs, assembled from the
     * per-face exact 21x21 Hessians that TinyAD produces in the same pass as the (verified)
     * energy and gradient. For each face: gather v at the 21 global slot indices, apply the
     * local 21x21 Hessian, scatter the result back. Returns the full-length (3*nV+nE) vector.
     */
    Eigen::VectorXd hessianVectorProductTinyAD(const Eigen::VectorXd & v)
    {
        requireMesh();
        mesh.updateDeformedConfiguration();

        const int nV = mesh.getNumberOfVertices();
        const int nF = mesh.getNumberOfFaces();
        const int nD = nDofs();
        if(v.size() != nD)
            throw std::invalid_argument("hessian_vector_product_tinyad: vector has the wrong length");

        const auto & topo = mesh.getTopology();
        const Eigen::MatrixXi F   = topo.getFace2Vertices();
        const Eigen::MatrixXi F2E = topo.getFace2Edges();
        const Eigen::MatrixXi E2F = topo.getEdge2Faces();
        const Eigen::MatrixXd V   = mesh.getCurrentConfiguration().getVertices();
        const Eigen::VectorXd phi = mesh.getCurrentConfiguration().getEdgeDirectors();
        const tVecMat2d & abars   = mesh.getRestConfiguration().getFirstFundamentalForms();

        using ADouble = TinyAD::Double<21>; // with_hessian = true

        Eigen::VectorXd out = Eigen::VectorXd::Zero(nD);

        for(int f = 0; f < nF; ++f)
        {
            int gidx[21];
            const ADouble energy = computeFaceEnergyAD<ADouble>(
                f, F, F2E, E2F, V, phi, abars, nV, gidx);

            Eigen::Matrix<double, 21, 1> v_local;
            v_local.setZero();
            for(int j = 0; j < 21; ++j)
                if(gidx[j] >= 0) v_local(j) = v(gidx[j]);

            const Eigen::Matrix<double, 21, 1> hv = energy.Hess * v_local;

            for(int j = 0; j < 21; ++j)
                if(gidx[j] >= 0) out(gidx[j]) += hv(j);
        }

        return out;
    }

    /**
     * STAGE 3: assemble the full sparse exact Hessian at the CURRENT dofs by scattering each
     * per-face 21x21 block (TinyAD) into global (row, col) triplets, then setFromTriplets.
     * pybind11/eigen maps Eigen::SparseMatrix<double> to scipy.sparse.csc automatically.
     */
    Eigen::SparseMatrix<double> hessianTinyAD()
    {
        requireMesh();
        mesh.updateDeformedConfiguration();

        const int nV = mesh.getNumberOfVertices();
        const int nF = mesh.getNumberOfFaces();
        const int nD = nDofs();

        const auto & topo = mesh.getTopology();
        const Eigen::MatrixXi F   = topo.getFace2Vertices();
        const Eigen::MatrixXi F2E = topo.getFace2Edges();
        const Eigen::MatrixXi E2F = topo.getEdge2Faces();
        const Eigen::MatrixXd V   = mesh.getCurrentConfiguration().getVertices();
        const Eigen::VectorXd phi = mesh.getCurrentConfiguration().getEdgeDirectors();
        const tVecMat2d & abars   = mesh.getRestConfiguration().getFirstFundamentalForms();

        using ADouble = TinyAD::Double<21>; // with_hessian = true

        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(nF * 21 * 21);

        for(int f = 0; f < nF; ++f)
        {
            int gidx[21];
            const ADouble energy = computeFaceEnergyAD<ADouble>(
                f, F, F2E, E2F, V, phi, abars, nV, gidx);

            const Eigen::Matrix<double, 21, 21> & Hloc = energy.Hess;
            for(int r = 0; r < 21; ++r)
            {
                if(gidx[r] < 0) continue;
                for(int c = 0; c < 21; ++c)
                {
                    if(gidx[c] < 0) continue;
                    trips.emplace_back(gidx[r], gidx[c], Hloc(r, c));
                }
            }
        }

        Eigen::SparseMatrix<double> H(nD, nD);
        H.setFromTriplets(trips.begin(), trips.end());
        return H;
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
        .def("energy_and_gradient_tinyad",
             &MonolayerShell::energyAndGradientTinyAD,
             "FULL Saint-Venant energy (stretching + bending) and its full-DOF gradient via "
             "TinyAD autodiff over all 21 per-face DOFs. Returns (energy, gradient); matches "
             "energy() and the analytic energy_and_gradient()[1] over all dofs. Monolayer only.")
        .def("hessian_vector_product_tinyad",
             &MonolayerShell::hessianVectorProductTinyAD, py::arg("v"),
             "Exact Hessian-vector product H*v at the current dofs, assembled from the per-face "
             "exact 21x21 TinyAD Hessians. Returns a length-(3*nV+nE) vector. Monolayer only.")
        .def("hessian_tinyad",
             &MonolayerShell::hessianTinyAD,
             "Exact full sparse Hessian (scipy.sparse.csc) at the current dofs, assembled from the "
             "per-face exact 21x21 TinyAD Hessian blocks. Monolayer only.")
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

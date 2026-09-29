#ifndef BENCHMARK_LOCAL_DIHEDRAL_CACHED_TINYAD_HESSIAN_BILAYER_HPP
#define BENCHMARK_LOCAL_DIHEDRAL_CACHED_TINYAD_HESSIAN_BILAYER_HPP

// Isolated benchmark candidate: local-dihedral AD and opt-in parallel face
// evaluation with deterministic serial CSC scatter. Per-face arithmetic and
// masking are retained. Never include this in production builds.

/**
 * TinyADHessian_Bilayer : the EXACT analytic Hessian of the bilayer SVK non-Euclidean shell
 * energy, via TinyAD forward-mode autodiff of a scalar-templated per-face energy.
 *
 * Ported from the pyshell/TinyAD work on the instrumentation branch, whose per-face energy was
 * verified to match this repo's CombinedOperator_Parametric<BilayerMesh, Material_Isotropic>
 * gradient to ~1e-15. The energy algebra, SVK coefficients, thickness prefactors, bilayer
 * mixed-coupling sign, 21-DOF stencil, DOF layout ([x|y|z|directors], column-major), and the
 * SMOOTH signed-dihedral form (finite second derivatives at the flat state) are all identical
 * to CombinedOperator_Parametric, so this is the exact Hessian of the energy the forming solve
 * minimizes -- not a finite-difference approximation (no noise floor).
 *
 * DOF layout: global index of vertex k coord d is d*nV + k; edge director e is 3*nV + e.
 * Per-face 21-DOF stencil: [v0 v1 v2 | opp(e0) opp(e1) opp(e2) | phi0 phi1 phi2].
 *
 * Only the isotropic material is supported -- the zigzag forming solve uses Material_Isotropic
 * (growth enters through abar, not the material), so that is all the Hessian needs.
 */

#include "common.hpp"
#include <TinyAD/Scalar.hh>
#include <Eigen/Sparse>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <array>
#include <algorithm>
#include <memory>
#include <limits>
#include <exception>
#ifdef _OPENMP
#include <omp.h>
#endif

template<typename tMesh>
class LocalDihedralCachedTinyADHessian_Bilayer
{
    const Real E, nu, h;
    struct Pattern
    {
        const tMesh* mesh=nullptr;
        Eigen::MatrixXi F,F2E,E2F;
        std::vector<char> bc;
        std::vector<std::array<int,21>> stencil;
        // CSC valuePtr offsets in face/r/c order; -1 denotes a masked DOF.
        std::vector<std::array<int,21*21>> offsets;
        std::vector<int> bcDiagonalOffsets;
        Eigen::SparseMatrix<double> structure;
    };
    std::unique_ptr<Pattern> pattern_;
    int cacheBuilds_=0;
    const int assemblyThreads_;
    int lastAssemblyThreads_=1;
    // Parallel-only scratch: 441 doubles/face (~35.5 MiB for 10560 faces).
    // This object, like its mesh/pattern cache, is not externally reentrant.
    std::vector<std::array<double,21*21>> faceHessians_;

    // ---- per-face energy, scalar-templated for TinyAD. Verbatim from the verified pyshell
    //      builder; material constants and bbar are passed in rather than read from a member. ----
    template<typename ADouble>
    ADouble computeFaceEnergyAD(
        const int f,
        const Eigen::MatrixXi & F, const Eigen::MatrixXi & F2E, const Eigen::MatrixXi & E2F,
        const Eigen::MatrixXd & V, const Eigen::VectorXd & phi,
        const tVecMat2d & abarsBot, const tVecMat2d & abarsTop,
        const Eigen::Matrix2d & bbar,
        const int nV, int gidx[21]) const
    {
        typedef Eigen::Matrix<ADouble, 3, 1> Vec3A;

        const Real c1   = 0.5 * E * nu / (1.0 - nu * nu);   // getStVenantFactor1()
        const Real c2   = 0.5 * E / (1.0 + nu);             // getStVenantFactor2()
        const Real h_aa = 0.5 * h / 4.0;                     // bilayer compute_h_aa (half)
        const Real h_bb = 0.5 * h * h * h / 12.0;            // bilayer compute_h_bb (half)
        const Real h_ab_bot = +h * h / 8.0;                  // bilayer compute_h_ab (bottom)
        const Real h_ab_top = -h * h / 8.0;                  // bilayer compute_h_ab (top)

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

        const ADouble aF11 = e1.dot(e1);
        const ADouble aF12 = e1.dot(e2);
        const ADouble aF22 = e2.dot(e2);

        // Isolated diagnostic: each signed dihedral sees the own face's 9
        // coordinates and one neighbor's opposite vertex (3), not all 21.
        // Lift its exact local derivatives back to 21 before the unchanged
        // bending/stretch-bend arithmetic. The duplicate own-normal work may
        // outweigh the reduction in AD width; timing must establish a gain.
        static_assert(ADouble::k_ == 21 && !ADouble::truncated_hessian_);
        using AD12 = TinyAD::Double<12, ADouble::with_hessian_>;
        using Vec3S = Eigen::Matrix<AD12,3,1>;
        ADouble theta[3];
        for(int i = 0; i < 3; ++i)
        {
            if(!interior[i]) { theta[i] = ADouble(0.0); continue; }
            Eigen::Matrix<double,12,1> local;
            local.template head<9>() = x0.template head<9>();
            local.template tail<3>() = x0.template segment<3>(9+3*i);
            const Eigen::Matrix<AD12,12,1> z = AD12::make_active(local);
            const Vec3S w0=z.template segment<3>(0),w1=z.template segment<3>(3),w2=z.template segment<3>(6);
            const Vec3S a0=w1-w0,a1=w2-w1,a2=w0-w2;
            const Vec3S ownUnnorm=a2.cross(a0);
            const Vec3S ownNormal=ownUnnorm/ownUnnorm.norm();
            auto pick = [&](int k) -> Vec3S {
                const int tag=nbrMap[i][k];
                if(tag==0)return w0;
                if(tag==1)return w1;
                if(tag==2)return w2;
                return Vec3S(z.template tail<3>());
            };
            const Vec3S nb0=pick(0),nb1=pick(1),nb2=pick(2);
            const Vec3S neighborUnnorm=(nb0-nb2).cross(nb1-nb0);
            const Vec3S neighborNormal=neighborUnnorm/neighborUnnorm.norm();
            const Vec3S ownEdges[3]={a0,a1,a2};
            const Vec3S edgeHat=ownEdges[i]/ownEdges[i].norm();
            const AD12 sine=(ownNormal.cross(neighborNormal)).dot(edgeHat);
            const AD12 cosine=ownNormal.dot(neighborNormal);
            const AD12 angle=atan2(sine,cosine);
            typename ADouble::GradType grad=ADouble::GradType::Zero();
            grad.template head<9>()=angle.grad.template head<9>();
            grad.template segment<3>(9+3*i)=angle.grad.template tail<3>();
            typename ADouble::HessType hess=ADouble::HessType::Zero();
            if constexpr (ADouble::with_hessian_)
            {
                hess.template topLeftCorner<9,9>()=angle.Hess.template topLeftCorner<9,9>();
                hess.template block<9,3>(0,9+3*i)=angle.Hess.template block<9,3>(0,9);
                hess.template block<3,9>(9+3*i,0)=angle.Hess.template block<3,9>(9,0);
                hess.template block<3,3>(9+3*i,9+3*i)=angle.Hess.template bottomRightCorner<3,3>();
            }
            theta[i]=ADouble::known_derivatives(angle.val,grad,hess);
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

        const double bb11 = bbar(0, 0), bb12 = bbar(0, 1), bb22 = bbar(1, 1);

        const ADouble d11 = b11 - bb11;
        const ADouble d12 = b12 - bb12;
        const ADouble d22 = b22 - bb22;

        auto layerEnergy = [&](const Eigen::Matrix2d & abarL, const Real h_ab_L) -> ADouble
        {
            const Eigen::Matrix2d abar_inv = abarL.inverse();
            const double ai11 = abar_inv(0, 0);
            const double ai12 = abar_inv(0, 1);
            const double ai22 = abar_inv(1, 1);
            const double area = 0.5 * std::sqrt(abarL.determinant());

            const ADouble Ea11 = ai11 * aF11 + ai12 * aF12 - 1.0;
            const ADouble Ea12 = ai11 * aF12 + ai12 * aF22;
            const ADouble Ea21 = ai12 * aF11 + ai22 * aF12;
            const ADouble Ea22 = ai12 * aF12 + ai22 * aF22 - 1.0;

            const ADouble Eb11 = ai11 * d11 + ai12 * d12;
            const ADouble Eb12 = ai11 * d12 + ai12 * d22;
            const ADouble Eb21 = ai12 * d11 + ai22 * d12;
            const ADouble Eb22 = ai12 * d12 + ai22 * d22;

            const ADouble tr_a  = Ea11 + Ea22;
            const ADouble tr_b  = Eb11 + Eb22;
            const ADouble trsq_a = Ea11 * Ea11 + 2.0 * Ea12 * Ea21 + Ea22 * Ea22;
            const ADouble trsq_b = Eb11 * Eb11 + 2.0 * Eb12 * Eb21 + Eb22 * Eb22;
            const ADouble tr_ab = Ea11 * Eb11 + Ea12 * Eb21 + Ea21 * Eb12 + Ea22 * Eb22;

            const ADouble stretch = h_aa   * (c1 * tr_a * tr_a + c2 * trsq_a) * area;
            const ADouble bend    = h_bb   * (c1 * tr_b * tr_b + c2 * trsq_b) * area;
            const ADouble mixed   = h_ab_L * (c1 * tr_a * tr_b + c2 * tr_ab)  * area;
            return stretch + bend + mixed;
        };

        const ADouble e_bot = layerEnergy(abarsBot[f], h_ab_bot);
        const ADouble e_top = layerEnergy(abarsTop[f], h_ab_top);
        return e_bot + e_top;
    }

    std::vector<char> constrainedDofMask(const tMesh & mesh) const
    {
        const int nV = mesh.getNumberOfVertices();
        const int nE = mesh.getNumberOfEdges();
        const int nD = 3 * nV + nE;
        std::vector<char> mask(nD, 0);

        const auto & bc = mesh.getBoundaryConditions();
        const auto vertices_bc = bc.getVertexBoundaryConditions();
        const auto edges_bc    = bc.getEdgeBoundaryConditions();

        if(vertices_bc.rows() == nV && vertices_bc.cols() == 3)
            for(int i = 0; i < nV; ++i)
                for(int j = 0; j < 3; ++j)
                    if(vertices_bc(i, j)) mask[j * nV + i] = 1;

        if(edges_bc.size() == nE)
            for(int e = 0; e < nE; ++e)
                if(edges_bc(e)) mask[3 * nV + e] = 1;

        return mask;
    }

public:
    LocalDihedralCachedTinyADHessian_Bilayer(const Real E_, const Real nu_, const Real h_,
                                          const int assemblyThreads=1)
    : E(E_), nu(nu_), h(h_), assemblyThreads_(assemblyThreads)
    {
        if(assemblyThreads<1)throw std::invalid_argument("Hessian assembly threads must be positive");
#ifndef _OPENMP
        if(assemblyThreads!=1)throw std::invalid_argument("parallel Hessian requires OpenMP");
#endif
    }
    int cacheBuilds() const {return cacheBuilds_;}
    int assemblyThreads() const {return assemblyThreads_;}
    int lastAssemblyThreads() const {return lastAssemblyThreads_;}

    // Diagnostic oracle value: sum the same face kernel used by the Hessian.
    double energy(tMesh & mesh) const
    {
        mesh.updateDeformedConfiguration();
        const int nv=mesh.getNumberOfVertices(),nf=mesh.getNumberOfFaces();
        const auto& topo=mesh.getTopology();
        const Eigen::MatrixXi F=topo.getFace2Vertices(),F2E=topo.getFace2Edges(),E2F=topo.getEdge2Faces();
        const Eigen::MatrixXd V=mesh.getCurrentConfiguration().getVertices();
        const Eigen::VectorXd phi=mesh.getCurrentConfiguration().getEdgeDirectors();
        const tVecMat2d& abB=mesh.getRestConfiguration().template getFirstFundamentalForms<bottom>();
        const tVecMat2d& abT=mesh.getRestConfiguration().template getFirstFundamentalForms<top>();
        using ADg=TinyAD::Double<21,false>;
        double sum=0;
        for(int f=0;f<nf;++f)
        {
            int gidx[21];
            const auto bbar=mesh.getRestConfiguration().getSecondFundamentalForm(f);
            sum+=computeFaceEnergyAD<ADg>(f,F,F2E,E2F,V,phi,abB,abT,bbar,nv,gidx).val;
        }
        return sum;
    }

    int nDofs(const tMesh & mesh) const
    { return 3 * mesh.getNumberOfVertices() + mesh.getNumberOfEdges(); }

    // ---- full gradient over all DOFs (for the machine-precision verification against
    //      CombinedOperator_Parametric) ----
    Eigen::VectorXd gradient(tMesh & mesh) const
    {
        mesh.updateDeformedConfiguration();
        const int nV = mesh.getNumberOfVertices();
        const int nF = mesh.getNumberOfFaces();
        const int nD = nDofs(mesh);
        const auto & topo = mesh.getTopology();
        const Eigen::MatrixXi F   = topo.getFace2Vertices();
        const Eigen::MatrixXi F2E = topo.getFace2Edges();
        const Eigen::MatrixXi E2F = topo.getEdge2Faces();
        const Eigen::MatrixXd V   = mesh.getCurrentConfiguration().getVertices();
        const Eigen::VectorXd phi = mesh.getCurrentConfiguration().getEdgeDirectors();
        const tVecMat2d & abB = mesh.getRestConfiguration().template getFirstFundamentalForms<bottom>();
        const tVecMat2d & abT = mesh.getRestConfiguration().template getFirstFundamentalForms<top>();
        const std::vector<char> bcMask = constrainedDofMask(mesh);

        using ADg = TinyAD::Double<21, false>;   // gradient only, no Hessian
        Eigen::VectorXd g = Eigen::VectorXd::Zero(nD);
        for(int f = 0; f < nF; ++f)
        {
            int gidx[21];
            const Eigen::Matrix2d bbar = mesh.getRestConfiguration().getSecondFundamentalForm(f);
            const ADg e = computeFaceEnergyAD<ADg>(f, F, F2E, E2F, V, phi, abB, abT, bbar, nV, gidx);
            for(int j = 0; j < 21; ++j)
                if(gidx[j] >= 0 && !bcMask[gidx[j]]) g(gidx[j]) += e.grad(j);
        }
        return g;
    }

    // ---- matrix-free exact Hessian-vector product ----
    Eigen::VectorXd hessianVectorProduct(tMesh & mesh, const Eigen::VectorXd & v) const
    {
        mesh.updateDeformedConfiguration();
        const int nV = mesh.getNumberOfVertices();
        const int nF = mesh.getNumberOfFaces();
        const int nD = nDofs(mesh);
        if(v.size() != nD)
            throw std::invalid_argument("TinyADHessian_Bilayer::hessianVectorProduct: wrong length");
        const auto & topo = mesh.getTopology();
        const Eigen::MatrixXi F   = topo.getFace2Vertices();
        const Eigen::MatrixXi F2E = topo.getFace2Edges();
        const Eigen::MatrixXi E2F = topo.getEdge2Faces();
        const Eigen::MatrixXd V   = mesh.getCurrentConfiguration().getVertices();
        const Eigen::VectorXd phi = mesh.getCurrentConfiguration().getEdgeDirectors();
        const tVecMat2d & abB = mesh.getRestConfiguration().template getFirstFundamentalForms<bottom>();
        const tVecMat2d & abT = mesh.getRestConfiguration().template getFirstFundamentalForms<top>();
        const std::vector<char> bcMask = constrainedDofMask(mesh);

        using ADouble = TinyAD::Double<21>;
        Eigen::VectorXd out = Eigen::VectorXd::Zero(nD);

        if(assemblyThreads_ == 1)
        {
            for(int f = 0; f < nF; ++f)
            {
                int gidx[21];
                const Eigen::Matrix2d bbar = mesh.getRestConfiguration().getSecondFundamentalForm(f);
                const ADouble e = computeFaceEnergyAD<ADouble>(f, F, F2E, E2F, V, phi, abB, abT, bbar, nV, gidx);
                Eigen::Matrix<double, 21, 1> v_local; v_local.setZero();
                for(int j = 0; j < 21; ++j)
                    if(gidx[j] >= 0 && !bcMask[gidx[j]]) v_local(j) = v(gidx[j]);
                const Eigen::Matrix<double, 21, 1> hv = e.Hess * v_local;
                for(int j = 0; j < 21; ++j)
                    if(gidx[j] >= 0 && !bcMask[gidx[j]]) out(gidx[j]) += hv(j);
            }
        }
#ifdef _OPENMP
        else
        {
            const tVecMat2d bbars = mesh.getRestConfiguration().getSecondFundamentalForms();
            std::vector<std::array<double, 21>> faceHv(nF);
            std::vector<std::array<int, 21>> faceGidx(nF);
            std::vector<std::exception_ptr> errors(nF);
#pragma omp parallel num_threads(assemblyThreads_)
            {
#pragma omp for schedule(static)
                for(int f = 0; f < nF; ++f)
                {
                    try
                    {
                        int gidx[21];
                        const ADouble e = computeFaceEnergyAD<ADouble>(f, F, F2E, E2F, V, phi, abB, abT, bbars[f], nV, gidx);
                        Eigen::Matrix<double, 21, 1> v_local; v_local.setZero();
                        for(int j = 0; j < 21; ++j)
                        {
                            faceGidx[f][j] = gidx[j];
                            if(gidx[j] >= 0 && !bcMask[gidx[j]]) v_local(j) = v(gidx[j]);
                        }
                        const Eigen::Matrix<double, 21, 1> hv = e.Hess * v_local;
                        for(int j = 0; j < 21; ++j) faceHv[f][j] = hv(j);
                    }
                    catch(...) { errors[f] = std::current_exception(); }
                }
            }
            for(const auto & error : errors) if(error) std::rethrow_exception(error);
            for(int f = 0; f < nF; ++f)
                for(int j = 0; j < 21; ++j)
                {
                    const int idx = faceGidx[f][j];
                    if(idx >= 0 && !bcMask[idx]) out(idx) += faceHv[f][j];
                }
        }
#endif
        for(int i = 0; i < nD; ++i)
            if(bcMask[i]) out(i) = v(i);
        return out;
    }

private:
    void initializePattern(tMesh& mesh,const Eigen::MatrixXi& F,
                           const Eigen::MatrixXi& F2E,const Eigen::MatrixXi& E2F,
                           const std::vector<char>& mask)
    {
        auto next=std::make_unique<Pattern>();
        next->mesh=&mesh;next->F=F;next->F2E=F2E;next->E2F=E2F;next->bc=mask;
        const int nv=mesh.getNumberOfVertices(),nd=nDofs(mesh),nf=mesh.getNumberOfFaces();
        next->stencil.resize(nf);
        std::vector<Eigen::Triplet<double>> trips;
        trips.reserve(std::size_t(nf)*21*21);
        for(int f=0;f<nf;++f)
        {
            auto& g=next->stencil[f];
            const int own[3]={F(f,0),F(f,1),F(f,2)};
            for(int a=0;a<3;++a)for(int d=0;d<3;++d)g[3*a+d]=d*nv+own[a];
            for(int i=0;i<3;++i)
            {
                const int edge=F2E(f,i),fa=E2F(edge,0),fb=E2F(edge,1);
                int opp=-1;
                if(fa>=0 && fb>=0)
                {
                    const int other=fa==f?fb:fa;
                    for(int k=0;k<3;++k)
                    {
                        const int v=F(other,k);
                        if(v!=own[0] && v!=own[1] && v!=own[2])opp=v;
                    }
                    if(opp<0)throw std::invalid_argument("invalid face neighbor in cached Hessian");
                }
                for(int d=0;d<3;++d)g[9+3*i+d]=opp<0?-1:d*nv+opp;
                g[18+i]=3*nv+edge;
            }
            for(int r=0;r<21;++r)
            {
                if(g[r]<0 || mask[g[r]])continue;
                for(int c=0;c<21;++c)
                    if(g[c]>=0 && !mask[g[c]])trips.emplace_back(g[r],g[c],1.0);
            }
        }
        for(int i=0;i<nd;++i)if(mask[i])trips.emplace_back(i,i,1.0);
        next->structure.resize(nd,nd);
        next->structure.setFromTriplets(trips.begin(),trips.end());
        next->structure.makeCompressed();
        const int* outer=next->structure.outerIndexPtr();
        const int* inner=next->structure.innerIndexPtr();
        auto offset=[&](int row,int col){
            const int* p=std::lower_bound(inner+outer[col],inner+outer[col+1],row);
            if(p==inner+outer[col+1] || *p!=row)
                throw std::logic_error("cached Hessian pattern lookup failed");
            return int(p-inner);
        };
        next->offsets.resize(nf);
        for(int f=0;f<nf;++f)
        {
            auto& dest=next->offsets[f];dest.fill(-1);
            const auto& g=next->stencil[f];
            for(int r=0;r<21;++r)
            {
                if(g[r]<0 || mask[g[r]])continue;
                for(int c=0;c<21;++c)
                    if(g[c]>=0 && !mask[g[c]])dest[21*r+c]=offset(g[r],g[c]);
            }
        }
        for(int i=0;i<nd;++i)if(mask[i])next->bcDiagonalOffsets.push_back(offset(i,i));
        pattern_=std::move(next);++cacheBuilds_;
    }

public:
    // ---- full sparse exact Hessian; parallel evaluation, ordered CSC scatter ----
    Eigen::SparseMatrix<double> assembleHessian(tMesh& mesh)
    {
        mesh.updateDeformedConfiguration();
        const int nv=mesh.getNumberOfVertices(),nf=mesh.getNumberOfFaces();
        const auto& topo=mesh.getTopology();
        const Eigen::MatrixXi F=topo.getFace2Vertices(),F2E=topo.getFace2Edges(),E2F=topo.getEdge2Faces();
        const std::vector<char> mask=constrainedDofMask(mesh);
        if(!pattern_)initializePattern(mesh,F,F2E,E2F,mask);
        const auto& cache=*pattern_;
        const auto same=[](const Eigen::MatrixXi& a,const Eigen::MatrixXi& b){
            return a.rows()==b.rows() && a.cols()==b.cols() && (a.array()==b.array()).all();
        };
        if(cache.mesh!=&mesh || cache.structure.rows()!=nDofs(mesh) ||
           cache.stencil.size()!=std::size_t(nf) || cache.bc!=mask ||
           !same(cache.F,F) || !same(cache.F2E,F2E) || !same(cache.E2F,E2F))
            throw std::invalid_argument("cached Hessian topology or constraint mask changed");
        const Eigen::MatrixXd V=mesh.getCurrentConfiguration().getVertices();
        const Eigen::VectorXd phi=mesh.getCurrentConfiguration().getEdgeDirectors();
        const tVecMat2d& abB=mesh.getRestConfiguration().template getFirstFundamentalForms<bottom>();
        const tVecMat2d& abT=mesh.getRestConfiguration().template getFirstFundamentalForms<top>();
        Eigen::SparseMatrix<double> H=cache.structure;
        std::fill_n(H.valuePtr(),H.nonZeros(),0.0);
        using ADouble=TinyAD::Double<21>;
        lastAssemblyThreads_=1;
        if(assemblyThreads_==1)
        {
            // Preserve the original serial path, without parallel scratch allocation.
            for(int f=0;f<nf;++f)
            {
                int gidx[21];
                const Eigen::Matrix2d bbar=mesh.getRestConfiguration().getSecondFundamentalForm(f);
                const ADouble e=computeFaceEnergyAD<ADouble>(f,F,F2E,E2F,V,phi,abB,abT,bbar,nv,gidx);
                const auto& positions=cache.offsets[f];
                for(int r=0;r<21;++r)
                {
                    if(gidx[r]!=cache.stencil[f][r])throw std::logic_error("cached Hessian stencil mismatch");
                    for(int c=0;c<21;++c)
                    {
                        const int k=positions[21*r+c];
                        if(k>=0)H.valuePtr()[k]+=e.Hess(r,c);
                    }
                }
            }
        }
#ifdef _OPENMP
        else
        {
            // No mesh access or global sparse writes inside the worker loop.
            const tVecMat2d bbars=mesh.getRestConfiguration().getSecondFundamentalForms();
            faceHessians_.resize(nf);
            std::vector<std::exception_ptr> errors(nf);
#pragma omp parallel num_threads(assemblyThreads_)
            {
#pragma omp single
                lastAssemblyThreads_=omp_get_num_threads();
#pragma omp for schedule(static)
                for(int f=0;f<nf;++f)
                {
                    // Exceptions cannot propagate across an OpenMP region. Report
                    // the first failing face in deterministic order after joining.
                    try
                    {
                        int gidx[21];
                        const ADouble e=computeFaceEnergyAD<ADouble>(f,F,F2E,E2F,V,phi,abB,abT,bbars[f],nv,gidx);
                        for(int r=0;r<21;++r)
                        {
                            if(gidx[r]!=cache.stencil[f][r])throw std::logic_error("cached Hessian stencil mismatch");
                            for(int c=0;c<21;++c)faceHessians_[f][21*r+c]=e.Hess(r,c);
                        }
                    }
                    catch(...){errors[f]=std::current_exception();}
                }
            }
            for(const auto& error:errors)if(error)std::rethrow_exception(error);
            // Exactly the serial face/r/c addition order: no atomics or reduction
            // tree, so parallelism does not change global summation order.
            for(int f=0;f<nf;++f)
                for(int j=0;j<21*21;++j)
                {
                    const int k=cache.offsets[f][j];
                    if(k>=0)H.valuePtr()[k]+=faceHessians_[f][j];
                }
        }
#endif
        for(int k:cache.bcDiagonalOffsets)H.valuePtr()[k]=1.0;
        return H;
    }
};

#endif

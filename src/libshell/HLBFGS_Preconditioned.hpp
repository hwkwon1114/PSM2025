#ifndef HLBFGS_PRECONDITIONED_HPP
#define HLBFGS_PRECONDITIONED_HPP

/**
 * HLBFGS_EnergyPrecond : HLBFGS preconditioned by the EXACT TinyAD Hessian via the library's
 * built-in ICFS (incomplete-Cholesky) sparse-Hessian mode.
 *
 * Plain HLBFGS is first-order and its iteration count grows super-linearly with mesh
 * resolution on the biharmonic shell (the ill-conditioning). Providing the exact Hessian as
 * a preconditioner -- rebuilt only every T iterations, so the amortized cost is small --
 * keeps HLBFGS's cheap gradient steps but conditions them with genuine second-order
 * curvature, which is the standard fix for the mesh-scaling.
 *
 * HLBFGS's Hessian mode (INFO[7]=1) calls an EVALFUNC_H callback every INFO[6]=T iterations;
 * it must fill f, g AND the HESSIAN_MATRIX. HESSIAN_MATRIX feeds the Fortran ICFS routine
 * (dicfs_), which wants the STRICT LOWER triangle in compressed-column form with 1-BASED
 * (Fortran) indices, and the diagonal in a separate array. dicfs handles indefiniteness
 * (the rigid-mode nullspace) internally via a diagonal shift, so no gauge fixing is needed.
 */

#include "HLBFGS_Wrapper.hpp"          // HLBFGS_Wrapper / HLBFGS_Energy + the static trampolines
#include "TinyADHessian_Bilayer.hpp"
#include <Eigen/Sparse>
#include <vector>
#include <string>

namespace HLBFGS_Methods
{
    template<typename tMesh, typename tMeshOperator, bool verbose>
    class HLBFGS_EnergyPrecond : public HLBFGS_Energy<tMesh, tMeshOperator, verbose>
    {
        typedef HLBFGS_Energy<tMesh, tMeshOperator, verbose> Base;

        const TinyADHessian_Bilayer<tMesh> & tad;

        // backing storage for the CSC (strict-lower, 1-based) Hessian handed to ICFS; refilled
        // on each Hessian-update call and kept alive for the duration of the HLBFGS solve.
        std::vector<double> Hvals, Hdiag;
        std::vector<int>    Hrow, Hcol;

    public:
        HLBFGS_EnergyPrecond(tMesh & mesh_in, const tMeshOperator & op_in,
                             const TinyADHessian_Bilayer<tMesh> & tad_in)
        : Base(mesh_in, op_in), tad(tad_in) {}

        // EVALFUNC_H: compute energy + gradient (as evaluate does) AND assemble the exact
        // Hessian into ICFS's strict-lower CSC (1-based) + diagonal format.
        void evaluateHessian(int N, double * /*x*/, double * /*prev_x*/,
                             double * f, double * g, HESSIAN_MATRIX & hm)
        {
            this->mesh.updateDeformedConfiguration();
            Eigen::Map<Eigen::VectorXd> eg(g, this->nVariables);
            eg.setZero();
            *f = this->op.compute(this->mesh, eg);

            Eigen::SparseMatrix<double> H = tad.assembleHessian(this->mesh);
            H.makeCompressed();

            const int n = N;
            Hdiag.assign(n, 0.0);
            Hcol.assign(n + 1, 0);
            Hvals.clear();
            Hrow.clear();
            Hvals.reserve(H.nonZeros());
            Hrow.reserve(H.nonZeros());

            int ptr = 1;                                   // 1-based (Fortran) running index
            for(int j = 0; j < n; ++j)
            {
                Hcol[j] = ptr;
                for(Eigen::SparseMatrix<double>::InnerIterator it(H, j); it; ++it)
                {
                    const int i = it.row();
                    if(i == j)      Hdiag[j] = it.value();                  // diagonal, separate
                    else if(i > j)  { Hvals.push_back(it.value());          // strict lower only
                                      Hrow.push_back(i + 1); ++ptr; }       // 1-based row index
                }
                if(Hdiag[j] == 0.0) Hdiag[j] = 1.0;        // guard: keep the diagonal nonzero for ICFS
            }
            Hcol[n] = ptr;

            hm.set_dimension(n);
            hm.set_nonzeros((int)Hvals.size());
            hm.set_values(Hvals.data());
            hm.set_rowind(Hrow.data());
            hm.set_colptr(Hcol.data());
            hm.set_diag(Hdiag.data());
        }

        static void s_evaluate_h(void * instance, int N, double * x, double * prev_x,
                                 double * f, double * g, HESSIAN_MATRIX & hm)
        {
            reinterpret_cast<HLBFGS_EnergyPrecond*>(instance)
                ->evaluateHessian(N, x, prev_x, f, g, hm);
        }

        // Preconditioned minimize: INFO[7]=1 (Hessian mode), INFO[6]=T (rebuild interval).
        int minimizePrecond(const std::string & outname, const Real eps, const int T = 5,
                            const int Mval = 10)
        {
            this->outFileName = outname;
            this->mesh.updateDeformedConfiguration();
            const Real energy0 = this->op.compute(this->mesh);
            Real * x = this->mesh.getDataPointer();

            double parameter[20];
            int info[20];
            this->default_setup(parameter, info, eps, verbose);
            info[6] = T;      // update the Hessian preconditioner every T iterations
            info[7] = 1;      // enable exact-Hessian (ICFS) preconditioning
            info[8] = 15;     // ICFS fill parameter p

            const int ret = HLBFGS(this->nVariables, Mval, x,
                                   HLBFGS_Methods::evaluate, s_evaluate_h,
                                   HLBFGS_UPDATE_Hessian, HLBFGS_Methods::newiteration,
                                   this, parameter, info);

            this->last_retcode = ret;
            this->last_iterations = info[2];
            const Real energy1 = this->op.compute(this->mesh);
            if(verbose)
                printf("[HLBFGS-precond] energy %.10e -> %.10e, iters=%d, |g|=%.6e (ret=%d)\n",
                       energy0, energy1, info[2], this->get_lastnorm(), ret);
            return (ret == 2 || ret == 3 || ret == 4) ? 0 : 1;
        }
    };
}

#endif

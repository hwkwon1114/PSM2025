#ifndef NEWTONCG_ENERGY_HPP
#define NEWTONCG_ENERGY_HPP

#include "common.hpp"   // Real (typedef double) + Eigen
#include <Eigen/Core>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

/**
 * NewtonCG_Energy : a matrix-free, curvature-aware energy minimizer.
 *
 * Ported from the Python solver-robustness work (trust-ncg + finite-difference
 * Hessian-vector products). Where HLBFGS is a Hessian-free quasi-Newton method that
 * converges linearly -- and grinds for 10^4 iterations to a tight tolerance on the
 * ill-conditioned biharmonic shell -- this uses genuine second-order curvature:
 *
 *   * The Hessian is never assembled. Each Hessian-vector product is a central
 *     difference of the ANALYTIC gradient the energy operator already provides:
 *         H v  ~=  ( grad(x + eps v) - grad(x - eps v) ) / (2 eps).
 *     So no new Hessian code is needed -- it reuses op.compute(mesh, grad).
 *   * Each Newton step solves the damped system (H + lambda I) p = -g by conjugate
 *     gradients. The lambda I (Levenberg-Marquardt) damping keeps the operator positive
 *     definite through the shell's 6 rigid-body nullspace directions, so no explicit
 *     rigid-mode projection is required, and it regularizes indefinite curvature away
 *     from a minimum. lambda is adapted from step success (down on accept, up on reject),
 *     giving Newton-like steps near the solution and gradient-descent-like steps far away.
 *
 * Interface mirrors HLBFGS_Energy: constructed from (mesh, op); the DOF vector is
 * mesh.getDataPointer() of length op.getNumberOfVariables(mesh); energy+gradient come from
 * op.compute(mesh, grad) after mesh.updateDeformedConfiguration().
 */
namespace NewtonCG_Methods
{
    template<typename tMesh, typename tMeshOperator, bool verbose = true>
    class NewtonCG_Energy
    {
    protected:
        tMesh & mesh;
        const tMeshOperator & op;
        const int n;

        Real last_gnorm;
        int last_iterations;

        // scratch buffers (sized once)
        Eigen::VectorXd gbuf, gp, gm, xtmp;

        // ---- write the DOF vector into the mesh and refresh geometry ----
        void set_x(const Eigen::VectorXd & xv)
        {
            Real * x = mesh.getDataPointer();
            Eigen::Map<Eigen::VectorXd>(x, n) = xv;
            mesh.updateDeformedConfiguration();
        }

        Eigen::Map<Eigen::VectorXd> x_map()
        {
            return Eigen::Map<Eigen::VectorXd>(mesh.getDataPointer(), n);
        }

        // energy only, at the CURRENT mesh DOFs
        Real energy_here()
        {
            mesh.updateDeformedConfiguration();
            return op.compute(mesh);
        }

        // energy + gradient at DOF vector xv; returns energy, fills g
        Real eval(const Eigen::VectorXd & xv, Eigen::VectorXd & g)
        {
            set_x(xv);
            g.setZero();
            Eigen::Ref<Eigen::VectorXd> gref(g);
            return op.compute(mesh, gref);
        }

        // gradient only at xv (energy discarded)
        void grad_at(const Eigen::VectorXd & xv, Eigen::VectorXd & g)
        {
            set_x(xv);
            g.setZero();
            Eigen::Ref<Eigen::VectorXd> gref(g);
            op.compute(mesh, gref);
        }

        // matrix-free Hessian-vector product, central difference of the analytic gradient.
        // damped: returns (H + lambda I) v. Costs two gradient evaluations.
        void hvp_damped(const Eigen::VectorXd & xc, const Eigen::VectorXd & v,
                        const Real lambda, Eigen::VectorXd & Hv)
        {
            const Real vn = v.norm();
            if(vn == 0.0){ Hv.setZero(); return; }
            // step scaled to the FD sweet spot ~ eps^(1/3), relative to ||x||
            const Real hstep = std::cbrt(std::numeric_limits<Real>::epsilon());
            const Real eps = hstep * std::max(Real(1), xc.norm()) / vn;
            xtmp = xc + eps * v;  grad_at(xtmp, gp);
            xtmp = xc - eps * v;  grad_at(xtmp, gm);
            Hv = (gp - gm) / (2.0 * eps);
            Hv += lambda * v;          // Levenberg-Marquardt damping
        }

        // Solve (H + lambda I) p = -g by conjugate gradients (matrix-free).
        // Truncated: stops on tolerance, negative curvature, or iteration cap.
        int cg_solve(const Eigen::VectorXd & xc, const Eigen::VectorXd & g,
                     const Real lambda, int maxit, Eigen::VectorXd & p)
        {
            p.setZero(n);
            Eigen::VectorXd r = -g;                 // residual = -g - (H+lI)p, p=0
            Eigen::VectorXd d = r;
            Eigen::VectorXd Hd(n);
            Real rr = r.squaredNorm();
            const Real r0 = std::sqrt(rr);
            if(r0 == 0.0) return 0;
            const Real cgtol = std::min(Real(0.5), std::sqrt(r0)) * r0;  // inexact-Newton forcing
            int it = 0;
            for(; it < maxit; ++it)
            {
                hvp_damped(xc, d, lambda, Hd);
                const Real dHd = d.dot(Hd);
                if(dHd <= 1e-14 * d.squaredNorm())    // non-positive curvature: stop with current p
                    break;
                const Real alpha = rr / dHd;
                p += alpha * d;
                r -= alpha * Hd;
                const Real rr_new = r.squaredNorm();
                if(std::sqrt(rr_new) <= cgtol) { ++it; break; }
                d = r + (rr_new / rr) * d;
                rr = rr_new;
            }
            return it;
        }

    public:
        NewtonCG_Energy(tMesh & mesh_in, const tMeshOperator & op_in):
        mesh(mesh_in), op(op_in), n(op_in.getNumberOfVariables(mesh_in)),
        last_gnorm(-1), last_iterations(0)
        {
            gbuf.resize(n); gp.resize(n); gm.resize(n); xtmp.resize(n);
        }

        Real get_lastnorm() const { return last_gnorm; }
        int  get_lastiterations() const { return last_iterations; }

        /**
         * Minimize to gradient-norm tolerance gtol. Returns 0 on convergence, 1 otherwise.
         */
        int minimize(const Real gtol = 1e-8, const int maxOuter = 200,
                     const int cgMax = 60)
        {
            Eigen::VectorXd x = x_map();      // start from current mesh DOFs
            Eigen::VectorXd g(n), p(n), xnew(n);
            Real E = eval(x, g);
            Real gnorm = g.norm();
            last_gnorm = gnorm;

            Real lambda = 1e-6 * std::max(Real(1), gnorm);   // initial damping
            const Real lam_min = 1e-14, lam_max = 1e12;
            int outer = 0, cg_total = 0;

            for(; outer < maxOuter; ++outer)
            {
                if(gnorm <= gtol) break;

                const int cgit = cg_solve(x, g, lambda, cgMax, p);
                cg_total += cgit;

                // trial step
                xnew = x + p;
                const Real Enew = eval(xnew, gbuf);   // gbuf = grad at trial (reused if accepted)

                if(std::isfinite(Enew) && Enew < E)
                {
                    // accept
                    x = xnew;
                    E = Enew;
                    g = gbuf;
                    gnorm = g.norm();
                    lambda = std::max(lam_min, lambda * 0.4);   // trust more
                }
                else
                {
                    // reject: increase damping, keep x (g already at x)
                    lambda = std::min(lam_max, lambda * 4.0);
                    set_x(x);                                    // restore mesh to accepted x
                }
                last_gnorm = gnorm;

                if(verbose && (outer % 5 == 0 || gnorm <= gtol))
                    printf("[NewtonCG] outer %3d  E=%.10e  |g|=%.6e  lambda=%.2e  cg=%d\n",
                           outer, E, gnorm, lambda, cgit);

                if(lambda >= lam_max) break;   // stalled
            }

            set_x(x);                          // leave the mesh at the found minimum
            last_iterations = outer;
            if(verbose)
                printf("[NewtonCG] done: outer=%d  cg_total=%d  final |g|=%.6e  (%s)\n",
                       outer, cg_total, gnorm, gnorm <= gtol ? "converged" : "stopped");
            return (gnorm <= gtol) ? 0 : 1;
        }
    };
}

#endif

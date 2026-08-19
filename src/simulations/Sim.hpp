//
//  Sim.hpp
//  Elasticity
//
//  Created by Wim van Rees on 21/02/16.
//  Modified by Vladislav Sushitskii on 03/29/22.
//  Copyright © 2022 Wim van Rees and Vladislav Sushitskii. All rights reserved.
//

#ifndef Sim_hpp
#define Sim_hpp

#include "common.hpp"
#include "ArgumentParser.hpp"
#include "ReadVTK.hpp"
#include "WriteVTK.hpp"
#include "WriteSTL.hpp"
#include "ComputeCurvatures.hpp"

#include "EnergyOperator.hpp"

#ifdef USELIBLBFGS
#include "LBFGS_Wrapper.hpp"
#endif

#ifdef USEHLBFGS
#include "HLBFGS_Wrapper.hpp"
#include "HLBFGS_ReducedWrapper.hpp"
#include "NewtonCG_Energy.hpp"
#include "TinyADHessian_Bilayer.hpp"
#include "HLBFGS_Preconditioned.hpp"
#include <Eigen/SparseCholesky>
#endif

#include <igl/writeOBJ.h>

/*! \class Sim
 * \brief Base class for simulations.
 *
 * This class is called from main, and performs the main simulation. Every class that derives from here can implement a simulation case.
 */
class BaseSim
{
protected:
    ArgumentParser & parser;

public:

    BaseSim(ArgumentParser & parser_in):
    parser(parser_in)
    {
        parser.save_defaults();// make sure all default values are printed as well from now on
        parser.save_options();
    }

    virtual void init() = 0;
    virtual void run() = 0;
    virtual int optimize()
    {
        std::cout << "Optimize not (yet) implemented for this class " << std::endl;
        return -1;
    };

    virtual ~BaseSim()
    {}
};


template<typename tMesh>
class Sim : public BaseSim
{
public:
    typedef typename tMesh::tCurrentConfigData tCurrentConfigData;
    typedef typename tMesh::tReferenceConfigData tReferenceConfigData;
protected:
    std::string tag;
    tMesh mesh;

    virtual void writeSTL(const std::string filename)
    {
        WriteSTL::write(mesh.getTopology(), mesh.getCurrentConfiguration(), filename);
    }
    virtual void dump(const size_t iter, const int nDigits=5)
    {
        const std::string filename = tag+"_"+helpers::ToString(iter, nDigits);
        dump(filename);
    }

    virtual void dump(const std::string filename)
    {
        const auto cvertices = mesh.getCurrentConfiguration().getVertices();
        const auto cface2vertices = mesh.getTopology().getFace2Vertices();
        WriteVTK writer(cvertices, cface2vertices);
        writer.write(filename);
    }

    virtual void dumpWithNormals(const std::string filename, const bool restconfig = false)
    {
        const TopologyData & topology = mesh.getTopology();
        const tReferenceConfigData & restState = mesh.getRestConfiguration();
        const tCurrentConfigData & currentState = mesh.getCurrentConfiguration();
        const BoundaryConditionsData & boundaryConditions = mesh.getBoundaryConditions();
        const int nFaces = topology.getNumberOfFaces();

        Eigen::MatrixXd normal_vectors(nFaces,3);
        if(restconfig)
            restState.computeFaceNormalsFromDirectors(topology, boundaryConditions, normal_vectors);
        else
            currentState.computeFaceNormalsFromDirectors(topology, boundaryConditions, normal_vectors);

        const auto cvertices = (restconfig ? mesh.getRestConfiguration().getVertices() : mesh.getCurrentConfiguration().getVertices());
        const auto cface2vertices = mesh.getTopology().getFace2Vertices();

        WriteVTK writer(cvertices, cface2vertices);
        writer.addVectorFieldToFaces(normal_vectors, "normals");
        //if(not restconfig) // always add curvatures of current config, even if it is on-top of the rest config
        {
            Eigen::VectorXd gauss(nFaces);
            Eigen::VectorXd mean(nFaces);
            Eigen::VectorXd PrincCurv1(nFaces);
            Eigen::VectorXd PrincCurv2(nFaces);
            Eigen::VectorXd CurvX(nFaces);
            Eigen::VectorXd CurvY(nFaces);

            Eigen::Vector3d Dir1 = (Eigen::Vector3d() <<  1, 0, 0).finished();
            Eigen::Vector3d Dir2 = (Eigen::Vector3d() <<  0, 1, 0).finished();

            ComputeCurvatures<tMesh> computeCurvatures;
            computeCurvatures.computeDir(mesh, gauss, mean, PrincCurv1, PrincCurv2, CurvX, CurvY, Dir1, Dir2);
            
            writer.addScalarFieldToFaces(gauss, "gauss");
            writer.addScalarFieldToFaces(mean, "mean");
            writer.addScalarFieldToFaces(PrincCurv1, "PrincCurv1");
            writer.addScalarFieldToFaces(PrincCurv2, "PrincCurv2");
            writer.addScalarFieldToFaces(CurvX, "CurvX");
            writer.addScalarFieldToFaces(CurvY, "CurvY");
        }
        writer.write(filename);
    }


    virtual void dumpWithCurvatures(const std::string filename)
    {
        const auto cvertices = mesh.getCurrentConfiguration().getVertices();
        const auto cface2vertices = mesh.getTopology().getFace2Vertices();

        const int nFaces = mesh.getNumberOfFaces();
        Eigen::VectorXd gauss(nFaces);
        Eigen::VectorXd mean(nFaces);
        ComputeCurvatures<tMesh> computeCurvatures;
        computeCurvatures.compute(mesh, gauss, mean);


        WriteVTK writer(cvertices, cface2vertices);
        writer.addScalarFieldToFaces(gauss, "gauss");
        writer.addScalarFieldToFaces(mean, "mean");

        writer.write(filename);
    }

    virtual void dumpObjWithTextureCoordinates(const std::string filename) const
    {
        // get vertices from current configuration
        const auto cvertices = mesh.getCurrentConfiguration().getVertices();
        const auto rvertices = mesh.getRestConfiguration().getVertices();

        // get face2vertices
        const auto cface2vertices = mesh.getTopology().getFace2Vertices();

        // UV coordinates : rest configuration (taking only xy components)
        const int nVertices = mesh.getNumberOfVertices();
        Eigen::MatrixXd uv_coordinates(nVertices,2);
        for(int i=0;i<nVertices;++i)
        for(int d=0;d<2;++d)
        uv_coordinates(i,d) = rvertices(i,d);

        // normals : dont care
        Eigen::MatrixXd normals;
        Eigen::MatrixXi face2verts_normals;
        //igl::writeOBJ<Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXd, Eigen::MatrixXi, Eigen::MatrixXd, Eigen::MatrixXi>(filename + ".obj", cvertices, cface2vertices, normals, face2verts_normals, uv_coordinates, cface2vertices);
        igl::writeOBJ(filename + ".obj", cvertices, cface2vertices, normals, face2verts_normals, uv_coordinates, cface2vertices);
    }


    /*! \struct MinimizationReport
     * \brief How the last minimizeEnergy / minimizeEnergyReduced call terminated.
     *
     * The int those methods return collapses several distinct HLBFGS outcomes into
     * success/failure, and counts a stagnated line-search as success. A solve cut short
     * by max_iter is therefore indistinguishable from a converged one -- both in the
     * dumps and as the starting guess for the next continuation step. Callers that care
     * whether a stage actually reached equilibrium should read this instead.
     */
    struct MinimizationReport
    {
        int code = -1;              //!< raw HLBFGS code, see HLBFGS_Energy::get_lastreturncode
        int iterations = 0;
        Real gradientNorm = -1.0;   //!< dimensional: scales with E, thickness and mesh

        /**
         * Whether the stage may be treated as an equilibrium.
         *
         * Code 5 (iteration cap) never qualifies. Codes 2 and 3 are genuine tolerance
         * hits, but note that minimizeEnergy passes epsMin -- machine epsilon by default
         * -- so in practice they never fire and the normal outcome is code 1 or 4, the
         * line search running out of progress. Those are accepted; pass a positive
         * gradientTolerance to additionally require the achieved gradient norm to meet
         * it. Calibrate that threshold from a known-good run, since the norm is
         * dimensional.
         */
        bool converged(const Real gradientTolerance = -1.0) const
        {
            if(code == 5) return false;
            if(code == 2 or code == 3) return true;
            if(code != 1 and code != 4) return false; // never ran, or unmodelled outcome
            if(gradientTolerance <= 0.0) return true;
            return (gradientNorm >= 0.0 and gradientNorm <= gradientTolerance);
        }
    };

    MinimizationReport lastMinimization;

    template<typename tMeshOperator, bool verbose = true>
    int minimizeEnergy(const tMeshOperator & op, Real & eps,
        const Real epsMin=std::numeric_limits<Real>::epsilon(), const bool stepWise = false,
        //ver-0122,set dump control and max iter control
        const std::vector<int>* dump_iters_ptr = nullptr,
                   const int max_iter = -1)
    {
        lastMinimization = MinimizationReport();

        std::cout << "[Sim] minimizeEnergy extended args received. "
          << "stepWise=" << stepWise
          << " eps_init=" << eps
          << " epsMin=" << epsMin
          << "dump_ptr=" << (dump_iters_ptr != nullptr)
          << " max_iter=" << max_iter << std::endl;

#ifdef USELIBLBFGS
        // use the LBFGS_Energy class to directly minimize the energy on the mesh with these operators
        // LBFGS does not use the hessian
        LBFGS::LBFGS_Energy<Real, tMesh, tMeshOperator, verbose> lbfgs_energy(mesh, op);
        int retval = 0;
        while(retval == 0 && eps > epsMin)
        {
            eps *= 0.1;
            retval = lbfgs_energy.minimize(eps);
        }
#else
#ifdef USEHLBFGS

        HLBFGS_Methods::HLBFGS_Energy<tMesh, tMeshOperator, verbose> hlbfgs_wrapper(mesh, op);
        // For debugging 0126
        std::cout << "[Sim] setting HLBFGS controls: dump_size="
          << (dump_iters_ptr ? dump_iters_ptr->size() : 0)
          << " max_iter=" << max_iter << std::endl;

        // if(dump_iters_ptr != nullptr)
        //     hlbfgs_wrapper.set_dump_schedule(tag, *dump_iters_ptr);

        // if(max_iter > 0)
        //     hlbfgs_wrapper.set_max_iter(max_iter);

        // back to original logic
        // ver-0122 
        // Optional: intermediate iteration dumps (true continuous trajectory)
        if(dump_iters_ptr != nullptr && !dump_iters_ptr->empty())
            hlbfgs_wrapper.set_dump_schedule(tag, *dump_iters_ptr);

        // Optional: override max iterations (default unchanged if max_iter <= 0)
        if(max_iter > 0)
            hlbfgs_wrapper.set_max_iter(max_iter);

        int retval = 0;
        if(stepWise)
        {
            while(retval == 0 && eps > epsMin)
            {
                eps *= 0.1;
                retval = hlbfgs_wrapper.minimize(tag+"_diagnostics.dat", eps);
            }
        }
        else
        {
            retval = hlbfgs_wrapper.minimize(tag+"_diagnostics.dat", epsMin);
            eps = hlbfgs_wrapper.get_lastnorm();
        }

        lastMinimization.code = hlbfgs_wrapper.get_lastreturncode();
        lastMinimization.iterations = hlbfgs_wrapper.get_lastiterations();
        lastMinimization.gradientNorm = hlbfgs_wrapper.get_lastnorm();

#else
        std::cout << "should use liblbfgs or hlbfgs\n";
#endif
#endif

        // store energies
        {
            std::vector<std::pair<std::string, Real>> energies;
            op.addEnergy(energies);
            FILE * f = fopen((tag+"_energies.dat").c_str(), "a");
            for(const auto & eng : energies)
            {
                fprintf(f, "%s \t\t %10.10e\n", eng.first.c_str(), eng.second);
            }
            fclose(f);
        }
        return retval;
    }


    /**
     * Curvature-aware alternative to minimizeEnergy: matrix-free Newton-CG (damped) using
     * finite-difference Hessian-vector products of the analytic gradient. Same operator
     * interface as minimizeEnergy; converges in few second-order steps where HLBFGS grinds
     * for many first-order ones. Returns 0 on convergence to gradient-norm tolerance gtol.
     */
    template<typename tMeshOperator, bool verbose = true>
    int minimizeEnergyNewtonCG(const tMeshOperator & op, const Real gtol = 1e-8,
                               const int maxOuter = 200, const int cgMax = 60)
    {
        lastMinimization = MinimizationReport();
        NewtonCG_Methods::NewtonCG_Energy<tMesh, tMeshOperator, verbose> ncg(mesh, op);
        const int ret = ncg.minimize(gtol, maxOuter, cgMax);
        lastMinimization.code = ret;
        lastMinimization.iterations = ncg.get_lastiterations();
        lastMinimization.gradientNorm = ncg.get_lastnorm();

        std::vector<std::pair<std::string, Real>> energies;
        op.addEnergy(energies);
        FILE * f = fopen((tag+"_energies.dat").c_str(), "a");
        for(const auto & eng : energies)
            fprintf(f, "%s \t\t %10.10e\n", eng.first.c_str(), eng.second);
        fclose(f);
        return ret;
    }


    /**
     * HLBFGS preconditioned by the exact TinyAD Hessian (ICFS mode). Keeps HLBFGS's cheap
     * gradient steps but conditions them with second-order curvature rebuilt every T iterations
     * -- the standard fix for the biharmonic mesh-scaling. eps is HLBFGS's (relative) accuracy.
     */
    template<typename tMeshOperator, bool verbose = true>
    int minimizeEnergyHLBFGSPrecond(const tMeshOperator & op,
                                    const TinyADHessian_Bilayer<tMesh> & tad,
                                    const Real eps, const int T = 5)
    {
        lastMinimization = MinimizationReport();
        HLBFGS_Methods::HLBFGS_EnergyPrecond<tMesh, tMeshOperator, verbose> solver(mesh, op, tad);
        const int ret = solver.minimizePrecond(tag + "_diagnostics.dat", eps, T);
        lastMinimization.code = ret;
        lastMinimization.iterations = solver.get_lastiterations();
        lastMinimization.gradientNorm = solver.get_lastnorm();

        std::vector<std::pair<std::string, Real>> energies;
        op.addEnergy(energies);
        FILE * f = fopen((tag+"_energies.dat").c_str(), "a");
        for(const auto & eng : energies)
            fprintf(f, "%s \t\t %10.10e\n", eng.first.c_str(), eng.second);
        fclose(f);
        return ret;
    }


    /**
     * TRUE Newton with the EXACT TinyAD Hessian: damped (Levenberg) Newton, reassembling the
     * exact sparse Hessian at each accepted iterate and taking p = -(H + lambda I)^{-1} g via a
     * sparse LDLT factorization. No finite-difference noise floor -- converges to machine
     * precision in a few second-order steps where HLBFGS grinds and FD-Newton plateaus.
     * lambda damps the free-panel rigid nullspace and indefinite curvature; it adapts from step
     * success. The exact Hessian is provided by TinyADHessian_Bilayer (verified to ~1e-15).
     */
    template<typename tMeshOperator, bool verbose = true>
    int minimizeEnergyNewtonExact(const tMeshOperator & op,
                                  const TinyADHessian_Bilayer<tMesh> & tad,
                                  const Real gtol = 1e-10, const int maxOuter = 100)
    {
        lastMinimization = MinimizationReport();
        const int n = op.getNumberOfVariables(mesh);

        auto eval_grad = [&](Eigen::VectorXd & g) -> Real {
            mesh.updateDeformedConfiguration();
            g.setZero();
            Eigen::Ref<Eigen::VectorXd> gr(g);
            return op.compute(mesh, gr);
        };
        auto set_x = [&](const Eigen::VectorXd & xv){
            Eigen::Map<Eigen::VectorXd>(mesh.getDataPointer(), n) = xv;
            mesh.updateDeformedConfiguration();
        };

        Eigen::VectorXd xcur = Eigen::Map<Eigen::VectorXd>(mesh.getDataPointer(), n);
        Eigen::VectorXd g(n);
        Real E = eval_grad(g);
        Real gnorm = g.norm();

        // CG solve of (H + lambda I) p = -g using cheap sparse matvecs -- a direct sparse
        // factorization of the biharmonic shell Hessian has heavy fill-in and is far too slow.
        auto cg_solve = [&](const Eigen::SparseMatrix<double> & H, const Eigen::VectorXd & rhs,
                            const Real lambda, const int maxit) -> Eigen::VectorXd {
            Eigen::VectorXd p = Eigen::VectorXd::Zero(n);
            Eigen::VectorXd r = rhs;                 // rhs = -g, p=0 so residual = rhs
            Eigen::VectorXd d = r, Hd(n);
            Real rr = r.squaredNorm();
            const Real r0 = std::sqrt(rr);
            if(r0 == 0.0) return p;
            const Real cgtol = std::min(Real(0.1), std::sqrt(r0)) * r0;
            for(int it = 0; it < maxit; ++it)
            {
                Hd.noalias() = H * d; Hd += lambda * d;
                const Real dHd = d.dot(Hd);
                if(dHd <= 0.0) break;
                const Real alpha = rr / dHd;
                p += alpha * d; r -= alpha * Hd;
                const Real rr_new = r.squaredNorm();
                if(std::sqrt(rr_new) <= cgtol) break;
                d = r + (rr_new / rr) * d; rr = rr_new;
            }
            return p;
        };

        Real lambda = 1e-8 * std::max(Real(1), gnorm);
        const Real lam_min = 1e-14, lam_max = 1e12;
        int outer = 0;

        for(; outer < maxOuter; ++outer)
        {
            if(gnorm <= gtol) break;
            set_x(xcur);                                   // ensure mesh at accepted x
            const Eigen::SparseMatrix<double> H = tad.assembleHessian(mesh);  // exact H at xcur

            bool stepped = false;
            for(int tries = 0; tries < 40 && !stepped; ++tries)
            {
                const Eigen::VectorXd p = cg_solve(H, -g, lambda, 300);
                const Eigen::VectorXd xnew = xcur + p;
                set_x(xnew);
                const Real Enew = op.compute(mesh);
                if(std::isfinite(Enew) && Enew < E)
                {
                    xcur = xnew; E = Enew;
                    (void)eval_grad(g); gnorm = g.norm();   // gradient at the accepted iterate
                    lambda = std::max(lam_min, lambda*0.3);
                    stepped = true;
                }
                else
                {
                    lambda = std::min(lam_max, lambda*4.0);
                    set_x(xcur);                            // restore
                }
            }
            if(verbose)
                printf("[NewtonExact] outer %2d  E=%.10e  |g|=%.6e  lambda=%.2e  nnz=%ld\n",
                       outer, E, gnorm, lambda, (long)H.nonZeros());
            if(!stepped || lambda >= lam_max) break;
        }
        set_x(xcur);
        lastMinimization.iterations = outer;
        lastMinimization.gradientNorm = gnorm;
        if(verbose)
            printf("[NewtonExact] done: outer=%d  final |g|=%.6e  (%s)\n",
                   outer, gnorm, gnorm <= gtol ? "converged" : "stopped");

        std::vector<std::pair<std::string, Real>> energies;
        op.addEnergy(energies);
        FILE * f = fopen((tag+"_energies.dat").c_str(), "a");
        for(const auto & eng : energies)
            fprintf(f, "%s \t\t %10.10e\n", eng.first.c_str(), eng.second);
        fclose(f);
        return (gnorm <= gtol) ? 0 : 1;
    }


    // Updates @08/06:
    // Minimize only over the currently free mesh DOFs.  The active fixed-DOF
    // mask is read from mesh.getBoundaryConditions() when the wrapper is built.
    template<typename tMeshOperator, bool verbose = true>
    int minimizeEnergyReduced(
        const tMeshOperator& op,
        Real& eps,
        const Real epsMin = std::numeric_limits<Real>::epsilon(),
        const bool stepWise = false,
        const std::vector<int>* dump_iters_ptr = nullptr,
        const int max_iter = -1)
    {
        lastMinimization = MinimizationReport();
#ifdef USEHLBFGS
        HLBFGS_Methods::HLBFGS_Energy_Reduced<
            tMesh, tMeshOperator, verbose> hlbfgs_wrapper(mesh, op);

        if(dump_iters_ptr != nullptr && !dump_iters_ptr->empty())
            hlbfgs_wrapper.set_dump_schedule(tag, *dump_iters_ptr);
        if(max_iter > 0)
            hlbfgs_wrapper.set_max_iter(max_iter);

        int retval = 0;
        if(stepWise)
        {
            while(retval == 0 && eps > epsMin)
            {
                eps *= 0.1;
                retval = hlbfgs_wrapper.minimize(
                    tag + "_diagnostics_reduced.dat", eps);
            }
        }
        else
        {
            retval = hlbfgs_wrapper.minimize(
                tag + "_diagnostics_reduced.dat", epsMin);
            eps = hlbfgs_wrapper.get_lastnorm();
        }

        lastMinimization.code = hlbfgs_wrapper.get_lastreturncode();
        lastMinimization.iterations = hlbfgs_wrapper.get_lastiterations();
        lastMinimization.gradientNorm = hlbfgs_wrapper.get_lastnorm();

        std::cout
            << "[Sim] reduced solve variables: free="
            << hlbfgs_wrapper.getNumberOfFreeVariables()
            << ", fixed="
            << hlbfgs_wrapper.getNumberOfFixedVariables()
            << ", full="
            << hlbfgs_wrapper.getNumberOfFullVariables()
            << std::endl;

        std::vector<std::pair<std::string, Real>> energies;
        op.addEnergy(energies);
        if(FILE* file = std::fopen((tag + "_energies.dat").c_str(), "a"))
        {
            for(const auto& energy : energies)
                std::fprintf(
                    file, "%s \t\t %10.10e\n",
                    energy.first.c_str(), energy.second);
            std::fclose(file);
        }
        return retval;
#else
        (void)op;
        (void)eps;
        (void)epsMin;
        (void)stepWise;
        (void)dump_iters_ptr;
        (void)max_iter;
        throw std::runtime_error(
            "minimizeEnergyReduced requires a build with USEHLBFGS.");
#endif
    }


public:

    Sim(ArgumentParser & parser_in):
    BaseSim(parser_in),
    tag("Sim")
    {
    }

    virtual ~Sim()
    {}
};

#endif /* Sim_hpp */

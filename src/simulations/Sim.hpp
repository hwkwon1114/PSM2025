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

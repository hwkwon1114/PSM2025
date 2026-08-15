//
//  Sim.hpp
//  Elasticity
//
//  Created by Wim van Rees on 21/02/16.
//  Copyright © 2016 Wim van Rees. All rights reserved.
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
            ComputeCurvatures<tMesh> computeCurvatures;
            computeCurvatures.compute(mesh, gauss, mean);
            writer.addScalarFieldToFaces(gauss, "gauss");
            writer.addScalarFieldToFaces(mean, "mean");
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
    
    virtual void dumpMomenta(const tUint iter, const Real t, const std::array<Real,3> & linMom, const std::array<Real,3> & angMom)
    {
        std::ofstream momfile;
        momfile.open(tag+"_mom.dat", (iter==0 ? std::ios::out : std::ios::app));
        momfile.precision(10);
        momfile << std::scientific;
        
        if(iter==0)
        {
            // print header
            momfile << "# \t iter \t time \t lin X \t lin Y \t lin Z \t ang X \t ang Y \t ang Z" << std::endl;
        }
        
        momfile << "\t" << iter << "\t" << t << "\t" <<  linMom[0] << "\t" << linMom[1] << "\t" << linMom[2] << "\t" << angMom[0] << "\t" << angMom[1] << "\t" << angMom[2] << std::endl;
        momfile.close();
    }
    
    virtual void dumpEnergy(const tUint iter, const Real t, const std::vector<std::pair<std::string, Real>> & energies)
    {
        std::ofstream engfile;
        engfile.open(tag+"_eng.dat", (iter==0 ? std::ios::out : std::ios::app));
        engfile.precision(10);
        engfile << std::scientific;
        
        if(iter==0)
        {
            // print header
            engfile << "# \t iter \t time";
            for(const auto & el : energies)
                engfile << "\t" << el.first;
            engfile << "\t total" << std::endl;
        }
        
        Real sum = 0.0;
        engfile << "\t" << iter << "\t" << t;
        for(const auto & el : energies)
        {
            sum += el.second;
            engfile << "\t" << el.second;
        }
        engfile << "\t" << sum << std::endl;
        engfile.close();
    }
    
    void computeMomenta(const Eigen::VectorXd & massVertices, const Eigen::MatrixXd & vVertices, std::array<Real,3> & lin, std::array<Real,3> & ang)
    {
        lin[0] = lin[1] = lin[2] = 0.0;
        ang[0] = ang[1] = ang[2] = 0.0;
        
        const int nVertices = mesh.getNumberOfVertices();
        const auto vertices = mesh.getCurrentConfiguration().getVertices();
        
        const Eigen::Vector3d org_vec = (Eigen::Vector3d() << 0,0,0).finished();
        
        for(int i=0;i<nVertices;++i)
        {
            const Real mass = massVertices(i);
            const Eigen::Vector3d vertex_pos = vertices.row(i);
            const Eigen::Vector3d position = vertex_pos - org_vec;
            const Eigen::Vector3d velocity = vVertices.row(i);
            const Eigen::Vector3d linMom = mass*velocity;
            const Eigen::Vector3d angMom = position.cross(linMom);
            for(int d=0;d<3;++d)
            {
                lin[d] += linMom(d);
                ang[d] += angMom(d);
            }
        }
    }
    
    template<int component>
    void addNoiseToVertices_c(const Real ampl, const unsigned int seed = 42)
    {
        std::mt19937 gen;
        gen.seed(seed);
        std::uniform_real_distribution<Real> distV(-ampl, ampl);
        auto perturb_v = [&](Eigen::Vector3d in)
        {
            Eigen::Vector3d retval;
            for(int d=0;d<3;++d)
                retval(d) = in(d) + (d == component ? distV(gen) : 0.0);
            return retval;
        };
        mesh.changeVertices(perturb_v);
    }
    
    void addNoiseToVertices(const Real ampl, const unsigned int seed = 42)
    {
        std::mt19937 gen;
        gen.seed(seed);
        std::uniform_real_distribution<Real> distV(-ampl, ampl);
        auto perturb_v = [&](Eigen::Vector3d in)
        {
            Eigen::Vector3d retval;
            retval << in(0)+distV(gen), in(1)+distV(gen), in(2)+distV(gen);
            return retval;
        };
        mesh.changeVertices(perturb_v);
    }
    
    void addNoiseToEdgeDirectors(const Real ampl, const unsigned int seed = 42)
    {
        std::mt19937 gen;
        gen.seed(seed);
        std::uniform_real_distribution<Real> distE(-ampl, ampl);
        auto perturb_e = [&](Real in)
        {
            return in + distE(gen);
        };
        
        mesh.changeEdgeDirectors(perturb_e);
    }
    
    /*! \struct MinimizationReport
     * \brief Outcome of the last minimizeEnergy call.
     *
     * The coarse int returned by minimizeEnergy only distinguishes 'HLBFGS was happy' from
     * 'it was not', and counts a stagnated line-search (code 4) as success. Callers that need
     * to know whether a stage actually reached equilibrium should inspect this instead.
     */
    struct MinimizationReport
    {
        int code = -1;              //!< raw HLBFGS termination code, see HLBFGS_Energy::get_lastreturncode
        int iterations = 0;         //!< iterations taken
        Real gradientNorm = -1.0;   //!< L2 norm of the gradient at termination (dimensional)
        int nVariables = 0;         //!< number of active DOFs, for the per-DOF (mesh-consistent) norm

        /**
         * Per-DOF RMS gradient : ||g|| / sqrt(N).
         *
         * The bare ||g|| grows like sqrt(N) under refinement (more residual entries to sum),
         * so a threshold on it is a different tightness on every mesh. Dividing by sqrt(N)
         * removes that growth, so a single -gradtol means the same thing across resolutions.
         * Falls back to the dimensional norm when N is unknown.
         */
        Real gradientNormPerDof() const
        {
            if(nVariables <= 0 or gradientNorm < 0.0) return gradientNorm;
            return gradientNorm / std::sqrt(static_cast<Real>(nVariables));
        }

        /**
         * Whether the stage may be treated as an equilibrium.
         *
         * Code 5 (iteration cap) is a truncated solve and never qualifies -- this is the case
         * -maxiterations makes reachable. Codes 2 and 3 are genuine tolerance hits, but note
         * that minimizeEnergy hands HLBFGS a tolerance at machine epsilon, so in practice they
         * never fire. The normal outcome is code 1 or 4 : the line search stopped making
         * progress. That is how a converged run actually terminates here, so it is accepted,
         * and -gradtol is what turns it into a real check by requiring the achieved gradient
         * to meet a threshold.
         *
         * The threshold is now applied to the *per-DOF RMS* gradient (gradientNormPerDof),
         * which is mesh-consistent : calibrate -gradtol once, not per resolution. It stays
         * material-dependent (it scales with E, h), so recalibrate if those change.
         *
         * NOTE : this is a first-order test. It cannot distinguish a minimum from a saddle,
         * which is exactly how the flat, pre-buckled state slips through as "converged".
         * The second-order (Hessian-eigenvalue) gate that closes that gap lives in the
         * Python driver (python/stage_check.py, assess_stage); a native C++ port is Phase 0
         * of docs/robust_optimization_spec.
         */
        bool converged(const Real gradientTolerance = -1.0) const
        {
            if(code == 5) return false;
            if(code == 2 or code == 3) return true;
            if(code != 1 and code != 4) return false; // never ran, or an outcome we do not model
            if(gradientTolerance <= 0.0) return true;
            return (gradientNorm >= 0.0 and gradientNormPerDof() <= gradientTolerance);
        }
    };

    MinimizationReport lastMinimization;

    template<typename tMeshOperator, bool verbose = true>
    int minimizeEnergy(const tMeshOperator & op, Real & eps,
                       const Real epsMin=std::numeric_limits<Real>::epsilon(),
                       const bool stepWise = false,
                       const int maxIterations = 1000000000)
    {
        lastMinimization = MinimizationReport();

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
        int retval = 0;
        if(stepWise)
        {
            while(retval == 0 && eps > epsMin)
            {
                eps *= 0.1;
                retval = hlbfgs_wrapper.minimize(tag+"_diagnostics.dat", eps, 10, maxIterations);
            }
        }
        else
        {
            retval = hlbfgs_wrapper.minimize(tag+"_diagnostics.dat", epsMin, 10, maxIterations);
            eps = hlbfgs_wrapper.get_lastnorm();
        }

        lastMinimization.code = hlbfgs_wrapper.get_lastreturncode();
        lastMinimization.iterations = hlbfgs_wrapper.get_lastiterations();
        lastMinimization.gradientNorm = hlbfgs_wrapper.get_lastnorm();
        lastMinimization.nVariables = op.getNumberOfVariables(mesh);
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

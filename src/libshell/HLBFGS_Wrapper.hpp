//
//  HLBFGS_Wrapper.hpp
//  Elasticity
//
//  Created by Wim van Rees on 30/03/16.
//  Copyright © 2016 Wim van Rees. All rights reserved.
//

#ifndef HLBFGS_Wrapper_hpp
#define HLBFGS_Wrapper_hpp

#ifdef USEHLBFGS

#include "HLBFGS.h"
#include "Profiler.hpp"
#include "WriteVTK.hpp"

namespace HLBFGS_Methods
{
    
    /**
     * @brief Wrapper class around the HLBFGS library
     *
     * @details
     * This class is wraps around the HLBFGS library (http://research.microsoft.com/en-us/UM/people/yangliu/software/HLBFGS/ ), by providing a default_setup and the framework for an evaluate and newiteration method
     */
    
    class HLBFGS_Wrapper
    {
    protected:
        
        /**
         * Default setup of HLBFGS parameters
         *
         *
         * These are the meanings of the parameter values:
         * - parameter 0 : function tolerance used in line-search
         * - parameter 1 : variable tolerance used in line-search
         * - parameter 2 : gradient tolerance used in line-search
         * - parameter 3 : stpmin used in line-search
         * - parameter 4 : stpmax used in line-search
         * - parameter 5 : the stop criterion ( ||G||/max(1,||X||) < PARAMETER[5] )
         * - parameter 6 : the stop criterion ( ||G|| < PARAMETER[6] )
         * - rest is reserved
         *
         * These are the meanings of the info values:
         * - info 0  : the max number of evaluation in line-search
         * - info 1  : the total number of evalfunc calls
         * - info 2  : the current number of iterations
         * - info 3  : The lbfgs strategy. 0: standard, 1: M1QN3 strategy[8](recommended).
         * - info 4  : the max number of iterations
         * - info 5  : 1: print message, 0: do nothing
         * - info 6  : T: the update interval of Hessian. (typical choices: 0-200)
         * - info 7  : 0: without hessian, 1: with accurate hessian
         * - info 8  : icfs parameter
         * - info 9  : 0: classical line-search; 1: modified line-search (do not set to 1 in practice)
         * - info 10 : 0: Disable preconditioned CG; 1: Enable preconditioned CG
         * - info 11 : 0 or 1 defines different methods for choosing beta in CG.
         * - info 12 : internal usage. 0: only update the diag in USER_DEFINED_HLBFGS_UPDATE_H; 1: default.
         *
         * Example settings:
         * - L-BFGS : M>=1, INFO[3]=0, INFO[7]=0, INFO[10]=0
         * - M1QN3 : M>=1, INFO[3]=1, INFO[7]=0, INFO[10]=0
         * - Preconditioned L-BFGS	 : M>=1, INFO[3]=0, INFO[6]=T>=0, INFO[7]=1, INFO[10]=0
         * - Preconditioned M1QN3	: M>=1, INFO[3]=1, INFO[6]=T>=0, INFO[7]=1, INFO[10]=0
         * - Preconditioned Conjugate Gradient without Hessian	 : M>=1, INFO[3]=0 or 1, INFO[7]=0, INFO[10]=1
         
         * @param [out] parameter a list of parameters that will be filled (will be passed to the main HLBFGS routine)
         * @param [out] info a list of settings that will be filled (will be passed to the main HBLFGS routine)
         * @param [in] eps non-dimensional accuracy
         * @param [in] verbose a flag to specify whether we are verbose or not
         */
        
        void default_setup(double parameter[20], int info[20], const Real eps, const bool verbose)
        {
            //initialize
            INIT_HLBFGS(parameter, info);
            
            parameter[0] = 1e-4; // ftol
            parameter[1] = 1e-16; // xtol
            parameter[2] = 0.9; // gtol
            parameter[5] = eps; // nondim accuracy : ||G||/max(1,||X||)
            parameter[6] = 1e-16; // dim accuracy : ||G||
            
            info[3] = 1;
            info[4] = 1e9;
            //info[4] = (max_iter_override > 0) ? max_iter_override : (int)1e9; //ver-0122, allow override of max iterations
            info[5] = verbose ? 1 : 0;
            info[6] = 0;
            info[7] = 0;
            info[10] = 0;
            info[11] = 1;
            info[13] = 3;
        }
        
    public:
        
        /**
         * evaluate one instance of the cost function and compute energy and gradient
         *
         * this method is responsible for computing the energy and gradient given the current choice of parameters
         *
         * @param [in] N size of the system (number of unknowns)
         * @param [in] x current value of the unknowns
         * @param [in] prev_x value of the unknowns at the previous iteration
         * @param [out] f value of the energy (scalar)
         * @param [out] g value of the gradient of the energy with respect to each of the unknowns (vector)
         */
        virtual void evaluate(int N, double * x, double * prev_x, double * f, double * g) = 0;
        
        /**
         * this method will be called in between iterations - can be used for output to screen or file
         *
         * @param [in] iter the iteration number
         * @param [in] call_iter the number of function calls made so far
         * @param [in] x current value of the unknowns
         * @param [in] f current value of the energy (scalar)
         * @param [in] g current value of the gradient of the energy with respect to each of the unknowns (vector)
         * @param [in] gnorm current value of the L2 norm of the gradient (sqrt of sum of gradient entries)
         */
        virtual void newiteration(int iter, int call_iter, double * x, double * f, double * g, double * gnorm) = 0;
    };
    
    /**
     * global helper method for evaluate, to deal with the callback C-style structure of the library 
     * 
     * performs a reinterpret cast of void * to HLBFGS_Wrapper* in order to access all internal variables inside the evaluate method
     */
    static void evaluate(void * instance, int N, double* x, double *prev_x, double* f, double* g)
    {
        HLBFGS_Wrapper * myHLBFGS = reinterpret_cast<HLBFGS_Wrapper*>(instance);
        myHLBFGS->evaluate(N, x, prev_x, f, g);
    }

    /**
     * helper method for newiteration, to deal with the callback C-style structure of the library
     * 
     * performs a reinterpret cast of void * to HLBFGS_Wrapper* in order to access all internal variables inside the evaluate method
     */
    static void newiteration(void * instance, int iter, int call_iter, double *x, double* f, double *g,  double* gnorm)
    {
        HLBFGS_Wrapper * myHLBFGS = reinterpret_cast<HLBFGS_Wrapper*>(instance);
        myHLBFGS->newiteration(iter, call_iter, x, f, g, gnorm);
    }
    
    
    
    
    /**
     * @brief Wrapper around HBLFGS library specific to our application of triangle mesh energy minimization
     * @tparam tMesh Type of the mesh (needs to expose updateDeformedConfiguration and getDataPointer)
     * @tparam tMeshOperator Type of the energy operator (needs to expose getNumberOfVariables, compute(mesh), compute(mesh,gradient) and printProfilerSummary)
     * @tparam verbose whether we should run the optimization silently or not
     *
     * @details
     * This class performs a minimization over the variables of mesh (obtained through getDataPointer) in order to minimize the energy returned by the tMeshOperator::compute(mesh) routine. To do so we use the HLBFGS library with settings as specific above, performing function evaluations and using the gradient to approximate the Hessian.
     * The number of variables we optimize over comes from the energy operator (op.getNumberOfVariables()) as this operator determines the active variables (could be just a subset of all the variables in the mesh). However, since we use mesh.getDataPointer() as the pointer, if we deal with a subset they should be in the top part of this data array (eg only the vertices)
     */
    template<typename tMesh, typename tMeshOperator, bool verbose>
    class HLBFGS_Energy : public HLBFGS_Wrapper
    {
    protected:
        Profiler profiler;
        
        tMesh & mesh;
        const tMeshOperator & op;
        const int nVariables;
        std::string outFileName;
        Real last_gnorm;

        // ver-0122, add dump control and max iter control
        // ---- intermediate dump control ----
        bool dump_enabled = false;
        std::vector<int> dump_iters;
        std::size_t next_dump_idx = 0;
        std::string dump_prefix;

        // ---- max iteration control ----
        int max_iter_override = -1; // -1 means: use default behavior

        // ---- termination reporting ----
        // The int returned by minimize() collapses several distinct outcomes into
        // success/failure, so a solve truncated by max_iter_override is indistinguishable
        // from a converged one. Keep the raw code and the iteration count.
        int last_retcode = -1;
        int last_iterations = 0;

    public:
        HLBFGS_Energy(tMesh & mesh, const tMeshOperator & engop):
        mesh(mesh),
        op(engop),
        nVariables(op.getNumberOfVariables(mesh)),
        outFileName("hlbfgs_output.dat"),
        last_gnorm(-1)
        {}

        // Enable dumps at specified optimizer iterations. Add on functions to set dump schedule and max iterations.
        // Ver-0122
        void set_dump_schedule(const std::string& prefix, const std::vector<int>& iters)
        {
        std::cout << "[HLBFGS] set_dump_schedule called: iters_size=" << iters.size()
          << " prefix=" << prefix << std::endl;

        if(!iters.empty())
        {
            std::cout << "[HLBFGS] dump schedule enabled: " << iters.size()
                    << " dumps, first=" << iters.front()
                    << ", last=" << iters.back()
                    << std::endl;
        }
        dump_prefix = prefix;
        dump_iters = iters;
        next_dump_idx = 0;
        dump_enabled = !dump_iters.empty();
        }

        // Optional: allow overriding max iterations (keeps default unchanged if not called)
        void set_max_iter(int max_iter)
        {
        // ver-0122, for debugging
        max_iter_override = max_iter;
        std::cout << "[HLBFGS] max_iter override set to " << max_iter_override << std::endl;
        }
        
        /**
         * Perform the actual energy minimization up to relative accuracy eps
         *
         * @param [in] eps the relative accuracy at which to stop
         * @return a bool that indicates whether the minimization was successful or not (if verbose, it will also print a message with the return value)
         */
        int minimize(const std::string outFileName_in, const Real eps = 1e-5, const int Mval = 10)
        {
            this->outFileName = outFileName_in;
            
            mesh.updateDeformedConfiguration();
            const Real energy0 = op.compute(mesh);
            
            // get variable pointer
            Real * x = mesh.getDataPointer();
            
            // init HLBFGS
            double parameter[20];
            int info[20];
            default_setup(parameter, info, eps, verbose);

            // ver-0122
            // Optional override of maximum iterations (default unchanged if max_iter_override <= 0)
            if(max_iter_override > 0)
                info[4] = max_iter_override;
            
            const int ret = HLBFGS(nVariables, Mval, x, HLBFGS_Methods::evaluate, 0, HLBFGS_UPDATE_Hessian, HLBFGS_Methods::newiteration, this, parameter, info);

            last_retcode = ret;
            last_iterations = info[2];

            if(verbose) std::cout << "HLBFGS return value = " << ret << std::endl;
            const Real energy1 = op.compute(mesh);
            if(verbose) printf("Energy went from %10.10e to %10.10e, using epsilon = %e, final eps = %e \n", energy0, energy1, eps, last_gnorm);
            
            {
                FILE * ff = fopen(outFileName.c_str(), "a");
                fprintf(ff, "# ---- \n %d \t\t %d \t %d \t\t %10.10e \t %10.10e \t %10.10e\n# ---- \n",ret, info[2], info[1], energy0, energy1, last_gnorm);
                fclose(ff);
            }
            return (ret==2 or ret==3 or ret==4) ? 0 : 1;
        }
        
        /**
         * Evaluate the cost function and gradient 
         * 
         * This method is called through the static global method evaluate, after recasting the void pointer to an instance of HLBFGS_Wrapper. It will fill the function value for the energy (f) and the vector for the gradient (g), by first updating the mesh variables and then computing the energy and filling the gradient vector in one call. We have no need for N and x here because they are accessible directly through the mesh methods
         *
         * @see HLBFGS_Wrapper::evaluate
         */
        void evaluate(int /*N*/, double* /*x*/, double */*prev_x*/, double* f, double* g) override
        {
            //*f is the energy, *g is the gradient

            this->profiler.push_start("update mesh");
            mesh.updateDeformedConfiguration();
            this->profiler.pop_stop();
            
            // wrap Eigen structures around the gradient pointer
            Eigen::Map<Eigen::VectorXd> eigenGrad(g, nVariables);
            eigenGrad.setZero();
            
            // compute energy
            this->profiler.push_start("compute energy");
            const Real energy = op.compute(mesh, eigenGrad);
            this->profiler.pop_stop();

            *f = energy;
        }
        
        /**
         * Print the progress of the minimization if we have verbose==true
         *
         * This method is called through the static global method evaluate, after recasting the void pointer to an instance of HLBFGS_Wrapper.
         *
         * @see HLBFGS_Wrapper::newiteration
         */
        void newiteration(int iter, int call_iter, double* /* x */, double* f, double* /* g */,  double* gnorm) override
        {
            last_gnorm = *gnorm;

            //ver-0122, for debugging
            if (iter == 1) {
                std::cout << "[HLBFGS] newiteration running. dump_enabled=" << dump_enabled
                        << " next_dump_idx=" << next_dump_idx
                        << " dump_iters_size=" << dump_iters.size() << std::endl;
            }


            // ---- scheduled intermediate dumps (true continuous trajectory) ----
            // ver-0122/0126
            if (dump_enabled && next_dump_idx < dump_iters.size())
            {
            // robust trigger: if iter has reached or passed the next requested dump iteration
            if (iter >= dump_iters[next_dump_idx])
            {
                // Ensure the geometry reflects the current iterate before writing
                mesh.updateDeformedConfiguration();

                // Use same IO convention as existing code: do NOT force ".vtp"
                // mesh.writeToFile(dump_prefix + "_mesh_iter_" + std::to_string(dump_iters[next_dump_idx]));

                // Use VTK format for better compatibility with ParaView, version 0126
                const auto cvertices = mesh.getCurrentConfiguration().getVertices();
                const auto cface2vertices = mesh.getTopology().getFace2Vertices();

                // write .vtp (WriteVTK appends ".vtp")
                WriteVTK writer(cvertices, cface2vertices);
                writer.write(dump_prefix + "_mesh_iter_" + std::to_string(dump_iters[next_dump_idx]));

                next_dump_idx++;
            }
            }
            // ---- end intermediate dumps ----
            
            if(iter%1000==0 and verbose)
            {
                printf("%d: \t %d \t %10.10e \t %10.10e\n",iter,call_iter, *f, *gnorm);
                op.printProfilerSummary();
                this->profiler.printSummary();
            }
            
            if(iter%100==0)
            {
                FILE * ff = fopen(outFileName.c_str(), "a");
                fprintf(ff, "%d \t %d \t %10.10e \t %10.10e\n", iter, call_iter, *f, *gnorm);
                fclose(ff);
            }
        }
        
        /**
         * Exposes the last value of the residual
         *
         */
        Real get_lastnorm() const
        {
            return last_gnorm;
        }

        /**
         * Raw HLBFGS termination code of the last minimize() call.
         *
         * 1 : line-search failure
         * 2 : ||G||/max(1,||X||) <= PARAMETER[5]
         * 3 : ||G|| <= PARAMETER[6]
         * 4 : line-search step outside [stpmin, stpmax] (stagnation, not a tolerance)
         * 5 : iteration cap INFO[4] exceeded -- what max_iter_override makes reachable
         */
        int get_lastreturncode() const
        {
            return last_retcode;
        }

        /// Iterations taken by the last minimize() call.
        int get_lastiterations() const
        {
            return last_iterations;
        }

    };
}
#endif

#endif /* HLBFGS_Wrapper_hpp */

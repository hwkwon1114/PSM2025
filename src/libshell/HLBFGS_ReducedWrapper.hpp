#pragma once

// Reduced-degree-of-freedom HLBFGS wrapper.
// Fixed vertex/edge variables are removed from the optimizer vector rather than
// being left in the full vector with a zero gradient.

#ifdef USEHLBFGS

#include "HLBFGS_Wrapper.hpp"

#include <Eigen/Dense>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace HLBFGS_Methods {

template<typename tMesh, typename tMeshOperator, bool verbose>
class HLBFGS_Energy_Reduced : public HLBFGS_Wrapper
{
protected:
    Profiler profiler;
    tMesh& mesh;
    const tMeshOperator& op;
    const int nVariablesFull;

    std::vector<int> freeIndices;
    std::vector<int> fixedIndices;
    Eigen::VectorXd reducedVariables;
    Eigen::VectorXd fixedFullState;
    Eigen::VectorXd fullGradient;

    std::string outFileName;
    Real last_gnorm;

    // Termination reporting: the int returned by minimize() collapses distinct
    // outcomes into success/failure, so a solve truncated by max_iter_override looks
    // identical to a converged one. Keep the raw code, the iteration count and the
    // number of energy/gradient evaluations the line search actually spent.
    int last_retcode = -1;
    int last_iterations = 0;
    int last_evaluations = 0;

    bool dump_enabled = false;
    std::vector<int> dump_iters;
    std::size_t next_dump_idx = 0;
    std::string dump_prefix;
    int max_iter_override = -1;

    void buildIndexSets()
    {
        const int nVertices = mesh.getNumberOfVertices();
        const int nEdges = mesh.getNumberOfEdges();
        const int nMeshVariables = 3 * nVertices + nEdges;
        if(nVariablesFull > nMeshVariables)
            throw std::runtime_error(
                "HLBFGS_Energy_Reduced: operator requests more variables than "
                "the mesh current configuration stores.");

        const auto vertexBC =
            mesh.getBoundaryConditions().getVertexBoundaryConditions();
        const auto edgeBC =
            mesh.getBoundaryConditions().getEdgeBoundaryConditions();
        if(vertexBC.rows() != nVertices || vertexBC.cols() != 3)
            throw std::runtime_error(
                "HLBFGS_Energy_Reduced: invalid vertex boundary-condition matrix.");
        if(edgeBC.rows() != nEdges)
            throw std::runtime_error(
                "HLBFGS_Energy_Reduced: invalid edge boundary-condition vector.");

        // Eigen::Map<MatrixXd>(ptr,nVertices,3) is column major.  Therefore
        // coordinate (vertex,component) is stored at vertex + component*nVertices.
        for(int component = 0; component < 3; ++component)
        {
            for(int vertex = 0; vertex < nVertices; ++vertex)
            {
                const int index = vertex + component * nVertices;
                if(index >= nVariablesFull) continue;
                if(vertexBC(vertex,component)) fixedIndices.push_back(index);
                else freeIndices.push_back(index);
            }
        }

        const int edgeOffset = 3 * nVertices;
        for(int edge = 0; edge < nEdges; ++edge)
        {
            const int index = edgeOffset + edge;
            if(index >= nVariablesFull) continue;
            if(edgeBC(edge)) fixedIndices.push_back(index);
            else freeIndices.push_back(index);
        }

        // Any future operator-specific variables beyond the standard DCS
        // portion are treated as free by default.
        for(int index = nMeshVariables; index < nVariablesFull; ++index)
            freeIndices.push_back(index);

        if(freeIndices.empty())
            throw std::runtime_error(
                "HLBFGS_Energy_Reduced: all optimization variables are fixed.");
    }

    void scatterReducedToMesh(const double* reduced)
    {
        Real* full = mesh.getDataPointer();
        for(const int index : fixedIndices)
            full[index] = fixedFullState(index);
        for(std::size_t k = 0; k < freeIndices.size(); ++k)
            full[freeIndices[k]] = reduced[k];
    }

    void gatherMeshToReduced()
    {
        const Real* full = mesh.getDataPointer();
        for(std::size_t k = 0; k < freeIndices.size(); ++k)
            reducedVariables(k) = full[freeIndices[k]];
    }

public:
    HLBFGS_Energy_Reduced(tMesh& mesh_in, const tMeshOperator& op_in)
    : mesh(mesh_in),
      op(op_in),
      nVariablesFull(op.getNumberOfVariables(mesh)),
      outFileName("hlbfgs_reduced_output.dat"),
      last_gnorm(-1)
    {
        fixedFullState = Eigen::Map<Eigen::VectorXd>(
            mesh.getDataPointer(), nVariablesFull);
        buildIndexSets();
        reducedVariables.resize(static_cast<int>(freeIndices.size()));
        fullGradient.resize(nVariablesFull);
        gatherMeshToReduced();
    }

    void set_dump_schedule(
        const std::string& prefix,
        const std::vector<int>& iters)
    {
        dump_prefix = prefix;
        dump_iters = iters;
        next_dump_idx = 0;
        dump_enabled = !dump_iters.empty();
    }

    void set_max_iter(const int max_iter)
    {
        max_iter_override = max_iter;
    }

    int minimize(
        const std::string& outFileName_in,
        const Real eps = 1e-5,
        const int Mval = 10)
    {
        outFileName = outFileName_in;
        scatterReducedToMesh(reducedVariables.data());
        mesh.updateDeformedConfiguration();
        const Real energy0 = op.compute(mesh);

        double parameter[20];
        int info[20];
        default_setup(parameter, info, eps, verbose);
        if(max_iter_override > 0)
            info[4] = max_iter_override;

        const int ret = HLBFGS(
            static_cast<int>(freeIndices.size()),
            Mval,
            reducedVariables.data(),
            HLBFGS_Methods::evaluate,
            0,
            HLBFGS_UPDATE_Hessian,
            HLBFGS_Methods::newiteration,
            this,
            parameter,
            info);

        last_retcode = ret;
        last_iterations = info[2];
        last_evaluations = info[1];

        scatterReducedToMesh(reducedVariables.data());
        mesh.updateDeformedConfiguration();
        const Real energy1 = op.compute(mesh);

        if(verbose)
        {
            std::cout << "HLBFGS reduced return value = " << ret << std::endl;
            std::printf(
                "Reduced energy went from %10.10e to %10.10e, "
                "using epsilon = %e, final eps = %e, free/full = %zu/%d\n",
                energy0, energy1, eps, last_gnorm,
                freeIndices.size(), nVariablesFull);
        }

        if(FILE* file = std::fopen(outFileName.c_str(), "a"))
        {
            std::fprintf(
                file,
                "# ---- reduced DOF solve ----\n"
                "%d\t%d\t%d\t%10.10e\t%10.10e\t%10.10e\t%zu\t%d\n"
                "# ----\n",
                ret, info[2], info[1], energy0, energy1, last_gnorm,
                freeIndices.size(), nVariablesFull);
            std::fclose(file);
        }

        return (ret == 2 || ret == 3 || ret == 4) ? 0 : 1;
    }

    void evaluate(
        int N,
        double* x,
        double* /*prev_x*/,
        double* f,
        double* g) override
    {
        if(N != static_cast<int>(freeIndices.size()))
            throw std::runtime_error(
                "HLBFGS_Energy_Reduced: callback variable count mismatch.");

        scatterReducedToMesh(x);

        profiler.push_start("update mesh");
        mesh.updateDeformedConfiguration();
        profiler.pop_stop();

        fullGradient.setZero();
        profiler.push_start("compute energy");
        *f = op.compute(mesh, fullGradient);
        profiler.pop_stop();

        for(std::size_t k = 0; k < freeIndices.size(); ++k)
            g[k] = fullGradient(freeIndices[k]);
    }

    void newiteration(
        int iter,
        int call_iter,
        double* x,
        double* f,
        double* /*g*/,
        double* gnorm) override
    {
        last_gnorm = *gnorm;
        scatterReducedToMesh(x);

        if(dump_enabled && next_dump_idx < dump_iters.size() &&
           iter >= dump_iters[next_dump_idx])
        {
            mesh.updateDeformedConfiguration();
            const auto vertices =
                mesh.getCurrentConfiguration().getVertices();
            const auto faces =
                mesh.getTopology().getFace2Vertices();
            WriteVTK writer(vertices, faces);
            writer.write(
                dump_prefix + "_mesh_iter_" +
                std::to_string(dump_iters[next_dump_idx]));
            ++next_dump_idx;
        }

        if(iter % 1000 == 0 && verbose)
        {
            std::printf(
                "%d:\t%d\t%10.10e\t%10.10e\n",
                iter, call_iter, *f, *gnorm);
            op.printProfilerSummary();
            profiler.printSummary();
        }

        if(iter % 100 == 0)
        {
            if(FILE* file = std::fopen(outFileName.c_str(), "a"))
            {
                std::fprintf(
                    file, "%d\t%d\t%10.10e\t%10.10e\n",
                    iter, call_iter, *f, *gnorm);
                std::fclose(file);
            }
        }
    }

    Real get_lastnorm() const
    {
        return last_gnorm;
    }

    /// Raw HLBFGS termination code: 1 line-search failure, 2/3 tolerance met,
    /// 4 step outside [stpmin,stpmax] (stagnation), 5 iteration cap exceeded.
    int get_lastreturncode() const
    {
        return last_retcode;
    }

    /// Iterations taken by the last minimize() call.
    int get_lastiterations() const
    {
        return last_iterations;
    }

    /// Energy/gradient evaluations spent by the last minimize() call (HLBFGS info[1]).
    int get_lastevaluations() const
    {
        return last_evaluations;
    }

    int getNumberOfFullVariables() const
    {
        return nVariablesFull;
    }

    int getNumberOfFreeVariables() const
    {
        return static_cast<int>(freeIndices.size());
    }

    int getNumberOfFixedVariables() const
    {
        return static_cast<int>(fixedIndices.size());
    }
};

} // namespace HLBFGS_Methods

#endif // USEHLBFGS

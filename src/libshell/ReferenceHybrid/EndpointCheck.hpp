#ifndef SHELL_NATIVE_ENDPOINT_CHECK_HPP
#define SHELL_NATIVE_ENDPOINT_CHECK_HPP

#include "ShellEquilibriumSolver.hpp"
#include <Eigen/SparseCholesky>
#include <Eigen/QR>
#include <limits>

// Diagnostic only. No optimizer steps, shifts, pseudoinverses, or gate changes.
// Sparse coordinate gauge, NOT the dense orthogonal gauge of the first prototype.
// Select exactly rank(R) coordinate conditions transverse to the admissible rigid
// modes R. This removes pose freedoms locally, not additional deformation DOFs.
// Away from stationarity different gauges need not give identical corrections.
namespace ShellNativeEndpointCheck
{
struct Report
{
    std::string status = "invalid_input";
    std::vector<int> gaugeDofs, freeDofs;
    double minPivot = std::numeric_limits<double>::quiet_NaN();
    double maxAbsDiagonal = std::numeric_limits<double>::quiet_NaN();
    double decrement = std::numeric_limits<double>::quiet_NaN();
    double predictedEnergyDrop = std::numeric_limits<double>::quiet_NaN();
    double relativeSolveResidual = std::numeric_limits<double>::quiet_NaN();
    Eigen::VectorXd correction;
};

// Diagonal-scaled, absolute-floor-regularized residual and decrement acceptance criterion.
// Evaluates ||Ay+b|| / (||b|| + maxAbsDiagonal*||y|| + 1e-16) <= 1e-6 and dotVal >= -1e-12*(||b||*||y|| + 1e-20).
inline std::string evaluateSolveAcceptance(
    const double residualNorm,
    const double bNorm,
    const double yNorm,
    const double maxAbsDiagonal,
    const double dotVal,
    double& relativeSolveResidual,
    double& decrementSquared)
{
    const double denom = bNorm + maxAbsDiagonal * yNorm + 1e-16;
    if(!std::isfinite(denom) || denom <= 0.0)
    {
        relativeSolveResidual = std::numeric_limits<double>::quiet_NaN();
        decrementSquared = 0.0;
        return "linear_solve_unresolved";
    }
    relativeSolveResidual = residualNorm / denom;
    if(!std::isfinite(dotVal) || dotVal < -1e-12 * (bNorm * yNorm + 1e-20))
    {
        decrementSquared = 0.0;
        return "linear_solve_unresolved";
    }
    decrementSquared = std::max(0.0, dotVal);
    if(!std::isfinite(relativeSolveResidual) || relativeSolveResidual > 1e-6)
    {
        return "linear_solve_unresolved";
    }
    return "accepted";
}

inline Report inspect(const Eigen::SparseMatrix<double>& H, const Eigen::VectorXd& g,
                      const Eigen::VectorXd& scales,
                      const ShellEquilibrium::RigidModeProjector& projector,
                      const double energyReference)
{
    Report out;
    const int n=g.size();
    if(n<=0 || H.rows()!=n || H.cols()!=n || scales.size()!=n || projector.nDofs()!=n ||
       !g.allFinite() || !scales.allFinite() || scales.minCoeff()<=0 ||
       !std::isfinite(energyReference) || energyReference<=0) return out;
    for(int j=0;j<H.outerSize();++j)
        for(Eigen::SparseMatrix<double>::InnerIterator it(H,j);it;++it)
            if(!std::isfinite(it.value())) return out;

    std::vector<char> fixed=projector.mask();
    const int k=projector.numberOfRigidModes();
    if(k>0)
    {
        const Eigen::MatrixXd R=scales.cwiseInverse().asDiagonal()*projector.basis();
        Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(R.transpose());
        qr.setThreshold(1e-10);
        if(qr.rank()!=k) { out.status="unresolved_rigid_gauge"; return out; }
        for(int j=0;j<k;++j)
        {
            const int i=qr.colsPermutation().indices()[j];
            if(fixed[i]) { out.status="unresolved_rigid_gauge"; return out; }
            fixed[i]=1; out.gaugeDofs.push_back(i);
        }
    }
    std::vector<int> reduced(n,-1);
    for(int i=0;i<n;++i) if(!fixed[i])
    { reduced[i]=out.freeDofs.size(); out.freeDofs.push_back(i); }
    const int m=out.freeDofs.size();
    if(m==0) { out.status="no_free_dofs"; return out; }
    std::vector<Eigen::Triplet<double>> entries;
    entries.reserve(H.nonZeros());
    for(int j=0;j<H.outerSize();++j)
        for(Eigen::SparseMatrix<double>::InnerIterator it(H,j);it;++it)
            if(reduced[it.row()]>=0 && reduced[it.col()]>=0)
                entries.emplace_back(reduced[it.row()],reduced[it.col()],
                    it.value()*scales[it.row()]*scales[it.col()]/energyReference);
    Eigen::SparseMatrix<double> A(m,m); A.setFromTriplets(entries.begin(),entries.end());
    Eigen::VectorXd b(m);
    for(int i=0;i<m;++i) b[i]=scales[out.freeDofs[i]]*g[out.freeDofs[i]]/energyReference;
    if(!b.allFinite()) return out;
    for(int j=0;j<A.outerSize();++j)
        for(Eigen::SparseMatrix<double>::InnerIterator it(A,j);it;++it)
            if(!std::isfinite(it.value())) return out;
    const Eigen::SparseMatrix<double> asym=A-Eigen::SparseMatrix<double>(A.transpose());
    if(asym.norm()>1e-10*A.norm()) { out.status="asymmetric_hessian"; return out; }
    A=0.5*(A+Eigen::SparseMatrix<double>(A.transpose()));
    out.maxAbsDiagonal=A.diagonal().cwiseAbs().maxCoeff();
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
    solver.compute(A);
    if(solver.info()!=Eigen::Success || !solver.vectorD().allFinite())
    { out.status="factorization_unresolved"; return out; }
    out.minPivot=solver.vectorD().minCoeff();
    const double floor=1e-12*out.maxAbsDiagonal;
    if(out.minPivot < -floor) { out.status="indefinite"; return out; }
    if(out.minPivot <= floor) { out.status="unresolved_curvature"; return out; }
    const Eigen::VectorXd y=solver.solve(-b);
    if(solver.info()!=Eigen::Success || !y.allFinite())
    { out.status="linear_solve_unresolved"; return out; }
    const double bNorm = b.norm();
    const double yNorm = y.norm();
    const double residualNorm = (A*y+b).norm();
    const double dotVal = -b.dot(y);
    double decrementSquared = 0.0;
    const std::string solveStatus = evaluateSolveAcceptance(
        residualNorm, bNorm, yNorm, out.maxAbsDiagonal, dotVal,
        out.relativeSolveResidual, decrementSquared);
    if(solveStatus != "accepted")
    {
        out.status = solveStatus;
        return out;
    }
    out.correction=Eigen::VectorXd::Zero(n);
    for(int i=0;i<m;++i) out.correction[out.freeDofs[i]]=scales[out.freeDofs[i]]*y[i];
    out.decrement=std::sqrt(decrementSquared);
    out.predictedEnergyDrop=0.5*energyReference*decrementSquared;
    if(!out.correction.allFinite() || !std::isfinite(out.predictedEnergyDrop))
    { out.status="linear_solve_unresolved"; return out; }
    out.status="positive_pivots"; // numerical local evidence; NOT global/convergence certification
    return out;
}
}
#endif

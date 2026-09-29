#ifndef SHIFTED_NEWTON_FACTORIZATION_HPP
#define SHIFTED_NEWTON_FACTORIZATION_HPP

#include <Eigen/SparseCholesky>
#include <cmath>
#include <stdexcept>

namespace NonconvexBenchmark
{
// Shared by the candidate engine and the frozen-matrix profiler. LLT stops at
// a nonpositive pivot; LDLT normally finishes even for indefinite matrices.
// Both use Eigen's default AMD ordering and the same strict pivot floor.
class ShiftedNewtonFactorization
{
    using Matrix=Eigen::SparseMatrix<double>;
    Eigen::SimplicialLLT<Matrix> llt;
    Eigen::SimplicialLDLT<Matrix> ldlt;
    bool early, factored=false;
public:
    explicit ShiftedNewtonFactorization(bool earlyCholesky=true):early(earlyCholesky){}
    void analyzePattern(const Matrix& A)
    {factored=false;if(early)llt.analyzePattern(A);else ldlt.analyzePattern(A);}
    void factorize(const Matrix& A)
    {factored=true;if(early)llt.factorize(A);else ldlt.factorize(A);}
    bool successful() const
    {return factored && (early?llt.info():ldlt.info())==Eigen::Success;}
    bool acceptable(double floor) const
    {
        if(!successful() || !std::isfinite(floor) || floor<=0)return false;
        if(!early)return ldlt.vectorD().allFinite() && ldlt.vectorD().size()>0 && ldlt.vectorD().minCoeff()>floor;
        // LLT diagonal squared is the corresponding unpivoted LDLT pivot in
        // exact arithmetic. Do not accept merely because factorization succeeds.
        const auto& L=llt.matrixL().nestedExpression();
        if(L.rows()==0)return false;
        for(int i=0;i<L.rows();++i)
        {
            const double d=L.coeff(i,i),pivot=d*d;
            if(!(d>0) || !std::isfinite(pivot) || !(pivot>floor))return false;
        }
        return true;
    }
    long long factorNonzeros() const
    {
        if(!successful())return -1;
        return early?llt.matrixL().nestedExpression().nonZeros():ldlt.matrixL().nestedExpression().nonZeros();
    }
    double columnSquareWorkProxy() const
    {
        if(!successful())return -1;
        const auto& L=early?llt.matrixL().nestedExpression():ldlt.matrixL().nestedExpression();
        double work=0;
        for(int j=0;j<L.cols();++j){double count=L.outerIndexPtr()[j+1]-L.outerIndexPtr()[j];work+=count*count;}
        return work; // structural proxy, not measured floating-point operations
    }
    Eigen::VectorXd solve(const Eigen::VectorXd& rhs) const
    {
        if(!successful())throw std::logic_error("solve requested without successful factorization");
        if(early)return llt.solve(rhs);
        return ldlt.solve(rhs);
    }
};
}
#endif

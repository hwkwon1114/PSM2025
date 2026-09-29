#ifndef BENCHMARK_NEWTON_FACTORIZATION_HPP
#define BENCHMARK_NEWTON_FACTORIZATION_HPP
#include "ShiftedNewtonFactorization.hpp"
#ifdef NONCONVEX_WITH_CHOLMOD
#include "CholmodFrozenFactorization.hpp"
#endif
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
namespace NonconvexBenchmark {
// Benchmark-only adapter. External Eigen AMD is retained for CHOLMOD; numeric
// values are remapped into a fixed CSC pattern, without inferring a shift.
class BenchmarkNewtonFactorization {
    using Matrix=Eigen::SparseMatrix<double>;
    ShiftedNewtonFactorization eigen;
    bool useCholmod;
    int threads_ = 1;
#ifdef NONCONVEX_WITH_CHOLMOD
    CholmodFrozenFactorization cholmod;
    Matrix ordered;
    std::vector<int> order,mapping,outer,inner;
#endif
public:
    explicit BenchmarkNewtonFactorization(const std::string& backend, int threads = 1)
        : useCholmod(backend=="cholmod"), threads_(std::max(1, threads))
#ifdef NONCONVEX_WITH_CHOLMOD
        , cholmod(threads)
#endif
    {
        if(backend!="eigen" && backend!="cholmod")throw std::invalid_argument("unknown Newton backend");
#ifndef NONCONVEX_WITH_CHOLMOD
        if(useCholmod)throw std::invalid_argument("CHOLMOD not compiled into benchmark");
#endif
    }
    void setThreads(int threads) {
        threads_ = std::max(1, threads);
#ifdef NONCONVEX_WITH_CHOLMOD
        if(useCholmod) cholmod.setThreads(threads_);
#endif
    }
    int threads() const { return threads_; }
    void analyzePattern(const Matrix& A) {
        if(!useCholmod){eigen.analyzePattern(A);return;}
#ifdef NONCONVEX_WITH_CHOLMOD
        Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic,int> P;Eigen::AMDOrdering<int>()(A,P);
        const int n=A.rows();order.resize(n);std::vector<int> inverse(n);
        for(int i=0;i<n;++i){order[i]=P.indices()[i];inverse[order[i]]=i;}
        outer.assign(A.outerIndexPtr(),A.outerIndexPtr()+n+1);inner.assign(A.innerIndexPtr(),A.innerIndexPtr()+A.nonZeros());
        std::vector<Eigen::Triplet<double>> t;t.reserve(A.nonZeros());
        for(int c=0;c<n;++c)for(Matrix::InnerIterator it(A,c);it;++it)t.emplace_back(inverse[it.row()],inverse[c],it.value());
        ordered.resize(n,n);ordered.setFromTriplets(t.begin(),t.end());mapping.resize(A.nonZeros());
        for(int c=0;c<n;++c)for(int k=outer[c];k<outer[c+1];++k){
            const int col=inverse[c],row=inverse[inner[k]];
            auto begin=ordered.innerIndexPtr()+ordered.outerIndexPtr()[col],end=ordered.innerIndexPtr()+ordered.outerIndexPtr()[col+1];
            auto found=std::lower_bound(begin,end,row);
            if(found==end || *found!=row)throw std::logic_error("permuted CSC entry missing");
            mapping[k]=found-ordered.innerIndexPtr();
        }
        cholmod.analyzePattern(ordered);
#endif
    }
    void factorize(const Matrix& A) {
        if(!useCholmod){eigen.factorize(A);return;}
#ifdef NONCONVEX_WITH_CHOLMOD
        if(A.rows()!=ordered.rows() || A.cols()!=ordered.cols() || !A.isCompressed() || size_t(A.nonZeros())!=inner.size() ||
           !std::equal(outer.begin(),outer.end(),A.outerIndexPtr()) || !std::equal(inner.begin(),inner.end(),A.innerIndexPtr()))
            throw std::invalid_argument("CHOLMOD CSC pattern changed after analysis");
        for(int k=0;k<A.nonZeros();++k)ordered.valuePtr()[mapping[k]]=A.valuePtr()[k];
        cholmod.factorize(ordered);
#endif
    }
    bool successful()const {
#ifdef NONCONVEX_WITH_CHOLMOD
        if(useCholmod)return cholmod.successful();
#endif
        return eigen.successful();
    }
    bool acceptable(double floor)const {
#ifdef NONCONVEX_WITH_CHOLMOD
        if(useCholmod)return cholmod.acceptable(floor);
#endif
        return eigen.acceptable(floor);
    }
    Eigen::VectorXd solve(const Eigen::VectorXd& b) {
#ifdef NONCONVEX_WITH_CHOLMOD
        if(useCholmod){Eigen::VectorXd rhs(b.size()),x(b.size());for(int i=0;i<b.size();++i)rhs[i]=b[order[i]];
            const Eigen::VectorXd y=cholmod.solve(rhs);for(int i=0;i<b.size();++i)x[order[i]]=y[i];return x;}
#endif
        return eigen.solve(b);
    }
    long long storedValues()const {
#ifdef NONCONVEX_WITH_CHOLMOD
        if(useCholmod)return cholmod.storedValues();
#endif
        return eigen.factorNonzeros();
    }
    double columnSquareWorkProxy()const {
#ifdef NONCONVEX_WITH_CHOLMOD
        if(useCholmod)return cholmod.lowerColumnSquareProxy();
#endif
        return eigen.columnSquareWorkProxy();
    }
};
// Process-local symbolic cache only. Numeric factors are always refreshed.
// Exact restricted-DOF order AND CSC indices are required; equal nnz is not enough.
class BenchmarkNewtonSymbolicCache {
    using Matrix=Eigen::SparseMatrix<double>;
    std::string backend;
    int threads_ = 1;
    std::unique_ptr<BenchmarkNewtonFactorization> factor;
    std::vector<int> freeIds,outer,inner;
    int dimension=-1;
public:
    explicit BenchmarkNewtonSymbolicCache(const std::string& name, int threads = 1):backend(name), threads_(std::max(1, threads)){}
    void setThreads(int threads) {
        threads_ = std::max(1, threads);
        if(factor) factor->setThreads(threads_);
    }
    int threads() const { return threads_; }
    bool prepare(const Matrix& A,const std::vector<int>& free) {
        if(!A.isCompressed() || A.rows()!=A.cols() || A.rows()<=0 || size_t(A.rows())!=free.size())
            throw std::invalid_argument("symbolic cache requires square compressed restriction");
        const bool same=factor && dimension==A.rows() && freeIds==free && inner.size()==size_t(A.nonZeros()) &&
            std::equal(outer.begin(),outer.end(),A.outerIndexPtr()) && std::equal(inner.begin(),inner.end(),A.innerIndexPtr());
        if(same)return false;
        // Discard the old state before attempting analysis; failure cannot expose stale factors.
        factor.reset();dimension=-1;freeIds.clear();outer.clear();inner.clear();
        auto fresh=std::make_unique<BenchmarkNewtonFactorization>(backend, threads_);fresh->analyzePattern(A);
        freeIds=free;dimension=A.rows();outer.assign(A.outerIndexPtr(),A.outerIndexPtr()+A.cols()+1);
        inner.assign(A.innerIndexPtr(),A.innerIndexPtr()+A.nonZeros());factor=std::move(fresh);return true;
    }
    BenchmarkNewtonFactorization& get(){if(!factor)throw std::logic_error("symbolic cache unprepared");return *factor;}
};
}
#endif

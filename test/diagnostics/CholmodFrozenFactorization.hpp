#ifndef CHOLMOD_FROZEN_FACTORIZATION_HPP
#define CHOLMOD_FROZEN_FACTORIZATION_HPP
// Opt-in isolated benchmark backend; never included by production solvers.
#include <cholmod.h>
#include <Eigen/SparseCore>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
class CholmodFrozenFactorization
{
    cholmod_common common{};
    cholmod_factor* factor=nullptr;
    bool completed=false;
    int threads_=1;
    static cholmod_sparse view(const Eigen::SparseMatrix<double>& A)
    {
        if(!A.isCompressed() || A.rows()!=A.cols())throw std::invalid_argument("CHOLMOD needs square compressed CSC");
        cholmod_sparse v{};v.nrow=A.rows();v.ncol=A.cols();v.nzmax=A.nonZeros();
        v.p=const_cast<int*>(A.outerIndexPtr());v.i=const_cast<int*>(A.innerIndexPtr());v.x=const_cast<double*>(A.valuePtr());
        v.stype=-1;v.itype=CHOLMOD_INT;v.xtype=CHOLMOD_REAL;v.dtype=CHOLMOD_DOUBLE;v.sorted=1;v.packed=1;return v;
    }
public:
    explicit CholmodFrozenFactorization(int nthreads = 1)
    {
        if(!cholmod_start(&common))throw std::runtime_error("cholmod_start failed");
        common.supernodal=CHOLMOD_SUPERNODAL;common.final_ll=1;common.final_super=1;
        common.nmethods=1;common.method[0].ordering=CHOLMOD_NATURAL;common.postorder=0;
        common.quick_return_if_not_posdef=1;common.dbound=0;common.useGPU=0;common.print=0;
        setThreads(nthreads);
    }
    void setThreads(int nthreads)
    {
        threads_ = std::max(1, nthreads);
        common.nthreads_max = threads_;
    }
    int threads() const { return threads_; }
    double chunk() const { return common.chunk; }
    void setChunk(double c) { common.chunk = c; }
    ~CholmodFrozenFactorization(){if(factor)cholmod_free_factor(&factor,&common);cholmod_finish(&common);}
    CholmodFrozenFactorization(const CholmodFrozenFactorization&)=delete;
    CholmodFrozenFactorization& operator=(const CholmodFrozenFactorization&)=delete;
    void analyzePattern(const Eigen::SparseMatrix<double>& A)
    {
        completed=false;if(factor)cholmod_free_factor(&factor,&common);auto a=view(A);factor=cholmod_analyze(&a,&common);
        if(!factor || common.status!=CHOLMOD_OK || !factor->is_super || factor->itype!=CHOLMOD_INT)
            throw std::runtime_error("CHOLMOD supernodal analysis failed");
        const auto p=static_cast<const int*>(factor->Perm);
        for(size_t i=0;i<factor->n;++i)if(p && p[i]!=int(i))throw std::runtime_error("unexpected CHOLMOD permutation");
    }
    void factorize(const Eigen::SparseMatrix<double>& A)
    {
        completed=false;if(!factor)throw std::logic_error("missing CHOLMOD analysis");
        for(int j=0;j<A.nonZeros();++j)if(!std::isfinite(A.valuePtr()[j]))return;
        auto a=view(A);common.status=CHOLMOD_OK;int ok=cholmod_factorize(&a,factor,&common);
        if(common.status<CHOLMOD_OK)throw std::runtime_error("CHOLMOD numeric error "+std::to_string(common.status));
        if(common.ndbounds_hit!=0)throw std::runtime_error("CHOLMOD changed diagonal bounds");
        completed=ok && common.status==CHOLMOD_OK && factor->minor==factor->n;
    }
    std::vector<double> pivots() const
    {
        if(!completed || !factor->is_super || !factor->is_ll || factor->xtype!=CHOLMOD_REAL || factor->dtype!=CHOLMOD_DOUBLE)
            throw std::logic_error("no successful double supernodal LLT");
        const auto super=static_cast<const int*>(factor->super),pi=static_cast<const int*>(factor->pi),px=static_cast<const int*>(factor->px);
        const auto x=static_cast<const double*>(factor->x);std::vector<double> d(factor->n,-1);
        for(size_t k=0;k<factor->nsuper;++k){const int cols=super[k+1]-super[k],rows=pi[k+1]-pi[k];
            if(cols<=0 || rows<cols)throw std::runtime_error("invalid supernode dimensions");
            for(int j=0;j<cols;++j){const size_t index=size_t(px[k])+size_t(j)*(rows+1);
                if(index>=factor->xsize || super[k]+j>=int(factor->n))throw std::runtime_error("invalid supernode offset");
                const double value=x[index];d[super[k]+j]=(value>0)?value*value:-1;}}
        return d;
    }
    bool acceptable(double floor) const
    {
        if(!completed || !std::isfinite(floor) || floor<=0)return false;
        for(double d:pivots())
        {
            if(!std::isfinite(d) || !(d>floor)) return false;
        }
        return true;
    }
    Eigen::VectorXd solve(const Eigen::VectorXd& b)
    {
        if(!completed || size_t(b.size())!=factor->n)throw std::logic_error("invalid CHOLMOD solve");
        cholmod_dense rhs{};rhs.nrow=b.size();rhs.ncol=1;rhs.nzmax=b.size();rhs.d=b.size();rhs.x=const_cast<double*>(b.data());rhs.xtype=CHOLMOD_REAL;rhs.dtype=CHOLMOD_DOUBLE;
        cholmod_dense* result=cholmod_solve(CHOLMOD_A,factor,&rhs,&common);
        if(!result || common.status!=CHOLMOD_OK){if(result)cholmod_free_dense(&result,&common);throw std::runtime_error("CHOLMOD solve failed");}
        Eigen::VectorXd x=Eigen::Map<const Eigen::VectorXd>(static_cast<const double*>(result->x),b.size());cholmod_free_dense(&result,&common);return x;
    }
    bool successful()const{return completed;}
    size_t supernodes()const{return factor?factor->nsuper:0;}
    size_t storedValues()const{return completed?factor->xsize:0;}
    double lowerColumnSquareProxy()const
    {
        if(!completed)return 0;
        double total=0;
        const auto s=static_cast<const int*>(factor->super),p=static_cast<const int*>(factor->pi);
        for(size_t k=0;k<factor->nsuper;++k)
        {
            for(int j=0;j<s[k+1]-s[k];++j)
            {
                double length=p[k+1]-p[k]-j;
                total+=length*length;
            }
        }
        return total;
    }
};
#endif

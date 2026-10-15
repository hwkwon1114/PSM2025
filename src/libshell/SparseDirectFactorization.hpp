#ifndef SPARSE_DIRECT_FACTORIZATION_HPP
#define SPARSE_DIRECT_FACTORIZATION_HPP

#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>
#include <Eigen/OrderingMethods>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef NONCONVEX_WITH_CHOLMOD
#include <cholmod.h>
#endif

namespace ShellEquilibrium
{
    /*! Sparse SPD factorization used by the assembled-Hessian Newton path.
     *
     * CHOLMOD is selected when the project is built with ENABLE_CHOLMOD; otherwise
     * Eigen's SimplicialLDLT is used. The caller owns regularization and must inspect
     * the positive-pivot gate before using the factor to solve.
     */
    class SparseDirectFactorization
    {
        using Matrix = Eigen::SparseMatrix<double>;

#ifdef NONCONVEX_WITH_CHOLMOD
        cholmod_common common_{};
        cholmod_factor *factor_ = nullptr;
        bool started_ = false;
        bool complete_ = false;

        static cholmod_sparse view(const Matrix &A)
        {
            if(!A.isCompressed() || A.rows() != A.cols())
                throw std::invalid_argument("CHOLMOD requires a square compressed CSC matrix");
            cholmod_sparse result{};
            result.nrow = static_cast<size_t>(A.rows());
            result.ncol = static_cast<size_t>(A.cols());
            result.nzmax = static_cast<size_t>(A.nonZeros());
            result.p = const_cast<int *>(A.outerIndexPtr());
            result.i = const_cast<int *>(A.innerIndexPtr());
            result.x = const_cast<double *>(A.valuePtr());
            result.stype = -1; // use the lower triangle of the symmetric matrix
            result.itype = CHOLMOD_INT;
            result.xtype = CHOLMOD_REAL;
            result.dtype = CHOLMOD_DOUBLE;
            result.sorted = 1;
            result.packed = 1;
            return result;
        }
#endif

#ifndef NONCONVEX_WITH_CHOLMOD
        Eigen::SimplicialLDLT<Matrix, Eigen::Lower, Eigen::AMDOrdering<int>> eigen_;
#endif

    public:
        explicit SparseDirectFactorization(const int threads = 1)
        {
#ifdef NONCONVEX_WITH_CHOLMOD
            if(!cholmod_start(&common_))
                throw std::runtime_error("cholmod_start failed");
            started_ = true;
            common_.supernodal = CHOLMOD_SUPERNODAL;
            common_.final_ll = 1;
            common_.final_super = 1;
            common_.nmethods = 1;
            common_.method[0].ordering = CHOLMOD_AMD;
            common_.postorder = 1;
            common_.quick_return_if_not_posdef = 1;
            common_.dbound = 0.0;
            common_.print = 0;
            setThreads(threads);
#else
            (void)threads;
#endif
        }

        ~SparseDirectFactorization()
        {
#ifdef NONCONVEX_WITH_CHOLMOD
            if(factor_) cholmod_free_factor(&factor_, &common_);
            if(started_) cholmod_finish(&common_);
#endif
        }

        SparseDirectFactorization(const SparseDirectFactorization &) = delete;
        SparseDirectFactorization &operator=(const SparseDirectFactorization &) = delete;

        void setThreads(const int threads)
        {
#if defined(NONCONVEX_WITH_CHOLMOD) && defined(CHOLMOD_MAIN_VERSION) && (CHOLMOD_MAIN_VERSION >= 4)
            common_.nthreads_max = std::max(1, threads);
#else
            (void)threads;
#endif
        }

        void analyzePattern(const Matrix &A)
        {
#ifdef NONCONVEX_WITH_CHOLMOD
            complete_ = false;
            if(factor_) cholmod_free_factor(&factor_, &common_);
            auto input = view(A);
            common_.status = CHOLMOD_OK;
            factor_ = cholmod_analyze(&input, &common_);
            if(!factor_ || common_.status != CHOLMOD_OK || !factor_->is_super || factor_->itype != CHOLMOD_INT)
                throw std::runtime_error("CHOLMOD symbolic analysis failed");
#else
            eigen_.analyzePattern(A);
            if(eigen_.info() != Eigen::Success)
                throw std::runtime_error("Eigen sparse symbolic analysis failed");
#endif
        }

        bool factorize(const Matrix &A, const double pivotFloor)
        {
#ifdef NONCONVEX_WITH_CHOLMOD
            complete_ = false;
            if(!factor_) throw std::logic_error("CHOLMOD factorization has no symbolic analysis");
            if(!A.isCompressed() || A.rows() != A.cols())
                throw std::invalid_argument("CHOLMOD requires a square compressed CSC matrix");
            for(int k = 0; k < A.nonZeros(); ++k)
                if(!std::isfinite(A.valuePtr()[k])) return false;
            auto input = view(A);
            common_.status = CHOLMOD_OK;
            const int ok = cholmod_factorize(&input, factor_, &common_);
            if(common_.status < CHOLMOD_OK)
                throw std::runtime_error("CHOLMOD numeric factorization failed: " + std::to_string(common_.status));
            complete_ = ok && common_.status == CHOLMOD_OK && factor_->minor == factor_->n;
            if(!complete_ || !factor_->is_super || !factor_->is_ll || factor_->xtype != CHOLMOD_REAL || factor_->dtype != CHOLMOD_DOUBLE)
                return false;

            const auto super = static_cast<const int *>(factor_->super);
            const auto pi = static_cast<const int *>(factor_->pi);
            const auto px = static_cast<const int *>(factor_->px);
            const auto values = static_cast<const double *>(factor_->x);
            for(size_t k = 0; k < factor_->nsuper; ++k)
            {
                const int columns = super[k + 1] - super[k];
                const int rows = pi[k + 1] - pi[k];
                if(columns <= 0 || rows < columns) return false;
                for(int j = 0; j < columns; ++j)
                {
                    const size_t index = static_cast<size_t>(px[k]) + static_cast<size_t>(j) * (rows + 1);
                    if(index >= factor_->xsize) return false;
                    const double diagonal = values[index];
                    if(!std::isfinite(diagonal) || !(diagonal > 0.0) ||
                       !std::isfinite(diagonal * diagonal) || diagonal * diagonal <= pivotFloor)
                        return false;
                }
            }
            return true;
#else
            for(int k = 0; k < A.nonZeros(); ++k)
                if(!std::isfinite(A.valuePtr()[k])) return false;
            eigen_.factorize(A);
            if(eigen_.info() != Eigen::Success) return false;
            const auto &diagonal = eigen_.vectorD();
            if(diagonal.size() != A.rows()) return false;
            for(Eigen::Index i = 0; i < diagonal.size(); ++i)
                if(!std::isfinite(diagonal[i]) || !(diagonal[i] > pivotFloor)) return false;
            return true;
#endif
        }

        Eigen::VectorXd solve(const Eigen::VectorXd &rhs)
        {
#ifdef NONCONVEX_WITH_CHOLMOD
            if(!complete_ || rhs.size() != static_cast<Eigen::Index>(factor_->n))
                throw std::logic_error("invalid CHOLMOD solve request");
            cholmod_dense input{};
            input.nrow = static_cast<size_t>(rhs.size());
            input.ncol = 1;
            input.nzmax = static_cast<size_t>(rhs.size());
            input.d = static_cast<size_t>(rhs.size());
            input.x = const_cast<double *>(rhs.data());
            input.xtype = CHOLMOD_REAL;
            input.dtype = CHOLMOD_DOUBLE;
            cholmod_dense *solution = cholmod_solve(CHOLMOD_A, factor_, &input, &common_);
            if(!solution || common_.status != CHOLMOD_OK)
            {
                if(solution) cholmod_free_dense(&solution, &common_);
                throw std::runtime_error("CHOLMOD triangular solve failed");
            }
            Eigen::VectorXd result = Eigen::Map<const Eigen::VectorXd>(
                static_cast<const double *>(solution->x), rhs.size());
            cholmod_free_dense(&solution, &common_);
            return result;
#else
            if(eigen_.info() != Eigen::Success)
                throw std::logic_error("Eigen sparse solve requested without a successful factorization");
            return eigen_.solve(rhs);
#endif
        }
    };
}

#endif

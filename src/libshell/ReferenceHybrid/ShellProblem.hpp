#ifndef SHELL_NATIVE_SHELLPROBLEM_HPP
#define SHELL_NATIVE_SHELLPROBLEM_HPP
#include "Core.hpp"
#include "Hessian.hpp"
#ifdef SHELL_NATIVE_CACHED_TINYAD
#include "CachedTinyADHessian_Bilayer.hpp"
#endif
#ifdef SHELL_NATIVE_LOCAL_DIHEDRAL
#include "Hessian.hpp"
#endif
#include <Eigen/QR>
#include <memory>

namespace ShellNativeHybrid
{
// Caller owns a fixed material/target/geometry bundle. Its independently verified
// content identity must be supplied; an arbitrary label is insufficient for a
// production restart. Benchmark engines do not mutate the rest targets.
template<class Mesh,class Energy>
class ShellProblem final : public Problem
{
    Mesh& mesh;
    Energy& energy;
    NativeHybridHessian_Bilayer<Mesh> ad;
#ifdef SHELL_NATIVE_CACHED_TINYAD
    std::unique_ptr<CachedTinyADHessian_Bilayer<Mesh>> cachedAd;
#endif
#ifdef SHELL_NATIVE_LOCAL_DIHEDRAL
    std::unique_ptr<NativeHybridHessian_Bilayer<Mesh>> localDihedralAd;
#endif
    std::string id;
    Vec scale, fixed, restArea;
    std::vector<char> mask;
    double reference;
public:
    ShellProblem(Mesh& m,Energy& e,double young,double nu,double h,std::string identity,
                 std::string assembly="triplets",int assemblyThreads=1)
        :mesh(m),energy(e),ad(young,nu,h),id(std::move(identity))
    {
        if(assemblyThreads<1 || (assemblyThreads!=1 && assembly!="local_dihedral_csc"))
            throw std::invalid_argument("parallel shell assembly requires local-dihedral Hessian and positive threads");
        if(assembly!="triplets" && assembly!="cached_csc" && assembly!="local_dihedral_csc")
            throw std::invalid_argument("unknown shell Hessian assembly");
        if(assembly=="cached_csc")
        {
#ifdef SHELL_NATIVE_CACHED_TINYAD
            cachedAd=std::make_unique<CachedTinyADHessian_Bilayer<Mesh>>(young,nu,h);
#else
            throw std::invalid_argument("cached shell Hessian not compiled into benchmark");
#endif
        }
        if(assembly=="local_dihedral_csc")
        {
#ifdef SHELL_NATIVE_LOCAL_DIHEDRAL
            localDihedralAd=std::make_unique<NativeHybridHessian_Bilayer<Mesh>>(young,nu,h,assemblyThreads);
#else
            throw std::invalid_argument("local-dihedral shell Hessian not compiled into benchmark");
#endif
        }
        const int n=size(),nv=mesh.getNumberOfVertices(),nf=mesh.getNumberOfFaces();
        fixed=Eigen::Map<const Vec>(mesh.getDataPointer(),n);
        mask=ShellEquilibrium::constrainedDofMask(mesh);
        const auto X=mesh.getRestConfiguration().getVertices();const auto F=mesh.getTopology().getFace2Vertices();
        restArea.resize(nf);double area=0;
        for(int f=0;f<nf;++f)
        {
            const Eigen::Vector3d a=(X.row(F(f,1))-X.row(F(f,0))).transpose();
            const Eigen::Vector3d b=(X.row(F(f,2))-X.row(F(f,0))).transpose();
            restArea[f]=0.5*a.cross(b).norm();area+=restArea[f];
        }
        if(!restArea.allFinite() || restArea.minCoeff()<=0 || young<=0 || h<=0 || !std::isfinite(young*h*area))
            throw std::invalid_argument("invalid shell reference geometry/material");
        scale=Vec::Ones(n);scale.head(3*nv).setConstant(std::sqrt(area));reference=young*h*area;
    }
    int size() const override{return 3*mesh.getNumberOfVertices()+mesh.getNumberOfEdges();}
    std::string identity() const override{return id;}
    Vec scales() const override{return scale;}
    double energyReference() const override{return reference;}
    void restore(const Vec& x) override
    {
        if(x.size()!=size())throw std::invalid_argument("shell state size mismatch");
        Eigen::Map<Vec>(mesh.getDataPointer(),size())=x;mesh.updateDeformedConfiguration();
    }
    Value evaluate(const Vec& x) override
    {
        if(!x.allFinite())throw std::runtime_error("nonfinite shell state");
        for(int i=0;i<size();++i)if(mask[i] && x[i]!=fixed[i])throw std::runtime_error("physical constraint violation");
        restore(x);const auto V=mesh.getCurrentConfiguration().getVertices();const auto F=mesh.getTopology().getFace2Vertices();
        for(int f=0;f<mesh.getNumberOfFaces();++f)
        {
            const Eigen::Vector3d a=(V.row(F(f,1))-V.row(F(f,0))).transpose();
            const Eigen::Vector3d b=(V.row(F(f,2))-V.row(F(f,0))).transpose();
            const double ratio=0.5*a.cross(b).norm()/restArea[f];
            if(!std::isfinite(ratio) || ratio<=1e-8)throw std::runtime_error("degenerate shell trial");
        }
        Vec g=Vec::Zero(size());const double f=energy.compute(mesh,g);return {f,g};
    }
    Sparse hessian(const Vec& x) override
    {
        restore(x);
#ifdef SHELL_NATIVE_CACHED_TINYAD
        if(cachedAd)return cachedAd->assembleHessian(mesh);
#endif
#ifdef SHELL_NATIVE_LOCAL_DIHEDRAL
        if(localDihedralAd){
            Sparse H=localDihedralAd->assembleHessian(mesh);
            if(localDihedralAd->lastAssemblyThreads()!=localDihedralAd->assemblyThreads())
                throw std::runtime_error("requested Hessian OpenMP team unavailable");
            return H;
        }
#endif
        return ad.assembleHessian(mesh);
    }
    int lastHessianThreads() const
    {
#ifdef SHELL_NATIVE_LOCAL_DIHEDRAL
        if(localDihedralAd)return localDihedralAd->lastAssemblyThreads();
#endif
        return 1;
    }
    Vec adGradient(const Vec& x){restore(x);return ad.gradient(mesh);}
    int hessianCacheBuilds() const
    {
#ifdef SHELL_NATIVE_CACHED_TINYAD
        if(cachedAd)return cachedAd->cacheBuilds();
#endif
#ifdef SHELL_NATIVE_LOCAL_DIHEDRAL
        if(localDihedralAd)return localDihedralAd->cacheBuilds();
#endif
        return 0;
    }
    std::vector<int> freeDofs(const Vec& x,const Vec& S) const override
    {
        // Construct the gauge from the explicit base state, never a lingering trial.
        if(x.size()!=size() || S.size()!=size() || !S.allFinite() || S.minCoeff()<=0)
            throw std::invalid_argument("invalid gauge scales");
        const ShellEquilibrium::RigidModeProjector P(
            Eigen::Map<const Eigen::MatrixXd>(x.data(),mesh.getNumberOfVertices(),3),
            mesh.getNumberOfEdges(),mask);
        std::vector<char> fixedMask=mask;
        const int k=P.numberOfRigidModes();
        if(k)
        {
            const Eigen::MatrixXd R=S.cwiseInverse().asDiagonal()*P.basis();
            Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(R.transpose());qr.setThreshold(1e-10);
            if(qr.rank()!=k)throw std::runtime_error("rank deficient rigid gauge");
            for(int j=0;j<k;++j)
            {
                const int i=qr.colsPermutation().indices()[j];
                if(fixedMask[i])throw std::runtime_error("gauge selected physical constraint");
                fixedMask[i]=1;
            }
        }
        std::vector<int> ids;for(int i=0;i<size();++i)if(!fixedMask[i])ids.push_back(i);return ids;
    }
};
}
#endif

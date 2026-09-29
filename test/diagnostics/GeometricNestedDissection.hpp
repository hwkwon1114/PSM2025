#ifndef GEOMETRIC_NESTED_DISSECTION_HPP
#define GEOMETRIC_NESTED_DISSECTION_HPP
#include <Eigen/SparseCore>
#include <Eigen/OrderingMethods>
#include <algorithm>
#include <functional>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace NonconvexBenchmark
{
struct NestedDissectionResult
{
    std::vector<int> order; // new index -> original index
    int splits=0,amdBlocks=0,maxSeparator=0;
};
// Experimental ordering only; not selected by any optimizer. Reference-plane
// median cuts, left-endpoint graph separators, AMD within leaves/separators.
inline NestedDissectionResult geometricNestedDissection(
    const Eigen::SparseMatrix<double>& A,const Eigen::MatrixXd& xy,int leafSize=256)
{
    const int n=A.rows();
    if(n<1 || A.cols()!=n || xy.rows()!=n || xy.cols()!=2 || !xy.allFinite() || leafSize<2)
        throw std::invalid_argument("invalid nested-dissection input");
    std::vector<std::vector<int>> graph(n);
    for(int j=0;j<A.outerSize();++j)for(Eigen::SparseMatrix<double>::InnerIterator it(A,j);it;++it)
        if(it.row()!=it.col()){graph[it.row()].push_back(it.col());graph[it.col()].push_back(it.row());}
    for(auto& v:graph){std::sort(v.begin(),v.end());v.erase(std::unique(v.begin(),v.end()),v.end());}
    NestedDissectionResult result;result.order.reserve(n);
    std::vector<int> stamp(n,0),side(n,0),local(n,-1);int generation=0;
    const auto amd=[&](const std::vector<int>& nodes){
        if(nodes.empty())return;
        const int tag=++generation;
        for(size_t i=0;i<nodes.size();++i){stamp[nodes[i]]=tag;local[nodes[i]]=i;}
        std::vector<Eigen::Triplet<double>> entries;
        for(int v:nodes){entries.emplace_back(local[v],local[v],1);
            for(int w:graph[v])if(stamp[w]==tag)entries.emplace_back(local[v],local[w],1);}
        Eigen::SparseMatrix<double> sub(nodes.size(),nodes.size());sub.setFromTriplets(entries.begin(),entries.end());
        Eigen::PermutationMatrix<Eigen::Dynamic,Eigen::Dynamic,int> p;
        Eigen::AMDOrdering<int>()(sub,p);
        for(int i=0;i<p.size();++i)result.order.push_back(nodes[p.indices()[i]]);
        ++result.amdBlocks;
    };
    std::function<void(std::vector<int>)> visit=[&](std::vector<int> nodes){
        if(int(nodes.size())<=leafSize){amd(nodes);return;}
        Eigen::Vector2d low=xy.row(nodes[0]).transpose(),high=low;
        for(int v:nodes){low=low.cwiseMin(xy.row(v).transpose());high=high.cwiseMax(xy.row(v).transpose());}
        const int axis=(high[1]-low[1]>high[0]-low[0])?1:0;
        if(!(high[axis]>low[axis])){amd(nodes);return;}
        std::sort(nodes.begin(),nodes.end(),[&](int a,int b){return xy(a,axis)==xy(b,axis)?a<b:xy(a,axis)<xy(b,axis);});
        const int tag=++generation;const size_t middle=nodes.size()/2;
        for(size_t i=0;i<nodes.size();++i){stamp[nodes[i]]=tag;side[nodes[i]]=i<middle?0:1;}
        std::vector<int> left,right,separator;
        for(int v:nodes){
            if(side[v]==1){right.push_back(v);continue;}
            bool crossing=false;for(int w:graph[v])if(stamp[w]==tag && side[w]==1){crossing=true;break;}
            (crossing?separator:left).push_back(v);
        }
        if(left.empty() || right.empty()){amd(nodes);return;}
        // Every remaining left/right edge must have been covered by separator.
        for(int v:left)for(int w:graph[v])if(stamp[w]==tag && side[w]==1)
            throw std::logic_error("invalid graph separator");
        ++result.splits;result.maxSeparator=std::max(result.maxSeparator,int(separator.size()));
        visit(std::move(left));visit(std::move(right));amd(separator);
    };
    std::vector<int> nodes(n);std::iota(nodes.begin(),nodes.end(),0);visit(std::move(nodes));
    auto check=result.order;std::sort(check.begin(),check.end());
    if(int(check.size())!=n)throw std::logic_error("incomplete ordering");
    for(int i=0;i<n;++i)if(check[i]!=i)throw std::logic_error("invalid ordering permutation");
    return result;
}
}
#endif

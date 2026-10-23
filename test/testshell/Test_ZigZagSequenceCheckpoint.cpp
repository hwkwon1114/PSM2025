#include "gtest/gtest.h"
#include "Geometry.hpp"
#include "Mesh.hpp"
#include "ZigZagSequenceCheckpoint.hpp"
#include "ZigZagSequenceBoundaryConditions.hpp"
#include "ShellEquilibriumSolver.hpp"
#include "MaterialProperties.hpp"
#include "CombinedOperator_Parametric.hpp"
#include "EnergyOperatorList.hpp"
#include <fstream>

namespace {
struct SequenceFixture {
    BilayerMesh mesh;
    Eigen::MatrixXd uv;
    Eigen::VectorXi counts;
    zigzag_checkpoint::Json identity,proof;
    SequenceFixture() {
        RectangularPlate geometry(.1,.15,.2,{false,false},{false,false});
        geometry.setQuiet();mesh.init(geometry,false);
        uv=mesh.getRestConfiguration().getVertices().leftCols(2);
        counts=Eigen::VectorXi::Constant(mesh.getNumberOfFaces(),2);
        identity=zigzag_checkpoint::immutable(mesh,uv,{{"test",true}},std::string(64,'a'));
        // Synthetic acceptance for serialization testing only: no solve claim.
        proof={{"accepted",true},{"termination","gradient_target"},{"curvature","positive_pivots"},
               {"gradient_norm",0.0},{"gradient_tolerance",5e-14}};
        auto x=mesh.getCurrentConfiguration().getVertices();
        for(int i=0;i<x.rows();++i) x(i,2)=.1*x(i,0)*x(i,0)+.2*x(i,1)*x(i,1)+.01;
        mesh.getCurrentConfiguration().getEdgeDirectors().setConstant(.003);
        mesh.getRestConfiguration().getFirstFundamentalForms<top>()[0]*=1.002;
        mesh.getRestConfiguration().getFirstFundamentalForms<bottom>()[0]*=1.001;
        mesh.updateDeformedConfiguration();
    }
    zigzag_checkpoint::Json saved() {
        return zigzag_checkpoint::capture(mesh,counts,1,proof,identity);
    }
};
}

TEST(SequenceCheckpoint, NontrivialRoundTripAndBranchIsolation) {
    SequenceFixture f;
    const auto saved=f.saved();
    const auto rest=zigzag_checkpoint::matrix(f.mesh.getRestConfiguration().getPositions());
    char dir[]="/tmp/sequence-checkpoint-XXXXXX";
    ASSERT_NE(::mkdtemp(dir),nullptr);
    const std::string path=std::string(dir)+"/accepted.json";
    zigzag_checkpoint::publish(path,saved);
    EXPECT_THROW(zigzag_checkpoint::publish(path,saved),std::runtime_error);
    const auto loaded=zigzag_checkpoint::read(path);
    EXPECT_EQ(loaded,saved);
    for(int branch=0;branch<2;++branch) {
        f.mesh.getCurrentConfiguration().getPositions().setZero();
        f.counts.setZero();
        f.mesh.getRestConfiguration().getFirstFundamentalForms<top>()[0]*=2.;
        EXPECT_EQ(zigzag_checkpoint::restore(f.mesh,f.counts,loaded,f.identity,2,5e-14),1);
        EXPECT_EQ(f.saved(),saved);
        EXPECT_EQ(zigzag_checkpoint::matrix(f.mesh.getRestConfiguration().getPositions()),rest);
    }
    std::ofstream corrupt(std::string(dir)+"/corrupt.json");corrupt<<"{\"payload\":";corrupt.close();
    EXPECT_ANY_THROW(zigzag_checkpoint::read(std::string(dir)+"/corrupt.json"));
}

TEST(SequenceCheckpoint, InvalidStateFailsBeforeMutation) {
    SequenceFixture f;
    const auto original=f.saved();
    for(int variant=0;variant<7;++variant) {
        auto bad=original;
        if(variant==0) bad["cycle"]=3;
        if(variant==1) bad["proof"]["accepted"]=false;
        if(variant==2) bad["state"][0][0]=nullptr;
        if(variant==3) bad["identity"]["faces"][0][0]=-1;
        if(variant==4) bad["counts"][0][0]=-1;
        if(variant==5) bad["top"][0][0][0]=-1.;
        if(variant==6) bad["identity"]["runner_identity"]=std::string(64,'b');
        EXPECT_ANY_THROW(zigzag_checkpoint::restore(f.mesh,f.counts,bad,f.identity,2,5e-14));
        EXPECT_EQ(f.saved(),original);
    }
}

TEST(SequenceCheckpoint, NeverClampedRestoreRejectsClampedIdentity) {
    SequenceFixture f;
    const auto free=f.saved();
    EXPECT_EQ(f.mesh.getBoundaryConditions().getVertexBoundaryConditions().count(),0);
    EXPECT_EQ(ShellEquilibrium::RigidModeProjector::fromMesh(f.mesh).numberOfRigidModes(),6);
    auto clamped=free;
    clamped["identity"]["vertex_mask"][0][0]=true;
    EXPECT_THROW(zigzag_checkpoint::restore(f.mesh,f.counts,clamped,f.identity,2,5e-14),std::runtime_error);
    EXPECT_EQ(f.saved(),free);
    f.mesh.getCurrentConfiguration().getPositions().setZero();
    EXPECT_EQ(zigzag_checkpoint::restore(f.mesh,f.counts,free,f.identity,2,5e-14),1);
    EXPECT_EQ(f.saved(),free);
    EXPECT_EQ(f.mesh.getBoundaryConditions().getVertexBoundaryConditions().count(),0);
    EXPECT_EQ(ShellEquilibrium::RigidModeProjector::fromMesh(f.mesh).numberOfRigidModes(),6);
}

TEST(SequenceCheckpoint, SixCoordinateGaugeAtNonzeroHeight) {
    SequenceFixture f;
    const auto gauge=zigzag_sequence_bc::buildMinimalReleaseGaugeMask(f.uv);
    EXPECT_EQ(zigzag_sequence_bc::countFixedDofs(gauge.vertex_mask),6);
    for(int id:gauge.vertex_ids) EXPECT_GT(f.mesh.getCurrentConfiguration().getVertices()(id,2),0);
    zigzag_sequence_bc::applyVertexMask(f.mesh,gauge.vertex_mask);
    EXPECT_EQ(ShellEquilibrium::RigidModeProjector::fromMesh(f.mesh).numberOfRigidModes(),0);
    zigzag_sequence_bc::applyVertexMask(f.mesh,Eigen::MatrixXb::Constant(f.uv.rows(),3,false));
    EXPECT_EQ(ShellEquilibrium::RigidModeProjector::fromMesh(f.mesh).numberOfRigidModes(),6);
    const auto full=zigzag_checkpoint::gaugeGeometry(f.mesh.getCurrentConfiguration().getVertices(),gauge.vertex_ids);
    EXPECT_TRUE(full.at("full_rank_at_relative_1e_10").get<bool>());
    auto x=f.mesh.getCurrentConfiguration().getVertices();
    x(gauge.vertex_ids[1],0)=x(gauge.vertex_ids[0],0);
    const auto singular=zigzag_checkpoint::gaugeGeometry(x,gauge.vertex_ids);
    EXPECT_FALSE(singular.at("full_rank_at_relative_1e_10").get<bool>());
}

TEST(SequenceCheckpoint, ConfirmedPadsSelectOnlyTheirMaterialRectangles) {
    // Explicit material points exercise exact pad edges without requiring a fine mesh.
    Eigen::MatrixXd uv(36,2);int row=0;
    for(double y:{.129,.13,.14,.15})
        for(double x:{-.1,-.075,-.05,-.025,0.,.025,.05,.075,.1}) uv.row(row++)<<x,y;
    zigzag_sequence::BoundaryConditionsConfig bc;
    bc.enabled=true;
    for(double x:{-.05,.05}) {
        zigzag_sequence::FixedRectangleConfig r;
        r.center_uv_m=Eigen::Vector2d(x,.14);r.size_uv_m=Eigen::Vector2d(.05,.02);
        bc.regions.push_back(r);
    }
    const auto mask=zigzag_sequence_bc::buildFullFixationMask(uv,bc);
    ASSERT_EQ(mask.region_vertex_counts.size(),2);
    EXPECT_GT(mask.region_vertex_counts[0],0);EXPECT_GT(mask.region_vertex_counts[1],0);
    for(int i=0;i<uv.rows();++i) {
        const double x=uv(i,0),y=uv(i,1);
        const bool expected=y>=.13-1e-12 && y<=.15+1e-12 &&
            ((x>=-.075-1e-12 && x<=-.025+1e-12) || (x>=.025-1e-12 && x<=.075+1e-12));
        for(int k=0;k<3;++k) EXPECT_EQ(mask.vertex_mask(i,k),expected);
    }
}

TEST(SequenceCheckpoint, UnmaskedReactionGradientMatchesFiniteDifference) {
    SequenceFixture f;
    MaterialProperties_Iso_Constant mat(1.,.33,.00077);
    CombinedOperator_Parametric<BilayerMesh,Material_Isotropic,bottom> bot(mat);
    CombinedOperator_Parametric<BilayerMesh,Material_Isotropic,top> topEnergy(mat);
    EnergyOperatorList<BilayerMesh> energy({&bot,&topEnergy});
    const auto gauge=zigzag_sequence_bc::buildMinimalReleaseGaugeMask(f.uv);
    zigzag_sequence_bc::applyVertexMask(f.mesh,gauge.vertex_mask);
    const Eigen::VectorXd state=f.mesh.getCurrentConfiguration().getPositions();
    Eigen::VectorXd masked=Eigen::VectorXd::Zero(state.size());
    const double maskedEnergy=energy.compute(f.mesh,masked);
    zigzag_sequence_bc::applyVertexMask(f.mesh,Eigen::MatrixXb::Constant(f.uv.rows(),3,false));
    Eigen::VectorXd raw=Eigen::VectorXd::Zero(state.size());
    const double rawEnergy=energy.compute(f.mesh,raw);
    EXPECT_DOUBLE_EQ(rawEnergy,maskedEnergy);
    const int nv=f.mesh.getNumberOfVertices();
    for(int i=0;i<nv;++i) for(int k=0;k<3;++k) if(gauge.vertex_mask(i,k)) {
        const int dof=k*nv+i;const double eps=1e-7;
        EXPECT_DOUBLE_EQ(masked[dof],0);
        auto current=f.mesh.getCurrentConfiguration().getPositions();
        current=state;current[dof]+=eps;f.mesh.updateDeformedConfiguration();
        const double plus=energy.compute(f.mesh);
        current=state;current[dof]-=eps;f.mesh.updateDeformedConfiguration();
        const double minus=energy.compute(f.mesh);
        EXPECT_NEAR((plus-minus)/(2*eps),raw[dof],1e-8*std::abs(raw[dof])+1e-13);
    }
    f.mesh.getCurrentConfiguration().getPositions()=state;
    zigzag_sequence_bc::applyVertexMask(f.mesh,gauge.vertex_mask);
    EXPECT_EQ((f.mesh.getCurrentConfiguration().getPositions()-state).norm(),0);
}

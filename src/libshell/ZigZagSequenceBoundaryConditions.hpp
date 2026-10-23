#pragma once

// Boundary-condition helpers for the JSON-driven recurring zigzag sequence.
// The physical clamp regions are selected once in persistent material (u,v)
// coordinates.  Each selected vertex is fully fixed in x, y, and z.

#include "ZigZagSequenceGrowth.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace zigzag_sequence_bc {

struct ClampMaskResult
{
    Eigen::MatrixXb vertex_mask;
    Eigen::VectorXd region_id;
    std::vector<int> region_vertex_counts;
    int fixed_vertex_count = 0;
    int fixed_dof_count = 0;
};

struct GaugeMaskResult
{
    Eigen::MatrixXb vertex_mask;
    std::array<int, 3> vertex_ids{{-1, -1, -1}};
    int fixed_dof_count = 6;
};

inline int countFixedDofs(const Eigen::Ref<const Eigen::MatrixXb>& mask)
{
    int count = 0;
    for(int i = 0; i < mask.rows(); ++i)
        for(int d = 0; d < mask.cols(); ++d)
            if(mask(i,d)) ++count;
    return count;
}

inline int countFixedVertices(const Eigen::Ref<const Eigen::MatrixXb>& mask)
{
    int count = 0;
    for(int i = 0; i < mask.rows(); ++i)
    {
        bool fixed = false;
        for(int d = 0; d < mask.cols(); ++d)
            fixed = fixed || mask(i,d);
        if(fixed) ++count;
    }
    return count;
}

inline Eigen::MatrixXb combineMasks(
    const Eigen::Ref<const Eigen::MatrixXb>& first,
    const Eigen::Ref<const Eigen::MatrixXb>& second)
{
    if(first.rows() != second.rows() || first.cols() != second.cols())
        throw std::runtime_error(
            "zigzag_sequence_BC: cannot combine boundary masks with different sizes.");

    Eigen::MatrixXb result(first.rows(), first.cols());
    for(int i = 0; i < first.rows(); ++i)
        for(int d = 0; d < first.cols(); ++d)
            result(i,d) = first(i,d) || second(i,d);
    return result;
}

inline ClampMaskResult buildFullFixationMask(
    const Eigen::Ref<const Eigen::MatrixXd>& material_coordinates,
    const zigzag_sequence::BoundaryConditionsConfig& config)
{
    if(material_coordinates.cols() != 2)
        throw std::runtime_error(
            "zigzag_sequence_BC: material coordinates must have two columns (u,v).");
    if(!config.enabled)
        throw std::runtime_error(
            "zigzag_sequence_BC: boundary_conditions.enabled must be true.");
    if(config.regions.empty())
        throw std::runtime_error(
            "zigzag_sequence_BC: at least one rectangular clamp region is required.");

    const int n_vertices = material_coordinates.rows();
    ClampMaskResult result;
    result.vertex_mask = Eigen::MatrixXb::Constant(n_vertices, 3, false);
    result.region_id = Eigen::VectorXd::Zero(n_vertices);
    result.region_vertex_counts.assign(config.regions.size(), 0);

    for(std::size_t region_index = 0;
        region_index < config.regions.size();
        ++region_index)
    {
        const auto& region = config.regions[region_index];
        const double angle = region.rotation_deg * M_PI / 180.0;
        const double c = std::cos(angle);
        const double s = std::sin(angle);
        const double half_u = 0.5 * region.size_uv_m(0);
        const double half_v = 0.5 * region.size_uv_m(1);
        const double tolerance = 1e-12 *
            std::max(1.0, std::max(std::abs(half_u), std::abs(half_v)));

        for(int vertex = 0; vertex < n_vertices; ++vertex)
        {
            const double du = material_coordinates(vertex,0) -
                              region.center_uv_m(0);
            const double dv = material_coordinates(vertex,1) -
                              region.center_uv_m(1);

            // Local coordinates = R(-theta) * (material_point - center).
            const double local_u =  c * du + s * dv;
            const double local_v = -s * du + c * dv;

            if(std::abs(local_u) <= half_u + tolerance &&
               std::abs(local_v) <= half_v + tolerance)
            {
                result.vertex_mask(vertex,0) = true;
                result.vertex_mask(vertex,1) = true;
                result.vertex_mask(vertex,2) = true;
                // Region IDs are one-based in VTK output; zero means unclamped.
                result.region_id(vertex) = static_cast<double>(region_index + 1);
                ++result.region_vertex_counts[region_index];
            }
        }

        if(result.region_vertex_counts[region_index] == 0)
        {
            throw std::runtime_error(
                "zigzag_sequence_BC: clamp region '" + region.name +
                "' contains no mesh vertices. Increase the rectangle size or "
                "check center_uv_mm and the material-coordinate convention.");
        }
    }

    result.fixed_vertex_count = countFixedVertices(result.vertex_mask);
    result.fixed_dof_count = countFixedDofs(result.vertex_mask);
    return result;
}

inline int nearestUnusedVertex(
    const Eigen::Ref<const Eigen::MatrixXd>& material_coordinates,
    const Eigen::Vector2d& target,
    const std::vector<int>& excluded)
{
    int best = -1;
    double best_distance = std::numeric_limits<double>::max();
    for(int vertex = 0; vertex < material_coordinates.rows(); ++vertex)
    {
        if(std::find(excluded.begin(), excluded.end(), vertex) != excluded.end())
            continue;
        const double distance =
            (material_coordinates.row(vertex).transpose() - target).squaredNorm();
        if(distance < best_distance)
        {
            best_distance = distance;
            best = vertex;
        }
    }
    return best;
}

inline GaugeMaskResult buildMinimalReleaseGaugeMask(
    const Eigen::Ref<const Eigen::MatrixXd>& material_coordinates)
{
    if(material_coordinates.cols() != 2 || material_coordinates.rows() < 3)
        throw std::runtime_error(
            "zigzag_sequence_BC: release gauge requires at least three material points.");

    const int n_vertices = material_coordinates.rows();
    const double u_min = material_coordinates.col(0).minCoeff();
    const double u_max = material_coordinates.col(0).maxCoeff();
    const double v_min = material_coordinates.col(1).minCoeff();

    GaugeMaskResult result;
    result.vertex_mask = Eigen::MatrixXb::Constant(n_vertices, 3, false);

    // A is near the lower-left material corner.
    const int vertex_a = nearestUnusedVertex(
        material_coordinates, Eigen::Vector2d(u_min, v_min), {});

    // B is near the lower-right corner, which makes the usual
    // A:xyz, B:yz, C:z gauge well conditioned for the panel coordinates.
    const int vertex_b = nearestUnusedVertex(
        material_coordinates, Eigen::Vector2d(u_max, v_min), {vertex_a});

    if(vertex_a < 0 || vertex_b < 0)
        throw std::runtime_error(
            "zigzag_sequence_BC: failed to select the first two release gauge vertices.");

    // Select C to maximize the material-space triangle area with A and B.
    const Eigen::Vector2d a = material_coordinates.row(vertex_a).transpose();
    const Eigen::Vector2d b = material_coordinates.row(vertex_b).transpose();
    const Eigen::Vector2d ab = b - a;
    int vertex_c = -1;
    double max_twice_area = -1.0;
    for(int vertex = 0; vertex < n_vertices; ++vertex)
    {
        if(vertex == vertex_a || vertex == vertex_b) continue;
        const Eigen::Vector2d ac =
            material_coordinates.row(vertex).transpose() - a;
        const double twice_area = std::abs(ab(0) * ac(1) - ab(1) * ac(0));
        if(twice_area > max_twice_area)
        {
            max_twice_area = twice_area;
            vertex_c = vertex;
        }
    }

    const double material_span = std::max(
        material_coordinates.col(0).maxCoeff() -
            material_coordinates.col(0).minCoeff(),
        material_coordinates.col(1).maxCoeff() -
            material_coordinates.col(1).minCoeff());
    const double area_tolerance = 1e-12 *
        std::max(1.0, material_span * material_span);
    if(vertex_c < 0 || max_twice_area <= area_tolerance)
        throw std::runtime_error(
            "zigzag_sequence_BC: release gauge vertices are collinear in material space.");

    // Six intended numerical gauge DOFs: A fixes x,y,z; B fixes y,z; C fixes z.
    // Material-space selection does not guarantee current-state rigid-gauge rank
    // or finite-motion compatibility; inspect the deformed gauge geometry.
    result.vertex_mask(vertex_a,0) = true;
    result.vertex_mask(vertex_a,1) = true;
    result.vertex_mask(vertex_a,2) = true;
    result.vertex_mask(vertex_b,1) = true;
    result.vertex_mask(vertex_b,2) = true;
    result.vertex_mask(vertex_c,2) = true;
    result.vertex_ids = {{vertex_a, vertex_b, vertex_c}};
    return result;
}

template<typename tMesh>
inline void applyVertexMask(
    tMesh& mesh,
    const Eigen::Ref<const Eigen::MatrixXb>& vertex_mask)
{
    if(vertex_mask.rows() != mesh.getNumberOfVertices() ||
       vertex_mask.cols() != 3)
        throw std::runtime_error(
            "zigzag_sequence_BC: vertex boundary mask has an incompatible size.");

    mesh.getBoundaryConditions().getVertexBoundaryConditions() = vertex_mask;
    mesh.updateDeformedConfiguration();
}

inline Eigen::VectorXd activeFixedDofCountPerVertex(
    const Eigen::Ref<const Eigen::MatrixXb>& vertex_mask)
{
    Eigen::VectorXd result = Eigen::VectorXd::Zero(vertex_mask.rows());
    for(int vertex = 0; vertex < vertex_mask.rows(); ++vertex)
        for(int d = 0; d < vertex_mask.cols(); ++d)
            if(vertex_mask(vertex,d)) result(vertex) += 1.0;
    return result;
}

} // namespace zigzag_sequence_bc

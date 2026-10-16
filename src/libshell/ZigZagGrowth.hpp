//
//  ZigZagGrowth.hpp
//  ZigZagGrowth for English Wheel process simulation
//
//  Created by Putong Kang on 2/1/26.
//  Copyright © 2026 Putong Kang. All rights reserved.
//

#pragma once
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace zigzag {

// ------------------------
// Config / CLI parameters
// ------------------------
enum class StartMode {
    LeftBottom_Up
};

// Updates @07/19:
// Placement convention for JSON-defined toolpaths. Material-frame shifts stay
// aligned with the fixed panel (u,v) axes; pattern-frame shifts rotate with
// the zigzag before being added to the absolute pattern center.
enum class ShiftFrame {
    Material,
    Pattern
};

struct PatternPlacement {
    bool use_material_bbox_center = true;
    Eigen::Vector2d center_uv_m = Eigen::Vector2d::Zero();
    Eigen::Vector2d shift_uv_m = Eigen::Vector2d::Zero();
    ShiftFrame shift_frame = ShiftFrame::Material;
    double rotation_deg = 0.0;
    bool require_inside_material_bounds = true;
};

// Add a switch
enum class TopProfileMode {
    Uniform,
    CenterPeak,
    DirectionalDrift
};

struct Params {
    double Lv_mm      = 140.0;   // vertical span in mm
    double alpha_deg  = 15.0;   // half-angle relative to vertical, deg
    int    N_total    = 6;      // total strips = (N_inclined + 2 vertical)
    double w_mm       = 8.0;    // strip width in mm

    // offsets to shift the zigzag pattern center from the panel center, in mm
    double offset_dx_mm = 0.0;   // shift of zigzag pattern center from panel center, in mm
    double offset_dy_mm = 0.0;   // shift of zigzag pattern center from panel center, in mm

    double rotation_deg = 0.0;   // rigid rotation of whole zigzag pattern, CCW, about local origin

    bool last_wins    = true;
    StartMode start_mode = StartMode::LeftBottom_Up;

    // If true: set all growth to zero first, then write only where strips cover
    // This matches "growth only on toolpath".
    bool zero_outside = true;

    // Along-strip modulation for growth_top only.
    // Uniform reproduces the original behavior.
    TopProfileMode top_profile_mode = TopProfileMode::Uniform;
    double top_end_ratio = 1.0;      // 1.0 => uniform; <1 lowers the two ends
    double top_profile_power = 1.0;  // 1 => sin(pi s), >1 sharper center peak
    double directional_drift_coeff = 0.0; // along-stroke linear drift slope (breaks pi-symmetry)
    bool reverse_traversal = false;       // reverse strip order and stroke travel endpoints
};

// ------------------------
// Small utilities
// ------------------------
inline double deg2rad(double deg) { return deg * M_PI / 180.0; }

// 2D rotation matrix for angle in radians
inline Eigen::Matrix2d rot2d(double ang_rad)
{
    const double c = std::cos(ang_rad);
    const double s = std::sin(ang_rad);
    Eigen::Matrix2d R;
    R << c, -s,
         s,  c;
    return R;
}

// Normalize orientation so opposite directions are equivalent (0..pi)
inline double normalize_angle_pi(double a) {
    // map to (-pi, pi]
    while (a <= -M_PI) a += 2.0 * M_PI;
    while (a >   M_PI) a -= 2.0 * M_PI;
    // identify opposite directions: map to [0, pi)
    if (a < 0.0) a += M_PI;
    return a;
}

// Parse comma list of Real; if fewer than N values provided, fill with default.
// If more than N, ignore extras.
inline std::vector<double> parseCommaListReal(const std::string& s, int N, double default_val)
{
    std::vector<double> out;
    out.reserve(N);

    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        // trim spaces
        size_t b = item.find_first_not_of(" \t\r\n");
        size_t e = item.find_last_not_of(" \t\r\n");
        if (b == std::string::npos) continue;
        std::string token = item.substr(b, e - b + 1);
        out.push_back(std::stod(token));
        if ((int)out.size() >= N) break;
    }
    while ((int)out.size() < N) out.push_back(default_val);
    return out;
}

// Distance from point p to segment [a,b] in 2D
inline double dist_point_segment_2d(const Eigen::Vector2d& p,
                                   const Eigen::Vector2d& a,
                                   const Eigen::Vector2d& b)
{
    const Eigen::Vector2d ab = b - a;
    const double ab2 = ab.squaredNorm();
    if (ab2 <= 1e-30) return (p - a).norm();

    const double t = std::max(0.0, std::min(1.0, (p - a).dot(ab) / ab2));
    const Eigen::Vector2d proj = a + t * ab;
    return (p - proj).norm();
}

// Parse profile mode
inline TopProfileMode parseTopProfileMode(const std::string& s) {
    if (s == "uniform") return TopProfileMode::Uniform;
    if (s == "center_peak") return TopProfileMode::CenterPeak;
    if (s == "directional_drift") return TopProfileMode::DirectionalDrift;
    throw std::runtime_error("zigzag profile mode must be 'uniform', 'center_peak', or 'directional_drift'");
}

// Normalized coordinate along a segment
inline double path_coord_01_on_segment_2d(const Eigen::Vector2d& p,
                                          const Eigen::Vector2d& a,
                                          const Eigen::Vector2d& b)
{
    const Eigen::Vector2d ab = b - a;
    const double ab2 = ab.squaredNorm();
    if (ab2 <= 1e-30) return 0.5;
    const double s = (p - a).dot(ab) / ab2;
    return std::max(0.0, std::min(1.0, s));
}

// Center-high profile with nonzero ends
inline double center_peak_profile_01(double s, double end_ratio, double power)
{
    const double endr = std::max(0.0, std::min(1.0, end_ratio));
    const double p = std::max(1e-12, power);
    const double hump = std::pow(std::sin(M_PI * s), p);
    return endr + (1.0 - endr) * hump;
}

// A single strip = a segment centerline + width + parameters
struct StripSegment {
    Eigen::Vector2d a;
    Eigen::Vector2d b;
    double width;          // meters
    double gtop;           // growth top
    double gbot;           // growth bot
    double ortho;          // ortho coeff
    double angle_rad;      // orientation in [0,pi)
};


// Updates @07/19:
// One physical toolpath hit on one face. Unlike the legacy "last_wins"
// representation, this event preserves every strip/face intersection in
// traversal order so overlap contributes multiple passes and can harden
// the face between successive hits.
struct MaterialHit {
    int face_idx = -1;
    int strip_idx = -1;
    double angle_rad = 0.0;   // material-space principal direction, including pattern rotation
    double gtop = 0.0;        // local/profiled scalar top growth for this hit
    double gbot = 0.0;        // scalar bottom growth for this hit
    double ortho = 0.0;       // orthotropy coefficient for this hit
    double path_coord_01 = 0.5;
    double top_profile = 1.0;
};

// Build strips from parametric zigzag (centered at origin)
inline std::vector<StripSegment> buildZigZagStrips(const Params& zz,
                                                   const std::vector<double>& gtop_list,
                                                   const std::vector<double>& gbot_list,
                                                   const std::vector<double>& ortho_list)
{
    if (zz.N_total < 2) throw std::runtime_error("zigzag_N must be >= 2");
    const int N_incl = zz.N_total - 2;

    const double Lv = zz.Lv_mm * 1e-3;
    const double w  = zz.w_mm  * 1e-3;
    const double a  = deg2rad(zz.alpha_deg);

    // p = Lv * tan(alpha) (alpha relative to vertical)
    const double p = Lv * std::tan(a);

    const double y_bot = -0.5 * Lv;
    const double y_top = +0.5 * Lv;

    // total horizontal span covered by inclined strips
    const double W = (N_incl > 0) ? (N_incl * p) : 0.0;
    const double x_L = -0.5 * W;
    const double x_R = +0.5 * W;

    // Build polyline points (start left-bottom -> up)
    std::vector<Eigen::Vector2d> pts;
    pts.reserve(2 + N_incl + 1);

    pts.emplace_back(x_L, y_bot);  // P0
    pts.emplace_back(x_L, y_top);  // P1  (left vertical)

    // inclined turning points: x increases by p each segment
    // j=1 -> first inclined goes from top to bottom at x_L+p
    for (int j = 1; j <= N_incl; ++j) {
        const double x = x_L + j * p;
        const double y = (j % 2 == 1) ? y_bot : y_top;
        pts.emplace_back(x, y);
    }

    // right vertical strip: go to opposite extreme at same x_R
    // If N_incl==0, pts.back() is P1 at (x_L,y_top) and x_R==x_L; still works.
    const Eigen::Vector2d plast = pts.back();
    const double y_other = (std::abs(plast.y() - y_top) < 1e-15) ? y_bot : y_top;
    pts.emplace_back(x_R, y_other);

    // Now construct strips from consecutive segments.
    // Segment index k corresponds to strip index k in your N_total lists:
    // k=0 left vertical, k=1..N_incl inclined, k=N_total-1 right vertical.
    std::vector<StripSegment> strips;
    strips.reserve(zz.N_total);

    auto make_strip = [&](int k, const Eigen::Vector2d& a2, const Eigen::Vector2d& b2) {
        StripSegment s;
        s.a = a2;
        s.b = b2;
        s.width = w;
        s.gtop  = gtop_list.at(k);
        s.gbot  = gbot_list.at(k);
        s.ortho = ortho_list.at(k);

        const Eigen::Vector2d d = b2 - a2;
        const double ang = std::atan2(d.y(), d.x());       // relative to +x
        s.angle_rad = normalize_angle_pi(ang);             // orientation only
        return s;
    };

    // pts layout:
    // 0->1 : left vertical
    // 1->2 : first inclined (if N_incl>=1) OR right vertical (if N_incl==0, because x_R==x_L)
    // ...
    // last-1 -> last : right vertical
    //
    // We want exactly N_total segments:
    // left vertical + N_incl inclined + right vertical = 2 + N_incl = N_total
    strips.push_back(make_strip(0, pts[0], pts[1]));

    for (int k = 1; k <= N_incl; ++k) {
        strips.push_back(make_strip(k, pts[k], pts[k+1]));
    }

    strips.push_back(make_strip(zz.N_total - 1, pts[pts.size()-2], pts[pts.size()-1]));

    if (zz.reverse_traversal) {
        std::reverse(strips.begin(), strips.end());
        for (auto& s : strips) {
            std::swap(s.a, s.b);
            const Eigen::Vector2d d = s.b - s.a;
            s.angle_rad = normalize_angle_pi(std::atan2(d.y(), d.x()));
        }
    }

    return strips;
}

// // New function to check if the shifted footprint of the zigzag pattern is still inside the panel.
// inline void checkShiftedFootprintInsidePanel(const std::vector<StripSegment>& strips,
//                                              double half_w,
//                                              double panel_lx,
//                                              double panel_ly,
//                                              double dx,
//                                              double dy)
// {
//     if (strips.empty()) return;

//     double xmin =  1e300, xmax = -1e300;
//     double ymin =  1e300, ymax = -1e300;

//     for (const auto& s : strips) {
//         xmin = std::min(xmin, std::min(s.a.x(), s.b.x()));
//         xmax = std::max(xmax, std::max(s.a.x(), s.b.x()));
//         ymin = std::min(ymin, std::min(s.a.y(), s.b.y()));
//         ymax = std::max(ymax, std::max(s.a.y(), s.b.y()));
//     }

//     // expand by strip half-width to get painted footprint
//     xmin -= half_w; xmax += half_w;
//     ymin -= half_w; ymax += half_w;

//     // apply requested shift
//     xmin += dx; xmax += dx;
//     ymin += dy; ymax += dy;

//     const double pxmin = -0.5 * panel_lx;
//     const double pxmax =  0.5 * panel_lx;
//     const double pymin = -0.5 * panel_ly;
//     const double pymax =  0.5 * panel_ly;

//     if (xmin < pxmin || xmax > pxmax || ymin < pymin || ymax > pymax) {
//         const double dx_min = pxmin - (xmin - dx);
//         const double dx_max = pxmax - (xmax - dx);
//         const double dy_min = pymin - (ymin - dy);
//         const double dy_max = pymax - (ymax - dy);

//         std::ostringstream oss;
//         oss << "zigzag offset moves footprint outside panel.\n"
//             << "Requested: dx=" << dx << " m, dy=" << dy << " m\n"
//             << "Allowed dx range: [" << dx_min << ", " << dx_max << "] m\n"
//             << "Allowed dy range: [" << dy_min << ", " << dy_max << "] m";
//         throw std::runtime_error(oss.str());
//     }
// }

// New function to check if the transformed footprint of the zigzag pattern is still inside the panel.
inline void checkTransformedFootprintInsidePanel(const std::vector<StripSegment>& strips,
                                                 double half_w,
                                                 double panel_lx,
                                                 double panel_ly,
                                                 double dx,
                                                 double dy,
                                                 double rot_rad)
{
    if (strips.empty()) return;

    double xmin =  1e300, xmax = -1e300;
    double ymin =  1e300, ymax = -1e300;

    for (const auto& s : strips) {
        xmin = std::min(xmin, std::min(s.a.x(), s.b.x()));
        xmax = std::max(xmax, std::max(s.a.x(), s.b.x()));
        ymin = std::min(ymin, std::min(s.a.y(), s.b.y()));
        ymax = std::max(ymax, std::max(s.a.y(), s.b.y()));
    }

    // expand by strip half-width to get painted footprint
    xmin -= half_w; xmax += half_w;
    ymin -= half_w; ymax += half_w;

    // rectangle corners in local zigzag frame
    std::vector<Eigen::Vector2d> corners = {
        {xmin, ymin},
        {xmin, ymax},
        {xmax, ymin},
        {xmax, ymax}
    };

    const Eigen::Matrix2d R = rot2d(rot_rad);
    const Eigen::Vector2d t(dx, dy);

    // transform corners to centered panel frame
    double txmin =  1e300, txmax = -1e300;
    double tymin =  1e300, tymax = -1e300;

    for (const auto& q : corners) {
        const Eigen::Vector2d qt = R * q + t;
        txmin = std::min(txmin, qt.x());
        txmax = std::max(txmax, qt.x());
        tymin = std::min(tymin, qt.y());
        tymax = std::max(tymax, qt.y());
    }

    const double pxmin = -0.5 * panel_lx;
    const double pxmax =  0.5 * panel_lx;
    const double pymin = -0.5 * panel_ly;
    const double pymax =  0.5 * panel_ly;

    if (txmin < pxmin || txmax > pxmax || tymin < pymin || tymax > pymax) {
        std::ostringstream oss;
        oss << "zigzag transform moves footprint outside panel.\n"
            << "Requested: dx=" << dx << " m, dy=" << dy << " m, rot=" << rot_rad * 180.0 / M_PI << " deg\n"
            << "Transformed footprint bounds: "
            << "x=[" << txmin << ", " << txmax << "] m, "
            << "y=[" << tymin << ", " << tymax << "] m\n"
            << "Panel bounds: "
            << "x=[" << pxmin << ", " << pxmax << "] m, "
            << "y=[" << pymin << ", " << pymax << "] m";
        throw std::runtime_error(oss.str());
    }
}

// ------------------------
// Main application function
// ------------------------
template <typename MeshType>
inline void apply(const MeshType& mesh,
                  const Params& zz,
                  const std::vector<double>& gtop_list,
                  const std::vector<double>& gbot_list,
                  const std::vector<double>& ortho_list,
                  Eigen::VectorXd& growthRates_t,
                  Eigen::VectorXd& growthRates_b,
                  Eigen::VectorXd& growthAngles,
                  Eigen::VectorXd& orthoCoeffFaces,
                  Eigen::VectorXi* passCountFaces = nullptr) // passCountFaces: optional output of how many strips cover each face (for debugging)
{
    const int nFaces = mesh.getNumberOfFaces();
    if (growthRates_t.size() != nFaces || growthRates_b.size() != nFaces)
        throw std::runtime_error("zigzag::apply: growthRates size mismatch");
    if (growthAngles.size() != nFaces || orthoCoeffFaces.size() != nFaces)
        throw std::runtime_error("zigzag::apply: growthAngles/orthoCoeffFaces size mismatch");
    if ((int)gtop_list.size() != zz.N_total || (int)gbot_list.size() != zz.N_total || (int)ortho_list.size() != zz.N_total)
        throw std::runtime_error("zigzag::apply: parameter lists must have size N_total");
    if (passCountFaces) {
        if (passCountFaces->size() != nFaces)
            throw std::runtime_error("zigzag::apply: passCountFaces size mismatch");
    }

    // Build strips (centered)
    const auto strips = buildZigZagStrips(zz, gtop_list, gbot_list, ortho_list);
    const double half_w = 0.5 * (zz.w_mm * 1e-3);

    // Apply offset for strips
    const double dx = zz.offset_dx_mm * 1e-3;
    const double dy = zz.offset_dy_mm * 1e-3;

    // Precompute inverse rotation for transforming face centroids into zigzag frame
    const double rot_rad = deg2rad(zz.rotation_deg);
    const Eigen::Matrix2d Rinv = rot2d(-rot_rad);

    // Optionally zero everything first
    if (zz.zero_outside) {
        growthRates_t.setZero();
        growthRates_b.setZero();
        // keep growthAngles as-is outside (could also set to 0); leaving as-is is fine
        // ortho outside should be 0 if no growth
        orthoCoeffFaces.setZero();
    }

    // Get vertices + faces
    const Eigen::MatrixXd V = mesh.getCurrentConfiguration().getVertices();
    const Eigen::MatrixXi F = mesh.getTopology().getFace2Vertices();

    // Center mesh XY so zigzag strips (built around origin) hit the plate
    const double xmin = V.col(0).minCoeff();
    const double xmax = V.col(0).maxCoeff();
    const double ymin = V.col(1).minCoeff();
    const double ymax = V.col(1).maxCoeff();
    const Eigen::Vector2d center(0.5*(xmin + xmax), 0.5*(ymin + ymax));

    // Check if the shifted footprint of the zigzag pattern is still inside the panel.
    const double panel_lx = xmax - xmin;
    const double panel_ly = ymax - ymin;

    // checkShiftedFootprintInsidePanel(strips, half_w, panel_lx, panel_ly, dx, dy);
    checkTransformedFootprintInsidePanel(strips, half_w, panel_lx, panel_ly, dx, dy, rot_rad);

    // For each strip in order: last wins = overwrite as we go
    for (int k = 0; k < (int)strips.size(); ++k) {
        const auto& s = strips[k];

        for (int i = 0; i < nFaces; ++i) {
            const int i0 = F(i,0);
            const int i1 = F(i,1);
            const int i2 = F(i,2);

            const Eigen::Vector2d c(
                (V(i0,0) + V(i1,0) + V(i2,0)) / 3.0,
                (V(i0,1) + V(i1,1) + V(i2,1)) / 3.0
            );

            // const Eigen::Vector2d cc = c - center; // center XY, shift to zigzag frame

            // const double d = dist_point_segment_2d(cc, s.a, s.b);

            // const Eigen::Vector2d cc = c - center;                 // centered panel frame
            // const Eigen::Vector2d p  = cc - Eigen::Vector2d(dx,dy); // shift into zigzag pattern frame

            const Eigen::Vector2d cc = c - center;                    // centered panel frame
            const Eigen::Vector2d q  = cc - Eigen::Vector2d(dx,dy);   // undo translation
            const Eigen::Vector2d p  = Rinv * q;                      // undo rigid rotation
            

            const double d = dist_point_segment_2d(p, s.a, s.b);

            // if (d <= half_w) {
            //     growthRates_t(i) = s.gtop;
            //     growthRates_b(i) = s.gbot;
            //     growthAngles(i)  = s.angle_rad;
            //     orthoCoeffFaces(i) = s.ortho;
            // }
            if (d <= half_w) {
                if (passCountFaces) (*passCountFaces)(i) += 1;  // history accumulation
                double gtop_local = s.gtop;

                if (zz.top_profile_mode == TopProfileMode::CenterPeak) {
                    // const double s01 = path_coord_01_on_segment_2d(cc, s.a, s.b);
                    const double s01 = path_coord_01_on_segment_2d(p, s.a, s.b);
                    const double profile = center_peak_profile_01(s01, zz.top_end_ratio, zz.top_profile_power);
                    gtop_local *= profile;
                }

                growthRates_t(i) = gtop_local;
                growthRates_b(i) = s.gbot;
                growthAngles(i)  = s.angle_rad;
                orthoCoeffFaces(i) = s.ortho;
            }
        }
    }
}



// Updates @07/19:
// Collect every material-coordinate zigzag hit without overwriting overlaps.
// The returned vector is ordered first by strip traversal order and then by
// face index. hitCountFaces reports the number of physical strip hits during
// one complete zigzag cycle, independent of whether the assigned growth is
// zero. This is the event source for the history-preserving recurring case.
inline void collectMaterialCoordinateHits(
                  const Eigen::Ref<const Eigen::MatrixXd> materialCoordinates,
                  const Eigen::Ref<const Eigen::MatrixXi> face2vertices,
                  const Params& zz,
                  const std::vector<double>& gtop_list,
                  const std::vector<double>& gbot_list,
                  const std::vector<double>& ortho_list,
                  std::vector<MaterialHit>& hits,
                  Eigen::VectorXi* hitCountFaces = nullptr)
{
    if(materialCoordinates.cols() < 2)
        throw std::runtime_error(
            "zigzag::collectMaterialCoordinateHits: materialCoordinates must have at least 2 columns.");

    const int nFaces = face2vertices.rows();
    if ((int)gtop_list.size() != zz.N_total ||
        (int)gbot_list.size() != zz.N_total ||
        (int)ortho_list.size() != zz.N_total)
        throw std::runtime_error(
            "zigzag::collectMaterialCoordinateHits: parameter lists must have size N_total.");

    if (hitCountFaces && hitCountFaces->size() != nFaces)
        throw std::runtime_error(
            "zigzag::collectMaterialCoordinateHits: hitCountFaces size mismatch.");

    if(face2vertices.size() > 0)
    {
        const int maxVertexIndex = face2vertices.maxCoeff();
        const int minVertexIndex = face2vertices.minCoeff();
        if(minVertexIndex < 0 || maxVertexIndex >= materialCoordinates.rows())
            throw std::runtime_error(
                "zigzag::collectMaterialCoordinateHits: face index exceeds material-coordinate array.");
    }

    const auto strips =
        buildZigZagStrips(zz, gtop_list, gbot_list, ortho_list);
    const double half_w = 0.5 * (zz.w_mm * 1e-3);
    const double dx = zz.offset_dx_mm * 1e-3;
    const double dy = zz.offset_dy_mm * 1e-3;
    const double rot_rad = deg2rad(zz.rotation_deg);
    const Eigen::Matrix2d Rinv = rot2d(-rot_rad);

    const double umin = materialCoordinates.col(0).minCoeff();
    const double umax = materialCoordinates.col(0).maxCoeff();
    const double vmin = materialCoordinates.col(1).minCoeff();
    const double vmax = materialCoordinates.col(1).maxCoeff();

    const Eigen::Vector2d center(
        0.5 * (umin + umax),
        0.5 * (vmin + vmax));

    checkTransformedFootprintInsidePanel(
        strips,
        half_w,
        umax - umin,
        vmax - vmin,
        dx,
        dy,
        rot_rad);

    hits.clear();
    if(hitCountFaces)
        hitCountFaces->setZero();

    for (int k = 0; k < (int)strips.size(); ++k)
    {
        const auto& strip = strips[k];

        for (int i = 0; i < nFaces; ++i)
        {
            const int i0 = face2vertices(i,0);
            const int i1 = face2vertices(i,1);
            const int i2 = face2vertices(i,2);

            const Eigen::Vector2d centroid(
                (materialCoordinates(i0,0) +
                 materialCoordinates(i1,0) +
                 materialCoordinates(i2,0)) / 3.0,
                (materialCoordinates(i0,1) +
                 materialCoordinates(i1,1) +
                 materialCoordinates(i2,1)) / 3.0);

            const Eigen::Vector2d centered = centroid - center;
            const Eigen::Vector2d untranslated =
                centered - Eigen::Vector2d(dx, dy);
            const Eigen::Vector2d local = Rinv * untranslated;

            if(dist_point_segment_2d(local, strip.a, strip.b) > half_w)
                continue;

            const double s01 =
                path_coord_01_on_segment_2d(local, strip.a, strip.b);

            double profile = 1.0;
            if(zz.top_profile_mode == TopProfileMode::CenterPeak)
            {
                profile = center_peak_profile_01(
                    s01,
                    zz.top_end_ratio,
                    zz.top_profile_power);
            }
            else if(zz.top_profile_mode == TopProfileMode::DirectionalDrift)
            {
                profile = std::max(0.0, 1.0 + zz.directional_drift_coeff * (s01 - 0.5));
            }

            MaterialHit hit;
            hit.face_idx = i;
            hit.strip_idx = k;
            hit.angle_rad =
                normalize_angle_pi(strip.angle_rad + rot_rad);
            hit.gtop = strip.gtop * profile;
            hit.gbot = strip.gbot;
            hit.ortho = strip.ortho;
            hit.path_coord_01 = s01;
            hit.top_profile = profile;
            hits.push_back(hit);

            if(hitCountFaces)
                (*hitCountFaces)(i) += 1;
        }
    }
}



// Updates @07/19:
// Check a placement against the actual material-coordinate bounds. Unlike the
// legacy helper, this routine supports an absolute center that need not equal
// the panel center and supports material- or pattern-frame shifts.
inline void checkPlacedFootprintInsideMaterialBounds(
    const std::vector<StripSegment>& strips,
    const double half_w,
    const double umin,
    const double umax,
    const double vmin,
    const double vmax,
    const PatternPlacement& placement)
{
    if(strips.empty() || !placement.require_inside_material_bounds)
        return;

    double xmin =  1e300, xmax = -1e300;
    double ymin =  1e300, ymax = -1e300;
    for(const auto& strip : strips)
    {
        xmin = std::min(xmin, std::min(strip.a.x(), strip.b.x()));
        xmax = std::max(xmax, std::max(strip.a.x(), strip.b.x()));
        ymin = std::min(ymin, std::min(strip.a.y(), strip.b.y()));
        ymax = std::max(ymax, std::max(strip.a.y(), strip.b.y()));
    }
    xmin -= half_w; xmax += half_w;
    ymin -= half_w; ymax += half_w;

    const Eigen::Vector2d materialCenter(
        0.5 * (umin + umax),
        0.5 * (vmin + vmax));
    const Eigen::Vector2d center =
        placement.use_material_bbox_center ?
            materialCenter : placement.center_uv_m;

    const double rot_rad = deg2rad(placement.rotation_deg);
    const Eigen::Matrix2d R = rot2d(rot_rad);
    const Eigen::Vector2d translation =
        placement.shift_frame == ShiftFrame::Material ?
            placement.shift_uv_m : R * placement.shift_uv_m;

    const Eigen::Vector2d corners[4] = {
        Eigen::Vector2d(xmin, ymin),
        Eigen::Vector2d(xmin, ymax),
        Eigen::Vector2d(xmax, ymin),
        Eigen::Vector2d(xmax, ymax)
    };

    double placedUmin =  1e300, placedUmax = -1e300;
    double placedVmin =  1e300, placedVmax = -1e300;
    for(const auto& corner : corners)
    {
        const Eigen::Vector2d placed = center + R * corner + translation;
        placedUmin = std::min(placedUmin, placed.x());
        placedUmax = std::max(placedUmax, placed.x());
        placedVmin = std::min(placedVmin, placed.y());
        placedVmax = std::max(placedVmax, placed.y());
    }

    const double tol = 1e-12 * std::max(1.0, std::max(umax-umin, vmax-vmin));
    if(placedUmin < umin - tol || placedUmax > umax + tol ||
       placedVmin < vmin - tol || placedVmax > vmax + tol)
    {
        std::ostringstream oss;
        oss << "zigzag placement moves footprint outside material bounds.\n"
            << "Placed footprint: u=[" << placedUmin << ", " << placedUmax
            << "], v=[" << placedVmin << ", " << placedVmax << "] m\n"
            << "Material bounds: u=[" << umin << ", " << umax
            << "], v=[" << vmin << ", " << vmax << "] m";
        throw std::runtime_error(oss.str());
    }
}

// Updates @07/19:
// Placement-aware event collector for JSON loading sequences. The canonical
// zigzag is constructed about its local origin and transformed by
//
//   p_uv = center + R(rotation) p_local + shift_material,
//
// or by center + R(rotation)(p_local + shift_pattern). The returned material
// angle includes the pattern rotation. Every face/strip intersection remains
// an independent hit so pass counting and hardening are unambiguous.
inline void collectMaterialCoordinateHits(
    const Eigen::Ref<const Eigen::MatrixXd> materialCoordinates,
    const Eigen::Ref<const Eigen::MatrixXi> face2vertices,
    const Params& zz,
    const PatternPlacement& placement,
    const std::vector<double>& gtop_list,
    const std::vector<double>& gbot_list,
    const std::vector<double>& ortho_list,
    std::vector<MaterialHit>& hits,
    Eigen::VectorXi* hitCountFaces = nullptr)
{
    if(materialCoordinates.cols() < 2)
        throw std::runtime_error(
            "zigzag::collectMaterialCoordinateHits(placement): materialCoordinates must have at least 2 columns.");

    const int nFaces = face2vertices.rows();
    if((int)gtop_list.size() != zz.N_total ||
       (int)gbot_list.size() != zz.N_total ||
       (int)ortho_list.size() != zz.N_total)
        throw std::runtime_error(
            "zigzag::collectMaterialCoordinateHits(placement): parameter lists must have size N_total.");
    if(hitCountFaces && hitCountFaces->size() != nFaces)
        throw std::runtime_error(
            "zigzag::collectMaterialCoordinateHits(placement): hitCountFaces size mismatch.");

    if(face2vertices.size() > 0)
    {
        const int minVertexIndex = face2vertices.minCoeff();
        const int maxVertexIndex = face2vertices.maxCoeff();
        if(minVertexIndex < 0 || maxVertexIndex >= materialCoordinates.rows())
            throw std::runtime_error(
                "zigzag::collectMaterialCoordinateHits(placement): face index exceeds material-coordinate array.");
    }

    const auto strips = buildZigZagStrips(
        zz, gtop_list, gbot_list, ortho_list);
    const double half_w = 0.5 * zz.w_mm * 1e-3;
    const double umin = materialCoordinates.col(0).minCoeff();
    const double umax = materialCoordinates.col(0).maxCoeff();
    const double vmin = materialCoordinates.col(1).minCoeff();
    const double vmax = materialCoordinates.col(1).maxCoeff();

    checkPlacedFootprintInsideMaterialBounds(
        strips, half_w, umin, umax, vmin, vmax, placement);

    const Eigen::Vector2d materialCenter(
        0.5 * (umin + umax),
        0.5 * (vmin + vmax));
    const Eigen::Vector2d center =
        placement.use_material_bbox_center ?
            materialCenter : placement.center_uv_m;

    const double rot_rad = deg2rad(placement.rotation_deg);
    const Eigen::Matrix2d R = rot2d(rot_rad);
    const Eigen::Matrix2d Rinv = rot2d(-rot_rad);
    const Eigen::Vector2d translation =
        placement.shift_frame == ShiftFrame::Material ?
            placement.shift_uv_m : R * placement.shift_uv_m;

    hits.clear();
    if(hitCountFaces) hitCountFaces->setZero();

    for(int k = 0; k < (int)strips.size(); ++k)
    {
        const auto& strip = strips[k];
        for(int face = 0; face < nFaces; ++face)
        {
            const int i0 = face2vertices(face,0);
            const int i1 = face2vertices(face,1);
            const int i2 = face2vertices(face,2);
            const Eigen::Vector2d centroid(
                (materialCoordinates(i0,0) + materialCoordinates(i1,0) + materialCoordinates(i2,0)) / 3.0,
                (materialCoordinates(i0,1) + materialCoordinates(i1,1) + materialCoordinates(i2,1)) / 3.0);

            const Eigen::Vector2d local =
                Rinv * (centroid - center - translation);
            if(dist_point_segment_2d(local, strip.a, strip.b) > half_w)
                continue;

            const double s01 = path_coord_01_on_segment_2d(
                local, strip.a, strip.b);
            double profile = 1.0;
            if(zz.top_profile_mode == TopProfileMode::CenterPeak)
                profile = center_peak_profile_01(
                    s01, zz.top_end_ratio, zz.top_profile_power);
            else if(zz.top_profile_mode == TopProfileMode::DirectionalDrift)
                profile = std::max(0.0, 1.0 + zz.directional_drift_coeff * (s01 - 0.5));

            MaterialHit hit;
            hit.face_idx = face;
            hit.strip_idx = k;
            hit.angle_rad = normalize_angle_pi(strip.angle_rad + rot_rad);
            hit.gtop = strip.gtop * profile;
            hit.gbot = strip.gbot;
            hit.ortho = strip.ortho;
            hit.path_coord_01 = s01;
            hit.top_profile = profile;
            hits.push_back(hit);
            if(hitCountFaces) (*hitCountFaces)(face) += 1;
        }
    }
}

// Updates @07/17:
// Apply the zigzag in a persistent flat/material coordinate domain (u,v)
// instead of using the current spatial x-y projection. This is the key
// toolpath-selection function for analytically curved panels.
//
// materialCoordinates: nVertices x 2 (or x >=2); columns are u and v.
// face2vertices:       nFaces x 3; same topology/indexing as the curved mesh.
//
// The rigid pattern rotation is included in the returned material-space
// principal direction:
//     theta_material = theta_segment + rotation_deg.
//
// The original mesh-based apply(...) above is intentionally left unchanged so
// existing flat-panel results remain reproducible.
inline void applyMaterialCoordinates(
                  const Eigen::Ref<const Eigen::MatrixXd> materialCoordinates,
                  const Eigen::Ref<const Eigen::MatrixXi> face2vertices,
                  const Params& zz,
                  const std::vector<double>& gtop_list,
                  const std::vector<double>& gbot_list,
                  const std::vector<double>& ortho_list,
                  Eigen::VectorXd& growthRates_t,
                  Eigen::VectorXd& growthRates_b,
                  Eigen::VectorXd& growthAngles,
                  Eigen::VectorXd& orthoCoeffFaces,
                  Eigen::VectorXi* passCountFaces = nullptr)
{
    if(materialCoordinates.cols() < 2)
        throw std::runtime_error(
            "zigzag::applyMaterialCoordinates: materialCoordinates must have at least 2 columns.");

    const int nFaces = face2vertices.rows();
    if (growthRates_t.size() != nFaces || growthRates_b.size() != nFaces)
        throw std::runtime_error(
            "zigzag::applyMaterialCoordinates: growthRates size mismatch.");
    if (growthAngles.size() != nFaces || orthoCoeffFaces.size() != nFaces)
        throw std::runtime_error(
            "zigzag::applyMaterialCoordinates: growthAngles/orthoCoeffFaces size mismatch.");
    if ((int)gtop_list.size() != zz.N_total ||
        (int)gbot_list.size() != zz.N_total ||
        (int)ortho_list.size() != zz.N_total)
        throw std::runtime_error(
            "zigzag::applyMaterialCoordinates: parameter lists must have size N_total.");
    if (passCountFaces && passCountFaces->size() != nFaces)
        throw std::runtime_error(
            "zigzag::applyMaterialCoordinates: passCountFaces size mismatch.");

    if(face2vertices.size() > 0)
    {
        const int maxVertexIndex = face2vertices.maxCoeff();
        if(maxVertexIndex >= materialCoordinates.rows())
            throw std::runtime_error(
                "zigzag::applyMaterialCoordinates: face index exceeds material-coordinate array.");
    }

    const auto strips =
        buildZigZagStrips(zz, gtop_list, gbot_list, ortho_list);
    const double half_w = 0.5 * (zz.w_mm * 1e-3);

    const double dx = zz.offset_dx_mm * 1e-3;
    const double dy = zz.offset_dy_mm * 1e-3;

    const double rot_rad = deg2rad(zz.rotation_deg);
    const Eigen::Matrix2d Rinv = rot2d(-rot_rad);

    if (zz.zero_outside)
    {
        growthRates_t.setZero();
        growthRates_b.setZero();
        orthoCoeffFaces.setZero();
    }

    if(passCountFaces)
        passCountFaces->setZero();

    const double umin = materialCoordinates.col(0).minCoeff();
    const double umax = materialCoordinates.col(0).maxCoeff();
    const double vmin = materialCoordinates.col(1).minCoeff();
    const double vmax = materialCoordinates.col(1).maxCoeff();

    const Eigen::Vector2d center(
        0.5 * (umin + umax),
        0.5 * (vmin + vmax));

    const double panel_lu = umax - umin;
    const double panel_lv = vmax - vmin;

    checkTransformedFootprintInsidePanel(
        strips, half_w, panel_lu, panel_lv, dx, dy, rot_rad);

    for (int k = 0; k < (int)strips.size(); ++k)
    {
        const auto& s = strips[k];

        for (int i = 0; i < nFaces; ++i)
        {
            const int i0 = face2vertices(i,0);
            const int i1 = face2vertices(i,1);
            const int i2 = face2vertices(i,2);

            const Eigen::Vector2d c(
                (materialCoordinates(i0,0) +
                 materialCoordinates(i1,0) +
                 materialCoordinates(i2,0)) / 3.0,
                (materialCoordinates(i0,1) +
                 materialCoordinates(i1,1) +
                 materialCoordinates(i2,1)) / 3.0);

            // Transform the material-space face centroid into the local
            // unrotated/untranslated zigzag frame.
            const Eigen::Vector2d cc = c - center;
            const Eigen::Vector2d q =
                cc - Eigen::Vector2d(dx, dy);
            const Eigen::Vector2d p = Rinv * q;

            const double d =
                dist_point_segment_2d(p, s.a, s.b);

            if (d <= half_w)
            {
                if (passCountFaces)
                    (*passCountFaces)(i) += 1;

                double gtop_local = s.gtop;
                if (zz.top_profile_mode == TopProfileMode::CenterPeak)
                {
                    const double s01 =
                        path_coord_01_on_segment_2d(p, s.a, s.b);
                    const double profile =
                        center_peak_profile_01(
                            s01,
                            zz.top_end_ratio,
                            zz.top_profile_power);
                    gtop_local *= profile;
                }

                growthRates_t(i) = gtop_local;
                growthRates_b(i) = s.gbot;

                // Unlike the legacy flat function, include the rigid rotation
                // because the physical eigenstrain direction follows the
                // rotated toolpath in the material domain.
                growthAngles(i) =
                    normalize_angle_pi(s.angle_rad + rot_rad);

                orthoCoeffFaces(i) = s.ortho;
            }
        }
    }
}


} // namespace zigzag


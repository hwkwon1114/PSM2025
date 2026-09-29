//
//  Sim_Bilayer_Growth.cpp
//  Elasticity
//
//  Created by Wim van Rees on 10/27/16.
//  Modified by Vladislav Sushitskii on 03/29/22.
//  Modified by Putong Kang on 06/10/24.
//  Copyright © 2026 Wim van Rees, Vladislav Sushitskii and Putong Kang. All rights reserved.
//

#include "Sim_Bilayer_Growth.hpp"
#include "Geometry.hpp"
#include "GrowthHelper.hpp"
#include "MaterialProperties.hpp"
#include "CombinedOperator_Parametric.hpp"
#include "EnergyOperatorList.hpp"
#include "TinyADHessian_Bilayer.hpp"
#include "ShellEquilibriumSolver.hpp"
#include <random>
#include "ComputeCurvatures.hpp"

// ver-0122
#include <sstream>
#include <algorithm>
#include <cctype>

// ver-0203
#include "ZigZagGrowth.hpp"
#include <memory>

// Updates @07/19: JSON-defined recurring toolpath sequence.
#include "ZigZagSequenceGrowth.hpp"
#include "ZigZagSequenceBoundaryConditions.hpp"

#include <stdexcept>
#include <fstream>   // Updates @07/19: sequence summary CSV
#include <iomanip>   // Updates @07/19: stable CSV precision
#include <limits>    // Updates @07/19: summary extrema

static std::vector<int> parse_int_list(const std::string& s)
{
    std::vector<int> out;
    std::string cleaned = s;

    // Replace commas with spaces so we can stream >> ints
    for(char& c : cleaned)
        if(c == ',') c = ' ';

    std::stringstream ss(cleaned);
    int v;
    while(ss >> v)
    {
        if(v > 0) out.push_back(v);
    }

    std::sort(out.begin(), out.end());
    out.erase(std::unique(out.begin(), out.end()), out.end());
    return out;
}

// Updates @09/05: optional prescribed-deformation input for the recurring
// zigzag sequence. One row per mesh vertex, "vertex_index,dx,dy,dz", with the
// displacement components in metres. A leading header line is tolerated, '#'
// starts a comment, and commas, semicolons, tabs or spaces separate fields.
// Every vertex index in [0, nVert) must appear exactly once and all four
// entries must be finite; anything else is a hard error, because a partially
// specified geometry would silently mix two different shapes.
static Eigen::MatrixXd loadVertexDisplacementCsv(
    const std::string& filename,
    const int nVert)
{
    std::ifstream input(filename);
    if(!input)
        throw std::runtime_error(
            "prescribed_deformation: cannot open -prescribed_disp_csv '" +
            filename + "'.");

    Eigen::MatrixXd displacement(nVert, 3);
    displacement.setZero();
    Eigen::VectorXi seen = Eigen::VectorXi::Zero(nVert);

    std::string line;
    int lineNumber = 0;
    int rowsRead = 0;
    bool headerSeen = false;
    while(std::getline(input, line))
    {
        ++lineNumber;
        const std::size_t comment = line.find('#');
        if(comment != std::string::npos)
            line.erase(comment);
        for(char& c : line)
            if(c == ',' || c == ';' || c == '\t' || c == '\r')
                c = ' ';
        if(line.find_first_not_of(' ') == std::string::npos)
            continue;

        std::istringstream fields(line);
        long long index = -1;
        Real dx = 0.0, dy = 0.0, dz = 0.0;
        if(!(fields >> index >> dx >> dy >> dz))
        {
            std::istringstream header(line);
            std::string label, xLabel, yLabel, zLabel, extra;
            if(rowsRead == 0 && !headerSeen &&
               (header >> label >> xLabel >> yLabel >> zLabel) &&
               label == "vertex_index" && xLabel == "dx" &&
               yLabel == "dy" && zLabel == "dz" && !(header >> extra))
            {
                headerSeen = true;
                continue;
            }
            throw std::runtime_error(
                "prescribed_deformation: malformed row at line " +
                std::to_string(lineNumber) + " of '" + filename +
                "'; expected 'vertex_index,dx,dy,dz'.");
        }
        std::string extra;
        if(fields >> extra)
            throw std::runtime_error(
                "prescribed_deformation: extra column at line " +
                std::to_string(lineNumber) + " of '" + filename + "'.");
        if(index < 0 || index >= nVert)
            throw std::runtime_error(
                "prescribed_deformation: vertex index " +
                std::to_string(index) + " at line " +
                std::to_string(lineNumber) + " of '" + filename +
                "' is outside [0," + std::to_string(nVert) + ").");
        if(seen(static_cast<int>(index)) != 0)
            throw std::runtime_error(
                "prescribed_deformation: vertex index " +
                std::to_string(index) + " appears more than once in '" +
                filename + "'.");
        if(!std::isfinite(dx) || !std::isfinite(dy) || !std::isfinite(dz))
            throw std::runtime_error(
                "prescribed_deformation: non-finite displacement for vertex " +
                std::to_string(index) + " in '" + filename + "'.");

        displacement(static_cast<int>(index), 0) = dx;
        displacement(static_cast<int>(index), 1) = dy;
        displacement(static_cast<int>(index), 2) = dz;
        seen(static_cast<int>(index)) = 1;
        ++rowsRead;
    }

    if(rowsRead != nVert || seen.sum() != nVert)
        throw std::runtime_error(
            "prescribed_deformation: '" + filename + "' provided " +
            std::to_string(rowsRead) + " valid rows for a mesh with " +
            std::to_string(nVert) +
            " vertices; a full per-vertex displacement field is required.");

    return displacement;
}

void Sim_Bilayer_Growth::run()
{
  const std::string runCase = parser.parse<std::string>("-case", "");
  if(runCase == "custom")
    TestCustomGrowth(); // custom growth pattern
  else if(runCase == "testR")
    TestRandomPatterns(); // random patterns on a rectangular plate
  else
    {
      std::cout << "No valid runCase defined. Options are \n";
      std::cout << "\t -case custom\n";
      std::cout << "\t -case testR\n";
    }
}

// Custom growth patterns
void Sim_Bilayer_Growth::TestCustomGrowth()
{
    const std::string growth_type = parser.parse<std::string>("-growth_type","chess");
    // Options: 
    // - homo (uniform expansion)
    // - chess (the checkerboard pattern)
    // - center (rectangular zone in the center of the plate)
    // - patch (rectangular zone in the center of the plate, with self-defined size and offset)
    // - wave (half of the plate is expanding on the bottom side, and half - on the top side)
    // - circle (circular zone in the center of the plate)
    // - external (projection of a pattern coming from another mesh)
    // - zigzag (zigzag pattern) New function added ver-0203
    // - zigzag_cycles (Updates @07/19: recurring, pass-count hardened zigzag)
    // Supported active growth types:
    // - zigzag_sequence (JSON-defined toolpath recipes and recurring forming cycles)
    // - zigzag_sequence_BC (persistent boundary clamp regions, then final release)
    // - zigzag (single zigzag toolpath)
    // - panel_ortho (uniform orthotropic growth over the whole panel)
    // - parallel_lines (uniform repeated parallel treated bands)
  
   
    const std::string geometryCase = parser.parse<std::string>("-geometry", ""); //see initForwardProblem()
    const Real margin_x = parser.parse<Real>("-margin_x", 0.0); // margins to simulate the clamping frame: no eigensrain in this zone. 0.001 = 1mm
    const Real margin_y = parser.parse<Real>("-margin_y", 0.0);
    tag = "bilayer_" + growth_type;

    initForwardProblem();

    const Real E = 1;
    // const Real E = 73100; // E,ν for Aluminum Alloy 2024-T3
    // Update E to match the material properties of the bilayer being simulated
    const bool enable_passE = parser.parse<bool>("-enable_passE", false);
    const Real passE_k = parser.parse<Real>("-passE_k", 0.05);      // max +5%
    const Real passE_n0 = parser.parse<Real>("-passE_n0", 2.0);     // saturation rate
    const int  passE_p0 = parser.parse<int> ("-passE_p0", 1);       // baseline pass count

    const Real nu = 0.33;
    const Real h_total = parser.parse<Real>("-h_total", 0.003);

    // init growth
    Real growthRate_t = parser.parse<Real>("-growth_top", 0.001);
    Real growthRate_b = parser.parse<Real>("-growth_bot", -0.001);

    // Correct angle conversion: degrees to radians
    const Real growthAngle = parser.parse<Real>("-growth_angle", 0.0)*M_PI/180.0; // principal growth direction (angle with respect to x-axis)
    const Real ortho_coeff = parser.parse<Real>("-ortho_coeff", 0.0); // orthotropy coefficient

    auto Vertices = mesh.getCurrentConfiguration().getVertices();
    const int nVert = mesh.getNumberOfVertices();
    const auto Connect = mesh.getTopology().getFace2Vertices();
    const int nFaces = mesh.getNumberOfFaces();

    // Updates @07/17:
    // Keep a separate flat/material coordinate array (u,v) for defining the
    // toolpath. For the analytical cylindrical panel, recover v from the
    // known inverse cylindrical mapping. The mechanics continue to use the
    // curved rest vertices X(u,v).
    const bool use_curved_material_mapping =
        (geometryCase == "curved_rectangle");

    Eigen::MatrixXd materialCoordinates(nVert, 2);
    {
        const Eigen::MatrixXd Xrest =
            mesh.getRestConfiguration().getVertices();

        if(use_curved_material_mapping)
        {
            const Real curve_radius =
                parser.parse<Real>("-curve_radius", 0.25);
            const Real curve_sign_input =
                parser.parse<Real>("-curve_sign", 1.0);
            const Real curve_sign =
                (curve_sign_input >= 0.0) ? 1.0 : -1.0;

            if(curve_radius <= 0.0)
                throw std::runtime_error(
                    "curved_rectangle: -curve_radius must be > 0.");

            for(int i = 0; i < nVert; ++i)
            {
                const Real u = Xrest(i,0);

                // From:
                // y = R sin(v/R)
                // z = s R (1-cos(v/R))
                // therefore:
                // v = R atan2(y, R-s z)
                const Real v = curve_radius * std::atan2(
                    Xrest(i,1),
                    curve_radius - curve_sign * Xrest(i,2));

                materialCoordinates(i,0) = u;
                materialCoordinates(i,1) = v;
            }
        }
        else
        {
            // Existing flat geometries: material coordinates coincide with
            // the rest x-y coordinates.
            materialCoordinates = Xrest.leftCols(2);
        }
    }

    Eigen::VectorXi passCountFaces(nFaces);
    passCountFaces.setZero();

    Eigen::VectorXi IndicV(nVert);
    IndicV.setZero();

    Eigen::VectorXd growthRates_b(nFaces);
    Eigen::VectorXd growthRates_t(nFaces);
    growthRates_b.setZero();
    growthRates_t.setZero();

    // Orthotropic growth fields: two principal in-plane growth components per face.
    // When growth_angle = 0, these align with the global x-y axes.
    Eigen::VectorXd growthRates_1_t(nFaces);
    Eigen::VectorXd growthRates_1_b(nFaces);
    Eigen::VectorXd growthRates_2_t(nFaces);
    Eigen::VectorXd growthRates_2_b(nFaces);

    growthRates_1_t.setZero();
    growthRates_1_b.setZero();
    growthRates_2_t.setZero();
    growthRates_2_b.setZero();

    bool use_direct_ortho = false;

    //

    Eigen::VectorXd E_face = Eigen::VectorXd::Constant(nFaces, E);

    if (enable_passE) {
        for (int i=0; i<nFaces; ++i) {
            const int p = passCountFaces(i);
            if (p > passE_p0) {
                const Real x = (Real)(p - passE_p0) / passE_n0;
                const Real mult = 1.0 + passE_k * (1.0 - std::exp(-x));  // increases & saturates
                E_face(i) = E * mult;
            }
        }
    }

    // Debug: Give declaration
    // --- helpers for quick sanity prints
    auto nnz = [](const Eigen::VectorXd& v, double eps=1e-20){
        return (v.array().abs() > eps).count();
    };
    auto vmin = [](const Eigen::VectorXd& v){
        return v.size() ? v.minCoeff() : 0.0;
    };
    auto vmax = [](const Eigen::VectorXd& v){
        return v.size() ? v.maxCoeff() : 0.0;
    };

    // ver -0203: Added declaration for zigzag growth
    Eigen::VectorXd growthAngles = Eigen::VectorXd::Constant(nFaces, growthAngle);
    Eigen::VectorXd orthoCoeffFaces = Eigen::VectorXd::Constant(nFaces, ortho_coeff);

    // Debug
    std::cout << "[growth] growth_type = '" << growth_type << "'\n";

    // Updates @07/19:
    // History-preserving recurring zigzag. The same material-space toolpath is
    // replayed for -cycle_rounds cycles. Every strip/face intersection is kept
    // as an individual hit (no last-wins overwrite), the target metrics are
    // updated sequentially, and the previous released geometry remains in
    // currentState as the starting geometry for the next cycle. The shared
    // target curvature b_r is never modified.
    if(growth_type == "zigzag_cycles")
    {
        if(enable_passE)
            throw std::runtime_error(
                "zigzag_cycles uses pass-count eigenstrain hardening; "
                "-enable_passE must be false.");

        const int nsteps_cycles = parser.parse<int>("-nsteps", 1);
        if(nsteps_cycles != 1)
            throw std::runtime_error(
                "zigzag_cycles owns the recurring loop; use -nsteps 1.");

        const int cycle_rounds =
            parser.parse<int>("-cycle_rounds", 1);
        if(cycle_rounds < 1)
            throw std::runtime_error(
                "zigzag_cycles: -cycle_rounds must be >= 1.");

        const std::string hardening_model =
            parser.parse<std::string>(
                "-cycle_hardening_model", "voce_decay");
        const Real hardening_beta =
            parser.parse<Real>("-cycle_hardening_beta", 0.0);
        const Real hardening_floor =
            parser.parse<Real>("-cycle_hardening_floor", 0.0);

        if(hardening_model != "none" &&
           hardening_model != "voce_decay")
            throw std::runtime_error(
                "zigzag_cycles: -cycle_hardening_model must be "
                "'none' or 'voce_decay'.");
        if(hardening_beta < 0.0)
            throw std::runtime_error(
                "zigzag_cycles: -cycle_hardening_beta must be >= 0.");
        if(hardening_floor < 0.0 || hardening_floor > 1.0)
            throw std::runtime_error(
                "zigzag_cycles: -cycle_hardening_floor must be in [0,1].");

        auto hardeningFactor =
            [&](const int previous_hits) -> Real
            {
                if(hardening_model == "none")
                    return 1.0;

                return hardening_floor +
                    (1.0 - hardening_floor) *
                    std::exp(-hardening_beta *
                             static_cast<Real>(previous_hits));
            };

        // Parse exactly the same zigzag inputs as the existing zigzag case.
        const Real Lv_mm =
            parser.parse<Real>("-zigzag_lv_mm", 40.0);
        const Real alpha_deg =
            parser.parse<Real>("-zigzag_alpha_deg", 15.0);
        const int N_total =
            parser.parse<int>("-zigzag_N", 6);
        const Real w_mm =
            parser.parse<Real>("-zigzag_w_mm", 2.0);
        const Real offset_dx_mm =
            parser.parse<Real>("-zigzag_offset_dx_mm", 0.0);
        const Real offset_dy_mm =
            parser.parse<Real>("-zigzag_offset_dy_mm", 0.0);
        const Real rotation_deg =
            parser.parse<Real>("-zigzag_rotation_deg", 0.0);

        const std::string gtop_s =
            parser.parse<std::string>("-zigzag_gtop_list", "");
        const std::string gbot_s =
            parser.parse<std::string>("-zigzag_gbot_list", "");
        const std::string ortho_s =
            parser.parse<std::string>("-zigzag_ortho_list", "");

        const std::string profile_mode_str =
            parser.parse<std::string>(
                "-zigzag_profile_mode", "uniform");
        const Real top_end_ratio =
            parser.parse<Real>("-zigzag_top_end_ratio", 1.0);
        const Real top_profile_power =
            parser.parse<Real>("-zigzag_top_profile_power", 1.0);

        std::vector<double> gtop_list(N_total, growthRate_t);
        std::vector<double> gbot_list(N_total, growthRate_b);
        std::vector<double> ortho_list(N_total, ortho_coeff);

        if(!gtop_s.empty())
            gtop_list = zigzag::parseCommaListReal(
                gtop_s, N_total, growthRate_t);
        if(!gbot_s.empty())
            gbot_list = zigzag::parseCommaListReal(
                gbot_s, N_total, growthRate_b);
        if(!ortho_s.empty())
            ortho_list = zigzag::parseCommaListReal(
                ortho_s, N_total, ortho_coeff);

        zigzag::Params zz;
        zz.Lv_mm = Lv_mm;
        zz.alpha_deg = alpha_deg;
        zz.N_total = N_total;
        zz.w_mm = w_mm;
        zz.offset_dx_mm = offset_dx_mm;
        zz.offset_dy_mm = offset_dy_mm;
        zz.rotation_deg = rotation_deg;
        zz.last_wins = false; // events are accumulated explicitly below
        zz.start_mode = zigzag::StartMode::LeftBottom_Up;
        zz.zero_outside = true;
        zz.top_profile_mode =
            zigzag::parseTopProfileMode(profile_mode_str);
        zz.top_end_ratio = top_end_ratio;
        zz.top_profile_power = top_profile_power;

        // Build the immutable one-cycle event sequence in material coordinates.
        std::vector<zigzag::MaterialHit> materialHits;
        Eigen::VectorXi hitsPerCycle(nFaces);
        hitsPerCycle.setZero();
        zigzag::collectMaterialCoordinateHits(
            materialCoordinates,
            Connect,
            zz,
            gtop_list,
            gbot_list,
            ortho_list,
            materialHits,
            &hitsPerCycle);

        // Preserve the existing margin convention: a face is removed only if
        // all three of its vertices belong to the material-coordinate margin.
        Eigen::VectorXi faceEnabled = Eigen::VectorXi::Ones(nFaces);
        if(margin_x > 0.0 || margin_y > 0.0)
        {
            const Real uMin = materialCoordinates.col(0).minCoeff();
            const Real uMax = materialCoordinates.col(0).maxCoeff();
            const Real vMin = materialCoordinates.col(1).minCoeff();
            const Real vMax = materialCoordinates.col(1).maxCoeff();

            Eigen::VectorXi marginVertex(nVert);
            marginVertex.setZero();
            for(int i = 0; i < nVert; ++i)
            {
                const Real u = materialCoordinates(i,0);
                const Real v = materialCoordinates(i,1);
                if(u <= uMin + margin_x ||
                   u >= uMax - margin_x ||
                   v <= vMin + margin_y ||
                   v >= vMax - margin_y)
                    marginVertex(i) = 1;
            }

            for(int i = 0; i < nFaces; ++i)
            {
                if(marginVertex(Connect(i,0)) == 1 &&
                   marginVertex(Connect(i,1)) == 1 &&
                   marginVertex(Connect(i,2)) == 1)
                    faceEnabled(i) = 0;
            }

            materialHits.erase(
                std::remove_if(
                    materialHits.begin(),
                    materialHits.end(),
                    [&](const zigzag::MaterialHit& hit)
                    {
                        return faceEnabled(hit.face_idx) == 0;
                    }),
                materialHits.end());

            hitsPerCycle.setZero();
            for(const auto& hit : materialHits)
                hitsPerCycle(hit.face_idx) += 1;
        }

        if(materialHits.empty())
            throw std::runtime_error(
                "zigzag_cycles: no faces are covered after applying "
                "the toolpath and margins.");

        // Store event indices per face. This supports exact VTK diagnostics for
        // every overlapping hit without reducing them to one last-win angle.
        std::vector<std::vector<int>> hitsByFace(nFaces);
        for(int h = 0; h < (int)materialHits.size(); ++h)
            hitsByFace[materialHits[h].face_idx].push_back(h);

        int maxHitsPerFace = 0;
        for(const auto& faceHits : hitsByFace)
            maxHitsPerFace = std::max(
                maxHitsPerFace,
                static_cast<int>(faceHits.size()));

        std::cout
            << "[zigzag_cycles] cycles=" << cycle_rounds
            << ", events_per_cycle=" << materialHits.size()
            << ", covered_faces=" << (hitsPerCycle.array() > 0).count()
            << ", max_hits_per_face_per_cycle=" << maxHitsPerFace
            << "\n";
        std::cout
            << "[zigzag_cycles] hardening_model=" << hardening_model
            << ", beta=" << hardening_beta
            << ", floor=" << hardening_floor
            << "\n";

        // The initialized rest forms are the cycle-zero natural state.
        // Top/bottom a_r evolve in place; the shared b_r is retained exactly.
        tVecMat2d& aformsBot =
            mesh.getRestConfiguration()
                .getFirstFundamentalForms<bottom>();
        tVecMat2d& aformsTop =
            mesh.getRestConfiguration()
                .getFirstFundamentalForms<top>();
        const tVecMat2d initialBforms =
            mesh.getRestConfiguration().getSecondFundamentalForms();

        Eigen::VectorXi totalPassCount(nFaces);
        totalPassCount.setZero();

        MaterialProperties_Iso_Constant matprop_bot(
            E, nu, h_total);
        MaterialProperties_Iso_Constant matprop_top(
            E, nu, h_total);
        CombinedOperator_Parametric<
            tMesh, Material_Isotropic, bottom>
            engOp_bot(matprop_bot);
        CombinedOperator_Parametric<
            tMesh, Material_Isotropic, top>
            engOp_top(matprop_top);
        EnergyOperatorList<tMesh> engOps(
            {&engOp_bot, &engOp_top});

        const std::string dump_iters_str =
            parser.parse<std::string>("-dump_iters", "");
        const std::vector<int> dump_iters =
            parse_int_list(dump_iters_str);
        const int max_iter =
            parser.parse<int>("-max_iter", -1);
        // Updates @07/19: Keep the default optimizer step immutable, but
        // create a writable copy for each cycle because minimizeEnergy()
        // accepts Real& and may update the step size internally.
        const Real eps_init_default = 1e-2;
        const Real tol =
            parser.parse<Real>("-tol", 1e-12);
        const bool stepwise =
            parser.parse<bool>("-stepwise", false);
        const bool write_cycle_state =
            parser.parse<bool>("-cycle_write_state", false);

        // Write the current-surface toolpath before each cycle. A separate
        // vector/mask pair is emitted for every hit rank on a face, so overlap
        // is visible without using a last-win diagnostic.
        auto writeCycleMapping =
            [&](const int cycle)
            {
                const Eigen::MatrixXd Xcurrent =
                    mesh.getCurrentConfiguration().getVertices();
                WriteVTK writer(Xcurrent, Connect);

                Eigen::VectorXd materialU = materialCoordinates.col(0);
                Eigen::VectorXd materialV = materialCoordinates.col(1);
                Eigen::VectorXd hitsPerCycleReal =
                    hitsPerCycle.cast<Real>();
                Eigen::VectorXd totalBeforeReal =
                    totalPassCount.cast<Real>();
                Eigen::VectorXd qBefore(nFaces);
                Eigen::VectorXd cycleField =
                    Eigen::VectorXd::Constant(
                        nFaces, static_cast<Real>(cycle));

                qBefore.setZero();
                for(int i = 0; i < nFaces; ++i)
                    if(hitsPerCycle(i) > 0)
                        qBefore(i) = hardeningFactor(
                            totalPassCount(i));

                writer.addScalarFieldToVertices(
                    materialU, "material_u");
                writer.addScalarFieldToVertices(
                    materialV, "material_v");
                writer.addScalarFieldToFaces(
                    cycleField, "cycle_index");
                writer.addScalarFieldToFaces(
                    hitsPerCycleReal, "hits_per_cycle");
                writer.addScalarFieldToFaces(
                    totalBeforeReal, "total_hits_before_cycle");
                writer.addScalarFieldToFaces(
                    qBefore, "hardening_factor_before_first_hit");

                Real maxTangencyError = 0.0;
                for(int rank = 0; rank < maxHitsPerFace; ++rank)
                {
                    Eigen::VectorXd angles =
                        Eigen::VectorXd::Zero(nFaces);
                    Eigen::VectorXd active =
                        Eigen::VectorXd::Zero(nFaces);
                    Eigen::VectorXd stripIndex =
                        Eigen::VectorXd::Constant(nFaces, -1.0);
                    Eigen::VectorXd baseGtop =
                        Eigen::VectorXd::Zero(nFaces);
                    Eigen::VectorXd baseGbot =
                        Eigen::VectorXd::Zero(nFaces);
                    Eigen::VectorXd baseOrtho =
                        Eigen::VectorXd::Zero(nFaces);

                    for(int i = 0; i < nFaces; ++i)
                    {
                        if(rank >= (int)hitsByFace[i].size())
                            continue;

                        const auto& hit =
                            materialHits[hitsByFace[i][rank]];
                        angles(i) = hit.angle_rad;
                        active(i) = 1.0;
                        stripIndex(i) =
                            static_cast<Real>(hit.strip_idx);
                        baseGtop(i) = hit.gtop;
                        baseGbot(i) = hit.gbot;
                        baseOrtho(i) = hit.ortho;
                    }

                    Eigen::MatrixXd directions;
                    GrowthHelper<tMesh>::
                        mapMaterialAnglesToCurrentShellDirections(
                            mesh,
                            materialCoordinates,
                            angles,
                            directions);

                    for(int i = 0; i < nFaces; ++i)
                    {
                        if(active(i) == 0.0)
                        {
                            directions.row(i).setZero();
                            continue;
                        }

                        const Eigen::Vector3d x0 =
                            Xcurrent.row(Connect(i,0)).transpose();
                        const Eigen::Vector3d x1 =
                            Xcurrent.row(Connect(i,1)).transpose();
                        const Eigen::Vector3d x2 =
                            Xcurrent.row(Connect(i,2)).transpose();
                        const Eigen::Vector3d normal =
                            (x1-x0).cross(x2-x0).normalized();
                        maxTangencyError = std::max(
                            maxTangencyError,
                            std::abs(normal.dot(
                                directions.row(i).transpose())));
                    }

                    const std::string suffix =
                        helpers::ToString(rank + 1, 2);
                    writer.addVectorFieldToFaces(
                        directions,
                        "growth_dir_3d_hit_" + suffix);
                    writer.addScalarFieldToFaces(
                        active,
                        "hit_active_" + suffix);
                    writer.addScalarFieldToFaces(
                        angles,
                        "growth_angle_material_hit_" + suffix);
                    writer.addScalarFieldToFaces(
                        stripIndex,
                        "strip_index_hit_" + suffix);
                    writer.addScalarFieldToFaces(
                        baseGtop,
                        "base_gtop_hit_" + suffix);
                    writer.addScalarFieldToFaces(
                        baseGbot,
                        "base_gbot_hit_" + suffix);
                    writer.addScalarFieldToFaces(
                        baseOrtho,
                        "base_ortho_hit_" + suffix);
                }

                std::cout
                    << "[zigzag_cycles] cycle " << cycle
                    << " mapping max |n dot d|="
                    << maxTangencyError << "\n";

                writer.write(
                    tag + "_cycle_" +
                    helpers::ToString(cycle, 3) +
                    "_mapping");
            };

        // Exact state output: target metric components are written directly,
        // avoiding an inaccurate reduction of overlapping directional history
        // to one equivalent growth angle.
        auto writeCycleState =
            [&](const int cycle,
                const std::string& phase,
                const Eigen::VectorXd& incG1Top,
                const Eigen::VectorXd& incG2Top,
                const Eigen::VectorXd& incG1Bot,
                const Eigen::VectorXd& incG2Bot,
                const Eigen::VectorXd& qFirst,
                const Eigen::VectorXd& qLast)
            {
                const Eigen::MatrixXd X0 =
                    mesh.getRestConfiguration().getVertices();
                const Eigen::MatrixXd X =
                    mesh.getCurrentConfiguration().getVertices();
                WriteVTK writer(X, Connect);

                const Eigen::MatrixXd U = X - X0;
                const Eigen::VectorXd U3 = U.col(2);
                const Eigen::VectorXd Umag = U.rowwise().norm();
                const Eigen::VectorXd materialU =
                    materialCoordinates.col(0);
                const Eigen::VectorXd materialV =
                    materialCoordinates.col(1);
                const Eigen::VectorXd hitCountReal =
                    totalPassCount.cast<Real>();
                const Eigen::VectorXd hitsPerCycleReal =
                    hitsPerCycle.cast<Real>();
                const Eigen::VectorXd cycleField =
                    Eigen::VectorXd::Constant(
                        nFaces, static_cast<Real>(cycle));
                Eigen::VectorXd qNext(nFaces);
                qNext.setZero();

                Eigen::VectorXd aTop11(nFaces), aTop12(nFaces), aTop22(nFaces);
                Eigen::VectorXd aBot11(nFaces), aBot12(nFaces), aBot22(nFaces);
                Eigen::VectorXd bRef11(nFaces), bRef12(nFaces), bRef22(nFaces);
                Eigen::VectorXd bRefDrift(nFaces);

                const tVecMat2d& currentBforms =
                    mesh.getRestConfiguration()
                        .getSecondFundamentalForms();

                for(int i = 0; i < nFaces; ++i)
                {
                    if(hitsPerCycle(i) > 0)
                        qNext(i) = hardeningFactor(
                            totalPassCount(i));

                    aTop11(i) = aformsTop[i](0,0);
                    aTop12(i) = aformsTop[i](0,1);
                    aTop22(i) = aformsTop[i](1,1);
                    aBot11(i) = aformsBot[i](0,0);
                    aBot12(i) = aformsBot[i](0,1);
                    aBot22(i) = aformsBot[i](1,1);
                    bRef11(i) = currentBforms[i](0,0);
                    bRef12(i) = currentBforms[i](0,1);
                    bRef22(i) = currentBforms[i](1,1);
                    bRefDrift(i) =
                        (currentBforms[i] - initialBforms[i]).norm();
                }

                writer.addVectorFieldToVertices(U, "U_from_cycle0");
                writer.addScalarFieldToVertices(U3, "U3_from_cycle0");
                writer.addScalarFieldToVertices(Umag, "Umag_from_cycle0");
                writer.addScalarFieldToVertices(materialU, "material_u");
                writer.addScalarFieldToVertices(materialV, "material_v");

                writer.addScalarFieldToFaces(cycleField, "cycle_index");
                writer.addScalarFieldToFaces(
                    hitsPerCycleReal, "hits_per_cycle");
                writer.addScalarFieldToFaces(
                    hitCountReal, "total_hit_count");
                writer.addScalarFieldToFaces(
                    qFirst, "hardening_factor_first_hit");
                writer.addScalarFieldToFaces(
                    qLast, "hardening_factor_last_hit");
                writer.addScalarFieldToFaces(
                    qNext, "hardening_factor_next_hit");

                writer.addScalarFieldToFaces(
                    incG1Top, "effective_increment_g1_top_sum");
                writer.addScalarFieldToFaces(
                    incG2Top, "effective_increment_g2_top_sum");
                writer.addScalarFieldToFaces(
                    incG1Bot, "effective_increment_g1_bot_sum");
                writer.addScalarFieldToFaces(
                    incG2Bot, "effective_increment_g2_bot_sum");

                writer.addScalarFieldToFaces(aTop11, "abar_top_11");
                writer.addScalarFieldToFaces(aTop12, "abar_top_12");
                writer.addScalarFieldToFaces(aTop22, "abar_top_22");
                writer.addScalarFieldToFaces(aBot11, "abar_bot_11");
                writer.addScalarFieldToFaces(aBot12, "abar_bot_12");
                writer.addScalarFieldToFaces(aBot22, "abar_bot_22");
                writer.addScalarFieldToFaces(bRef11, "bbar_ref_11");
                writer.addScalarFieldToFaces(bRef12, "bbar_ref_12");
                writer.addScalarFieldToFaces(bRef22, "bbar_ref_22");
                writer.addScalarFieldToFaces(
                    bRefDrift, "bbar_reference_drift_norm");

                if(phase == "final")
                {
                    Eigen::VectorXd gauss(nFaces);
                    Eigen::VectorXd mean(nFaces);
                    ComputeCurvatures<tMesh> computeCurvatures;
                    computeCurvatures.compute(mesh, gauss, mean);
                    writer.addScalarFieldToFaces(gauss, "gauss");
                    writer.addScalarFieldToFaces(mean, "mean");
                }

                writer.write(
                    tag + "_cycle_" +
                    helpers::ToString(cycle, 3) +
                    "_" + phase);
            };

        Eigen::VectorXd zeroField =
            Eigen::VectorXd::Zero(nFaces);
        writeCycleState(
            0,
            "initial",
            zeroField,
            zeroField,
            zeroField,
            zeroField,
            zeroField,
            zeroField);

        for(int cycle = 1; cycle <= cycle_rounds; ++cycle)
        {
            // The current vertices here are exactly the previous cycle's final
            // released geometry. They are not copied into restState.
            writeCycleMapping(cycle);

            Eigen::VectorXd incG1Top =
                Eigen::VectorXd::Zero(nFaces);
            Eigen::VectorXd incG2Top =
                Eigen::VectorXd::Zero(nFaces);
            Eigen::VectorXd incG1Bot =
                Eigen::VectorXd::Zero(nFaces);
            Eigen::VectorXd incG2Bot =
                Eigen::VectorXd::Zero(nFaces);
            Eigen::VectorXd qFirst =
                Eigen::VectorXd::Zero(nFaces);
            Eigen::VectorXd qLast =
                Eigen::VectorXd::Zero(nFaces);
            Eigen::VectorXi processedThisCycle(nFaces);
            processedThisCycle.setZero();

            for(const auto& hit : materialHits)
            {
                const int face = hit.face_idx;
                const Real q =
                    hardeningFactor(totalPassCount(face));

                if(processedThisCycle(face) == 0)
                    qFirst(face) = q;
                qLast(face) = q;

                const Real g1Top =
                    q * hit.gtop * (1.0 + hit.ortho);
                const Real g2Top =
                    q * hit.gtop * (1.0 - hit.ortho);
                const Real g1Bot =
                    q * hit.gbot * (1.0 + hit.ortho);
                const Real g2Bot =
                    q * hit.gbot * (1.0 - hit.ortho);

                GrowthHelper<tMesh>::
                    updateAbarWithMaterialGrowthIncrement(
                        materialCoordinates,
                        Connect,
                        face,
                        hit.angle_rad,
                        g1Top,
                        g2Top,
                        aformsTop[face]);
                GrowthHelper<tMesh>::
                    updateAbarWithMaterialGrowthIncrement(
                        materialCoordinates,
                        Connect,
                        face,
                        hit.angle_rad,
                        g1Bot,
                        g2Bot,
                        aformsBot[face]);

                // These sums are diagnostics only. The exact tensor history is
                // stored in aformsTop/aformsBot through the congruence updates.
                incG1Top(face) += g1Top;
                incG2Top(face) += g2Top;
                incG1Bot(face) += g1Bot;
                incG2Bot(face) += g2Bot;

                totalPassCount(face) += 1;
                processedThisCycle(face) += 1;
            }

            Real maxBformDrift = 0.0;
            const tVecMat2d& currentBforms =
                mesh.getRestConfiguration()
                    .getSecondFundamentalForms();
            for(int i = 0; i < nFaces; ++i)
                maxBformDrift = std::max(
                    maxBformDrift,
                    (currentBforms[i] - initialBforms[i]).norm());

            if(maxBformDrift > 1e-13)
                throw std::runtime_error(
                    "zigzag_cycles: b_r changed unexpectedly; "
                    "history-preserving metric-only treatment requires "
                    "the shared reference curvature to remain fixed.");

            // Updates @07/19: Reset the optimizer step for every cycle.
            // Passing the previous cycle's modified epsilon would couple the
            // numerical optimizer history to the physical loading history.
            Real eps_cycle = eps_init_default;
            minimizeEnergy(
                engOps,
                eps_cycle,
                tol,
                stepwise,
                (dump_iters.empty() ? nullptr : &dump_iters),
                max_iter);
            mesh.updateDeformedConfiguration();

            const Real totalEnergy =
                engOp_bot.getLastStretchingEnergy() +
                engOp_bot.getLastBendingEnergy() +
                engOp_bot.getLastABEnergy() +
                engOp_top.getLastStretchingEnergy() +
                engOp_top.getLastBendingEnergy() +
                engOp_top.getLastABEnergy();

            std::cout
                << "[zigzag_cycles] cycle " << cycle
                << " complete: max_total_hits="
                << totalPassCount.maxCoeff()
                << ", total_energy=" << totalEnergy
                << ", max_bbar_drift=" << maxBformDrift
                << "\n";

            writeCycleState(
                cycle,
                "final",
                incG1Top,
                incG2Top,
                incG1Bot,
                incG2Bot,
                qFirst,
                qLast);

            if(write_cycle_state)
                mesh.writeToFile(
                    tag + "_cycle_" +
                    helpers::ToString(cycle, 3) +
                    "_state");
        }

        const bool export_stl =
            parser.parse<bool>("-export_stl", false);
        const bool stl_ascii =
            parser.parse<bool>("-stl_ascii", false);
        if(export_stl)
            WriteSTL::write(
                mesh.getTopology(),
                mesh.getCurrentConfiguration(),
                tag + "_final_deformed",
                stl_ascii);

        // Do not execute the legacy mesh.init_rest(final_vtp) block below.
        // Returning here preserves the original b_r and accumulated target
        // metrics as the constitutive history.
        return;
    }


    // Updates @07/19:
    // Flexible JSON loading sequence. Each JSON item defines one toolpath
    // recipe; every repeat is executed as a separate physical cycle:
    // apply the complete toolpath, minimize/release, then remap the next repeat
    // to the newly released current geometry. Top/bottom target metrics and
    // face hit-count history persist globally; the shared b_r stays unchanged.
    if(growth_type == "zigzag_sequence" ||
       growth_type == "zigzag_sequence_BC")
    {
        // Both CLI names use the same recurring-sequence implementation.
        // The JSON boundary_conditions.enabled flag is the source of truth
        // for whether physical clamps and final release are activated.
        const bool requested_sequence_bc_alias =
            (growth_type == "zigzag_sequence_BC");

        if(enable_passE)
            throw std::runtime_error(
                growth_type + " uses hit-count eigenstrain hardening; "
                "-enable_passE must be false.");

        const int sequence_nsteps = parser.parse<int>("-nsteps", 1);
        if(sequence_nsteps != 1)
            throw std::runtime_error(
                growth_type + " owns the loading/release loop; use -nsteps 1.");

        const std::string cycle_file =
            parser.parse<std::string>("-cycle_file", "");
        if(cycle_file.empty())
            throw std::runtime_error(
                growth_type + ": provide -cycle_file sequence.json.");

        const zigzag_sequence::SequenceConfig sequence =
            zigzag_sequence::loadSequenceJson(cycle_file);

        const bool use_sequence_bc =
            sequence.boundary_conditions.enabled;
        const std::string sequence_solve_mode =
            parser.parse<std::string>("-sequence_solve_mode", "every_cycle");
        if(sequence_solve_mode != "every_cycle" && sequence_solve_mode != "final_only")
            throw std::runtime_error("-sequence_solve_mode must be every_cycle or final_only.");
        if(use_sequence_bc && sequence_solve_mode == "final_only")
            throw std::runtime_error("final_only currently requires boundary_conditions.enabled=false.");
        int requested_cycles = 0;
        for(const auto& tp : sequence.toolpaths) requested_cycles += tp.repeat;
        int equilibrium_solves = 0;
        std::string calibration_final_file;
        Real calibration_energy = 0.0;
        zigzag_sequence::json solve_records = zigzag_sequence::json::array();
        std::ofstream history("sequence_history.csv");
        if(!history) throw std::runtime_error("Cannot write sequence_history.csv");
        history << "cycle,solved,hit_events,max_total_hits\n";

        // Normalize output tags to the actual mechanics mode rather than the
        // compatibility CLI alias. This makes a BC-disabled run identical to
        // the legacy zigzag_sequence naming, even if the user supplied
        // -growth_type zigzag_sequence_BC.
        tag = use_sequence_bc
            ? "bilayer_zigzag_sequence_BC"
            : "bilayer_zigzag_sequence";

        if(requested_sequence_bc_alias && !use_sequence_bc)
            std::cout
                << "[zigzag_sequence] boundary_conditions.enabled=false; "
                << "running the unconstrained recurring sequence.\n";
        else if(!requested_sequence_bc_alias && use_sequence_bc)
            std::cout
                << "[zigzag_sequence] boundary_conditions.enabled=true; "
                << "activating persistent clamps and final release.\n";

        Eigen::MatrixXb physicalClampMask =
            Eigen::MatrixXb::Constant(nVert, 3, false);
        Eigen::VectorXd physicalClampRegionId =
            Eigen::VectorXd::Zero(nVert);
        std::vector<int> clampRegionVertexCounts;
        std::ofstream cycleReadme;

        if(use_sequence_bc)
        {
            if(sequence.boundary_conditions.regions.size() != 2)
                throw std::runtime_error(
                    "zigzag_sequence_BC first version requires exactly two "
                    "rectangular clamp regions.");

            const Eigen::MatrixXb initialBC =
                mesh.getBoundaryConditions().getVertexBoundaryConditions();
            if(zigzag_sequence_bc::countFixedDofs(initialBC) != 0)
                throw std::runtime_error(
                    "zigzag_sequence_BC first version requires a geometry "
                    "without pre-existing vertex boundary conditions. Use "
                    "rectangle, curved_rectangle, or an unconstrained external mesh.");

            const auto clampResult =
                zigzag_sequence_bc::buildFullFixationMask(
                    materialCoordinates,
                    sequence.boundary_conditions);
            physicalClampMask = clampResult.vertex_mask;
            physicalClampRegionId = clampResult.region_id;
            clampRegionVertexCounts = clampResult.region_vertex_counts;
            zigzag_sequence_bc::applyVertexMask(mesh, physicalClampMask);

            cycleReadme.open("README_cycles.md");
            if(!cycleReadme)
                throw std::runtime_error(
                    "zigzag_sequence_BC: cannot open README_cycles.md.");
            cycleReadme
                << "# zigzag_sequence_BC cycle map\n\n"
                << "- JSON file: `" << cycle_file << "`\n"
                << "- Physical clamp coordinate system: material `(u,v)`\n"
                << "- Physical clamp mode: full fixation of `x`, `y`, and `z`\n"
                << "- Fixed physical-clamp vertices: "
                << clampResult.fixed_vertex_count << "\n"
                << "- Fixed physical-clamp DOFs: "
                << clampResult.fixed_dof_count << "\n"
                << "- Release after final cycle: "
                << (sequence.boundary_conditions.release_after_final_cycle ?
                    "true" : "false") << "\n\n";
            for(std::size_t region = 0;
                region < sequence.boundary_conditions.regions.size();
                ++region)
            {
                const auto& r = sequence.boundary_conditions.regions[region];
                cycleReadme
                    << "- Clamp " << (region + 1) << " (`" << r.name
                    << "`): center_uv_mm = ["
                    << r.center_uv_m(0) * 1e3 << ", "
                    << r.center_uv_m(1) * 1e3 << "], size_uv_mm = ["
                    << r.size_uv_m(0) * 1e3 << ", "
                    << r.size_uv_m(1) * 1e3 << "], rotation_deg = "
                    << r.rotation_deg << ", selected vertices = "
                    << clampRegionVertexCounts[region] << "\n";
            }
            cycleReadme
                << "\n| File | Physical cycle | Toolpath | Repeat | State |\n"
                << "|---|---:|---|---:|---|\n";
        }

        auto sanitizeIdentifier = [](const std::string& input)
        {
            std::string output;
            output.reserve(input.size());
            for(const char c : input)
            {
                if(std::isalnum(static_cast<unsigned char>(c)) ||
                   c == '_' || c == '-')
                    output.push_back(c);
                else
                    output.push_back('_');
            }
            return output.empty() ? std::string("toolpath") : output;
        };

        auto csvQuote = [](const std::string& input)
        {
            std::string escaped = "\"";
            for(const char c : input)
            {
                if(c == '\"') escaped += "\"\"";
                else escaped.push_back(c);
            }
            escaped += "\"";
            return escaped;
        };

        std::cout
            << "[" << growth_type << "] file=" << cycle_file
            << ", enabled_toolpaths=" << sequence.toolpaths.size()
            << ", hardening_beta=" << sequence.hardening.beta
            << ", hardening_floor=" << sequence.hardening.floor
            << "\n";

        // Natural state at cycle zero. a_r(top/bottom) evolves sequentially;
        // b_r is retained exactly for the full sequence.
        tVecMat2d& aformsBot =
            mesh.getRestConfiguration()
                .getFirstFundamentalForms<bottom>();
        tVecMat2d& aformsTop =
            mesh.getRestConfiguration()
                .getFirstFundamentalForms<top>();
        const tVecMat2d initialBforms =
            mesh.getRestConfiguration().getSecondFundamentalForms();

        Eigen::VectorXi totalPassCount(nFaces);
        totalPassCount.setZero();

        MaterialProperties_Iso_Constant matprop_bot(E, nu, h_total);
        MaterialProperties_Iso_Constant matprop_top(E, nu, h_total);
        CombinedOperator_Parametric<tMesh, Material_Isotropic, bottom>
            engOp_bot(matprop_bot);
        CombinedOperator_Parametric<tMesh, Material_Isotropic, top>
            engOp_top(matprop_top);
        EnergyOperatorList<tMesh> engOps({&engOp_bot, &engOp_top});

        // ---- exact-Hessian (TinyAD) verification gate ----------------------------------
        // Perturb to a curved state (so bending terms are active), then check the TinyAD
        // gradient against CombinedOperator_Parametric to machine precision, plus Hessian
        // symmetry and sparse-vs-HvP consistency. Verification-only; returns without solving.
        if(parser.parse<bool>("-verify_hessian", false))
        {
            const int nVv = mesh.getNumberOfVertices();
            const int nEv = mesh.getNumberOfEdges();
            const int nDv = 3 * nVv + nEv;
            Eigen::Map<Eigen::VectorXd> xmap(mesh.getDataPointer(), nDv);
            std::mt19937 rng(1);
            std::uniform_real_distribution<double> Ud(-1.0, 1.0);
            for(int k = 0; k < nVv; ++k) xmap(2 * nVv + k) += 0.02 * Ud(rng);   // z
            for(int e = 0; e < nEv; ++e) xmap(3 * nVv + e) += 0.02 * Ud(rng);   // directors
            mesh.updateDeformedConfiguration();

            Eigen::VectorXd gOp = Eigen::VectorXd::Zero(nDv);
            engOps.compute(mesh, gOp);

            const int hessian_threads =
                std::max(1, parser.parse<int>("-hessian_threads", 1));
            TinyADHessian_Bilayer<tMesh> tad(E, nu, h_total, hessian_threads);
            const Eigen::VectorXd gAD = tad.gradient(mesh);

            const double gdiff  = (gOp - gAD).cwiseAbs().maxCoeff();
            const double gscale = std::max(gOp.cwiseAbs().maxCoeff(), 1e-30);
            printf("[verify_hessian] gradient: ||g_op - g_tinyad||_inf = %.3e   rel = %.3e\n",
                   gdiff, gdiff / gscale);

            const Eigen::SparseMatrix<double> H = tad.assembleHessian(mesh);
            Eigen::VectorXd v = Eigen::VectorXd::Random(nDv);
            Eigen::VectorXd w = Eigen::VectorXd::Random(nDv);
            const Eigen::VectorXd Hv = tad.hessianVectorProduct(mesh, v);
            const Eigen::VectorXd Hw = tad.hessianVectorProduct(mesh, w);
            const double sym = std::abs(w.dot(Hv) - v.dot(Hw));
            const double spd = (H * v - Hv).cwiseAbs().maxCoeff();
            printf("[verify_hessian] symmetry |wHv - vHw| = %.3e   ||H@v - HvP||_inf = %.3e   nnz = %ld\n",
                   sym, spd, (long)H.nonZeros());
            printf("[verify_hessian] DONE (verification-only run)\n");
            return;
        }

        const std::string dump_iters_str =
            parser.parse<std::string>("-dump_iters", "");
        const std::vector<int> dump_iters =
            parse_int_list(dump_iters_str);
        const int max_iter =
            parser.parse<int>("-max_iter", -1);
        const Real eps_init_default = 1e-2;
        const Real tol = parser.parse<Real>("-tol", 1e-12);
        const bool stepwise = parser.parse<bool>("-stepwise", false);
        const bool write_cycle_state =
            parser.parse<bool>("-cycle_write_state", false);
        const std::string equilibrium_solver =
            parser.parse<std::string>("-equilibrium_solver", "hlbfgs");
        if(equilibrium_solver != "hlbfgs" &&
           equilibrium_solver != "trust_region" &&
           equilibrium_solver != "hybrid")
            throw std::runtime_error(
                "zigzag_sequence: -equilibrium_solver must be 'hlbfgs', "
                "'trust_region', or 'hybrid'.");
        const int hybrid_warmup_iters =
            parser.parse<int>("-hybrid_warmup_iters", 1000);
        const Real hybrid_gate_tol =
            parser.parse<Real>("-hybrid_gate_tol", 1e-4);
        if(hybrid_warmup_iters < 0)
            throw std::runtime_error(
                "zigzag_sequence: -hybrid_warmup_iters must be non-negative.");
        if(!std::isfinite(hybrid_gate_tol) || hybrid_gate_tol <= 0.0)
            throw std::runtime_error(
                "zigzag_sequence: -hybrid_gate_tol must be finite and > 0.");
        const Real equilibrium_gradient_tolerance =
            parser.parse<Real>("-equilibrium_grad_tol", 10.0 * tol);
        if(!std::isfinite(equilibrium_gradient_tolerance) ||
           equilibrium_gradient_tolerance <= 0.0)
            throw std::runtime_error(
                "zigzag_sequence: -equilibrium_grad_tol must be finite and > 0.");
        const bool adaptive_continuation =
            parser.parse<bool>("-sequence_adaptive", true);
        const Real continuation_initial_step =
            parser.parse<Real>("-sequence_initial_step", 1.0);
        const Real continuation_max_step =
            parser.parse<Real>("-sequence_max_step", 1.0);
        const Real continuation_min_step =
            parser.parse<Real>("-sequence_min_step", 1.0 / 64.0);
        const Real continuation_growth =
            parser.parse<Real>("-sequence_step_growth", 2.0);
        const int continuation_max_retries =
            parser.parse<int>("-sequence_max_retries", 8);
        if(!std::isfinite(continuation_initial_step) ||
           !std::isfinite(continuation_max_step) ||
           !std::isfinite(continuation_min_step) ||
           !std::isfinite(continuation_growth) ||
           continuation_initial_step <= 0.0 ||
           continuation_max_step <= 0.0 ||
           continuation_initial_step > continuation_max_step ||
           continuation_min_step > continuation_initial_step ||
           continuation_min_step <= 0.0 ||
           continuation_max_step > 1.0 ||
           continuation_growth < 1.0 ||
           continuation_max_retries < 0)
            throw std::runtime_error(
                "zigzag_sequence: invalid adaptive-continuation controls.");
        ShellEquilibrium::TrustRegionNewtonOptions trustRegionOptions;
        trustRegionOptions.gradientTolerance = equilibrium_gradient_tolerance;
        trustRegionOptions.maxIterations =
            parser.parse<int>("-trust_max_iterations", 100);
        trustRegionOptions.initialTrustRadius =
            parser.parse<Real>("-trust_initial_radius", 0.25);
        trustRegionOptions.maxTrustRadius =
            parser.parse<Real>("-trust_max_radius", 4.0);
        trustRegionOptions.minTrustRadius =
            parser.parse<Real>("-trust_min_radius", 1e-8);
        trustRegionOptions.vertexLengthScale =
            parser.parse<Real>("-trust_vertex_scale", h_total);
        trustRegionOptions.directorAngleScale =
            parser.parse<Real>("-trust_director_scale", 1.0);
        trustRegionOptions.cg.maxIterations =
            parser.parse<int>("-trust_cg_max_iterations", 250);
        const int hessian_threads =
            std::max(1, parser.parse<int>("-hessian_threads", 1));
        const bool track_sequence_stability =
            parser.parse<bool>("-sequence_stability", false);
        ShellEquilibrium::SmallestRitzPairOptions stabilityOptions;
        stabilityOptions.krylovDimension =
            parser.parse<int>("-stability_krylov_dimension", 40);
        stabilityOptions.maxRestarts =
            parser.parse<int>("-stability_max_restarts", 20);
        stabilityOptions.absoluteResidualTolerance =
            parser.parse<Real>("-stability_abs_residual_tol", 1e-10);
        stabilityOptions.relativeResidualTolerance =
            parser.parse<Real>("-stability_rel_residual_tol", 1e-8);
        const bool sequence_warm_start =
            parser.parse<bool>("-sequence_warm_start", true);
        const std::string metric_update_name =
            parser.parse<std::string>("-metric_update", "multiplicative");
        GrowthMetricUpdate metric_update;
        if(metric_update_name == "multiplicative")
            metric_update = GrowthMetricUpdate::Multiplicative;
        else if(metric_update_name == "recursive_linearized")
            metric_update = GrowthMetricUpdate::RecursiveLinearized;
        else if(metric_update_name == "reference_additive_linearized")
            metric_update = GrowthMetricUpdate::ReferenceAdditiveLinearized;
        else
            throw std::runtime_error(
                "zigzag_sequence: -metric_update must be 'multiplicative', "
                "'recursive_linearized', or 'reference_additive_linearized'.");
        const int minimize_every =
            parser.parse<int>("-sequence_minimize_every", 1);
        if(minimize_every < 1)
            throw std::runtime_error(
                "zigzag_sequence: -sequence_minimize_every must be >= 1.");
        if(adaptive_continuation && minimize_every != 1)
            throw std::runtime_error(
                "zigzag_sequence: adaptive continuation requires "
                "-sequence_minimize_every 1 so every physical path is equilibrated.");

        // ---- optional prescribed deformation at a physical-cycle boundary ----
        // Disabled unless -prescribed_at_cycle >= 0, so every existing run is
        // bit-for-bit unchanged. When enabled it rewrites the CURRENT vertex
        // DOFs once, at the boundary after physical cycle K, before cycle K+1's
        // toolpath mapping and load. The retained target metrics a_r(top/bot),
        // the reference curvature b_r, and the per-face pass history are left
        // untouched: this imposes an initial state for the next path, it is not
        // a maintained displacement boundary condition and it is not a
        // stress-free reset.
        const int prescribed_at_cycle =
            parser.parse<int>("-prescribed_at_cycle", -1);
        const bool prescribed_enabled = prescribed_at_cycle >= 0;
        const std::string prescribed_disp_csv =
            parser.parse<std::string>("-prescribed_disp_csv", "");
        const std::string prescribed_shape =
            parser.parse<std::string>("-prescribed_shape", "none");
        const Real prescribed_amplitude =
            parser.parse<Real>("-prescribed_amplitude", 0.0);
        const std::string prescribed_reference =
            parser.parse<std::string>("-prescribed_reference", "rest");
        const bool prescribed_reequilibrate =
            parser.parse<bool>("-prescribed_reequilibrate", false);
        const std::string prescribed_reeq_solver =
            parser.parse<std::string>(
                "-prescribed_reeq_solver", equilibrium_solver);
        const Real prescribed_reeq_grad_tol =
            parser.parse<Real>(
                "-prescribed_reeq_grad_tol", equilibrium_gradient_tolerance);
        const std::string prescribed_tag =
            parser.parse<std::string>("-prescribed_tag", "prescribed");
        if(prescribed_enabled)
        {
            const bool has_csv = !prescribed_disp_csv.empty();
            const bool has_shape = prescribed_shape != "none";
            if(has_csv == has_shape)
                throw std::runtime_error(
                    "prescribed_deformation: supply exactly one of "
                    "-prescribed_disp_csv <file> or -prescribed_shape "
                    "<crown|saddle>.");
            if(has_shape &&
               prescribed_shape != "crown" &&
               prescribed_shape != "saddle")
                throw std::runtime_error(
                    "prescribed_deformation: -prescribed_shape must be "
                    "'none', 'crown', or 'saddle'.");
            if(has_shape &&
               (!std::isfinite(prescribed_amplitude) ||
                prescribed_amplitude == 0.0))
                throw std::runtime_error(
                    "prescribed_deformation: -prescribed_amplitude must be "
                    "finite and non-zero when -prescribed_shape is used.");
            if(prescribed_reference != "rest" &&
               prescribed_reference != "current")
                throw std::runtime_error(
                    "prescribed_deformation: -prescribed_reference must be "
                    "'rest' (x_new = X_rest + U) or 'current' "
                    "(x_new = X_current + U).");
            if(prescribed_reeq_solver != "hlbfgs" &&
               prescribed_reeq_solver != "trust_region" &&
               prescribed_reeq_solver != "hybrid")
                throw std::runtime_error(
                    "prescribed_deformation: -prescribed_reeq_solver must be "
                    "'hlbfgs', 'trust_region', or 'hybrid'.");
            if(!std::isfinite(prescribed_reeq_grad_tol) ||
               prescribed_reeq_grad_tol <= 0.0)
                throw std::runtime_error(
                    "prescribed_deformation: -prescribed_reeq_grad_tol must be "
                    "finite and > 0.");
        }
        else if(!prescribed_disp_csv.empty() ||
                prescribed_shape != "none" ||
                prescribed_reequilibrate)
            throw std::runtime_error(
                "prescribed_deformation: -prescribed_disp_csv, "
                "-prescribed_shape and -prescribed_reequilibrate require "
                "-prescribed_at_cycle <k>; refusing to silently ignore them.");

        const std::string summary_filename =
            use_sequence_bc ? "cycle_summary.csv" : tag + "_summary.csv";
        std::ofstream summary(summary_filename);
        if(!summary)
            throw std::runtime_error(
                "zigzag_sequence: cannot open summary CSV: " +
                summary_filename);
        std::ofstream convergenceSummary("sequence_convergence.csv");
        if(!convergenceSummary)
            throw std::runtime_error(
                "zigzag_sequence: cannot open sequence_convergence.csv.");
        convergenceSummary
            << "executed_cycle_index,minimization_performed,state_kind,"
            << "substep_attempt,"
            << "lambda_from,lambda_trial,step_size,retry_count,solver,"
            << "solver_code,iterations,evaluations,final_gradient_norm,"
            << "equilibrium_accepted,recomputed_energy\n";
        convergenceSummary << std::setprecision(17);
        std::ofstream stabilitySummary("sequence_stability.csv");
        if(!stabilitySummary)
            throw std::runtime_error(
                "zigzag_sequence: cannot open sequence_stability.csv.");
        stabilitySummary
            << "executed_cycle_index,evaluated,smallest_ritz_value,"
            << "residual_absolute,residual_relative,residual_converged,"
            << "classification,rigid_modes,operator_evaluations\n";
        stabilitySummary << std::setprecision(17);
        summary << std::setprecision(17);
        if(use_sequence_bc)
            summary
                << "output_index,is_release_state,physical_clamps_active,";
        summary
            << "executed_cycle_index,toolpath_id,toolpath_sequence_index,"
            << "repeat_index,repeat_count,operation_count,hit_event_count,"
            << "covered_face_count,max_hits_on_one_face_this_cycle,"
            << "max_total_face_hit_count,hardening_factor_min,"
            << "hardening_factor_mean,hardening_factor_max,"
            << "max_abs_top_increment,max_abs_bottom_increment,"
            << "max_displacement,min_U3,max_U3,total_energy,"
            << "max_tangency_error,max_bbar_drift,mapping_file,final_file\n";
        summary.flush();

        // Material-coordinate margin filtering is applied independently to
        // each toolpath recipe, after its face/strip events are collected.
        auto filterHitsByMargins =
            [&](std::vector<zigzag::MaterialHit>& hits,
                Eigen::VectorXi& hitsThisCycle)
            {
                if(margin_x <= 0.0 && margin_y <= 0.0)
                    return;

                const Real uMin = materialCoordinates.col(0).minCoeff();
                const Real uMax = materialCoordinates.col(0).maxCoeff();
                const Real vMin = materialCoordinates.col(1).minCoeff();
                const Real vMax = materialCoordinates.col(1).maxCoeff();

                Eigen::VectorXi marginVertex(nVert);
                marginVertex.setZero();
                for(int i = 0; i < nVert; ++i)
                {
                    const Real u = materialCoordinates(i,0);
                    const Real v = materialCoordinates(i,1);
                    if(u <= uMin + margin_x ||
                       u >= uMax - margin_x ||
                       v <= vMin + margin_y ||
                       v >= vMax - margin_y)
                        marginVertex(i) = 1;
                }

                Eigen::VectorXi faceEnabled =
                    Eigen::VectorXi::Ones(nFaces);
                for(int face = 0; face < nFaces; ++face)
                {
                    if(marginVertex(Connect(face,0)) == 1 &&
                       marginVertex(Connect(face,1)) == 1 &&
                       marginVertex(Connect(face,2)) == 1)
                        faceEnabled(face) = 0;
                }

                hits.erase(
                    std::remove_if(
                        hits.begin(),
                        hits.end(),
                        [&](const zigzag::MaterialHit& hit)
                        {
                            return faceEnabled(hit.face_idx) == 0;
                        }),
                    hits.end());

                hitsThisCycle.setZero();
                for(const auto& hit : hits)
                    hitsThisCycle(hit.face_idx) += 1;
            };

        int bcOutputIndex = -1;
        bool bcOutputIsRelease = false;
        bool bcOutputPhysicalClampsActive = false;
        bool bcOutputGaugeActive = false;

        // Mapping diagnostic on the current start-of-cycle geometry. Returns
        // max |n dot d| for the compact sequence summary.
        auto writeSequenceMapping =
            [&](const int executed_cycle,
                const std::string& filebase,
                const std::vector<zigzag::MaterialHit>& hits,
                const Eigen::VectorXi& hitsThisCycle,
                const std::vector<std::vector<int>>& hitsByFace,
                const int maxHitsPerFace,
                const zigzag_sequence::ZigZagOperationConfig& operation) -> Real
            {
                const Eigen::MatrixXd Xcurrent =
                    mesh.getCurrentConfiguration().getVertices();
                WriteVTK writer(Xcurrent, Connect);

                const Eigen::VectorXd materialU =
                    materialCoordinates.col(0);
                const Eigen::VectorXd materialV =
                    materialCoordinates.col(1);
                const Eigen::VectorXd hitsReal =
                    hitsThisCycle.cast<Real>();
                const Eigen::VectorXd totalBefore =
                    totalPassCount.cast<Real>();
                const Eigen::VectorXd cycleField =
                    Eigen::VectorXd::Constant(
                        nFaces, static_cast<Real>(executed_cycle));

                writer.addScalarFieldToVertices(materialU, "material_u");
                writer.addScalarFieldToVertices(materialV, "material_v");
                writer.addScalarFieldToFaces(cycleField, "executed_cycle_index");
                writer.addScalarFieldToFaces(hitsReal, "hits_this_cycle");
                writer.addScalarFieldToFaces(
                    totalBefore, "total_hits_before_cycle");

                Real maxTangencyError = 0.0;
                for(int rank = 0; rank < maxHitsPerFace; ++rank)
                {
                    Eigen::VectorXd angles =
                        Eigen::VectorXd::Zero(nFaces);
                    Eigen::VectorXd active =
                        Eigen::VectorXd::Zero(nFaces);
                    Eigen::VectorXd stripIndex =
                        Eigen::VectorXd::Constant(nFaces, -1.0);
                    Eigen::VectorXd baseGtop =
                        Eigen::VectorXd::Zero(nFaces);
                    Eigen::VectorXd baseGbot =
                        Eigen::VectorXd::Zero(nFaces);
                    Eigen::VectorXd baseOrtho = Eigen::VectorXd::Zero(nFaces);
                    Eigen::VectorXd baseOrthoBottom = Eigen::VectorXd::Zero(nFaces);

                    for(int face = 0; face < nFaces; ++face)
                    {
                        if(rank >= (int)hitsByFace[face].size())
                            continue;
                        const auto& hit = hits[hitsByFace[face][rank]];
                        angles(face) = hit.angle_rad;
                        active(face) = 1.0;
                        stripIndex(face) =
                            static_cast<Real>(hit.strip_idx);
                        baseGtop(face) = hit.gtop;
                        baseGbot(face) = hit.gbot;
                        baseOrtho(face) = operation.ortho_top.at(hit.strip_idx);
                        baseOrthoBottom(face) = operation.ortho_bottom.at(hit.strip_idx);
                    }

                    Eigen::MatrixXd directions;
                    GrowthHelper<tMesh>::
                        mapMaterialAnglesToCurrentShellDirections(
                            mesh,
                            materialCoordinates,
                            angles,
                            directions);

                    for(int face = 0; face < nFaces; ++face)
                    {
                        if(active(face) == 0.0)
                        {
                            directions.row(face).setZero();
                            continue;
                        }
                        const Eigen::Vector3d x0 =
                            Xcurrent.row(Connect(face,0)).transpose();
                        const Eigen::Vector3d x1 =
                            Xcurrent.row(Connect(face,1)).transpose();
                        const Eigen::Vector3d x2 =
                            Xcurrent.row(Connect(face,2)).transpose();
                        const Eigen::Vector3d normal =
                            (x1-x0).cross(x2-x0).normalized();
                        maxTangencyError = std::max(
                            maxTangencyError,
                            std::abs(normal.dot(
                                directions.row(face).transpose())));
                    }

                    const std::string suffix =
                        helpers::ToString(rank + 1, 2);
                    writer.addVectorFieldToFaces(
                        directions,
                        "growth_dir_3d_hit_" + suffix);
                    writer.addScalarFieldToFaces(
                        active,
                        "hit_active_" + suffix);
                    writer.addScalarFieldToFaces(
                        angles,
                        "growth_angle_material_hit_" + suffix);
                    writer.addScalarFieldToFaces(
                        stripIndex,
                        "strip_index_hit_" + suffix);
                    writer.addScalarFieldToFaces(
                        baseGtop,
                        "base_gtop_hit_" + suffix);
                    writer.addScalarFieldToFaces(
                        baseGbot,
                        "base_gbot_hit_" + suffix);
                    writer.addScalarFieldToFaces(
                        baseOrtho,
                        "base_ortho_hit_" + suffix); // Legacy alias for top.
                    writer.addScalarFieldToFaces(baseOrtho, "base_ortho_top_hit_" + suffix);
                    writer.addScalarFieldToFaces(baseOrthoBottom, "base_ortho_bottom_hit_" + suffix);
                }

                writer.write(filebase);
                return maxTangencyError;
            };

        // Set only while an imposed prescribed state is exported, so the
        // cycle-0 output of an unmodified run keeps its exact field list.
        bool forceStateCurvature = false;

        auto writeSequenceState =
            [&](const int executed_cycle,
                const std::string& filebase,
                const Eigen::VectorXi& hitsThisCycle,
                const Eigen::VectorXd& incG1Top,
                const Eigen::VectorXd& incG2Top,
                const Eigen::VectorXd& incG1Bot,
                const Eigen::VectorXd& incG2Bot,
                const Eigen::VectorXd& qFirst,
                const Eigen::VectorXd& qLast)
            {
                const Eigen::MatrixXd X0 =
                    mesh.getRestConfiguration().getVertices();
                const Eigen::MatrixXd X =
                    mesh.getCurrentConfiguration().getVertices();
                WriteVTK writer(X, Connect);

                const Eigen::MatrixXd U = X - X0;
                const Eigen::VectorXd U3 = U.col(2);
                const Eigen::VectorXd Umag = U.rowwise().norm();
                const Eigen::VectorXd materialU =
                    materialCoordinates.col(0);
                const Eigen::VectorXd materialV =
                    materialCoordinates.col(1);
                const Eigen::VectorXd hitsReal =
                    hitsThisCycle.cast<Real>();
                const Eigen::VectorXd totalHitsReal =
                    totalPassCount.cast<Real>();
                const Eigen::VectorXd cycleField =
                    Eigen::VectorXd::Constant(
                        nFaces, static_cast<Real>(executed_cycle));

                Eigen::VectorXd qNext =
                    Eigen::VectorXd::Zero(nFaces);
                Eigen::VectorXd aTop11(nFaces), aTop12(nFaces), aTop22(nFaces);
                Eigen::VectorXd aBot11(nFaces), aBot12(nFaces), aBot22(nFaces);
                Eigen::VectorXd bRef11(nFaces), bRef12(nFaces), bRef22(nFaces);
                Eigen::VectorXd bRefDrift(nFaces);

                const tVecMat2d& currentBforms =
                    mesh.getRestConfiguration()
                        .getSecondFundamentalForms();
                for(int face = 0; face < nFaces; ++face)
                {
                    if(hitsThisCycle(face) > 0)
                        qNext(face) =
                            zigzag_sequence::hardeningFactor(
                                sequence.hardening,
                                totalPassCount(face));

                    aTop11(face) = aformsTop[face](0,0);
                    aTop12(face) = aformsTop[face](0,1);
                    aTop22(face) = aformsTop[face](1,1);
                    aBot11(face) = aformsBot[face](0,0);
                    aBot12(face) = aformsBot[face](0,1);
                    aBot22(face) = aformsBot[face](1,1);
                    bRef11(face) = currentBforms[face](0,0);
                    bRef12(face) = currentBforms[face](0,1);
                    bRef22(face) = currentBforms[face](1,1);
                    bRefDrift(face) =
                        (currentBforms[face] - initialBforms[face]).norm();
                }

                writer.addVectorFieldToVertices(U, "U_from_cycle0");
                writer.addScalarFieldToVertices(U3, "U3_from_cycle0");
                writer.addScalarFieldToVertices(Umag, "Umag_from_cycle0");
                writer.addScalarFieldToVertices(materialU, "material_u");
                writer.addScalarFieldToVertices(materialV, "material_v");

                if(use_sequence_bc)
                {
                    const Eigen::VectorXd outputIndexField =
                        Eigen::VectorXd::Constant(
                            nVert, static_cast<Real>(bcOutputIndex));
                    const Eigen::VectorXd releaseField =
                        Eigen::VectorXd::Constant(
                            nVert, bcOutputIsRelease ? 1.0 : 0.0);
                    const Eigen::VectorXd physicalActiveField =
                        Eigen::VectorXd::Constant(
                            nVert,
                            bcOutputPhysicalClampsActive ? 1.0 : 0.0);
                    const Eigen::VectorXd gaugeActiveField =
                        Eigen::VectorXd::Constant(
                            nVert, bcOutputGaugeActive ? 1.0 : 0.0);
                    const Eigen::VectorXd activeFixedDofs =
                        zigzag_sequence_bc::activeFixedDofCountPerVertex(
                            mesh.getBoundaryConditions()
                                .getVertexBoundaryConditions());
                    writer.addScalarFieldToVertices(
                        physicalClampRegionId, "physical_clamp_region_id");
                    writer.addScalarFieldToVertices(
                        outputIndexField, "output_index");
                    writer.addScalarFieldToVertices(
                        releaseField, "is_release_state");
                    writer.addScalarFieldToVertices(
                        physicalActiveField, "physical_clamps_active");
                    writer.addScalarFieldToVertices(
                        gaugeActiveField, "numerical_gauge_active");
                    writer.addScalarFieldToVertices(
                        activeFixedDofs, "active_fixed_dof_count");
                }

                writer.addScalarFieldToFaces(
                    cycleField, "executed_cycle_index");
                writer.addScalarFieldToFaces(
                    hitsReal, "hits_this_cycle");
                writer.addScalarFieldToFaces(
                    totalHitsReal, "total_hit_count");
                writer.addScalarFieldToFaces(
                    qFirst, "hardening_factor_first_hit");
                writer.addScalarFieldToFaces(
                    qLast, "hardening_factor_last_hit");
                writer.addScalarFieldToFaces(
                    qNext, "hardening_factor_next_hit");

                writer.addScalarFieldToFaces(
                    incG1Top, "effective_increment_g1_top_sum");
                writer.addScalarFieldToFaces(
                    incG2Top, "effective_increment_g2_top_sum");
                writer.addScalarFieldToFaces(
                    incG1Bot, "effective_increment_g1_bot_sum");
                writer.addScalarFieldToFaces(
                    incG2Bot, "effective_increment_g2_bot_sum");

                writer.addScalarFieldToFaces(aTop11, "abar_top_11");
                writer.addScalarFieldToFaces(aTop12, "abar_top_12");
                writer.addScalarFieldToFaces(aTop22, "abar_top_22");
                writer.addScalarFieldToFaces(aBot11, "abar_bot_11");
                writer.addScalarFieldToFaces(aBot12, "abar_bot_12");
                writer.addScalarFieldToFaces(aBot22, "abar_bot_22");
                writer.addScalarFieldToFaces(bRef11, "bbar_ref_11");
                writer.addScalarFieldToFaces(bRef12, "bbar_ref_12");
                writer.addScalarFieldToFaces(bRef22, "bbar_ref_22");
                writer.addScalarFieldToFaces(
                    bRefDrift, "bbar_reference_drift_norm");

                // Curvature is skipped for the flat cycle-0 output. A
                // prescribed state can be curved at any boundary, including
                // k = 0, so it opts in explicitly.
                if(executed_cycle > 0 || forceStateCurvature)
                {
                    Eigen::VectorXd gauss(nFaces);
                    Eigen::VectorXd mean(nFaces);
                    ComputeCurvatures<tMesh> computeCurvatures;
                    computeCurvatures.compute(mesh, gauss, mean);
                    writer.addScalarFieldToFaces(gauss, "gauss");
                    writer.addScalarFieldToFaces(mean, "mean");
                }

                writer.write(filebase);
            };

        const Eigen::VectorXi zeroHits =
            Eigen::VectorXi::Zero(nFaces);
        const Eigen::VectorXd zeroField =
            Eigen::VectorXd::Zero(nFaces);
        if(use_sequence_bc)
        {
            // Output index 000 is the undeformed initial geometry. The
            // physical clamp mask is already active, but no growth cycle has
            // been applied and no equilibrium solve has been performed yet.
            bcOutputIndex = 0;
            bcOutputIsRelease = false;
            bcOutputPhysicalClampsActive = true;
            bcOutputGaugeActive = false;

            const std::string initialBase = "cycle_000";
            writeSequenceState(
                0,
                initialBase,
                zeroHits,
                zeroField,
                zeroField,
                zeroField,
                zeroField,
                zeroField,
                zeroField);

            summary
                << 0 << ","       // output_index
                << 0 << ","       // is_release_state
                << 1 << ","       // physical_clamps_active
                << 0 << ","       // executed_cycle_index
                << csvQuote("INITIAL") << ","
                << 0 << ","       // toolpath_sequence_index
                << 0 << ","       // repeat_index
                << 0 << ","       // repeat_count
                << 0 << ","       // operation_count
                << 0 << ","       // hit_event_count
                << 0 << ","       // covered_face_count
                << 0 << ","       // max_hits_on_one_face_this_cycle
                << 0 << ","       // max_total_face_hit_count
                << 0.0 << ","     // hardening_factor_min
                << 0.0 << ","     // hardening_factor_mean
                << 0.0 << ","     // hardening_factor_max
                << 0.0 << ","     // max_abs_top_increment
                << 0.0 << ","     // max_abs_bottom_increment
                << 0.0 << ","     // max_displacement
                << 0.0 << ","     // min_U3
                << 0.0 << ","     // max_U3
                << 0.0 << ","     // total_energy at the natural initial state
                << 0.0 << ","     // max_tangency_error
                << 0.0 << ","     // max_bbar_drift
                << csvQuote("") << ","
                << csvQuote(initialBase + ".vtp") << "\n";
            summary.flush();

            cycleReadme
                << "| `" << initialBase << ".vtp` | 0 | - | - | "
                << "initial geometry; physical clamps active; no growth applied |\n";
            cycleReadme.flush();
        }
        else
        {
            // Preserve the original zigzag_sequence initial output exactly.
            writeSequenceState(
                0,
                tag + "_cycle_000_initial",
                zeroHits,
                zeroField,
                zeroField,
                zeroField,
                zeroField,
                zeroField,
                zeroField);
        }

        int executed_cycle = 0;
        int toolpath_sequence_index = 0;
        int total_sequence_cycles = 0;
        for(const auto& scheduled : sequence.toolpaths)
            total_sequence_cycles += scheduled.repeat;

        if(prescribed_enabled &&
           prescribed_at_cycle >= total_sequence_cycles)
            throw std::runtime_error(
                "prescribed_deformation: -prescribed_at_cycle " +
                std::to_string(prescribed_at_cycle) +
                " has no following physical cycle; the sequence executes " +
                std::to_string(total_sequence_cycles) +
                " cycles, so valid boundaries are 0.." +
                std::to_string(total_sequence_cycles - 1) +
                " (k means 'after cycle k, before cycle k+1').");

        // Imposes the prescribed geometry on the CURRENT vertex DOFs at the
        // boundary after physical cycle 'boundary_cycle'. a_r(top/bottom), b_r
        // and totalPassCount are deliberately never touched here, so the
        // elastic energy functional is exactly the one the sequence had already
        // accumulated: the imposed geometry is a new initial state for the next
        // toolpath, not a stress-free configuration and not a constraint that
        // is held during the following solve.
        int prescribedAppliedCount = 0;
        auto imposePrescribedDeformation =
            [&](const int boundary_cycle)
            {
                ++prescribedAppliedCount;
                const int nEdges = mesh.getNumberOfEdges();
                const int nDofs = 3 * nVert + nEdges;
                const std::string boundaryLabel =
                    helpers::ToString(boundary_cycle, 3);
                const std::string imposedBase =
                    tag + "_" + prescribed_tag + "_after_cycle_" +
                    boundaryLabel + "_imposed";
                const std::string reeqBase =
                    tag + "_" + prescribed_tag + "_after_cycle_" +
                    boundaryLabel + "_reequilibrated";

                const Eigen::MatrixXd Xrest =
                    mesh.getRestConfiguration().getVertices();
                const Eigen::MatrixXd Xpre =
                    mesh.getCurrentConfiguration().getVertices();

                Eigen::MatrixXd requested(nVert, 3);
                requested.setZero();
                if(!prescribed_disp_csv.empty())
                    requested =
                        loadVertexDisplacementCsv(prescribed_disp_csv, nVert);
                else
                {
                    // Smooth analytic z-displacement over the material domain,
                    // for controlled runs where generating a full CSV is not
                    // worth it. Identical convention as the CSV path.
                    const Real uMin = materialCoordinates.col(0).minCoeff();
                    const Real uMax = materialCoordinates.col(0).maxCoeff();
                    const Real vMin = materialCoordinates.col(1).minCoeff();
                    const Real vMax = materialCoordinates.col(1).maxCoeff();
                    const Real uHalf = 0.5 * (uMax - uMin);
                    const Real vHalf = 0.5 * (vMax - vMin);
                    if(uHalf <= 0.0 || vHalf <= 0.0)
                        throw std::runtime_error(
                            "prescribed_deformation: degenerate material "
                            "bounding box; cannot build an analytic shape.");
                    const Real uMid = 0.5 * (uMax + uMin);
                    const Real vMid = 0.5 * (vMax + vMin);
                    for(int i = 0; i < nVert; ++i)
                    {
                        const Real un =
                            (materialCoordinates(i,0) - uMid) / uHalf;
                        const Real vn =
                            (materialCoordinates(i,1) - vMid) / vHalf;
                        requested(i,2) =
                            prescribed_shape == "crown" ?
                                prescribed_amplitude *
                                std::cos(0.5 * M_PI * un) *
                                std::cos(0.5 * M_PI * vn) :
                                prescribed_amplitude * (un*un - vn*vn);
                    }
                }

                const Eigen::MatrixXd& base =
                    prescribed_reference == "rest" ? Xrest : Xpre;

                const Eigen::Ref<const Eigen::MatrixXb> vertexBC =
                    mesh.getBoundaryConditions()
                        .getVertexBoundaryConditions();
                const bool hasVertexBC = (vertexBC.rows() == nVert);

                Eigen::Ref<Eigen::MatrixXd> vertices =
                    mesh.getCurrentConfiguration().getVertices();
                int blockedDofs = 0;
                Real maxBlockedRequest = 0.0;
                for(int i = 0; i < nVert; ++i)
                    for(int j = 0; j < 3; ++j)
                    {
                        const Real target = base(i,j) + requested(i,j);
                        if(!std::isfinite(target))
                            throw std::runtime_error(
                                "prescribed_deformation: non-finite target "
                                "coordinate for vertex " + std::to_string(i) +
                                ".");
                        if(hasVertexBC && vertexBC(i,j))
                        {
                            // A fixed DOF stays where the boundary conditions
                            // put it; the request is reported, never applied.
                            ++blockedDofs;
                            maxBlockedRequest = std::max(
                                maxBlockedRequest,
                                std::abs(target - Xpre(i,j)));
                            continue;
                        }
                        vertices(i,j) = target;
                    }
                mesh.updateDeformedConfiguration();

                // Residual of the imposed state under the retained metrics,
                // restricted to the free DOFs. A large value is expected and is
                // exactly the point: the imposed shape is not an equilibrium.
                auto freeGradientNorm = [&]() -> Real
                {
                    Eigen::VectorXd gradient = Eigen::VectorXd::Zero(nDofs);
                    engOps.compute(mesh, gradient);
                    const Eigen::Ref<const Eigen::VectorXb> edgeBC =
                        mesh.getBoundaryConditions()
                            .getEdgeBoundaryConditions();
                    const bool hasEdgeBC = (edgeBC.size() == nEdges);
                    Real sum = 0.0;
                    for(int i = 0; i < nVert; ++i)
                        for(int j = 0; j < 3; ++j)
                            if(!(hasVertexBC && vertexBC(i,j)))
                                sum += gradient(j*nVert + i) *
                                       gradient(j*nVert + i);
                    for(int e = 0; e < nEdges; ++e)
                        if(!(hasEdgeBC && edgeBC(e)))
                            sum += gradient(3*nVert + e) *
                                   gradient(3*nVert + e);
                    return std::sqrt(sum);
                };

                // Full per-vertex state export, keyed by vertex index, in
                // metres. U_from_rest is the displacement from the reference
                // geometry; U_step is the change produced by this imposition.
                auto exportStateCsv =
                    [&](const std::string& filename)
                    {
                        std::ofstream out(filename);
                        if(!out)
                            throw std::runtime_error(
                                "prescribed_deformation: cannot open " +
                                filename);
                        out << std::setprecision(17);
                        out << "vertex_index,material_u,material_v,x,y,z,"
                            << "U_from_rest_x,U_from_rest_y,U_from_rest_z,"
                            << "U_step_x,U_step_y,U_step_z,"
                            << "requested_dx,requested_dy,requested_dz,"
                            << "bc_fixed_x,bc_fixed_y,bc_fixed_z\n";
                        const Eigen::MatrixXd Xnow =
                            mesh.getCurrentConfiguration().getVertices();
                        for(int i = 0; i < nVert; ++i)
                        {
                            out << i << ","
                                << materialCoordinates(i,0) << ","
                                << materialCoordinates(i,1) << ","
                                << Xnow(i,0) << ","
                                << Xnow(i,1) << ","
                                << Xnow(i,2) << ","
                                << (Xnow(i,0) - Xrest(i,0)) << ","
                                << (Xnow(i,1) - Xrest(i,1)) << ","
                                << (Xnow(i,2) - Xrest(i,2)) << ","
                                << (Xnow(i,0) - Xpre(i,0)) << ","
                                << (Xnow(i,1) - Xpre(i,1)) << ","
                                << (Xnow(i,2) - Xpre(i,2)) << ","
                                << requested(i,0) << ","
                                << requested(i,1) << ","
                                << requested(i,2) << ",";
                            for(int j = 0; j < 3; ++j)
                                out << ((hasVertexBC && vertexBC(i,j)) ? 1 : 0)
                                    << (j == 2 ? '\n' : ',');
                        }
                    };

                // Summary row reusing the sequence schema: hit statistics are
                // zero because no toolpath was applied.
                auto writePrescribedSummaryRow =
                    [&](const std::string& kind,
                        const std::string& filebase,
                        const Real energy)
                    {
                        const Eigen::MatrixXd Xnow =
                            mesh.getCurrentConfiguration().getVertices();
                        const Eigen::MatrixXd U = Xnow - Xrest;
                        Real maxDrift = 0.0;
                        const tVecMat2d& bforms =
                            mesh.getRestConfiguration()
                                .getSecondFundamentalForms();
                        for(int face = 0; face < nFaces; ++face)
                            maxDrift = std::max(
                                maxDrift,
                                (bforms[face] - initialBforms[face]).norm());
                        if(use_sequence_bc)
                            summary
                                << boundary_cycle << ","  // output_index
                                << 0 << ","               // is_release_state
                                << 1 << ",";              // clamps active
                        summary
                            << boundary_cycle << ","
                            << csvQuote(kind) << ","
                            // toolpath_sequence_index, repeat_index,
                            // repeat_count, operation_count, hit_event_count,
                            // covered_face_count, max_hits_on_one_face
                            << 0 << "," << 0 << "," << 0 << "," << 0 << ","
                            << 0 << "," << 0 << "," << 0 << ","
                            // The retained pass history is reported as-is: no
                            // toolpath ran, but nothing was reset either.
                            << totalPassCount.maxCoeff() << ","
                            // hardening min/mean/max, top/bottom increments
                            << 0.0 << "," << 0.0 << "," << 0.0 << ","
                            << 0.0 << "," << 0.0 << ","
                            << U.rowwise().norm().maxCoeff() << ","
                            << U.col(2).minCoeff() << ","
                            << U.col(2).maxCoeff() << ","
                            << energy << ","
                            << 0.0 << ","
                            << maxDrift << ","
                            << csvQuote("") << ","
                            << csvQuote(filebase + ".vtp") << "\n";
                        summary.flush();
                    };

                const Real imposedEnergy = engOps.compute(mesh);
                const Real imposedGradient = freeGradientNorm();

                convergenceSummary
                    << boundary_cycle << ",0,prescribed_imposed,0,1,1,0,0,"
                    << prescribed_reeq_solver << ",-1,0,0,"
                    << imposedGradient << ",0," << imposedEnergy << "\n";
                convergenceSummary.flush();

                bcOutputIndex = boundary_cycle;
                bcOutputIsRelease = false;
                bcOutputPhysicalClampsActive = use_sequence_bc;
                bcOutputGaugeActive = false;
                forceStateCurvature = true;
                writeSequenceState(
                    boundary_cycle,
                    imposedBase,
                    zeroHits,
                    zeroField,
                    zeroField,
                    zeroField,
                    zeroField,
                    zeroField,
                    zeroField);
                exportStateCsv(imposedBase + "_vertices.csv");
                writePrescribedSummaryRow(
                    "PRESCRIBED_IMPOSED", imposedBase, imposedEnergy);
                if(write_cycle_state)
                    mesh.writeToFile(imposedBase + "_state");

                std::cout
                    << "[prescribed_deformation] boundary_after_cycle="
                    << boundary_cycle
                    << ", source="
                    << (prescribed_disp_csv.empty() ?
                        ("analytic:" + prescribed_shape) :
                        ("csv:" + prescribed_disp_csv))
                    << ", reference=" << prescribed_reference
                    << ", max_requested_norm="
                    << requested.rowwise().norm().maxCoeff()
                    << ", blocked_fixed_dofs=" << blockedDofs
                    << ", max_blocked_request=" << maxBlockedRequest
                    << ", edge_directors_retained=" << nEdges
                    << ", energy=" << imposedEnergy
                    << ", free_gradient_norm=" << imposedGradient
                    << ", abar_top_bot_unchanged=1, bbar_unchanged=1"
                    << ", pass_history_unchanged=1"
                    << ", reequilibrate="
                    << (prescribed_reequilibrate ? 1 : 0)
                    << "\n";

                if(prescribed_reequilibrate)
                {
                    // Same retained target state: only the current DOFs move.
                    int code = -1;
                    int iterations = 0;
                    int evaluations = 0;
                    Real gradient =
                        std::numeric_limits<Real>::quiet_NaN();
                    bool accepted = false;
                    Real energy = std::numeric_limits<Real>::quiet_NaN();

                    if(prescribed_reeq_solver == "hlbfgs")
                    {
                        Real eps_reeq = eps_init_default;
                        if(use_sequence_bc)
                            minimizeEnergyReduced(
                                engOps,
                                eps_reeq,
                                tol,
                                stepwise,
                                nullptr,
                                max_iter);
                        else
                            minimizeEnergy(
                                engOps,
                                eps_reeq,
                                tol,
                                stepwise,
                                nullptr,
                                max_iter);
                        mesh.updateDeformedConfiguration();
                        code = lastMinimization.code;
                        iterations = lastMinimization.iterations;
                        evaluations = lastMinimization.evaluations;
                        gradient = lastMinimization.gradientNorm;
                        accepted = lastMinimization.accepted(
                            prescribed_reeq_grad_tol);
                        energy = engOps.compute(mesh);
                    }
                    else if(prescribed_reeq_solver == "hybrid")
                    {
                        int lbfgs_iters = 0;
                        int lbfgs_evals = 0;
                        if(hybrid_warmup_iters > 0)
                        {
                            Real eps_reeq = eps_init_default;
                            if(use_sequence_bc)
                                minimizeEnergyReduced(
                                    engOps,
                                    eps_reeq,
                                    hybrid_gate_tol,
                                    false,
                                    nullptr,
                                    hybrid_warmup_iters);
                            else
                                minimizeEnergy(
                                    engOps,
                                    eps_reeq,
                                    hybrid_gate_tol,
                                    false,
                                    nullptr,
                                    hybrid_warmup_iters);
                            mesh.updateDeformedConfiguration();
                            lbfgs_iters = lastMinimization.iterations;
                            lbfgs_evals = lastMinimization.evaluations;
                        }

                        TinyADHessian_Bilayer<tMesh> reeqHessian(
                            E, nu, h_total, hessian_threads);
                        ShellEquilibrium::TrustRegionNewtonOptions reeqOptions =
                            trustRegionOptions;
                        reeqOptions.gradientTolerance =
                            prescribed_reeq_grad_tol;
                        const ShellEquilibrium::TrustRegionNewtonReport report =
                            ShellEquilibrium::solveShellEquilibrium(
                                mesh,
                                engOps,
                                reeqHessian,
                                reeqOptions);
                        code = static_cast<int>(report.status);
                        iterations = lbfgs_iters + report.iterations;
                        evaluations =
                            lbfgs_evals + report.energyEvaluations +
                            report.gradientEvaluations;
                        gradient = report.gradientNorm;
                        accepted = report.accepted;
                        energy = report.energy;
                    }
                    else
                    {
                        TinyADHessian_Bilayer<tMesh> reeqHessian(
                            E, nu, h_total, hessian_threads);
                        ShellEquilibrium::TrustRegionNewtonOptions reeqOptions =
                            trustRegionOptions;
                        reeqOptions.gradientTolerance =
                            prescribed_reeq_grad_tol;
                        const ShellEquilibrium::TrustRegionNewtonReport report =
                            ShellEquilibrium::solveShellEquilibrium(
                                mesh,
                                engOps,
                                reeqHessian,
                                reeqOptions);
                        code = static_cast<int>(report.status);
                        iterations = report.iterations;
                        evaluations =
                            report.energyEvaluations +
                            report.gradientEvaluations;
                        gradient = report.gradientNorm;
                        accepted = report.accepted;
                        energy = report.energy;
                    }

                    convergenceSummary
                        << boundary_cycle << ",1,prescribed_reeq,1,1,1,0,0,"
                        << prescribed_reeq_solver << ","
                        << code << ","
                        << iterations << ","
                        << evaluations << ","
                        << gradient << ","
                        << (accepted ? 1 : 0) << ","
                        << energy << "\n";
                    convergenceSummary.flush();

                    if(!accepted)
                        throw std::runtime_error(
                            "prescribed_deformation: re-equilibration at the "
                            "prescribed state failed the strict residual test "
                            "(solver=" + prescribed_reeq_solver +
                            ", code=" + std::to_string(code) +
                            ", gradient_norm=" + std::to_string(gradient) +
                            ", tolerance=" +
                            std::to_string(prescribed_reeq_grad_tol) + ").");

                    writeSequenceState(
                        boundary_cycle,
                        reeqBase,
                        zeroHits,
                        zeroField,
                        zeroField,
                        zeroField,
                        zeroField,
                        zeroField,
                        zeroField);
                    exportStateCsv(reeqBase + "_vertices.csv");
                    writePrescribedSummaryRow(
                        "PRESCRIBED_REEQ", reeqBase, energy);
                    if(write_cycle_state)
                        mesh.writeToFile(reeqBase + "_state");

                    std::cout
                        << "[prescribed_deformation] re-equilibrated at the "
                        << "retained target state: solver="
                        << prescribed_reeq_solver
                        << ", iterations=" << iterations
                        << ", gradient_norm=" << gradient
                        << ", tolerance=" << prescribed_reeq_grad_tol
                        << ", energy=" << energy << "\n";
                }
                forceStateCurvature = false;
            };

        for(const auto& toolpath : sequence.toolpaths)
        {
            ++toolpath_sequence_index;
            const auto& op = toolpath.operation;

            zigzag::Params zz;
            zz.Lv_mm = op.lv_mm;
            zz.alpha_deg = op.alpha_deg;
            zz.N_total = op.n_strips;
            zz.w_mm = op.width_mm;
            zz.offset_dx_mm = 0.0;
            zz.offset_dy_mm = 0.0;
            zz.rotation_deg = 0.0;
            zz.last_wins = false;
            zz.start_mode = sequence.defaults.start_mode;
            zz.zero_outside = true;
            zz.top_profile_mode = op.profile.mode;
            zz.top_end_ratio = op.profile.top_end_ratio;
            zz.top_profile_power = op.profile.top_profile_power;

            zigzag::PatternPlacement placement;
            placement.use_material_bbox_center =
                op.use_material_bbox_center;
            placement.center_uv_m = op.center_uv_m;
            placement.shift_uv_m = op.shift_uv_m;
            placement.shift_frame = op.shift_frame;
            placement.rotation_deg = op.rotation_deg;
            placement.require_inside_material_bounds = true;

            std::vector<zigzag::MaterialHit> materialHits;
            Eigen::VectorXi hitsThisCycle(nFaces);
            hitsThisCycle.setZero();
            zigzag::collectMaterialCoordinateHits(
                materialCoordinates,
                Connect,
                zz,
                placement,
                op.gtop,
                op.gbot,
                op.ortho,
                materialHits,
                &hitsThisCycle);
            if(!op.active_strips.empty())
            {
                materialHits.erase(
                    std::remove_if(
                        materialHits.begin(),
                        materialHits.end(),
                        [&](const zigzag::MaterialHit& hit)
                        {
                            return !std::binary_search(
                                op.active_strips.begin(),
                                op.active_strips.end(),
                                hit.strip_idx);
                        }),
                    materialHits.end());
                hitsThisCycle.setZero();
                for(const auto& hit : materialHits)
                    ++hitsThisCycle(hit.face_idx);
            }

            filterHitsByMargins(materialHits, hitsThisCycle);
            if(materialHits.empty())
                throw std::runtime_error(
                    "zigzag_sequence: toolpath '" + toolpath.id +
                    "' covers no faces after placement and margins.");

            std::vector<std::vector<int>> hitsByFace(nFaces);
            for(int hit_index = 0;
                hit_index < (int)materialHits.size();
                ++hit_index)
                hitsByFace[materialHits[hit_index].face_idx]
                    .push_back(hit_index);

            int maxHitsPerFace = 0;
            for(const auto& faceHits : hitsByFace)
                maxHitsPerFace = std::max(
                    maxHitsPerFace,
                    static_cast<int>(faceHits.size()));

            const std::string safeId =
                sanitizeIdentifier(toolpath.id);

            for(int repeat_index = 1;
                repeat_index <= toolpath.repeat;
                ++repeat_index)
            {
                if(!sequence_warm_start)
                    mesh.resetToRestState();
                // Reset before mapping so cold-start diagnostics describe the
                // geometry actually passed to the nonlinear minimizer.
                ++executed_cycle;
                // Prescribed geometry lands before this cycle's mapping and
                // load, so the toolpath is projected onto the imposed shape
                // and the imposed state is the initial guess for the solve.
                if(prescribed_enabled &&
                   executed_cycle == prescribed_at_cycle + 1)
                    imposePrescribedDeformation(prescribed_at_cycle);
                // BC output index 000 is reserved for the initial geometry.
                // Therefore physical cycle k is written as cycle_k.
                const int output_index = executed_cycle;
                const bool minimize_this_cycle =
                    executed_cycle % minimize_every == 0 ||
                    executed_cycle == total_sequence_cycles;
                const std::string cyclePrefix =
                    use_sequence_bc ?
                        ("cycle_" + helpers::ToString(output_index, 3)) :
                        (tag + "_cycle_" +
                         helpers::ToString(executed_cycle, 3) +
                         "_" + safeId + "_r" +
                         helpers::ToString(repeat_index, 3));
                const std::string mappingBase =
                    use_sequence_bc ?
                        ("mapping_" + helpers::ToString(output_index, 3)) :
                        cyclePrefix + "_mapping";
                const std::string finalBase =
                    (use_sequence_bc ? cyclePrefix : cyclePrefix + "_final") +
                    (minimize_this_cycle ? "" : "_pending");

                // Target metrics and hardening always advance in original hit order.
                // Skipping equilibria is an approximation to the solution branch only.
                const bool solveThisCycle = sequence_solve_mode == "every_cycle" ||
                                            executed_cycle == requested_cycles;
                const Real maxTangencyError = solveThisCycle ?
                    writeSequenceMapping(
                        executed_cycle,
                        mappingBase,
                        materialHits,
                        hitsThisCycle,
                        hitsByFace,
                        maxHitsPerFace,
                        op) : 0.0;
                const tVecMat2d pathStartAformsTop = aformsTop;
                const tVecMat2d pathStartAformsBot = aformsBot;
                const Eigen::VectorXi pathStartPassCount = totalPassCount;
                const int pathDofCount =
                    3 * mesh.getNumberOfVertices() + mesh.getNumberOfEdges();
                Eigen::Map<Eigen::VectorXd> pathState(
                    mesh.getDataPointer(), pathDofCount);
                const Eigen::VectorXd pathStartState = pathState;

                Eigen::VectorXd incG1Top =
                    Eigen::VectorXd::Zero(nFaces);
                Eigen::VectorXd incG2Top =
                    Eigen::VectorXd::Zero(nFaces);
                Eigen::VectorXd incG1Bot =
                    Eigen::VectorXd::Zero(nFaces);
                Eigen::VectorXd incG2Bot =
                    Eigen::VectorXd::Zero(nFaces);
                Eigen::VectorXd qFirst =
                    Eigen::VectorXd::Zero(nFaces);
                Eigen::VectorXd qLast =
                    Eigen::VectorXd::Zero(nFaces);
                Eigen::VectorXi processedThisCycle(nFaces);
                processedThisCycle.setZero();

                Real qMin = std::numeric_limits<Real>::max();
                Real qMax = -std::numeric_limits<Real>::max();
                Real qSum = 0.0;
                int qCount = 0;

                for(const auto& hit : materialHits)
                {
                    const int face = hit.face_idx;
                    const Real q =
                        zigzag_sequence::hardeningFactor(
                            sequence.hardening,
                            totalPassCount(face));

                    if(processedThisCycle(face) == 0)
                        qFirst(face) = q;
                    qLast(face) = q;
                    processedThisCycle(face) += 1;

                    qMin = std::min(qMin, q);
                    qMax = std::max(qMax, q);
                    qSum += q;
                    ++qCount;

                    const Real g1Top =
                        q * hit.gtop * (1.0 + op.ortho_top.at(hit.strip_idx));
                    const Real g2Top =
                        q * hit.gtop * (1.0 - op.ortho_top.at(hit.strip_idx));
                    const Real g1Bot =
                        q * hit.gbot * (1.0 + op.ortho_bottom.at(hit.strip_idx));
                    const Real g2Bot =
                        q * hit.gbot * (1.0 - op.ortho_bottom.at(hit.strip_idx));

                    GrowthHelper<tMesh>::
                        updateAbarWithMaterialGrowthIncrement(
                            materialCoordinates,
                            Connect,
                            face,
                            hit.angle_rad,
                            g1Top,
                            g2Top,
                            aformsTop[face],
                            metric_update);
                    GrowthHelper<tMesh>::
                        updateAbarWithMaterialGrowthIncrement(
                            materialCoordinates,
                            Connect,
                            face,
                            hit.angle_rad,
                            g1Bot,
                            g2Bot,
                            aformsBot[face],
                            metric_update);

                    incG1Top(face) += g1Top;
                    incG2Top(face) += g2Top;
                    incG1Bot(face) += g1Bot;
                    incG2Bot(face) += g2Bot;

                    if(zigzag_sequence::countsTowardHistory(
                           sequence.hardening, hit))
                        totalPassCount(face) += 1;
                }

                Real maxBformDrift = 0.0;
                const tVecMat2d& currentBforms =
                    mesh.getRestConfiguration()
                        .getSecondFundamentalForms();
                for(int face = 0; face < nFaces; ++face)
                    maxBformDrift = std::max(
                        maxBformDrift,
                        (currentBforms[face] -
                         initialBforms[face]).norm());

                if(maxBformDrift > 1e-13)
                    throw std::runtime_error(
                        "zigzag_sequence: b_r changed unexpectedly; "
                        "metric-only treatment requires fixed reference curvature.");

                const tVecMat2d pathEndAformsTop = aformsTop;
                const tVecMat2d pathEndAformsBot = aformsBot;
                const Eigen::VectorXi pathEndPassCount = totalPassCount;
                auto setPathLoad = [&](const Real lambda)
                {
                    if(lambda <= 0.0)
                    {
                        aformsTop = pathStartAformsTop;
                        aformsBot = pathStartAformsBot;
                        return;
                    }
                    if(lambda >= 1.0)
                    {
                        aformsTop = pathEndAformsTop;
                        aformsBot = pathEndAformsBot;
                        return;
                    }
                    for(int face = 0; face < nFaces; ++face)
                    {
                        aformsTop[face] =
                            pathStartAformsTop[face] +
                            lambda *
                            (pathEndAformsTop[face] -
                             pathStartAformsTop[face]);
                        aformsBot[face] =
                            pathStartAformsBot[face] +
                            lambda *
                            (pathEndAformsBot[face] -
                             pathStartAformsBot[face]);
                    }
                };

                history << executed_cycle << "," << (solveThisCycle ? 1 : 0) << ","
                        << materialHits.size() << "," << totalPassCount.maxCoeff() << "\n";
                history.flush();
                if(!solveThisCycle) continue;

                int solveCode = -1;
                int solveIterations = 0;
                int solveEvaluations = 0;
                Real solveGradient = std::numeric_limits<Real>::quiet_NaN();
                bool solveConverged = false;
                Real totalEnergy = std::numeric_limits<Real>::quiet_NaN();
                if(minimize_this_cycle)
                {
                    TinyADHessian_Bilayer<tMesh> exactHessian(
                        E, nu, h_total, hessian_threads);
                    Real acceptedLambda = 0.0;
                    Real stepSize = adaptive_continuation ?
                        continuation_initial_step : 1.0;
                    int retries = 0;
                    int attempt = 0;
                    Eigen::VectorXd acceptedState = pathStartState;
                    totalPassCount = pathStartPassCount;
                    setPathLoad(0.0);

                    while(acceptedLambda < 1.0)
                    {
                        ++attempt;
                        const Real trialLambda =
                            std::min<Real>(1.0, acceptedLambda + stepSize);
                        setPathLoad(trialLambda);
                        pathState = acceptedState;
                        mesh.updateDeformedConfiguration();

                        if(equilibrium_solver == "hlbfgs")
                        {
                            Real eps_cycle = eps_init_default;
                            const std::vector<int>* trialDumps =
                                adaptive_continuation ? nullptr :
                                (dump_iters.empty() ? nullptr : &dump_iters);
                            if(use_sequence_bc)
                                minimizeEnergyReduced(
                                    engOps,
                                    eps_cycle,
                                    tol,
                                    stepwise,
                                    trialDumps,
                                    max_iter);
                            else
                                minimizeEnergy(
                                    engOps,
                                    eps_cycle,
                                    tol,
                                    stepwise,
                                    trialDumps,
                                    max_iter);
                            mesh.updateDeformedConfiguration();

                            solveCode = lastMinimization.code;
                            solveIterations = lastMinimization.iterations;
                            solveEvaluations = lastMinimization.evaluations;
                            solveGradient = lastMinimization.gradientNorm;
                            solveConverged = lastMinimization.accepted(
                                equilibrium_gradient_tolerance);
                            totalEnergy = engOps.compute(mesh);
                        }
                        else if(equilibrium_solver == "hybrid")
                        {
                            int lbfgs_iters = 0;
                            int lbfgs_evals = 0;
                            if(hybrid_warmup_iters > 0)
                            {
                                Real eps_cycle = eps_init_default;
                                const std::vector<int>* trialDumps =
                                    adaptive_continuation ? nullptr :
                                    (dump_iters.empty() ? nullptr : &dump_iters);
                                if(use_sequence_bc)
                                    minimizeEnergyReduced(
                                        engOps,
                                        eps_cycle,
                                        hybrid_gate_tol,
                                        false,
                                        trialDumps,
                                        hybrid_warmup_iters);
                                else
                                    minimizeEnergy(
                                        engOps,
                                        eps_cycle,
                                        hybrid_gate_tol,
                                        false,
                                        trialDumps,
                                        hybrid_warmup_iters);
                                mesh.updateDeformedConfiguration();
                                lbfgs_iters = lastMinimization.iterations;
                                lbfgs_evals = lastMinimization.evaluations;
                            }

                            const ShellEquilibrium::TrustRegionNewtonReport
                                report =
                                    ShellEquilibrium::solveShellEquilibrium(
                                        mesh,
                                        engOps,
                                        exactHessian,
                                        trustRegionOptions);
                            solveCode = static_cast<int>(report.status);
                            solveIterations = lbfgs_iters + report.iterations;
                            solveEvaluations =
                                lbfgs_evals + report.energyEvaluations +
                                report.gradientEvaluations;
                            solveGradient = report.gradientNorm;
                            solveConverged = report.accepted;
                            totalEnergy = report.energy;
                        }
                        else
                        {
                            const ShellEquilibrium::TrustRegionNewtonReport
                                report =
                                    ShellEquilibrium::solveShellEquilibrium(
                                        mesh,
                                        engOps,
                                        exactHessian,
                                        trustRegionOptions);
                            solveCode = static_cast<int>(report.status);
                            solveIterations = report.iterations;
                            solveEvaluations =
                                report.energyEvaluations +
                                report.gradientEvaluations;
                            solveGradient = report.gradientNorm;
                            solveConverged = report.accepted;
                            totalEnergy = report.energy;
                        }
                        convergenceSummary
                            << executed_cycle << ",1,physical,"
                            << attempt << ","
                            << acceptedLambda << ","
                            << trialLambda << ","
                            << (trialLambda - acceptedLambda) << ","
                            << retries << ","
                            << equilibrium_solver << ","
                            << solveCode << ","
                            << solveIterations << ","
                            << solveEvaluations << ","
                            << solveGradient << ","
                            << (solveConverged ? 1 : 0) << ","
                            << totalEnergy << "\n";
                        convergenceSummary.flush();

                        if(solveConverged)
                        {
                            acceptedLambda = trialLambda;
                            acceptedState = pathState;
                            retries = 0;
                            if(adaptive_continuation)
                                stepSize = std::min(
                                    continuation_max_step,
                                    stepSize * continuation_growth);
                            continue;
                        }

                        pathState = acceptedState;
                        mesh.updateDeformedConfiguration();
                        setPathLoad(acceptedLambda);
                        ++retries;
                        if(!adaptive_continuation ||
                           retries > continuation_max_retries ||
                           stepSize <=
                               continuation_min_step *
                               (1.0 + 10.0 *
                                std::numeric_limits<Real>::epsilon()))
                            throw std::runtime_error(
                                "zigzag_sequence: equilibrium corrector failed "
                                "at cycle " + std::to_string(executed_cycle) +
                                ", lambda=" + std::to_string(trialLambda) +
                                ", gradient_norm=" +
                                std::to_string(solveGradient) + ".");
                        stepSize = std::max(
                            continuation_min_step, 0.5 * stepSize);
                    }

                    // Preserve the exact constitutive endpoint rather than its
                    // floating-point interpolation at lambda = 1.
                    aformsTop = pathEndAformsTop;
                    aformsBot = pathEndAformsBot;
                    totalPassCount = pathEndPassCount;
                    ++equilibrium_solves;
                    solve_records.push_back({{"cycle", executed_cycle},
                        {"return_code", solveCode}, {"reported_eps", solveGradient},
                        {"return_code_available", !use_sequence_bc}});
                }
                else
                {
                    convergenceSummary
                        << executed_cycle << ",0,pending,0,0,0,0,0,"
                        << equilibrium_solver
                        << ",-1,0,0,nan,0,nan\n";
                    convergenceSummary.flush();
                }
                if(minimize_this_cycle && track_sequence_stability)
                {
                    TinyADHessian_Bilayer<tMesh> stabilityHessian(
                        E, nu, h_total, hessian_threads);
                    const ShellEquilibrium::RigidModeProjector projector =
                        ShellEquilibrium::RigidModeProjector::fromMesh(mesh);
                    const ShellEquilibrium::RitzPairReport stability =
                        ShellEquilibrium::smallestProjectedRitzPairForMesh(
                            mesh,
                            stabilityHessian,
                            projector,
                            Eigen::VectorXd(),
                            stabilityOptions);
                    std::string classification = "indeterminate";
                    if(stability.residualConverged)
                    {
                        if(stability.eigenvalue -
                           stability.residualAbsolute > 0.0)
                            classification = "positive";
                        else if(stability.eigenvalue +
                                stability.residualAbsolute < 0.0)
                            classification = "negative";
                    }
                    stabilitySummary
                        << executed_cycle << ",1,"
                        << stability.eigenvalue << ","
                        << stability.residualAbsolute << ","
                        << stability.residualRelative << ","
                        << (stability.residualConverged ? 1 : 0) << ","
                        << classification << ","
                        << projector.numberOfRigidModes() << ","
                        << stability.operatorEvaluations << "\n";
                }
                else
                    stabilitySummary
                        << executed_cycle
                        << ",0,nan,nan,nan,0,not_evaluated,0,0\n";
                stabilitySummary.flush();

                if(!std::isfinite(totalEnergy))
                    throw std::runtime_error("Sequence energy is not finite.");
                calibration_energy = totalEnergy;
                calibration_final_file = finalBase + ".vtp";
                bcOutputIndex = output_index;
                bcOutputIsRelease = false;
                bcOutputPhysicalClampsActive = use_sequence_bc;
                bcOutputGaugeActive = false;

                writeSequenceState(
                    executed_cycle,
                    finalBase,
                    hitsThisCycle,
                    incG1Top,
                    incG2Top,
                    incG1Bot,
                    incG2Bot,
                    qFirst,
                    qLast);

                if(write_cycle_state)
                    mesh.writeToFile(cyclePrefix + "_state");

                const Eigen::MatrixXd X0 =
                    mesh.getRestConfiguration().getVertices();
                const Eigen::MatrixXd X =
                    mesh.getCurrentConfiguration().getVertices();
                const Eigen::MatrixXd U = X - X0;
                const Eigen::VectorXd U3 = U.col(2);
                const Eigen::VectorXd Umag = U.rowwise().norm();

                const Real maxAbsTopIncrement = std::max(
                    incG1Top.cwiseAbs().maxCoeff(),
                    incG2Top.cwiseAbs().maxCoeff());
                const Real maxAbsBotIncrement = std::max(
                    incG1Bot.cwiseAbs().maxCoeff(),
                    incG2Bot.cwiseAbs().maxCoeff());
                const int coveredFaces =
                    (hitsThisCycle.array() > 0).count();

                if(qCount == 0)
                {
                    qMin = 0.0;
                    qMax = 0.0;
                }
                const Real qMean =
                    qCount > 0 ? qSum / static_cast<Real>(qCount) : 0.0;

                if(use_sequence_bc)
                    summary
                        << output_index << ","
                        << 0 << ","
                        << 1 << ",";

                summary
                    << executed_cycle << ","
                    << csvQuote(toolpath.id) << ","
                    << toolpath_sequence_index << ","
                    << repeat_index << ","
                    << toolpath.repeat << ","
                    << 1 << ","
                    << materialHits.size() << ","
                    << coveredFaces << ","
                    << hitsThisCycle.maxCoeff() << ","
                    << totalPassCount.maxCoeff() << ","
                    << qMin << ","
                    << qMean << ","
                    << qMax << ","
                    << maxAbsTopIncrement << ","
                    << maxAbsBotIncrement << ","
                    << Umag.maxCoeff() << ","
                    << U3.minCoeff() << ","
                    << U3.maxCoeff() << ","
                    << totalEnergy << ","
                    << maxTangencyError << ","
                    << maxBformDrift << ","
                    << csvQuote(mappingBase + ".vtp") << ","
                    << csvQuote(finalBase + ".vtp") << "\n";
                summary.flush();

                if(use_sequence_bc)
                {
                    cycleReadme
                        << "| `" << finalBase << ".vtp` | "
                        << executed_cycle << " | `" << toolpath.id
                        << "` | " << repeat_index
                        << " | constrained; two physical clamps active |\n";
                    cycleReadme.flush();
                }

                std::cout
                    << "[" << growth_type << "] cycle=" << executed_cycle
                    << ", toolpath='" << toolpath.id << "'"
                    << ", repeat=" << repeat_index
                    << "/" << toolpath.repeat
                    << ", events=" << materialHits.size()
                    << ", max_total_hits="
                    << totalPassCount.maxCoeff()
                    << ", energy=" << totalEnergy
                    << "\n";
            }
        }

        if(prescribed_enabled && prescribedAppliedCount != 1)
            throw std::runtime_error(
                "prescribed_deformation: requested boundary after cycle " +
                std::to_string(prescribed_at_cycle) +
                " was imposed " + std::to_string(prescribedAppliedCount) +
                " times instead of exactly once.");


        if(use_sequence_bc &&
           sequence.boundary_conditions.release_after_final_cycle)
        {
            const auto gauge =
                zigzag_sequence_bc::buildMinimalReleaseGaugeMask(
                    materialCoordinates);
            zigzag_sequence_bc::applyVertexMask(mesh, gauge.vertex_mask);

            int releaseCode = -1;
            int releaseIterations = 0;
            int releaseEvaluations = 0;
            Real releaseGradient = std::numeric_limits<Real>::quiet_NaN();
            bool releaseAccepted = false;
            Real releaseEnergy = std::numeric_limits<Real>::quiet_NaN();
            if(equilibrium_solver == "hlbfgs")
            {
                Real eps_release = eps_init_default;
                minimizeEnergyReduced(
                    engOps,
                    eps_release,
                    tol,
                    stepwise,
                    (dump_iters.empty() ? nullptr : &dump_iters),
                    max_iter);
                mesh.updateDeformedConfiguration();
                releaseCode = lastMinimization.code;
                releaseIterations = lastMinimization.iterations;
                releaseEvaluations = lastMinimization.evaluations;
                releaseGradient = lastMinimization.gradientNorm;
                releaseAccepted = lastMinimization.accepted(
                    equilibrium_gradient_tolerance);
                releaseEnergy = engOps.compute(mesh);
            }
            else if(equilibrium_solver == "hybrid")
            {
                int lbfgs_iters = 0;
                int lbfgs_evals = 0;
                if(hybrid_warmup_iters > 0)
                {
                    Real eps_release = eps_init_default;
                    minimizeEnergyReduced(
                        engOps,
                        eps_release,
                        hybrid_gate_tol,
                        false,
                        (dump_iters.empty() ? nullptr : &dump_iters),
                        hybrid_warmup_iters);
                    mesh.updateDeformedConfiguration();
                    lbfgs_iters = lastMinimization.iterations;
                    lbfgs_evals = lastMinimization.evaluations;
                }

                TinyADHessian_Bilayer<tMesh> releaseHessian(
                    E, nu, h_total, hessian_threads);
                const ShellEquilibrium::TrustRegionNewtonReport report =
                    ShellEquilibrium::solveShellEquilibrium(
                        mesh, engOps, releaseHessian, trustRegionOptions);
                releaseCode = static_cast<int>(report.status);
                releaseIterations = lbfgs_iters + report.iterations;
                releaseEvaluations =
                    lbfgs_evals + report.energyEvaluations +
                    report.gradientEvaluations;
                releaseGradient = report.gradientNorm;
                releaseAccepted = report.accepted;
                releaseEnergy = report.energy;
            }
            else
            {
                TinyADHessian_Bilayer<tMesh> releaseHessian(
                    E, nu, h_total, hessian_threads);
                const ShellEquilibrium::TrustRegionNewtonReport report =
                    ShellEquilibrium::solveShellEquilibrium(
                        mesh, engOps, releaseHessian, trustRegionOptions);
                releaseCode = static_cast<int>(report.status);
                releaseIterations = report.iterations;
                releaseEvaluations =
                    report.energyEvaluations + report.gradientEvaluations;
                releaseGradient = report.gradientNorm;
                releaseAccepted = report.accepted;
                releaseEnergy = report.energy;
            }
            convergenceSummary
                << executed_cycle << ",1,release,1,1,1,0,0,"
                << equilibrium_solver << ","
                << releaseCode << ","
                << releaseIterations << ","
                << releaseEvaluations << ","
                << releaseGradient << ","
                << (releaseAccepted ? 1 : 0) << ","
                << releaseEnergy << "\n";
            convergenceSummary.flush();
            if(!releaseAccepted)
                throw std::runtime_error(
                    "zigzag_sequence: final release equilibrium failed, "
                    "gradient_norm=" + std::to_string(releaseGradient) + ".");
            // The release follows the last constrained physical cycle.
            const int release_output_index = executed_cycle + 1;
            const std::string releaseBase =
                "cycle_" + helpers::ToString(release_output_index, 3);

            bcOutputIndex = release_output_index;
            bcOutputIsRelease = true;
            bcOutputPhysicalClampsActive = false;
            bcOutputGaugeActive = true;
            writeSequenceState(
                executed_cycle,
                releaseBase,
                zeroHits,
                zeroField,
                zeroField,
                zeroField,
                zeroField,
                zeroField,
                zeroField);

            const Eigen::MatrixXd X0 =
                mesh.getRestConfiguration().getVertices();
            const Eigen::MatrixXd X =
                mesh.getCurrentConfiguration().getVertices();
            const Eigen::MatrixXd U = X - X0;
            const Eigen::VectorXd U3 = U.col(2);
            const Eigen::VectorXd Umag = U.rowwise().norm();

            summary
                << release_output_index << ","
                << 1 << ","
                << 0 << ","
                << executed_cycle << ","
                << csvQuote("FINAL_RELEASE") << ","
                << 0 << ","
                << 0 << ","
                << 0 << ","
                << 0 << ","
                << 0 << ","
                << 0 << ","
                << 0 << ","
                << totalPassCount.maxCoeff() << ","
                << 0.0 << ","
                << 0.0 << ","
                << 0.0 << ","
                << 0.0 << ","
                << 0.0 << ","
                << Umag.maxCoeff() << ","
                << U3.minCoeff() << ","
                << U3.maxCoeff() << ","
                << releaseEnergy << ","
                << 0.0 << ","
                << 0.0 << ","
                << csvQuote("") << ","
                << csvQuote(releaseBase + ".vtp") << "\n";
            summary.flush();

            cycleReadme
                << "| `" << releaseBase << ".vtp` | "
                << executed_cycle
                << " | - | - | released; physical clamps removed; "
                << "six-DOF numerical gauge active |\n\n"
                << "## Final-release numerical gauge\n\n"
                << "- Vertex A (x,y,z fixed): " << gauge.vertex_ids[0] << "\n"
                << "- Vertex B (y,z fixed): " << gauge.vertex_ids[1] << "\n"
                << "- Vertex C (z fixed): " << gauge.vertex_ids[2] << "\n";
            cycleReadme.flush();

            std::cout
                << "[zigzag_sequence_BC] final release output="
                << releaseBase << ".vtp, energy=" << releaseEnergy
                << ", gauge_vertices=[" << gauge.vertex_ids[0] << ","
                << gauge.vertex_ids[1] << "," << gauge.vertex_ids[2]
                << "]\n";
        }

        const bool export_stl =
            parser.parse<bool>("-export_stl", false);
        const bool stl_ascii =
            parser.parse<bool>("-stl_ascii", false);
        if(export_stl)
            WriteSTL::write(
                mesh.getTopology(),
                mesh.getCurrentConfiguration(),
                tag + "_final_deformed",
                stl_ascii);

        std::cout
            << "[" << growth_type << "] completed " << executed_cycle
            << " physical cycles. Summary: "
            << summary_filename << "\n";

        // ---- second-order certification of the FINAL state (exact TinyAD Hessian) --------
        // Answers: is the final springback state a genuine (local) energy minimum, or a
        // saddle / soft shoulder the |g|-based stop accepted prematurely? Assemble the exact
        // sparse Hessian once, then power-iterate for lam_max and (via the spectral shift
        // c*I - H) for lam_min. A free panel has 6 rigid zero modes, so lam_min ~ 0 means
        // "minimum up to rigid modes"; lam_min clearly negative means saddle/false stop.
        const bool certify_final = parser.parse<bool>("-certify_final", false);
        const bool seed_escape   = parser.parse<bool>("-seed_escape", false);
        if(certify_final || seed_escape)
        {
            TinyADHessian_Bilayer<tMesh> tad(E, nu, h_total, hessian_threads);
            const int nVv = mesh.getNumberOfVertices();
            const int nDv = 3 * nVv + mesh.getNumberOfEdges();
            std::mt19937 rng(3);
            std::uniform_real_distribution<double> Ud(-1.0, 1.0);

            // certify the CURRENT state: returns lam_min, fills lam_max and the buckling
            // eigenvector (eigenvector of lam_min) via spectral-shift power iteration.
            const int piters = parser.parse<int>("-certify_iters", 1200);
            auto certify = [&](Real & lam_max_out, Eigen::VectorXd & evec) -> Real
            {
                mesh.updateDeformedConfiguration();
                const Eigen::SparseMatrix<double> H = tad.assembleHessian(mesh);
                Eigen::VectorXd v(nDv);
                for(int i = 0; i < nDv; ++i) v(i) = Ud(rng);
                v.normalize();
                Real lm = 0;
                for(int it = 0; it < 150; ++it) { v = H * v; lm = v.norm(); v /= lm; }
                lam_max_out = lm;
                const Real c = 1.05 * lm;
                Eigen::VectorXd w(nDv);
                for(int i = 0; i < nDv; ++i) w(i) = Ud(rng);
                w.normalize();
                Real mu = 0;
                for(int it = 0; it < piters; ++it)
                { Eigen::VectorXd Hw = H * w; w = c * w - Hw; mu = w.norm(); w /= mu; }
                evec = w;
                return c - mu;
            };

            // The saddle-escape loop (Phase-1 branch seeding): while the state has negative
            // curvature, kick along +/- the buckling eigenvector (a few amplitudes, scaled to
            // unit physical max|z|), re-minimize, keep the lowest-energy candidate, repeat.
            // Deterministic escape off the saddle cascade instead of tolerance-grinding.
            const Real seed_amp  = parser.parse<Real>("-seed_amp", 10.0 * h_total);
            const int  seed_max  = parser.parse<int>("-seed_max", 12);
            const Real tolEsc    = parser.parse<Real>("-tol", 1e-12);
            int esc = 0;
            while(true)
            {
                mesh.updateDeformedConfiguration();
                Eigen::VectorXd gfin = Eigen::VectorXd::Zero(nDv);
                const Real Ecur = engOps.compute(mesh, gfin);
                Real lam_max = 0;
                Eigen::VectorXd wmode(nDv);
                const Real lam_min = certify(lam_max, wmode);
                const bool is_min = (lam_min >= -1e-6 * lam_max);
                printf("[certify_final] esc=%d  E=%.6e  |g|=%.3e  lam_max=%.3e  lam_min=%+.3e  -> %s\n",
                       esc, Ecur, gfin.norm(), lam_max, lam_min,
                       is_min ? "MINIMUM (up to rigid modes)" : "saddle");
                fflush(stdout);
                if(is_min || !seed_escape || esc >= seed_max) break;

                // normalize the mode to unit physical out-of-plane amplitude
                const Real zmax = wmode.segment(2 * nVv, nVv).cwiseAbs().maxCoeff();
                if(zmax > 1e-12) wmode /= zmax; else wmode /= wmode.norm();

                const Eigen::VectorXd x0 =
                    Eigen::Map<const Eigen::VectorXd>(mesh.getDataPointer(), nDv);
                Real bestE = Ecur;
                Eigen::VectorXd bestX = x0;
                const Real facs[3] = {1.0, 0.25, 2.5};
                for(int fi = 0; fi < 3; ++fi)
                for(int sgn = -1; sgn <= 1; sgn += 2)
                {
                    Eigen::Map<Eigen::VectorXd>(mesh.getDataPointer(), nDv) =
                        x0 + (sgn * facs[fi] * seed_amp) * wmode;
                    Real eps_esc = 1e-2;
                    minimizeEnergy(engOps, eps_esc, tolEsc);
                    const Real Etry = engOps.compute(mesh);
                    if(std::isfinite(Etry) && Etry < bestE)
                    {
                        bestE = Etry;
                        bestX = Eigen::Map<const Eigen::VectorXd>(mesh.getDataPointer(), nDv);
                    }
                }
                Eigen::Map<Eigen::VectorXd>(mesh.getDataPointer(), nDv) = bestX;
                mesh.updateDeformedConfiguration();
                if(bestE >= Ecur - 1e-18)
                {
                    printf("[seed_escape] no seeded candidate lowered the energy -- stopping\n");
                    break;
                }
                const Real z_now = [&]{
                    const auto vv = mesh.getCurrentConfiguration().getVertices();
                    return vv.col(2).cwiseAbs().maxCoeff(); }();
                printf("[seed_escape] escape %d accepted: E %.6e -> %.6e   max|z|=%.4e\n",
                       esc + 1, Ecur, bestE, z_now);
                fflush(stdout);
                ++esc;
            }
        }

        if(!use_sequence_bc)
        {
            std::string backend = equilibrium_solver;
            if(backend.empty())
            {
#if defined(USELIBLBFGS)
                backend = "liblbfgs";
#elif defined(USEHLBFGS)
                backend = "hlbfgs";
#endif
            }
            zigzag_sequence::json result = {
                {"schema_version", 1}, {"completed", true},
                {"solve_mode", sequence_solve_mode}, {"cycles", executed_cycle},
                {"equilibrium_solves", equilibrium_solves},
                {"initial_mesh", tag + "_cycle_000_initial.vtp"},
                {"final_mesh", calibration_final_file}, {"total_energy", calibration_energy},
                {"backend", backend}, {"stepwise", stepwise}, {"solves", solve_records}};
            std::ofstream resultFile("sequence_result.json");
            if(!resultFile) throw std::runtime_error("Cannot write sequence_result.json");
            resultFile << result.dump(2) << "\n";
            resultFile.close();
            if(!resultFile) throw std::runtime_error("Failed writing sequence_result.json");
        }
        }

        // Preserve target-form history and the original b_r; do not enter the
        // legacy final mesh.init_rest(...) block below.
        return;
    }

    if (growth_type == "homo"){
      growthRates_b = Eigen::VectorXd::Constant(nFaces, growthRate_b);
      growthRates_t = Eigen::VectorXd::Constant(nFaces, growthRate_t);
    }
    else if (growth_type == "chess"){
      Real CenterX = 0.0;
      Real CenterY = 0.0;

      for (int i=0; i<nVert; ++i){
        if ((std::abs(Vertices(i,0))<10e-9 && std::abs(Vertices(i,1))<10e-9)){
            CenterX = Vertices(i,0);
            CenterY = Vertices(i,1);
        }
      }

      //first quarter

      for (int i=0; i<nVert; ++i){
        if (Vertices(i,0) > CenterX && Vertices(i,1) > CenterY){
  	       IndicV(i) = 1;
        }
      }

      for (int i=0; i<nFaces; ++i){
  	     if (IndicV(Connect(i,0))==1 || IndicV(Connect(i,1))==1 || IndicV(Connect(i,2))==1) {
           growthRates_b(i) = growthRate_b;
           growthRates_t(i) = growthRate_t;
  	     }
  	     else{
           // growthRates_b(i) = 0.0;
           // growthRates_t(i) = 0.0;
           growthRates_b(i) = growthRate_t;
           growthRates_t(i) = growthRate_b;
  	     }
      }

      //second quarter

      IndicV.setZero();

      for (int i=0; i<nVert; ++i){
        //IndicV(i) = 0;
        if (Vertices(i,0) < CenterX && Vertices(i,1) < CenterY){
  	       IndicV(i) = 1;
        }
      }

      for (int i=0; i<nFaces; ++i){
  	     if (IndicV(Connect(i,0))==1 || IndicV(Connect(i,1))==1 || IndicV(Connect(i,2))==1) {
           growthRates_b(i) = growthRate_b;
           growthRates_t(i) = growthRate_t;
  	     }
      }
    }

    else if (growth_type == "center"){
      const Real Lx = parser.parse<Real>("-lx", 0.5);
      const Real Ly = parser.parse<Real>("-ly", 0.5);

      const Real size = parser.parse<Real>("-size", 0.4);
      const Real size_hole = parser.parse<Real>("-size_hole", 0.0);

      for (int i=0; i<nVert; ++i){
        if (abs(Vertices(i,0))<=Lx*size && abs(Vertices(i,1))<=Ly*size){
          IndicV(i) = 1;
        } else{
          IndicV(i) = 0;
        }
        if (abs(Vertices(i,0))<=Lx*size_hole && abs(Vertices(i,1))<=Ly*size_hole){
          IndicV(i) = 0;
        }
      }

      for (int i=0; i<nFaces; ++i){
        if (IndicV(Connect(i,0))==1 && IndicV(Connect(i,1))==1 && IndicV(Connect(i,2))==1) {
          growthRates_b(i) = growthRate_b;
          growthRates_t(i) = growthRate_t;
          //growthRates_t(i) = 0.0;
          //growthRates_b(i) = 0.0;
        }
        else{
          growthRates_t(i) = 0.0;
          growthRates_b(i) = 0.0;
          //growthRates_b(i) = growthRate_b;
          //growthRates_t(i) = growthRate_t;
        }
      }
    }

    else if (growth_type == "patch") {
      const Real Lx = parser.parse<Real>("-lx", 0.5);
      [[maybe_unused]] const Real Ly = parser.parse<Real>("-ly", 0.5);

      const Real patch_lx = parser.parse<Real>("-patch_lx", 0.120);
      const Real patch_ly = parser.parse<Real>("-patch_ly", 0.010);

      // Patch placement: distance from the LEFT end (x = -Lx) to the patch's left edge
      const Real patch_x_from_left = parser.parse<Real>("-patch_x_from_left", 0.120);

      // Optional: centerline offset in y (default 0 means centered)
      const Real patch_yc = parser.parse<Real>("-patch_yc", 0.0);

      // Compute patch bounds in centered coordinates
      const Real x_min = (-Lx) + patch_x_from_left;
      const Real x_max = x_min + patch_lx;

      const Real y_min = patch_yc - 0.5*patch_ly;
      const Real y_max = patch_yc + 0.5*patch_ly;

      // 1) Vertex indicator: inside patch
      for (int i = 0; i < nVert; ++i) {
        const Real x = Vertices(i, 0);
        const Real y = Vertices(i, 1);

        if (x >= x_min && x <= x_max && y >= y_min && y <= y_max) {
          IndicV(i) = 1;
        } else {
          IndicV(i) = 0;
        }
      }

      // 2) Face mask: apply growth only if entire triangle is inside
      for (int i = 0; i < nFaces; ++i) {
        const bool inPatch =
          (IndicV(Connect(i, 0)) == 1 &&
          IndicV(Connect(i, 1)) == 1 &&
          IndicV(Connect(i, 2)) == 1);

        if (inPatch) {
          growthRates_b(i) = growthRate_b;
          growthRates_t(i) = growthRate_t;
        } else {
          growthRates_b(i) = 0.0;
          growthRates_t(i) = 0.0;
        }
      }
    }

    else if (growth_type == "wave"){

      const bool sym_x = parser.parse<bool>("-sym_x", false);
      const bool inv_wave = parser.parse<bool>("-inv_wave", false);
      const Real overlap = parser.parse<Real>("-overlap", 0.0);

      Real CenterX=0.0;
      Real CenterY=0.0;
      for (int i=0; i<nVert; ++i){
          CenterX += Vertices(i,0);
          CenterY += Vertices(i,1);
      }
      CenterX /= nVert;
      CenterY /= nVert;

      if (sym_x){
        for (int i=0; i<nVert; ++i){
          if (Vertices(i,1)<= CenterY + 1e-9){
            IndicV(i) = 1;
          } else{
            IndicV(i) = 0;
          }

          if ((Vertices(i,1)<= CenterY + overlap) && (Vertices(i,1)>= CenterY - overlap)){
            IndicV(i) = 2;
          }

        }
      } else {
        for (int i=0; i<nVert; ++i){
          if (Vertices(i,0)<= CenterX + 1e-9){
            IndicV(i) = 1;
          } else{
            IndicV(i) = 0;
          }

          if ((Vertices(i,0)<= CenterX + overlap) && (Vertices(i,0)>= CenterX - overlap)){
            IndicV(i) = 2;
          }
        }
      }

      for (int i=0; i<nFaces; ++i){
        if (inv_wave){
          if (IndicV(Connect(i,0))==1 && IndicV(Connect(i,1))==1 && IndicV(Connect(i,2))==1) {
            growthRates_b(i) = growthRate_b;
            growthRates_t(i) = growthRate_t;
          }
          else{
            growthRates_b(i) = growthRate_t;
            growthRates_t(i) = growthRate_b;
          }
        } else{
          if (IndicV(Connect(i,0))==1 && IndicV(Connect(i,1))==1 && IndicV(Connect(i,2))==1) {
            growthRates_b(i) = growthRate_t;
            growthRates_t(i) = growthRate_b;
          }
          else{
            growthRates_b(i) = growthRate_b;
            growthRates_t(i) = growthRate_t;
          }
        }

        if (IndicV(Connect(i,0))==2 || IndicV(Connect(i,1))==2 || IndicV(Connect(i,2))==2) {
            growthRates_b(i) = growthRate_b + growthRate_t;
            growthRates_t(i) = growthRate_t + growthRate_b;
        }
      }
    }

    else if (growth_type == "circle"){
      const Real radius = parser.parse<Real>("-radius", 0.2);

      for (int i=0; i<nVert; ++i){
        if (std::pow(Vertices(i,0),2) + std::pow(Vertices(i,1),2) <= std::pow(radius,2)){
  	       IndicV(i) = 1;
        }
      }

      for (int i=0; i<nFaces; ++i){
  	     if (IndicV(Connect(i,0))==1 && IndicV(Connect(i,1))==1 && IndicV(Connect(i,2))==1) {
           growthRates_b(i) = growthRate_b;
           growthRates_t(i) = growthRate_t;
  	     }
  	     else{
           growthRates_b(i) = 0.0;
           growthRates_t(i) = 0.0;
  	     }
      }
    }

    else if (growth_type == "external"){

      const bool samemesh = parser.parse<bool>("-samemesh", false); 
      // whether the external pattern is generated on the same mesh that we use or not
      // if true, then we just activate the elements specified in the text files
      // if false, then we project the pattern specified on another mesh onto our initial mesh

      if (samemesh){
        const std::string growth_tag = parser.template parse<std::string>("-growth_tag", "");
        const Real correction_coeff = parser.parse<Real>("-correction_coeff", 1.0);
        const int size1 = parser.parse<int>("-size1", nFaces);
        const int size2 = parser.parse<int>("-size2", 1);

        // two .txt files that specify "active" elements on the top and bottom layers
        // each file contains a vector of 0 or 1, where 1 corresponds to the active elements
        Eigen::VectorXi active_b(nFaces);
        Eigen::VectorXi active_t(nFaces);
        helpers::read_matrix(growth_tag + "_peening_bot.txt", active_b, size1, size2);
        helpers::read_matrix(growth_tag + "_peening_top.txt", active_t, size1, size2);

        // std::cout<<active_b<<std::endl;

        for (int i=0; i<nFaces; i++){
          if (active_b(i) == 1){
            growthRates_b(i) = growthRate_t;
            growthRates_t(i) = growthRate_b;
          }
          if (active_t(i) == 1){
            growthRates_t(i) += growthRate_t;
            growthRates_b(i) += growthRate_b;
          }
        }

        Real CenterX = 0.0;
        Real CenterY = 0.0;
        Real CenterZ = 0.0;

        for (int i=0; i<nVert; ++i){
          CenterX += Vertices(i,0);
          CenterY += Vertices(i,1);
          CenterZ += Vertices(i,2);
        } 

        CenterX /= nVert;
        CenterY /= nVert;
        CenterZ /= nVert;

        for (int i=0; i<nVert; i++){
          Vertices(i,0) -= CenterX;
          Vertices(i,1) -= CenterY;
          Vertices(i,2) -= CenterZ;

          Vertices(i,0) *= correction_coeff;
          Vertices(i,1) *= correction_coeff;
        }

        mesh.getCurrentConfiguration().getVertices() = Vertices;
        mesh.getRestConfiguration().getVertices() = Vertices;
      }
      else {
        const std::string growth_tag = parser.template parse<std::string>("-growth_tag", "");

        // specify the number of faces and vertices in the mesh containing the pattern
        const int nFaces_Reg = parser.template parse<int>("-nFaces_Reg", 2178);
        const int nVert_Reg = parser.template parse<int>("-nVert_Reg", 1156);

        Eigen::MatrixXi Reg_Vertices(nVert_Reg,3);
        Eigen::MatrixXi Reg_Faces(nFaces_Reg,3);
        Eigen::VectorXi Peening_bot(nFaces_Reg);
        Eigen::VectorXi Peening_top(nFaces_Reg);
        Eigen::VectorXi Reg_Clusters(nFaces_Reg);
        Reg_Clusters.setZero();

        // two .txt files that specify "active" elements on the top and bottom layers,
        // each file contains a vector of 0 or 1, where 1 corresponds to the active elements
        helpers::read_matrix(growth_tag + "_peening_bot.txt", Peening_bot, nFaces_Reg, 1);
        helpers::read_matrix(growth_tag + "_peening_top.txt", Peening_top, nFaces_Reg, 1);
        // two .txt files that define the mesh containing the pattern
        helpers::read_matrix(growth_tag + "_vertices.txt", Reg_Vertices, nVert_Reg, 3);
        helpers::read_matrix(growth_tag + "_faces.txt", Reg_Faces, nFaces_Reg, 3);

        //scale the pattern before projection
        const Real mesh_ratio_x = (Vertices.col(0).maxCoeff() - Vertices.col(0).minCoeff()) / (Reg_Vertices.col(0).maxCoeff() - Reg_Vertices.col(0).minCoeff());
        const Real mesh_ratio_y = (Vertices.col(1).maxCoeff() - Vertices.col(1).minCoeff()) / (Reg_Vertices.col(1).maxCoeff() - Reg_Vertices.col(1).minCoeff());
        for (int i=0; i<nVert_Reg; i++){
          Reg_Vertices(i,0) *= mesh_ratio_x;
          Reg_Vertices(i,1) *= mesh_ratio_y;
        }

        const Real CenterX = Reg_Vertices.col(0).mean();
        const Real CenterY = Reg_Vertices.col(1).mean();
        for (int i=0; i<nVert_Reg; i++){
          Reg_Vertices(i,0) -= CenterX;
          Reg_Vertices(i,1) -= CenterY;
          Reg_Vertices(i,2) = 0;
        }

        for (int i=0; i<nFaces_Reg; i++){
          if(Peening_bot(i)==1){
            Reg_Clusters(i) = 2;
          }
          if(Peening_top(i)==1){
            Reg_Clusters(i) = 1;
          }
        }

        // we draw a circle around each vertex and look if the vertices in the regular pattern
        // that lie inside this circle are all "activated". This is a simplified formulation that works bad if the pattern is complex

        // So the sizes and positions of the two plates must perfectly coincide!

        //First we assign the IndicV to the initial pattern
        Eigen::VectorXi IndicV_Reg(nVert_Reg);
        IndicV_Reg.setZero();
        for (int i=0; i<nFaces_Reg; i++){
          if(Reg_Clusters(i)==1 || Reg_Clusters(i)==2){
            for (int j=0; j<3; j++){
              IndicV_Reg(Reg_Faces(i,j)) = Reg_Clusters(i);
            }
          }
        }

        //we divide the -lx by number of elms along this side and multiply by sqrt(2)/2 (if it falls exactly in center of square)
        const Real search_rad = parser.template parse<Real>("-search_rad", ((Reg_Vertices.col(0).maxCoeff() - Reg_Vertices.col(0).minCoeff())/std::sqrt(nFaces_Reg*0.5)) * std::sqrt(2.0)*0.5 * 1.1);
        for (int i=0; i<nVert; i++){
          Eigen::VectorXi Inside_Circle(nVert_Reg);
          Inside_Circle.setZero();
          int olol=0;
          for (int j=0; j<nVert_Reg; j++){
            if (std::pow(Reg_Vertices(j,0) - Vertices(i,0),2) + std::pow(Reg_Vertices(j,1) - Vertices(i,1),2) < std::pow(search_rad*1.01,2)){
              Inside_Circle(olol) = IndicV_Reg(j);
              olol ++;
            }
          }

          int not_clust_1 = 0;
          int not_clust_2 = 0;
          for (int j=0; j<olol; j++){
            if(Inside_Circle(j) != 1){
              not_clust_1 = 1;
            }
            if(Inside_Circle(j) != 2){
              not_clust_2 = 1;
            }
          }

          if (not_clust_1 == 0){
            IndicV(i) = 1;
          } else if (not_clust_2 == 0){
            IndicV(i) = 2;
          }

        }

        for (int i=0; i<nFaces; ++i){
          if (IndicV(Connect(i,0))==1 || IndicV(Connect(i,1))==1 || IndicV(Connect(i,2))==1) {
            growthRates_b(i) = growthRate_b;
            growthRates_t(i) = growthRate_t;
          }
          else if (IndicV(Connect(i,0))==2 || IndicV(Connect(i,1))==2 || IndicV(Connect(i,2))==2) {
            growthRates_b(i) = growthRate_t;
            growthRates_t(i) = growthRate_b;
          }
        }
      }
    }

    // ver-0203 ADDED zigzag pattern
    else if (growth_type == "zigzag")
    {
      std::cout << "[zigzag] ENTER (TestCustomGrowth) line=" << __LINE__ << "\n"; // For debug

      // ---- CLI (mm units) ----
      const Real Lv_mm     = parser.parse<Real>("-zigzag_lv_mm", 40.0);
      const Real alpha_deg = parser.parse<Real>("-zigzag_alpha_deg", 15.0); // relative to vertical
      const int  N_total   = parser.parse<int> ("-zigzag_N", 6);
      const Real w_mm      = parser.parse<Real>("-zigzag_w_mm", 2.0);

      const Real offset_dx_mm = parser.parse<Real>("-zigzag_offset_dx_mm", 0.0);
      const Real offset_dy_mm = parser.parse<Real>("-zigzag_offset_dy_mm", 0.0);

      const Real rotation_deg = parser.parse<Real>("-zigzag_rotation_deg", 0.0);

      // per-strip grouped parameters (length = N_total)
      const std::string gtop_s  = parser.parse<std::string>("-zigzag_gtop_list", "");
      const std::string gbot_s  = parser.parse<std::string>("-zigzag_gbot_list", "");
      const std::string ortho_s = parser.parse<std::string>("-zigzag_ortho_list", "");

      // Theta related modifications
      const std::string zigzag_profile_mode_str =
          parser.parse<std::string>("-zigzag_profile_mode", "uniform");
      const double zigzag_top_end_ratio =
          parser.parse<Real>("-zigzag_top_end_ratio", 1.0);
      const double zigzag_top_profile_power =
          parser.parse<Real>("-zigzag_top_profile_power", 1.0);

      // defaults fall back to global values if lists not provided
      const Real gtop_default  = growthRate_t;
      const Real gbot_default  = growthRate_b;
      const Real ortho_default = ortho_coeff;

      std::vector<double> gtop_list(N_total, gtop_default);
      std::vector<double> gbot_list(N_total, gbot_default);
      std::vector<double> ortho_list(N_total, ortho_default);

      if(!gtop_s.empty())  gtop_list  = zigzag::parseCommaListReal(gtop_s,  N_total, gtop_default);
      if(!gbot_s.empty())  gbot_list  = zigzag::parseCommaListReal(gbot_s,  N_total, gbot_default);
      if(!ortho_s.empty()) ortho_list = zigzag::parseCommaListReal(ortho_s, N_total, ortho_default);

      zigzag::Params zz;
      zz.Lv_mm      = Lv_mm;
      zz.alpha_deg  = alpha_deg;
      zz.N_total    = N_total;
      zz.w_mm       = w_mm;
      zz.last_wins  = true;
      zz.start_mode = zigzag::StartMode::LeftBottom_Up; // your preference

      zz.offset_dx_mm = offset_dx_mm;
      zz.offset_dy_mm = offset_dy_mm;

      zz.rotation_deg = rotation_deg;

      //
      zz.top_profile_mode = zigzag::parseTopProfileMode(zigzag_profile_mode_str);
      zz.top_end_ratio = zigzag_top_end_ratio;
      zz.top_profile_power = zigzag_top_profile_power;

      // debug: print parsed parameters
      std::cout << "[zigzag] gtop_list size=" << gtop_list.size()
        << " gbot_list size=" << gbot_list.size()
        << " ortho_list size=" << ortho_list.size()
        << "\n";

      // Updates @07/17:
      // Use the outer passCountFaces array so the coverage history remains
      // available after this branch for diagnostics and later extensions.
      passCountFaces.setZero();

      if(use_curved_material_mapping)
      {
          // Define coverage and direction in the original flat/material
          // coordinates, then map the selected faces to the curved mesh by
          // shared topology/indexing.
          zigzag::applyMaterialCoordinates(
              materialCoordinates,
              Connect,
              zz,
              gtop_list,
              gbot_list,
              ortho_list,
              growthRates_t,
              growthRates_b,
              growthAngles,
              orthoCoeffFaces,
              &passCountFaces);
      }
      else
      {
          // Preserve the existing flat-panel path for regression.
          zigzag::apply(
              mesh,
              zz,
              gtop_list,
              gbot_list,
              ortho_list,
              growthRates_t,
              growthRates_b,
              growthAngles,
              orthoCoeffFaces,
              &passCountFaces);
      }

      // Debug, print nonzero counts + min/max
      std::cout << "[zigzag] list sizes: gtop=" << gtop_list.size()
        << " gbot=" << gbot_list.size()
        << " ortho=" << ortho_list.size()
        << "\n";

      std::cout << "[zigzag] after apply: "
                << "nnz_top=" << nnz(growthRates_t)
                << " nnz_bot=" << nnz(growthRates_b)
                << " top[min,max]=[" << vmin(growthRates_t) << "," << vmax(growthRates_t) << "]"
                << " bot[min,max]=[" << vmin(growthRates_b) << "," << vmax(growthRates_b) << "]\n";
    }

    else if (growth_type == "panel_ortho")
    {
        const Real exx_top = parser.parse<Real>("-exx_top", 0.001);
        const Real eyy_top = parser.parse<Real>("-eyy_top", 0.001);
        const Real exx_bot = parser.parse<Real>("-exx_bot", -0.001);
        const Real eyy_bot = parser.parse<Real>("-eyy_bot", -0.001);

        // Optional rotation of the orthotropic frame relative to global x-y
        growthAngles = Eigen::VectorXd::Constant(nFaces, growthAngle);

        // Whole panel active
        growthRates_1_t = Eigen::VectorXd::Constant(nFaces, exx_top);
        growthRates_2_t = Eigen::VectorXd::Constant(nFaces, eyy_top);
        growthRates_1_b = Eigen::VectorXd::Constant(nFaces, exx_bot);
        growthRates_2_b = Eigen::VectorXd::Constant(nFaces, eyy_bot);

        // Keep scalar fields populated for diagnostics / compatibility if useful
        growthRates_t = 0.5 * (growthRates_1_t + growthRates_2_t);
        growthRates_b = 0.5 * (growthRates_1_b + growthRates_2_b);

        use_direct_ortho = true;

        std::cout << "[panel_ortho] exx_top=" << exx_top
                  << " eyy_top=" << eyy_top
                  << " exx_bot=" << exx_bot
                  << " eyy_bot=" << eyy_bot
                  << " growth_angle_deg=" << growthAngle * 180.0 / M_PI
                  << "\n";
    }



    else if (growth_type == "parallel_lines")
    {
        const Real width_mm   = parser.parse<Real>("-parline_width_mm", 1.0);
        const Real spacing_mm = parser.parse<Real>("-parline_spacing_mm", 2.0);
        const Real offset_mm  = parser.parse<Real>("-parline_offset_mm", 0.0);

        const Real exx_top = parser.parse<Real>("-parline_exx_top", 0.001);
        const Real exx_bot = parser.parse<Real>("-parline_exx_bot", -0.001);
        const Real eyy_top = parser.parse<Real>("-parline_eyy_top", 0.0);
        const Real eyy_bot = parser.parse<Real>("-parline_eyy_bot", 0.0);

        if (spacing_mm <= 0.0) {
            throw std::runtime_error("parallel_lines: spacing must be > 0");
        }
        if (width_mm <= 0.0) {
            throw std::runtime_error("parallel_lines: width must be > 0");
        }
        if (width_mm > spacing_mm) {
            throw std::runtime_error("parallel_lines: width must be <= spacing");
        }

        const Real width   = width_mm   * 1e-3;  // mm -> m
        const Real spacing = spacing_mm * 1e-3;  // mm -> m
        const Real offset  = offset_mm  * 1e-3;  // mm -> m

        const Real theta = growthAngle;

        // Tangent direction of the lines
        [[maybe_unused]] const Real tx = std::cos(theta);
        [[maybe_unused]] const Real ty = std::sin(theta);

        // Normal direction across the lines
        const Real nx_dir = -std::sin(theta);
        const Real ny_dir =  std::cos(theta);

        // Use direct orthotropic input
        use_direct_ortho = true;

        // Principal material frame used by computeAbarsOrthoGrowth
        growthAngles = Eigen::VectorXd::Constant(nFaces, theta);

        for (int i = 0; i < nFaces; ++i)
        {
            const int v0 = Connect(i, 0);
            const int v1 = Connect(i, 1);
            const int v2 = Connect(i, 2);

            const Real xc = (Vertices(v0,0) + Vertices(v1,0) + Vertices(v2,0)) / 3.0;
            const Real yc = (Vertices(v0,1) + Vertices(v1,1) + Vertices(v2,1)) / 3.0;

            // Coordinate across the parallel lines
            const Real s = xc * nx_dir + yc * ny_dir - offset;

            // Fold into one period, centered about zero
            Real s_mod = std::fmod(s, spacing);
            if (s_mod < 0.0) s_mod += spacing;
            if (s_mod > 0.5 * spacing) s_mod -= spacing;

            const bool in_line = (std::abs(s_mod) <= 0.5 * width);

            if (in_line)
            {
                growthRates_1_t(i) = exx_top;
                growthRates_1_b(i) = exx_bot;
                growthRates_2_t(i) = eyy_top;
                growthRates_2_b(i) = eyy_bot;

                // Compatibility/debug only
                growthRates_t(i) = 0.5 * (exx_top + eyy_top);
                growthRates_b(i) = 0.5 * (exx_bot + eyy_bot);
            }
            else
            {
                growthRates_1_t(i) = 0.0;
                growthRates_1_b(i) = 0.0;
                growthRates_2_t(i) = 0.0;
                growthRates_2_b(i) = 0.0;

                growthRates_t(i) = 0.0;
                growthRates_b(i) = 0.0;
            }
        }

        std::cout << "[parallel_lines] width_mm=" << width_mm
                  << " spacing_mm=" << spacing_mm
                  << " offset_mm=" << offset_mm
                  << " angle_deg=" << growthAngle * 180.0 / M_PI
                  << "\n";

        std::cout << "[parallel_lines] exx_top=" << exx_top
                  << " exx_bot=" << exx_bot
                  << " eyy_top=" << eyy_top
                  << " eyy_bot=" << eyy_bot
                  << "\n";
    }
    
    else {
        throw std::runtime_error("Unknown growth type: " + growth_type);
    }

    // Updates @07/17:
    // On a curved panel, define the no-growth margin in the original
    // material domain, not in the projected spatial x-y coordinates.
    auto MarginCutMaterialCoordinates =
        [&](const Real margin_u,
            const Real margin_v,
            Eigen::Ref<Eigen::VectorXd> rates_bot,
            Eigen::Ref<Eigen::VectorXd> rates_top)
        {
            const Real uMin = materialCoordinates.col(0).minCoeff();
            const Real uMax = materialCoordinates.col(0).maxCoeff();
            const Real vMin = materialCoordinates.col(1).minCoeff();
            const Real vMax = materialCoordinates.col(1).maxCoeff();

            Eigen::VectorXi boundaryVertex(nVert);
            boundaryVertex.setZero();

            for(int i = 0; i < nVert; ++i)
            {
                const Real u = materialCoordinates(i,0);
                const Real v = materialCoordinates(i,1);

                if(u <= uMin + margin_u ||
                   u >= uMax - margin_u ||
                   v <= vMin + margin_v ||
                   v >= vMax - margin_v)
                    boundaryVertex(i) = 1;
            }

            for(int i = 0; i < nFaces; ++i)
            {
                if(boundaryVertex(Connect(i,0)) == 1 &&
                   boundaryVertex(Connect(i,1)) == 1 &&
                   boundaryVertex(Connect(i,2)) == 1)
                {
                    rates_bot(i) = 0.0;
                    rates_top(i) = 0.0;
                }
            }
        };

    if (margin_x > 0.0 || margin_y > 0.0)
    {
        if (use_direct_ortho)
        {
            if(use_curved_material_mapping)
            {
                MarginCutMaterialCoordinates(
                    margin_x, margin_y,
                    growthRates_1_b, growthRates_1_t);
                MarginCutMaterialCoordinates(
                    margin_x, margin_y,
                    growthRates_2_b, growthRates_2_t);
            }
            else
            {
                MarginCut(
                    margin_x, margin_y,
                    growthRates_1_b, growthRates_1_t);
                MarginCut(
                    margin_x, margin_y,
                    growthRates_2_b, growthRates_2_t);
            }
        }
        else
        {
            if(use_curved_material_mapping)
                MarginCutMaterialCoordinates(
                    margin_x, margin_y,
                    growthRates_b, growthRates_t);
            else
                MarginCut(
                    margin_x, margin_y,
                    growthRates_b, growthRates_t);
        }
    }

    // Debug, print nonzero counts + min/max
    // std::cout << "[zigzag] after MarginCut: "
    //       << "nnz_top=" << nnz(growthRates_t)
    //       << " nnz_bot=" << nnz(growthRates_b)
    //       << " top[min,max]=[" << vmin(growthRates_t) << "," << vmax(growthRates_t) << "]"
    //       << " bot[min,max]=[" << vmin(growthRates_b) << "," << vmax(growthRates_b) << "]\n";

    if (use_direct_ortho) {
        std::cout << "[panel_ortho] after MarginCut: "
                  << " top_dir1[min,max]=[" << vmin(growthRates_1_t) << "," << vmax(growthRates_1_t) << "]"
                  << " top_dir2[min,max]=[" << vmin(growthRates_2_t) << "," << vmax(growthRates_2_t) << "]"
                  << " bot_dir1[min,max]=[" << vmin(growthRates_1_b) << "," << vmax(growthRates_1_b) << "]"
                  << " bot_dir2[min,max]=[" << vmin(growthRates_2_b) << "," << vmax(growthRates_2_b) << "]\n";
    } else {
        std::cout << "[growth] after MarginCut: "
                  << "nnz_top=" << nnz(growthRates_t)
                  << " nnz_bot=" << nnz(growthRates_b)
                  << " top[min,max]=[" << vmin(growthRates_t) << "," << vmax(growthRates_t) << "]"
                  << " bot[min,max]=[" << vmin(growthRates_b) << "," << vmax(growthRates_b) << "]\n";
    }

    // Might be an issue for other cases than zigzag if some faces are completely inactive, which causes issues for the growth helper. So we set them to zero growth and zero orthotropic coefficient explicitly here.
    // const double eps0 = 1e-20;
    // for (int i=0; i<nFaces; ++i) {
    //     if (std::abs(growthRates_t(i)) + std::abs(growthRates_b(i)) <= eps0) {
    //         passCountFaces(i) = 0;
    //     }
    // }
    const double eps0 = 1e-20;
    for (int i=0; i<nFaces; ++i) {
        if (use_direct_ortho) {
            const Real mag =
                std::abs(growthRates_1_t(i)) + std::abs(growthRates_2_t(i)) +
                std::abs(growthRates_1_b(i)) + std::abs(growthRates_2_b(i));
            if (mag <= eps0) passCountFaces(i) = 0;
        } else {
            if (std::abs(growthRates_t(i)) + std::abs(growthRates_b(i)) <= eps0) {
                passCountFaces(i) = 0;
            }
        }
    }

    // const Eigen::VectorXd growthAngles = Eigen::VectorXd::Constant(nFaces, growthAngle);

    // Eigen::VectorXd growthRates_1_t(nFaces);
    // Eigen::VectorXd growthRates_1_b(nFaces);
    // Eigen::VectorXd growthRates_2_t(nFaces);
    // Eigen::VectorXd growthRates_2_b(nFaces);

    // for (int i=0; i<nFaces; i++){
    //   const Real oc = orthoCoeffFaces(i);
    //   growthRates_1_t(i) = growthRates_t(i)*(1.0+oc);
    //   growthRates_1_b(i) = growthRates_b(i)*(1.0+oc);
    //   growthRates_2_t(i) = growthRates_t(i)*(1.0-oc);
    //   growthRates_2_b(i) = growthRates_b(i)*(1.0-oc);
    // }
    if (!use_direct_ortho) {
        for (int i=0; i<nFaces; i++){
          const Real oc = orthoCoeffFaces(i);
          growthRates_1_t(i) = growthRates_t(i)*(1.0+oc);
          growthRates_1_b(i) = growthRates_b(i)*(1.0+oc);
          growthRates_2_t(i) = growthRates_t(i)*(1.0-oc);
          growthRates_2_b(i) = growthRates_b(i)*(1.0-oc);
        }
    }

    // Updates @07/17:
    // Convert each material-space growth angle into a 3D tangent direction on
    // the curved rest surface. Flat cases continue to use the legacy angle-
    // based growth helper.
    Eigen::MatrixXd growthDirections3D;
    if(use_curved_material_mapping)
    {
        GrowthHelper<tMesh>::mapMaterialAnglesToShellDirections(
            mesh,
            materialCoordinates,
            growthAngles,
            growthDirections3D);
    }

    auto updateRestMetrics =
        [&]()
        {
            if(use_curved_material_mapping)
            {
                GrowthHelper<tMesh>::computeAbarsOrthoGrowthShell(
                    mesh,
                    growthDirections3D,
                    growthRates_1_b,
                    growthRates_2_b,
                    mesh.getRestConfiguration()
                        .getFirstFundamentalForms<bottom>());

                GrowthHelper<tMesh>::computeAbarsOrthoGrowthShell(
                    mesh,
                    growthDirections3D,
                    growthRates_1_t,
                    growthRates_2_t,
                    mesh.getRestConfiguration()
                        .getFirstFundamentalForms<top>());
            }
            else
            {
                GrowthHelper<tMesh>::computeAbarsOrthoGrowth(
                    mesh,
                    growthAngles,
                    growthRates_1_b,
                    growthRates_2_b,
                    mesh.getRestConfiguration()
                        .getFirstFundamentalForms<bottom>());

                GrowthHelper<tMesh>::computeAbarsOrthoGrowth(
                    mesh,
                    growthAngles,
                    growthRates_1_t,
                    growthRates_2_t,
                    mesh.getRestConfiguration()
                        .getFirstFundamentalForms<top>());
            }
        };

    updateRestMetrics();

    // Updates @07/17:
    // Write a dedicated mapping diagnostic before minimization. This file lets
    // us verify the unwrapped material coordinates, selected faces, mapped 3D
    // path directions, and tangency to the curved panel.
    if(use_curved_material_mapping)
    {
        const Eigen::MatrixXd Xrest =
            mesh.getRestConfiguration().getVertices();

        WriteVTK mappingWriter(Xrest, Connect);

        Eigen::VectorXd materialU = materialCoordinates.col(0);
        Eigen::VectorXd materialV = materialCoordinates.col(1);
        Eigen::VectorXd passCountReal =
            passCountFaces.cast<Real>();

        mappingWriter.addScalarFieldToVertices(
            materialU, "material_u");
        mappingWriter.addScalarFieldToVertices(
            materialV, "material_v");
        mappingWriter.addVectorFieldToFaces(
            growthDirections3D, "growth_dir_3d");
        mappingWriter.addScalarFieldToFaces(
            growthAngles, "growth_angle_material");
        mappingWriter.addScalarFieldToFaces(
            passCountReal, "pass_count");
        mappingWriter.addScalarFieldToFaces(
            growthRates_1_b, "rate1_bot");
        mappingWriter.addScalarFieldToFaces(
            growthRates_2_b, "rate2_bot");
        mappingWriter.addScalarFieldToFaces(
            growthRates_1_t, "rate1_top");
        mappingWriter.addScalarFieldToFaces(
            growthRates_2_t, "rate2_top");

        Real maxNormalDot = 0.0;
        for(int i = 0; i < nFaces; ++i)
        {
            const Eigen::Vector3d x0 =
                Xrest.row(Connect(i,0)).transpose();
            const Eigen::Vector3d x1 =
                Xrest.row(Connect(i,1)).transpose();
            const Eigen::Vector3d x2 =
                Xrest.row(Connect(i,2)).transpose();

            const Eigen::Vector3d normal =
                (x1 - x0).cross(x2 - x0).normalized();
            const Eigen::Vector3d direction =
                growthDirections3D.row(i).transpose();

            maxNormalDot = std::max(
                maxNormalDot,
                std::abs(normal.dot(direction)));
        }

        std::cout
            << "[curved_mapping] max |n dot d1| = "
            << maxNormalDot << "\n";

        mappingWriter.write(tag + "_curved_mapping");
    }

    // write initial condition
    mesh.writeToFile(tag+"_init");

    // dump
    dumpOrtho(growthRates_1_b, growthRates_2_b, growthRates_1_t, growthRates_2_t, growthAngles, tag+"_init");

    // define the material and operator
    // Check here later for different material properties for the top and bottom layers
    // MaterialProperties_Iso_Constant matprop_bot(E, nu, h_total);
    // MaterialProperties_Iso_Constant matprop_top(E, nu, h_total);
    std::unique_ptr<MaterialProperties<Material_Isotropic>> matprop_bot_ptr;
    std::unique_ptr<MaterialProperties<Material_Isotropic>> matprop_top_ptr;

    if (enable_passE) {
        matprop_bot_ptr = std::make_unique<MaterialProperties_Iso_Array>(E_face, nu, h_total);
        matprop_top_ptr = std::make_unique<MaterialProperties_Iso_Array>(E_face, nu, h_total);
    } else {
        matprop_bot_ptr = std::make_unique<MaterialProperties_Iso_Constant>(E, nu, h_total);
        matprop_top_ptr = std::make_unique<MaterialProperties_Iso_Constant>(E, nu, h_total);
    }

    CombinedOperator_Parametric<tMesh, Material_Isotropic, bottom> engOp_bot(*matprop_bot_ptr);
    CombinedOperator_Parametric<tMesh, Material_Isotropic, top>    engOp_top(*matprop_top_ptr);
    EnergyOperatorList<tMesh> engOps({&engOp_bot, &engOp_top});

    // CombinedOperator_Parametric<tMesh, Material_Isotropic, bottom> engOp_bot(matprop_bot);
    // CombinedOperator_Parametric<tMesh, Material_Isotropic, top> engOp_top(matprop_top);
    // EnergyOperatorList<tMesh> engOps({&engOp_bot, &engOp_top});

    // dump 0 swelling rate (initial condition) for nicer movies afterwards
    dumpOrtho(growthRates_1_b, growthRates_2_b, growthRates_1_t, growthRates_2_t, growthAngles, tag+"_final_"+helpers::ToString(0,2));

    // ver-0122, parse CLI args once and pass to the minimize call in TestCustomGrowth()
    const std::string dump_iters_str = parser.parse<std::string>("-dump_iters", "");
    const std::vector<int> dump_iters = parse_int_list(dump_iters_str);
    const int max_iter = parser.parse<int>("-max_iter", -1);

    // swelling loop
    const int nSwellingRuns = parser.parse<int>("-nsteps", 1);
    const Real swelling_step = 1.0/((Real)nSwellingRuns);
    const int startidx = 0;

    std::string curTag;
    for(int s=startidx;s<nSwellingRuns;++s)
    {
        curTag = tag+"_final_"+helpers::ToString(s+1,2); // s+1 since we already dumped s=0 as the initial condition

        const Real swelling_fac = (s+1)*swelling_step;

        // Updates @07/17: use the same flat or curved metric-update path
        // selected above. For this first curved test, run with -nsteps 1.
        updateRestMetrics();

        // minimize energy, old version
        // Real eps = 1e-2;
        // minimizeEnergy(engOps, eps);

        // ver-0122
        // Real eps = 1e-2;
        // minimizeEnergy(engOps, eps,
        //               std::numeric_limits<Real>::epsilon(),
        //               false,
        //               //(dump_iters.empty() ? nullptr : &dump_iters),
        //               &dump_iters,
        //               max_iter);

        Real eps_init = 1e-2;                           // keep
        Real tol = parser.parse<Real>("-tol", 1e-6);    // new
        bool stepwise = parser.parse<bool>("-stepwise", false);

        minimizeEnergy(engOps, eps_init,
                      tol,
                      stepwise,
                      (dump_iters.empty() ? nullptr : &dump_iters),
                      max_iter);

        // Record how the solve terminated. Without this a step truncated by -max_iter,
        // or one whose line search gave up early, is indistinguishable from an
        // equilibrium -- both in the dumps and as the starting guess for the next step.
        {
            const Real gradTol = parser.parse<Real>("-gradtol", -1.0);
            const bool converged = lastMinimization.converged(gradTol);
            const std::string cfile = tag + "_convergence.dat";
            FILE * f = fopen(cfile.c_str(), (s == startidx) ? "w" : "a");
            if(f != nullptr)
            {
                if(s == startidx)
                    fprintf(f, "# step \t swelling \t hlbfgs code \t iterations \t grad norm \t converged\n");
                fprintf(f, "%d \t %10.10e \t %d \t %d \t %10.10e \t %d\n",
                        s, swelling_fac, lastMinimization.code,
                        lastMinimization.iterations, lastMinimization.gradientNorm,
                        converged ? 1 : 0);
                fclose(f);
            }
            if(not converged)
                printf("WARNING : swelling step %d (fac %10.10e) did not reach equilibrium -- HLBFGS code %d after %d iterations, |g| = %10.10e\n",
                       s, swelling_fac, lastMinimization.code,
                       lastMinimization.iterations, lastMinimization.gradientNorm);
        }

        // dump
        // dumpIso(growthRates_b, growthRates_t, curTag);
        dumpOrtho(growthRates_1_b, growthRates_2_b, growthRates_1_t, growthRates_2_t, growthAngles, curTag);
    }

    const std::string fname1 = curTag + ".vtp";
    IOGeometry geometry_dummy(fname1);
    mesh.init_rest(geometry_dummy);
    dumpOrtho(growthRates_1_b, growthRates_2_b, growthRates_1_t, growthRates_2_t, growthAngles, tag+"_final");
    
    dumpWithNormals(tag+"_final_curv");

}





// Generate n random patterns on a rectangular plate and solve the forward problem for each of them
void Sim_Bilayer_Growth::TestRandomPatterns()
{

    initForwardProblem();

    const int nFaces = mesh.getNumberOfFaces();

    Eigen::MatrixXi adj_faces(nFaces,3);
    const auto face2edges = mesh.getTopology().getFace2Edges();
    const auto edge2faces = mesh.getTopology().getEdge2Faces();

    Real adj_faceidx = 0.0;

    for(int i=0;i<nFaces;++i)
    {
      for(int e0=0;e0<3;++e0){
        for(int f0=0;f0<2;++f0)
        {
          adj_faceidx = edge2faces(face2edges(i, e0), f0);
          if (adj_faceidx!=i){
            adj_faces(i,e0) = adj_faceidx;
          }
        }
      }
    }

    const int nVert = mesh.getNumberOfVertices();
    const int nEdges = mesh.getNumberOfEdges();
    const auto vertex2faces = mesh.getTopology().getVertex2Faces();
    const auto edge2vertices = mesh.getTopology().getEdge2Vertices();

    Eigen::VectorXi vertex_edge(nVert);
    vertex_edge.setZero();

    for (int i=0;i<nVert;++i){
      for (int j=0;j<nEdges;++j){
        for (int k=0;k<2;++k){
          if(edge2vertices(j,k) == i){
            ++vertex_edge(i);
          }
        }
      }
    }

    int ntests = parser.parse<int>("-ntests", 3);

    Eigen::VectorXd h_ac (4);
    Eigen::VectorXd growth (4);

    Eigen::VectorXd growthRate1 (4);
    Eigen::VectorXd growthRate2 (4);

    growthRate1(0) = parser.parse<Real>("-growth1t", 0.00047);
    growthRate2(0) = parser.parse<Real>("-growth1b", -growthRate1(1));

    growthRate1(1) = parser.parse<Real>("-growth2t", 0.00082);
    growthRate2(1) = parser.parse<Real>("-growth2b", -growthRate1(2));

    growthRate1(2) = parser.parse<Real>("-growth3t", 0.0012);
    growthRate2(2) = parser.parse<Real>("-growth3b", -growthRate1(3));

    growthRate1(3) = parser.parse<Real>("-growth4t", 0.001475);
    growthRate2(3) = parser.parse<Real>("-growth4b", -growthRate1(4));

    Eigen::VectorXd maxerror(ntests);
    Eigen::VectorXd relative_error(ntests);
    Eigen::MatrixXd h_nu(ntests,2);

    Eigen::VectorXi ActiveElms_top(nFaces);
    ActiveElms_top.setZero();

    Eigen::VectorXi ActiveElms_bot(nFaces);
    ActiveElms_bot.setZero();

    const Real E = 1;

    Eigen::VectorXd growthRates_bot(nFaces);
    Eigen::VectorXd growthRates_top(nFaces);

    growthRates_bot.setZero();
    growthRates_top.setZero();

    Eigen::VectorXd growthRates_bot_eqv(nFaces);
    Eigen::VectorXd growthRates_top_eqv(nFaces);

    growthRates_bot_eqv.setZero();
    growthRates_top_eqv.setZero();

    Eigen::VectorXd h_bot(nFaces);
    Eigen::VectorXd h_top(nFaces);



    for(int k=0;k<ntests;++k){

      growthRates_bot.setZero();
      growthRates_top.setZero();

      growthRates_bot_eqv.setZero();
      growthRates_top_eqv.setZero();

      tag = "TestRandom_"+ helpers::ToString(k,2);

      Real h_total = std::rand() % 12 + 4;
      h_total = h_total/1000.0;
      std::cout << "\n" << "h_total" <<  " = " << h_total << "\n" << std::endl;

//      Real h_total = parser.parse<Real>("-h_total", 0.005);

      Real nu = std::rand() % 5 + 32;
      nu = nu/100.0;
      std::cout << "\n" << "nu" <<  " = " << nu << "\n" << std::endl;

//      Real nu = parser.parse<Real>("-nu", 0.33);

      h_nu(k,0)=h_total;
      h_nu(k,1)=nu;

      for (int i=0 ; i<nFaces ; ++i){
        h_bot(i) = h_total;
        h_top(i) = h_total;
      }

      Real CenterX;
      Real CenterY;

      int regime;

      auto Vertices = mesh.getCurrentConfiguration().getVertices();
      const auto Connect = mesh.getTopology().getFace2Vertices();

      helpers::write_matrix("VERTICES.txt", Vertices);
      helpers::write_matrix("CONNEC.txt", Connect);

      Eigen::VectorXi IndicV(nVert);

      Eigen::VectorXi Nregime(nFaces);
      Nregime.setZero();

      Eigen::VectorXi Active_top(nFaces);
      Active_top.setZero();

      Eigen::VectorXi Active_bot(nFaces);
      Active_bot.setZero();

      Eigen::VectorXd GrowthFacs_bot_temp(nFaces);
      GrowthFacs_bot_temp.setZero();

      Eigen::VectorXd GrowthFacs_top_temp(nFaces);
      GrowthFacs_top_temp.setZero();

      const int nspots_bot = std::rand() % 6 + 1;
      const int nspots_top = std::rand() % 6 + 1;

      //////////// BOTTOM LAYER ////////////

      for(int n=0;n<nspots_bot;++n){

          Real spotsize = std::rand() % 40 + 5;
          spotsize = spotsize/100.0;

          IndicV.setZero();

          regime = std::rand() % 4 + 1;
          CenterX = std::rand() % 1000 - 500;
          CenterY = std::rand() % 1000 - 500;

          CenterX = CenterX/1000.0;
          CenterY = CenterY/1000.0;


          for (int i=0; i<nVert; ++i){

            if (Vertices(i,0)>(CenterX-spotsize) && Vertices(i,0)<(CenterX+spotsize) && Vertices(i,1)>(CenterY-spotsize) && Vertices(i,1)<(CenterY+spotsize)){
  	           IndicV(i) = 1;
            }

            if ((CenterX-spotsize)<-0.5){ // check if we overpassed the borders along X
              if (Vertices(i,0)>(0.5+(CenterX-spotsize+0.5)) && Vertices(i,1)>(CenterY-spotsize) && Vertices(i,1)<(CenterY+spotsize)){
                IndicV(i) = 1;
              }
            }

            if ((CenterX+spotsize)>0.5){
                if (Vertices(i,0)<(-0.5+(CenterX+spotsize-0.5)) && Vertices(i,1)>(CenterY-spotsize) && Vertices(i,1)<(CenterY+spotsize)){
                  IndicV(i) = 1;
                }
            }

            if ((CenterY-spotsize)<-0.5){
                if (Vertices(i,1)>(0.5+(CenterY-spotsize+0.5)) && Vertices(i,0)>(CenterX-spotsize) && Vertices(i,0)<(CenterX+spotsize)){
                  IndicV(i) = 1;
                }
            }

            if ((CenterY+spotsize)>0.5){
                if (Vertices(i,1)<(-0.5+(CenterY+spotsize-0.5)) && Vertices(i,0)>(CenterX-spotsize) && Vertices(i,0)<(CenterX+spotsize)){
                  IndicV(i) = 1;
                }
            }
          }

          for (int i=0; i<nFaces; ++i){
               if (IndicV(Connect(i,0))==1 && IndicV(Connect(i,1))==1 && IndicV(Connect(i,2))==1) {

                 if (growthRates_bot(i) == 0){
                   ActiveElms_bot(i) = ActiveElms_bot(i)+1;
                 }

                 h_bot(i) = h_ac(regime-1);
                 h_top(i) = h_ac(regime-1);
                 //h_top(i) = h_total - h_bot(i);
                 growthRates_bot(i) = growth(regime-1);

                 growthRates_bot_eqv(i) = growthRate1(regime-1);
                 growthRates_top_eqv(i) = growthRate2(regime-1);
                 Active_bot(i) = 1;
                 Nregime(i) = regime-1;
               }
             //}
          }

      }

      //////////// TOP LAYER ////////////

      for(int n=0;n<nspots_top;++n){

          Real spotsize = std::rand() % 40 + 5;
          spotsize = spotsize/100.0;

          IndicV.setZero();

          regime = std::rand() % 4 + 1;

          CenterX = std::rand() % 1000 - 500;
          CenterY = std::rand() % 1000 - 500;

          CenterX = CenterX/1000.0;
          CenterY = CenterY/1000.0;

          for (int i=0; i<nVert; ++i){

            if (Vertices(i,0)>(CenterX-spotsize) && Vertices(i,0)<(CenterX+spotsize) && Vertices(i,1)>(CenterY-spotsize) && Vertices(i,1)<(CenterY+spotsize)){
  	           IndicV(i) = 1;
            }

            if ((CenterX-spotsize)<-0.5){ // check if we overpassed the borders along X
              if (Vertices(i,0)>(0.5+(CenterX-spotsize+0.5)) && Vertices(i,1)>(CenterY-spotsize) && Vertices(i,1)<(CenterY+spotsize)){
                IndicV(i) = 1;
              }
            }

            if ((CenterX+spotsize)>0.5){
                if (Vertices(i,0)<(-0.5+(CenterX+spotsize-0.5)) && Vertices(i,1)>(CenterY-spotsize) && Vertices(i,1)<(CenterY+spotsize)){
                  IndicV(i) = 1;
                }
            }

            if ((CenterY-spotsize)<-0.5){
                if (Vertices(i,1)>(0.5+(CenterY-spotsize+0.5)) && Vertices(i,0)>(CenterX-spotsize) && Vertices(i,0)<(CenterX+spotsize)){
                  IndicV(i) = 1;
                }
            }

            if ((CenterY+spotsize)>0.5){
                if (Vertices(i,1)<(-0.5+(CenterY+spotsize-0.5)) && Vertices(i,0)>(CenterX-spotsize) && Vertices(i,0)<(CenterX+spotsize)){
                  IndicV(i) = 1;
                }
            }
          }

          for (int i=0; i<nFaces; ++i){
  	         if (IndicV(Connect(i,0))==1 && IndicV(Connect(i,1))==1 && IndicV(Connect(i,2))==1) {

               if (growthRates_top(i) == 0){
                 ActiveElms_top(i) = ActiveElms_top(i)+1;
               }

               growthRates_top(i) = growth(regime-1);
               h_top(i) = h_ac(regime-1);
               Active_top(i) = 1;

               if (Active_bot(i) != 1){
                 growthRates_top_eqv(i) = growthRate1(regime-1);
                 growthRates_bot_eqv(i) = growthRate2(regime-1);
                 h_bot(i) = h_ac(regime-1);
               }

               if (Active_bot(i) == 1){
                 growthRates_top_eqv(i) = growthRate2(Nregime(i)) + growthRate1(regime-1);
                 growthRates_bot_eqv(i) = growthRate1(Nregime(i)) + growthRate2(regime-1);
               }
              }
          }

      }

      helpers::write_matrix(tag+"_growthRates_bot.txt", growthRates_bot);
      helpers::write_matrix(tag+"_thickness_bot.txt", h_bot);
      
      helpers::write_matrix(tag+"_growthRates_top.txt", growthRates_top);
      helpers::write_matrix(tag+"_thickness_top.txt", h_top);
      
      helpers::write_matrix(tag+"_growthRates_bot_eqv.txt", growthRates_bot_eqv);
      helpers::write_matrix(tag+"_growthRates_top_eqv.txt", growthRates_top_eqv);
      
      helpers::write_matrix_binary(tag+"_growthRates_bot.dat", growthRates_bot);
      helpers::write_matrix_binary(tag+"_thickness_bot.dat", h_bot);
      
      helpers::write_matrix_binary(tag+"_growthRates_top.dat", growthRates_top);
      helpers::write_matrix_binary(tag+"_thickness_top.dat", h_top);
      
      helpers::write_matrix_binary(tag+"_growthRates_bot_eqv.dat", growthRates_bot_eqv);
      helpers::write_matrix_binary(tag+"_growthRates_top_eqv.dat", growthRates_top_eqv);

      /////////////////////////////////////////////

      tag = "TestRandom_"+ helpers::ToString(k,2);

      initForwardProblem();

      mesh.resetToRestState();

      GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, growthRates_bot_eqv, mesh.getRestConfiguration().getFirstFundamentalForms<bottom>());
      GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, growthRates_top_eqv, mesh.getRestConfiguration().getFirstFundamentalForms<top>());

      MaterialProperties_Iso_Constant matprop_bot_eqv(E, nu, h_total, 0.0);
      MaterialProperties_Iso_Constant matprop_top_eqv(E, nu, h_total, 0.0);

      CombinedOperator_Parametric<tMesh, Material_Isotropic, bottom> engOp_bot_eqv(matprop_bot_eqv);
      CombinedOperator_Parametric<tMesh, Material_Isotropic, top> engOp_top_eqv(matprop_top_eqv);
      EnergyOperatorList<tMesh> engOps_eqv({&engOp_bot_eqv, &engOp_top_eqv});

      // std::unique_ptr<MaterialProperties<Material_Isotropic>> matprop_bot_ptr;
      // std::unique_ptr<MaterialProperties<Material_Isotropic>> matprop_top_ptr;

      // if (enable_passE) {
      //     matprop_bot_ptr = std::make_unique<MaterialProperties_Iso_Array>(E_face, nu, h_total);
      //     matprop_top_ptr = std::make_unique<MaterialProperties_Iso_Array>(E_face, nu, h_total);
      // } else {
      //     matprop_bot_ptr = std::make_unique<MaterialProperties_Iso_Constant>(E, nu, h_total);
      //     matprop_top_ptr = std::make_unique<MaterialProperties_Iso_Constant>(E, nu, h_total);
      // }

      // CombinedOperator_Parametric<tMesh, Material_Isotropic, bottom> engOp_bot(*matprop_bot_ptr);
      // CombinedOperator_Parametric<tMesh, Material_Isotropic, top>    engOp_top(*matprop_top_ptr);
      // EnergyOperatorList<tMesh> engOps({&engOp_bot, &engOp_top});

      // write initial condition
      mesh.writeToFile(tag+"_init");

      // dump
      dumpIso(growthRates_bot_eqv, growthRates_top_eqv, tag+"_init");

      // ver-0122, parse CLI args once and pass to the minimize call in TestCustomGrowth()
      const std::string dump_iters_str = parser.parse<std::string>("-dump_iters", "");
      const std::vector<int> dump_iters = parse_int_list(dump_iters_str);
      const int max_iter = parser.parse<int>("-max_iter", -1);

      const int nSwellingRuns = parser.parse<int>("-nsteps", 1);
      const Real steppo = 1.0/nSwellingRuns;

      for(int s=0;s<nSwellingRuns;++s)
      {

          const Real swelling_fac = steppo*(s+1);

          // apply swelling
          GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, swelling_fac*growthRates_bot_eqv, mesh.getRestConfiguration().getFirstFundamentalForms<bottom>());
          GrowthHelper<tMesh>::computeAbarsIsoGrowth(mesh, swelling_fac*growthRates_top_eqv, mesh.getRestConfiguration().getFirstFundamentalForms<top>());
          
          // old version
          // Real eps = 1e-2;
          // minimizeEnergy(engOps_eqv, eps);

          // New version for control over tolerances

          // Real eps = 1e-2;
          // minimizeEnergy(engOps_eqv, eps,
          //               std::numeric_limits<Real>::epsilon(),
          //               false,
          //               (dump_iters.empty() ? nullptr : &dump_iters),
          //               max_iter);

          Real eps_init = 1e-2;                           // keep
          Real tol = parser.parse<Real>("-tol", 1e-6);    // new
          bool stepwise = parser.parse<bool>("-stepwise", false);

          minimizeEnergy(engOps_eqv, eps_init,
                        tol,
                        stepwise,
                        (dump_iters.empty() ? nullptr : &dump_iters),
                        max_iter);

      }

      // dump
      dumpIso(growthRates_bot_eqv, growthRates_top_eqv, tag+"_final");

      // write
      mesh.writeToFile(tag+"_final");

      auto Vertices_Final_Eqv = mesh.getCurrentConfiguration().getVertices();
      helpers::write_matrix(tag+"_vertices_final.txt", Vertices_Final_Eqv);

    }

    tag = "TestRandom";
    helpers::write_matrix(tag+"_h_nu.txt", h_nu);
    helpers::write_matrix_binary(tag+"_h_nu.dat", h_nu);

}

void Sim_Bilayer_Growth::initForwardProblem()
{
    const std::string geometryCase = parser.parse<std::string>("-geometry", "");

    if (geometryCase == "external")
    // external mesh from the fname file
    {
        const std::string fname = parser.template parse<std::string>("-basename", "");
        IOGeometry geometry(fname);
        mesh.init(geometry);
    }
    else if (geometryCase == "external_rectangle_clamped")
    // external mesh from the fname file
    // all translations of vertices that lie in the margins are restricted
    {
        const std::string fname = parser.template parse<std::string>("-basename", "");
        const Real margin_width = parser.parse<Real>("-margin_width", 0.0); //margins to simulate the clamping frame: no eigensrain in this zone. 0.001 = 1mm
        const Real fixedX = parser.parse<Real>("-fixedX", true); // either we fix the borders aligned with x- or not
        const Real fixedY = parser.parse<Real>("-fixedY", true); // either we fix the borders aligned with y- or not
        IOGeometry_Rectangle_Clamped geometry(fname, margin_width, fixedX, fixedY);
        const Real clamped = parser.parse<Real>("-clamped", true);
        mesh.init(geometry, clamped);
    }
    else if (geometryCase == "curved_rectangle")
    // Updates @07/17:
    // Analytical cylindrical panel with the same regular topology as the
    // existing right-angle rectangular plate. Curvature is across material y.
    {
      const Real res =
          parser.parse<Real>("-res", 0.01);
      const Real Lx =
          parser.parse<Real>("-lx", 0.5);
      const Real Ly =
          parser.parse<Real>("-ly", 0.5);
      const Real curve_radius =
          parser.parse<Real>("-curve_radius", 0.25);
      const Real curve_sign =
          parser.parse<Real>("-curve_sign", 1.0);

      const Real relArea = 2.0 * Lx * res;

      CurvedRectangularPlate_RightAngle geometry(
          Lx,
          Ly,
          relArea,
          curve_radius,
          curve_sign);

      mesh.init(geometry);

      std::cout
          << "[curved_rectangle] Lx=" << Lx
          << ", Ly=" << Ly
          << ", R=" << curve_radius
          << ", sign=" << ((curve_sign >= 0.0) ? 1.0 : -1.0)
          << ", total opening angle="
          << 2.0 * Ly / curve_radius
          << " rad\n";
    }
    else if (geometryCase == "rectangle")
    // regular mesh
    {
      const Real res = parser.parse<Real>("-res", 0.03); // production default selected by the field-convergence sweep
      const Real Lx = parser.parse<Real>("-lx", 0.5);
      const Real Ly = parser.parse<Real>("-ly", 0.5);
      const Real relArea = 2.0*Lx*res;
      RectangularPlate_RightAngle geometry(Lx, Ly, relArea, false, false);
      mesh.init(geometry);

    }
    else if (geometryCase == "rectangle_allclamped")
    // regular mesh
    // all translations of vertices that lie in the margins are restricted
    {
      const Real res = parser.parse<Real>("-res", 0.01); //res = 1/(quantity of nodes per boundary)
      const Real Lx = parser.parse<Real>("-lx", 0.5);
      const Real Ly = parser.parse<Real>("-ly", 0.5);
      const Real relArea = 2.0*Lx*res;
      const Real margin_width = parser.parse<Real>("-margin_width", 0.0);
      const Real fixedX = parser.parse<Real>("-fixedX", true);   // either we fix the borders aligned with x- or not
      const Real fixedY = parser.parse<Real>("-fixedY", true);   // either we fix the borders aligned with y- or not
      RectangularPlate_RightAngle_Clamped geometry(Lx, Ly, relArea, margin_width, fixedX, fixedY);
      const Real clamped = parser.parse<Real>("-clamped", true);
      mesh.init(geometry, clamped);
    }
    else if (geometryCase == "rectangle_3clampvert")
    // regular mesh
    // one vertex is fixed along all three axes, one vertex is fixed along x- and z-, and one vertex is fixed only along z-
    {
      const Real res = parser.parse<Real>("-res", 0.01); //res = 1/(quantity of nodes per boundary)
      const Real Lx = parser.parse<Real>("-lx", 0.5);
      const Real Ly = parser.parse<Real>("-ly", 0.5);
      const Real relArea = 2.0*Lx*res;
      RectangularPlate_3clampvert geometry(Lx, Ly, relArea, false, false);
      mesh.init(geometry);
    }
    else if (geometryCase == "rectangle_irreg") 
    //irregular mesh
    {
      const Real res = parser.parse<Real>("-res", 0.01); //res = 1/(quantity of nodes per boundary)
      const Real Lx = parser.parse<Real>("-lx", 0.5);
      const Real Ly = parser.parse<Real>("-ly", 0.5);
      const Real relArea = 2.0*Lx*res;
      RectangularPlate geometry(Lx, Ly, relArea, {false,false}, {false,false});
      mesh.init(geometry);
    }
    else
    {
        std::cout << "No valid geometry defined. Options are \n";
        std::cout << "\t -geometry external\n";
        std::cout << "\t -geometry external_rectangle_clamped\n";
        std::cout << "\t -geometry curved_rectangle\n"; // Updates @07/17
        std::cout << "\t -geometry rectangle\n";
        std::cout << "\t -geometry rectangle_allclamped\n";
        std::cout << "\t -geometry rectangle_3clampvert\n";
        std::cout << "\t -geometry rectangle_irreg\n";
        helpers::catastrophe("no valid geometry", __FILE__, __LINE__);
    }
}





void Sim_Bilayer_Growth::dumpIso(const Eigen::Ref<const Eigen::VectorXd> growthRates_bot, const Eigen::Ref<const Eigen::VectorXd> growthRates_top, const std::string filename, const bool restConfig)
{
    // const auto cvertices = restConfig ? mesh.getRestConfiguration().getVertices() : mesh.getCurrentConfiguration().getVertices();;
    // const auto cface2vertices = mesh.getTopology().getFace2Vertices();

    // WriteVTK writer(cvertices, cface2vertices);

    // New version for displacement field
    // --- Always write geometry in REST configuration (reference mesh) ---
    const auto X0 = mesh.getRestConfiguration().getVertices();      // nV x 3
    const auto X  = mesh.getCurrentConfiguration().getVertices();   // nV x 3
    const auto cface2vertices = mesh.getTopology().getFace2Vertices();

    WriteVTK writer(X0, cface2vertices);

    // --- Displacement fields (PointData) ---
    Eigen::MatrixXd U = X - X0;                   // nV x 3
    Eigen::VectorXd U3 = U.col(2);                // nV
    Eigen::VectorXd Umag = U.rowwise().norm();    // nV

    writer.addVectorFieldToVertices(U, "U");
    writer.addScalarFieldToVertices(U3, "U3");
    writer.addScalarFieldToVertices(Umag, "Umag");
    writer.addVectorFieldToVertices(X, "X_current"); // helpful for debugging

    // --- Growth fields (existing face/vertex logic) ---
    if(growthRates_bot.rows() == mesh.getNumberOfFaces())
    {
        writer.addScalarFieldToFaces(growthRates_bot, "growthrates_bot");
        writer.addScalarFieldToFaces(growthRates_top, "growthrates_top");
    }
    else if(growthRates_bot.rows() == mesh.getNumberOfVertices())
    {
        writer.addScalarFieldToVertices(growthRates_bot, "growthrates_bot");
        writer.addScalarFieldToVertices(growthRates_top, "growthrates_top");
    }
    else
    {
        const std::string errmsg = ("Problem  : number of rows in growthRates = "+std::to_string(growthRates_bot.rows())+" while nVertices / nFaces = "+std::to_string(mesh.getNumberOfVertices())+" , "+std::to_string(mesh.getNumberOfFaces()));
        helpers::catastrophe(errmsg, __FILE__, __LINE__, false);
    }

    const int nFaces = mesh.getNumberOfFaces();
    Eigen::VectorXd gauss(nFaces);
    Eigen::VectorXd mean(nFaces);
    ComputeCurvatures<tMesh> computeCurvatures;
    computeCurvatures.compute(mesh, gauss, mean);
    writer.addScalarFieldToFaces(gauss, "gauss");
    writer.addScalarFieldToFaces(mean, "mean");

    writer.write(filename);

    // Optional: also write deformed STL (geometry only)
    const bool export_stl = parser.parse<bool>("-export_stl", false);
    const bool stl_ascii  = parser.parse<bool>("-stl_ascii", false);
    
    // Only write STL for FINAL dumps (and only for current config)
    auto ends_with = [](const std::string& s, const std::string& suf) {
      return s.size() >= suf.size() &&
            s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
    };

    const bool is_final = ends_with(filename, "_final");

    if (export_stl && is_final && !restConfig)
    {
        WriteSTL::write(mesh.getTopology(), mesh.getCurrentConfiguration(),
                        filename + "_deformed", stl_ascii);
    }
}

void Sim_Bilayer_Growth::dumpOrtho(Eigen::Ref<Eigen::VectorXd> growthRates_1_bot, Eigen::Ref<Eigen::VectorXd> growthRates_2_bot, Eigen::Ref<Eigen::VectorXd> growthRates_1_top, Eigen::Ref<Eigen::VectorXd> growthRates_2_top, const Eigen::Ref<const Eigen::VectorXd> growthAngles, const std::string filename, const bool restConfig)
{
  // const auto cvertices = restConfig ? mesh.getRestConfiguration().getVertices() : mesh.getCurrentConfiguration().getVertices();
  // const auto cface2vertices = mesh.getTopology().getFace2Vertices();
  // const int nFaces = mesh.getNumberOfFaces();
  // WriteVTK writer(cvertices, cface2vertices);

  // new version for displacement field
  // --- Always write geometry in REST configuration (reference mesh) ---
  const auto X0 = mesh.getRestConfiguration().getVertices();      // nV x 3
  const auto X  = mesh.getCurrentConfiguration().getVertices();   // nV x 3
  const auto cface2vertices = mesh.getTopology().getFace2Vertices();
  const int nFaces = mesh.getNumberOfFaces();

  WriteVTK writer(X0, cface2vertices);

  // --- Displacement field (Abaqus-like U) as PointData ---
  Eigen::MatrixXd U = X - X0;                   // nV x 3
  Eigen::VectorXd U3 = U.col(2);                // nV
  Eigen::VectorXd Umag = U.rowwise().norm();    // nV

  writer.addVectorFieldToVertices(U, "U");
  writer.addScalarFieldToVertices(U3, "U3");
  writer.addScalarFieldToVertices(Umag, "Umag");
  writer.addVectorFieldToVertices(X, "X_current");

  // --- Existing state handles ---
  const TopologyData & topology = mesh.getTopology();
  const tReferenceConfigData & restState = mesh.getRestConfiguration();
  const tCurrentConfigData & currentState = mesh.getCurrentConfiguration();
  const BoundaryConditionsData & boundaryConditions = mesh.getBoundaryConditions();

  Eigen::MatrixXd normal_vectors(nFaces,3);
  if(restConfig)
      restState.computeFaceNormalsFromDirectors(topology, boundaryConditions, normal_vectors);
  else
      currentState.computeFaceNormalsFromDirectors(topology, boundaryConditions, normal_vectors);

  writer.addVectorFieldToFaces(normal_vectors, "normals");

  if (restConfig == false)
  {
      Eigen::VectorXd gauss(nFaces);
      Eigen::VectorXd mean(nFaces);
      Eigen::VectorXd PrincCurv1(nFaces);
      Eigen::VectorXd PrincCurv2(nFaces);
      Eigen::VectorXd CurvX(nFaces);
      Eigen::VectorXd CurvY(nFaces);

      Eigen::Vector3d Dir1 = (Eigen::Vector3d() <<  1, 0, 0).finished();
      Eigen::Vector3d Dir2 = (Eigen::Vector3d() <<  0, 1, 0).finished();

      ComputeCurvatures<tMesh> computeCurvatures;
      computeCurvatures.computeDir(mesh, gauss, mean, PrincCurv1, PrincCurv2, CurvX, CurvY, Dir1, Dir2);

      writer.addScalarFieldToFaces(gauss, "gauss");
      writer.addScalarFieldToFaces(mean, "mean");
      writer.addScalarFieldToFaces(PrincCurv1, "PrincCurv1");
      writer.addScalarFieldToFaces(PrincCurv2, "PrincCurv2");
      writer.addScalarFieldToFaces(CurvX, "CurvX");
      writer.addScalarFieldToFaces(CurvY, "CurvY");
  }

  writer.addScalarFieldToFaces(growthRates_1_bot, "rate1_bot");
  writer.addScalarFieldToFaces(growthRates_2_bot, "rate2_bot");
  writer.addScalarFieldToFaces(growthAngles, "dir_growth");
  writer.addScalarFieldToFaces(growthRates_1_top, "rate1_top");
  writer.addScalarFieldToFaces(growthRates_2_top, "rate2_top");

  writer.write(filename);

  const bool export_stl = parser.parse<bool>("-export_stl", false);
  const bool stl_ascii  = parser.parse<bool>("-stl_ascii", false);

  // Only write STL for FINAL dumps (and only for current config)
  auto ends_with = [](const std::string& s, const std::string& suf) {
    return s.size() >= suf.size() &&
           s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
  };

  const bool is_final = ends_with(filename, "_final");

  if (export_stl && is_final && !restConfig)
  {
      WriteSTL::write(mesh.getTopology(), mesh.getCurrentConfiguration(),
                      filename + "_deformed", stl_ascii);
  }

}



// eliminate any expansion from the margins
void Sim_Bilayer_Growth::MarginCut(const Real margin_x, const Real margin_y, Eigen::Ref<Eigen::VectorXd> GrowthFacs_bot, Eigen::Ref<Eigen::VectorXd> GrowthFacs_top)
{
  const auto Vertices = mesh.getRestConfiguration().getVertices();
  const auto Connect = mesh.getTopology().getFace2Vertices();

  const int nFaces = mesh.getNumberOfFaces();
  const int nVert = mesh.getNumberOfVertices();

  Eigen::VectorXi IndicV(nVert);
  IndicV.setZero();
  auto MinVert = Vertices.colwise().minCoeff();
  auto MaxVert = Vertices.colwise().maxCoeff();

  for (int i=0; i<nVert; ++i){
    if (Vertices(i,0)<=(MinVert(0)+margin_x) || Vertices(i,0)>=(MaxVert(0)-margin_x) || Vertices(i,1)<=(MinVert(1)+margin_y) || Vertices(i,1)>=(MaxVert(1)-margin_y)){
      IndicV(i) = 1;
    }
  }

  for (int i=0; i<nFaces; ++i){
     if (IndicV(Connect(i,0))==1 && IndicV(Connect(i,1))==1 && IndicV(Connect(i,2))==1) {
       GrowthFacs_bot(i) = 0.0;
       GrowthFacs_top(i) = 0.0;
     }
  }
}




void Sim_Bilayer_Growth::init()
{
}




void  Sim_Bilayer_Growth::computeQuadraticForms(tVecMat2d & firstFF, tVecMat2d & secondFF)
{
    const int nFaces = mesh.getNumberOfFaces();
    const auto currentState = mesh.getCurrentConfiguration();

    firstFF.resize(nFaces);
    secondFF.resize(nFaces);

    for(int i=0;i<nFaces;++i)
    {
        firstFF[i] = currentState.getTriangleInfo(i).computeFirstFundamentalForm();
        secondFF[i] = currentState.getTriangleInfo(i).computeSecondFundamentalForm();
    }
}




// reformulate the growth from the step profile to the bilayer profile 
void Sim_Bilayer_Growth::Trilayer_2_Bilayer(Real epseq, Real heq, const Real h, Real *theta_top, Real *theta_bot)
{
  *theta_top = heq*epseq*(3*h-2*heq)/std::pow(h,2);
  *theta_bot = heq*epseq*(2*heq-h)/std::pow(h,2);
}

// reformulate the growth from the bilayer profile to the step profile 
void Sim_Bilayer_Growth::Bilayer_2_Trilayer(Real *epseq, Real *heq, const Real h, Real theta_top, Real theta_bot)
{
  *heq=(theta_top+3.0*theta_bot)*h/((theta_top+theta_bot)*2.0);
  *epseq=std::pow((theta_top+theta_bot),2)/(theta_top+3.0*theta_bot);
}

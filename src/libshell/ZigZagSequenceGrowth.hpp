#pragma once

/**
 * ZigZagSequenceGrowth: Production JSON Engine for Recurring English Wheeling Sequences.
 *
 * Architecture:
 *  1. Toolpath Recipe & Repeat Lifecycle:
 *     - Defines an ordered sequence of physical forming cycles in JSON format.
 *     - Each toolpath recipe specifies geometric parameters (Lv, alpha, N, width, rotation),
 *       eigenstrain (gtop, gbot, ortho), profile (center_peak, uniform), and repeat counts.
 *     - Every repeat executes as a full physical forming cycle: toolpath application,
 *       nonlinear elastic energy minimization (HLBFGS), and VTP/CSV diagnostic output.
 *
 *  2. Metric Tensor Compounding:
 *     - Each hit computes an incremental growth tensor G(theta, g1, g2) in the flat
 *       material plane (u,v).
 *     - Compounds the metric via exact tensor pullback:
 *       abar_new = Dm^T * G^T * (Dm^(-T) * abar_old * Dm^(-1)) * G * Dm
 *
 *  3. Strain Hardening Model (Voce Decay):
 *     - Tracks face-level plastic hit counts across consecutive passes.
 *     - Scaling factor q(N_hits) = floor + (1 - floor) * exp(-beta * N_hits)
 *       reduces incremental plastic yield as metal work-hardens.
 *
 *  4. Boundary Clamping & Release:
 *     - Optionally enforces persistent physical clamp regions (e.g. edge clamps)
 *       during forming cycles, followed by unconstrained release minimization.
 */
#include "ZigZagGrowth.hpp"

#include <Eigen/Dense>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace zigzag_sequence {

using json = nlohmann::json;

enum class HardeningModel {
    None,
    VoceDecay
};

enum class CountRule {
    PlasticHits,
    AllToolHits
};

struct HardeningConfig {
    HardeningModel model = HardeningModel::VoceDecay;
    std::string history_variable = "hit_count";
    double beta = 0.0;
    double floor = 0.0;
    std::string scope = "face_shared";
    CountRule count_rule = CountRule::PlasticHits;
};

struct ProfileConfig {
    zigzag::TopProfileMode mode = zigzag::TopProfileMode::Uniform;
    double top_end_ratio = 1.0;
    double top_profile_power = 1.0;
};

struct DefaultsConfig {
    zigzag::StartMode start_mode = zigzag::StartMode::LeftBottom_Up;
    ProfileConfig profile;
};

struct ZigZagOperationConfig {
    double lv_mm = 40.0;
    double alpha_deg = 15.0;
    int n_strips = 6;
    double width_mm = 2.0;

    bool use_material_bbox_center = true;
    Eigen::Vector2d center_uv_m = Eigen::Vector2d::Zero();
    Eigen::Vector2d shift_uv_m = Eigen::Vector2d::Zero();
    zigzag::ShiftFrame shift_frame = zigzag::ShiftFrame::Material;
    double rotation_deg = 0.0;

    std::vector<double> gtop;
    std::vector<double> gbot;
    std::vector<double> ortho; // Legacy carrier for material-hit collection.
    std::vector<double> ortho_top, ortho_bottom;
    // Optional zero-based indices selecting physical straight strips from
    // the generated zigzag. Empty means all strips.
    std::vector<int> active_strips;

    ProfileConfig profile;
};

struct ToolpathConfig {
    std::string id;
    bool enabled = true;
    int repeat = 1;
    ZigZagOperationConfig operation;
};

// Updates @08/06:
// Persistent material-coordinate clamp regions for zigzag_sequence_BC.
// The first version supports only fully fixed rectangular regions.
struct FixedRectangleConfig {
    std::string name;
    Eigen::Vector2d center_uv_m = Eigen::Vector2d::Zero();
    Eigen::Vector2d size_uv_m = Eigen::Vector2d::Zero();
    double rotation_deg = 0.0;
};

struct BoundaryConditionsConfig {
    bool enabled = false;
    std::string coordinate_system = "material_uv";
    std::string application = "persistent";
    bool release_after_final_cycle = true;
    std::vector<FixedRectangleConfig> regions;
};

struct SequenceConfig {
    int schema_version = 1;
    HardeningConfig hardening;
    DefaultsConfig defaults;
    BoundaryConditionsConfig boundary_conditions;
    std::vector<ToolpathConfig> toolpaths;
};

inline void rejectUnknownKeys(const json& object,
                              const std::set<std::string>& allowed,
                              const std::string& context)
{
    if(!object.is_object())
        throw std::runtime_error(context + " must be a JSON object.");

    for(const auto& item : object.items())
    {
        if(allowed.count(item.key()) == 0)
            throw std::runtime_error(
                context + ": unknown key '" + item.key() + "'.");
    }
}

inline std::string getStringOrDefault(const json& j,
                                      const std::string& key,
                                      const std::string& fallback)
{
    if(!j.contains(key) || j.at(key).is_null())
        return fallback;
    if(!j.at(key).is_string())
        throw std::runtime_error("zigzag_sequence: key '" + key + "' must be a string.");
    return j.at(key).get<std::string>();
}

inline bool getBoolOrDefault(const json& j,
                             const std::string& key,
                             const bool fallback)
{
    if(!j.contains(key) || j.at(key).is_null())
        return fallback;
    if(!j.at(key).is_boolean())
        throw std::runtime_error("zigzag_sequence: key '" + key + "' must be boolean.");
    return j.at(key).get<bool>();
}

inline int getIntOrDefault(const json& j,
                           const std::string& key,
                           const int fallback)
{
    if(!j.contains(key) || j.at(key).is_null())
        return fallback;
    if(!j.at(key).is_number_integer())
        throw std::runtime_error("zigzag_sequence: key '" + key + "' must be an integer.");
    return j.at(key).get<int>();
}

inline double getDoubleOrDefault(const json& j,
                                 const std::string& key,
                                 const double fallback)
{
    if(!j.contains(key) || j.at(key).is_null())
        return fallback;
    if(!j.at(key).is_number())
        throw std::runtime_error("zigzag_sequence: key '" + key + "' must be numeric.");
    return j.at(key).get<double>();
}

inline Eigen::Vector2d readVec2Mm(const json& j,
                                  const std::string& key,
                                  const Eigen::Vector2d& fallback,
                                  bool* was_provided = nullptr)
{
    if(!j.contains(key) || j.at(key).is_null())
    {
        if(was_provided) *was_provided = false;
        return fallback;
    }

    const auto& value = j.at(key);
    if(!value.is_array() || value.size() != 2 ||
       !value.at(0).is_number() || !value.at(1).is_number())
        throw std::runtime_error(
            "zigzag_sequence: key '" + key + "' must be [u_mm, v_mm].");

    if(was_provided) *was_provided = true;
    return Eigen::Vector2d(
        value.at(0).get<double>() * 1e-3,
        value.at(1).get<double>() * 1e-3);
}

// Updates @07/19:
// Lists are the canonical JSON form. For quick trials a scalar is accepted and
// broadcast to all strips. Arrays must have exactly n_strips entries; unlike
// the legacy CLI parser, this routine never pads or truncates silently.
inline std::vector<double> readScalarOrExactList(const json& j,
                                                 const std::string& key,
                                                 const int n_strips,
                                                 const bool required,
                                                 const double omitted_default)
{
    if(!j.contains(key) || j.at(key).is_null())
    {
        if(required)
            throw std::runtime_error(
                "zigzag_sequence: operation must provide '" + key + "'.");
        return std::vector<double>(n_strips, omitted_default);
    }

    const auto& value = j.at(key);
    if(value.is_number())
        return std::vector<double>(n_strips, value.get<double>());

    if(!value.is_array())
        throw std::runtime_error(
            "zigzag_sequence: '" + key + "' must be a scalar or numeric array.");

    if(static_cast<int>(value.size()) != n_strips)
    {
        std::ostringstream oss;
        oss << "zigzag_sequence: '" << key << "' has " << value.size()
            << " entries, but n_strips=" << n_strips << ".";
        throw std::runtime_error(oss.str());
    }

    std::vector<double> result;
    result.reserve(n_strips);
    for(int i = 0; i < n_strips; ++i)
    {
        if(!value.at(i).is_number())
            throw std::runtime_error(
                "zigzag_sequence: every entry of '" + key + "' must be numeric.");
        result.push_back(value.at(i).get<double>());
    }
    return result;
}

inline HardeningModel parseHardeningModel(const std::string& value)
{
    if(value == "none") return HardeningModel::None;
    if(value == "voce_decay") return HardeningModel::VoceDecay;
    throw std::runtime_error(
        "zigzag_sequence: hardening.model must be 'none' or 'voce_decay'.");
}

inline CountRule parseCountRule(const std::string& value)
{
    if(value == "plastic_hits") return CountRule::PlasticHits;
    if(value == "all_tool_hits") return CountRule::AllToolHits;
    throw std::runtime_error(
        "zigzag_sequence: hardening.count_rule must be 'plastic_hits' or 'all_tool_hits'.");
}

inline zigzag::ShiftFrame parseShiftFrame(const std::string& value)
{
    if(value == "material") return zigzag::ShiftFrame::Material;
    if(value == "pattern") return zigzag::ShiftFrame::Pattern;
    throw std::runtime_error(
        "zigzag_sequence: shift_frame must be 'material' or 'pattern'.");
}

inline zigzag::StartMode parseStartMode(const std::string& value)
{
    if(value == "left_bottom_up")
        return zigzag::StartMode::LeftBottom_Up;
    throw std::runtime_error(
        "zigzag_sequence: start_mode currently supports only 'left_bottom_up'.");
}

inline ProfileConfig parseProfile(const json& object,
                                  const ProfileConfig& fallback,
                                  const std::string& context)
{
    if(object.is_null()) return fallback;
    rejectUnknownKeys(
        object,
        {"mode", "top_end_ratio", "top_profile_power"},
        context);

    ProfileConfig result = fallback;
    result.mode = zigzag::parseTopProfileMode(
        getStringOrDefault(object, "mode",
            result.mode == zigzag::TopProfileMode::CenterPeak ?
                "center_peak" : "uniform"));
    result.top_end_ratio = getDoubleOrDefault(
        object, "top_end_ratio", result.top_end_ratio);
    result.top_profile_power = getDoubleOrDefault(
        object, "top_profile_power", result.top_profile_power);

    if(result.top_end_ratio < 0.0 || result.top_end_ratio > 1.0)
        throw std::runtime_error(
            context + ": top_end_ratio must be in [0,1].");
    if(result.top_profile_power <= 0.0)
        throw std::runtime_error(
            context + ": top_profile_power must be > 0.");
    return result;
}

inline SequenceConfig loadSequenceJson(const std::string& filename)
{
    std::ifstream input(filename);
    if(!input)
        throw std::runtime_error(
            "zigzag_sequence: cannot open cycle file: " + filename);

    json root;
    input >> root;
    rejectUnknownKeys(
        root,
        {"schema_version", "units", "hardening", "defaults",
         "boundary_conditions", "toolpaths"},
        "zigzag_sequence root");

    SequenceConfig sequence;
    sequence.schema_version = getIntOrDefault(root, "schema_version", 1);
    if(sequence.schema_version != 1)
        throw std::runtime_error(
            "zigzag_sequence: only schema_version=1 is supported.");

    if(root.contains("units"))
    {
        const auto& units = root.at("units");
        rejectUnknownKeys(
            units,
            {"length", "angle", "growth"},
            "zigzag_sequence units");

        if(getStringOrDefault(units, "length", "mm") != "mm")
            throw std::runtime_error(
                "zigzag_sequence: units.length must be 'mm'.");
        if(getStringOrDefault(units, "angle", "deg") != "deg")
            throw std::runtime_error(
                "zigzag_sequence: units.angle must be 'deg'.");
        if(getStringOrDefault(units, "growth", "engineering_strain") !=
           "engineering_strain")
            throw std::runtime_error(
                "zigzag_sequence: units.growth must be 'engineering_strain'.");
    }


    if(root.contains("boundary_conditions"))
    {
        const auto& bc = root.at("boundary_conditions");
        rejectUnknownKeys(
            bc,
            {"enabled", "coordinate_system", "application",
             "release_after_final_cycle", "regions"},
            "zigzag_sequence boundary_conditions");

        auto& config = sequence.boundary_conditions;
        config.enabled = getBoolOrDefault(bc, "enabled", false);
        config.coordinate_system = getStringOrDefault(
            bc, "coordinate_system", "material_uv");
        config.application = getStringOrDefault(
            bc, "application", "persistent");
        config.release_after_final_cycle = getBoolOrDefault(
            bc, "release_after_final_cycle", true);

        if(config.coordinate_system != "material_uv")
            throw std::runtime_error(
                "zigzag_sequence: boundary_conditions.coordinate_system "
                "currently supports only 'material_uv'.");
        if(config.application != "persistent")
            throw std::runtime_error(
                "zigzag_sequence: boundary_conditions.application "
                "currently supports only 'persistent'.");

        if(bc.contains("regions"))
        {
            if(!bc.at("regions").is_array())
                throw std::runtime_error(
                    "zigzag_sequence: boundary_conditions.regions must be an array.");

            int region_index = 0;
            for(const auto& item : bc.at("regions"))
            {
                const std::string context =
                    "zigzag_sequence boundary_conditions.regions[" +
                    std::to_string(region_index) + "]";
                rejectUnknownKeys(
                    item,
                    {"name", "shape", "center_uv_mm", "size_uv_mm",
                     "rotation_deg"},
                    context);

                if(getStringOrDefault(item, "shape", "rectangle") !=
                   "rectangle")
                    throw std::runtime_error(
                        context + ": shape must be 'rectangle'.");

                FixedRectangleConfig region;
                region.name = getStringOrDefault(
                    item, "name", "clamp_" + std::to_string(region_index + 1));
                region.center_uv_m = readVec2Mm(
                    item, "center_uv_mm", Eigen::Vector2d::Zero());
                region.size_uv_m = readVec2Mm(
                    item, "size_uv_mm", Eigen::Vector2d::Zero());
                region.rotation_deg = getDoubleOrDefault(
                    item, "rotation_deg", 0.0);

                if(region.name.empty())
                    throw std::runtime_error(context + ": name must not be empty.");
                if(region.size_uv_m(0) <= 0.0 || region.size_uv_m(1) <= 0.0)
                    throw std::runtime_error(
                        context + ": size_uv_mm entries must both be > 0.");

                config.regions.push_back(std::move(region));
                ++region_index;
            }
        }

        if(config.enabled && config.regions.empty())
            throw std::runtime_error(
                "zigzag_sequence: enabled boundary_conditions require at least one region.");
    }

    if(root.contains("hardening"))
    {
        const auto& h = root.at("hardening");
        rejectUnknownKeys(
            h,
            {"model", "history_variable", "beta", "floor", "scope", "count_rule"},
            "zigzag_sequence hardening");

        sequence.hardening.model = parseHardeningModel(
            getStringOrDefault(h, "model", "voce_decay"));
        sequence.hardening.history_variable = getStringOrDefault(
            h, "history_variable", "hit_count");
        sequence.hardening.beta = getDoubleOrDefault(h, "beta", 0.0);
        sequence.hardening.floor = getDoubleOrDefault(h, "floor", 0.0);
        sequence.hardening.scope = getStringOrDefault(
            h, "scope", "face_shared");
        sequence.hardening.count_rule = parseCountRule(
            getStringOrDefault(h, "count_rule", "plastic_hits"));
    }

    if(sequence.hardening.history_variable != "hit_count")
        throw std::runtime_error(
            "zigzag_sequence: hardening.history_variable currently supports only 'hit_count'.");
    if(sequence.hardening.scope != "face_shared")
        throw std::runtime_error(
            "zigzag_sequence: hardening.scope currently supports only 'face_shared'.");
    if(sequence.hardening.beta < 0.0)
        throw std::runtime_error(
            "zigzag_sequence: hardening.beta must be >= 0.");
    if(sequence.hardening.floor < 0.0 || sequence.hardening.floor > 1.0)
        throw std::runtime_error(
            "zigzag_sequence: hardening.floor must be in [0,1].");

    if(root.contains("defaults"))
    {
        const auto& defaults = root.at("defaults");
        rejectUnknownKeys(
            defaults,
            {"start_mode", "profile"},
            "zigzag_sequence defaults");

        sequence.defaults.start_mode = parseStartMode(
            getStringOrDefault(defaults, "start_mode", "left_bottom_up"));
        if(defaults.contains("profile"))
            sequence.defaults.profile = parseProfile(
                defaults.at("profile"),
                sequence.defaults.profile,
                "zigzag_sequence defaults.profile");
    }

    if(!root.contains("toolpaths") || !root.at("toolpaths").is_array())
        throw std::runtime_error(
            "zigzag_sequence: root must contain a toolpaths array.");

    std::set<std::string> ids;
    int toolpath_index = 0;
    for(const auto& item : root.at("toolpaths"))
    {
        const std::string context =
            "zigzag_sequence toolpaths[" + std::to_string(toolpath_index) + "]";
        rejectUnknownKeys(
            item,
            {"id", "enabled", "repeat", "operation"},
            context);

        ToolpathConfig toolpath;
        toolpath.id = getStringOrDefault(
            item, "id", "toolpath_" + std::to_string(toolpath_index + 1));
        toolpath.enabled = getBoolOrDefault(item, "enabled", true);
        toolpath.repeat = getIntOrDefault(item, "repeat", 1);

        if(toolpath.id.empty())
            throw std::runtime_error(context + ": id must not be empty.");
        if(!ids.insert(toolpath.id).second)
            throw std::runtime_error(
                "zigzag_sequence: duplicate toolpath id '" + toolpath.id + "'.");
        if(toolpath.repeat < 1)
            throw std::runtime_error(
                context + ": repeat must be >= 1.");

        if(!item.contains("operation"))
            throw std::runtime_error(context + ": missing operation object.");
        const auto& operation = item.at("operation");
        rejectUnknownKeys(
            operation,
            {"type", "lv_mm", "alpha_deg", "n_strips", "width_mm",
             "center_uv_mm", "shift_uv_mm", "shift_frame", "rotation_deg",
             "gtop", "gbot", "ortho", "ortho_top", "ortho_bottom", "profile", "active_strips"},
            context + ".operation");

        if(getStringOrDefault(operation, "type", "zigzag") != "zigzag")
            throw std::runtime_error(
                context + ": operation.type must be 'zigzag'.");

        auto& op = toolpath.operation;
        op.lv_mm = getDoubleOrDefault(operation, "lv_mm", 40.0);
        op.alpha_deg = getDoubleOrDefault(operation, "alpha_deg", 15.0);
        op.n_strips = getIntOrDefault(operation, "n_strips", 6);
        op.width_mm = getDoubleOrDefault(operation, "width_mm", 2.0);
        op.rotation_deg = getDoubleOrDefault(operation, "rotation_deg", 0.0);

        bool center_provided = false;
        op.center_uv_m = readVec2Mm(
            operation,
            "center_uv_mm",
            Eigen::Vector2d::Zero(),
            &center_provided);
        op.use_material_bbox_center = !center_provided;
        op.shift_uv_m = readVec2Mm(
            operation,
            "shift_uv_mm",
            Eigen::Vector2d::Zero());
        op.shift_frame = parseShiftFrame(
            getStringOrDefault(operation, "shift_frame", "material"));

        if(op.lv_mm <= 0.0)
            throw std::runtime_error(context + ": lv_mm must be > 0.");
        if(op.width_mm <= 0.0)
            throw std::runtime_error(context + ": width_mm must be > 0.");
        if(op.n_strips < 2)
            throw std::runtime_error(context + ": n_strips must be >= 2.");

        op.gtop = readScalarOrExactList(
            operation, "gtop", op.n_strips, true, 0.0);
        op.gbot = readScalarOrExactList(
            operation, "gbot", op.n_strips, false, 0.0);
        const bool separate = operation.contains("ortho_top") ||
                              operation.contains("ortho_bottom");
        if(separate && (operation.contains("ortho") ||
                        !operation.contains("ortho_top") ||
                        !operation.contains("ortho_bottom")))
            throw std::runtime_error(context +
                ": supply either ortho, or BOTH ortho_top and ortho_bottom.");
        if(separate)
        {
            op.ortho_top = readScalarOrExactList(
                operation, "ortho_top", op.n_strips, true, 0.0);
            op.ortho_bottom = readScalarOrExactList(
                operation, "ortho_bottom", op.n_strips, true, 0.0);
        }
        else
        {
            op.ortho_top = readScalarOrExactList(
                operation, "ortho", op.n_strips, false, 0.0);
            op.ortho_bottom = op.ortho_top;
        }
        op.ortho = op.ortho_top;

        if(operation.contains("active_strips"))
        {
            const auto& active = operation.at("active_strips");
            if(!active.is_array())
                throw std::runtime_error(
                    context + ".operation.active_strips must be an integer array.");
            std::set<int> unique;
            for(const auto& value : active)
            {
                if(!value.is_number_integer())
                    throw std::runtime_error(
                        context + ".operation.active_strips must contain integers.");
                const int strip = value.get<int>();
                if(strip < 0 || strip >= op.n_strips)
                    throw std::runtime_error(
                        context + ".operation.active_strips index out of range.");
                unique.insert(strip);
            }
            op.active_strips.assign(unique.begin(), unique.end());
        }

        for(int i = 0; i < op.n_strips; ++i)
        {
            if(!std::isfinite(op.gtop[i]) || !std::isfinite(op.gbot[i]) ||
               !std::isfinite(op.ortho_top[i]) || !std::isfinite(op.ortho_bottom[i]) ||
               std::abs(op.ortho_top[i]) > 1.0 || std::abs(op.ortho_bottom[i]) > 1.0)
                throw std::runtime_error(
                    context + ": growth must be finite; layer ortho must be finite in [-1,1].");

            const double g1t = op.gtop[i] * (1.0 + op.ortho_top[i]);
            const double g2t = op.gtop[i] * (1.0 - op.ortho_top[i]);
            const double g1b = op.gbot[i] * (1.0 + op.ortho_bottom[i]);
            const double g2b = op.gbot[i] * (1.0 - op.ortho_bottom[i]);
            if(1.0 + g1t <= 0.0 || 1.0 + g2t <= 0.0 ||
               1.0 + g1b <= 0.0 || 1.0 + g2b <= 0.0)
                throw std::runtime_error(
                    context + ": a strip produces a nonpositive incremental stretch ratio.");
        }

        op.profile = sequence.defaults.profile;
        if(operation.contains("profile"))
            op.profile = parseProfile(
                operation.at("profile"),
                op.profile,
                context + ".operation.profile");

        if(toolpath.enabled)
            sequence.toolpaths.push_back(std::move(toolpath));
        ++toolpath_index;
    }

    if(sequence.toolpaths.empty())
        throw std::runtime_error(
            "zigzag_sequence: no enabled toolpaths were found.");

    return sequence;
}

inline double hardeningFactor(const HardeningConfig& hardening,
                              const int previous_hits)
{
    if(hardening.model == HardeningModel::None)
        return 1.0;
    return hardening.floor +
        (1.0 - hardening.floor) *
        std::exp(-hardening.beta * static_cast<double>(previous_hits));
}

inline bool countsTowardHistory(const HardeningConfig& hardening,
                                const zigzag::MaterialHit& hit,
                                const double tolerance = 1e-20)
{
    if(hardening.count_rule == CountRule::AllToolHits)
        return true;
    return std::abs(hit.gtop) + std::abs(hit.gbot) > tolerance;
}

} // namespace zigzag_sequence

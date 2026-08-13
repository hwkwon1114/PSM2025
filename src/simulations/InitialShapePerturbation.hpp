//
// InitialShapePerturbation.hpp
//
// Deterministic starting-shape profiles for numerical experiments.
//


#ifndef InitialShapePerturbation_hpp
#define InitialShapePerturbation_hpp

#include "common.hpp"

#include <stdexcept>

namespace initial_shape
{

inline Real height(const std::string & mode,
                   const Real x,
                   const Real y,
                   const Real radius,
                   const Real amplitude,
                   const int waves,
                   const Real randomValue = 0.0)
{
    if(radius <= 0.0)
        throw std::invalid_argument("Initial-shape radius must be positive");

    const Real xNormalized = x / radius;
    const Real yNormalized = y / radius;
    const Real radiusSquared = xNormalized*xNormalized + yNormalized*yNormalized;
    const Real boundaryEnvelope = std::max<Real>(0.0, 1.0 - radiusSquared);

    if(mode == "flat")
        return 0.0;
    if(mode == "random")
        return amplitude * boundaryEnvelope * randomValue;
    if(mode == "dome_up")
        return amplitude * boundaryEnvelope;
    if(mode == "dome_down")
        return -amplitude * boundaryEnvelope;
    if(mode == "saddle")
        return amplitude * (xNormalized*xNormalized - yNormalized*yNormalized);
    if(mode == "wrinkle")
    {
        if(waves <= 0)
            throw std::invalid_argument("Wrinkle initialization requires a positive wave count");
        const Real angle = std::atan2(y, x);
        return amplitude * boundaryEnvelope * std::cos(waves * angle);
    }

    throw std::invalid_argument(
        "Unknown initial-shape mode '" + mode
        + "'. Expected flat, random, dome_up, dome_down, saddle, or wrinkle");
}

} // namespace initial_shape

#endif /* InitialShapePerturbation_hpp */

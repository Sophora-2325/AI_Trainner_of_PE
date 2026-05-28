#define _USE_MATH_DEFINES
#include "retargeting.hpp"
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ─── Conversion functions ───────────────────────────────────────────

double retargetKnee(double humanDeg) {
    constexpr double NEUTRAL = 180.0;
    constexpr double MAX_DEVIATION = 120.0;
    constexpr double OP2_MAX_BEND = 2.0;

    double deviation = NEUTRAL - humanDeg;
    deviation = std::max(0.0, std::min(MAX_DEVIATION, deviation));
    return (deviation / MAX_DEVIATION) * OP2_MAX_BEND;
}

double retargetHip(double humanDeg) {
    constexpr double NEUTRAL = 180.0;
    constexpr double MAX_DEVIATION = 135.0;
    constexpr double OP2_MAX_FLEX = 1.5;

    double deviation = NEUTRAL - humanDeg;
    deviation = std::max(0.0, std::min(MAX_DEVIATION, deviation));
    return (deviation / MAX_DEVIATION) * OP2_MAX_FLEX;
}

double retargetAnkle(double humanDeg) {
    constexpr double NEUTRAL = 90.0;
    constexpr double MAX_DEVIATION = 20.0;
    constexpr double OP2_MAX_DORSI = 0.5;

    double deviation = NEUTRAL - humanDeg;
    deviation = std::max(-MAX_DEVIATION, std::min(MAX_DEVIATION, deviation));
    return (deviation / MAX_DEVIATION) * OP2_MAX_DORSI;
}

double retargetElbow(double humanDeg) {
    constexpr double NEUTRAL = 180.0;
    constexpr double MAX_DEVIATION = 135.0;
    constexpr double OP2_MAX_BEND = 2.5;

    double deviation = NEUTRAL - humanDeg;
    deviation = std::max(0.0, std::min(MAX_DEVIATION, deviation));
    return (deviation / MAX_DEVIATION) * OP2_MAX_BEND;
}

double retargetShoulder(double humanDeg) {
    constexpr double GAIN = 0.01;
    return humanDeg * GAIN;
}

double retargetAbduction(double humanDeg) {
    constexpr double GAIN = 0.005;
    return humanDeg * GAIN;
}

// ─── Clamp ──────────────────────────────────────────────────────────

double clamp(double value, double lo, double hi) {
    return std::max(lo, std::min(hi, value));
}

// ─── Full frame mapping ─────────────────────────────────────────────

std::map<std::string, double> mapHumanAnglesToOP2(
    const std::map<std::string, double>& humanAngles)
{
    std::map<std::string, double> positions;
    auto limits = getMotorLimits();
    auto standingOffset = getStandingOffset();
    auto humanToOP2 = getHumanToOP2Map();

    // 1. Initialize all motors to 0
    for (int i = 0; i < NMOTORS; i++) {
        positions[MOTOR_NAMES[i]] = 0.0;
    }

    // 2. Map each human angle to its OP2 motor
    for (const auto& [angleName, motorName] : humanToOP2) {
        auto it = humanAngles.find(angleName);
        if (it == humanAngles.end()) continue;

        double humanDeg = it->second;
        double rawRad = 0.0;

        // Determine angle type (strip _r/_l suffix)
        std::string base = angleName;
        if (base.size() > 2) base = base.substr(0, base.size() - 2);

        if (base == "knee_angle")
            rawRad = retargetKnee(humanDeg);
        else if (base == "hip_flexion")
            rawRad = retargetHip(humanDeg);
        else if (base == "ankle_angle")
            rawRad = retargetAnkle(humanDeg);
        else if (base == "elbow_angle")
            rawRad = retargetElbow(humanDeg);
        else if (base == "shoulder_angle")
            rawRad = retargetShoulder(humanDeg);
        else if (base == "hip_abduction")
            rawRad = retargetAbduction(humanDeg);

        // Symmetry: left-side motors get negated values
        if (angleName.size() >= 2 &&
            angleName.substr(angleName.size() - 2) == "_l") {
            rawRad = -rawRad;
        }

        positions[motorName] = rawRad;
    }

    // 3. Distribute lumbar extension across hip + shoulder pitch
    auto lumbarIt = humanAngles.find("lumbar_extension");
    if (lumbarIt != humanAngles.end()) {
        double lumbarDeg = lumbarIt->second;
        double hipOffset = (lumbarDeg * LUMBAR_HIP_RATIO) * M_PI / 180.0;
        double shoulderOffset = (lumbarDeg * LUMBAR_SHOULDER_RATIO) * M_PI / 180.0;

        positions["LegUpperR"] += hipOffset;
        positions["LegUpperL"] -= hipOffset;
        positions["ShoulderR"] += shoulderOffset;
        positions["ShoulderL"] -= shoulderOffset;
    }

    // 4. Apply standing offset
    for (const auto& [name, offset] : standingOffset) {
        positions[name] += offset;
    }

    // 5. Clamp all positions to motor limits
    for (auto& [name, value] : positions) {
        auto lim = limits.find(name);
        if (lim != limits.end()) {
            value = clamp(value, lim->second.min, lim->second.max);
        }
    }

    return positions;
}

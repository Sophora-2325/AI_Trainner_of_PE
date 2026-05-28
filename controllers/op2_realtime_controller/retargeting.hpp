#ifndef RETARGETING_HPP
#define RETARGETING_HPP

#include <map>
#include <string>
#include <vector>

// ─── Motor name constants (port from joint_config_op2.py) ───
constexpr const char* MOTOR_NAMES[] = {
    "ShoulderR", "ShoulderL", "ArmUpperR", "ArmUpperL",
    "ArmLowerR", "ArmLowerL", "PelvYR", "PelvYL",
    "PelvR", "PelvL", "LegUpperR", "LegUpperL",
    "LegLowerR", "LegLowerL", "AnkleR", "AnkleL",
    "FootR", "FootL", "Neck", "Head"
};
constexpr int NMOTORS = 20;

// ─── Human angle name → OP2 motor name mapping ───
// Left-side motors get negated values (symmetry from Symmetry.cpp)
inline std::map<std::string, std::string> getHumanToOP2Map() {
    return {
        {"knee_angle_r",    "LegLowerR"},
        {"knee_angle_l",    "LegLowerL"},
        {"hip_flexion_r",   "LegUpperR"},
        {"hip_flexion_l",   "LegUpperL"},
        {"ankle_angle_r",   "AnkleR"},
        {"ankle_angle_l",   "AnkleL"},
        {"elbow_angle_r",   "ArmLowerR"},
        {"elbow_angle_l",   "ArmLowerL"},
        {"shoulder_angle_r","ShoulderR"},
        {"shoulder_angle_l","ShoulderL"},
        {"hip_abduction_r", "PelvR"},
        {"hip_abduction_l", "PelvL"},
    };
}

// ─── Standing offset: makes "human standing" → stable OP2 crouch ───
inline std::map<std::string, double> getStandingOffset() {
    return {
        {"LegLowerR",  0.30},
        {"LegLowerL", -0.30},
        {"LegUpperR",  0.10},
        {"LegUpperL", -0.10},
        {"AnkleR",    -0.10},
        {"AnkleL",     0.10},
    };
}

// ─── Motor limits (radians) ───
struct MotorLimit { double min; double max; };
inline std::map<std::string, MotorLimit> getMotorLimits() {
    return {
        {"ShoulderR", {-2.0, 2.0}}, {"ShoulderL", {-2.0, 2.0}},
        {"ArmUpperR", {-1.5, 1.5}}, {"ArmUpperL", {-1.5, 1.5}},
        {"ArmLowerR", {-2.5, 2.5}}, {"ArmLowerL", {-2.5, 2.5}},
        {"PelvYR",    {-1.0, 1.0}}, {"PelvYL",    {-1.0, 1.0}},
        {"PelvR",     {-0.5, 0.5}}, {"PelvL",     {-0.5, 0.5}},
        {"LegUpperR", {-1.5, 1.5}}, {"LegUpperL", {-1.5, 1.5}},
        {"LegLowerR", {-2.5, 2.5}}, {"LegLowerL", {-2.5, 2.5}},
        {"AnkleR",    {-1.0, 1.0}}, {"AnkleL",    {-1.0, 1.0}},
        {"FootR",     {-0.5, 0.5}}, {"FootL",     {-0.5, 0.5}},
        {"Neck",      {-1.5, 1.5}}, {"Head",      {-0.8, 0.8}},
    };
}

// ─── Retargeting gains ───
constexpr double LUMBAR_HIP_RATIO     = 0.6;
constexpr double LUMBAR_SHOULDER_RATIO = 0.4;

// ─── Conversion functions: human degrees → OP2 radians ───
double retargetKnee(double humanDeg);
double retargetHip(double humanDeg);
double retargetAnkle(double humanDeg);
double retargetElbow(double humanDeg);
double retargetShoulder(double humanDeg);
double retargetAbduction(double humanDeg);

// ─── Full frame mapping ───
// Takes human joint angles dict → returns OP2 motor positions dict
std::map<std::string, double> mapHumanAnglesToOP2(
    const std::map<std::string, double>& humanAngles);

// Clamp a value to [lo, hi]
double clamp(double value, double lo, double hi);

#endif // RETARGETING_HPP

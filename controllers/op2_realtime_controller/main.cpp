// OP2 Real-Time Mirror Controller
// Receives human joint angles via TCP and drives OP2 motors in real time.
//
// Key controls:
//   Space — Pause / Resume mirroring
//   S     — Return to standing pose
//   ESC   — Quit

#include "retargeting.hpp"

#include <webots/Accelerometer.hpp>
#include <webots/Keyboard.hpp>
#include <webots/Motor.hpp>
#include <webots/PositionSensor.hpp>
#include <webots/Robot.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #pragma comment(lib, "ws2_32.lib")
  using socklen_t = int;
#else
  #include <arpa/inet.h>
  #include <netinet/in.h>
  #include <sys/socket.h>
  #include <unistd.h>
  #define SOCKET int
  #define INVALID_SOCKET (-1)
  #define SOCKET_ERROR (-1)
  #define closesocket close
#endif

using namespace webots;

// ─── Configuration ──────────────────────────────────────────────────
constexpr int   TCP_PORT          = 10020;
constexpr int   TRANSITION_STEPS  = 20;
constexpr int   STARTUP_GRACE     = 100;
constexpr int   FALL_THRESHOLD    = 50;
constexpr double FALL_AY_MIN      = 3.0;

// ─── JSON parsing helpers ───────────────────────────────────────────

static double extractJsonNumber(const std::string& json, const std::string& key, double defaultValue = 0.0) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return defaultValue;

    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return defaultValue;

    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t'))
        pos++;

    char* end = nullptr;
    const char* start = json.c_str() + pos;
    double value = std::strtod(start, &end);
    if (end == start) return defaultValue;
    return value;
}

static std::string extractJsonObject(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";

    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";

    pos = json.find('{', pos);
    if (pos == std::string::npos) return "";

    size_t start = pos;
    int depth = 1;
    pos++;
    while (pos < json.size() && depth > 0) {
        if (json[pos] == '{') depth++;
        else if (json[pos] == '}') depth--;
        pos++;
    }

    return json.substr(start, pos - start);
}

// ─── Main Controller Class ──────────────────────────────────────────

class OP2RealtimeController {
public:
    OP2RealtimeController() : robot_() {
        timeStep_ = (int)robot_.getBasicTimeStep();

        for (int i = 0; i < NMOTORS; i++) {
            Motor* motor = robot_.getMotor(MOTOR_NAMES[i]);
            motors_[MOTOR_NAMES[i]] = motor;
            PositionSensor* sensor = robot_.getPositionSensor(
                std::string(MOTOR_NAMES[i]) + "S");
            sensor->enable(timeStep_);
            positionSensors_[MOTOR_NAMES[i]] = sensor;

            motorLimits_[MOTOR_NAMES[i]] = {
                motor->getMinPosition(),
                motor->getMaxPosition()
            };
        }

        accelerometer_ = robot_.getAccelerometer("Accelerometer");
        accelerometer_->enable(timeStep_);

        keyboard_ = robot_.getKeyboard();
        keyboard_->enable(timeStep_);

        for (int i = 0; i < NMOTORS; i++) {
            targets_[MOTOR_NAMES[i]] = 0.0;
        }
    }

    ~OP2RealtimeController() {
        tcpRunning_ = false;
        if (tcpThread_.joinable()) tcpThread_.join();
#ifdef _WIN32
        WSACleanup();
#endif
        if (clientSock_ != INVALID_SOCKET) closesocket(clientSock_);
        if (serverSock_ != INVALID_SOCKET) closesocket(serverSock_);
    }

    void run() {
        printf("[OP2 Realtime] Starting on port %d\n", TCP_PORT);
        printf("[OP2 Realtime] Controls: Space=Pause, S=Stand, ESC=Quit\n");

        setInitialPose();

        for (int i = 0; i < 150; i++)
            robot_.step(timeStep_);

        for (int i = 0; i < NMOTORS; i++) {
            standingPose_[MOTOR_NAMES[i]] =
                positionSensors_[MOTOR_NAMES[i]]->getValue();
        }
        printf("[OP2 Realtime] Standing pose calibrated\n");

        startTCPServer();

        int stepCount = 0;
        while (robot_.step(timeStep_) != -1) {
            stepCount++;

            if (stepCount > STARTUP_GRACE && checkFallen()) {
                recoverFromFall();
            }

            handleKeyboard();

            if (transitionActive_) {
                advanceTransition();
            }

            if (!paused_ && !transitionActive_) {
                auto positions = getTargetPositions();
                setMotorsWithInterpolation(positions);
            }
        }

        printf("[OP2 Realtime] Shutting down.\n");
    }

private:
    Robot robot_;
    int timeStep_;

    std::map<std::string, Motor*> motors_;
    std::map<std::string, PositionSensor*> positionSensors_;
    std::map<std::string, MotorLimit> motorLimits_;
    Accelerometer* accelerometer_ = nullptr;
    Keyboard* keyboard_ = nullptr;

    std::mutex mutex_;
    std::map<std::string, double> targets_;
    std::map<std::string, double> motorTargets_;
    bool hasNewData_ = false;
    bool paused_ = false;
    bool mirroringActive_ = false;

    std::map<std::string, double> standingPose_;
    int fallCount_ = 0;

    bool transitionActive_ = false;
    int transitionStep_ = 0;
    int transitionTotal_ = TRANSITION_STEPS;
    std::map<std::string, double> transitionStart_;
    std::map<std::string, double> transitionEnd_;
    std::map<std::string, double> currentPositions_;

    SOCKET serverSock_ = INVALID_SOCKET;
    SOCKET clientSock_ = INVALID_SOCKET;
    std::thread tcpThread_;
    bool tcpRunning_ = false;

    void setInitialPose() {
        for (int i = 0; i < NMOTORS; i++) {
            const char* name = MOTOR_NAMES[i];
            double pos = 0.0;

            if (std::strcmp(name, "LegLowerR") == 0) pos = 0.3;
            else if (std::strcmp(name, "LegLowerL") == 0) pos = -0.3;
            else if (std::strcmp(name, "LegUpperR") == 0) pos = 0.1;
            else if (std::strcmp(name, "LegUpperL") == 0) pos = -0.1;
            else if (std::strcmp(name, "AnkleR") == 0) pos = -0.1;
            else if (std::strcmp(name, "AnkleL") == 0) pos = 0.1;

            motors_[name]->setPosition(pos);
            currentPositions_[name] = pos;
        }
    }

    void startTCPServer() {
#ifdef _WIN32
        WSADATA wsaData;
        WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif
        tcpRunning_ = true;
        tcpThread_ = std::thread(&OP2RealtimeController::tcpLoop, this);
    }

    void tcpLoop() {
        serverSock_ = socket(AF_INET, SOCK_STREAM, 0);
        if (serverSock_ == INVALID_SOCKET) {
            printf("[OP2 TCP] Failed to create socket\n");
            return;
        }

#ifdef _WIN32
        u_long mode = 1;
        ioctlsocket(serverSock_, FIONBIO, &mode);
#else
        int flags = fcntl(serverSock_, F_GETFL, 0);
        fcntl(serverSock_, F_SETFL, flags | O_NONBLOCK);
#endif

        int opt = 1;
        setsockopt(serverSock_, SOL_SOCKET, SO_REUSEADDR,
                   (const char*)&opt, sizeof(opt));

        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = INADDR_ANY;
        addr.sin_port = htons(TCP_PORT);

        if (bind(serverSock_, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR) {
            printf("[OP2 TCP] Bind failed\n");
            return;
        }

        listen(serverSock_, 1);
        printf("[OP2 TCP] Listening on port %d\n", TCP_PORT);

        while (tcpRunning_) {
            if (clientSock_ == INVALID_SOCKET) {
                sockaddr_in clientAddr{};
                socklen_t clientLen = sizeof(clientAddr);
                clientSock_ = accept(serverSock_, (sockaddr*)&clientAddr, &clientLen);

                if (clientSock_ != INVALID_SOCKET) {
                    printf("[OP2 TCP] Client connected\n");
                    mirroringActive_ = true;
                }
            }

            if (clientSock_ != INVALID_SOCKET) {
                if (!receiveMessage()) {
                    closesocket(clientSock_);
                    clientSock_ = INVALID_SOCKET;
                    mirroringActive_ = false;
                    printf("[OP2 TCP] Client disconnected\n");
                }
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }

    bool receiveMessage() {
        uint8_t lenBuf[4];
        int received = 0;
        while (received < 4) {
            int n = recv(clientSock_, (char*)(lenBuf + received), 4 - received, 0);
            if (n <= 0) return false;
            received += n;
        }

        uint32_t msgLen = (uint32_t(lenBuf[0]) << 24) |
                          (uint32_t(lenBuf[1]) << 16) |
                          (uint32_t(lenBuf[2]) << 8)  |
                          uint32_t(lenBuf[3]);

        if (msgLen > 65536) return false;

        std::vector<char> msgBuf(msgLen);
        received = 0;
        while (received < (int)msgLen) {
            int n = recv(clientSock_, msgBuf.data() + received,
                        (int)msgLen - received, 0);
            if (n <= 0) return false;
            received += n;
        }

        std::string json(msgBuf.data(), msgLen);
        parseAndApply(json);
        return true;
    }

    void parseAndApply(const std::string& json) {
        std::string targetsJson = extractJsonObject(json, "targets");
        if (targetsJson.empty()) return;

        std::map<std::string, double> humanAngles;
        const char* knownKeys[] = {
            "knee_angle_r", "knee_angle_l",
            "hip_flexion_r", "hip_flexion_l",
            "ankle_angle_r", "ankle_angle_l",
            "elbow_angle_r", "elbow_angle_l",
            "shoulder_angle_r", "shoulder_angle_l",
            "hip_abduction_r", "hip_abduction_l",
            "lumbar_extension",
            "knee_valgus_angle_r", "knee_valgus_angle_l",
            "knee_symmetry", "hip_symmetry", "knee_torque_r",
            nullptr
        };

        for (int i = 0; knownKeys[i] != nullptr; i++) {
            double val = extractJsonNumber(targetsJson, knownKeys[i], -999.0);
            if (val != -999.0) {
                humanAngles[knownKeys[i]] = val;
            }
        }

        if (humanAngles.empty()) return;

        auto motorPositions = mapHumanAnglesToOP2(humanAngles);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (const auto& [name, pos] : motorPositions) {
                motorTargets_[name] = pos;
            }
            hasNewData_ = true;
        }
    }

    std::map<std::string, double> getTargetPositions() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (mirroringActive_ && hasNewData_) {
            return motorTargets_;
        }
        std::map<std::string, double> hold;
        for (int i = 0; i < NMOTORS; i++) {
            hold[MOTOR_NAMES[i]] =
                positionSensors_[MOTOR_NAMES[i]]->getValue();
        }
        return hold;
    }

    void setMotorsWithInterpolation(const std::map<std::string, double>& targets) {
        for (const auto& [name, target] : targets) {
            double current = positionSensors_[name]->getValue();
            double alpha = 0.3;
            double smoothed = current + (target - current) * alpha;
            motors_[name]->setPosition(smoothed);
            currentPositions_[name] = smoothed;
        }
    }

    void startTransition(const std::map<std::string, double>& target, int steps = 30) {
        transitionActive_ = true;
        transitionStep_ = 0;
        transitionTotal_ = steps;
        transitionEnd_ = target;

        for (int i = 0; i < NMOTORS; i++) {
            transitionStart_[MOTOR_NAMES[i]] =
                positionSensors_[MOTOR_NAMES[i]]->getValue();
        }

        {
            std::lock_guard<std::mutex> lock(mutex_);
            mirroringActive_ = false;
        }
    }

    void advanceTransition() {
        transitionStep_++;
        if (transitionStep_ >= transitionTotal_) {
            for (const auto& [name, pos] : transitionEnd_) {
                motors_[name]->setPosition(pos);
                currentPositions_[name] = pos;
            }
            transitionActive_ = false;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                mirroringActive_ = (clientSock_ != INVALID_SOCKET);
            }
            return;
        }

        double t = (double)transitionStep_ / transitionTotal_;
        t = t * t * (3.0 - 2.0 * t);

        for (int i = 0; i < NMOTORS; i++) {
            const char* name = MOTOR_NAMES[i];
            double start = transitionStart_[name];
            double end = transitionEnd_.count(name) ? transitionEnd_[name] : start;
            double pos = start + (end - start) * t;
            motors_[name]->setPosition(pos);
            currentPositions_[name] = pos;
        }
    }

    bool checkFallen() {
        const double* acc = accelerometer_->getValues();
        double ay = acc[1];

        if (std::abs(ay) < FALL_AY_MIN) {
            fallCount_++;
        } else {
            fallCount_ = std::max(0, fallCount_ - 1);
        }

        return fallCount_ > FALL_THRESHOLD;
    }

    void recoverFromFall() {
        paused_ = false;
        transitionActive_ = false;
        fallCount_ = 0;
        printf("[OP2 Realtime] Fall detected! Returning to standing pose.\n");
        startTransition(standingPose_, 60);
    }

    void handleKeyboard() {
        int key;
        while ((key = keyboard_->getKey()) >= 0) {
            switch (key) {
            case ' ':
                paused_ = !paused_;
                printf("[OP2 Realtime] %s\n", paused_ ? "PAUSED" : "RESUMED");
                break;
            case 'S':
            case 's':
                paused_ = false;
                printf("[OP2 Realtime] Returning to standing pose\n");
                startTransition(standingPose_, 40);
                break;
            case 27:
                printf("[OP2 Realtime] Quit requested\n");
                tcpRunning_ = false;
                break;
            }
        }
    }
};

int main(int argc, char** argv) {
    OP2RealtimeController controller;
    controller.run();
    return 0;
}

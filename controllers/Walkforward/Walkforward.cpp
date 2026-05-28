#include <webots/Robot.hpp>
#include <webots/Keyboard.hpp>
#include <webots/Motor.hpp>

using namespace webots;

int main(int argc, char **argv) {
  Robot *robot = new Robot();

  // 获取键盘并启用
  Keyboard *keyboard = robot->getKeyboard();
  keyboard->enable(robot->getBasicTimeStep());

  // 获取电机设备，名称需与你的机器人模型中的电机名一致
  Motor *leftMotor = robot->getMotor("leftFootSlot");
  Motor *rightMotor = robot->getMotor("rightFootSlot");

  // 设置为速度控制模式
  leftMotor->setPosition(INFINITY);
  rightMotor->setPosition(INFINITY);

  int timeStep = (int)robot->getBasicTimeStep();
  const double speed = 5.0;  // 前进速度

  while (robot->step(timeStep) != -1) {
    int key = keyboard->getKey();

    if (key == 'W') {
      // 按下W键，径直向前
      leftMotor->setVelocity(speed);
      rightMotor->setVelocity(speed);
    } else {
      // 松开则停止
      leftMotor->setVelocity(0.0);
      rightMotor->setVelocity(0.0);
    }
  }

  delete robot;
  return 0;
}

#!/usr/bin/env python3
# coding=utf-8
from pylimo import limo
import time
limo=limo.LIMO()

''' https://pypi.org/project/pylimo/
Possible functions are:
    EnableCommand()
    SetMotionCommand()
    GetLinearVelocity()
    GetAngularVelocity()
    GetSteeringAngle()
    GetLateralVelocity()
    GetControlMode()
    GetBatteryVoltage()
    GetErrorCode()
    GetRightWheelOdem()
    GetLeftWheelOdem()
    GetIMUAccelData()
    GetIMUGyroData()
    GetIMUYawData()
    GetIMUPichData()
    GetIMURollData()
    '''

# tells it we're going to tell it to do stuff
limo.EnableCommand()

#example motion forward
# while True:
#     limo.SetMotionCommand(linear_vel=0.1,angular_vel=-0.01)
#     time.sleep(0.1)
# parameters
# linear_vel:(float)
# angular_vel:(float) diff
# lateral_velocity:(float) manamu
# steering_angle:(float) ackerman

#go straight example
# limo.SetMotionCommand(linear_vel=0.1,angular_vel=-0.01)
# time.sleep(5)

# # left circle example
# for i in range(50):
#     limo.SetMotionCommand(linear_vel=0.1,angular_vel=-0.01, steering_angle=30)
#     time.sleep(0.1)
#
# # right circle example
# for i in range(50):
#     limo.SetMotionCommand(linear_vel=0.1,angular_vel=-0.01, steering_angle=-30)
#     time.sleep(0.1)
#
# # left circle example
# for i in range(50):
#     limo.SetMotionCommand(linear_vel=-0.1,angular_vel=0.01, steering_angle=-30)
#     time.sleep(0.1)
#
# # right circle example
# for i in range(50):
#     limo.SetMotionCommand(linear_vel=-0.1,angular_vel=0.01, steering_angle=30)
#     time.sleep(0.1)

for i in range(50):
    limo.SetMotionCommand(linear_vel=-0.1,steering_angle=0)
    time.sleep(0.1)

for i in range(50):
    limo.SetMotionCommand(linear_vel=-0.1,steering_angle=60)
    time.sleep(0.1)

for i in range(50):
    limo.SetMotionCommand(linear_vel=-0.1, steering_angle=-60)
    time.sleep(0.1)
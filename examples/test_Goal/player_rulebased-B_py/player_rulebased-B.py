#!/usr/bin/python3

from __future__ import print_function

from twisted.internet import reactor
from twisted.internet.defer import inlineCallbacks

from autobahn.wamp.serializer import MsgPackSerializer
from autobahn.wamp.types import ComponentConfig
from autobahn.twisted.wamp import ApplicationSession, ApplicationRunner

import argparse
import random
import math
import sys

import base64
import numpy as np

import helper

# reset_reason
NONE = 0
GAME_START = 1
SCORE_MYTEAM = 2
SCORE_OPPONENT = 3
GAME_END = 4
DEADLOCK = 5
GOALKICK = 6
CORNERKICK = 7
PENALTYKICK = 8
HALFTIME = 9
EPISODE_END = 10

# game_state
STATE_DEFAULT = 0
STATE_KICKOFF = 1
STATE_GOALKICK = 2
STATE_CORNERKICK = 3
STATE_PENALTYKICK = 4

# coordinates
MY_TEAM = 0
OP_TEAM = 1
BALL = 2
X = 0
Y = 1
TH = 2
ACTIVE = 3
TOUCH = 4


class Received_Image(object):
    def __init__(self, resolution, colorChannels):
        self.resolution = resolution
        self.colorChannels = colorChannels
        # need to initialize the matrix at timestep 0
        self.ImageBuffer = np.zeros((resolution[1], resolution[0], colorChannels))  # rows, columns, colorchannels

    def update_image(self, received_parts):
        self.received_parts = received_parts
        for i in range(0, len(received_parts)):
            dec_msg = base64.b64decode(self.received_parts[i].b64, '-_')  # decode the base64 message
            np_msg = np.fromstring(dec_msg, dtype=np.uint8)  # convert byte array to numpy array
            reshaped_msg = np_msg.reshape((self.received_parts[i].height, self.received_parts[i].width, 3))
            for j in range(0, self.received_parts[i].height):  # y axis
                for k in range(0, self.received_parts[i].width):  # x axis
                    self.ImageBuffer[j + self.received_parts[i].y, k + self.received_parts[i].x, 0] = reshaped_msg[
                        j, k, 0]  # blue channel
                    self.ImageBuffer[j + self.received_parts[i].y, k + self.received_parts[i].x, 1] = reshaped_msg[
                        j, k, 1]  # green channel
                    self.ImageBuffer[j + self.received_parts[i].y, k + self.received_parts[i].x, 2] = reshaped_msg[
                        j, k, 2]  # red channel


class SubImage(object):
    def __init__(self, x, y, width, height, b64):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.b64 = b64


class Frame(object):
    def __init__(self):
        self.time = None
        self.score = None
        self.reset_reason = None
        self.game_state = None
        self.subimages = None
        self.coordinates = None
        self.half_passed = None


class Component(ApplicationSession):
    """
    AI Base + Rule Based Algorithm
    """

    def __init__(self, config):
        ApplicationSession.__init__(self, config)

    def printConsole(self, message):
        print(message)
        sys.__stdout__.flush()

    def onConnect(self):
        self.join(self.config.realm)

    @inlineCallbacks
    def onJoin(self, details):

        ##############################################################################
        def init_variables(self, info):
            # Here you have the information of the game (virtual init() in random_walk.cpp)
            # List: game_time, number_of_robots
            #       field, goal, penalty_area, goal_area, resolution Dimension: [x, y]
            #       ball_radius, ball_mass,
            #       robot_size, robot_height, axle_length, robot_body_mass, ID: [0, 1, 2, 3, 4]
            #       wheel_radius, wheel_mass, ID: [0, 1, 2, 3, 4]
            #       max_linear_velocity, max_torque, codewords, ID: [0, 1, 2, 3, 4]
            self.game_time = info['game_time']
            self.number_of_robots = info['number_of_robots']

            self.field = info['field']
            self.goal = info['goal']
            self.penalty_area = info['penalty_area']
            # self.goal_area = info['goal_area']
            self.resolution = info['resolution']

            self.ball_radius = info['ball_radius']
            # self.ball_mass = info['ball_mass']

            self.robot_size = info['robot_size']
            # self.robot_height = info['robot_height']
            # self.axle_length = info['axle_length']
            # self.robot_body_mass = info['robot_body_mass']

            # self.wheel_radius = info['wheel_radius']
            # self.wheel_mass = info['wheel_mass']

            self.max_linear_velocity = info['max_linear_velocity']
            # self.max_torque = info['max_torque']
            # self.codewords = info['codewords']

            self.colorChannels = 3
            self.end_of_frame = False
            self.image = Received_Image(self.resolution, self.colorChannels)
            self.cur_posture = []
            self.cur_ball = []
            self.prev_posture = []
            self.prev_ball = []
            self.previous_frame = Frame()
            self.received_frame = Frame()

            self.touch = [False,False,False,False,False]
            self.def_idx = 0
            self.atk_idx = 0

            self.wheels = [0 for _ in range(10)]
            return

        ##############################################################################

        try:
            info = yield self.call(u'aiwc.get_info', args.key)
        except Exception as e:
            self.printConsole("Error: {}".format(e))
        else:
            try:
                self.sub = yield self.subscribe(self.on_event, args.key)
            except Exception as e2:
                self.printConsole("Error: {}".format(e2))

        init_variables(self, info)

        try:
            yield self.call(u'aiwc.ready', args.key)
        except Exception as e:
            self.printConsole("Error: {}".format(e))
        else:
            self.printConsole("I am ready for the game!")

    # set the left and right wheel velocities of robot with id 'id'
    # 'max_velocity' scales the velocities up to the point where at least one of wheel is operating at max velocity
    def set_wheel_velocity(self, id, left_wheel, right_wheel, max_velocity):
        multiplier = 1

        # wheel velocities need to be scaled so that none of wheels exceed the maximum velocity available
        # otherwise, the velocity above the limit will be set to the max velocity by the simulation program
        # if that happens, the velocity ratio between left and right wheels will be changed that the robot may not execute
        # turning actions correctly.
        if (abs(left_wheel) > self.max_linear_velocity[id] or abs(right_wheel) > self.max_linear_velocity[
            id] or max_velocity):
            if (abs(left_wheel) > abs(right_wheel)):
                multiplier = self.max_linear_velocity[id] / abs(left_wheel)
            else:
                multiplier = self.max_linear_velocity[id] / abs(right_wheel)

        self.wheels[2 * id] = left_wheel * multiplier
        self.wheels[2 * id + 1] = right_wheel * multiplier

    # let the robot with id 'id' move to a target position (x, y)
    # the trajectory to reach the target position is determined by several different parameters
    def set_target_position(self, id, x, y, scale, mult_lin, mult_ang, max_velocity):
        damping = 0.35
        ka = 0
        sign = 1

        # calculate how far the target position is from the robot
        dx = x - self.cur_posture[id][X]
        dy = y - self.cur_posture[id][Y]
        d_e = math.sqrt(math.pow(dx, 2) + math.pow(dy, 2))

        # calculate how much the direction is off
        desired_th = (math.pi / 2) if (dx == 0 and dy == 0) else math.atan2(dy, dx)
        d_th = desired_th - self.cur_posture[id][TH]
        while (d_th > math.pi):
            d_th -= 2 * math.pi
        while (d_th < -math.pi):
            d_th += 2 * math.pi

        # based on how far the target position is, set a parameter that
        # decides how much importance should be put into changing directions
        # farther the target is, less need to change directions fastly
        if (d_e > 1):
            ka = 17 / 90
        elif (d_e > 0.5):
            ka = 19 / 90
        elif (d_e > 0.3):
            ka = 21 / 90
        elif (d_e > 0.2):
            ka = 23 / 90
        else:
            ka = 25 / 90

        # if the target position is at rear of the robot, drive backward instead
        if (d_th > helper.d2r(95)):
            d_th -= math.pi
            sign = -1
        elif (d_th < helper.d2r(-95)):
            d_th += math.pi
            sign = -1

        # if the direction is off by more than 85 degrees,
        # make a turn first instead of start moving toward the target
        if (abs(d_th) > helper.d2r(85)):
            self.set_wheel_velocity(id, -mult_ang * d_th, mult_ang * d_th, False)
        # otherwise
        else:
            # scale the angular velocity further down if the direction is off by less than 40 degrees
            if (d_e < 5 and abs(d_th) < helper.d2r(40)):
                ka = 0.1
            ka *= 4

            # set the wheel velocity
            # 'sign' determines the direction [forward, backward]
            # 'scale' scales the overall velocity at which the robot is driving
            # 'mult_lin' scales the linear velocity at which the robot is driving
            # larger distance 'd_e' scales the base linear velocity higher
            # 'damping' slows the linear velocity down
            # 'mult_ang' and 'ka' scales the angular velocity at which the robot is driving
            # larger angular difference 'd_th' scales the base angular velocity higher
            # if 'max_velocity' is true, the overall velocity is scaled to the point
            # where at least one wheel is operating at maximum velocity
            self.set_wheel_velocity(id,
                                    sign * scale * (mult_lin * (
                                                1 / (1 + math.exp(-3 * d_e)) - damping) - mult_ang * ka * d_th),
                                    sign * scale * (mult_lin * (
                                                1 / (1 + math.exp(-3 * d_e)) - damping) + mult_ang * ka * d_th),
                                    max_velocity)

    # copy coordinates from frames to different variables just for convenience
    def get_coord(self):
        self.cur_ball = self.received_frame.coordinates[BALL]
        self.cur_posture = self.received_frame.coordinates[MY_TEAM]
        self.cur_posture_op = self.received_frame.coordinates[OP_TEAM]
        self.prev_ball = self.previous_frame.coordinates[BALL]
        self.prev_posture = self.previous_frame.coordinates[MY_TEAM]
        self.prev_posture_op = self.previous_frame.coordinates[OP_TEAM]

    # find a defender and a forward closest to the ball
    def find_closest_robot(self):
        # find the closest defender
        min_idx = 0
        min_dist = 9999.99
        def_dist = 9999.99

        all_dist = []

        for i in [1, 2]:
            measured_dist = helper.dist(self.cur_ball[X], self.cur_posture[i][X], self.cur_ball[Y],
                                                self.cur_posture[i][Y])
            all_dist.append(measured_dist)
            if (measured_dist < min_dist):
                min_dist = measured_dist
                def_dist = min_dist
                min_idx = i

        self.def_idx = min_idx

        # find the closest forward
        min_idx = 0
        min_dist = 9999.99
        atk_dist = 9999.99

        for i in [3, 4]:
            measured_dist = helper.dist(self.cur_ball[X], self.cur_posture[i][X], self.cur_ball[Y],
                                                self.cur_posture[i][Y])
            all_dist.append(measured_dist)
            if (measured_dist < min_dist):
                min_dist = measured_dist
                atk_dist = min_dist
                min_idx = i

        self.atk_idx = min_idx

        # record the robot closer to the ball between the two too
        self.closest_order = np.argsort(all_dist) + 1

    # predict where the ball will be located after 'steps' steps
    def predict_ball_location(self, steps):
        dx = self.cur_ball[X] - self.prev_ball[X]
        dy = self.cur_ball[Y] - self.prev_ball[Y]
        return [self.cur_ball[X] + steps * dx, self.cur_ball[Y] + steps * dy]

    # let the robot face toward specific direction
    def face_specific_position(self, id, x, y):
        dx = x - self.cur_posture[id][X]
        dy = y - self.cur_posture[id][Y]

        desired_th = (math.pi / 2) if (dx == 0 and dy == 0) else math.atan2(dy, dx)

        self.angle(id, desired_th)

    # returns the angle toward a specific position from current robot posture
    def direction_angle(self, id, x, y):
        dx = x - self.cur_posture[id][X]
        dy = y - self.cur_posture[id][Y]

        return ((math.pi / 2) if (dx == 0 and dy == 0) else math.atan2(dy, dx))

    # turn to face 'desired_th' direction
    def angle(self, id, desired_th):
        mult_ang = 0.4

        d_th = desired_th - self.cur_posture[id][TH]
        d_th = helper.trim_radian(d_th)

        # the robot instead puts the direction rear if the angle difference is large
        if (d_th > helper.d2r(95)):
            d_th -= math.pi
            sign = -1
        elif (d_th < helper.d2r(-95)):
            d_th += math.pi
            sign = -1

        self.set_wheel_velocity(id, -mult_ang * d_th, mult_ang * d_th, False)

    # checks if a certain position is inside the penalty area of 'team'
    def in_penalty_area(self, obj, team):
        if (abs(obj[Y]) > self.penalty_area[Y] / 2):
            return False

        if (team == MY_TEAM):
            return (obj[X] < -self.field[X] / 2 + self.penalty_area[X])
        else:
            return (obj[X] > self.field[X] / 2 - self.penalty_area[X])

    # check if the ball is coming toward the robot
    def ball_coming_toward_robot(self, id):
        x_dir = abs(self.cur_posture[id][X] - self.prev_ball[X]) > abs(self.cur_posture[id][X] - self.cur_ball[X])
        y_dir = abs(self.cur_posture[id][Y] - self.prev_ball[Y]) > abs(self.cur_posture[id][Y] - self.cur_ball[Y])

        # ball is coming closer
        if (x_dir and y_dir):
            return True
        else:
            return False

    # check if the robot with id 'id' has a chance to shoot
    def shoot_chance(self, id):
        dx = self.cur_ball[X] - self.cur_posture[id][X]
        dy = self.cur_ball[Y] - self.cur_posture[id][Y]

        # if the ball is located further on left than the robot, it will be hard to shoot
        if (dx < 0):
            return False

        # if the robot->ball direction aligns with opponent's goal, the robot can shoot
        y = (self.field[X] / 2 - self.cur_ball[X]) * dy / dx + self.cur_posture[id][Y]
        if (abs(y) < self.goal[Y] / 2):
            return True
        else:
            return False

    @inlineCallbacks
    def on_event(self, f):

        @inlineCallbacks
        def set_wheel(self, robot_wheels):
            yield self.call(u'aiwc.set_speed', args.key, robot_wheels)
            return

        # a basic goalkeeper rulbased algorithm
        def goalkeeper(self, id):

            # self.set_target_position(id, 1, 1, 1.4, 5.0, 0.4, False)
            # return
            # default desired position
            x = (-self.field[X] / 2) + (self.robot_size[id] / 2) + 0.05
            y = max(min(self.cur_ball[Y], (self.goal[Y] / 2 - self.robot_size[id] / 2)),
                    -self.goal[Y] / 2 + self.robot_size[id] / 2)

            # if the robot is inside the goal, try to get out
            if (self.cur_posture[id][X] < -self.field[X] / 2):
                if (self.cur_posture[id][Y] < 0):
                    self.set_target_position(id, x, self.cur_posture[id][Y] + 0.2, 1.4, 5.0, 0.4, False)
                else:
                    self.set_target_position(id, x, self.cur_posture[id][Y] - 0.2, 1.4, 5.0, 0.4, False)
            # if the goalkeeper is outside the penalty area
            elif (not self.in_penalty_area(self.cur_posture[id], MY_TEAM)):
                # return to the desired position
                self.set_target_position(id, x, y, 1.4, 5.0, 0.4, True)
            # if the goalkeeper is inside the penalty area
            else:
                # if the ball is inside the penalty area
                if (self.in_penalty_area(self.cur_ball, MY_TEAM)):
                    # if the ball is behind the goalkeeper
                    if (self.cur_ball[X] < self.cur_posture[id][X]):
                        # if the ball is not blocking the goalkeeper's path
                        if (abs(self.cur_ball[Y] - self.cur_posture[id][Y]) > 2 * self.robot_size[id]):
                            # try to get ahead of the ball
                            self.set_target_position(id, self.cur_ball[X] - self.robot_size[id], self.cur_posture[id][Y], 1.4, 5.0,
                                          0.4, False)
                        else:
                            # just give up and try not to make a suicidal goal
                            self.angle(id, math.pi / 2)
                    # if the ball is ahead of the goalkeeper
                    else:
                        desired_th = self.direction_angle(id, self.cur_ball[X], self.cur_ball[Y])
                        rad_diff = helper.trim_radian(desired_th - self.cur_posture[id][TH])
                        # if the robot direction is too away from the ball direction
                        if (rad_diff > math.pi / 3):
                            # give up kicking the ball and block the goalpost
                            self.set_target_position(id, x, y, 1.4, 5.0, 0.4, False)
                        else:
                            # try to kick the ball away from the goal
                            self.set_target_position(id, self.cur_ball[X], self.cur_ball[Y], 1.4, 3.0, 0.8, True)
                # if the ball is not in the penalty area
                else:
                    # if the ball is within alert range and y position is not too different
                    if (self.cur_ball[X] < -self.field[X] / 2 + 1.5 * self.penalty_area[X] and abs(
                            self.cur_ball[Y]) < 1.5 * self.penalty_area[Y] / 2 and abs(
                            self.cur_ball[Y] - self.cur_posture[id][Y]) < 0.2):
                        self.face_specific_position(id, self.cur_ball[X], self.cur_ball[Y])
                    # otherwise
                    else:
                        self.set_target_position(id, x, y, 1.4, 5.0, 0.4, True)

        # a basic defender rulebased algorithm
        def defender(self, id):
            self.set_target_position(id, -6, 5, 1.4, 5.0, 0.4, False)
            return
            # # if the robot is inside the goal, try to get out
            # if (self.cur_posture[id][X] < -self.field[X] / 2):
            #     if (self.cur_posture[id][Y] < 0):
            #         self.set_target_position(id, -0.7 * self.field[X] / 2, self.cur_posture[id][Y] + 0.2, 1.4, 3.5, 0.6, False)
            #     else:
            #         self.set_target_position(id, -0.7 * self.field[X] / 2, self.cur_posture[id][Y] - 0.2, 1.4, 3.5, 0.6, False)
            #     return
            # # the defender may try to shoot if condition meets
            # if (id == self.def_idx and self.shoot_chance(id) and self.cur_ball[X] < 0.3 * self.field[X] / 2):
            #     self.set_target_position(id, self.cur_ball[X], self.cur_ball[Y], 1.4, 5.0, 0.4, True)
            #     return

            # # if this defender is closer to the ball than the other defender
            # if (id == self.def_idx):
            #     # ball is on our side
            #     if (self.cur_ball[X] < 0):
            #         # if the robot can push the ball toward opponent's side, do it
            #         if (self.cur_posture[id][X] < self.cur_ball[X] - self.ball_radius):
            #             self.set_target_position(id, self.cur_ball[X], self.cur_ball[Y], 1.4, 5.0, 0.4, True)
            #         else:
            #             # otherwise go behind the ball
            #             if (abs(self.cur_ball[Y] - self.cur_posture[id][Y]) > 0.3):
            #                 self.set_target_position(id, max(self.cur_ball[X] - 0.5, -self.field[X] / 2 + self.robot_size[id] / 2),
            #                               self.cur_ball[Y], 1.4, 3.5, 0.6, False)
            #             else:
            #                 self.set_target_position(id, max(self.cur_ball[X] - 0.5, -self.field[X] / 2 + self.robot_size[id] / 2),
            #                               self.cur_posture[id][Y], 1.4, 3.5, 0.6, False)
            #     else:
            #         self.set_target_position(id, -0.7 * self.field[X] / 2, self.cur_ball[Y], 1.4, 3.5, 0.4, False)
            # # if this defender is not closer to the ball than the other defender
            # else:
            #     # ball is on our side
            #     if (self.cur_ball[X] < 0):
            #         # ball is on our left
            #         if (self.cur_ball[Y] > self.goal[Y] / 2 + 0.15):
            #             self.set_target_position(id,
            #                           max(self.cur_ball[X] - 0.5, -self.field[X] / 2 + self.robot_size[id] / 2 + 0.1),
            #                           self.goal[Y] / 2 + 0.15, 1.4, 3.5, 0.4, False)
            #         # ball is on our right
            #         elif (self.cur_ball[Y] < -self.goal[Y] / 2 - 0.15):
            #             self.set_target_position(id,
            #                           max(self.cur_ball[X] - 0.5, -self.field[X] / 2 + self.robot_size[id] / 2 + 0.1),
            #                           -self.goal[Y] / 2 - 0.15, 1.4, 3.5, 0.4, False)
            #         # ball is in center
            #         else:
            #             self.set_target_position(id,
            #                           max(self.cur_ball[X] - 0.5, -self.field[X] / 2 + self.robot_size[id] / 2 + 0.1),
            #                           self.cur_ball[Y], 1.4, 3.5, 0.4, False)
            #     else:
            #         # ball is on right side
            #         if (self.cur_ball[Y] < 0):
            #             self.set_target_position(id, -0.7 * self.field[X] / 2,
            #                           min(self.cur_ball[Y] + 0.5, self.field[Y] / 2 - self.robot_size[id] / 2), 1.4,
            #                           3.5, 0.4, False)
            #         # ball is on left side
            #         else:
            #             self.set_target_position(id, -0.7 * self.field[X] / 2,
            #                           max(self.cur_ball[Y] - 0.5, -self.field[Y] / 2 + self.robot_size[id] / 2), 1.4,
            #                           3.5, 0.4, False)

        # a basic forward rulebased algorithm
        def forward(self, id):
            self.set_target_position(id, -5, 5, 1.4, 3.5, 0.6, False)
            return
            # if the robot is blocking the ball's path toward opponent side
            # if (self.cur_ball[X] > -0.3 * self.field[X] / 2 and self.cur_ball[X] < 0.3 * self.field[X] / 2 and
            #         self.cur_posture[id][X] > self.cur_ball[X] + 0.1 and abs(
            #                 self.cur_posture[id][Y] - self.cur_ball[Y]) < 0.3):
            #     if (self.cur_ball[Y] < 0):
            #         self.set_target_position(id, self.cur_posture[id][X] - 0.25, self.cur_ball[Y] + 0.75, 1.4, 3.0, 0.8, False)
            #     else:
            #         self.set_target_position(id, self.cur_posture[id][X] - 0.25, self.cur_ball[Y] - 0.75, 1.4, 3.0, 0.8, False)
            #     return

            # # if the robot can shoot from current position
            # if (id == self.atk_idx and self.shoot_chance(id)):
            #     pred_ball = self.predict_ball_location(2)
            #     self.set_target_position(id, pred_ball[X], pred_ball[Y], 1.4, 5.0, 0.4, True)
            #     return

            # # if the ball is coming toward the robot, seek for shoot chance
            # if (id == self.atk_idx and self.ball_coming_toward_robot(id)):
            #     dx = self.cur_ball[X] - self.prev_ball[X]
            #     dy = self.cur_ball[Y] - self.prev_ball[Y]
            #     pred_x = (self.cur_posture[id][Y] - self.cur_ball[Y]) * dx / dy + self.cur_ball[X]
            #     steps = (self.cur_posture[id][Y] - self.cur_ball[Y]) / dy

            #     # if the ball will be located in front of the robot
            #     if (pred_x > self.cur_posture[id][X]):
            #         pred_dist = pred_x - self.cur_posture[id][X]
            #         # if the predicted ball location is close enough
            #         if (pred_dist > 0.1 and pred_dist < 0.3 and steps < 10):
            #             # find the direction towards the opponent goal and look toward it
            #             goal_angle = self.direction_angle(id, self.field[X] / 2, 0)
            #             self.angle(id, goal_angle)
            #             return

            # # if this forward is closer to the ball than the other forward
            # if (id == self.atk_idx):
            #     if (self.cur_ball[X] > -0.3 * self.field[X] / 2):
            #         # if the robot can push the ball toward opponent's side, do it
            #         if (self.cur_posture[id][X] < self.cur_ball[X] - self.ball_radius):
            #             self.set_target_position(id, self.cur_ball[X], self.cur_ball[Y], 1.4, 5.0, 0.4, True)
            #         else:
            #             # otherwise go behind the ball
            #             if (abs(self.cur_ball[Y] - self.cur_posture[id][Y]) > 0.3):
            #                 self.set_target_position(id, self.cur_ball[X] - 0.2, self.cur_ball[Y], 1.4, 3.5, 0.6, False)
            #             else:
            #                 self.set_target_position(id, self.cur_ball[X] - 0.2, self.cur_posture[id][Y], 1.4, 3.5, 0.6, False)
            #     else:
            #         self.set_target_position(id, -0.1 * self.field[X] / 2, self.cur_ball[Y], 1.4, 3.5, 0.4, False)
            # # if this forward is not closer to the ball than the other forward
            # else:
            #     if (self.cur_ball[X] > -0.3 * self.field[X] / 2):
            #         # ball is on our right
            #         if (self.cur_ball[Y] < 0):
            #             self.set_target_position(id, self.cur_ball[X] - 0.25, self.goal[Y] / 2, 1.4, 3.5, 0.4, False)
            #         # ball is on our left
            #         else:
            #             self.set_target_position(id, self.cur_ball[X] - 0.25, -self.goal[Y] / 2, 1.4, 3.5, 0.4, False)
            #     else:
            #         # ball is on right side
            #         if (self.cur_ball[Y] < 0):
            #             self.set_target_position(id, -0.1 * self.field[X] / 2,
            #                           min(-self.cur_ball[Y] - 0.5, self.field[Y] / 2 - self.robot_size[id] / 2), 1.4,
            #                           3.5, 0.4, False)
            #         # ball is on left side
            #         else:
            #             self.set_target_position(id, -0.1 * self.field[X] / 2,
            #                           max(-self.cur_ball[Y] + 0.5, -self.field[Y] / 2 + self.robot_size[id] / 2), 1.4,
            #                           3.5, 0.4, False)

                #가장 빠른 공격수 공격
        def attack(self, id):
            self.face_specific_position(id, self.cur_ball[X], self.cur_ball[Y])
            self.set_target_position(id, self.cur_ball[X], self.cur_ball[Y], 1.4, 5.0, 0.4, False)
            # self.atk_idx may try to shoot if condition meets
            if (self.shoot_chance(id) and self.cur_ball[X] < 0.3 * self.field[X] / 2):
                self.set_target_position(id, self.cur_ball[X], self.cur_ball[Y], 1.4, 5.0, 0.4, True)

        #자살골용 주변 분산
        def go_away(self, id):
            self.set_target_position(id, 3, 2.5, 1.4, 5.0, 0.4, True)
            return

        # initiate empty frame
        if (self.end_of_frame):
            self.received_frame = Frame()
            self.end_of_frame = False
        received_subimages = []

        if 'time' in f:
            self.received_frame.time = f['time']
        if 'score' in f:
            self.received_frame.score = f['score']
        if 'reset_reason' in f:
            self.received_frame.reset_reason = f['reset_reason']
        if 'game_state' in f:
            self.received_frame.game_state = f['game_state']
        if 'ball_ownership' in f:
            self.received_frame.ball_ownership = f['ball_ownership']
        if 'half_passed' in f:
            self.received_frame.half_passed = f['half_passed']
        if 'subimages' in f:
            self.received_frame.subimages = f['subimages']
            for s in self.received_frame.subimages:
                received_subimages.append(SubImage(s['x'],
                                                   s['y'],
                                                   s['w'],
                                                   s['h'],
                                                   s['base64'].encode('utf8')))
            self.image.update_image(received_subimages)
        if 'coordinates' in f:
            self.received_frame.coordinates = f['coordinates']
        if 'EOF' in f:
            self.end_of_frame = f['EOF']

        if (self.end_of_frame):
            # to get the image at the end of each frame use the variable:
            # self.image.ImageBuffer

            if (self.received_frame.reset_reason != NONE):
                self.previous_frame = self.received_frame

            self.get_coord()
            self.find_closest_robot()

            if (self.received_frame.reset_reason == EPISODE_END):
                # EPISODE_END is sent instead of GAME_END when 'repeat' option is set to 'true'
                # to mark the end of episode
                # you can reinitialize the parameters, count the number of episodes done, etc. here

                # this example does not do anything at episode end
                pass

            if (self.received_frame.reset_reason == HALFTIME):
                # halftime is met - from next frame, self.received_frame.half_passed will be set to True
                # although the simulation switches sides,
                # coordinates and images given to your AI soccer algorithm will stay the same
                # that your team is red and located on left side whether it is 1st half or 2nd half

                # this example does not do anything at halftime
                pass

            ##############################################################################
            if (self.received_frame.game_state == STATE_DEFAULT):
                # robot functions in STATE_DEFAULT

                #go_away(self,0)

                # 골키퍼 다른구역에 보내기
                #self.set_target_position(0, -2, 2.5, 1.4, 5.0, 0.4, True)

                # 골키퍼역할 제대로하기
                # goalkeeper(self, 0)

                # 수비(1,2) 공격(3,4) 명령
                # defender(self, 1)
                # defender(self, 2)
                # forward(self, 3)
                # forward(self, 4)

                # 블루팀의 공격(1,공을 찾아 드리블 2.슛찬스가 났을시 슈팅시도)
                # attack(self,4)
                # attack(self,1)
                # attack(self,2)
                # attack(self,3)

                # 선수들 특정 영역으로 보내기
                # self.set_target_position(1, 3, 3, 1.4, 5.0, 0.4, True)
                # self.set_target_position(2, 3, 3, 1.4, 5.0, 0.4, True)
                # self.set_target_position(3, 3, 3, 1.4, 5.0, 0.4, True)
                # self.set_target_position(4, 3, 3, 1.4, 5.0, 0.4, True)

                self.printConsole("blue team : STATE_DEFAULT")

                ##특정 위치로 모든 로봇을 옮기기
                #goalkeeper(self, 0)

                #set_wheel(self, self.wheels)
                return
            ##############################################################################
            elif (self.received_frame.game_state == STATE_KICKOFF):
                #  if the ball belongs to my team, initiate kickoff
                # if (self.received_frame.ball_ownership):
                #     self.set_target_position(4, 0, 0, 1.4, 3.0, 0.4, False)

                # defender(self, 1)
                # defender(self, 2)
                # forward(self, 3)
                # forward(self, 4)

                set_wheel(self, self.wheels)
            ##############################################################################
            elif (self.received_frame.game_state == STATE_GOALKICK):
                # if the ball belongs to my team,
                # drive the goalkeeper to kick the ball
                if (self.received_frame.ball_ownership):
                    self.set_wheel_velocity(0, self.max_linear_velocity[0], self.max_linear_velocity[0], True)

                set_wheel(self, self.wheels)
            ##############################################################################
            elif (self.received_frame.game_state == STATE_CORNERKICK):
                # just play as simple as possible
#                goalkeeper(self, 0)
#                defender(self, 1)
#                defender(self, 2)
                forward(self, 3)
                forward(self, 4)

                set_wheel(self, self.wheels)
            ##############################################################################
            elif (self.received_frame.game_state == STATE_PENALTYKICK):
                # if the ball belongs to my team,
                # drive the forward to kick the ball
                if (self.received_frame.ball_ownership):
                    self.set_wheel_velocity(4, self.max_linear_velocity[0], self.max_linear_velocity[0], True)

                set_wheel(self, self.wheels)
            ##############################################################################
            if (self.received_frame.reset_reason == GAME_END):
                # (virtual finish() in random_walk.cpp)
                # save your data
                with open(args.datapath + '/result.txt', 'w') as output:
                    # output.write('yourvariables')
                    output.close()
                # unsubscribe; reset or leave
                yield self.sub.unsubscribe()
                try:
                    yield self.leave()
                except Exception as e:
                    self.printConsole("Error: {}".format(e))
            ##############################################################################

            self.end_of_frame = False
            self.previous_frame = self.received_frame

    def onDisconnect(self):
        if reactor.running:
            reactor.stop()


if __name__ == '__main__':

    try:
        unicode
    except NameError:
        # Define 'unicode' for Python 3
        def unicode(s, *_):
            return s


    def to_unicode(s):
        return unicode(s, "utf-8")


    parser = argparse.ArgumentParser()
    parser.add_argument("server_ip", type=to_unicode)
    parser.add_argument("port", type=to_unicode)
    parser.add_argument("realm", type=to_unicode)
    parser.add_argument("key", type=to_unicode)
    parser.add_argument("datapath", type=to_unicode)

    args = parser.parse_args()

    ai_sv = "rs://" + args.server_ip + ":" + args.port
    ai_realm = args.realm

    # create a Wamp session object
    session = Component(ComponentConfig(ai_realm, {}))

    # initialize the msgpack serializer
    serializer = MsgPackSerializer()

    # use Wamp-over-rawsocket
    runner = ApplicationRunner(ai_sv, ai_realm, serializers=[serializer])

    runner.run(session, auto_reconnect=False)

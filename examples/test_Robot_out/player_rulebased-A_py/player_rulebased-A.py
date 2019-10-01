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

            self.cur_count = 0
            self.end_count = 0
            self.prev_sender = None
            self.sender = None
            self.touch = [False,False,False,False,False]
            self.prev_receiver = None
            self.receiver = None
            self.def_idx = 0
            self.atk_idx = 0
            self.closest_order = []
            self.player_state = [None,None,None,None,None]

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

    # check if sender/receiver pair should be reset
    def reset_condition(self) :
        # if the time is over, setting is reset
        if (self.end_count > 0 and self.end_count - self.cur_count < 0) :
            return True

        # if there is no sender and receiver is not in shoot chance, setting is cleared
        if not self.sender is None :
            if not self.shoot_chance(self.sender) :
                return True
        return False

    # check if a sender can be selected
    def set_sender_condition(self) :
        for i in range(1,5) :
            # if this robot is near the ball, it will be a sender candidate
            dist = helper.dist(self.cur_posture[i][X], self.cur_ball[X], self.cur_posture[i][Y], self.cur_ball[Y])
            if dist < 0.5 and self.cur_posture[i][ACTIVE]: return True
        return False

    # check if a receiver should be selected
    def set_receiver_condition(self) :
        # if a sender exists, any other robots can be receiver candidates
        if self.sender != None and self.receiver == None: return True
        return False

    # select a sender
    def set_sender(self, _player_list):
        distance_list = []
        for sender in _player_list :
            predict_ball = self.predict_ball_location(3)
            ball_distance = helper.dist(predict_ball[X], self.cur_posture[sender][X], predict_ball[Y], self.cur_posture[sender][Y])
            distance_list.append(ball_distance)

        # if the distance between ball and sender is less than 1, choose the closest robot as the sender
        if min(distance_list) < 1.0 :
            return distance_list.index(min(distance_list)) + 1

        # otherwise, there is no sender
        return None

    # select a receiver
    def set_receiver(self, _player_list):
        receiver_op_dist_list = []
        for receiver in _player_list :
            temp_receiver_op_dist_list = []
            # the sender is not a receiver candidate
            if receiver == self.sender :
                receiver_op_dist_list.append(999)
                continue

            # the distance between the robot and opponents
            for op in range(1, 5) : #[1,2,3,4]
                op_distance = helper.dist(self.cur_posture[receiver][X], self.cur_posture_op[op][X], self.cur_posture[receiver][Y], self.cur_posture_op[op][Y])
                temp_receiver_op_dist_list.append(op_distance)

            # save the shortest distance between this robot and one of opponents
            receiver_op_dist_list.append(min(temp_receiver_op_dist_list))

        receiver_ball_list = []
        for r in receiver_op_dist_list :
            # if the minimum distance between player and opponent's player is less than 0.5, this robot cannot be receiver
            if r < 0.5 or r == 999:
                receiver_ball_list.append(999)
                continue
            id = receiver_op_dist_list.index(r) + 1
            receiver_ball_distance = helper.dist(self.cur_ball[X], self.cur_posture[id][X], self.cur_ball[Y], self.cur_posture[id][Y])
            receiver_ball_list.append(receiver_ball_distance)

        if min(receiver_ball_list) < 999 :
            min_id = receiver_ball_list.index(min(receiver_ball_list)) + 1
            return min_id
        return None

    def pass_ball(self):
        #print("pass_ ball()!!!!!!!!!@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
        if self.prev_sender == self.receiver or self.prev_receiver == self.sender :# and not None in [self.prev_sender, self.prev_receiver, self.sender, self.receiver] :
            self.sender = self.prev_sender
            self.receiver = self.prev_receiver

        self.receive_ball()
        self.send_ball()

        self.prev_sender = self.sender
        self.prev_receiver = self.receiver

    def send_ball(self) :
        if self.sender == None :
            self.printConsole("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            self.face_specific_position(3, self.cur_ball[X], self.cur_ball[Y])
            self.set_target_position(3, self.cur_ball[X], self.cur_ball[Y], 1.4, 5.0, 0.4, False)
            self.actions(3, 'backward')
            return

        goal_dist = helper.dist(4.0, self.cur_posture[self.sender][X], 0, self.cur_posture[self.sender][Y])
        # if the sender has a shoot chance, it tries to shoot
        if self.shoot_chance(self.sender) :
            if goal_dist > 0.3 * self.field[X] / 2:
                self.actions(self.sender, 'dribble',refine=True)
                return
            else :
                ##self.actions(self.sender, 'kick')
                self.actions(self.sender, 'dribble')
                return

        # if the receiver exists, get the distance between the sender and the receiver
        sender_receiver_dist = None
        if not self.receiver == None :
            sender_receiver_dist = helper.dist(self.cur_posture[self.sender][X], self.cur_posture[self.receiver][X],self.cur_posture[self.sender][Y], self.cur_posture[self.receiver][Y])

        # # # if the sender is close to the receiver, the sender kicks the ball
        if not sender_receiver_dist == None :
            if sender_receiver_dist < 0.3 and not self.cur_posture[self.receiver][TOUCH]:
                ##self.actions(self.sender, 'kick')
                self.actions(self.sender, 'dribble')
                return

        # # ift, theta_diff = self.is_facing_target(self.sender, self.cur_ball[X], self.cur_ball[Y])
        # # if not ift :
        # #     # after the sender kicks, it stops
        # #     if theta_diff > math.pi * 3/4 :
        # #         self.actions(self.sender, None)
        # #         return
        # #     else :
        # #         self.actions(self.sender, 'follow',refine=True)
        # #         return

        # if the ball is in front of the sender and sender is moving backward
        if self.cur_posture[self.sender][X] < - 0.8 * self.field[X] / 2 :
            if self.cur_posture[self.sender][X] - self.prev_posture[self.sender][X] < 0 :
                self.actions(self.sender, 'backward')

        ##self.actions(self.sender, 'dribble',refine=True)
        return

    def receive_ball(self) :
        # if receiver does not exist, do nothing
        if self.receiver == None :
            self.printConsole("receiver doesn't exist")
            return

        goal_dist = helper.dist(4.0, self.cur_posture[self.receiver][X], 0, self.cur_posture[self.receiver][Y])
        # if sender is in shoot chance, receiver does nothing(reset)
        if self.shoot_chance(self.sender) :
            self.actions(self.receiver,None)
            return
        # if receiver is in shoot chance, receiver try to shoot
        if self.shoot_chance(self.receiver) :
            self.printConsole("reciever get a chance to shoot!!")
            if goal_dist > 0.3 * self.field[X] / 2:
                self.printConsole("Reciever try to dribble")
                self.actions(self.receiver, 'dribble',refine=True)
                return
            else :
                self.printConsole("Reciever try to kick")
                ##self.actions(self.receiver, 'kick')
                self.actions(self.receiver, 'dribble')
                return

        # if sender exists
        if not self.sender == None :
            s2risFace, _ = self.is_facing_target(self.sender, self.cur_posture[self.receiver][X], self.cur_posture[self.receiver][Y],4)
            r2sisFace, _ = self.is_facing_target(self.receiver, self.cur_posture[self.sender][X], self.cur_posture[self.sender][Y],4)
            # if sender and receiver directs each other
            if s2risFace and r2sisFace :
                if self.cur_posture[self.receiver][TH] > 0 or self.cur_posture[self.receiver][TH] < -3 :
                    self.actions(self.receiver,'follow', [self.prev_posture[self.receiver][X], self.prev_posture[self.receiver][Y] - 0.5 * self.field[Y]])
                    return
                self.actions(self.receiver, 'follow',[self.prev_posture[self.receiver][X], self.prev_posture[self.receiver][Y] + 0.5 * self.field[Y]])
                return

        r_point = self.cur_ball
        # if sender exists
        if not self.sender == None:
            r_point = self.receive_position()
        receiver_ball_dist = helper.dist(self.cur_ball[X], self.cur_posture[self.receiver][X], self.cur_ball[Y],self.cur_posture[self.receiver][Y])
        # if ball is close to receiver
        if receiver_ball_dist > 0.3 * self.field[X] / 2 :
            self.printConsole("ball is close to receiver")
            self.actions(self.receiver, 'follow', [r_point[X], r_point[Y]],refine=True)
            return

        r2bisFace, _ = self.is_facing_target(self.receiver, self.cur_ball[X], self.cur_ball[Y], 4)
        if not r2bisFace :
            self.printConsole("receiver is follow")
            self.actions(self.receiver, 'follow',refine=True)
            return
        # if receiver is moving to our goal area
        if self.cur_posture[self.receiver][X] < - 0.8 * self.field[X] / 2 :
            if self.cur_posture[self.receiver][X] - self.prev_posture[self.receiver][X] < 0 :
                self.printConsole("receiver is backward")
                self.actions(self.receiver, 'backward')

        ##self.actions(self.receiver, 'dribble')
        return

    # let robot with id 'id' execute an action directed by 'mode'
    def actions(self, id, mode = None, target_pts = None, params = None, refine = False) :
        if id == None :
            return

        # if the player state is set to 'stop', force the mode to be 'stop'
        if self.player_state[id] == 'stop' :
            mode = 'stop'

        if mode == None :
            # reset all robot status
            if self.sender == id :
                self.sender = None
                self.touch = [False, False, False, False, False]
            if self.receiver == id :
                self.receiver = None
            self.player_state[id] = None
            return
        if mode == 'follow' :
            # let the robot follow the ball
            if target_pts == None :
                target_pts = self.predict_ball_location(3)
            if params == None :
                params = [1.0, 1.0, 0.0, False]
            if refine :
                self.set_pos_parameters(id, target_pts, params)
            self.set_target_position(id, target_pts[X], target_pts[Y], params[0], params[1], params[2], params[3])
            self.player_state[id] = 'follow'
            return
        if mode == 'dribble' :
            # let the robot follow the ball but at a faster speed
            if target_pts == None :
                target_pts = self.cur_ball
            if params == None :
                params = [1.4, 10.0, 0.0, False]
            if refine :
                self.set_pos_parameters(id, target_pts, params)
            self.set_target_position(id, target_pts[X], target_pts[Y], params[0], params[1], params[2], params[3])
            self.player_state[id] = 'dribble'
            return
        if mode == 'kick' :
            # kick the ball
            # if target_pts == None :
            #     target_pts = self.cur_ball
            # if params == None :
            #     params = [1.4, 5.0, 0.8, True]
            # if self.end_count == 0 and not self.touch[id] :
            #     self.end_count = self.cur_count + 10 # 0.05 * cnt seconds
            # self.player_state[id] = 'kick'
            # if self.touch[id] :
            #     self.player_state[id] = 'stop'
            # if not self.touch[id] :
            #     self.touch[id] = self.cur_posture[id][TOUCH]
            # if self.player_state[id] == 'stop' :
            #     params = [0.0, 0.0, 0.0, False]
            target_pts = self.cur_ball
            params = [1.4, 5.0, 0.8, True]
            self.set_target_position(id, target_pts[X], target_pts[Y], params[0], params[1], params[2], params[3])
            return
        if mode == 'stop' :
            # stop while counter is on
            if params == None :
                params = [0.0, 0.0, False]
            self.set_wheel_velocity(id, params[0], params[1], params[2])
            if self.end_count == 0 :
                self.end_count = self.cur_count + 5 # 0.05 * cnt seconds
            self.player_state[id] = 'stop'
            if self.end_count - 1 == self.cur_count :
                self.player_state[id] = None
            return
        if mode == 'backward' :
            # retreat from the current position
            if target_pts == None :
                target_pts = [self.cur_posture[id][X] + 0.2, self.cur_posture[id][Y]]
            if params == None :
                params = [1.4, 5.0, 0.8, False]
            if refine :
                self.set_pos_parameters(id, target_pts, params)
            self.set_target_position(id, target_pts[X], target_pts[Y], params[0], params[1], params[2], params[3])
            self.player_state[id] = 'backward'
            return
        if mode == 'position' :
            # go toward target position
            self.set_target_position(id, target_pts[X], target_pts[Y], params[0], params[1], params[2], params[3])
            return

    def set_pos_parameters(self,id,target_pts,params,mult = 1.2):
        prev_dist = helper.dist(self.prev_posture[id][X],target_pts[X],self.prev_posture[id][Y],target_pts[Y])
        cur_dist = helper.dist(self.cur_posture[id][X],target_pts[X],self.cur_posture[id][Y],target_pts[Y])
        if cur_dist > prev_dist - 0.02 :
            params = [params[0] * mult, params[1] * mult, params[2] * mult, params[3]]
        return params

    def is_facing_target(self, id, x, y, div = 4):
        dx = x - self.cur_posture[id][X]
        dy = y - self.cur_posture[id][Y]
        ds = math.sqrt(dx * dx + dy * dy)
        desired_th = (self.cur_posture[id][TH] if (ds == 0) else math.acos(dx / ds))

        theta = self.cur_posture[id][TH]
        if desired_th < 0:
            desired_th += math.pi * 2
        if theta < 0:
            theta += math.pi * 2
        diff_theta = abs(desired_th - theta)
        if diff_theta > math.pi:
            diff_theta = min(diff_theta, math.pi * 2 - diff_theta)
        if diff_theta < math.pi / div or diff_theta > math.pi * (1 -  1 / div):
            return [True, diff_theta]
        return [False, diff_theta]

    def receive_position(self):
        step = 5
        ball_receiver_dist = helper.dist(self.cur_ball[X], self.cur_posture[self.receiver][X], self.cur_ball[Y],
                                             self.cur_posture[self.receiver][Y])
        prev_ball_receiver_dist = helper.dist(self.prev_ball[X], self.prev_posture[self.receiver][X],
                                                  self.prev_ball[Y], self.prev_posture[self.receiver][Y])

        diff_dist = prev_ball_receiver_dist - ball_receiver_dist
        if diff_dist > 0:
            step = ball_receiver_dist # diff_dist

        step = min(step, 15)

        predict_pass_point = self.predict_ball_location(step)

        ball_goal_dist = helper.dist(self.cur_ball[X], self.field[X] / 2, self.cur_ball[Y], 0)
        prev_ball_goal_dist = helper.dist(self.prev_ball[X], self.field[X] / 2, self.prev_ball[Y], 0)
        if ball_goal_dist > prev_ball_goal_dist:
            predict_pass_point[X] = predict_pass_point[X] - 0.15

        return predict_pass_point

    @inlineCallbacks
    def on_event(self, f):

        @inlineCallbacks
        def set_wheel(self, robot_wheels):
            yield self.call(u'aiwc.set_speed', args.key, robot_wheels)
            return

        # a basic goalkeeper rulbased algorithm
        def goalkeeper(self, id):
            # # default desired position

            # self.set_target_position(id, -2, 10, 1.4, 5.0, 0.4, False)
            # return

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

            # self.set_target_position(id, -2, 2, 1.4, 5.0, 0.4, False)
            # return

            # if the robot is inside the goal, try to get out
            if (self.cur_posture[id][X] < -self.field[X] / 2):
                if (self.cur_posture[id][Y] < 0):
                    self.set_target_position(id, -0.7 * self.field[X] / 2, self.cur_posture[id][Y] + 0.2, 1.4, 3.5, 0.6, False)
                else:
                    self.set_target_position(id, -0.7 * self.field[X] / 2, self.cur_posture[id][Y] - 0.2, 1.4, 3.5, 0.6, False)
                return
            # the defender may try to shoot if condition meets
            if (id == self.def_idx and self.shoot_chance(id) and self.cur_ball[X] < 0.3 * self.field[X] / 2):
                self.set_target_position(id, self.cur_ball[X], self.cur_ball[Y], 1.4, 5.0, 0.4, True)
                return

            # if this defender is closer to the ball than the other defender
            if (id == self.def_idx):
                # ball is on our side
                if (self.cur_ball[X] < 0):
                    # if the robot can push the ball toward opponent's side, do it
                    if (self.cur_posture[id][X] < self.cur_ball[X] - self.ball_radius):
                        self.set_target_position(id, self.cur_ball[X], self.cur_ball[Y], 1.4, 5.0, 0.4, True)
                    else:
                        # otherwise go behind the ball
                        if (abs(self.cur_ball[Y] - self.cur_posture[id][Y]) > 0.3):
                            self.set_target_position(id, max(self.cur_ball[X] - 0.5, -self.field[X] / 2 + self.robot_size[id] / 2),
                                          self.cur_ball[Y], 1.4, 3.5, 0.6, False)
                        else:
                            self.set_target_position(id, max(self.cur_ball[X] - 0.5, -self.field[X] / 2 + self.robot_size[id] / 2),
                                          self.cur_posture[id][Y], 1.4, 3.5, 0.6, False)
                else:
                    self.set_target_position(id, -0.7 * self.field[X] / 2, self.cur_ball[Y], 1.4, 3.5, 0.4, False)
            # if this defender is not closer to the ball than the other defender
            else:
                # ball is on our side
                if (self.cur_ball[X] < 0):
                    # ball is on our left
                    if (self.cur_ball[Y] > self.goal[Y] / 2 + 0.15):
                        self.set_target_position(id,
                                      max(self.cur_ball[X] - 0.5, -self.field[X] / 2 + self.robot_size[id] / 2 + 0.1),
                                      self.goal[Y] / 2 + 0.15, 1.4, 3.5, 0.4, False)
                    # ball is on our right
                    elif (self.cur_ball[Y] < -self.goal[Y] / 2 - 0.15):
                        self.set_target_position(id,
                                      max(self.cur_ball[X] - 0.5, -self.field[X] / 2 + self.robot_size[id] / 2 + 0.1),
                                      -self.goal[Y] / 2 - 0.15, 1.4, 3.5, 0.4, False)
                    # ball is in center
                    else:
                        self.set_target_position(id,
                                      max(self.cur_ball[X] - 0.5, -self.field[X] / 2 + self.robot_size[id] / 2 + 0.1),
                                      self.cur_ball[Y], 1.4, 3.5, 0.4, False)
                else:
                    # ball is on right side
                    if (self.cur_ball[Y] < 0):
                        self.set_target_position(id, -0.7 * self.field[X] / 2,
                                      min(self.cur_ball[Y] + 0.5, self.field[Y] / 2 - self.robot_size[id] / 2), 1.4,
                                      3.5, 0.4, False)
                    # ball is on left side
                    else:
                        self.set_target_position(id, -0.7 * self.field[X] / 2,
                                      max(self.cur_ball[Y] - 0.5, -self.field[Y] / 2 + self.robot_size[id] / 2), 1.4,
                                      3.5, 0.4, False)

        # a basic forward rulebased algorithm
        def forward(self, id):
            # if the robot is blocking the ball's path toward opponent side
            if (self.cur_ball[X] > -0.3 * self.field[X] / 2 and self.cur_ball[X] < 0.3 * self.field[X] / 2 and
                    self.cur_posture[id][X] > self.cur_ball[X] + 0.1 and abs(
                            self.cur_posture[id][Y] - self.cur_ball[Y]) < 0.3):
                if (self.cur_ball[Y] < 0):
                    self.set_target_position(id, self.cur_posture[id][X] - 0.25, self.cur_ball[Y] + 0.75, 1.4, 3.0, 0.8, False)
                else:
                    self.set_target_position(id, self.cur_posture[id][X] - 0.25, self.cur_ball[Y] - 0.75, 1.4, 3.0, 0.8, False)
                return

            # if the robot can shoot from current position
            if (id == self.atk_idx and self.shoot_chance(id)):
                pred_ball = self.predict_ball_location(2)
                self.set_target_position(id, pred_ball[X], pred_ball[Y], 1.4, 5.0, 0.4, True)
                return

            # if the ball is coming toward the robot, seek for shoot chance
            if (id == self.atk_idx and self.ball_coming_toward_robot(id)):
                dx = self.cur_ball[X] - self.prev_ball[X]
                dy = self.cur_ball[Y] - self.prev_ball[Y]
                pred_x = (self.cur_posture[id][Y] - self.cur_ball[Y]) * dx / dy + self.cur_ball[X]
                steps = (self.cur_posture[id][Y] - self.cur_ball[Y]) / dy

                # if the ball will be located in front of the robot
                if (pred_x > self.cur_posture[id][X]):
                    pred_dist = pred_x - self.cur_posture[id][X]
                    # if the predicted ball location is close enough
                    if (pred_dist > 0.1 and pred_dist < 0.3 and steps < 10):
                        # find the direction towards the opponent goal and look toward it
                        goal_angle = self.direction_angle(id, self.field[X] / 2, 0)
                        self.angle(id, goal_angle)
                        return

            # if this forward is closer to the ball than the other forward
            if (id == self.atk_idx):
                if (self.cur_ball[X] > -0.3 * self.field[X] / 2):
                    # if the robot can push the ball toward opponent's side, do it
                    if (self.cur_posture[id][X] < self.cur_ball[X] - self.ball_radius):
                        self.set_target_position(id, self.cur_ball[X], self.cur_ball[Y], 1.4, 5.0, 0.4, True)
                    else:
                        # otherwise go behind the ball
                        if (abs(self.cur_ball[Y] - self.cur_posture[id][Y]) > 0.3):
                            self.set_target_position(id, self.cur_ball[X] - 0.2, self.cur_ball[Y], 1.4, 3.5, 0.6, False)
                        else:
                            self.set_target_position(id, self.cur_ball[X] - 0.2, self.cur_posture[id][Y], 1.4, 3.5, 0.6, False)
                else:
                    self.set_target_position(id, -0.1 * self.field[X] / 2, self.cur_ball[Y], 1.4, 3.5, 0.4, False)
            # if this forward is not closer to the ball than the other forward
            else:
                if (self.cur_ball[X] > -0.3 * self.field[X] / 2):
                    # ball is on our right
                    if (self.cur_ball[Y] < 0):
                        self.set_target_position(id, self.cur_ball[X] - 0.25, self.goal[Y] / 2, 1.4, 3.5, 0.4, False)
                    # ball is on our left
                    else:
                        self.set_target_position(id, self.cur_ball[X] - 0.25, -self.goal[Y] / 2, 1.4, 3.5, 0.4, False)
                else:
                    # ball is on right side
                    if (self.cur_ball[Y] < 0):
                        self.set_target_position(id, -0.1 * self.field[X] / 2,
                                      min(-self.cur_ball[Y] - 0.5, self.field[Y] / 2 - self.robot_size[id] / 2), 1.4,
                                      3.5, 0.4, False)
                    # ball is on left side
                    else:
                        self.set_target_position(id, -0.1 * self.field[X] / 2,
                                      max(-self.cur_ball[Y] + 0.5, -self.field[Y] / 2 + self.robot_size[id] / 2), 1.4,
                                      3.5, 0.4, False)

        def default_rulebased(self, player_list):
            for p in player_list:
                # Add actions instead of default rulebase(goalkeeper, defender, forward) actions
                # If this robot is stuck at field sides, move forward the center
                if pow(self.prev_posture[p][X] - self.cur_posture[p][X],2) + pow(self.prev_posture[p][Y] - self.cur_posture[p][Y],2) < 5e-6:
                    if self.cur_posture[p][Y] > 0 :
                        self.set_target_position(p, 0, 0, 1.4, 3.5, 0.4, False)
                        continue
                if p == 0:
                    goalkeeper(self, 0)
                    continue
                if p == 1 or p == 2:
                    defender(self, p)
                    continue
                if p == 3 or p == 4:
                    forward(self, p)
                    continue

        def passing_play(self, player_list):
            def find_active_player(self, ids):
                _ids = []
                for i in ids:
                    if self.cur_posture[i][ACTIVE] :
                        _ids.append(i)
                return _ids
            # select only alive player
            _player_list = find_active_player(self, player_list)
            self.cur_count = round(self.received_frame.time * 20)  # count = 50 ms

            if self.end_count == self.cur_count :
                self.end_count = 0

            if self.reset_condition() :
                self.sender = None
                self.sender_touch = False
                self.receiver = None
            # check if sender exists
            if self.set_sender_condition() :
                self.printConsole("sender exists!!")
                self.sender = self.set_sender( _player_list)
            # check if receiver exists
            if self.set_receiver_condition():
                self.receiver = self.set_receiver(_player_list)

            if (self.sender != None and self.receiver != None):
                self.pass_ball()
                # if player is sender
                if self.sender in _player_list:
                    _player_list.remove(self.sender)
                # if player is receiver
                if self.receiver in _player_list:
                    _player_list.remove(self.receiver)

            default_rulebased(self, _player_list)
            return

        #패스용 주변으로 분산
        def go_away_pass(self, id):
            self.set_target_position(id, 3, 2, 1.4, 5.0, 0.4, True)
            return

        # 주변 분산
        def go_away(self, id):
            self.set_target_position(id, -3, 2.5, 1.4, 10.0, 0.4, True)
            return

        #자살골용 주변 분산
        def go_away_sucide(self, id):
            self.set_target_position(id, -1, 2.5, 1.4, 5.0, 0.4, True)
            return

        #가장 빠른 공격수 공격
        def attack(self, id):
            self.face_specific_position(id, self.cur_ball[X], self.cur_ball[Y])
            self.set_target_position(id, self.cur_ball[X], self.cur_ball[Y], 1.4, 10.0, 0.4, False)
            # self.atk_idx may try to shoot if condition meets
            if (self.shoot_chance(id) and self.cur_ball[X] < 0.3 * self.field[X] / 2):
               self.actions(id, 'backward')
               self.set_target_position(id, self.cur_ball[X], self.cur_ball[Y], 1.4, 10.0, 0.4, True)

######################################################## 상황 #######################################################

        #shooting directly
        def shoot_direct():
                attack(self,3)
                attack(self,4)
                goalkeeper(self,0)

        #자살골 상황
        def suicide_shoot():
                go_away_sucide(self,3)
                go_away_sucide(self,1)
                go_away_sucide(self,0)
                self.set_target_position(4, self.cur_ball[X], self.cur_ball[Y], 1.4, 5.0, 0.4, True)

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
            #self.printConsole(self.received_frame.game_state)
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
                self.printConsole("half time")
                # halftime is met - from next frame, self.received_frame.half_passed will be set to True
                # although the simulation switches sides,
                # coordinates and images given to your AI soccer algorithm will stay the same
                # that your team is red and located on left side whether it is 1st half or 2nd half

                # this example does not do anything at halftime
                pass
            ##############################################################################
            if (self.received_frame.game_state == STATE_DEFAULT):

                # robot functions in STATE_DEFAULT
                # goalkeeper simply executes goalkeeper algorithm on its own
                # goalkeeper(self, 0)

                    #self.actions(self.atk_idx, 'kick')

                # Deadlock 상황
                # set_wheel(self, self.wheels)만 주석

                # redteam blueteam 전체 공격
                # attack(self,1)
                # attack(self,2)
                # attack(self,3)
                # attack(self,4)

                #드리블
                #self.actions(4, 'dribble')

                ## 골넣기(장거리슈팅)
                # shoot_direct()
                # 유효슈팅 + 슈팅을 막는경우(골키퍼)
                # self.set_target_position(4, 3, 3, 1.4, 5.0, 0.4, True)
                # self.actions(3, 'backward')
                # self.actions(3, 'kick')

                ## Struggle & Steal
                #player_rulebased-B의 struggle 함수 활성화 시키기 + shoot_direct()
                #shoot_direct()

                ## Defence against oponent robots
                # defender(self, 1)
                # defender(self, 2)
                # defender(self, 3)
                # defender(self, 4)

                # 골넣기
                # shoot_direct()
                # 자살골
                # suicide_shoot()

                # 1대 1상황
                ## 상대팀 다른곳으로 보내기
                # attack(self,2)
                # self.set_target_position(3, -1, -3, 1.4, 5.0, 0.4, True)
                # self.set_target_position(4, -1, -3, 1.4, 5.0, 0.4, True)

                # 패스
                # # defender(self, 2)
                # # go_away_pass(self,1)
                # # go_away_pass(self,4)
                # # go_away_pass(self,3)

                # # player = 2
                # # measured_dist_pur = helper.dist(self.cur_ball[X], self.cur_posture[player][X], self.cur_ball[Y],self.cur_posture[player][Y])
                # # self.set_target_position(player, self.cur_ball[X], self.cur_ball[Y], 1.4, 5.0, 0.4, True)

                # # if(measured_dist_pur < 2):
                    
                # #     purpose = self.direction_angle(player, self.cur_ball[X], self.cur_ball[Y])
                # #     self.angle(player, purpose)

                # #     self.set_target_position(player, self.cur_ball[X], self.cur_ball[Y], 1.4, 5.0, 0.4, True)

                # defenders and forwards can pass ball to each other if necessary
                #passing_play(self, [1, 2, 3, 4])

                # 게임진행                
                # defender(self, 1)
                # forward(self, 2)
                # forward(self, 3)
                # forward(self, 4)

                # attack(self, 1)
                # attack(self, 2)
                # attack(self, 3)
                # attack(self, 4)

                # 슈팅하기 (단 첫슛은 실패함)
                # self.set_target_position(4, 3, 3, 1.4, 5.0, 0.4, True)
                # self.actions(3, 'backward')
                # self.actions(3, 'kick')

                # 액션별 정리
                #self.actions(1, 'kick')
                #self.actions(4, 'dribble')
                #self.actions(3, 'backward')
                #self.actions(3, 'kick')
                #self.actions(1, 'stop')

                # #다른곳으로 보내기
                # goalkeeper(self, 0)
                # go_away(self,1)
                # go_away(self,2)
                # go_away(self,3)
                # go_away(self,4)

                # 공 쫓아가기
                goalkeeper(self, 0)
                self.actions(1,'follow')
                self.actions(2,'follow')
                self.actions(3,'dribble')
                self.actions(4,'follow')
                # self.set_target_position(1, self.cur_ball[X], self.cur_ball[Y], 1.4, 1.5, 0.4, False)
                # self.set_target_position(2, self.cur_ball[X], self.cur_ball[Y], 1.4, 1.5, 0.4, False)
                # self.set_target_position(3, self.cur_ball[X], self.cur_ball[Y], 1.4, 1.5, 0.4, False)
                # self.set_target_position(4, self.cur_ball[X], self.cur_ball[Y], 1.4, 1.5, 0.4, False)


                # self.set_target_position(4, self.cur_ball[X], self.cur_ball[Y], 1.4, 10.0, 0.4, True)
                # self.actions(4,'kick')
                # self.set_target_position(3, self.cur_ball[X], self.cur_ball[Y], 1.4, 10.0, 0.4, True)
                # self.actions(3,'kick')

                self.printConsole("default state")
                #passing_play(self, [1,2,3])
                set_wheel(self, self.wheels)
            ##############################################################################
            elif (self.received_frame.game_state == STATE_KICKOFF):
                #  if the ball belongs to my team, initiate kickoff
                
                #if (self.received_frame.ball_ownership):
                    #self.set_target_position(4, 0, 0, 1.4, 3.0, 0.4, False)

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
                # goalkeeper(self, 0)
                # defender(self, 1)
                # defender(self, 2)
                # forward(self, 3)
                # forward(self, 4)

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

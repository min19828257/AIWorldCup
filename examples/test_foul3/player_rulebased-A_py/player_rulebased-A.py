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
STATE_DEFAULT = 0 #KICKOFF, GALKICK 등이 진행되고나서 바로 기본상태로 돌아간다. #게임 상태를 전체적으로 평가하기 위해 존재하는 변수.
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
# => 리스트를 딕셔너리처럼 사용할 수 있게 만드는 변수들

# TestCase
TestCase = 1

class Received_Image(object):
    def __init__(self, resolution, colorChannels):
        self.resolution = resolution
        self.colorChannels = colorChannels
        # need to initialize the matrix at timestep 0
        self.ImageBuffer = np.zeros((resolution[1], resolution[0], colorChannels))  # rows, columns, colorchann

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
            self.touch = [False, False, False, False,False]
            self.prev_receiver = None
            self.receiver = None
            self.def_idx = 0
            self.atk_idx = 0
            self.closest_order = []
            self.player_state = [None, None, None, None, None]

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
    #left, right wheel은 방향에만, 각도에만 영향을 주는지?
        # wheel velocities need to be scaled so that none of wheels exceed the maximum velocity available
        # otherwise, the velocity above the limit will be set to the max velocity by the simulation program
        # if that happens, the velocity ratio between left and right wheels will be changed that the robot may not execute
        # turning actions correctly.
        if (abs(left_wheel) > self.max_linear_velocity[id] or abs(right_wheel) > self.max_linear_velocity[
                id] or max_velocity):
            if abs(left_wheel) > abs(right_wheel):
                multiplier = self.max_linear_velocity[id] / abs(left_wheel)
            else:
                multiplier = self.max_linear_velocity[id] / abs(right_wheel)

        self.wheels[2 * id] = left_wheel * multiplier
        self.wheels[2 * id + 1] = right_wheel * multiplier

    # let the robot with id 'id' move to a target position (x, y)
    # the trajectory to reach the target position is determined by several different parameters
    
    def set_target_position(self, id, x, y, scale, mult_lin, mult_ang, max_velocity):
        #해당 값으로 이동한다는 것은 어쩄든, 초기좌표에 영향줄 수 있는 변수도 볼 수 있다는 게 아닐까?
        # => set_wheel_velocity 로 넘어가기 때문에 파악 불가.
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
        while d_th > math.pi:
            d_th -= 2 * math.pi
        while d_th < -math.pi:
            d_th += 2 * math.pi

        # based on how far the target position is, set a parameter that
        # decides how much importance should be put into changing directions
        # farther the target is, less need to change directions fastly
        if d_e > 1:
            ka = 17 / 90
        elif d_e > 0.5:
            ka = 19 / 90
        elif d_e > 0.3:
            ka = 21 / 90
        elif d_e > 0.2:
            ka = 23 / 90
        else:
            ka = 25 / 90

        # if the target position is at rear of the robot, drive backward instead
        if d_th > helper.d2r(95):
            d_th -= math.pi
            sign = -1
        elif d_th < helper.d2r(-95):
            d_th += math.pi
            sign = -1

        # if the direction is off by more than 85 degrees,
        # make a turn first instead of start moving toward the target
        if abs(d_th) > helper.d2r(85):
            self.set_wheel_velocity(id, -mult_ang * d_th, mult_ang * d_th, False)
        # otherwise
        else:
            # scale the angular velocity further down if the direction is off by less than 40 degrees
            if d_e < 5 and abs(d_th) < helper.d2r(40):
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

        self.atk_idx = min_idx #atk_idx를 정하는게 여기밖에 없는데 무조건 0값 아닌가? => x min_dix가 정하잖아. =>
        # min_idx는 0아니면 1의 값이므로 atk_dix는 플레이어리스트가 아니라, 0과 1만 들어간다. 근데 왜 id값이랑 비교하는거지?

        # record the robot closer to the ball between the two too
        self.closest_order = np.argsort(all_dist) + 1

    # predict where the ball will be located after 'steps' steps
    def predict_ball_location(self, steps): #스텝은 프레임 같은 것? => 여기서 어떻게 활용되는지, 던지는 값의 문맥은 어떤지를 봐야 알 수 있다.
        # 결국 이들도 문맥에 귀결되늰 요소와 동일하다.
        dx = self.cur_ball[X] - self.prev_ball[X] #현재 위치와 이전 위치의 볼좌표 X
        dy = self.cur_ball[Y] - self.prev_ball[Y] #현재 위치와 이전 위치의 볼좌표 Y
        return [self.cur_ball[X] + steps * dx, self.cur_ball[Y] + steps * dy] #현재볼좌표X + 스텝 * dx, 현재볼좌표Y + 스텝 * dy
        
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
        dx = self.cur_ball[X] - self.cur_posture[id][X] #현재볼위치와 플레이어 위치의 차
        dy = self.cur_ball[Y] - self.cur_posture[id][Y] 

        # if the ball is located further on left than the robot, it will be hard to shoot
        if (dx < 0): #X, Y는 좌표평면상 음수 양수 방향과 동일하므로 => 플레이어가 볼의 위치보다 오른쪽에 있다는 것을 의미한다.
            #반대로 말하면 볼의 소유권이 없거나, 공격을 당하고 있거나 등으로 해석할 수 있다.
            return False #True 값은 슛 찬스가 있다는 걸 얘기하는거겠지.

        # => 가장 첫 조건으로 와야 올바른 순서로 검증을 진행할 수 있음을 알려준다.

        # if the robot->ball direction aligns with opponent's goal, the robot can shoot
        y = (self.field[X] / 2 - self.cur_ball[X]) * dy / dx + self.cur_posture[id][Y] 
        #필드의 절반 / 2 - 현재볼X좌표) * Y좌표차 / X좌표차 + 현재플레이어 Y좌표
        
        if (abs(y) < self.goal[Y] / 2): #절대값y가 공 크기 세로절반보다 작다면,
            return True #슈팅찬스 반환
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
            # 현재 플레이어와 볼의 거리를 가져온다

            if dist < 0.5 and self.cur_posture[i][ACTIVE]: return True
            #ACTIVE가 소유권을 가지고 있는 얘 라면, 0.5보다 더 작은 값에 가까이 할 때 소유권을 인정함을 알 수 있지. #애(아이) 얘(이 아이) 라면,

        return False #샌더조건을 트루, 펄스로 리턴.

    # check if a receiver should be selected
    def set_receiver_condition(self) :
        # if a sender exists, any other robots can be receiver candidates
        if self.sender != None and self.receiver == None: return True
        return False

    # select a sender
    def set_sender(self, _player_list): #플레이어 리스트를 가져온다. 매개변수로 가져온 것들은 그 구조가 어떻든 다이렉트로 가리킨다고 보면된다.
        distance_list = [] #여기서 활용하기 위해 새롭게 만든 변수. #기초적인 것, 기본적인 것을 항상 베이스로두고 행동한다는 관점을 가지자.
        for sender in _player_list : #sender에 받은 플레이선수를 차례로 넣어준다
            #이렇게 보면 sender는 결국 player list에 의해 단순히 결정되는 것 뿐인데 특별한게 있나?

            predict_ball = self.predict_ball_location(3) #predict_ball_location은 볼에 관련된 특정 X, Y좌표를 리턴한다는 개념으로 설계되었다.
            ball_distance = helper.dist(predict_ball[X], self.cur_posture[sender][X], predict_ball[Y], self.cur_posture[sender][Y])
            #ㅎhelper.dist 함수에 predict_ball이 받은 [X], [Y] 인덱스와, sender의 X,Y 좌표를 보낸다.
            # helper.dist의 리턴값을 볼거리에 넣음. 
            distance_list.append(ball_distance) #거리 리스트에 ball 거리를 넣어준다. 거리리스트에는 뭐가 들어가는지는 모르지만,
            #순서없이 추가하는게 인상적.

        # if the distance between ball and sender is less than 1, choose the closest robot as the sender
        if min(distance_list) < 1.0 :
            return distance_list.index(min(distance_list)) + 1
        #sender를 모든 로봇이라고 단순히 보고, 그 중 가장 가까운 로봇을 샌더로 지정한다.
        # => 이렇게 보니 리팩토링이 완전한 코드가 아님을 알 수 있다. 샌더를 결정하는 코드에서 sender 변수를 사용했는데, 이는 플레이어 전체를 말하는거지
        # 샌더 자체를 지정하지 않는다. 그래서 실질적으로 정해지는 sender의 의미를 손실하여 직관성을 훼손한 사실에 이르렀다.

        # otherwise, there is no sender
        return None
        #sender의 리턴은 두 가지로 나뉜다.

    # select a receiver
    def set_receiver(self, _player_list): #player_list란 0~4 전 선수를 말하는 것. 그러한 선수들을 받기 위해 틀을 짜내었다.
        #실제로 이 함수를 쓸 때, 플레이어 리스트를 가리키는 변수를 써줘야겠지.
        receiver_op_dist_list = []
        for receiver in _player_list : #변수에 담긴 요소 하나하나에 원하는 과정을 대입하기 위함.
            temp_receiver_op_dist_list = []
            # the sender is not a receiver candidate
            if receiver == self.sender : #self.sender를 이해하기 위해서는 2가지 => 가르치는 용도의 방법으로 사용해왔지만, 항상 그럴 필요가 없다. 잠시 내려놓고 더 효율적인걸 사용하자.
                #sender를 정의 한 후에 receiver를 사용할 수 있다.
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
        if self.prev_sender == self.receiver or self.prev_receiver == self.sender :# and not None in [self.prev_sender, self.prev_receiver, self.sender, self.receiver] :
            self.sender = self.prev_sender
            self.receiver = self.prev_receiver

        self.receive_ball()
        self.send_ball()

        self.prev_sender = self.sender #prev_sender에 현재 sender인 self.sender를 집어넣는 이유는, prev_를 가리키기 위한 값이 cur-> self.sender에
        #있기 때문이다. => 매우 기초적이고 기본적이고 설명도 필요없는 내용이라 볼 수 있는 당연한 내용이지만, 기초적인 걸 표현하고 행동했을 때 새로운 게 탄생하거나,
        #기존 것의 농도를 더 증진시킨다.
        self.prev_receiver = self.receiver

    def send_ball(self) :
        if self.sender == None :
            return

        goal_dist = helper.dist(4.0, self.cur_posture[self.sender][X], 0, self.cur_posture[self.sender][Y])
        #4.0과 0은 무엇을 지칭하는지는 모르나, 현재 샌더와 거리를 리턴하여 goal_dist에 대입하고 있다.
        #두 점 사이의 거리를 알게되면 거리별로 특정 행동을 정의할 수 있겠지. => 내가 보는 활용관점

        # if the sender has a shoot chance, it tries to shoot
        if self.shoot_chance(self.sender) : #sender 속성을 슛 찬스에 보내고 트루 펄스를 판단 => sender는 공을 차고 있는 플레이어인가? 
            if goal_dist > 0.3 * self.field[X] / 2:
                self.actions(self.sender, 'dribble',refine=True)
                return
            else :
                self.actions(self.sender, 'kick')
                return

        # if the receiver exists, get the distance between the sender and the receiver
        sender_receiver_dist = None
        if not self.receiver == None :
            sender_receiver_dist = helper.dist(self.cur_posture[self.sender][X], self.cur_posture[self.receiver][X],self.cur_posture[self.sender][Y], self.cur_posture[self.receiver][Y])

        # if the sender is close to the receiver, the sender kicks the ball
        if not sender_receiver_dist == None :
            if sender_receiver_dist < 0.3 and not self.cur_posture[self.receiver][TOUCH]:
                self.actions(self.sender, 'kick')
                return

        ift, theta_diff = self.is_facing_target(self.sender, self.cur_ball[X], self.cur_ball[Y])
        if not ift :
            # after the sender kicks, it stops
            if theta_diff > math.pi * 3/4 :
                self.actions(self.sender, None)
                return
            else :
                self.actions(self.sender, 'follow',refine=True)
                return

        # if the ball is in front of the sender and sender is moving backward
        if self.cur_posture[self.sender][X] < - 0.8 * self.field[X] / 2 :
            if self.cur_posture[self.sender][X] - self.prev_posture[self.sender][X] < 0 :
                self.actions(self.sender, 'backward')

        self.actions(self.sender, 'dribble',refine=True)
        return

    def receive_ball(self) :
        # if receiver does not exist, do nothing
        if self.receiver == None : #리시버가 없다는 것은, 리시버로 결정될 만한 플레이어가 아무도 없다는 뜻이다. 그렇게 됬을 경우
            # 함수를 실행한다는 것은 자원낭비를 뜻하므로, return으로 되돌리게 된다.
            return

        goal_dist = helper.dist(4.0, self.cur_posture[self.receiver][X], 0, self.cur_posture[self.receiver][Y])
        # if sender is in shoot chance, receiver does nothing(reset)
        if self.shoot_chance(self.sender) :
            self.actions(self.receiver,None)
            return
        # if receiver is in shoot chance, receiver try to shoot
        if self.shoot_chance(self.receiver) :
            if goal_dist > 0.3 * self.field[X] / 2:
                self.actions(self.receiver, 'dribble',refine=True)
                return
            else :
                self.actions(self.receiver, 'kick')
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
            self.actions(self.receiver, 'follow', [r_point[X], r_point[Y]],refine=True)
            return

        r2bisFace, _ = self.is_facing_target(self.receiver, self.cur_ball[X], self.cur_ball[Y], 4)
        if not r2bisFace :
            self.actions(self.receiver, 'follow',refine=True)
            return
        # if receiver is moving to our goal area
        if self.cur_posture[self.receiver][X] < - 0.8 * self.field[X] / 2 :
            if self.cur_posture[self.receiver][X] - self.prev_posture[self.receiver][X] < 0 :
                self.actions(self.receiver, 'backward')

        self.actions(self.receiver, 'dribble')
        return

    # let robot with id 'id' execute an action directed by 'mode'
    def actions(self, id, mode = None, target_pts = None, params = None, refine = False) :
        #set_target_position이 해석되니까 이 부분의 target_pts, params의 역할도 파악할 수 있다.
        if id == None :
            return

        # if the player state is set to 'stop', force the mode to be 'stop'
        if self.player_state[id] == 'stop' :
            mode = 'stop'

        if mode == 'test' : 
            # kick the ball
            if target_pts == None :
                target_pts = self.cur_ball #set_target_position 좌표를 위환 포지션
            if params == None :
                params = [1.4, 5.0, 0.8, True] # scale, mult_lin, mult_ang, max_velocity 를 정하는 요소
                #실제 게임에서 어느정도의 효과가 있는지는 모르겠다. 이 수치들이 어느정도의 영향을 미치는지, 어떤 그림을 그려내는지에 대한
            if self.end_count == 0 and not self.touch[id] :
                self.end_count = self.cur_count + 10 # 0.05 * cnt seconds

            #player_state[id]에 각 플레이어의 상태를 지칭하는 속성을 담음.
            self.player_state[id] = 'kick'
            if self.touch[id] :
                self.player_state[id] = 'stop'
            if not self.touch[id] :
                self.touch[id] = self.cur_posture[id][TOUCH]
            if self.player_state[id] == 'stop' :
                params = [0.0, 0.0, 0.0, False]
            self.set_target_position(id, target_pts[X], target_pts[Y], params[0], params[1], params[2], params[3])
            #target_pts에 현재 볼 위치를 담아서 보내는 것.
            return

        if mode == None :
            # reset all robot status
            if self.sender == id : #현재 샌더가 이 함수를 호출한 매개변수 값과 동일하다면,
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
                params = [1.0, 3.0, 0.6, False]
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
                params = [1.4, 5.0, 0.8, False]
            if refine :
                self.set_pos_parameters(id, target_pts, params)
            self.set_target_position(id, target_pts[X], target_pts[Y], params[0], params[1], params[2], params[3])
            self.player_state[id] = 'dribble'
            return
        if mode == 'kick' :
            # kick the ball
            if target_pts == None :
                target_pts = self.cur_ball #set_target_position 좌표를 위환 포지션
            if params == None :
                params = [1.4, 5.0, 0.8, True] # scale, mult_lin, mult_ang, max_velocity 를 정하는 요소
                #실제 게임에서 어느정도의 효과가 있는지는 모르겠다. 이 수치들이 어느정도의 영향을 미치는지, 어떤 그림을 그려내는지에 대한
            if self.end_count == 0 and not self.touch[id] :
                self.end_count = self.cur_count + 10 # 0.05 * cnt seconds
            self.player_state[id] = 'kick'
            if self.touch[id] :
                self.player_state[id] = 'stop'
            if not self.touch[id] :
                self.touch[id] = self.cur_posture[id][TOUCH]
            if self.player_state[id] == 'stop' :
                params = [0.0, 0.0, 0.0, False]
            self.set_target_position(id, target_pts[X], target_pts[Y], params[0], params[1], params[2], params[3])
            #킥은 무조건 현재 볼의 좌표로 보내고 시작한다.
            return
        if mode == 'stop' :
            # stop while counter is on
            if params == None :
                params = [0.0, 0.0, False]
            self.set_wheel_velocity(id, params[0], params[1], params[2])
            #set target position에서 target은 플레이어를 대상으로 지목한게 아니라, 좌표공간 자체를 지목한 의미였다.
            #갑자기 이걸보니 확 떠오른 중의적 차이를 알 수 있었다. 신기하게도.. 왜지? 큰 관련성은 없어보이는데 말이지.
            #결국 기초와 기본이 이러한 연관성을 끌어당기는 핵심임을 알 수 있네. 기본기를 다스리자.
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
    def on_event(self, f): #어느 변수든 on_event에 f를 넘겨주네, f를 넘겨준다고 해당 변수의 이름이 f는 아니니까, on_event를 찾아봐야 알 수 있다.
#on_event(로 검색이 안된다는 것은, 다른 언어에서 이 함수를 호출하는 틀이 다르다는 것을 알 수 있다. => 그렇다면 on_event 자체로 검색
        @inlineCallbacks
        def set_wheel(self, robot_wheels):
            yield self.call(u'aiwc.set_speed', args.key, robot_wheels)
            return



        # a basic goalkeeper rulbased algorithm
        def goalkeeper(self, id):
            
            #id, left_wheel, right_wheel, max_velocity
            #초기좌표를 건드릴 수 없으면, 최대 속도를 줘서 초기좌표와 같은 효과를 주면 된다. => 원하는 대로 안움직인다.

            #default desired position
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
                #1. soccer filed를 반으로 나누어야 중간 지점에서 계산처리를 할 수 있다. => 일반적으로 필드를 활용하여 뭔가를 계산할 때는 양 끝보다
                #중앙좌표가 포커싱되기 마련이다.
                #2. 중간을 나누어 -0.3을 곱했다는 것은 왼쪽 공간의 30% 영역을 가지고 계산하는 것을 의미. 뒤에 and는 오른쪽 30%
                #필드의 x와 y에 음수값은 존재하지 않지만, 볼과 플레이어는 x와 y를 기준으로 움직인다. 볼과 플레이어의 x, y는 음수 양수로 절반을 활용하기 때문에
                #각기 절대값으로 변경 후 합해야만 필드 x와 y의 값과 동일선상에 위치해있다고 말할 수 있다. (동일하지 않다고 표현하는 것은 표현 단위가 다르기 때문에
                #동일한 양으로 칠 수 없기 때문이다.)
                #그 뒤의 and는 볼의 위치보다 앞섬을 의미. (red 기준) 그 뒤의 abs는 0아니면 무조건 1같은 자연수값이 나오는게 아닌가?
                #자연수라는 표현은 그 자체가 애매하므로 배제시켜 활용한다. => 내 기준에서 자연수가 의미를 가지려면 1부터 시작해야함. 고로 다시 쓴다.
        
    
                if (self.cur_ball[Y] < 0): #공이 아래에 가있다면,
                    self.set_target_position(id, self.cur_posture[id][X] - 0.25, self.cur_ball[Y] + 0.75, 1.4, 3.0, 0.8, False)
                    
                else:
                    self.set_target_position(id, self.cur_posture[id][X] - 0.25, self.cur_ball[Y] - 0.75, 1.4, 3.0, 0.8, False)
                return
                #이러한 조건들을 직관적으로 보려면, 각 조건을 이미지화시켜 이어그려 한 화면에 표시되게 하는 것.

            # if the robot can shoot from current position
            if (id == self.atk_idx and self.shoot_chance(id)): #어택하고 있는 아이디와 동일하다면? 은 디팬더와 공격수 4명 중 한명을 가려내기 위함?
                pred_ball = self.predict_ball_location(2)
                self.set_target_position(id, pred_ball[X], pred_ball[Y], 1.4, 5.0, 0.4, True)
                # 예상지역으로 움직인 후, 1.4 5.0 0.4 True 값을 주면 공격을 하나보다.
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
                        self.set_target_position(p, 0, 0, 1.4, 3.5, 0.4, False) #초기 패스에 관여하여 이동하는 식. (defender 1)
                        continue

                #플레이어 리스트에 따라 id 값을 지정하는 식.
                #특별하게 id값을 부여하여 뭘 가져오는 줄 알았더니, 그냥 플레이어 리스트를 p로 처리하여 담는 식이었다.
                #=> 여기는 한 프레임당 몇번 동작하고 가려나? => 동일한 라이프 사이클에서 0.05 프레임이 돌아간다면, 맞물리는데 이상이 없겠지.
                
                if p == 0: 
                    #goalkeeper(self, 0)
                    continue
                if p == 1 or p == 2:
                    #defender(self, p)
                    continue
                if p == 3 or p == 4:
                    #forward(self, p)
                    continue

        def passing_play(self, player_list):
            def find_active_player(self, ids):
                _ids = []
                for i in ids:
                    if self.cur_posture[i][ACTIVE] :
                        _ids.append(i)
                return _ids
            # select only alive player

            #함수안에 함수를 넣어서 코드 직관성을 높이려나보다.
            _player_list = find_active_player(self, player_list)
            self.cur_count = round(self.received_frame.time * 20)  # count = 50 ms

            if self.end_count == self.cur_count :
                self.end_count = 0

            if self.reset_condition() :
                self.sender = None
                self.sender_touch = False
                self.receiver = None


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
        if 'EOF' in f: #f안에 EOF가 들어있다면, (안정성을 위해 이 작업을 해주는듯. 굳이 안해줘도 작동은 잘 할 것이다.)
            self.end_of_frame = f['EOF']

        if (self.end_of_frame):
            # to get the image at the end of each frame use the variable:
            # self.image.ImageBuffer
            # 이 코드 영역이 0.05초마다 들어오는 프레임인지 확인하는 방법은 간단하다.
            # self.printConsole(self.received_frame.time) => 이 영역에서도 존재함을 알 수 있다. 그냥 위에서 보고 써도 되지만, 
            # 인텔리전스를 통해 더 빠른 체크가 가능.
    
            if (self.received_frame.reset_reason != NONE):
                self.previous_frame = self.received_frame
            
            self.get_coord()
            self.find_closest_robot()

            if (self.received_frame.reset_reason == EPISODE_END):
                # EPISODE_END is sent instead of GAME_END when 'repeat' option is set to 'true'
                # to mark the end of episode
                # you can reinitialize the parameters, count the number of episodes done, etc. here

                # this example does not do anything at episode end
                pass #아무 문장도 없으면 오류가 나오므로, 문장에 대한 널을 표현하는 역할.

            if (self.received_frame.reset_reason == HALFTIME):
                # halftime is met - from next frame, self.received_frame.half_passed will be set to True
                # although the simulation switches sides,
                # coordinates and images given to your AI soccer algorithm will stay the same
                # that your team is red and located on left side whether it is 1st half or 2nd half

                # this example does not do anything at halftime
                pass

            ##############################################################################
            if (self.received_frame.game_state == STATE_DEFAULT):
                pass
            # Foul 1 => 상대방에게 패널티킥 줌.
                # self.set_target_position(1, -3.5, 0, 1.4, 3.0, 0.4, False)
                # self.set_target_position(2, -3.5, 0, 1.4, 3.0, 0.4, False)
                # self.set_target_position(3, -3.5, 0, 1.4, 3.0, 0.4, False)

            # Foul 2 => 공이 패널티 라인안에, 적 로봇이 패널티 라인에 2마리 이상 -> 아군 골킥부여
                # self.set_target_position(1, -3, 3, 1.4, 10.0, 0.4, False)
                # self.set_target_position(2, -2, -3, 1.4, 10.0, 0.4, False)
                # self.set_target_position(3, -2, 3, 1.4, 10.0, 0.4, False)
                # self.set_target_position(4, 0, 3, 1.4, 10.0, 0.4, False)

            # Foul 3 => 공격수가 적 패널티 안에 1초 이상 들어가면 랜덤 재배치
                self.set_target_position(1, -3, 3, 1.4, 10.0, 0.4, False)
                self.set_target_position(2, -2, -3, 1.4, 10.0, 0.4, False)
                self.set_target_position(3, -2, 3, 1.4, 10.0, 0.4, False)
                

            # Foul 4 => 키퍼 벗어났을 때
                # self.set_target_position(0, 3, 3, 1.4, 3.0, 0.4, False)

            # Rule 1 => 공이 나갔을 때

                # self.set_target_position(3, -1, 3, 1.4, 4.0, 0.0, False)

                # if(self.received_frame.time > 3 and self.received_frame.time < 3.2):
                #     self.set_wheel_velocity(4, 0.0, 0.1, False) #해당각도로 위치를 잡겠다.
                    
                # elif(self.received_frame.time == 3.3):    
                #     self.set_wheel_velocity(4, 0.0, 0, False)

                # elif(self.received_frame.time >= 3.3 and self.received_frame.time <= 4.3):        
                #     self.set_target_position(4, 0, 0, 1.4, 4.0, 0.0, False) #앵글값은 그 각도로 해당포지션을 가겠다는 의미.
                    #앵글 값을 주지않으면, 이동시에 앞 뒤로만 움직이게 된다. 즉, 이동 시 각도를 정해주는 역할.
                # elif(self.received_frame.time >= 5.0 and self.received_frame.time <= 10.0):        
                #     self.set_target_position(4, self.cur_ball[X], self.cur_ball[Y], 1.4, 4.0, 0.0, False)

                    

                # self.set_target_position(2, 0, -1, 1.4, 10.0, 0.4, False)

            # Rule 2 => 플레이어가 넘어졌을 때
                # self.set_target_position(2, 0, -1, 1.4, 10.0, 0.4, False)


                # robot functions in STATE_DEFAULT
                # goalkeeper simply executes goalkeeper algorithm on its own
                # goalkeeper(self, 0)                
                # defenders and forwards can pass ball to each other if necessary
                # passing_play(self, [1, 2, 3, 4])
                # => 패싱플레이가 1 -> 2 -> 3 -> 4로 이어지는건가?
                set_wheel(self, self.wheels)
            ##############################################################################
            elif (self.received_frame.game_state == STATE_KICKOFF): #off라는 말과는 반대로 시작하다, 켜다라는 의미를 가진 KICK OFF
                
                # def set_target_position(self, id, x, y, scale, mult_lin, mult_ang, max_velocity):                
                # def set_wheel_velocity(self, id, left_wheel, right_wheel, max_velocity):

                #  if the ball belongs to my team, initiate kickoff
                if (self.received_frame.ball_ownership):                   
                    pass
                # Foul 1 
                    # self.set_target_position(4, 0, 0, 1.4, 2.3, 0.4, False)

                # Foul 2 
                    # self.set_target_position(4, 0, 0, 1.4, 2.0, 0.4, False)
                                        
                # Foul 3                     
                    if(self.received_frame.time > 0 and self.received_frame.time < 0.2):
                        self.set_wheel_velocity(4, 0.0, 0.7, False)
                        
                    elif(self.received_frame.time == 0.3):    
                        self.set_wheel_velocity(4, 0.0, 0, False)

                    elif(self.received_frame.time > 0.3):        
                        self.set_target_position(4, -0.2, 0, 1.4, 4.0, 0.0, False)                                    

                # Rule 1
                    # self.set_target_position(4, 0, 0, 1.4, 2.0, 0.4, False)                    

                # Rule 2 
                    # self.set_target_position(4, 0, 0, 1.4, 2.0, 0.4, False)
                    
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
                # 그냥 배치만하고 하던 대로 행동.
                goalkeeper(self, 0)
                defender(self, 1)
                defender(self, 2)
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

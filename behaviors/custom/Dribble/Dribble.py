import pickle
import numpy as np
from agent.Base_Agent import Base_Agent
from behaviors.custom.Dribble.Env import Env
from math_ops.math_ext import get_active_directory, vector_angle, normalize_deg
from math_ops.Neural_Network import run_mlp


class Dribble():

    def __init__(self, base_agent : Base_Agent) -> None:
        self.behavior = base_agent.behavior
        self.path_manager = base_agent.path_manager
        self.world = base_agent.world
        self.description = "RL dribble"
        self.auto_head = True
        self.env = Env(base_agent, 0.9 if self.world.robot.type == 3 else 1.2)

        with open(get_active_directory([
            "/behaviors/custom/Dribble/dribble_R0.pkl",
            "/behaviors/custom/Dribble/dribble_R1.pkl",
            "/behaviors/custom/Dribble/dribble_R2.pkl",
            "/behaviors/custom/Dribble/dribble_R3.pkl",
            "/behaviors/custom/Dribble/dribble_R4.pkl"
            ][self.world.robot.type]), 'rb') as f:
            self.model = pickle.load(f)

    def define_approach_orientation(self):

        w = self.world
        b = w.ball_abs_pos[:2]
        me = w.robot.loc_head_position[:2]

        self.approach_orientation = None

        MARGIN = 0.8 # safety margin (if ball is near the field limits by this amount, the approach orientation is considered)
        M90 = 90/MARGIN # auxiliary variable
        DEV = 25 # when standing on top of sidelines or endlines, the approach direction deviates from that line by this amount
        MDEV = (90+DEV)/MARGIN # auxiliary variable

        a1 = -180 # angle range start (counterclockwise rotation)
        a2 = 180  # angle range end   (counterclockwise rotation)

        if b[1] < -10 + MARGIN:
            if b[0] < -15 + MARGIN:
                a1 = DEV - M90 * (b[1]+10)
                a2 = 90 - DEV + M90 * (b[0]+15)
            elif b[0] > 15 - MARGIN:
                a1 = 90 + DEV - M90 * (15-b[0])
                a2 = 180 - DEV + M90 * (b[1]+10)
            else:
                a1 = DEV - MDEV * (b[1]+10)
                a2 = 180 - DEV + MDEV * (b[1]+10)
        elif b[1] > 10 - MARGIN:
            if b[0] < -15 + MARGIN:
                a1 = -90 + DEV - M90 * (b[0]+15)
                a2 = -DEV + M90 * (10-b[1])
            elif b[0] > 15 - MARGIN:
                a1 = 180 + DEV - M90 * (10-b[1])
                a2 = 270 - DEV + M90 * (15-b[0])
            else:
                a1 = -180 + DEV - MDEV * (10-b[1])
                a2 = -DEV + MDEV * (10-b[1])
        elif b[0] < -15 + MARGIN:
            a1 = -90 + DEV - MDEV * (b[0]+15)
            a2 = 90 - DEV + MDEV * (b[0]+15)
        elif b[0] > 15 - MARGIN and abs(b[1]) > 1.2:
            a1 = 90 + DEV - MDEV * (15-b[0])
            a2 = 270 - DEV + MDEV * (15-b[0])


        cad = vector_angle(b - me) # current approach direction

        a1 = normalize_deg(a1)
        a2 = normalize_deg(a2)

        if a1<a2:
            if a1 <= cad <= a2:
                return # current approach orientation is within accepted range
        else:
            if a1 <= cad or cad <= a2:
                return # current approach orientation is within accepted range

        a1_diff = abs(normalize_deg(a1 - cad))
        a2_diff = abs(normalize_deg(a2 - cad))

        self.approach_orientation = a1 if a1_diff < a2_diff else a2  # fixed normalized orientation


    def execute(self, reset, orientation, is_orientation_absolute, speed=1, stop=False):
        '''
        参数
        ----------
        orientation : float
            身体的绝对或相对朝向（相对于 imu_torso_orientation），单位为度。
            如果设置为 None，则向对方球门运球（此时忽略 is_orientation_absolute 参数）。
        is_orientation_absolute : bool
            如果为 True，则 orientation 是相对于场地的；如果为 False，则是相对于机器人的身体。
        speed : float
            速度，范围从 0 到 1（比例不是线性的）。
        stop : bool
            如果正在行走，立即返回 True；如果正在运球，则逐渐停止，并在可能时返回 True。
        '''
        w = self.world
        r = self.world.robot
        me = r.loc_head_position[:2]
        b = w.ball_abs_pos[:2]
        b_rel = w.ball_rel_torso_cart_pos[:2]
        b_dist = np.linalg.norm(b-me)
        behavior = self.behavior
        reset_dribble = False
        lost_ball = (w.ball_last_seen <= w.time_local_ms - w.VISUALSTEP_MS) or np.linalg.norm(b_rel)>0.4

        if reset:
            self.phase = 0
            if behavior.previous_behavior == "Push_RL" and 0<b_rel[0]<0.25 and abs(b_rel[1])<0.07:
                self.phase = 1
                reset_dribble = True

        if self.phase == 0: # walk to ball
            reset_walk = reset and behavior.previous_behavior != "Walk" and behavior.previous_behavior != "Push_RL" # reset walk if it wasn't the previous behavior

            #------------------------ 1. Decide if a better approach orientation is needed (when ball is nearly out of bounds)
            if reset or b_dist > 0.4: # stop defining orientation after getting near the ball to avoid noise
                self.define_approach_orientation()

            #------------------------ 2A. A better approach orientation is needed (ball is almost out of bounds)
            if self.approach_orientation is not None:
                next_pos, next_ori, dist_to_final_target = self.path_manager.get_path_to_ball(
                    x_ori=self.approach_orientation, x_dev=-0.24, torso_ori=self.approach_orientation, safety_margin=0.4)

                if b_rel[0]<0.26 and b_rel[0]>0.18 and abs(b_rel[1])<0.04 and w.ball_is_visible: # ready to start dribble
                    self.phase += 1
                    reset_dribble = True
                else:
                    dist = max(0.08, dist_to_final_target*0.7)
                    behavior.execute_sub_behavior("Walk", reset_walk, next_pos, True, next_ori, True, dist) # target, is_target_abs, ori, is_ori_abs, distance

            #------------------------ 2B. A better approach orientation is not needed but the robot cannot see the ball
            elif w.time_local_ms - w.ball_last_seen > 200: # walk to absolute target if ball was not seen
                abs_ori = vector_angle( b - me )
                behavior.execute_sub_behavior("Walk", reset_walk, b, True, abs_ori, True, None) # target, is_target_abs, ori, is_ori_abs, distance

            #------------------------ 2C. A better approach orientation is not needed and the robot can see the ball
            else: # walk to relative target
                if 0.18<b_rel[0]<0.25 and abs(b_rel[1])<0.05 and w.ball_is_visible: # ready to start dribble
                    self.phase += 1
                    reset_dribble = True
                else:
                    rel_target = b_rel+(-0.23,0)    # relative target is a circle (center: ball, radius:0.23m)
                    rel_ori = vector_angle(b_rel) # relative orientation of ball, NOT the target!
                    dist = max(0.08, np.linalg.norm(rel_target)*0.7) # slow approach
                    behavior.execute_sub_behavior("Walk", reset_walk, rel_target, False, rel_ori, False, dist) # target, is_target_abs, ori, is_ori_abs, distance

            if stop:
                return True

        if self.phase == 1 and (stop or (b_dist > 0.5 and lost_ball)): # go back to walking
            self.phase += 1
        elif self.phase == 1: # dribble
            #------------------------ 1. Define dribble parameters
            self.env.dribble_speed = speed

            # Relative orientation values are decreased to avoid overshoot
            if orientation is None:
                if b[0] < 0: # dribble to the sides
                    if b[1] > 0:
                        dribble_target = (15,5)
                    else:
                        dribble_target = (15,-5)
                else:
                    dribble_target = None # go to goal
                self.env.dribble_rel_orientation = self.path_manager.get_dribble_path(optional_2d_target=dribble_target)[1]
            elif is_orientation_absolute:
                self.env.dribble_rel_orientation = normalize_deg( orientation - r.imu_torso_orientation )
            else:
                self.env.dribble_rel_orientation = float(orientation) # copy if numpy float

            #------------------------ 2. Execute behavior
            obs = self.env.observe(reset_dribble)
            action = run_mlp(obs, self.model)
            self.env.execute(action)

        # wind down dribbling, and then reset phase
        if self.phase > 1:
            WIND_DOWN_STEPS = 60
            #------------------------ 1. Define dribble wind down parameters
            self.env.dribble_speed = 1 - self.phase/WIND_DOWN_STEPS
            self.env.dribble_rel_orientation = 0

            #------------------------ 2. Execute behavior
            obs = self.env.observe(reset_dribble, virtual_ball=True)
            action = run_mlp(obs, self.model)
            self.env.execute(action)

            #------------------------ 3. Reset behavior
            self.phase += 1
            if self.phase >= WIND_DOWN_STEPS - 5:
                self.phase = 0
                return True

        return False

    def is_ready(self):
        ''' Returns True if this behavior is ready to start/continue under current game/robot conditions '''
        return True

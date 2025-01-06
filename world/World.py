from collections import deque
from enum import IntEnum
import numpy as np

from cpp.modules import utils, localization
from logs.Logger import Logger
from math_ops.Matrix_4x4 import Matrix_4x4
from world.commons.Draw import Draw
from world.commons.Other_Robot import Other_Robot, RobotList
from world.Robot import Robot


class OurMode(IntEnum):
    "play mode in our favor"
    KICK_OFF = 0
    KICK_IN = 1
    CORNER_KICK = 2
    GOAL_KICK = 3
    FREE_KICK = 4
    PASS = 5
    DIR_FREE_KICK = 6
    GOAL = 7
    OFFSIDE = 8


class TheirMode(IntEnum):
    "play modes in their favor"
    KICKOFF = 9
    KICK_IN = 10
    CORNER_KICK = 11
    GOAL_KICK = 12
    FREE_KICK = 13
    PASS = 14
    DIR_FREE_KICK = 15
    GOAL = 16
    OFFSIDE = 17


class NeuMode(IntEnum):
    "neutral play modes"
    BEFORE_KICKOFF = 18
    GAME_OVER = 19
    PLAY_ON = 20


class PlayMode(IntEnum):
    "play mode groups"
    OUR_KICK = 0
    THEIR_KICK = 1
    ACTIVE_BEAM = 2
    PASSIVE_BEAM = 3
    OTHER = 4  # play on, game over


class World:
    STEPTIME = 0.02    # Fixed step time
    STEPTIME_MS = 20   # Fixed step time in milliseconds
    VISUALSTEP = 0.04  # Fixed visual step time
    VISUALSTEP_MS = 40  # Fixed visual step time in milliseconds

    FLAGS_CORNERS_POS = ((-15, -10, 0), (-15, +10, 0),
                         (+15, -10, 0), (+15, +10, 0))
    FLAGS_POSTS_POS = ((-15, -1.05, 0.8), (-15, +1.05, 0.8),
                       (+15, -1.05, 0.8), (+15, +1.05, 0.8))

    class Ball:
        """足球相关信息"""
        RelativeHeadSphPos = np.zeros(3)  # 相对头部的球的位置（球坐标系）
        RelativeHeadCartPos = np.zeros(3)  # 相对头部的球的位置（直角坐标系）
        RelativeTorsoCartPos = np.zeros(3)  # 相对躯干的球的位置（直角坐标系）
        RelativeTorsoCartPosHistory = deque(maxlen=20)  # 相对躯干的球的位置历史
        AbsolutePos = np.zeros(3)  # 绝对位置
        AbsolutePosHistory = deque(maxlen=20)  # 绝对位置历史
        AbsolutePosLastUpdate = 0  # 上次更新绝对位置的时间
        AbsoluteVel = np.zeros(3)  # 速度
        AbsoluteSpeed = 0  # 速度大小
        IsVisible = False  # 是否可见
        IsFromVision = False
        LastSeen = 0  # 上次看到球的时间
        CheatAbsPos = np.zeros(3)
        CheatAbsVel = np.zeros(3)
        Predicted2DPos = np.zeros((1, 2))
        Predicted2DVel = np.zeros((1, 2))
        Predicted2DSpeed = np.zeros(1)

    def __init__(self, robot_type: int, team_name: str, unum: int, apply_play_mode_correction: bool,
                 enable_draw: bool, logger: Logger, host: str) -> None:

        self.team_name = team_name
        self.team_name_opponent: str = None

        # 是否根据游戏模式调整球的位置
        self.apply_play_mode_correction = apply_play_mode_correction

        # 接收到的模拟步骤总数（总是和 self.time_total_ms 保持同步）
        self.step = 0

        # Time, in seconds, as indicated by the server (this time is NOT reliable, use only for synchronization between agents)
        self.time_server = 0.0
        # Reliable simulation time in milliseconds, use this when possible (it is incremented 20ms for every TCP message)
        self.time_local_ms = 0
        self.time_game = 0.0    # Game time, in seconds, as indicated by the server
        self.goals_scored = 0   # Goals score by our team
        self.goals_conceded = 0  # Goals conceded by our team
        # True if our team plays on the left side (this value is later changed by the world parser)
        self.team_side_is_left: bool = None
        # Play mode of the soccer game, provided by the server
        self.play_mode = None
        # Certain play modes share characteristics, so it makes sense to group them
        self.play_mode_group = None
        # corner flags, key=(x,y,z), always assume we play on the left side
        self.flags_corners: dict = None
        # goal   posts, key=(x,y,z), always assume we play on the left side
        self.flags_posts: dict = None

        # *at intervals of 0.02 s until ball comes to a stop or gets out of bounds (according to prediction)
        # Position of visible lines, relative to head, start_pos+end_pos (spherical coordinates) (m, deg, deg, m, deg, deg)
        self.lines = np.zeros((30, 6))
        self.line_count = 0                      # Number of visible lines
        # World.time_local_ms when last vision update was received
        self.vision_last_update = 0
        # True if the last server message contained vision information
        self.vision_is_up_to_date = False
        # List of teammates, ordered by unum
        self.teammates = RobotList(12, True)
        # self.teammates = [Other_Robot(i, True) for i in range(1, 12)]
        # List of opponents, ordered by unum
        self.opponents = RobotList(12, False)
        # self.opponents = [Other_Robot(i, False) for i in range(1, 12)]
        # This teammate is self
        self.teammates[unum-1].is_self = True
        # Draw object for current player
        self.draw = Draw(enable_draw, unum, host, 32769)
        # Draw object shared with teammates
        self.team_draw = Draw(enable_draw, 0, host, 32769)
        self.logger = logger
        self.robot = Robot(unum, robot_type)

    def log(self, msg: str):
        '''
        Shortcut for:

        self.logger.write(msg, True, self.step)

        Parameters
        ----------
        msg : str
            message to be written after the simulation step
        '''
        self.logger.write(msg, True, self.step)

    def get_ball_rel_vel(self, history_steps: int):
        '''
        Get ball velocity, relative to torso (m/s)

        Parameters
        ----------
        history_steps : int
            number of history steps to consider [1,20]

        Examples
        --------
        get_ball_rel_vel(1) is equivalent to (current rel pos - last rel pos)      / 0.04
        get_ball_rel_vel(2) is equivalent to (current rel pos - rel pos 0.08s ago) / 0.08
        get_ball_rel_vel(3) is equivalent to (current rel pos - rel pos 0.12s ago) / 0.12
        '''
        assert 1 <= history_steps <= 20, "Argument 'history_steps' must be in range [1,20]"

        if len(self.Ball.RelativeTorsoCartPosHistory) == 0:
            return np.zeros(3)

        h_step = min(history_steps, len(self.Ball.RelativeTorsoCartPosHistory))
        t = h_step * World.VISUALSTEP

        return (self.Ball.RelativeTorsoCartPos - self.Ball.RelativeTorsoCartPosHistory[h_step-1]) / t

    def get_ball_abs_vel(self, history_steps: int):
        '''
        Get ball absolute velocity (m/s)

        Parameters
        ----------
        history_steps : int
            number of history steps to consider [1,20]

        Examples
        --------
        get_ball_abs_vel(1) is equivalent to (current abs pos - last abs pos)      / 0.04
        get_ball_abs_vel(2) is equivalent to (current abs pos - abs pos 0.08s ago) / 0.08
        get_ball_abs_vel(3) is equivalent to (current abs pos - abs pos 0.12s ago) / 0.12
        '''
        assert 1 <= history_steps <= 20, "Argument 'history_steps' must be in range [1,20]"

        if len(self.Ball.AbsolutePosHistory) == 0:
            return np.zeros(3)

        h_step = min(history_steps, len(self.Ball.AbsolutePosHistory))
        t = h_step * World.VISUALSTEP

        return (self.Ball.AbsolutePos - self.Ball.AbsolutePosHistory[h_step-1]) / t

    def get_predicted_ball_pos(self, max_speed):
        '''
        Get predicted 2D ball position when its predicted speed is equal to or less than `max_speed`
        In case that position exceeds the prediction horizon, the last available prediction is returned

        Parameters
        ----------
        max_speed : float
            maximum speed at which the ball will be moving at returned future position
        '''
        b_sp = self.Ball.Predicted2DSpeed
        index = len(b_sp) - \
            max(1, np.searchsorted(b_sp[::-1], max_speed, side='right'))
        return self.Ball.Predicted2DPos[index]

    def get_intersection_point_with_ball(self, player_speed):
        '''
        Get 2D intersection point with moving ball, based on `self.ball_2d_pred_pos`

        Parameters
        ----------
        player_speed : float
            average speed at which the robot will chase the ball

        Returns
        -------
        2D intersection point : ndarray
            2D intersection point with moving ball, assuming the robot moves at an avg. speed of `player_speed`
        intersection distance : float
            distance between current robot position and intersection point
        '''

        params = np.array([*self.robot.location.Head.Position[:2],
                          player_speed*0.02, *self.Ball.Predicted2DPos.flat], np.float32)
        pred_ret = utils.get_intersection(params)
        return pred_ret[:2], pred_ret[2]

    def update(self):
        r = self.robot

        # reset variables
        r.location.is_up_to_date = False
        r.location.Head.head_z_is_up_to_date = False

        # update play mode groups

        if self.play_mode in (NeuMode.PLAY_ON, NeuMode.GAME_OVER):  # most common group
            self.play_mode_group = PlayMode.OTHER
        elif isinstance(self.play_mode, OurMode) and self.play_mode != OurMode.GOAL:
            self.play_mode_group = PlayMode.OUR_KICK
        elif isinstance(self.play_mode, TheirMode):
            if self.play_mode != TheirMode.GOAL:
                self.play_mode_group = PlayMode.THEIR_KICK
            else:
                self.play_mode_group = PlayMode.ACTIVE_BEAM
        elif self.play_mode == NeuMode.BEFORE_KICKOFF:
            self.play_mode_group = PlayMode.ACTIVE_BEAM
        elif self.play_mode == OurMode.GOAL:
            self.play_mode_group = PlayMode.PASSIVE_BEAM
        elif self.play_mode is not None:
            raise ValueError(f'Unexpected play mode ID: {self.play_mode}')

        r.update_pose()  # update forward kinematics

        if self.Ball.IsVisible:
            # Compute ball position, relative to torso
            self.Ball.RelativeTorsoCartPos = r.head_to_body_part_transform(
                "torso", self.Ball.RelativeHeadCartPos)

        if self.vision_is_up_to_date:  # update vision based localization

            # Prepare all variables for localization

            feet_contact = np.zeros(6)

            lf_contact = r.frp.get('lf', None)
            rf_contact = r.frp.get('rf', None)
            if lf_contact is not None:
                feet_contact[0:3] = Matrix_4x4(r.body_parts["lfoot"].transform).translate(
                    lf_contact[0:3], True).get_translation()
            if rf_contact is not None:
                feet_contact[3:6] = Matrix_4x4(r.body_parts["rfoot"].transform).translate(
                    rf_contact[0:3], True).get_translation()

            ball_pos = np.concatenate(
                (self.Ball.RelativeHeadCartPos, self.Ball.RelativeHeadCartPos))

            corners_list = [[key in self.flags_corners, 1.0, *key, *
                             self.flags_corners.get(key, (0, 0, 0))] for key in World.FLAGS_CORNERS_POS]
            posts_list = [[key in self.flags_posts, 0.0, *key, *
                           self.flags_posts.get(key, (0, 0, 0))] for key in World.FLAGS_POSTS_POS]
            all_landmarks = np.array(corners_list + posts_list, float)

            # Compute localization

            loc = localization.compute(
                r.feet_toes_are_touching['lf'],
                r.feet_toes_are_touching['rf'],
                feet_contact,
                self.Ball.IsVisible,
                ball_pos,
                r.cheat_abs_pos,
                all_landmarks,
                self.lines[0:self.line_count])

            r.update_localization(loc, self.time_local_ms)

            # Update self in teammates list (only the most useful parameters, add as needed)
            me = self.teammates[r.unum-1]
            me.state_last_update = r.location.last_update
            me.state_abs_pos = r.location.Head.Position
            # uses same criterion as for other teammates - not as reliable as player.behavior.is_ready("Get_Up")
            me.state_fallen = r.location.Head.Head_Z < 0.3
            me.state_orientation = r.location.Torso.Orientation
            # relevant for localization demo
            me.state_ground_area = (r.location.Head.Position[:2], 0.2)

            # Save last ball position to history at every vision cycle (even if not up to date)
            self.Ball.AbsolutePosHistory.appendleft(
                self.Ball.AbsolutePos)  # from vision or radio
            self.Ball.RelativeTorsoCartPosHistory.appendleft(
                self.Ball.RelativeTorsoCartPos)

            # Get ball position based on vision or play mode
            # Sources:
            # Corner kick position - rcssserver3d/plugin/soccer/soccerruleaspect/soccerruleaspect.cpp:1927 (May 2022)
            # Goal   kick position - rcssserver3d/plugin/soccer/soccerruleaspect/soccerruleaspect.cpp:1900 (May 2022)
            ball = None
            if self.apply_play_mode_correction:
                if self.play_mode == OurMode.CORNER_KICK:
                    ball = np.array(
                        [15, 5.483 if self.Ball.AbsolutePos[1] > 0 else -5.483, 0.042], float)
                elif self.play_mode == TheirMode.CORNER_KICK:
                    ball = np.array(
                        [-15, 5.483 if self.Ball.AbsolutePos[1] > 0 else -5.483, 0.042], float)
                elif self.play_mode in [
                        OurMode.KICK_OFF, TheirMode.KICKOFF, OurMode.GOAL, TheirMode.GOAL]:
                    ball = np.array([0, 0, 0.042], float)
                elif self.play_mode == OurMode.GOAL_KICK:
                    ball = np.array([-14, 0, 0.042], float)
                elif self.play_mode == TheirMode.GOAL_KICK:
                    ball = np.array([14, 0, 0.042], float)

                # Discard hard-coded ball position if robot is near that position (in favor of its own vision)
                if ball is not None and np.linalg.norm(r.location.Head.Position[:2] - ball[:2]) < 1:
                    ball = None

            if ball is None and self.Ball.IsVisible and r.location.is_up_to_date:
                ball = r.location.Head.ToFieldTransform(
                    self.Ball.RelativeHeadCartPos)
                ball[2] = max(ball[2], 0.042)  # lowest z = ball radius
                # for compatibility with tests without active soccer rules
                if self.play_mode != NeuMode.BEFORE_KICKOFF:
                    # force ball position to be inside field
                    ball[:2] = np.clip(ball[:2], [-15, -10], [15, 10])

            # Update internal ball position (also updated by Radio)
            if ball is not None:
                time_diff = (self.time_local_ms -
                             self.Ball.AbsolutePosLastUpdate) / 1000
                self.Ball.AbsoluteVel = (
                    ball - self.Ball.AbsolutePos) / time_diff
                self.Ball.AbsoluteSpeed = np.linalg.norm(self.Ball.AbsoluteVel)
                self.Ball.AbsolutePosLastUpdate = self.time_local_ms
                self.Ball.AbsolutePos = ball
                self.Ball.IsFromVision = True

            # Velocity decay for teammates and opponents (it is later neutralized if the velocity is updated)
            for p in self.teammates:
                p.state_filtered_velocity *= p.vel_decay
            for p in self.opponents:
                p.state_filtered_velocity *= p.vel_decay

            # Update teammates and opponents
            if r.location.is_up_to_date:
                for p in self.teammates:
                    if not p.is_self:                     # if teammate is not self
                        if p.is_visible:                  # if teammate is visible, execute full update
                            self.update_other_robot(p)
                        # otherwise update its horizontal distance (assuming last known position)
                        elif p.state_abs_pos is not None:
                            p.state_horizontal_dist = np.linalg.norm(
                                r.location.Head.Position[:2] - p.state_abs_pos[:2])

                for p in self.opponents:
                    if p.is_visible:                  # if opponent is visible, execute full update
                        self.update_other_robot(p)
                    # otherwise update its horizontal distance (assuming last known position)
                    elif p.state_abs_pos is not None:
                        p.state_horizontal_dist = np.linalg.norm(
                            r.location.Head.Position[:2] - p.state_abs_pos[:2])

        # Update prediction of ball position/velocity
        if self.play_mode_group != PlayMode.OTHER:  # not 'play on' nor 'game over', so ball must be stationary
            self.Ball.Predicted2DPos = self.Ball.AbsolutePos[:2].copy().reshape(
                1, 2)
            self.Ball.Predicted2DVel = np.zeros((1, 2))
            self.Ball.Predicted2DSpeed = np.zeros(1)

        # make new prediction for new ball position (from vision or radio)
        elif self.Ball.AbsolutePosLastUpdate == self.time_local_ms:

            params = np.array(
                [*self.Ball.AbsolutePos[:2], *np.copy(self.get_ball_abs_vel(6)[:2])], np.float32)
            pred_ret = utils.predict_rolling_ball(params)
            sample_no = len(pred_ret) // 5 * 2
            self.Ball.Predicted2DPos = pred_ret[:sample_no].reshape(-1, 2)
            self.Ball.Predicted2DVel = pred_ret[sample_no:sample_no *
                                                2].reshape(-1, 2)
            self.Ball.Predicted2DSpeed = pred_ret[sample_no*2:]

        # otherwise, advance to next predicted step, if available
        elif len(self.Ball.Predicted2DPos) > 1:
            self.Ball.Predicted2DPos = self.Ball.Predicted2DPos[1:]
            self.Ball.Predicted2DVel = self.Ball.Predicted2DVel[1:]
            self.Ball.Predicted2DSpeed = self.Ball.Predicted2DSpeed[1:]

        # update imu (must be executed after localization)
        r.update_imu(self.time_local_ms)

    def update_other_robot(self, other_robot: Other_Robot):
        ''' 
        Update other robot state based on the relative position of visible body parts
        (also updated by Radio, with the exception of state_orientation)
        '''
        o = other_robot
        r = self.robot

        # update body parts absolute positions
        o.state_body_parts_abs_pos = o.body_parts_cart_rel_pos.copy()
        for bp, pos in o.body_parts_cart_rel_pos.items():
            # Using the IMU could be beneficial if we see other robots but can't self-locate
            o.state_body_parts_abs_pos[bp] = r.location.Head.ToFieldTransform(
                pos, False)

        # auxiliary variables
        bps_apos = o.state_body_parts_abs_pos                 # read-only shortcut
        # list of body parts' 2D absolute positions
        bps_2d_apos_list = [v[:2] for v in bps_apos.values()]
        # 2D avg pos of visible body parts
        avg_2d_pt = np.average(bps_2d_apos_list, axis=0)
        head_is_visible = 'head' in bps_apos

        # evaluate robot's state (unchanged if head is not visible)
        if head_is_visible:
            o.state_fallen = bps_apos['head'][2] < 0.3

        # compute velocity if head is visible
        if o.state_abs_pos is not None:
            time_diff = (self.time_local_ms - o.state_last_update) / 1000
            if head_is_visible:
                # if last position is 2D, we assume that the z coordinate did not change, so that v.z=0
                old_p = o.state_abs_pos if len(o.state_abs_pos) == 3 else np.append(
                    o.state_abs_pos, bps_apos['head'][2])
                velocity = (bps_apos['head'] - old_p) / time_diff
                decay = o.vel_decay  # neutralize decay in all axes
            else:  # if head is not visible, we only update the x & y components of the velocity
                velocity = np.append(
                    (avg_2d_pt - o.state_abs_pos[:2]) / time_diff, 0)
                # neutralize decay (except in the z-axis)
                decay = (o.vel_decay, o.vel_decay, 1)
            # apply filter
            # otherwise assume it was beamed
            if np.linalg.norm(velocity - o.state_filtered_velocity) < 4:
                o.state_filtered_velocity /= decay  # neutralize decay
                o.state_filtered_velocity += o.vel_filter * \
                    (velocity-o.state_filtered_velocity)

        # compute robot's position (preferably based on head)
        if head_is_visible:
            # 3D head position, if head is visible
            o.state_abs_pos = bps_apos['head']
        else:
            o.state_abs_pos = avg_2d_pt  # 2D avg pos of visible body parts

        # compute robot's horizontal distance (head distance, or avg. distance of visible body parts)
        o.state_horizontal_dist = np.linalg.norm(
            r.location.Head.Position[:2] - o.state_abs_pos[:2])

        # compute orientation based on pair of lower arms or feet, or average of both
        lr_vec = None
        if 'llowerarm' in bps_apos and 'rlowerarm' in bps_apos:
            lr_vec = bps_apos['rlowerarm'] - bps_apos['llowerarm']

        if 'lfoot' in bps_apos and 'rfoot' in bps_apos:
            if lr_vec is None:
                lr_vec = bps_apos['rfoot'] - bps_apos['lfoot']
            else:
                lr_vec = (lr_vec + (bps_apos['rfoot'] - bps_apos['lfoot'])) / 2

        if lr_vec is not None:
            o.state_orientation = np.rad2deg(
                np.arctan2(lr_vec[1], lr_vec[0])) + 90

        # compute projection of player area on ground (circle)
        if o.state_horizontal_dist < 4:  # we don't need precision if the robot is farther than 4m
            max_dist = np.max(np.linalg.norm(
                bps_2d_apos_list - avg_2d_pt, axis=1))
        else:
            max_dist = 0.2
        o.state_ground_area = (avg_2d_pt, max_dist)

        # update timestamp
        o.state_last_update = self.time_local_ms

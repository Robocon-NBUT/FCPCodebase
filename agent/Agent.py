import numpy as np
from agent.Base_Agent import Base_Agent
from world.World import OurMode, NeuMode, PlayMode, TheirMode
from math_ops.math_ext import (
    vector_angle, normalize_deg, normalize_vec, target_abs_angle, target_rel_angle)


class Agent(Base_Agent):
    def __init__(self, host: str, agent_port: int, monitor_port: int, unum: int,
                 team_name: str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:

        # define robot type
        robot_type = (0, 1, 1, 1, 2, 3, 3, 3, 3, 3, 4)[unum-1]

        # Initialize base agent
        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw, play mode correction, Wait for Server, Hear Callback
        super().__init__(host, agent_port, monitor_port, unum, robot_type,
                         team_name, enable_log, enable_draw, True, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = 0  # 0-Normal, 1-Getting up, 2-Kicking
        self.kick_direction = 0
        self.kick_distance = 0
        self.fat_proxy_cmd = "" if is_fat_proxy else None
        # filtered walk parameters for fat proxy
        self.fat_proxy_walk = np.zeros(3)

        self.init_pos = ([-14, 0], [-9, -5], [-9, 0], [-9, 5], [-5, -5], [-5, 0], [-5, 5],
                         # initial formation
                         [-1, -6], [-1, -2.5], [-1, 2.5], [-1, 6])[unum-1]

    def beam(self, avoid_center_circle=False):
        r = self.world.robot
        pos = self.init_pos[:]  # copy position list
        self.state = 0

        # Avoid center circle by moving the player back
        if avoid_center_circle and np.linalg.norm(self.init_pos) < 2.5:
            pos[0] = -2.3

        if np.linalg.norm(pos - r.location.Head.position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
            # beam to initial position, face coordinate (0,0)
            self.server.commit_beam(pos, vector_angle((-pos[0], -pos[1])))
        else:
            if self.fat_proxy_cmd is None:  # normal behavior
                self.behavior.execute("Zero_Bent_Knees_Auto_Head")
            else:  # fat proxy behavior
                self.fat_proxy_cmd += "(proxy dash 0 0 0)"
                self.fat_proxy_walk = np.zeros(3)  # reset fat proxy walk

    def move(self, target_2d=(0, 0), orientation=None, is_orientation_absolute=True,
             avoid_obstacles=True, priority_unums=[], is_aggressive=False, timeout=3000):
        '''
        Walk to target position

        Parameters
        ----------
        target_2d : array_like
            2D target in absolute coordinates
        orientation : float
            absolute or relative orientation of torso, in degrees
            set to None to go towards the target (is_orientation_absolute is ignored)
        is_orientation_absolute : bool
            True if orientation is relative to the field, False if relative to the robot's torso
        avoid_obstacles : bool
            True to avoid obstacles using path planning (maybe reduce timeout arg if this function is called multiple times per simulation cycle)
        priority_unums : list
            list of teammates to avoid (since their role is more important)
        is_aggressive : bool
            if True, safety margins are reduced for opponents
        timeout : float
            restrict path planning to a maximum duration (in microseconds)    
        '''
        r = self.world.robot

        if self.fat_proxy_cmd is not None:  # fat proxy behavior
            self.fat_proxy_move(target_2d, orientation,
                                is_orientation_absolute)  # ignore obstacles
            return

        if avoid_obstacles:
            target_2d, _, distance_to_final_target = self.path_manager.get_path_to_target(
                target_2d, priority_unums=priority_unums, is_aggressive=is_aggressive, timeout=timeout)
        else:
            distance_to_final_target = np.linalg.norm(
                target_2d - r.location.Head.position[:2])

        # Args: target, is_target_abs, ori, is_ori_abs, distance
        self.behavior.execute("Walk", target_2d, True, orientation,
                              is_orientation_absolute, distance_to_final_target)

    def kick(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball

        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.server.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None:  # normal behavior
            # Basic_Kick has no kick distance control
            return self.behavior.execute("Basic_Kick", self.kick_direction, abort)
        else:  # fat proxy behavior
            return self.fat_proxy_kick()

    def kick_long(self, kick_direction=None, kick_distance=None, abort=False, enable_pass_command=False):
        '''
        Walk to ball and kick

        Parameters
        ----------
        kick_direction : float
            kick direction, in degrees, relative to the field
        kick_distance : float
            kick distance in meters
        abort : bool
            True to abort.
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        avoid_pass_command : bool
            When False, the pass command will be used when at least one opponent is near the ball

        Returns
        -------
        finished : bool
            Returns True if the behavior finished or was successfully aborted.
        '''

        if self.min_opponent_ball_dist < 1.45 and enable_pass_command:
            self.server.commit_pass_command()

        self.kick_direction = self.kick_direction if kick_direction is None else kick_direction
        self.kick_distance = self.kick_distance if kick_distance is None else kick_distance

        if self.fat_proxy_cmd is None:  # normal behavior
            # Basic_Kick has no kick distance control
            return self.behavior.execute("Kick_Long", self.kick_direction)
        else:  # fat proxy behavior
            return self.fat_proxy_kick()

    def dribble(self):
        '''
        Dribble to ball

        Parameters
        ----------
        orientation : float
            absolute or relative orientation of torso, in degrees
            set to None to dribble towards the ball (is_orientation_absolute is ignored)
        is_orientation_absolute : bool
            True if orientation is relative to the field, False if relative to the robot's torso
        '''
        if self.fat_proxy_cmd is None:  # normal behavior
            return self.behavior.execute("Dribble", None, True)
        else:  # fat proxy behavior
            self.fat_proxy_cmd += "(proxy dribble 0 0 0)"

    def think_and_send(self):
        w = self.world
        r = self.world.robot
        my_head_pos_2d = r.location.Head.position[:2]
        my_ori = r.imu_torso_orientation
        ball_2d = w.ball_abs_pos[:2]  # 球的二维坐标
        ball_vec = ball_2d - my_head_pos_2d  # 球相对于机器人头部的位置向量
        ball_dir = vector_angle(ball_vec)  # 球相对于机器人头部的角度
        ball_dist = np.linalg.norm(ball_vec)  # 球相对于机器人头部的距离
        ball_sq_dist = ball_dist * ball_dist  # 球距离的平方，用于加快比较速度
        ball_speed = np.linalg.norm(w.get_ball_abs_vel(6)[:2])  # 球的速度
        behavior = self.behavior  # 机器人行为对象
        goal_dir = target_abs_angle(ball_2d, (15.05, 0))  # 球门方向角度
        path_draw_options = self.path_manager.draw_options  # 路径绘制选项
        opposing_goal = (15.05, 0)  # 对方球门位置

        # --------------------------------------- 1. 预处理

        # 当球速<=0.5 m/s时预测的未来2D球位置
        slow_ball_pos = w.get_predicted_ball_pos(0.5)

        # 队友（包括自己）与慢球的距离平方的列表（在某些条件下距离平方被设置为1000）
        teammates_ball_sq_dist = [
            # 队友与球之间的距离平方
            np.sum((p.state_abs_pos[:2] - slow_ball_pos) ** 2)
            if p.state_last_update != 0 and (w.time_local_ms - p.state_last_update <= 360 or p.is_self) and not p.state_fallen
            # 如果队友不存在，或者状态信息不新（360毫秒），或者已经倒下，则强制设置为大距离
            else 1000
            for p in w.teammates]

        # 对手与慢球的距离平方的列表（在某些条件下距离平方被设置为1000）
        opponents_ball_sq_dist = [
            # 对手与球之间的距离平方
            np.sum((p.state_abs_pos[:2] - slow_ball_pos) ** 2)
            if p.state_last_update != 0 and w.time_local_ms - p.state_last_update <= 360 and not p.state_fallen
            # 如果对手不存在，或者状态信息不新（360毫秒），或者已经倒下，则强制设置为大距离
            else 1000
            for p in w.opponents]

        min_teammate_ball_sq_dist = min(teammates_ball_sq_dist)  # 最近的队友与球的距离平方
        self.min_teammate_ball_dist = np.sqrt(
            min_teammate_ball_sq_dist)  # 最近的队友与球的距离
        self.min_opponent_ball_dist = np.sqrt(
            min(opponents_ball_sq_dist))  # 最近的对手与球的距离

        # 最近的队友的编号
        active_player_unum = teammates_ball_sq_dist.index(
            min_teammate_ball_sq_dist) + 1

        # --------------------------------------- 2. 决定动作

        if w.play_mode == NeuMode.GAME_OVER:
            pass
        elif w.play_mode_group == PlayMode.ACTIVE_BEAM:
            self.beam()  # 主动传送
        elif w.play_mode_group == PlayMode.PASSIVE_BEAM:
            self.beam(True)  # 被动传送，避免进入中心圆
        elif self.state == 1 or (behavior.is_ready("Get_Up") and self.fat_proxy_cmd is None):
            # 如果站起来行为完成，则返回正常状态
            self.state = 0 if behavior.execute("Get_Up") else 1
        elif w.play_mode == OurMode.KICK_OFF:
            if r.unum == 9:
                # 如果游戏模式不是Play On，则无需更改状态
                self.kick(120, 3)
            else:
                self.move(self.init_pos, orientation=ball_dir)  # 原地行走
        elif w.play_mode == TheirMode.KICKOFF:
            self.move(self.init_pos, orientation=ball_dir)  # 原地行走
        elif active_player_unum != r.unum:  # 当前队员不是活跃球员
            if r.unum == 1:  # 当前球员是守门员
                self.move(self.init_pos, orientation=ball_dir)  # 原地行走
            elif r.unum in (2, 3, 4):  # 当前球员是后卫
                # 根据球的位置计算基本阵型位置
                new_x = max(0.5, (ball_2d[0]+15)/15) * \
                    (self.init_pos[0]+15) - 15
                self.move((new_x, self.init_pos[1]), orientation=ball_dir, priority_unums=[
                    active_player_unum])
            else:
                # # 根据球的位置计算基本阵型位置
                # new_x = max(0.5, (ball_2d[0]+15)/15) * \
                #     (self.init_pos[0]+15) - 15
                # if self.min_teammate_ball_dist < self.min_opponent_ball_dist:
                #     # 如果球队控球，则前进
                #     new_x = min(new_x + 3.5, 13)
                # self.move((new_x, self.init_pos[1]), orientation=ball_dir, priority_unums=[
                #     active_player_unum])
                if r.unum == 10:
                    if ball_2d[0] < 0:
                        self.move((ball_2d[0]+15, ball_2d[1]*15/(15-ball_2d[0])), orientation=ball_dir, priority_unums=[
                            active_player_unum])
                    else:
                        self.move((13, (13-ball_2d[0])*ball_2d[1]/(15-ball_2d[0])), orientation=ball_dir, priority_unums=[
                            active_player_unum])
                elif r.unum == 8:
                    self.move(((ball_2d[0]+opposing_goal[0])/2, ball_2d[1]/2), orientation=ball_dir, priority_unums=[
                        active_player_unum])
                else:
                    if r.unum % 2 == 0:
                        self.move((ball_2d[0]-0.5, ball_2d[1]-0.5), orientation=ball_dir, priority_unums=[
                            active_player_unum])
                    else:
                        self.move((ball_2d[0]-0.5, ball_2d[1]+0.5), orientation=ball_dir, priority_unums=[
                            active_player_unum])
        else:  # 我是活跃球员
            # 启用活跃球员的路径绘制（如果self.enable_draw为False则忽略）
            path_draw_options(enable_obstacles=True,
                              enable_path=True, use_team_drawing_channel=True)
            enable_pass_command = (
                w.play_mode == NeuMode.PLAY_ON and ball_2d[0] < 6)

            if r.unum == 1 and w.play_mode_group == PlayMode.THEIR_KICK:  # 对手开球期间的守门员
                self.move(self.init_pos, orientation=ball_dir)  # 原地行走
            if w.play_mode == OurMode.CORNER_KICK:
                # 将球踢到对方球门前的空位
                self.kick_long(-np.sign(ball_2d[1])*95, 5.5)
            # 如果对手明显更接近球，则防守
            elif self.min_opponent_ball_dist + 0.5 < self.min_teammate_ball_dist:
                if self.state == 2:  # 中止踢球并提交
                    self.state = 0 if self.kick_long(abort=True) else 2
                else:  # 向球移动，但将自己定位在球和我方球门之间
                    self.move(slow_ball_pos + normalize_vec(
                        (-16, 0) - slow_ball_pos) * 0.2, is_aggressive=True)
            else:
                self.state = 0 if self.kick_long(
                    goal_dir, 9, False, enable_pass_command) else 2

            # 禁用路径绘制
            path_draw_options(enable_obstacles=False, enable_path=False)

        # --------------------------------------- 3. Broadcast
        self.radio.broadcast()

        # --------------------------------------- 4. 发送到服务器
        if self.fat_proxy_cmd is None:  # 正常行为
            self.server.commit_and_send(r.get_command())  # 提交并发送命令
        else:  # 特殊代理行为
            self.server.commit_and_send(
                self.fat_proxy_cmd.encode())  # 提交并发送特殊命令
            self.fat_proxy_cmd = ""

        # ---------------------- annotations for debugging
        if self.enable_draw:
            d = w.draw
            if active_player_unum == r.unum:
                # 当球速<=0.5 m/s时预测的未来2D球位置
                d.point(slow_ball_pos, 3, d.Color.pink, "status", False)
                # 最后一次球的预测位置
                d.point(w.ball_2d_pred_pos[-1], 5,
                        d.Color.pink, "status", False)
                d.annotation((*my_head_pos_2d, 0.6), "I've got it!",
                             d.Color.yellow, "status")
            else:
                d.clear("status")  # 清除状态信息

    # --------------------------------------- Fat proxy auxiliary methods

    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot
        ball_2d = w.ball_abs_pos[:2]
        my_head_pos_2d = r.location.Head.position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            # fat proxy kick arguments: power [0,10]; relative horizontal angle [-180,180]; vertical angle [0,70]
            self.fat_proxy_cmd += f"(proxy kick 10 {normalize_deg(
                self.kick_direction - r.imu_torso_orientation):.2f} 20)"
            self.fat_proxy_walk = np.zeros(3)  # reset fat proxy walk
            return True
        else:
            self.fat_proxy_move(ball_2d-(-0.1, 0), None,
                                True)  # ignore obstacles
            return False

    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        r = self.world.robot

        target_dist = np.linalg.norm(target_2d - r.location.Head.position[:2])
        target_dir = target_rel_angle(
            r.location.Head.position[:2], r.imu_torso_orientation, target_2d)

        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += (f"(proxy dash {100} {0} {0})")
            return

        if target_dist < 0.1:
            if is_orientation_absolute:
                orientation = normalize_deg(
                    orientation - r.imu_torso_orientation)
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += (f"(proxy dash {0} {0} {target_dir:.1f})")
        else:
            self.fat_proxy_cmd += (f"(proxy dash {20} {0} {target_dir:.1f})")

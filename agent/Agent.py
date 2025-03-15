import numpy as np
from agent.Base_Agent import Base_Agent
from world.World import OurMode, NeuMode, PlayMode, TheirMode
from math_ops.math_ext import (
    vector_angle, normalize_deg, normalize_vec, target_abs_angle, target_rel_angle)


class Agent(Base_Agent):
    def __init__(self, host: str, agent_port: int, monitor_port: int, unum: int,
                 team_name: str, enable_log, enable_draw, wait_for_server=True, is_fat_proxy=False) -> None:

        # define robot type
        robot_type = (1, 1, 1, 1, 3, 3, 3, 3, 3, 0, 0)[unum-1]

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

        if np.linalg.norm(pos - r.location.Head.Position[:2]) > 0.1 or self.behavior.is_ready("Get_Up"):
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
        """
        参数
        ----------
        target_2d : array_like
            绝对坐标系下的2D目标位置
        orientation : float
            躯干的绝对或相对朝向角度（单位为度）
            设为None时会自动朝向目标方向（此时is_orientation_absolute参数将被忽略）
        is_orientation_absolute : bool
            True表示朝向角度是相对于场地坐标系，False表示相对于机器人自身躯干朝向
        avoid_obstacles : bool
            是否启用障碍物规避路径规划（若在单个仿真周期内多次调用此函数，可能需要调小timeout参数）
        priority_unums : list
            需要优先避让的队友号码列表（这些队友承担更重要的场上角色）
        is_aggressive : bool
            激进模式开关（开启时会减少对对手的安全距离限制）
        timeout : float
            路径规划的最大允许耗时（单位：微秒）
        """
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
                target_2d - r.location.Head.Position[:2])

        # Args: target, is_target_abs, ori, is_ori_abs, distance
        self.behavior.execute("Walk", target_2d, True, orientation,
                              is_orientation_absolute, distance_to_final_target)
        # self.behavior.execute("Dribble", target_2d, orientation, is_orientation_absolute)

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

    def kick_short(self, kick_direction=None, kick_distance=3, abort=False, enable_pass_command=False):
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
            return self.behavior.execute("Kick", self.kick_direction)
        else:  # fat proxy behavior
            return self.fat_proxy_kick()

    def kick_long(self, kick_direction=None, kick_distance=None, allow_aerial=True, enable_pass_command=False):
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
            return self.behavior.execute("Kick_Long", self.kick_direction, allow_aerial)
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

    def nearest_teammate(self, pos, active_player_unum):
        w = self.world

        sorted_teammates = w.teammates.sort_distance(
            pos, w.time_local_ms)

        i = 0

        while sorted_teammates[i].unum == 1 or sorted_teammates[i].unum == active_player_unum:
            i += 1

        return sorted_teammates[i].unum

    def deliberate_kick(self, ball_2d, enable_pass_command):
        goal_dist = np.sqrt((15.05 - ball_2d[0])**2 + ball_2d[1]**2)
        if goal_dist > 15:
            if ball_2d[1] > 0:
                if self.kick_long(kick_direction=target_abs_angle(ball_2d, (15.05, 0.4)), enable_pass_command=enable_pass_command):
                    self.state = 0
                else:
                    self.state = 2
            else:
                if self.kick_long(kick_direction=target_abs_angle(ball_2d, (15.05, -0.4)), enable_pass_command=enable_pass_command):
                    self.state = 0
                else:
                    self.state = 2
        elif goal_dist > 10:
            if ball_2d[1] > 0:
                if self.kick_long(kick_direction=target_abs_angle(ball_2d, (15.05, 0.5)), enable_pass_command=enable_pass_command):
                    self.state = 0
                else:
                    self.state = 2
            else:
                if self.kick_long(kick_direction=target_abs_angle(ball_2d, (15.05, -0.5)), enable_pass_command=enable_pass_command):
                    self.state = 0
                else:
                    self.state = 2
        elif goal_dist > 6:
            if ball_2d[1] > 0:
                if self.kick_long(kick_direction=target_abs_angle(ball_2d, (15.05, 0.6)), allow_aerial=False, enable_pass_command=enable_pass_command):
                    self.state = 0
                else:
                    self.state = 2
            else:
                if self.kick_long(kick_direction=target_abs_angle(ball_2d, (15.05, -0.6)), allow_aerial=False, enable_pass_command=enable_pass_command):
                    self.state = 0
                else:
                    self.state = 2
        else:
            if ball_2d[1] > 0:
                if self.kick_short(kick_direction=target_abs_angle(ball_2d, (15.05, 0.88)), enable_pass_command=enable_pass_command):
                    self.state = 0
                else:
                    self.state = 2
            else:
                if self.kick_short(kick_direction=target_abs_angle(ball_2d, (15.05, -0.88)), enable_pass_command=enable_pass_command):
                    self.state = 0
                else:
                    self.state = 2

    def dash_to_ball(self):
        w = self.world
        r = self.world.robot
        ball_x_center = 0.20
        ball_y_center = -0.04
        ball_2d = w.Ball.AbsolutePos[:2]
        my_head_pos_2d = r.location.Head.Position[:2]
        ball_vec = ball_2d - my_head_pos_2d
        ball_dir = vector_angle(ball_vec)
        slow_ball_pos = w.get_predicted_ball_pos(0.5)
        direction = 0
        bias_dir = [0.09, 0.1, 0.14, 0.08, 0.05][r.type]
        biased_dir = normalize_deg(direction + bias_dir)  # add bias to rectify direction
        next_pos, next_ori, dist_to_final_target = self.path_manager.get_path_to_ball(
            x_ori=biased_dir, x_dev=-ball_x_center, y_dev=-ball_y_center, torso_ori=biased_dir)
        if ball_dir > -30 and ball_dir < 30:  # to avoid kicking immediately without preparation & stability
            self.move((slow_ball_pos[0]+1, slow_ball_pos[1]), orientation=0)
        else:
            dist = max(0.07, dist_to_final_target)
            reset_walk = self.behavior.previous_behavior != "Walk"  # reset walk if it wasn't the previous behavior
            self.behavior.execute_sub_behavior("Walk", reset_walk, next_pos, True, next_ori, True,
                                                dist)  # target, is_target_abs, ori, is_ori_abs, distance

    def think_and_send(self):
        w = self.world
        r = self.world.robot
        my_head_pos_2d = r.location.Head.Position[:2]
        my_ori = r.IMU.TorsoOrientation
        ball_2d = w.Ball.AbsolutePos[:2]  # 球的二维坐标
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

        sorted_teammates = w.teammates.sort_distance(
            slow_ball_pos, w.time_local_ms)
        sorted_opponents = w.opponents.sort_distance(
            slow_ball_pos, w.time_local_ms)

        self.min_teammate_ball_dist = min(
            w.teammates.distance(slow_ball_pos, w.time_local_ms))
        self.min_opponent_ball_dist = min(
            w.opponents.distance(slow_ball_pos, w.time_local_ms))

        active_player_unum = sorted_teammates[0].unum
        second_active_player_unum = sorted_teammates[1].unum

        goalkeeper_is_active_player = False
        if active_player_unum == 1:
            goalkeeper_is_active_player = True
            active_player_unum = second_active_player_unum

        if ball_2d[0] > 8:
            pos_x = 2
        else:
            pos_x = -2
        if ball_2d[1] > 0:
            pos_y = 7
        else:
            pos_y = -7

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
                self.kick(kick_direction=130, kick_distance=9)
            else:
                self.move(self.init_pos, orientation=ball_dir)  # 原地行走
        elif w.play_mode == TheirMode.KICKOFF:
            self.move(self.init_pos, orientation=ball_dir)  # 原地行走
        elif active_player_unum != r.unum:  # 当前队员不是活跃球员
            enable_pass_command = (
                w.play_mode == NeuMode.PLAY_ON and ball_2d[0] < 6)
            if r.unum == 1:  # 当前球员是守门员
                if slow_ball_pos[0] < -13 and slow_ball_pos[1] > -3 and slow_ball_pos[1] < 3:
                    if goalkeeper_is_active_player:
                        if self.min_opponent_ball_dist > 0.5 or enable_pass_command:
                            if self.kick_long(kick_direction=goal_dir, enable_pass_command=enable_pass_command):
                                self.state = 0
                            else:
                                self.state = 2
                        else:
                            if self.kick_short(kick_direction=40, enable_pass_command=enable_pass_command):
                                self.state = 0
                            else:
                                self.state = 2
                    else:
                        if slow_ball_pos[0] == -15:
                            k = -slow_ball_pos[1] / (-15.1 - slow_ball_pos[0])
                        else:
                            k = -slow_ball_pos[1] / (-15 - slow_ball_pos[0])
                        x = slow_ball_pos[0]
                        y = k * (x + 15)
                        if x > -14.2:
                            x = -14.2
                            y = k * (x + 15)
                        elif x < -14.8:
                            x = -14.8
                            y = k * (x + 15)
                        if y > 1.5:
                            y = 1.5
                        elif y < -1.5:
                            y = -1.5
                        self.move((x-0.4, y), orientation=ball_dir)
                elif w.play_mode == OurMode.GOAL_KICK:
                    if self.kick_long(kick_direction=goal_dir, enable_pass_command=enable_pass_command):
                        self.state = 0
                    else:
                        self.state = 2
                else:
                    if slow_ball_pos[0] == -15:
                        k = -slow_ball_pos[1] / (-15.1 - slow_ball_pos[0])
                    else:
                        k = -slow_ball_pos[1] / (-15 - slow_ball_pos[0])
                    x = slow_ball_pos[0]
                    y = k * (x + 15)
                    if x > -14.2:
                        x = -14.2
                        y = k * (x + 15)
                    elif x < -14.8:
                        x = -14.8
                        y = k * (x + 15)
                    if y > 1.5:
                        y = 1.5
                    elif y < -1.5:
                        y = -1.5
                    self.move((x, y), orientation=ball_dir)
            elif r.unum == self.nearest_teammate((slow_ball_pos[0]+15, 1), active_player_unum):
                if slow_ball_pos[0]+15 > 13:
                    if slow_ball_pos[1] > 0:
                        self.move((12.5, 1), orientation=goal_dir)
                    else:
                        self.move((12.5, -1), orientation=goal_dir)
                else:
                    if slow_ball_pos[1] > 0:
                        self.move((slow_ball_pos[0]+15, 1), orientation=ball_dir)
                    else:
                        self.move((slow_ball_pos[0]+15,  -1), orientation=ball_dir)
            elif r.unum == self.nearest_teammate((slow_ball_pos[0]+7, -1), active_player_unum):
                if slow_ball_pos[0]+7 > 13:
                    if slow_ball_pos[1] > 0:
                        self.move((12, 0.8), orientation=goal_dir)
                    else:
                        self.move((12, -0.8), orientation=goal_dir)
                else:
                    if slow_ball_pos[1] > 0:
                        self.move((slow_ball_pos[0]+7, 1), orientation=ball_dir)
                    else:
                        self.move((slow_ball_pos[0]+7, -1), orientation=ball_dir)
            elif r.unum in (2, 3, 4):
                if r.unum == self.nearest_teammate((-13, 0), active_player_unum):
                    if slow_ball_pos[0] == -15:
                        k = -slow_ball_pos[1] / (-15.1 - slow_ball_pos[0])
                    else:
                        k = -slow_ball_pos[1] / (-15 - slow_ball_pos[0])
                    x = slow_ball_pos[0]
                    y = k * (x + 15)
                    if x > -14.2:
                        x = -14.2
                        y = k * (x + 15)
                    elif x < -14.8:
                        x = -14.8
                        y = k * (x + 15)
                    if y > 1.5:
                        y = 1.5
                    elif y < -1.5:
                        y = -1.5
                    self.move((x+0.5, y+0.5), orientation=ball_dir)
                else:
                    new_x = max(0.5, (slow_ball_pos[0]+15)/15) * \
                        (self.init_pos[0]+15) - 15
                    self.move((new_x, self.init_pos[1]), orientation=ball_dir, priority_unums=[
                        active_player_unum])
            elif r.unum == self.nearest_teammate((slow_ball_pos[0]+2, slow_ball_pos[1]+0.5), active_player_unum):
                self.move(
                    (slow_ball_pos[0]+2, slow_ball_pos[1]+0.5), orientation=goal_dir)
            elif r.unum == self.nearest_teammate((pos_x, pos_y), active_player_unum):
                self.move(
                    (pos_x, pos_y), orientation=goal_dir)
            else:
                # 优化位置选择和移动策略
                if r.unum % 2 == 0:
                    # 假设偶数球员在场地左侧
                    target_pos = (slow_ball_pos[0] - 0.5, slow_ball_pos[1] - 0.5)
                else:
                    # 假设奇数球员在场地右侧
                    target_pos = (slow_ball_pos[0] - 0.5, slow_ball_pos[1] + 0.5)

                # 确保球员在移动时能够避开对手，并且能够快速到达目标位置
                self.move(target_pos, orientation=ball_dir, priority_unums=[active_player_unum])
        else:  # 我是活跃球员
            # 启用活跃球员的路径绘制（如果self.enable_draw为False则忽略）
            path_draw_options(enable_obstacles=True,
                            enable_path=True, use_team_drawing_channel=True)
            enable_pass_command = (
                w.play_mode == NeuMode.PLAY_ON and ball_2d[0] < 6)

            distance_diff = abs(self.min_opponent_ball_dist -
                                self.min_teammate_ball_dist)  # 队友和对方与球距离的差值
            if w.play_mode == OurMode.CORNER_KICK:
                # 将球踢到对方球门前的空位
                self.kick_short(-np.sign(ball_2d[1])*95)
            elif goalkeeper_is_active_player:
                self.move((slow_ball_pos[0]+0, 0.66), orientation=goal_dir)
            elif ball_2d[0] > 12.5 and ball_2d[1] > -1.1 and ball_2d[1] < 1.1:
                self.dash_to_ball()
            elif self.min_opponent_ball_dist + 0.5 - self.min_teammate_ball_dist >= 0:
                if enable_pass_command:
                    self.deliberate_kick(ball_2d, enable_pass_command)
                elif distance_diff < 0.5:
                    if self.kick_short(kick_direction=goal_dir, enable_pass_command=enable_pass_command):
                        self.state = 0
                    else:
                        self.state = 2
                else:
                    self.deliberate_kick(ball_2d, enable_pass_command)
            # 如果对手明显更接近球，则防守
            else:
                if self.state == 2:  # 中止踢球并提交
                    self.state = 0 if self.kick_short(abort=True) else 2
                else:  # 向球移动，但将自己定位在球和我方球门之间
                    self.move(slow_ball_pos + normalize_vec(
                        (-16, 0) - slow_ball_pos) * 0.2, is_aggressive=True)

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
                d.point(w.Ball.Predicted2DPos[-1], 5,
                        d.Color.pink, "status", False)
                d.annotation((*my_head_pos_2d, 0.6), "I've got it!",
                            d.Color.yellow, "status")
            else:
                d.clear("status")  # 清除状态信息

    def fat_proxy_kick(self):
        w = self.world
        r = self.world.robot
        ball_2d = w.Ball.AbsolutePos[:2]
        my_head_pos_2d = r.location.Head.Position[:2]

        if np.linalg.norm(ball_2d - my_head_pos_2d) < 0.25:
            # fat proxy kick arguments: power [0,10]; relative horizontal angle [-180,180]; vertical angle [0,70]
            self.fat_proxy_cmd += f"(proxy kick 10 {normalize_deg(
                self.kick_direction - r.IMU.TorsoOrientation): .2f} 20)"
            self.fat_proxy_walk = np.zeros(3)  # reset fat proxy walk
            return True
        else:
            self.fat_proxy_move(ball_2d-(-0.1, 0), None,
                                True)  # ignore obstacles
            return False

    def fat_proxy_move(self, target_2d, orientation, is_orientation_absolute):
        r = self.world.robot

        target_dist = np.linalg.norm(target_2d - r.location.Head.Position[:2])
        target_dir = target_rel_angle(
            r.location.Head.Position[:2], r.IMU.TorsoOrientation, target_2d)

        if target_dist > 0.1 and abs(target_dir) < 8:
            self.fat_proxy_cmd += (f"(proxy dash {100} {0} {0})")
            return

        if target_dist < 0.1:
            if is_orientation_absolute:
                orientation = normalize_deg(
                    orientation - r.IMU.TorsoOrientation)
            target_dir = np.clip(orientation, -60, 60)
            self.fat_proxy_cmd += (f"(proxy dash {0} {0} {target_dir:.1f})")
        else:
            self.fat_proxy_cmd += (f"(proxy dash {20} {0} {target_dir:.1f})")

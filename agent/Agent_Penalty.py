import random
import numpy as np

from agent.Base_Agent import Base_Agent
from math_ops.math_ext import vector_from_angle
from world.World import NeuMode, OurMode, TheirMode


class RobotState:
    "机器人状态"
    NORMAL = 0  # 正常
    GET_UP = 1  # 起身
    DIVE_LEFT = 2  # 向左扑救
    DIVE_RIGHT = 3  # 向右扑救
    WAIT = 4  # 等待


class Agent(Base_Agent):

    def __init__(self, host: str, agent_port: int, monitor_port: int, unum: int,
                 team_name: str, enable_log, enable_draw, wait_for_server=True,
                 is_fat_proxy=False) -> None:

        # 定义机器人类型：1号为守门员，其他为前锋
        robot_type = 0 if unum == 1 else 4

        # 初始化基类Agent
        super().__init__(host, agent_port, monitor_port, unum, robot_type,
                         team_name, enable_log, enable_draw, False, wait_for_server, None)

        self.enable_draw = enable_draw
        self.state = RobotState.NORMAL
        self.kick_dir = 0  # 踢球方向
        self.reset_kick = True  # 是否重置踢球方向

    def think_and_send(self):
        # 获取当前世界和机器人状态
        w = self.world
        r = self.world.robot
        my_head_pos_2d = r.location.Head.Position[:2]
        ball_2d = w.Ball.AbsolutePos[:2]

        # --------------------------------------- 1. 决策行为

        # 在开球前或进球后，回到初始位置并等待
        if w.play_mode in [NeuMode.BEFORE_KICKOFF, TheirMode.GOAL, OurMode.GOAL]:
            self.state = RobotState.NORMAL
            self.reset_kick = True
            # 守门员初始位置在 (-14, 0)，前锋在 (4.9, 0)
            pos = (-14, 0) if r.unum == 1 else (4.9, 0)
            if np.linalg.norm(pos - my_head_pos_2d) > 0.1 or self.behavior.is_ready("Get_Up"):
                self.server.commit_beam(pos, 0)  # 传送到初始位置
            else:
                self.behavior.execute("Zero_Bent_Knees")  # 等待
        elif self.state == RobotState.DIVE_LEFT:  # 执行向左扑救
            # 扑救完成后进入等待状态
            self.state = RobotState.WAIT if self.behavior.execute(
                "Dive_Left") else RobotState.DIVE_LEFT
        elif self.state == RobotState.DIVE_RIGHT:  # 执行向右扑救
            # 扑救完成后进入等待状态
            self.state = RobotState.WAIT if self.behavior.execute(
                "Dive_Right") else RobotState.DIVE_RIGHT
        elif self.state == RobotState.WAIT:  # 扑救后或对方开球时等待
            pass
        # 起身或已倒地
        elif self.state == RobotState.GET_UP or self.behavior.is_ready("Get_Up"):
            # 起身动作完成后恢复正常状态
            self.state = RobotState.NORMAL if self.behavior.execute(
                "Get_Up") else RobotState.GET_UP
        elif (w.play_mode == OurMode.KICK_OFF and r.unum == 1) or \
                (w.play_mode == TheirMode.KICKOFF and r.unum != 1):
            self.state = RobotState.WAIT  # 守门员在开球或对方开球时等待
        elif r.unum == 1:  # 守门员逻辑
            y_coordinate = np.clip(ball_2d[1], -1.1, 1.1)
            # 守门员根据球的位置调整位置
            self.behavior.execute(
                "Walk", (-14, y_coordinate), True, 0, True, None)
            if ball_2d[0] < -10:  # 当球靠近球门时，执行扑救
                self.state = RobotState.DIVE_LEFT if ball_2d[1] > 0 else RobotState.DIVE_RIGHT
        else:  # 前锋逻辑
            if w.play_mode == OurMode.KICK_OFF and ball_2d[0] > 5:  # 开球时选择踢球方向
                if self.reset_kick:
                    self.kick_dir = random.choice([-7.5, 7.5])
                    self.reset_kick = False
                self.behavior.execute("Basic_Kick", self.kick_dir)
            else:
                self.behavior.execute("Zero_Bent_Knees")  # 等待

        # --------------------------------------- 2. 广播状态信息
        self.radio.broadcast()

        # --------------------------------------- 3. 发送指令到服务器
        self.server.commit_and_send(r.get_command())

        # --------------------------------------- 4. 调试注释和绘图
        if self.enable_draw:
            d = w.draw
            role = "Goalkeeper" if r.unum == 1 else "Kicker"
            d.annotation((*my_head_pos_2d, 0.8), role,
                         d.Color.yellow, "status")
            if r.unum != 1 and w.play_mode == OurMode.KICK_OFF:  # 前锋在开球时绘制箭头指示踢球方向
                target_pos = ball_2d + 5 * vector_from_angle(self.kick_dir)
                d.arrow(ball_2d, target_pos, 0.4, 3,
                        d.Color.cyan_light, "Target")

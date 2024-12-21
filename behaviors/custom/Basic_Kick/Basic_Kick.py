from agent.Base_Agent import Base_Agent
from behaviors.custom.Step.Step_Generator import Step_Generator
from math_ops.math_ext import normalize_deg


class Basic_Kick():

    def __init__(self, base_agent: Base_Agent) -> None:
        self.behavior = base_agent.behavior
        self.path_manager = base_agent.path_manager
        self.world = base_agent.world
        self.description = "Walk to ball and perform a basic kick"
        self.auto_head = True

        r_type = self.world.robot.type
        self.bias_dir = [22, 29, 26, 29, 22][self.world.robot.type]
        self.ball_x_limits = ((0.19, 0.215), (0.2, 0.22),
                              (0.19, 0.22), (0.2, 0.215), (0.2, 0.215))[r_type]
        self.ball_y_limits = ((-0.115, -0.1), (-0.125, -0.095),
                              (-0.12, -0.1), (-0.13, -0.105), (-0.09, -0.06))[r_type]
        self.ball_x_center = (self.ball_x_limits[0] + self.ball_x_limits[1])/2
        self.ball_y_center = (self.ball_y_limits[0] + self.ball_y_limits[1])/2

    def execute(self, reset: bool, direction: float, abort=False) -> bool:  # You can add more arguments
        """
        Parameters
        ----------
        reset: 用于重置行为状态
        direction : 相对于球场的踢球的方向，单位为度
        abort : 指示是否应当中断当前行为的变量
            The method returns True upon successful abortion, which is immediate while the robot is aligning itself. 
            However, if the abortion is requested during the kick, it is delayed until the kick is completed.
        """

        w = self.world
        r = self.world.robot
        b = w.Ball.RelativeTorsoCartPos
        t = w.time_local_ms
        gait: Step_Generator = self.behavior.get_custom_behavior_object(
            "Walk").env.step_generator

        if reset:
            self.phase = 0
            self.reset_time = t

        if self.phase == 0:
            # add bias to rectify direction
            biased_dir = normalize_deg(direction + self.bias_dir)
            # the reset was learned with loc, not IMU
            ang_diff = abs(normalize_deg(
                biased_dir - r.location.Torso.Orientation))

            next_pos, next_ori, dist_to_final_target = self.path_manager.get_path_to_ball(
                x_ori=biased_dir, x_dev=-self.ball_x_center, y_dev=-self.ball_y_center, torso_ori=biased_dir)

            if (w.Ball.LastSeen > t - w.VISUALSTEP_MS and ang_diff < 5 and       # ball is visible & aligned
                # ball is in kick area (x)
                self.ball_x_limits[0] < b[0] < self.ball_x_limits[1] and
                # ball is in kick area (y)
                self.ball_y_limits[0] < b[1] < self.ball_y_limits[1] and
                # ball absolute location is recent
                t - w.Ball.AbsolutePosLastUpdate < 100 and
                # if absolute ball position is updated
                dist_to_final_target < 0.03 and
                # walk gait phase is adequate
                not gait.state_is_left_active and gait.state_current_ts == 2 and
                    t - self.reset_time > 500):  # to avoid kicking immediately without preparation & stability

                self.phase += 1

                return self.behavior.execute_sub_behavior("Kick_Motion", True)
            else:
                dist = max(0.07, dist_to_final_target)
                # reset walk if it wasn't the previous behavior
                reset_walk = reset and self.behavior.previous_behavior != "Walk"
                # target, is_target_abs, ori, is_ori_abs, distance
                self.behavior.execute_sub_behavior(
                    "Walk", reset_walk, next_pos, True, next_ori, True, dist)
                return abort  # abort only if self.phase == 0

        else:  # define kick parameters and execute
            return self.behavior.execute_sub_behavior("Kick_Motion", False)

    def is_ready(self) -> any:  # You can add more arguments
        ''' Returns True if this behavior is ready to start/continue under current game/robot conditions '''
        return True

import pickle
from behaviors.custom.Kick_Long.Env import Env
from agent.Base_Agent import Base_Agent
from math_ops.math_ext import get_active_directory, normalize_deg
from math_ops.Neural_Network import run_mlp


class Kick_Long:

    # def __init__(self, world) -> None:
    #     self.world = world
    def __init__(self, base_agent: Base_Agent) -> None:
        self.behavior = base_agent.behavior
        self.path_manager = base_agent.path_manager
        self.world = base_agent.world
        self.description = "Walk to ball and perform a long kick"
        self.auto_head = False
        self.env = Env(base_agent)
        with open(get_active_directory([
            '/behaviors/custom/Kick_Long/long_kick_R0.pkl',
            '/behaviors/custom/Kick_Long/long_kick_R1.pkl',
            '/behaviors/custom/Kick_Long/long_kick_R2.pkl',
            '/behaviors/custom/Kick_Long/long_kick_R3.pkl',
                '/behaviors/custom/Kick_Long/long_kick_R0.pkl'][self.world.robot.type]), "rb") as f:
            self.model = pickle.load(f)

    def execute(self, reset, orientation, allow_aerial=True, wait=False):
        """
        Parameters
        ----------
        orientation : float
            absolute orientation of torso (relative to imu_torso_orientation), in degrees
        allow_aerial : float
            allow aerial kicks, not recommeded near goal
        """
        w = self.world
        r = self.world.robot
        step_gen = self.behavior.get_custom_behavior_object(
            "Walk").env.step_generator
        reset_kick = False
        if reset:
            self.phase = 0
            self.reset_time = w.time_local_ms
        elif self.phase == 0:
            next_pos, next_ori, dist_to_final_target = self.path_manager.get_path_to_ball(
                x_ori=orientation, x_dev=(-0.21), y_dev=0.045, torso_ori=orientation)
            ang_diff = abs(normalize_deg(
                orientation - r.location.Torso.Orientation))
            dist_factor = 0
            if dist_to_final_target < 0.018 and ang_diff < 6 and (step_gen.state_is_left_active or step_gen.state_current_ts) == 2 and w.time_local_ms - w.Ball.AbsolutePosLastUpdate < 100 and (wait or w.time_local_ms - self.reset_time) > 500:
                self.phase += 1
                reset_kick = True
            else:
                dist_factor = 0.9
                # if w.penalty_shootout:
                #     dist_factor = 0.5
                # else:
                #     if w.front_standing_opp_d is not None:
                #         dist_factor = max(
                #             min(0.9, 0.9 - w.front_standing_opp_d * 0.08), 0.7)
            dist = max(0.07, dist_to_final_target * dist_factor)
            d = w.draw
            # d.annotation((*r.loc_head_position[:2], *(1.2, )), f"{w.front_standing_opp_d} {dist_factor}", d.Color.red, "role")
            reset_walk = reset and self.behavior.previous_behavior != "Walk"
            self.behavior.execute_sub_behavior(
                "Walk", reset_walk, next_pos, True, next_ori, True, dist)
        if self.phase == 1:
            self.env.kick_ori = orientation
            obs = self.env.observe(reset_kick)
            action = run_mlp(obs, self.model)
            return self.env.execute(action, allow_aerial)
        return False

    def is_ready(self):
        """ Returns True if Kick Behavior is ready to start under current game/robot conditions """
        return True

# okay decompiling behaviors.custom.Kick_Long_RL.Kick_Long_RL.pyc

import numpy as np
from agent.Base_Agent import Base_Agent
from math_ops.math_ext import normalize_deg
import world.World as World


class Env:

    # def __init__(self, world: World) -> None:
    def __init__(self, base_agent: Base_Agent) -> None:
        self.agent = base_agent
        self.behavior = base_agent.behavior
        self.path_manager = base_agent.path_manager
        self.world = base_agent.world
        self.obs = np.zeros(63, np.float32)
        self.DEFAULT_ARMS = np.array(
            [-90, -90, 8, 8, 90, 90, 70, 70], np.float32)
        self.kick_ori = None

    def observe(self, init=False):
        w = self.world
        r = self.world.robot
        if init:
            self.step_counter = 0
            self.act = np.zeros(16, np.float32)
        self.obs[0] = self.step_counter / 20
        self.obs[1] = r.loc_head_z * 3
        self.obs[2] = r.loc_head_z_vel / 2
        self.obs[3] = r.imu_torso_roll / 15
        self.obs[4] = r.imu_torso_pitch / 15
        self.obs[5:8] = r.gyro / 100
        self.obs[8:11] = r.acc / 10
        self.obs[11:17] = r.frp.get("lf", np.zeros(6)) * (10, 10, 10, 0.01, 0.01,
                                                            0.01)
        self.obs[17:23] = r.frp.get("rf", np.zeros(6)) * (10, 10, 10, 0.01, 0.01,
                                                            0.01)
        self.obs[23:39] = np.array([joint.position for joint in r.joints[2:18]]) / 100
        self.obs[39:55] = np.array([joint.speed for joint in r.joints[2:18]]) / 6.1395
        ball_rel_hip_center = self.agent.inv_kinematics.torso_to_hip_transform(
            w.ball_rel_torso_cart_pos)
        if init:
            self.obs[55:58] = (0, 0, 0)
        else:
            if w.ball_is_visible:
                self.obs[55:58] = (
                    ball_rel_hip_center - self.obs[58:61]) * 10
            self.obs[58:61] = ball_rel_hip_center
            self.obs[61] = np.linalg.norm(ball_rel_hip_center) * 2
            self.obs[62] = normalize_deg(
                self.kick_ori - r.imu_torso_orientation) / 30
        return self.obs

    def execute(self, action, allow_aerial=True):
        w = self.world
        r = self.world.robot
        if self.step_counter < 6:
            arr = action * [2, 2, 1, 1, 0.2, 0.2, 0.2, 0.2, 2, 2, 1, 1, 1, 1, 1, 1]
            for i in range(2, 18):
                r.joints[i].target_speed = arr[i-2]
            r.joints[12].target_speed += 1
            r.joints[6].target_speed += 3
            r.joints[5].target_speed -= 1
            r.joints[8].target_speed += 3
            r.joints[9].target_speed -= 5
        else:
            arr = action * [6, 6, 3, 3, 6, 6, 6, 6, 6, 6, 2, 2, 1, 1, 1, 1]
            for i in range(2, 18):
                r.joints[i].target_speed = arr[i-2]
            r.joints[6].target_speed -= 3
            r.joints[7].target_speed += 3
            r.joints[9].target_speed += 3
            r.joints[11].target_speed -= 1
    
            if not allow_aerial:
                r.joints[7].target_speed = np.clip(
                    r.joints[7].target_speed, -10, 5)
            r.set_joints_target_position_direct(
                [0, 1], (np.array([0, -44], float)), harmonize=False)
        self.step_counter += 1
        return self.step_counter >= 16

# okay decompiling behaviors.custom.Kick_Long_RL.Env.pyc

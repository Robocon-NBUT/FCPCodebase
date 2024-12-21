from agent.Base_Agent import Base_Agent
from math_ops.math_ext import get_active_directory
from math_ops.Neural_Network import run_mlp
import pickle
import numpy as np


class Fall():

    def __init__(self, base_agent: Base_Agent) -> None:
        self.world = base_agent.world
        self.description = "Fall example"
        self.auto_head = False

        with open(get_active_directory("/behaviors/custom/Fall/fall.pkl"), 'rb') as f:
            self.model = pickle.load(f)

        # extracted from size of Neural Network's last layer bias
        self.action_size = len(self.model[-1][0])
        self.obs = np.zeros(self.action_size+1, np.float32)

        # compatibility between different robot types
        self.controllable_joints = min(
            self.world.robot.no_of_joints, self.action_size)

    def observe(self):
        r = self.world.robot

        for i in range(self.action_size):
            self.obs[i] = r.joints[i].position / \
                100  # naive scale normalization

        # head.z (alternative: r.loc_head_z)
        self.obs[self.action_size] = r.cheat_abs_pos[2]

    def execute(self, reset) -> bool:
        self.observe()
        action = run_mlp(self.obs, self.model)

        self.world.robot.set_joints_target_position_direct(  # commit actions:
            range(self.controllable_joints),  # act on trained joints
            action*10,                       # scale actions up to motivate early exploration
            # there is no point in harmonizing actions if the targets change at every step
            harmonize=False
        )

        # finished when head height < 0.15 m
        return self.world.robot.location.Head.Head_Z < 0.15

    def is_ready(self) -> any:
        ''' Returns True if this behavior is ready to start/continue under current game/robot conditions '''
        return True

'''
Objective:
----------
Fall and get up
'''
from itertools import count
import numpy as np

from agent.Base_Agent import Base_Agent as Agent
from scripts.commons.Script import Script


class Get_Up:
    def __init__(self, script: Script) -> None:
        self.script = script
        self.player: Agent = None

    def sync(self):
        r = self.player.world.robot
        self.player.server.commit_and_send(r.get_command())
        self.player.server.receive()

    def execute(self):

        a = self.script.args
        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name
        player = self.player = Agent(a.i, a.p, a.m, a.u, a.r, a.t)
        behavior = player.behavior
        r = player.world.robot

        player.server.commit_beam((-3, 0), 0)
        print("\nPress ctrl+c to return.")

        for i in count():
            rnd = np.random.uniform(-6, 6, r.no_of_joints)

            # Fall
            while r.location.Head.Head_Z > 0.3 and r.IMU.TorsoInclination < 50:
                if i < 4:
                    # First, fall deterministically
                    behavior.execute(
                        ["Fall_Front", "Fall_Back", "Fall_Left", "Fall_Right"][i % 4])
                else:
                    for index in range(r.no_of_joints):
                        r.joints[index].target_speed = rnd[index]
                self.sync()

            # Get up
            behavior.execute_to_completion("Get_Up")
            behavior.execute_to_completion("Zero_Bent_Knees")

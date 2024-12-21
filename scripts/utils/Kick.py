'''
Objective:
----------
Demonstrate kick
'''
import numpy as np
from agent.Base_Agent import Base_Agent as Agent
from math_ops.math_ext import normalize_vec, vector_angle
from scripts.commons.Script import Script


class Kick():
    def __init__(self, script: Script) -> None:
        self.script = script

    def execute(self):

        a = self.script.args
        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name
        player = Agent(a.i, a.p, a.m, a.u, a.r, a.t)
        # enable drawings of obstacles and path to ball
        player.path_manager.draw_options(
            enable_obstacles=True, enable_path=True)
        behavior = player.behavior
        w = player.world
        r = w.robot

        print("\nThe robot will kick towards the center of the field")
        print("Try to manually relocate the ball")
        print("Press ctrl+c to return\n")

        player.server.unofficial_set_play_mode("PlayOn")
        player.server.unofficial_beam((-3, 0, r.beam_height), 0)
        vec = (1, 0)

        while True:
            player.server.unofficial_set_game_time(0)
            b = w.Ball.AbsolutePos[:2]

            # speed of zero is likely to indicate prolongued inability to see the ball
            if 0 < np.linalg.norm(w.get_ball_abs_vel(6)) < 0.02:
                # update kick if ball is further than 0.5 m
                if np.linalg.norm(w.Ball.RelativeHeadCartPos[:2]) > 0.5:
                    if max(abs(b)) < 0.5:
                        vec = np.array([6, 0])
                    else:
                        vec = normalize_vec((0, 0)-b) * 6

                w.draw.point(b+vec, 8, w.draw.Color.pink, "target")

            behavior.execute("Basic_Kick", vector_angle(vec))

            player.server.commit_and_send(r.get_command())
            player.server.receive()

            if behavior.is_ready("Get_Up"):
                player.server.unofficial_beam(
                    (*r.location.Head.position[0:2], r.beam_height), 0)
                behavior.execute_to_completion("Zero_Bent_Knees")

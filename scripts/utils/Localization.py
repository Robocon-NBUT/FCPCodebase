from agent.Agent import Agent
from cpp.localization import localization
from math_ops.math_ext import to_3d
from scripts.commons.Script import Script
from world.commons.Draw import Draw
from world.commons.Other_Robot import Other_Robot
import numpy as np

class Localization():

    def __init__(self, script: Script) -> None:
        self.script = script

    def execute(self):

        a = self.script.args
        # using independent draw object so that the internal agent drawings can be disabled
        d = self.draw = Draw(True, 0, a.i, 32769)

        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Team name, Enable Log, Enable Draw
        # one teammate (dummy goalkeeper without communication)
        self.script.batch_create(
            Agent, ((a.i, a.p, a.m, 1, a.t, False, False),))
        self.script.batch_create(
            # one opponent
            Agent, ((a.i, a.p, a.m, 5, "Opponent", False, False),))
        # one main agent (the one who draws its world)
        self.script.batch_create(
            Agent, ((a.i, a.p, a.m, 9, a.t, False, False),))

        # Beam dummy goalkeeper
        self.script.batch_unofficial_beam(((-14, 0, 0.5, 0),), slice(0, 1))

        p: Agent = self.script.players[-1]  # p identifies the main agent
        p.server.unofficial_set_play_mode("PlayOn")

        # Execute
        while True:
            self.script.batch_commit_and_send(
                slice(0, 1))  # dummy agent does not think
            self.script.batch_execute_agent(
                slice(1, None))  # execute normal agents
            # receive & don't update dummy's world state (to save cpu resources)
            self.script.batch_receive(slice(0, 1), False)
            # receive & update world state
            self.script.batch_receive(slice(1, None))

            if p.world.vision_is_up_to_date:
                if p.world.robot.location.is_up_to_date:     # localization will draw the world of the last agent to be executed
                    # print data received by the localization module
                    localization.print_python_data()
                    localization.draw_visible_elements(
                        not p.world.team_side_is_left)  # draw visible elements
                    localization.print_report()         # print report with stats
                    print("\nPress ctrl+c to return.")
                    d.circle(p.world.Ball.AbsolutePos, 0.1, 6,
                             Draw.Color.purple_magenta, "world", False)
                else:
                    d.annotation(p.world.robot.cheat_abs_pos,
                                 "Not enough visual data!", Draw.Color.red, "world", False)

                for o in p.world.teammates:
                    if o.state_last_update != 0 and not o.is_self:  # skip if other robot was not yet seen
                        self._draw_other_robot(p, o, Draw.Color.white)

                for o in p.world.opponents:
                    if o.state_last_update != 0:  # skip if other robot was not yet seen
                        self._draw_other_robot(p, o, Draw.Color.red)

                d.flush("world")

    def _draw_other_robot(self, p: Agent, o: Other_Robot, team_color):
        # p - player that sees
        # o - other robot (player that is seen)

        d = self.draw
        white = Draw.Color.white
        green = Draw.Color.green_light
        gray = Draw.Color.gray_20

        time_diff = p.world.time_local_ms - o.state_last_update
        if time_diff > 0:
            white = Draw.Color.gray_40
            green = Draw.Color.get(107, 139, 107)
            gray = Draw.Color.gray_50

        # orientation
        if len(o.state_abs_pos) == 3:
            line_tip = o.state_abs_pos + \
                (0.5*np.cos(np.deg2rad(o.state_orientation)),
                 0.5*np.sin(np.deg2rad(o.state_orientation)), 0)
            d.line(o.state_abs_pos, line_tip, 3, white, "world", False)
        else:
            temp_pos = to_3d(o.state_abs_pos, 0.3)
            line_tip = temp_pos + \
                (0.5*np.cos(np.deg2rad(o.state_orientation)),
                 0.5*np.sin(np.deg2rad(o.state_orientation)), 0)
            d.line(temp_pos, line_tip, 3, Draw.Color.yellow, "world", False)

        # body parts
        for pos in o.state_body_parts_abs_pos.values():
            d.sphere(pos, 0.07, green, "world", False)

        # player ground area
        d.circle(
            o.state_ground_area[0], o.state_ground_area[1], 6, team_color, "world", False)

        # distance
        midpoint = (o.state_abs_pos[0:2] +
                    p.world.robot.location.Head.position[0:2])/2
        d.line(
            o.state_abs_pos[0:2], p.world.robot.location.Head.position[0:2], 1, gray, "world", False)
        d.annotation(
            midpoint, f'{o.state_horizontal_dist:.2f}m', white, "world", False)

        # velocity
        arrow_tip = o.state_abs_pos[0:2] + o.state_filtered_velocity[0:2]
        d.arrow(o.state_abs_pos[0:2], arrow_tip, 0.2, 4, green, "world", False)

        # state
        state_color = white if not o.state_fallen else Draw.Color.yellow
        d.annotation(
            (o.state_abs_pos[0], o.state_abs_pos[1], 1),
            f"({o.unum}) {'Fallen' if o.state_fallen else 'Normal'}", state_color, "world", False)

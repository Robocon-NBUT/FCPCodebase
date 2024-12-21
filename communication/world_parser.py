from pathlib import Path
import numpy as np

from math_ops.math_ext import deg_sph2cart
from world.Robot import Robot
from world.World import World, OurMode, TheirMode, NeuMode


class WorldParser:
    def __init__(self, world: World, hear_callback) -> None:
        self.file_name = Path(__file__).name + ": "
        self.world = world
        self.hear_callback = hear_callback
        self.exp: bytearray = None
        self.depth = None
        self.LEFT_SIDE_FLAGS = {
            b'F2L': (-15, -10, 0),
            b'F1L': (-15, +10, 0),
            b'F2R': (+15, -10, 0),
            b'F1R': (+15, +10, 0),
            b'G2L': (-15, -1.05, 0.8),
            b'G1L': (-15, +1.05, 0.8),
            b'G2R': (+15, -1.05, 0.8),
            # mapping between flag names and their corrected location, when playing on the left side
            b'G1R': (+15, +1.05, 0.8)
        }
        self.RIGHT_SIDE_FLAGS = {
            b'F2L': (+15, +10, 0),
            b'F1L': (+15, -10, 0),
            b'F2R': (-15, +10, 0),
            b'F1R': (-15, -10, 0),
            b'G2L': (+15, +1.05, 0.8),
            b'G1L': (+15, -1.05, 0.8),
            b'G2R': (-15, +1.05, 0.8),
            b'G1R': (-15, -1.05, 0.8)
        }
        self.play_mode_to_id = None
        self.LEFT_PLAY_MODE_TO_ID = {
            "KickOff_Left": OurMode.KICK_OFF, "KickIn_Left": OurMode.KICK_IN,
            "corner_kick_left": OurMode.CORNER_KICK, "goal_kick_left": OurMode.GOAL_KICK,
            "free_kick_left": OurMode.FREE_KICK, "pass_left": OurMode.PASS,
            "direct_free_kick_left": OurMode.DIR_FREE_KICK, "Goal_Left": OurMode.GOAL,
            "offside_left": OurMode.OFFSIDE, "KickOff_Right": TheirMode.KICKOFF,
            "KickIn_Right": TheirMode.KICK_IN, "corner_kick_right": TheirMode.CORNER_KICK,
            "goal_kick_right": TheirMode.GOAL_KICK, "free_kick_right": TheirMode.FREE_KICK,
            "pass_right": TheirMode.PASS, "direct_free_kick_right": TheirMode.DIR_FREE_KICK,
            "Goal_Right": TheirMode.GOAL, "offside_right": TheirMode.OFFSIDE,
            "BeforeKickOff": NeuMode.BEFORE_KICKOFF, "GameOver": NeuMode.GAME_OVER,
            "PlayOn": NeuMode.PLAY_ON
        }
        self.RIGHT_PLAY_MODE_TO_ID = {
            "KickOff_Left": TheirMode.KICKOFF, "KickIn_Left": TheirMode.KICK_IN,
            "corner_kick_left": TheirMode.CORNER_KICK, "goal_kick_left": TheirMode.GOAL_KICK,
            "free_kick_left": TheirMode.FREE_KICK, "pass_left": TheirMode.PASS,
            "direct_free_kick_left": TheirMode.DIR_FREE_KICK, "Goal_Left": TheirMode.GOAL,
            "offside_left": TheirMode.OFFSIDE, "KickOff_Right": OurMode.KICK_OFF,
            "KickIn_Right": OurMode.KICK_IN, "corner_kick_right": OurMode.CORNER_KICK,
            "goal_kick_right": OurMode.GOAL_KICK, "free_kick_right": OurMode.FREE_KICK,
            "pass_right": OurMode.PASS, "direct_free_kick_right": OurMode.DIR_FREE_KICK,
            "Goal_Right": OurMode.GOAL, "offside_right": OurMode.OFFSIDE,
            "BeforeKickOff": NeuMode.BEFORE_KICKOFF, "GameOver": NeuMode.GAME_OVER,
            "PlayOn": NeuMode.PLAY_ON
        }

    def find_non_digit(self, start):
        while True:
            if (self.exp[start] < ord('0') or self.exp[start] > ord('9')) and self.exp[start] != ord('.'):
                return start
            start += 1

    def read_float(self, start):
        if self.exp[start:start+3] == b'nan':
            return float('nan'), start+3  # handle nan values (they exist)
        # we assume the first one is a digit or minus sign
        end = self.find_non_digit(start+1)
        try:
            retval = float(self.exp[start:end])
        except:
            self.world.log(
                f"{self.file_name}String to float conversion failed: {self.exp[start:end]} at msg[{start},{end}], \nMsg: {self.exp.decode()}")
            retval = 0
        return retval, end

    def read_int(self, start):
        # we assume the first one is a digit or minus sign
        end = self.find_non_digit(start+1)
        return int(self.exp[start:end]), end

    def read_bytes(self, start):
        end = start
        while True:
            if self.exp[end] == ord(' ') or self.exp[end] == ord(')'):
                break
            end += 1

        return self.exp[start:end], end

    def read_str(self, start):
        b, end = self.read_bytes(start)
        return b.decode(), end

    def get_next_tag(self, start):
        min_depth = self.depth
        while True:
            if self.exp[start] == ord(")"):  # monitor xml element depth
                self.depth -= 1
                if min_depth > self.depth:
                    min_depth = self.depth
            elif self.exp[start] == ord("("):
                break
            start += 1
            if start >= len(self.exp):
                return None, start, 0

        self.depth += 1
        start += 1
        end = self.exp.find(ord(" "), start)
        return self.exp[start:end], end, min_depth

    def parse(self, exp):

        self.exp = exp  # used by other member functions
        self.depth = 0  # xml element depth
        self.world.step += 1
        self.world.line_count = 0
        self.world.robot.frp = {}
        self.world.flags_posts = {}
        self.world.flags_corners = {}
        self.world.vision_is_up_to_date = False
        self.world.Ball.IsVisible = False
        self.world.robot.feet_toes_are_touching = dict.fromkeys(
            self.world.robot.feet_toes_are_touching, False)
        self.world.time_local_ms += World.STEPTIME_MS

        for p in self.world.teammates:
            p.is_visible = False
        for p in self.world.opponents:
            p.is_visible = False

        tag, end, _ = self.get_next_tag(0)

        while end < len(exp):

            if tag == b'time':
                while True:
                    tag, end, min_depth = self.get_next_tag(end)
                    if min_depth == 0:
                        break

                    if tag == b'now':
                        # last_time = self.world.time_server
                        self.world.time_server, end = self.read_float(end+1)

                        # Test server time reliability
                        # increment = self.world.time_server - last_time
                        # if increment < 0.019: print ("down",last_time,self.world.time_server)
                        # if increment > 0.021: print ("up",last_time,self.world.time_server)
                    else:
                        self.world.log(
                            f"{self.file_name}Unknown tag inside 'time': {tag} at {end}, \nMsg: {exp.decode()}")

            elif tag == b'GS':
                while True:
                    tag, end, min_depth = self.get_next_tag(end)
                    if min_depth == 0:
                        break

                    if tag == b'unum':
                        # We already know our unum
                        _, end = self.read_int(end+1)
                    elif tag == b'team':
                        aux, end = self.read_str(end+1)
                        is_left = bool(aux == "left")
                        if self.world.team_side_is_left != is_left:
                            self.world.team_side_is_left = is_left
                            self.play_mode_to_id = self.LEFT_PLAY_MODE_TO_ID if is_left else self.RIGHT_PLAY_MODE_TO_ID
                            self.world.draw.set_team_side(not is_left)
                            self.world.team_draw.set_team_side(not is_left)
                    elif tag == b'sl':
                        if self.world.team_side_is_left:
                            self.world.goals_scored, end = self.read_int(end+1)
                        else:
                            self.world.goals_conceded, end = self.read_int(
                                end+1)
                    elif tag == b'sr':
                        if self.world.team_side_is_left:
                            self.world.goals_conceded, end = self.read_int(
                                end+1)
                        else:
                            self.world.goals_scored, end = self.read_int(end+1)
                    elif tag == b't':
                        self.world.time_game, end = self.read_float(end+1)
                    elif tag == b'pm':
                        aux, end = self.read_str(end+1)
                        if self.play_mode_to_id is not None:
                            self.world.play_mode = self.play_mode_to_id[aux]
                    else:
                        self.world.log(
                            f"{self.file_name}Unknown tag inside 'GS': {tag} at {end}, \nMsg: {exp.decode()}")

            elif tag == b'GYR':
                while True:
                    tag, end, min_depth = self.get_next_tag(end)
                    if min_depth == 0:
                        break

                    # The gyroscope measures the robot's torso angular velocity (rotation rate vector)
                    # The angular velocity's orientation is given by the right-hand rule.

                    # Original reference frame:
                    #     X:left(-)/right(+)      Y:back(-)/front(+)      Z:down(-)/up(+)

                    # New reference frame:
                    #     X:back(-)/front(+)      Y:right(-)/left(+)      Z:down(-)/up(+)
                    if tag == b'n':
                        pass
                    elif tag == b'rt':
                        self.world.robot.gyro[1], end = self.read_float(end+1)
                        self.world.robot.gyro[0], end = self.read_float(end+1)
                        self.world.robot.gyro[2], end = self.read_float(end+1)
                        self.world.robot.gyro[1] *= -1
                    else:
                        self.world.log(
                            f"{self.file_name}Unknown tag inside 'GYR': {tag} at {end}, \nMsg: {exp.decode()}")

            elif tag == b'ACC':
                while True:
                    tag, end, min_depth = self.get_next_tag(end)
                    if min_depth == 0:
                        break

                    # The accelerometer measures the acceleration relative to freefall. It will read zero during any type of free fall.
                    # When at rest relative to the Earth's surface, it will indicate an upwards acceleration of 9.81m/s^2 (in SimSpark).

                    # Original reference frame:
                    #     X:left(-)/right(+)      Y:back(-)/front(+)      Z:down(-)/up(+)

                    # New reference frame:
                    #     X:back(-)/front(+)      Y:right(-)/left(+)      Z:down(-)/up(+)

                    if tag == b'n':
                        pass
                    elif tag == b'a':
                        self.world.robot.acc[1], end = self.read_float(end+1)
                        self.world.robot.acc[0], end = self.read_float(end+1)
                        self.world.robot.acc[2], end = self.read_float(end+1)
                        self.world.robot.acc[1] *= -1
                    else:
                        self.world.log(
                            f"{self.file_name}Unknown tag inside 'ACC': {tag} at {end}, \nMsg: {exp.decode()}")

            elif tag == b'HJ':
                while True:
                    tag, end, min_depth = self.get_next_tag(end)
                    if min_depth == 0:
                        break

                    if tag == b'n':
                        joint_name, end = self.read_str(end+1)
                        joint_index = Robot.MAP_PERCEPTOR_TO_INDEX[joint_name]
                    elif tag == b'ax':
                        joint_angle, end = self.read_float(end+1)

                        # Fix symmetry issues 2/4 (perceptors)
                        if joint_name in Robot.FIX_PERCEPTOR_SET:
                            joint_angle = -joint_angle

                        old_angle = self.world.robot.joints[joint_index].position
                        self.world.robot.joints[joint_index].speed = (
                            joint_angle - old_angle) / World.STEPTIME * np.pi / 180
                        self.world.robot.joints[joint_index].position = joint_angle
                    else:
                        self.world.log(
                            f"{self.file_name}Unknown tag inside 'HJ': {tag} at {end}, \nMsg: {exp.decode()}")

            elif tag == b'FRP':
                while True:
                    tag, end, min_depth = self.get_next_tag(end)
                    if min_depth == 0:
                        break

                    # The reference frame is used for the contact point and force vector applied to that point
                    #     Note: The force vector is applied to the foot, so it usually points up

                    # Original reference frame:
                    #     X:left(-)/right(+)      Y:back(-)/front(+)      Z:down(-)/up(+)

                    # New reference frame:
                    #     X:back(-)/front(+)      Y:right(-)/left(+)      Z:down(-)/up(+)

                    if tag == b'n':
                        foot_toe_id, end = self.read_str(end+1)
                        self.world.robot.frp[foot_toe_id] = foot_toe_ref = np.empty(
                            6)
                        self.world.robot.feet_toes_last_touch[foot_toe_id] = self.world.time_local_ms
                        self.world.robot.feet_toes_are_touching[foot_toe_id] = True
                    elif tag == b'c':
                        foot_toe_ref[1], end = self.read_float(end+1)
                        foot_toe_ref[0], end = self.read_float(end+1)
                        foot_toe_ref[2], end = self.read_float(end+1)
                        foot_toe_ref[1] *= -1
                    elif tag == b'f':
                        foot_toe_ref[4], end = self.read_float(end+1)
                        foot_toe_ref[3], end = self.read_float(end+1)
                        foot_toe_ref[5], end = self.read_float(end+1)
                        foot_toe_ref[4] *= -1
                    else:
                        self.world.log(
                            f"{self.file_name}Unknown tag inside 'FRP': {tag} at {end}, \nMsg: {exp.decode()}")

            elif tag == b'See':
                self.world.vision_is_up_to_date = True
                self.world.vision_last_update = self.world.time_local_ms

                while True:
                    tag, end, min_depth = self.get_next_tag(end)
                    if min_depth == 0:
                        break

                    # since bytearray is not hashable, it cannot be used as key for dictionaries
                    tag_bytes = bytes(tag)

                    if tag in (b'G1R', b'G2R', b'G1L', b'G2L'):
                        _, end, _ = self.get_next_tag(end)

                        c1, end = self.read_float(end+1)
                        c2, end = self.read_float(end+1)
                        c3, end = self.read_float(end+1)

                        aux = self.LEFT_SIDE_FLAGS[tag_bytes] if self.world.team_side_is_left else self.RIGHT_SIDE_FLAGS[tag_bytes]
                        self.world.flags_posts[aux] = (c1, c2, c3)

                    elif tag in (b'F1R', b'F2R', b'F1L', b'F2L'):
                        _, end, _ = self.get_next_tag(end)

                        c1, end = self.read_float(end+1)
                        c2, end = self.read_float(end+1)
                        c3, end = self.read_float(end+1)

                        aux = self.LEFT_SIDE_FLAGS[tag_bytes] if self.world.team_side_is_left else self.RIGHT_SIDE_FLAGS[tag_bytes]
                        self.world.flags_corners[aux] = (c1, c2, c3)

                    elif tag == b'B':
                        _, end, _ = self.get_next_tag(end)

                        self.world.Ball.RelativeHeadSphPos[0], end = self.read_float(
                            end+1)
                        self.world.Ball.RelativeHeadSphPos[1], end = self.read_float(
                            end+1)
                        self.world.Ball.RelativeHeadSphPos[2], end = self.read_float(
                            end+1)
                        self.world.Ball.RelativeHeadCartPos = deg_sph2cart(
                            self.world.Ball.RelativeHeadSphPos)
                        self.world.Ball.IsVisible = True
                        self.world.Ball.LastSeen = self.world.time_local_ms

                    elif tag == b'mypos':

                        self.world.robot.cheat_abs_pos[0], end = self.read_float(
                            end+1)
                        self.world.robot.cheat_abs_pos[1], end = self.read_float(
                            end+1)
                        self.world.robot.cheat_abs_pos[2], end = self.read_float(
                            end+1)

                    elif tag == b'myorien':

                        self.world.robot.cheat_ori, end = self.read_float(
                            end+1)

                    elif tag == b'ballpos':

                        c1, end = self.read_float(end+1)
                        c2, end = self.read_float(end+1)
                        c3, end = self.read_float(end+1)

                        self.world.Ball.CheatAbsVel[0] = (
                            c1 - self.world.Ball.CheatAbsPos[0]) / World.VISUALSTEP
                        self.world.Ball.CheatAbsVel[1] = (
                            c2 - self.world.Ball.CheatAbsPos[1]) / World.VISUALSTEP
                        self.world.Ball.CheatAbsVel[2] = (
                            c3 - self.world.Ball.CheatAbsPos[2]) / World.VISUALSTEP

                        self.world.Ball.CheatAbsPos[0] = c1
                        self.world.Ball.CheatAbsPos[1] = c2
                        self.world.Ball.CheatAbsPos[2] = c3

                    elif tag == b'P':

                        while True:
                            previous_depth = self.depth
                            previous_end = end
                            tag, end, min_depth = self.get_next_tag(end)
                            if min_depth < 2:  # if =1 we are still inside 'See', if =0 we are already outside 'See'
                                # The "P" tag is special because it's the only variable particle inside 'See'
                                end = previous_end
                                self.depth = previous_depth
                                break  # we restore the previous tag, and let 'See' handle it

                            if tag == b'team':
                                player_team, end = self.read_str(end+1)
                                is_teammate = bool(
                                    player_team == self.world.team_name)
                                if self.world.team_name_opponent is None and not is_teammate:  # register opponent team name
                                    self.world.team_name_opponent = player_team
                            elif tag == b'id':
                                player_id, end = self.read_int(end+1)
                                player = self.world.teammates[player_id -
                                                              1] if is_teammate else self.world.opponents[player_id-1]
                                player.body_parts_cart_rel_pos = {}  # reset seen body parts
                                player.is_visible = True
                            elif tag in (b'llowerarm', b'rlowerarm', b'lfoot', b'rfoot', b'head'):
                                tag_str = tag.decode()
                                _, end, _ = self.get_next_tag(end)

                                c1, end = self.read_float(end+1)
                                c2, end = self.read_float(end+1)
                                c3, end = self.read_float(end+1)

                                if is_teammate:
                                    self.world.teammates[
                                        player_id - 1].body_parts_sph_rel_pos[tag_str] = (c1, c2, c3)
                                    self.world.teammates[player_id-1].body_parts_cart_rel_pos[tag_str] = deg_sph2cart(
                                        (c1, c2, c3))
                                else:
                                    self.world.opponents[
                                        player_id - 1].body_parts_sph_rel_pos[tag_str] = (c1, c2, c3)
                                    self.world.opponents[player_id-1].body_parts_cart_rel_pos[tag_str] = deg_sph2cart(
                                        (c1, c2, c3))
                            else:
                                self.world.log(
                                    f"{self.file_name}Unknown tag inside 'P': {tag} at {end}, \nMsg: {exp.decode()}")

                    elif tag == b'L':
                        l = self.world.lines[self.world.line_count]

                        _, end, _ = self.get_next_tag(end)
                        l[0], end = self.read_float(end+1)
                        l[1], end = self.read_float(end+1)
                        l[2], end = self.read_float(end+1)
                        _, end, _ = self.get_next_tag(end)
                        l[3], end = self.read_float(end+1)
                        l[4], end = self.read_float(end+1)
                        l[5], end = self.read_float(end+1)

                        if np.isnan(l).any():
                            self.world.log(
                                f"{self.file_name}Received field line with NaNs {l}")
                        else:
                            self.world.line_count += 1  # accept field line if there are no NaNs

                    else:
                        self.world.log(
                            f"{self.file_name}Unknown tag inside 'see': {tag} at {end}, \nMsg: {exp.decode()}")

            elif tag == b'hear':

                team_name, end = self.read_str(end+1)

                if team_name == self.world.team_name:   # discard message if it's not from our team

                    timestamp, end = self.read_float(end+1)

                    # this message was sent by oneself
                    if self.exp[end+1] == ord('s'):
                        direction, end = "self", end+5
                    else:                               # this message was sent by teammate
                        direction, end = self.read_float(end+1)

                    msg, end = self.read_bytes(end+1)
                    self.hear_callback(msg, direction, timestamp)

                tag, end, _ = self.get_next_tag(end)

            else:
                self.world.log(
                    f"{self.file_name}Unknown root tag: {tag} at {end}, \nMsg: {exp.decode()}")
                tag, end, min_depth = self.get_next_tag(end)

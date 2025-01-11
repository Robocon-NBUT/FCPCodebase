from functools import cmp_to_key

import numpy as np

# Note: When other robot is seen, all previous body part positions are deleted
# E.g. we see 5 body parts at 0 seconds -> body_parts_cart_rel_pos contains 5 elements
#      we see 1 body part  at 1 seconds -> body_parts_cart_rel_pos contains 1 element


class Other_Robot:
    def __init__(self, unum: int, is_teammate: bool) -> None:
        # convenient variable to indicate uniform number (same as other robot's index + 1)
        self.unum = unum
        self.is_self = False            # convenient flag to indicate if this robot is self
        # convenient variable to indicate if this robot is from our team
        self.is_teammate = is_teammate
        # True if this robot was seen in the last message from the server (it doesn't mean we know its absolute location)
        self.is_visible = False
        # cartesian relative position of the robot's visible body parts
        self.body_parts_cart_rel_pos = {}
        # spherical relative position of the robot's visible body parts
        self.body_parts_sph_rel_pos = {}
        # EMA filter coefficient applied to self.state_filtered_velocity
        self.vel_filter = 0.3
        # velocity decay at every vision cycle (neutralized if velocity is updated)
        self.vel_decay = 0.95

        # State variables: these are computed when this robot is visible and when the original robot is able to self-locate
        # true if the robot is lying down  (updated when head is visible)
        self.state_fallen = False
        # World.time_local_ms when the state was last updated
        self.state_last_update = 0
        # horizontal head distance if head is visible, otherwise, average horizontal distance of visible body parts (the distance is updated by vision or radio when state_abs_pos gets a new value, but also when the other player is not visible, by assuming its last position)
        self.state_horizontal_dist = 0
        # 3D head position if head is visible, otherwise, 2D average position of visible body parts, or, 2D radio head position
        self.state_abs_pos = None
        # orientation based on pair of lower arms or feet, or average of both (WARNING: may be older than state_last_update)
        self.state_orientation = 0
        # (pt_2d,radius) projection of player area on ground (circle), not precise if farther than 3m (for performance), useful for obstacle avoidance when it falls
        self.state_ground_area = None
        # 3D absolute position of each body part
        self.state_body_parts_abs_pos = {}
        # 3D filtered velocity (m/s) (if the head is not visible, the 2D part is updated and v.z decays)
        self.state_filtered_velocity = np.zeros(3)


class RobotList:
    def __init__(self, length: int, is_teammate: bool):
        self._robots = [Other_Robot(i + 1, is_teammate) for i in range(length)]
        self._ball_pos: np.ndarray = []
        self._time_local_ms = 0

    def distance(self, ball_pos: np.ndarray, time_local_ms: int):
        """
        获取队员与球员的距离
        """
        self._ball_pos = ball_pos
        self._time_local_ms = time_local_ms
        for robot in self._robots:
            yield self._single_distance(robot)

    def _single_distance(self, r: Other_Robot):
        # 如果对手不存在，或者状态信息不新（360毫秒），或者已经倒下，则强制设置为大距离
        if r.state_last_update != 0 and (self._time_local_ms - r.state_last_update <= 360 or r.is_self) and not r.state_fallen:
            return np.sum((r.state_abs_pos[:2] - self._ball_pos) ** 2)
        return 1000

    def _compare(self, r1: Other_Robot, r2: Other_Robot):
        """
        比较两个球员离球的距离
        """
        return self._single_distance(r1) - self._single_distance(r2)

    def sort_distance(self, ball_pos: np.ndarray, time_local_ms: int):
        """
        按照和球的距离对球员进行排序
        """
        self._ball_pos = ball_pos
        self._time_local_ms = time_local_ms
        _sorted_arr = sorted(self._robots, key=cmp_to_key(self._compare))
        new_list = RobotList(0, False)
        new_list._robots = _sorted_arr
        return new_list
        # self._robots = _sorted_arr
        # return self

    def __getitem__(self, index: int):
        return self._robots[index]

    def __iter__(self):
        return iter(self._robots)

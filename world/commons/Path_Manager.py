import numpy as np

from cpp.a_star import a_star
from math_ops.math_ext import (
    vector_angle, vector_from_angle, normalize_deg, normalize_vec)
from world.World import PlayMode, World, TheirMode


class Path_Manager:
    MODE_CAUTIOUS = 0
    MODE_DRIBBLE = 1    # safety margins are increased
    MODE_AGGRESSIVE = 2  # safety margins are reduced for opponents

    STATUS_SUCCESS = 0  # the pathfinding algorithm was executed normally
    # timeout before the target was reached (may be impossible)
    STATUS_TIMEOUT = 1
    # impossible to reach target (all options were tested)
    STATUS_IMPOSSIBLE = 2
    # no obstacles between start and target (path contains only 2 points: the start and target)
    STATUS_DIRECT = 3

    # hot start prediction distance (when walking)
    HOT_START_DIST_WALK = 0.05
    # hot start prediction distance (when dribbling)
    HOT_START_DIST_DRIBBLE = 0.10

    def __init__(self, world: World) -> None:
        self.world = world

        self._draw_obstacles = False   # enabled by function 'draw_options'
        self._draw_path = False        # enabled by function 'draw_options'
        self._use_team_channel = False  # enabled by function 'draw_options'

        # internal variables to bootstrap the path to start from a prediction (to reduce path instability)
        self.last_direction_rad = None
        self.last_update = 0
        self.last_start_dist = None

    def draw_options(self, enable_obstacles, enable_path, use_team_drawing_channel=False):
        '''
        Enable or disable drawings, and change drawing channel
        If self.world.draw.enable is False, these options are ignored

        Parameters
        ----------
        enable_obstacles : bool
            draw relevant obstacles for path planning
        enable_path : bool
            draw computed path
        use_team_drawing_channel : bool
            True to use team drawing channel, otherwise use individual channel
            Using individual channels for each player means that drawings with the same name can coexist
            With the team channel, drawings with the same name will replace previous drawings, even if drawn by a teammate
        '''
        self._draw_obstacles = enable_obstacles
        self._draw_path = enable_path
        self._use_team_channel = use_team_drawing_channel

    def get_obstacles(
            self, include_teammates, include_opponents, include_play_mode_restrictions,
            max_distance=4, max_age=500, ball_safety_margin=0, goalpost_safety_margin=0,
            mode=MODE_CAUTIOUS, priority_unums=[]):
        '''
        获取障碍物列表

        参数
        ----------
        include_teammates : bool
            是否包括队友在返回的障碍物列表中
        include_opponents : bool
            是否包括对手在返回的障碍物列表中
        max_distance : float
            只有距离小于 `max_distance` 米的队友或对手才会被考虑
        max_age : float
            只有在过去 `max_age` 毫秒内被看到的队友或对手才会被考虑
        ball_safety_margin : float
            球的软排斥半径的最小值
            当比赛停止时，以及球几乎出界时，此值会增加
            默认值为零，球被忽略
        goalpost_safety_margin : float
            对手球门周围的硬排斥半径
            默认值为零，使用最小边距
        mode : int
            对安全边距的总体态度（涉及队友和对手）
        priority_unums : list
            需要避免的队友列表（因为他们的角色更重要）

        返回值
        -------
        obstacles : list
            障碍物列表，每个障碍物是一个包含 5 个浮点数的元组（x, y, 硬半径, 软半径, 排斥力）
        '''
        w = self.world

        ball_2d = w.ball_abs_pos[:2]
        obstacles = []

        # 'comparator' 是 lambda 的局部变量，它捕获了 (w.time_local_ms - max_age) 的当前值
        check_age = lambda last_update, comparator = w.time_local_ms - \
            max_age: last_update > 0 and last_update >= comparator

        # ---------------------------------------------- 获取最近看到的近距离队友
        if include_teammates:
            # 软半径：排斥力在中心处最大并逐渐减弱
            soft_radius = 1.1 if mode == Path_Manager.MODE_DRIBBLE else 0.6

            def get_hard_radius(t):
                if t.unum in priority_unums:
                    return 1.0  # 优先角色的额外距离
                else:
                    return t.state_ground_area[1]+0.2

            # 获取近距离队友（中心，硬半径，软半径，力）
            obstacles.extend(
                (*t.state_ground_area[0],
                 get_hard_radius(t),
                 1.5 if t.unum in priority_unums else soft_radius,
                 1.0)  # 排斥力
                for t in w.teammates if not t.is_self and check_age(t.state_last_update) and t.state_horizontal_dist < max_distance)

        # ---------------------------------------------- 获取最近看到的近距离对手
        if include_opponents:

            # 软半径：排斥力在中心处最大并逐渐减弱
            if mode == Path_Manager.MODE_AGGRESSIVE:
                soft_radius = 0.6

                def hard_radius(o):
                    return 0.2
            elif mode == Path_Manager.MODE_DRIBBLE:
                soft_radius = 2.3

                def hard_radius(o):
                    return o.state_ground_area[1]+0.9
            else:
                soft_radius = 1.0

                def hard_radius(o):
                    return o.state_ground_area[1]+0.2

            # 获取近距离对手（中心，硬半径，软半径，力）
            obstacles.extend(
                (*o.state_ground_area[0],
                 hard_radius(o),
                 soft_radius,
                 # 排斥力（门将的额外值）
                 1.5 if o.unum == 1 else 1.0)
                for o in w.opponents if o.state_last_update > 0 and w.time_local_ms - o.state_last_update <= max_age and o.state_horizontal_dist < max_distance)

        # ---------------------------------------------- 获取比赛模式限制
        if include_play_mode_restrictions:
            if w.play_mode == TheirMode.GOAL_KICK:
                # 5 个圆形障碍物以覆盖对方球门区域
                obstacles.extend((15, i, 2.1, 0, 0) for i in range(-2, 3))
            elif w.play_mode == TheirMode.PASS:
                obstacles.append((*ball_2d, 1.2, 0, 0))
            elif w.play_mode in [TheirMode.KICK_IN, TheirMode.CORNER_KICK, TheirMode.FREE_KICK, TheirMode.DIR_FREE_KICK, TheirMode.OFFSIDE]:
                obstacles.append((*ball_2d, 2.5, 0, 0))

        # ---------------------------------------------- 获取球
        if ball_safety_margin > 0:
            # 在某些比赛场景下增加球的安全边距
            if (w.play_mode_group != PlayMode.OTHER) or abs(ball_2d[1]) > 9.5 or abs(ball_2d[0]) > 14.5:
                ball_safety_margin += 0.12

            obstacles.append((*ball_2d, 0, ball_safety_margin, 8))

        # ---------------------------------------------- 获取球门柱
        if goalpost_safety_margin > 0:
            obstacles.append((14.75, 1.10, goalpost_safety_margin, 0, 0))
            obstacles.append((14.75, -1.10, goalpost_safety_margin, 0, 0))

        # ---------------------------------------------- 绘制障碍物
        if self._draw_obstacles:
            d = w.team_draw if self._use_team_channel else w.draw
            if d.enabled:
                for o in obstacles:
                    if o[3] > 0:
                        d.circle(o[:2], o[3], o[4]/2, d.Color.orange,
                                 "path_obstacles", False)
                    if o[2] > 0:
                        d.circle(o[:2], o[2], 1, d.Color.red,
                                 "path_obstacles", False)
                d.flush("path_obstacles")

        return obstacles

    def _get_hot_start(self, start_distance):
        '''
        Get hot start position for path (considering the previous path)
        (as opposed to a cold start, where the path starts at the player)
        '''
        if self.last_update > 0 and self.world.time_local_ms - self.last_update == 20 and self.last_start_dist == start_distance:
            return self.world.robot.loc_head_position[:2] + vector_from_angle(self.last_direction_rad, is_rad=True) * start_distance
        else:
            # return cold start if start_distance was different or the position was not updated in the last step
            return self.world.robot.loc_head_position[:2]

    def _update_hot_start(self, next_dir_rad, start_distance):
        ''' Update hot start position for next run '''
        self.last_direction_rad = next_dir_rad
        self.last_update = self.world.time_local_ms
        self.last_start_dist = start_distance

    def _extract_target_from_path(self, path, path_len, ret_segments):
        ret_seg_ceil = int(np.ceil(ret_segments))

        if path_len >= ret_seg_ceil:
            i = ret_seg_ceil * 2  # path index of ceil point (x)
            if ret_seg_ceil == ret_segments:
                return path[i:i+2]
            else:
                floor_w = ret_seg_ceil-ret_segments
                return path[i-2:i] * floor_w + path[i:i+2] * (1-floor_w)
        else:
            return path[-2:]  # path end

    def get_path_to_ball(
            self, x_ori=None, x_dev=-0.2, y_dev=0, torso_ori=None, torso_ori_thrsh=1,
            priority_unums: list = [], is_aggressive=True, safety_margin=0.25, timeout=3000):
        '''
        获取从当前位置到球的下一个目标位置和方向（路径）

        参数
        ----------
        x_ori : float
            （此变量允许在自定义参考系中相对于球指定目标位置。）
            自定义参考系 x 轴的绝对方向
            如果为 None，则方向由向量（机器人->球）决定
        x_dev : float
            （此变量允许在自定义参考系中相对于球指定目标位置。）
            自定义参考系 x 轴上的目标位置偏移量
        y_dev : float
            （此变量允许在自定义参考系中相对于球指定目标位置。）
            自定义参考系 y 轴上的目标位置偏移量
        torso_ori : float
            机器人的目标绝对方向（参见 `torso_ori_thrsh`）
            如果为 None，则方向由向量（机器人->目标）决定
        torso_ori_thrsh : float
            当机器人与最终目标的距离小于 `torso_ori_thrsh` 米时，才会应用 `torso_ori`
            否则，机器人将朝向最终目标
        priority_unums : list
            需要避免的队友列表（因为他们的角色更重要）
        is_aggressive : bool
            如果为 True，则减少对对手的安全边距
        safety_margin : float
            球周围的排斥半径，以避免与球碰撞
        timeout : float
            最大执行时间（以微秒为单位）

        返回值
        -------
        next_pos : ndarray
            从路径到球的下一个绝对位置
        next_ori : float
            下一个绝对方向
        distance : float
            最小值（距离最终目标）和（距离球）

        示例
        -------
        -------------------------------------------------------------------------------------------
        x_ori        |  x_dev  |  y_dev  |  torso_ori  |  OBS
        -------------+---------+---------+-------------+-------------------------------------------
        None =>      |    -    |   !0    |      -      |  不推荐。无法收敛。
        (orient. of: |    0    |    0    |     None    |  正面追球，预期*慢速接近
        robot->ball) |    0    |    0    |    value    |  定向追球，预期*慢速接近
                     |   >0    |    0    |      -      |  不推荐。无法收敛。
                     |   <0    |    0    |     None    |  正面追球直到距离 == x_dev
                     |   <0    |    0    |    value    |  定向追球直到距离 == x_dev
        -------------+---------+---------+-------------+-------------------------------------------
        value        |    -    |    -    |     None    |  正面追点
                     |    -    |    -    |    value    |  定向追点
        -------------------------------------------------------------------------------------------
        * 取决于调用函数（预期在目标附近慢速行走）
        `torso_ori` 仅在机器人与最终目标的距离小于 `torso_ori_thrsh` 米时才会应用
        '''

        w = self.world
        r = w.robot
        dev = np.array([x_dev, y_dev])
        dev_len = np.linalg.norm(dev)
        dev_mult = 1

        # 如果机器人距离球超过 0.5 米且处于 PlayOn 模式，则使用球预测
        if np.linalg.norm(w.ball_abs_pos[:2] - r.loc_head_position[:2]) > 0.5 and w.play_mode_group == PlayMode.OTHER:
            # 交点，假设移动速度为 0.4 m/s
            ball_2d = w.get_intersection_point_with_ball(0.4)[0]
        else:
            ball_2d = w.ball_abs_pos[:2]

        # 自定义参考系方向
        vec_me_ball = ball_2d - r.loc_head_position[:2]
        if x_ori is None:
            x_ori = vector_angle(vec_me_ball)

        distance_boost = 0  # 返回距离目标的增量
        if torso_ori is not None and dev_len > 0:
            approach_ori_diff = abs(normalize_deg(
                r.imu_torso_orientation - torso_ori))
            if approach_ori_diff > 15:
                # 如果机器人远离接近方向，增加目标距离
                distance_boost = 0.15
            if approach_ori_diff > 30:
                # 如果机器人远离接近方向，增加目标到球的距离
                dev_mult = 1.3
            if approach_ori_diff > 45:
                # 如果机器人远离接近方向，增加球周围的安全边距
                safety_margin = max(0.32, safety_margin)

        # ------------------------------------------- 获取目标

        front_unit_vec = vector_from_angle(x_ori)
        left_unit_vec = np.array([-front_unit_vec[1], front_unit_vec[0]])

        rel_target = front_unit_vec * dev[0] + left_unit_vec * dev[1]
        target = ball_2d + rel_target * dev_mult
        target_vec = target - r.loc_head_position[:2]
        target_dist = np.linalg.norm(target_vec)

        if self._draw_path:
            d = self.world.team_draw if self._use_team_channel else self.world.draw
            # 如果绘图对象未被内部禁用，则绘制目标点
            d.point(target, 4, d.Color.red, "path_target")

        # ------------------------------------------- 获取障碍物

        # 如果距离球的偏移量大于 0，则忽略球（在目标的同一侧）
        if dev_len > 0 and np.dot(vec_me_ball, rel_target) < -0.10:
            safety_margin = 0

        obstacles = self.get_obstacles(
            include_teammates=True, include_opponents=True, include_play_mode_restrictions=True,
            ball_safety_margin=safety_margin,
            mode=Path_Manager.MODE_AGGRESSIVE if is_aggressive else Path_Manager.MODE_CAUTIOUS,
            priority_unums=priority_unums)

        # 在目标相对的一侧添加障碍物
        if dev_len > 0 and safety_margin > 0:
            center = ball_2d - normalize_vec(rel_target) * safety_margin
            obstacles.append((*center, 0, safety_margin*0.9, 5))
            if self._draw_obstacles:
                d = w.team_draw if self._use_team_channel else w.draw
                if d.enabled:
                    d.circle(center, safety_margin*0.8, 2.5,
                             d.Color.orange, "path_obstacles_1")

        # ------------------------------------------- 获取路径

        # 根据目标距离决定起始位置
        start_pos = self._get_hot_start(
            Path_Manager.HOT_START_DIST_WALK) if target_dist > 0.4 else w.robot.loc_head_position[:2]

        path, path_len, path_status, path_cost = self.get_path(
            start_pos, True, obstacles, target, timeout)
        path_end = path[-2:]  # A* 算法允许的最后位置

        # ------------------------------------------- 获取相关距离

        if w.ball_last_seen > w.time_local_ms - w.VISUALSTEP_MS:  # 球在视野中
            # - 机器人中心与球中心之间的距离
            raw_ball_dist = np.linalg.norm(w.ball_rel_torso_cart_pos[:2])
        # 否则使用绝对坐标计算距离
        else:
            # - 头部中心与球中心之间的距离
            raw_ball_dist = np.linalg.norm(vec_me_ball)

        avoid_touching_ball = (w.play_mode_group != PlayMode.OTHER)
        distance_to_final_target = np.linalg.norm(
            path_end - r.loc_head_position[:2])
        distance_to_ball = max(
            0.07 if avoid_touching_ball else 0.14, raw_ball_dist - 0.13)
        caution_dist = min(distance_to_ball, distance_to_final_target)

        # ------------------------------------------- 获取下一个目标位置

        next_pos = self._extract_target_from_path(
            path, path_len, ret_segments=1 if caution_dist < 1 else 2)

        # ------------------------------------------ 获取下一个目标方向

        # 如果给定了方向，则使用给定的方向；否则，如果目标距离足够远，则使用目标的方向；否则使用当前方向
        if torso_ori is not None:
            if caution_dist > torso_ori_thrsh:
                next_ori = vector_angle(target_vec)
            else:
                mid_ori = normalize_deg(vector_angle(
                    vec_me_ball) - vector_angle(-dev) - x_ori + torso_ori)
                mid_ori_diff = abs(normalize_deg(
                    mid_ori - r.imu_torso_orientation))
                final_ori_diff = abs(normalize_deg(
                    torso_ori - r.imu_torso_orientation))
                next_ori = mid_ori if mid_ori_diff + 10 < final_ori_diff else torso_ori
        elif target_dist > 0.1:
            next_ori = vector_angle(target_vec)
        else:
            next_ori = r.imu_torso_orientation

        # ------------------------------------------ 更新下一次运行的热启动

        # 定义热启动距离：
        # 如果路径长度为零，则没有热启动，因为我们已经在目标位置（距离=0）
        # 如果目标很近，则不应用热启动（见上文）
        # 如果下一个位置非常近（由于硬障碍物），
        # 热启动位置是下一个位置（距离 < Path_Manager.HOT_START_DIST_WALK）
        # 否则，热启动距离是常数（距离 = Path_Manager.HOT_START_DIST_WALK）
        if path_len != 0:
            next_pos_vec = next_pos - w.robot.loc_head_position[:2]
            next_pos_dist = np.linalg.norm(next_pos_vec)
            self._update_hot_start(vector_angle(next_pos_vec, is_rad=True), min(
                Path_Manager.HOT_START_DIST_WALK, next_pos_dist))

        return next_pos, next_ori, min(distance_to_ball, distance_to_final_target + distance_boost)

    def get_path_to_target(
            self, target, ret_segments=1.0, torso_ori=None,
            priority_unums: list = [], is_aggressive=True, timeout=3000):
        '''
        获取从当前位置到目标位置的下一个绝对位置和下一个绝对方向（路径）

        参数
        ----------
        ret_segments : float
            返回的目标的最大距离（以路径段为单位，从起始位置开始计算）
            实际距离为 min(ret_segments, path_length)
            每个路径段的长度为 0.10 米（如果是对角线则为 0.1*sqrt(2) 米）
            如果 `ret_segments` 为 0，则返回当前的位置
        torso_ori : float
            机器人的目标绝对方向
            如果为 None，则方向由向量（机器人到目标）决定
        priority_unums : list
            需要避免的队友列表（因为他们的角色更重要）
        is_aggressive : bool
            如果为 True，则减少对对手的安全边距
        timeout : float
            最大执行时间（以微秒为单位）
        '''

        w = self.world

        # ------------------------------------------- 获取目标

        target_vec = target - w.robot.loc_head_position[:2]  # 计算目标向量
        target_dist = np.linalg.norm(target_vec)  # 计算目标距离

        # ------------------------------------------- 获取障碍物

        # 获取障碍物，是否包括队友、对手和比赛模式限制
        obstacles = self.get_obstacles(
            include_teammates=True, include_opponents=True, include_play_mode_restrictions=True,
            mode=Path_Manager.MODE_AGGRESSIVE if is_aggressive else Path_Manager.MODE_CAUTIOUS, priority_unums=priority_unums)

        # ------------------------------------------- 获取路径

        # 根据目标距离决定起始位置
        start_pos = self._get_hot_start(
            Path_Manager.HOT_START_DIST_WALK) if target_dist > 0.4 else w.robot.loc_head_position[:2]

        # 通过 `get_path` 函数获取从起始位置到目标的路径
        path, path_len, path_status, path_cost = self.get_path(
            start_pos, True, obstacles, target, timeout)
        path_end = path[-2:]  # A* 算法允许的最后位置

        # ------------------------------------------- 获取下一个目标位置

        # 从路径中提取下一个目标位置
        next_pos = self._extract_target_from_path(path, path_len, ret_segments)

        # ------------------------------------------ 获取下一个目标方向

        # 如果给定了方向，则使用给定的方向；否则，如果目标距离足够远，则使用目标的方向；否则使用当前方向
        if torso_ori is not None:
            next_ori = torso_ori
        elif target_dist > 0.1:
            next_ori = vector_angle(target_vec)
        else:
            next_ori = w.robot.imu_torso_orientation

        # ------------------------------------------ 更新下一次运行的热启动

        # 定义热启动距离：
        # 如果路径长度为零，则没有热启动，因为我们已经在目标位置（距离=0）
        # 如果目标很近，则不应用热启动（见上文）
        # 如果下一个位置非常近（由于硬障碍物），
        # 热启动位置是下一个位置（距离 < Path_Manager.HOT_START_DIST_WALK）
        # 否则，热启动距离是常数（距离 = Path_Manager.HOT_START_DIST_WALK）
        if path_len != 0:
            next_pos_vec = next_pos - w.robot.loc_head_position[:2]
            next_pos_dist = np.linalg.norm(next_pos_vec)
            self._update_hot_start(vector_angle(next_pos_vec, is_rad=True), min(
                Path_Manager.HOT_START_DIST_WALK, next_pos_dist))

        # 计算到最终目标的距离
        distance_to_final_target = np.linalg.norm(
            path_end - w.robot.loc_head_position[:2])

        return next_pos, next_ori, distance_to_final_target

    def get_dribble_path(self, ret_segments=None, optional_2d_target=None, goalpost_safety_margin=0.4, timeout=3000):
        '''
        Get next position from path to target (next relative orientation)
        Path is optimized for dribble

        Parameters
        ----------
        ret_segments : float
            returned target's maximum distance (measured in path segments from hot start position)
            actual distance: min(ret_segments,path_length)
            each path segment has 0.10 m or 0.1*sqrt(2) m (if diagonal)
            if `ret_segments` is 0, the current position is returned
            if `ret_segments` is None, it is dynamically set according to the robot's speed
        optional_2d_target : float
            2D target
            if None, the target is the opponent's goal (the specific goal point is decided by the A* algorithm)
        goalpost_safety_margin : float
            hard repulsion radius around the opponents' goalposts
            if zero, the minimum margin is used
        timeout : float
            maximum execution time (in microseconds)
        '''

        r = self.world.robot
        ball_2d = self.world.ball_abs_pos[:2]

        # ------------------------------------------- get obstacles

        obstacles = self.get_obstacles(
            include_teammates=True, include_opponents=True, include_play_mode_restrictions=False,
            max_distance=5, max_age=1000, goalpost_safety_margin=goalpost_safety_margin,
            mode=Path_Manager.MODE_DRIBBLE)

        # ------------------------------------------- get path

        start_pos = self._get_hot_start(Path_Manager.HOT_START_DIST_DRIBBLE)

        path, path_len, path_status, path_cost = self.get_path(
            start_pos, False, obstacles, optional_2d_target, timeout)

        # ------------------------------------------- get next target position & orientation

        if ret_segments is None:
            ret_segments = 2.0

        next_pos = self._extract_target_from_path(path, path_len, ret_segments)
        next_rel_ori = normalize_deg(vector_angle(
            next_pos - ball_2d) - r.imu_torso_orientation)

        # ------------------------------------------ update hot start for next run

        if path_len != 0:
            self._update_hot_start(np.deg2rad(
                r.imu_torso_orientation), Path_Manager.HOT_START_DIST_DRIBBLE)

        # ------------------------------------------ draw
        if self._draw_path and path_status != Path_Manager.STATUS_DIRECT:
            d = self.world.team_draw if self._use_team_channel else self.world.draw
            # will not draw if drawing object is internally disabled
            d.point(next_pos, 2, d.Color.pink, "path_next_pos", False)
            # will not draw if drawing object is internally disabled
            d.line(ball_2d, next_pos, 2, d.Color.pink, "path_next_pos")

        return next_pos, next_rel_ori

    def get_push_path(self, ret_segments=1.5, optional_2d_target=None, avoid_opponents=False, timeout=3000):
        '''
        获取从当前球位置到目标位置的下一个绝对位置（优化路径用于关键推球）

        参数
        ----------
        ret_segments : float
            返回的目标的最大距离（以路径段为单位，从起始位置开始计算）
            实际距离为 min(ret_segments, path_length)
            每个路径段的长度为 0.10 米（如果是对角线则为 0.1*sqrt(2) 米）
            如果 `ret_segments` 为 0，则返回当前的位置
        optional_2d_target : float
            2D 目标点
            如果为 None，目标点将是对方的球门（具体的目标点由 A* 算法决定）
        avoid_opponents : bool
            是否避免对方球员
        timeout : float
            最大执行时间（以微秒为单位）
        '''

        # 获取球的当前绝对位置（2D 坐标）
        ball_2d = self.world.ball_abs_pos[:2]

        # ------------------------------------------- 获取障碍物

        # 获取障碍物，是否包括队友、对手和比赛模式限制
        obstacles = self.get_obstacles(False, avoid_opponents, False)

        # ------------------------------------------- 获取路径

        # 通过 `get_path` 函数获取从球的位置到目标的路径
        path, path_len, path_status, path_cost = self.get_path(
            ball_2d, False, obstacles, optional_2d_target, timeout)

        # ------------------------------------------- 获取下一个目标位置

        # 从路径中提取下一个目标位置
        next_pos = self._extract_target_from_path(path, path_len, ret_segments)

        return next_pos

    def get_path(self, start, allow_out_of_bounds, obstacles=[], optional_2d_target=None, timeout=3000):
        '''
        参数
        ----------
        allow_out_of_bounds : bool
            是否允许路径越界，当带球时应为 False
        obstacles : list
            障碍物列表，每个障碍物是一个包含 5 个浮点数 (x, y, 硬半径, 软半径, 排斥力) 的元组
        optional_2d_target : float
            2D 目标点
            如果为 None，目标点将是对方的球门（具体的目标点由 A* 算法决定）
        timeout : float
            最大执行时间（以微秒为单位）
        '''

        # 如果没有指定 2D 目标点，则目标点为对方球门
        go_to_goal = int(optional_2d_target is None)

        # 如果没有指定 2D 目标点，默认设置为 (0, 0)
        if optional_2d_target is None:
            optional_2d_target = (0, 0)

        # 将障碍物列表展平成一个元组
        obstacles = sum(obstacles, tuple())
        # 确保每个障碍物由恰好 5 个浮点值描述
        assert len(obstacles) % 5 == 0, "每个障碍物应该由正好 5 个浮点数描述"

        # 路径参数：起点、是否允许越界、是否前往球门、可选目标点、超时时间（微秒）、障碍物
        params = np.array([*start, int(allow_out_of_bounds), go_to_goal,
                           *optional_2d_target, timeout, *obstacles], np.float32)
        # 使用 A* 算法计算路径
        path_ret = a_star.compute(params)
        # 获取路径（去掉最后两个返回值）
        path = path_ret[:-2]
        # 获取路径状态
        path_status = path_ret[-2]

        # ---------------------------------------------- 绘制路径段
        if self._draw_path:
            d = self.world.team_draw if self._use_team_channel else self.world.draw
            if d.enabled:
                # 根据路径状态选择颜色：绿色草坪、黄色、红色、青色
                c = {0: d.Color.green_lawn, 1: d.Color.yellow,
                     2: d.Color.red, 3: d.Color.cyan}[path_status]
                # 绘制路径段
                for j in range(0, len(path)-2, 2):
                    d.line((path[j], path[j+1]), (path[j+2],
                                                  path[j+3]), 1, c, "path_segments", False)
                # 刷新绘图
                d.flush("path_segments")

        # 返回路径、路径段数量、路径状态、路径成本（A* 成本）
        return path, len(path)//2-1, path_status, path_ret[-1]

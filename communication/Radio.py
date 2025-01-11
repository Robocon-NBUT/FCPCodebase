import numpy as np

from world.Other_Robot import Other_Robot
from world.World import World


class Radio:
    '''
    地图限制是硬编码的：
        队友/对手的位置 (x,y) 在 ([-16,16],[-11,11]) 范围内
        球的位置 (x,y) 在 ([-15,15],[-10,10]) 范围内
    已知服务器限制：
        声称：所有从 0x20 到 0x7E 的 ASCII 字符，除了 ' ', '(', ')'
        已知错误：
            - 单引号 ' 或双引号 " 会截断消息
            - 反斜杠 '\' 在结尾或接近另一个反斜杠时出错
            - 消息开头的 ';' 会出错
    '''
    # 地图限制是硬编码的：

    # 定义各种参数和常量，用于编码和解码位置数据
    TP = 321, 221, 160, 110, 10,  10, 70941, 141882  # 队友位置参数
    OP = 201, 111, 100, 55, 6.25, 5, 22311, 44622    # 对手位置参数
    BP = 301, 201, 150, 100, 10,  10, 60501          # 球位置参数

    # 定义可用的符号集，用于编码消息
    SYMB = "!#$%&*+,-./0123456789:<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[]^_`abcdefghijklmnopqrstuvwxyz{|}~;"
    SLEN = len(SYMB)
    SYMB_TO_IDX = {ord(s): i for i, s in enumerate(SYMB)}  # 符号到索引的映射

    def __init__(self, world: World, commit_announcement) -> None:
        self.world = world
        self.commit_announcement = commit_announcement
        r = world.robot
        t = world.teammates
        o = world.opponents
        # 定义不同的组，每个组包含特定的队友和对手
        self.groups = (
            # 组0：包含2个队友，5个对手和球
            [(t[9], t[10], o[6], o[7], o[8], o[9], o[10]), True],
            # 组1：包含7个队友
            [(t[0], t[1], t[2], t[3], t[4], t[5], t[6]), False],
            # 组2：包含2个队友，6个对手
            [(t[7], t[8], o[0], o[1], o[2], o[3], o[4], o[5]), False]
        )
        # 检查每个组中是否包含自身
        for g in self.groups:
            g.append(any(i.is_self for i in g[0]))

    def get_player_combination(self, pos, is_unknown, is_down, info):
        '''
        获取玩家位置的组合索引

        参数：
        ----------
        pos : array
            玩家的位置坐标
        is_unknown : bool
            玩家位置是否未知
        is_down : bool
            玩家是否倒下
        info : tuple
            玩家位置的相关参数（根据是队友还是对手有所不同）

        返回值：
        -------
        combination : int
            位置的组合索引
        total_combinations : int
            可能的组合总数
        '''
        if is_unknown:
            return info[7] + 1, info[7] + 2  # 未知位置的组合索引

        x, y = pos[:2]

        if x < -17 or x > 17 or y < -12 or y > 12:
            # 超出边界的位置组合索引
            return info[7], info[7] + 2

        # 将坐标转换为整数索引，并限制在有效范围内
        l = int(np.clip(round(info[4] * x + info[2]), 0, info[0] - 1))
        c = int(np.clip(round(info[5] * y + info[3]), 0, info[1] - 1))

        # 返回有效的组合索引
        return (l * info[1] + c) + (info[6] if is_down else 0), info[7] + 2

    def get_ball_combination(self, x, y):
        '''
        获取球位置的组合索引

        参数：
        ----------
        x : float
            球的x坐标
        y : float
            球的y坐标

        返回值：
        -------
        combination : int
            球位置的组合索引
        total_combinations : int
            可能的组合总数
        '''
        # 将球的位置坐标转换为整数索引，并限制在有效范围内
        l = int(
            np.clip(round(Radio.BP[4] * x + Radio.BP[2]), 0, Radio.BP[0] - 1))
        c = int(
            np.clip(round(Radio.BP[5] * y + Radio.BP[3]), 0, Radio.BP[1] - 1))

        # 返回有效的组合索引
        return l * Radio.BP[1] + c, Radio.BP[6]

    def get_ball_position(self, comb):
        '''
        根据组合索引获取球的位置坐标

        参数：
        ----------
        comb : int
            球位置的组合索引

        返回值：
        -------
        position : np.array
            球的位置坐标数组 [x, y, z]
        '''
        l = comb // Radio.BP[1]
        c = comb % Radio.BP[1]

        # 假设球在地面上，z坐标为固定值
        return np.array([l / Radio.BP[4] - 15, c / Radio.BP[5] - 10, 0.042])

    def get_player_position(self, comb, info):
        '''
        根据组合索引获取玩家的位置和状态

        参数：
        ----------
        comb : int
            玩家位置的组合索引
        info : tuple
            玩家位置的相关参数（根据是队友还是对手有所不同）

        返回值：
        -------
        result : tuple 或 int
            如果位置有效，返回 (x, y, is_down)
            如果位置无效，返回 -1 或 -2 表示出界或未知位置
        '''
        if comb == info[7]:
            return -1  # 玩家出界
        if comb == info[7] + 1:
            return -2  # 玩家位置未知

        is_down = comb >= info[6]
        if is_down:
            comb -= info[6]

        l = comb // info[1]
        c = comb % info[1]

        # 计算实际坐标
        return l / info[4] - 16, c / info[5] - 11, is_down

    def check_broadcast_requirements(self):
        '''
        检查广播组是否有效

        Returns

        -------
        ready: bool
            如果所有要求都得到满足则返回 True

        Check if broadcast group is valid

        Returns
        -------
        ready : bool
            True if all requirements are met

        Sequence: g0,g1,g2, ig0,ig1,ig2, iig0,iig1,iig2  (whole cycle: 0.36s)
            igx  means      'incomplete group', where <=1 element  can be MIA recently
            iigx means 'very incomplete group', where <=2 elements can be MIA recently
            Rationale: prevent incomplete messages from monopolizing the broadcast space 

        However:
        - 1st round: when 0 group  members are missing,          that group will update 3 times every 0.36s
        - 2nd round: when 1 group  member  is  recently missing, that group will update 2 times every 0.36s
        - 3rd round: when 2 group  members are recently missing, that group will update 1 time  every 0.36s
        -            when >2 group members are recently missing, that group will not be updated

        Players that have never been seen or heard are not considered for the 'recently missing'.
        If there is only 1 group member since the beginning, the respective group can be updated, except in the 1st round.
        In this way, the 1st round cannot be monopolized by clueless agents, which is important during games with 22 players.
        '''

        w = self.world
        r = w.robot
        ago40ms = w.time_local_ms - 40
        ago370ms = w.time_local_ms - 370

        # 根据服务器时间确定当前的消息序列和允许的缺失成员数量
        idx9 = int((w.time_server * 25) + 0.1) % 9  # 9个阶段的序列
        max_MIA = idx9 // 3  # 允许的最大缺失成员数
        group_idx = idx9 % 3  # 当前组索引
        group, has_ball, is_self_included = self.groups[group_idx]

        # 检查球的位置是否最新
        if has_ball and w.Ball.AbsolutePosLastUpdate < ago40ms:
            return False

        # 检查自身的位置是否最新
        if is_self_included and r.location.last_update < ago40ms:
            return False

        # 检查组成员的状态，统计最近缺失的成员数量
        MIAs = [
            not ot.is_self and ot.state_last_update < ago370ms and ot.state_last_update > 0
            for ot in group
        ]
        self.MIAs = [
            ot.state_last_update == 0 or MIAs[i]
            for i, ot in enumerate(group)
        ]

        # 如果缺失成员数量超过允许的最大值，返回 False
        if sum(MIAs) > max_MIA:
            return False

        # 在特定条件下，如果有未曾见过的成员，也返回 False
        if (max_MIA == 0 and any(self.MIAs)) or all(self.MIAs):
            return False

        # 检查成员的数据是否有效和最新
        if any(
            not ot.is_self and not self.MIAs[i] and (
                ot.state_last_update < ago40ms or
                ot.state_last_update == 0 or
                len(ot.state_abs_pos) < 3
            )
            for i, ot in enumerate(group)
        ):
            return False

        return True

    def broadcast(self):
        '''
        如果满足条件，发送广播消息给队友
        消息包含所有移动实体的位置和状态
        '''
        if not self.check_broadcast_requirements():
            return

        w = self.world

        # 根据服务器时间确定当前组索引
        group_idx = int((w.time_server * 25) + 0.1) % 3
        group, has_ball, _ = self.groups[group_idx]

        # 初始化组合索引和总组合数
        combination = group_idx
        no_of_combinations = 3

        # 添加球的位置组合索引
        if has_ball:
            c, n = self.get_ball_combination(
                w.Ball.AbsolutePos[0], w.Ball.AbsolutePos[1]
            )
            combination += c * no_of_combinations
            no_of_combinations *= n

        # 添加组成员的位置组合索引
        for i, ot in enumerate(group):
            c, n = self.get_player_combination(
                ot.state_abs_pos,                       # 玩家位置
                self.MIAs[i],                           # 位置是否未知
                ot.state_fallen,                        # 玩家是否倒下
                Radio.TP if ot.is_teammate else Radio.OP  # 使用队友或对手的参数
            )
            combination += c * no_of_combinations
            no_of_combinations *= n

        # 确保组合数不超过限制
        assert (no_of_combinations < 9.61e38)

        # 构建消息字符串
        msg = Radio.SYMB[combination % (Radio.SLEN - 1)]
        combination //= (Radio.SLEN - 1)

        while combination:
            msg += Radio.SYMB[combination % Radio.SLEN]
            combination //= Radio.SLEN

        # 提交广播消息
        self.commit_announcement(msg.encode())

    def receive(self, msg: bytearray):
        '''
        接收并解析广播消息，更新世界状态

        参数：
        ----------
        msg : bytearray
            接收到的广播消息
        '''
        w = self.world
        r = w.robot
        ago40ms = w.time_local_ms - 40
        ago110ms = w.time_local_ms - 110
        msg_time = w.time_local_ms - 20  # 消息在上一个时间步发送

        # 解析组合索引
        combination = Radio.SYMB_TO_IDX[msg[0]]
        total_combinations = Radio.SLEN - 1

        if len(msg) > 1:
            for m in msg[1:]:
                combination += total_combinations * Radio.SYMB_TO_IDX[m]
                total_combinations *= Radio.SLEN

        # 获取消息编号和对应的组
        message_no = combination % 3
        combination //= 3
        group, has_ball, _ = self.groups[message_no]

        # 解析球的位置组合索引
        if has_ball:
            ball_comb = combination % Radio.BP[6]
            combination //= Radio.BP[6]

        # 解析组成员的位置组合索引
        players_combs = []
        for ot in group:
            info = Radio.TP if ot.is_teammate else Radio.OP
            players_combs.append(combination % (info[7] + 2))
            combination //= info[7] + 2

        # 更新球的位置
        if has_ball and w.Ball.AbsolutePosLastUpdate < ago40ms:
            time_diff = (msg_time - w.Ball.AbsolutePosLastUpdate) / 1000
            ball = self.get_ball_position(ball_comb)
            w.Ball.AbsoluteVel = (ball - w.Ball.AbsolutePos) / time_diff
            w.Ball.AbsoluteSpeed = np.linalg.norm(w.Ball.AbsoluteVel)
            w.Ball.AbsolutePosLastUpdate = msg_time  # 误差：0-40 ms
            w.Ball.AbsolutePos = ball
            w.Ball.IsFromVision = False

        # 更新组成员的位置
        for c, ot in zip(players_combs, group):
            # 如果是自身，特殊处理
            if ot.is_self:
                if r.location.last_update < ago110ms:
                    data = self.get_player_position(c, Radio.TP)
                    if isinstance(data, tuple):
                        x, y, is_down = data
                        r.location.Head.Position[:2] = x, y  # 保持z坐标不变
                        r.location.Head.PositionLastUpdate = msg_time
                        r.radio_fallen_state = is_down
                        r.radio_last_update = msg_time
                continue

            # 如果其他机器人最近被看到，不更新
            if ot.state_last_update >= ago40ms:
                continue

            info = Radio.TP if ot.is_teammate else Radio.OP
            data = self.get_player_position(c, info)
            if isinstance(data, tuple):
                x, y, is_down = data
                p = np.array([x, y])

                # 更新速度信息
                if ot.state_abs_pos is not None:
                    time_diff = (msg_time - ot.state_last_update) / 1000
                    velocity = np.append(
                        (p - ot.state_abs_pos[:2]) / time_diff, 0)  # v.z = 0
                    vel_diff = velocity - ot.state_filtered_velocity
                    if np.linalg.norm(vel_diff) < 4:
                        ot.state_filtered_velocity /= (
                            ot.vel_decay, ot.vel_decay, 1)
                        ot.state_filtered_velocity += ot.vel_filter * vel_diff

                ot.state_fallen = is_down
                ot.state_last_update = msg_time
                ot.state_body_parts_abs_pos = {"head": p}
                ot.state_abs_pos = p
                ot.state_horizontal_dist = np.linalg.norm(
                    p - r.location.Head.Position[:2])
                ot.state_ground_area = (p, 0.3 if is_down else 0.2)

import numpy as np
from math_ops.Matrix_3x3 import Matrix_3x3
from math_ops.math_ext import acos

class Inverse_Kinematics():

    # leg y deviation, upper leg height, upper leg depth, lower leg length, knee extra angle, max ankle z
    NAO_SPECS_PER_ROBOT = (
        (0.055,      0.12,        0.005, 0.1,         np.arctan(0.005/0.12),        -0.091),
        (0.055,      0.13832,     0.005, 0.11832,     np.arctan(0.005/0.13832),     -0.106),
        (0.055,      0.12,        0.005, 0.1,         np.arctan(0.005/0.12),        -0.091),
        (0.072954143,0.147868424, 0.005, 0.127868424, np.arctan(0.005/0.147868424), -0.114),
        (0.055,      0.12,        0.005, 0.1,         np.arctan(0.005/0.12),        -0.091))

    TORSO_HIP_Z = 0.115 # distance in the z-axis, between the torso and each hip (same for all robots)
    TORSO_HIP_X = 0.01  # distance in the x-axis, between the torso and each hip (same for all robots) (hip is 0.01m to the back)

    def __init__(self, robot) -> None:
        self.robot = robot
        self.NAO_SPECS = Inverse_Kinematics.NAO_SPECS_PER_ROBOT[robot.type]

    def torso_to_hip_transform(self, coords, is_batch=False):
        '''
        Convert cartesian coordinates that are relative to torso to coordinates
        that are relative the center of both hip joints
        
        Parameters
        ----------
        coords : array_like
            One 3D position or list of 3D positions
        is_batch : `bool`
            Indicates if coords is a batch of 3D positions

        Returns
        -------
        coord : `list` or ndarray
            A numpy array is returned if is_batch is False, otherwise, a list of arrays is returned   
        '''
        if is_batch:
            return [c + (Inverse_Kinematics.TORSO_HIP_X, 0, Inverse_Kinematics.TORSO_HIP_Z) for c in coords]
        else:
            return coords + (Inverse_Kinematics.TORSO_HIP_X, 0, Inverse_Kinematics.TORSO_HIP_Z)


    def head_to_hip_transform(self, coords, is_batch=False):
        '''
        Convert cartesian coordinates that are relative to head to coordinates
        that are relative the center of both hip joints
        
        Parameters
        ----------
        coords : array_like
            One 3D position or list of 3D positions
        is_batch : `bool`
            Indicates if coords is a batch of 3D positions

        Returns
        -------
        coord : `list` or ndarray
            A numpy array is returned if is_batch is False, otherwise, a list of arrays is returned   
        '''
        coords_rel_torso = self.robot.head_to_body_part_transform( "torso", coords, is_batch )
        return self.torso_to_hip_transform(coords_rel_torso, is_batch)

    def get_body_part_pos_relative_to_hip(self, body_part_name):
        ''' Get body part position relative to the center of both hip joints '''
        bp_rel_head = self.robot.body_parts[body_part_name].transform.get_translation()
        return self.head_to_hip_transform(bp_rel_head)

    def get_ankle_pos_relative_to_hip(self, is_left):
        ''' Internally calls get_body_part_pos_relative_to_hip() '''
        return self.get_body_part_pos_relative_to_hip("lankle" if is_left else "rankle")

    def get_linear_leg_trajectory(
            self, is_left: bool, p1, p2=None, foot_ori3d=(0, 0, 0),
            dynamic_pose: bool = True, resolution=100):
        """
        计算腿部轨迹，使脚踝在两个3D点（相对于髋部）之间线性移动

        参数
        ----------
        is_left : `bool`
            设置为True选择左腿，设置为False选择右腿
        p1 : array_like, 长度为3
            如果p2为None:
                p1是目标位置（相对于髋部），起始点由脚踝的当前位置给出
            如果p2不为None:
                p1是起始点（相对于髋部）
        p2 : array_like, 长度为3 / `None`
            目标位置（相对于髋部）或None（参见p1）
        foot_ori3d : array_like, 长度为3
            绕x, y, z轴的旋转（绕x和y轴的旋转是相对于垂直姿势或启用动态姿势时的偏移）
        dynamic_pose : `bool`
            启用基于IMU的动态脚旋转，使其与地面平行
        resolution : int
            插值分辨率；更高的分辨率总是更好，但需要更多的计算时间；
            拥有更多的点并不会使运动变慢，因为如果有过多的点，它们会在分析优化过程中被删除

        返回
        -------
        trajecory : `tuple`
            索引, [[values_1,error_codes_1], [values_2,error_codes_2], ...]
            有关详细信息，请参见leg()函数
        """

        if p2 is None:
            # 如果p2为None，则将p1视为目标位置，并获取当前脚踝相对于髋部的位置作为起始点
            p2 = np.asarray(p1, float)
            p1 = self.get_body_part_pos_relative_to_hip(
                'lankle' if is_left else 'rankle')
        else:
            # 否则，将p1和p2转换为浮点型的numpy数组
            p1 = np.asarray(p1, float)
            p2 = np.asarray(p2, float)

        # 计算从p1到p2的向量，并将其除以分辨率得到每一步的位移
        vec = (p2 - p1) / resolution

        # 生成从p1到p2的插值点
        hip_points = [p1 + vec * i for i in range(1, resolution+1)]
        interpolation = [self.leg(p, foot_ori3d, is_left, dynamic_pose)
                        for p in hip_points]

        # 定义要调整的关节索引（根据是左腿还是右腿）
        indices = [2, 4, 6, 8, 10, 12] if is_left else [3, 5, 7, 9, 11, 13]

        # 获取当前的关节位置（不包括脚的关节，以计算脚踝轨迹）
        last_joint_values = self.robot.joints_position[indices[0:4]]  # 排除脚关节

        # 初始化下一步的插值点
        next_step = interpolation[0]
        trajectory = []

        # 遍历插值点以生成轨迹
        for p in interpolation[1:-1]:
            # 如果当前关节位置与上一步的关节位置差异超过阈值，则将当前的next_step添加到轨迹中
            if np.any(np.abs(p[1][0:4] - last_joint_values) > 7.03):
                trajectory.append(next_step[1:3])
                last_joint_values = next_step[1][0:4]
                next_step = p
            else:
                next_step = p

        # 将最后一个插值点添加到轨迹中
        trajectory.append(interpolation[-1][1:3])

        # 返回关节索引和生成的轨迹
        return indices, trajectory

    def leg(self, ankle_pos3d, foot_ori3d, is_left: bool, dynamic_pose: bool):
        """
        计算腿部的逆向运动学，输入为脚踝的相对3D位置和脚的3D方向*。
        *可以直接控制偏航角，但俯仰角和横滚角是偏差（见下文）

        参数
        ----------
        ankle_pos3d : array_like, 长度为3
            脚踝的(x,y,z)位置，相对于两个髋关节的中心
        foot_ori3d : array_like, 长度为3
            绕x,y,z轴的旋转（绕x和y轴的旋转是相对于垂直姿势或启用动态姿势时的偏移）
        is_left : `bool`
            设置为True选择左腿，设置为False选择右腿
        dynamic_pose : `bool`
            启用基于IMU的动态脚旋转，使其与地面平行

        返回
        -------
        indices : `list`
            计算出的关节索引
        values : `list`
            计算出的关节值
        error_codes : `list`
            错误代码列表
                错误代码:
                    (-1) 脚过远（不可达）
                    (x)  关节x超出范围
        """

        error_codes = []
        leg_y_dev, upper_leg_height, upper_leg_depth, lower_leg_len, knee_extra_angle, _ = self.NAO_SPECS
        sign = -1 if is_left else 1

        # 将坐标转换为相对于腿部原点（通过平移y坐标）
        ankle_pos3d = np.asarray(ankle_pos3d) + (0, sign * leg_y_dev, 0)

        # 先旋转腿部，然后旋转坐标以抽象出旋转
        ankle_pos3d = Matrix_3x3().rotate_z_deg(-foot_ori3d[2]).multiply(ankle_pos3d)

        # 使用几何解法计算膝关节角度和脚的俯仰角
        dist = np.linalg.norm(ankle_pos3d)  # 髋关节到脚踝的距离
        sq_dist = dist * dist
        sq_upper_leg_h = upper_leg_height * upper_leg_height
        sq_lower_leg_l = lower_leg_len * lower_leg_len
        sq_upper_leg_l = upper_leg_depth * upper_leg_depth + sq_upper_leg_h
        upper_leg_len = np.sqrt(sq_upper_leg_l)
        knee = acos((sq_upper_leg_l + sq_lower_leg_l - sq_dist) /
                    (2 * upper_leg_len * lower_leg_len)) + knee_extra_angle  # 余弦定理
        foot = acos((sq_lower_leg_l + sq_dist - sq_upper_leg_l) /
                    (2 * lower_leg_len * dist))  # 脚垂直于向量(原点->ankle_pos)

        # 检查目标是否可达
        if dist > upper_leg_len + lower_leg_len:
            error_codes.append(-1)

        # 膝盖和脚
        knee_angle = np.pi - knee
        foot_pitch = foot - \
            np.arctan(ankle_pos3d[0] / np.linalg.norm(ankle_pos3d[1:3]))
        # 避免脚滚动的不稳定性（在-0.05m以上时无关紧要）
        foot_roll = np.arctan(
            ankle_pos3d[1] / min(-0.05, ankle_pos3d[2])) * -sign

        # 如果所有关节都直接移动的情况下的原始髋关节角度
        raw_hip_yaw = foot_ori3d[2]
        raw_hip_pitch = foot_pitch - knee_angle
        raw_hip_roll = -sign * foot_roll

        # 由于偏航关节的方向，旋转45度，然后旋转偏航、横滚和俯仰角
        m = Matrix_3x3().rotate_y_rad(raw_hip_pitch).rotate_x_rad(
            raw_hip_roll).rotate_z_deg(raw_hip_yaw).rotate_x_deg(-45 * sign)

        # 考虑偏航关节方向的实际髋关节角度
        hip_roll = (np.pi / 4) - (sign * np.arcsin(m.m[1, 2]))  # 由于45度旋转，增加pi/4
        hip_pitch = -np.arctan2(m.m[0, 2], m.m[2, 2])
        hip_yaw = sign * np.arctan2(m.m[1, 0], m.m[1, 1])

        # 将弧度转换为角度
        values = np.array([hip_yaw, hip_roll, hip_pitch, -knee_angle,
                        foot_pitch, foot_roll]) * 57.2957795  # 弧度到角度

        # 设置脚旋转的偏差（基于垂直姿势或动态姿势）
        values[4] -= foot_ori3d[1]
        values[5] -= foot_ori3d[0] * sign

        indices = [2, 4, 6, 8, 10, 12] if is_left else [3, 5, 7, 9, 11, 13]

        if dynamic_pose:
            # 躯干相对于脚的旋转
            m: Matrix_3x3 = Matrix_3x3.from_rotation_deg(
                (self.robot.imu_torso_roll, self.robot.imu_torso_pitch, 0))
            m.rotate_z_deg(foot_ori3d[2], True)

            roll = m.get_roll_deg()
            pitch = m.get_pitch_deg()

            # 简单的平衡算法
            correction = 1  # 修正以保持垂直躯干（以度数表示）
            roll = 0 if abs(roll) < correction else roll - \
                np.copysign(correction, roll)
            pitch = 0 if abs(pitch) < correction else pitch - \
                np.copysign(correction, pitch)

            values[4] += pitch
            values[5] += roll * sign

        # 检查并限制关节的范围
        for index, value in enumerate(indices):
            if values[index] < self.robot.joints[value].info.min or values[index] > self.robot.joints[value].info.max:
                error_codes.append(value)
                values[index] = np.clip(
                    values[index], self.robot.joints[value].info.min, self.robot.joints[value].info.max)

        return indices, values, error_codes

import xml.etree.ElementTree as xmlp
from collections import deque
from math_ops.math_ext import get_active_directory
from math_ops.Matrix_3x3 import Matrix_3x3
from math_ops.Matrix_4x4 import Matrix_4x4
from world.commons.Body_Part import Body_Part
from world.commons.Joint_Info import Joint_Info
import numpy as np


class Joint:
    """
    关节类
    """
    def __init__(self):
        self.position = 0.0
        self.speed = 0.0
        self.target_speed = 0.0
        self.target_last_speed = 0.0
        self.info: Joint_Info = None
        self.transform = Matrix_4x4()
        self.fix_effector_mask = 1


class Robot:
    STEPTIME = 0.02   # Fixed step time
    VISUALSTEP = 0.04  # Fixed visual step time
    SQ_STEPTIME = STEPTIME * STEPTIME
    GRAVITY = np.array([0, 0, -9.81])
    IMU_DECAY = 0.996  # IMU's velocity decay

    # ------------------ constants to force symmetry in joints/effectors

    MAP_PERCEPTOR_TO_INDEX = {
        "hj1": 0,  "hj2": 1,  "llj1": 2, "rlj1": 3,
        "llj2": 4, "rlj2": 5, "llj3": 6, "rlj3": 7,
        "llj4": 8, "rlj4": 9, "llj5": 10, "rlj5": 11,
        "llj6": 12, "rlj6": 13, "laj1": 14, "raj1": 15,
        "laj2": 16, "raj2": 17, "laj3": 18, "raj3": 19,
        "laj4": 20, "raj4": 21, "llj7": 22, "rlj7": 23}

    # Fix symmetry issues 1a/4 (identification)
    FIX_PERCEPTOR_SET = {'rlj2', 'rlj6', 'raj2', 'laj3', 'laj4'}
    FIX_INDICES_LIST = [5, 13, 17, 18, 20]

    # Recommended height for unofficial beam (near ground)
    BEAM_HEIGHTS = [0.4, 0.43, 0.4, 0.46, 0.4]

    def __init__(self, unum: int, robot_type: int) -> None:
        # 加载机器人XML文件，通常文件名为"naoX.xml"，其中X表示机器人类型
        robot_xml = "nao" + str(robot_type) + ".xml"
        self.type = robot_type
        self.beam_height = Robot.BEAM_HEIGHTS[robot_type]
        self.no_of_joints = 24 if robot_type == 4 else 22
        self.joints = [Joint() for _ in range(self.no_of_joints)]

        # 修复对称性问题1b/4（标识）
        for index in Robot.FIX_INDICES_LIST:
            self.joints[index].fix_effector_mask = -1

        self.body_parts = {}  # 保存机器人的身体部件，键为部件名称，值为Body_Part对象
        self.unum = unum  # 机器人的编号
        self.gyro = np.zeros(3)  # 机器人躯干在三自由度轴上的角速度（单位：度/秒）
        self.acc = np.zeros(3)  # 机器人躯干在三自由度轴上的加速度（单位：m/s²）
        self.frp = {}  # 足部和脚趾的阻力传感器数据，例如 {"lf":(px,py,pz,fx,fy,fz)}
        self.feet_toes_last_touch = {"lf": 0, "rf": 0,
                                     "lf1": 0, "rf1": 0}  # 记录足部和脚趾最后一次接触地面的时间
        self.feet_toes_are_touching = {
            "lf": False, "rf": False, "lf1": False, "rf1": False}  # 标记足部和脚趾是否接触地面
        self.fwd_kinematics_list = None  # 保存按照依赖关系排序的身体部件列表
        self.rel_cart_CoM_position = np.zeros(3)  # 质心相对于头部的位置（笛卡尔坐标系，单位：米）

        # 相对于头部的定位变量
        self.loc_head_to_field_transform = Matrix_4x4()  # 从头部到场地的变换矩阵
        self.loc_field_to_head_transform = Matrix_4x4()  # 从场地到头部的变换矩阵
        self.loc_rotation_head_to_field = Matrix_3x3()  # 从头部到场地的旋转矩阵
        self.loc_rotation_field_to_head = Matrix_3x3()  # 从场地到头部的旋转矩阵
        self.loc_head_position = np.zeros(3)  # 头部的绝对位置（单位：米）
        # 头部的绝对位置历史（最多保存40个旧位置，时间间隔为0.04秒，索引0为上一个位置）
        self.loc_head_position_history = deque(maxlen=40)
        self.loc_head_velocity = np.zeros(3)  # 头部的绝对速度（单位：米/秒，注意：可能有噪声）
        self.loc_head_orientation = 0  # 头部的方向（单位：度）
        self.loc_is_up_to_date = False  # 如果这不是视觉步，或可见的元素不足，则为False
        self.loc_last_update = 0  # 定位的最后更新时间（单位：World.time_local_ms）
        # 头部位置最后通过视觉或无线电更新的时间（单位：World.time_local_ms）
        self.loc_head_position_last_update = 0
        self.radio_fallen_state = False  # 如果无线电数据表明机器人倒下，并且无线电数据显著比定位数据更新，则为True
        self.radio_last_update = 0  # 无线电状态最后更新时间（单位：World.time_local_ms）

        # 相对于躯干的定位变量
        self.loc_torso_to_field_rotation = Matrix_3x3()  # 从躯干到场地的旋转矩阵
        self.loc_torso_to_field_transform = Matrix_4x4()  # 从躯干到场地的变换矩阵
        self.loc_torso_roll = 0  # 躯干的横滚角度（单位：度）
        self.loc_torso_pitch = 0  # 躯干的俯仰角度（单位：度）
        self.loc_torso_orientation = 0  # 躯干的方向（单位：度）
        self.loc_torso_inclination = 0  # 躯干的倾斜角度（单位：度）（躯干z轴相对于场地z轴的倾斜角度）
        self.loc_torso_position = np.zeros(3)  # 躯干的绝对位置（单位：米）
        self.loc_torso_velocity = np.zeros(3)  # 躯干的绝对速度（单位：米/秒）
        self.loc_torso_acceleration = np.zeros(3)  # 躯干的绝对加速度（单位：米/秒²）

        # 其他定位变量
        self.cheat_abs_pos = np.zeros(3)  # 服务器提供的头部绝对位置（作弊，单位：米）
        self.cheat_ori = 0.0  # 服务器提供的头部绝对方向（作弊，单位：度）
        self.loc_CoM_position = np.zeros(3)  # 质心的绝对位置（单位：米）
        self.loc_CoM_velocity = np.zeros(3)  # 质心的绝对速度（单位：米/秒）

        # 特殊定位变量
        '''
        self.loc_head_z 通常等于 self.loc_head_position[2]，但有时会有所不同。
        在某些情况下，尽管无法计算旋转和位移，仍然可以通过视觉获取z坐标，
        在这种情况下：
            self.loc_is_up_to_date 为False
            self.loc_head_z_is_up_to_date 为True
        它应当在依赖z作为独立坐标的应用中使用，
        例如检测机器人是否倒下，或作为机器学习的观测值。
        它绝不应在3D变换中使用。
        '''
        self.loc_head_z = 0  # 头部的绝对z坐标（单位：米），见上面的解释
        self.loc_head_z_is_up_to_date = False  # 如果这不是视觉步，或可见的元素不足，则为False
        self.loc_head_z_last_update = 0  # 最后计算loc_head_z的时间（单位：World.time_local_ms）
        self.loc_head_z_vel = 0  # 头部的绝对z速度（单位：米/秒）

        # 定位 + 陀螺仪
        # 这些变量是可靠的。当等待下一个视觉周期时，陀螺仪用于更新旋转
        self.imu_torso_roll = 0  # 躯干的横滚角度（单位：度）（来源：定位+陀螺仪）
        self.imu_torso_pitch = 0  # 躯干的俯仰角度（单位：度）（来源：定位+陀螺仪）
        self.imu_torso_orientation = 0  # 躯干的方向（单位：度）（来源：定位+陀螺仪）
        self.imu_torso_inclination = 0  # 躯干的倾斜角度（单位：度）（来源：定位+陀螺仪）
        self.imu_torso_to_field_rotation = Matrix_3x3()  # 从躯干到场地的旋转矩阵（来源：定位+陀螺仪）
        self.imu_last_visual_update = 0  # 最后一次使用视觉信息更新IMU数据的时间（单位：World.time_local_ms）

        # 定位 + 陀螺仪 + 加速度计
        # 注意：这些变量不可靠，因为定位方向中的小误差会导致错误的加速度->错误的速度->错误的位置
        self.imu_weak_torso_to_field_transform = Matrix_4x4()  # 从躯干到场地的变换矩阵（来源：定位+陀螺仪+加速度计）
        self.imu_weak_head_to_field_transform = Matrix_4x4()  # 从头部到场地的变换矩阵（来源：定位+陀螺仪+加速度计）
        self.imu_weak_field_to_head_transform = Matrix_4x4()  # 从场地到头部的变换矩阵（来源：定位+陀螺仪+加速度计）
        self.imu_weak_torso_position = np.zeros(
            3)  # 躯干的绝对位置（单位：米）（来源：定位+陀螺仪+加速度计）
        self.imu_weak_torso_velocity = np.zeros(
            3)  # 躯干的绝对速度（单位：米/秒）（来源：定位+陀螺仪+加速度计）
        self.imu_weak_torso_acceleration = np.zeros(
            3)  # 躯干的绝对加速度（单位：米/秒²）（来源：定位+陀螺仪+加速度计）
        self.imu_weak_torso_next_position = np.zeros(
            3)  # 预测下一步的绝对位置（单位：米）（来源：定位+陀螺仪+加速度计）
        self.imu_weak_torso_next_velocity = np.zeros(
            3)  # 预测下一步的绝对速度（单位：米/秒）（来源：定位+陀螺仪+加速度计）
        self.imu_weak_CoM_position = np.zeros(
            3)  # 质心的绝对位置（单位：米）（来源：定位+陀螺仪+加速度计）
        self.imu_weak_CoM_velocity = np.zeros(
            3)  # 质心的绝对速度（单位：米/秒）（来源：定位+陀螺仪+加速度计）

        # 使用显式变量以启用IDE建议
        self.J_HEAD_YAW = 0
        self.J_HEAD_PITCH = 1
        self.J_LLEG_YAW_PITCH = 2
        self.J_RLEG_YAW_PITCH = 3
        self.J_LLEG_ROLL = 4
        self.J_RLEG_ROLL = 5
        self.J_LLEG_PITCH = 6
        self.J_RLEG_PITCH = 7
        self.J_LKNEE = 8
        self.J_RKNEE = 9
        self.J_LFOOT_PITCH = 10
        self.J_RFOOT_PITCH = 11
        self.J_LFOOT_ROLL = 12
        self.J_RFOOT_ROLL = 13
        self.J_LARM_PITCH = 14
        self.J_RARM_PITCH = 15
        self.J_LARM_ROLL = 16
        self.J_RARM_ROLL = 17
        self.J_LELBOW_YAW = 18
        self.J_RELBOW_YAW = 19
        self.J_LELBOW_ROLL = 20
        self.J_RELBOW_ROLL = 21
        self.J_LTOE_PITCH = 22
        self.J_RTOE_PITCH = 23

        # ------------------ 解析机器人XML文件

        dir = get_active_directory("/world/commons/robots/")
        robot_xml_root = xmlp.parse(dir + robot_xml).getroot()

        joint_no = 0
        for child in robot_xml_root:
            if child.tag == "bodypart":  # 如果是身体部件
                self.body_parts[child.attrib['name']] = Body_Part(
                    child.attrib['mass'])  # 保存部件质量信息
            elif child.tag == "joint":  # 如果是关节
                self.joints[joint_no].info = Joint_Info(child)
                self.joints[joint_no].position = 0.0
                ji = self.joints[joint_no].info

                # 如果身体部件是第一个锚点，则保存关节信息（简化模型遍历的方向）
                self.body_parts[ji.anchor0_part].joints.append(
                    Robot.MAP_PERCEPTOR_TO_INDEX[ji.perceptor])

                joint_no += 1
                if joint_no == self.no_of_joints:
                    break  # ignore extra joints

            else:
                raise NotImplementedError

        assert joint_no == self.no_of_joints, "The Robot XML and the robot type don't match!"

    def get_head_abs_vel(self, history_steps: int):
        '''
        Get robot's head absolute velocity (m/s)

        Parameters
        ----------
        history_steps : int
            number of history steps to consider [1,40]

        Examples
        --------
        get_head_abs_vel(1) is equivalent to (current abs pos - last abs pos)      / 0.04
        get_head_abs_vel(2) is equivalent to (current abs pos - abs pos 0.08s ago) / 0.08
        get_head_abs_vel(3) is equivalent to (current abs pos - abs pos 0.12s ago) / 0.12
        '''
        assert 1 <= history_steps <= 40, "Argument 'history_steps' must be in range [1,40]"

        if len(self.loc_head_position_history) == 0:
            return np.zeros(3)

        h_step = min(history_steps, len(self.loc_head_position_history))
        t = h_step * Robot.VISUALSTEP

        return (self.loc_head_position - self.loc_head_position_history[h_step-1]) / t

    def _initialize_kinematics(self):

        # starting with head
        parts = {"head"}
        sequential_body_parts = ["head"]

        while len(parts) > 0:
            part = parts.pop()

            for j in self.body_parts[part].joints:

                p = self.joints[j].info.anchor1_part

                # add body part if it is the 1st anchor of some joint
                if len(self.body_parts[p].joints) > 0:
                    parts.add(p)
                    sequential_body_parts.append(p)

        self.fwd_kinematics_list = [(self.body_parts[part], j, self.body_parts[self.joints[j].info.anchor1_part])
                                    for part in sequential_body_parts for j in self.body_parts[part].joints]

        # Fix symmetry issues 4/4 (kinematics)
        for i in Robot.FIX_INDICES_LIST:
            self.joints[i].info.axes *= -1
            aux = self.joints[i].info.min
            self.joints[i].info.min = -self.joints[i].info.max
            self.joints[i].info.max = -aux

    def update_localization(self, localization_raw, time_local_ms):

        # parse raw data
        # 32bits to 64bits for consistency
        loc = localization_raw.astype(float)
        self.loc_is_up_to_date = bool(loc[32])
        self.loc_head_z_is_up_to_date = bool(loc[34])

        if self.loc_head_z_is_up_to_date:
            time_diff = (time_local_ms - self.loc_head_z_last_update) / 1000
            self.loc_head_z_vel = (loc[33] - self.loc_head_z) / time_diff
            self.loc_head_z = loc[33]
            self.loc_head_z_last_update = time_local_ms

        # Save last position to history at every vision cycle (even if not up to date) (update_localization is only called at vision cycles)
        self.loc_head_position_history.appendleft(
            np.copy(self.loc_head_position))

        if self.loc_is_up_to_date:
            time_diff = (time_local_ms - self.loc_last_update) / 1000
            self.loc_last_update = time_local_ms
            self.loc_head_to_field_transform.m[:] = loc[0:16].reshape((4, 4))
            self.loc_field_to_head_transform.m[:] = loc[16:32].reshape((4, 4))

            # extract data (related to the robot's head)
            self.loc_rotation_head_to_field = self.loc_head_to_field_transform.get_rotation()
            self.loc_rotation_field_to_head = self.loc_field_to_head_transform.get_rotation()
            p = self.loc_head_to_field_transform.get_translation()
            self.loc_head_velocity = (p - self.loc_head_position) / time_diff
            self.loc_head_position = p
            self.loc_head_position_last_update = time_local_ms
            self.loc_head_orientation = self.loc_head_to_field_transform.get_yaw_deg()
            self.radio_fallen_state = False

            # extract data (related to the center of mass)
            p = self.loc_head_to_field_transform(self.rel_cart_CoM_position)
            self.loc_CoM_velocity = (p - self.loc_CoM_position) / time_diff
            self.loc_CoM_position = p

            # extract data (related to the robot's torso)
            t = self.get_body_part_to_field_transform('torso')
            self.loc_torso_to_field_transform = t
            self.loc_torso_to_field_rotation = t.get_rotation()
            self.loc_torso_orientation = t.get_yaw_deg()
            self.loc_torso_pitch = t.get_pitch_deg()
            self.loc_torso_roll = t.get_roll_deg()
            self.loc_torso_inclination = t.get_inclination_deg()
            p = t.get_translation()
            self.loc_torso_velocity = (p - self.loc_torso_position) / time_diff
            self.loc_torso_position = p
            self.loc_torso_acceleration = self.loc_torso_to_field_rotation.multiply(
                self.acc) + Robot.GRAVITY

    def head_to_body_part_transform(self, body_part_name, coords, is_batch=False):
        '''
        If coord is a vector or list of vectors:
        Convert cartesian coordinates that are relative to head to coordinates that are relative to a body part 

        If coord is a Matrix_4x4 or a list of Matrix_4x4:
        Convert pose that is relative to head to a pose that is relative to a body part 

        Parameters
        ----------
        body_part_name : `str`
            name of body part (given by the robot's XML)
        coords : array_like
            One 3D position or list of 3D positions
        is_batch : `bool`
            Indicates if coords is a batch of 3D positions

        Returns
        -------
        coord : `list` or ndarray
            A numpy array is returned if is_batch is False, otherwise, a list of arrays is returned
        '''
        head_to_bp_transform: Matrix_4x4 = self.body_parts[body_part_name].transform.invert(
        )

        if is_batch:
            return [head_to_bp_transform(c) for c in coords]
        else:
            return head_to_bp_transform(coords)

    def get_body_part_to_field_transform(self, body_part_name) -> Matrix_4x4:
        '''
        Computes the transformation matrix from body part to field, from which we can extract its absolute position and rotation.
        For best results, use this method when self.loc_is_up_to_date is True. Otherwise, the forward kinematics
        will not be synced with the localization data and strange results may occur.
        '''
        return self.loc_head_to_field_transform.multiply(self.body_parts[body_part_name].transform)

    def get_body_part_abs_position(self, body_part_name) -> np.ndarray:
        '''
        Computes the absolute position of a body part considering the localization data and forward kinematics.
        For best results, use this method when self.loc_is_up_to_date is True. Otherwise, the forward kinematics
        will not be synced with the localization data and strange results may occur.
        '''
        return self.get_body_part_to_field_transform(body_part_name).get_translation()

    def get_joint_to_field_transform(self, joint_index) -> Matrix_4x4:
        '''
        Computes the transformation matrix from joint to field, from which we can extract its absolute position and rotation.
        For best results, use this method when self.loc_is_up_to_date is True. Otherwise, the forward kinematics
        will not be synced with the localization data and strange results may occur.
        '''
        return self.loc_head_to_field_transform.multiply(self.joints[joint_index].transform)

    def get_joint_abs_position(self, joint_index) -> np.ndarray:
        '''
        Computes the absolute position of a joint considering the localization data and forward kinematics.
        For best results, use this method when self.loc_is_up_to_date is True. Otherwise, the forward kinematics
        will not be synced with the localization data and strange results may occur.
        '''
        return self.get_joint_to_field_transform(joint_index).get_translation()

    def update_pose(self):

        if self.fwd_kinematics_list is None:
            self._initialize_kinematics()

        for body_part, j, child_body_part in self.fwd_kinematics_list:
            ji = self.joints[j].info
            self.joints[j].transform.m[:] = body_part.transform.m
            self.joints[j].transform.translate(ji.anchor0_axes, True)
            child_body_part.transform.m[:] = self.joints[j].transform.m
            child_body_part.transform.rotate_deg(
                ji.axes, self.joints[j].position, True)
            child_body_part.transform.translate(ji.anchor1_axes_neg, True)

        self.rel_cart_CoM_position = np.average([b.transform.get_translation() for b in self.body_parts.values()], 0,
                                                [b.mass for b in self.body_parts.values()])

    def update_imu(self, time_local_ms):

        # update IMU
        if self.loc_is_up_to_date:
            self.imu_torso_roll = self.loc_torso_roll
            self.imu_torso_pitch = self.loc_torso_pitch
            self.imu_torso_orientation = self.loc_torso_orientation
            self.imu_torso_inclination = self.loc_torso_inclination
            self.imu_torso_to_field_rotation.m[:
                                               ] = self.loc_torso_to_field_rotation.m
            self.imu_weak_torso_to_field_transform.m[:
                                                     ] = self.loc_torso_to_field_transform.m
            self.imu_weak_head_to_field_transform.m[:
                                                    ] = self.loc_head_to_field_transform.m
            self.imu_weak_field_to_head_transform.m[:
                                                    ] = self.loc_field_to_head_transform.m
            self.imu_weak_torso_position[:] = self.loc_torso_position
            self.imu_weak_torso_velocity[:] = self.loc_torso_velocity
            self.imu_weak_torso_acceleration[:] = self.loc_torso_acceleration
            self.imu_weak_torso_next_position = self.loc_torso_position + self.loc_torso_velocity * \
                Robot.STEPTIME + self.loc_torso_acceleration * \
                (0.5 * Robot.SQ_STEPTIME)
            self.imu_weak_torso_next_velocity = self.loc_torso_velocity + \
                self.loc_torso_acceleration * Robot.STEPTIME
            self.imu_weak_CoM_position[:] = self.loc_CoM_position
            self.imu_weak_CoM_velocity[:] = self.loc_CoM_velocity
            self.imu_last_visual_update = time_local_ms
        else:
            g = self.gyro / 50  # convert degrees per second to degrees per step

            self.imu_torso_to_field_rotation.multiply(
                Matrix_3x3.from_rotation_deg(g), in_place=True, reverse_order=True)

            self.imu_torso_orientation = self.imu_torso_to_field_rotation.get_yaw_deg()
            self.imu_torso_pitch = self.imu_torso_to_field_rotation.get_pitch_deg()
            self.imu_torso_roll = self.imu_torso_to_field_rotation.get_roll_deg()

            self.imu_torso_inclination = np.atan(np.sqrt(
                np.tan(self.imu_torso_roll/180*np.pi)**2+np.tan(self.imu_torso_pitch/180*np.pi)**2))*180/np.pi

            # Update position and velocity until 0.2 seconds has passed since last visual update
            if time_local_ms < self.imu_last_visual_update + 200:
                self.imu_weak_torso_position[:] = self.imu_weak_torso_next_position
                if self.imu_weak_torso_position[2] < 0:
                    # limit z coordinate to positive values
                    self.imu_weak_torso_position[2] = 0
                # stability tradeoff
                self.imu_weak_torso_velocity[:] = self.imu_weak_torso_next_velocity * \
                    Robot.IMU_DECAY
            else:
                # without visual updates for 0.2s, the position is locked, and the velocity decays to zero
                self.imu_weak_torso_velocity *= 0.97

            # convert proper acceleration to coordinate acceleration and fix rounding bias
            self.imu_weak_torso_acceleration = self.imu_torso_to_field_rotation.multiply(
                self.acc) + Robot.GRAVITY
            self.imu_weak_torso_to_field_transform = Matrix_4x4.from_3x3_and_translation(
                self.imu_torso_to_field_rotation, self.imu_weak_torso_position)
            self.imu_weak_head_to_field_transform = self.imu_weak_torso_to_field_transform.multiply(
                self.body_parts["torso"].transform.invert())
            self.imu_weak_field_to_head_transform = self.imu_weak_head_to_field_transform.invert()
            p = self.imu_weak_head_to_field_transform(
                self.rel_cart_CoM_position)
            self.imu_weak_CoM_velocity = (
                p-self.imu_weak_CoM_position)/Robot.STEPTIME
            self.imu_weak_CoM_position = p

            # Next Position = x0 + v0*t + 0.5*a*t^2,   Next velocity = v0 + a*t
            self.imu_weak_torso_next_position = self.imu_weak_torso_position + self.imu_weak_torso_velocity * \
                Robot.STEPTIME + self.imu_weak_torso_acceleration * \
                (0.5 * Robot.SQ_STEPTIME)
            self.imu_weak_torso_next_velocity = self.imu_weak_torso_velocity + \
                self.imu_weak_torso_acceleration * Robot.STEPTIME

    def set_joints_target_position_direct(
            self,
            indices: int | list | slice | np.ndarray,
            values: np.ndarray,
            harmonize: bool = True,
            max_speed=7.03,
            tolerance=0.012,
            limit_joints=True) -> int:
        '''
        Computes the speed of a list of joints, taking as argument the target position

        Parameters
        ----------
        indices : `int`/`list`/`slice`/numpy array
            joint indices
        values : numpy array 
            target position for each listed joint index
        harmonize : `bool`
            if True, all joints reach target at same time
        max_speed : `float`
            max. speed for all joints in deg/step
            Most joints have a maximum speed of 351.77 deg/s according to rcssserver3d/data/rsg/agent/nao/hingejoint.rsg
            That translates as 7.0354 deg/step or 6.1395 rad/s
        tolerance : `float`
            angle error tolerance (in degrees) to return that target was reached (returns -1)
        limit_joints : `bool`
            limit values to the joints' range of motion

        Returns
        -------
        remaining_steps : `int`
            predicted number of remaining steps or -1 if target was already reached
        '''
        assert isinstance(
            values, np.ndarray), "'values' argument must be a numpy array"
        # Replace NaN with zero and infinity with large finite numbers
        np.nan_to_num(values, copy=False)

        if isinstance(indices, slice):
            print(indices.start, indices.stop, indices.step)
            indices = list(range(indices.start, indices.stop, indices.step))

        predicted_diff = []
        joints_positions = []

        for i, idx in enumerate(indices):
            joint: Joint = self.joints[idx]

            if limit_joints:
                values[i] = np.clip(values[i], joint.info.min, joint.info.max)

            predicted_diff.append(joint.target_last_speed * 1.1459156)
            joints_positions.append(joint.position)  # joint's current position

        predicted_diff = np.array(predicted_diff)
        joints_positions = np.array(joints_positions)

        np.clip(predicted_diff, -7.03, 7.03, out=predicted_diff)

        reported_dist = values - joints_positions
        if np.all((np.abs(reported_dist) < tolerance)) and np.all((np.abs(predicted_diff) < tolerance)):
            for index in indices:
                self.joints[index].target_speed = 0
            return -1

        deg_per_step = reported_dist - predicted_diff

        relative_max = np.max(np.abs(deg_per_step)) / max_speed
        remaining_steps = np.ceil(relative_max)

        if remaining_steps == 0:
            for index in indices:
                self.joints[index].target_speed = 0
            return 0

        if harmonize:
            deg_per_step /= remaining_steps
        else:
            np.clip(deg_per_step, -max_speed, max_speed,
                    out=deg_per_step)  # limit maximum speed

        # convert to rad/s

        for i, idx in enumerate(indices):
            self.joints[idx].target_speed = deg_per_step[i] * 0.87266463

        return remaining_steps

    def get_command(self) -> bytes:
        '''
        Builds commands string from self.joints_target_speed
        '''
        cmd = ""
        j_speed = []
        for i, joint in enumerate(self.joints):
            j_speed.append(joint.target_speed * joint.fix_effector_mask)
            cmd += f"({joint.info.effector} {j_speed[i]:.5f})"
            joint.target_last_speed = joint.target_speed
            joint.target_speed = 0.0
        return cmd.encode("utf-8")

"""
    Robot Joint
"""

import xml.etree.ElementTree

import numpy as np
from math_ops.Matrix_4x4 import Matrix_4x4


class JointInfo:
    def __init__(self, xml_element: xml.etree.ElementTree.Element) -> None:
        self.perceptor = xml_element.attrib['perceptor']
        self.effector = xml_element.attrib['effector']
        self.axes = np.array([
            float(xml_element.attrib['xaxis']),
            float(xml_element.attrib['yaxis']),
            float(xml_element.attrib['zaxis'])])
        self.min = int(xml_element.attrib['min'])
        self.max = int(xml_element.attrib['max'])

        self.anchor0_part = xml_element[0].attrib['part']
        self.anchor0_axes = np.array([
            float(xml_element[0].attrib['y']),
            float(xml_element[0].attrib['x']),
            float(xml_element[0].attrib['z'])])  # x and y axes are switched

        self.anchor1_part = xml_element[1].attrib['part']
        self.anchor1_axes_neg = np.array([
            -float(xml_element[1].attrib['y']),
            -float(xml_element[1].attrib['x']),
            -float(xml_element[1].attrib['z'])])  # x and y axes are switched


class BodyPart:
    def __init__(self, mass) -> None:
        self.mass = float(mass)
        self.joints = []
        self.transform = Matrix_4x4()  # body part to head transformation matrix


class Joint:
    """
    关节类
    """

    def __init__(self):
        self.position = 0.0
        self.speed = 0.0
        self.target_speed = 0.0
        self.target_last_speed = 0.0
        self.info: JointInfo = None
        self.transform = Matrix_4x4()
        self.fix_effector_mask = 1

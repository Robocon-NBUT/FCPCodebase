"""
Extension of Python's built-in math module.
"""
import sys
import numpy as np

try:
    GLOBAL_DIR = sys._MEIPASS # Pyinstaller Template Folder
except AttributeError:
    GLOBAL_DIR = "."

type Number = int | float

def deg_sph2cart(spherical_vec):
    """
    Converts SimSpark's spherical coordinates in degrees to cartesian coordinates
    """
    r = spherical_vec[0]
    h = spherical_vec[1] * np.pi / 180
    v = spherical_vec[2] * np.pi / 180
    return np.array([r * np.cos(v) * np.cos(h), r * np.cos(v) * np.sin(h), r * np.sin(v)])

def deg_sin(deg: Number) -> float:
    """
    返回用角度计算的 sin 值
    """
    return np.sin(deg * np.pi / 180)

def deg_cos(deg: Number) -> float:
    """
    返回用角度计算的 cos 值
    """
    return np.cos(deg * np.pi / 180)

def to_3d(vec_2d, value = 0) -> np.ndarray:
    """
    Returns new 3d vector from 2d vector
    """
    return np.append(vec_2d, value)

def project_to_plane(vec_3d) ->np.ndarray:
    """
    将三维向量投影到二维平面
    """
    new_vec = np.copy(vec_3d)
    new_vec[2] = 0
    return new_vec

def normalize_vec(vec) -> np.ndarray:
    """
    将向量单位化
    """
    size = np.linalg.norm(vec)
    if size == 0:
        return vec
    return vec / size

def get_active_directory(_dir: str) -> str:
    global GLOBAL_DIR
    return GLOBAL_DIR + _dir

def acos(value: Number) -> float:
    """
    计算 np.arccos ，同时限制其输入范围
    """
    return np.arccos(np.clip(value, -1, 1))

def asin(value: Number) -> float:
    """
    计算 np.arcsin ，同时限制其输入范围
    """
    return np.arcsin(np.clip(value, -1, 1))

def normalize_deg(deg: Number) -> float:
    """
    将输入的角度映射到 [-180, 180) 范围内
    """
    return (deg + 180.0) % 360 - 180

def normalize_rad(rad: Number) -> float:
    """
    将输入的弧度映射到 [-pi, pi) 范围内
    """
    return (rad + np.pi) % (2 * np.pi) - np.pi

def deg_to_rad(deg: Number) -> float:
    """
    角度转弧度
    """
    return deg * np.pi / 180

def rad_to_deg(rad: Number) -> float:
    """
    弧度转角度
    """
    return rad / np.pi * 180

def vector_angle(vector, is_rad=False):
    ''' angle (degrees or radians) of 2D vector '''
    if is_rad:
        return np.arctan2(vector[1], vector[0])
    else:
        return np.arctan2(vector[1], vector[0]) * 180 / np.pi

def vectors_angle(vec1, vec2, is_rad=False):
    ''' get angle between vectors (degrees or radians) '''
    ang_rad = acos(np.dot(normalize_vec(vec1), normalize_vec(vec2)))
    return ang_rad if is_rad else ang_rad * 180 / np.pi

def vector_from_angle(angle, is_rad=False):
    ''' unit vector with direction given by `angle` '''
    if is_rad:
        return np.array([np.cos(angle), np.sin(angle)], float)
    else:
        return np.array([deg_cos(angle), deg_sin(angle)], float)

def target_abs_angle(pos2d, target, is_rad=False):
    ''' angle (degrees or radians) of vector (target-pos2d) '''
    if is_rad:
        return np.arctan2(target[1]-pos2d[1], target[0]-pos2d[0])
    else:
        return np.arctan2(target[1]-pos2d[1], target[0]-pos2d[0]) * 180 / np.pi

def target_rel_angle(pos2d, ori, target, is_rad=False):
    ''' relative angle (degrees or radians) of target if we're located at 'pos2d' with orientation 'ori' (degrees or radians) '''
    if is_rad:
        return normalize_rad( np.arctan2(target[1]-pos2d[1], target[0]-pos2d[0]) - ori )
    else:
        return normalize_deg( np.arctan2(target[1]-pos2d[1], target[0]-pos2d[0]) * 180 / np.pi - ori )

def rotate_2d_vec(vec, angle, is_rad=False):
    ''' rotate 2D vector anticlockwise around the origin by `angle` '''
    cos_ang = np.cos(angle) if is_rad else np.cos(angle * np.pi / 180)
    sin_ang = np.sin(angle) if is_rad else np.sin(angle * np.pi / 180)
    return np.array([cos_ang*vec[0]-sin_ang*vec[1], sin_ang*vec[0]+cos_ang*vec[1]])

def distance_point_to_line(p:np.ndarray, a:np.ndarray, b:np.ndarray) -> tuple[float, str]:
    ''' 
    计算点 P 到线段 ab 的距离
    并且观察 P 在 ab 的左侧还是右侧（方向为 a -> b)

    
    Parameters
    ----------
    a, b, p 都是 ndarray 且均为二维坐标

    Returns
    -------
    distance : float
        计算点 P 到线段 ab 的距离
    side : str
        P 在 ab 的左侧还是右侧（方向为 a -> b)
    '''
    line_len = np.linalg.norm(b - a)

    # 如果 ab 长度为 0 代表 a, b 重合
    if line_len == 0:
        dist = sdist = np.linalg.norm(p - a)
    else:
        sdist = np.cross(b - a,p - a) / line_len
        dist = abs(sdist)

    return dist, "left" if sdist > 0 else "right"

def distance_point_to_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray):
    """
    计算点 p 到二维线段 ab 的最短距离

    参数：
    p -- 待求距离的点
    a, b -- 定义线段的两个端点
    """
    ap = p - a
    ab = b - a

    ad = vector_projection(ap, ab)

    # 通过比较向量分量避免除法时可能出现的除以零错误，计算投影点d在线段ab上的相对位置系数k

    k = ad[0] / ab[0] if abs(ab[0]) > abs(ab[1]) else ad[1] / ab[1]

    if k <= 0:
        return np.linalg.norm(ap)
    elif k >= 1:
        return np.linalg.norm(p - b)
    else:
        return np.linalg.norm(p - (ad + a)) # p - d

def distance_point_to_ray(p:np.ndarray, ray_start:np.ndarray, ray_direction:np.ndarray):
    ''' Distance from point p to 2d ray '''

    rp = p-ray_start
    rd = vector_projection(rp,ray_direction)

    # Is d in ray? We can find k in (rd = k * ray_direction) without computing any norm
    # we use the largest dimension of ray_direction to avoid division by 0
    k = rd[0]/ray_direction[0] if abs(ray_direction[0])>abs(ray_direction[1]) else rd[1]/ray_direction[1]

    if k <= 0:
        return np.linalg.norm(rp)
    else:
        return np.linalg.norm(p-(rd + ray_start)) # p-d

def closest_point_on_ray_to_point(p:np.ndarray, ray_start:np.ndarray, ray_direction:np.ndarray):
    ''' Point on ray closest to point p '''

    rp = p-ray_start
    rd = vector_projection(rp,ray_direction)

    # Is d in ray? We can find k in (rd = k * ray_direction) without computing any norm
    # we use the largest dimension of ray_direction to avoid division by 0
    k = rd[0]/ray_direction[0] if abs(ray_direction[0])>abs(ray_direction[1]) else rd[1]/ray_direction[1]

    if   k <= 0: return ray_start
    else:        return rd + ray_start

def does_circle_intersect_segment(p:np.ndarray, r, a:np.ndarray, b:np.ndarray):
    ''' Returns true if circle (center p, radius r) intersect 2d line segment '''

    ap = p-a
    ab = b-a

    ad = vector_projection(ap,ab)

    # Is d in ab? We can find k in (ad = k * ab) without computing any norm
    # we use the largest dimension of ab to avoid division by 0
    k = ad[0]/ab[0] if abs(ab[0])>abs(ab[1]) else ad[1]/ab[1]

    if   k <= 0: return np.dot(ap,ap)   <= r*r
    elif k >= 1: return np.dot(p-b,p-b) <= r*r

    dp = p-(ad + a)
    return np.dot(dp,dp) <= r*r

def vector_projection(a:np.ndarray, b:np.ndarray):
    ''' Vector projection of a onto b '''
    b_dot = np.dot(b,b)
    return b * np.dot(a,b) / b_dot if b_dot != 0 else b

def do_noncollinear_segments_intersect(a,b,c,d):
    ''' 
    检查二维线段 'ab' 是否和非共线的二维线段 'cd' 相交
    Explanation: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/ 
    '''

    ccw = lambda a,b,c: (c[1]-a[1]) * (b[0]-a[0]) > (b[1]-a[1]) * (c[0]-a[0])
    return ccw(a,c,d) != ccw(b,c,d) and ccw(a,b,c) != ccw(a,b,d)

def intersection_segment_opp_goal(a:np.ndarray, b:np.ndarray):
    ''' Computes the intersection point of 2d segment 'ab' and the opponents' goal (front line) '''
    vec_x = b[0]-a[0]

    # Collinear intersections are not accepted
    if vec_x == 0:
        return None

    k = (15.01-a[0])/vec_x

    # No collision
    if k < 0 or k > 1:
        return None

    intersection_pt = a + (b-a) * k

    if -1.01 <= intersection_pt[1] <= 1.01:
        return intersection_pt
    else:
        return None

def intersection_circle_opp_goal(p:np.ndarray, r):
    ''' 
    Computes the intersection segment of circle (center p, radius r) and the opponents' goal (front line)
    Only the y coordinates are returned since the x coordinates are always equal to 15
    '''

    x_dev = abs(15-p[0])

    if x_dev > r:
        return None # no intersection with x=15

    y_dev = np.sqrt(r*r - x_dev*x_dev)

    p1 = max(p[1] - y_dev, -1.01)
    p2 = min(p[1] + y_dev,  1.01)

    if p1 == p2:
        return p1 # return the y coordinate of a single intersection point
    elif p2 < p1:
        return None # no intersection
    else:
        return p1, p2 # return the y coordinates of the intersection segment

def distance_point_to_opp_goal(p: np.ndarray):
    """
    计算点 p 到对手球门的距离 (front line)
    """

    if p[1] < -1.01:
        return np.linalg.norm( p-(15,-1.01) )
    elif p[1] > 1.01:
        return np.linalg.norm( p-(15, 1.01) )
    else:
        return abs(15-p[0])

def circle_line_segment_intersection(circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9):
    """
    计算一个圆与线段的交点，可能有0、1或2个交点。

    circle_center: 圆心坐标(x, y)
    circle_radius: 圆的坐标(x, y)
    pt1: 线段起点坐标(x, y)
    pt2: 线段终点坐标(x, y)
    full_line: 如果为 True，将寻找整条线上的交点，而不仅仅在线段上；
                如果为 False，将只返回线段内的交点。
    tangent_tol: 误差值

    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """
    cx, cy = circle_center
    pt1x, pt1y = pt1
    x1, y1 = pt1[0] - cx, pt1[1] - cy
    x2, y2 = pt2[0] - cx, pt2[1] - cy

    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2)**.5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2

    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant**.5) / dr ** 2,
            cy + (-big_d * dx + sign * abs(dy) * discriminant**.5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [
                (xi - pt1x) / dx if abs(dx) > abs(dy) else (yi - pt1y) / dy for xi, yi in intersections]
            intersections = [pt for pt, frac in zip(
                intersections, fraction_along_segment) if 0 <= frac <= 1]
        # If line is tangent to circle, return just one point (as both intersections have same location)
        if len(intersections) == 2 and abs(discriminant) <= tangent_tol:
            return [intersections[0]]
        else:
            return intersections

# adapted from https://stackoverflow.com/questions/3252194/numpy-and-line-intersections

def get_line_intersection(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return np.array([float('inf'), float('inf')])
    return np.array([x/z, y/z],float)

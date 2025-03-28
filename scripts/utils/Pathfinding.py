"""
::::::::::::::::::::::::::::::::::::::::::
::::::::a_star.compute(param_vec):::::::::
::::::::::::::::::::::::::::::::::::::::::

A* 自动寻路优化算法

param_vec: numpy 数组，类型为 float32

param_vec[0] - 起始点 x 坐标
param_vec[1] - 起始点 y 坐标
param_vec[2] - 是否允许球员走出球场边界（无球状态下有效）
param_vec[3] - 是否以球门为终点（自动设为对方球门最佳区域）
param_vec[4] - 目标点 x 坐标（仅当 param_vec[3] == 0 时生效）
param_vec[5] - 目标点 y 坐标（仅当 param_vec[3] == 0 时生效）
param_vec[6] - 超时时间（微秒，最大执行时间）
-------------- [optional] ----------------
param_vec[7-11] - 障碍物 1: x, y, 硬半径（最大5m），软半径（最大5m），软半径排斥力（最小0）
                - 障碍物 2: x, y, 硬半径（最大5m），软半径（最大5m），软半径排斥力（最小0）
                - 障碍物 n: x, y, 硬半径（最大5m），软半径（最大5m），软半径排斥力（最小0）
---------------- return ------------------
path_ret : numpy 数组 (float32)
    path_ret[:-2]
        包含从起点到目标的路径（最多1024个坐标点）
        每个坐标点由x,y组成（最多2048个坐标值）
        返回数组为一维扁平结构（例如 [x1,y1,x2,y2,x3,y3,...]）
        路径未达目标的情况可能包括：
            - 路径超过1024个点（至少102米！）
            - 无法到达或超时（返回最接近目标的路径）
    path_ret[-2]
        路径状态码：
        0 - 成功
        1 - 超时未达目标（可能不可达）
        2 - 目标不可达（已穷尽所有可能）
        3 - 起点与目标间无障碍（仅含起点和终点）
    path_ret[-1]
        A* 路径总成本
::::::::::::::::::::::::::::::::::::::::::
::::::::::::::::注意事项::::::::::::::::::
::::::::::::::::::::::::::::::::::::::::::

Map of field:
    - The algorithm has a 32m by 22m map with a precision of 10cm (same dimension as field +1 meter border)
    - The map contains information about field lines, goal posts and goal net
    - The path may go outside the field (out of bounds) if the user allows it, 
      but it may never go through goal posts or the goal net (these are considered static inaccessible obstacles)
    - The user must only specify dynamic obstacles through the arguments

Repulsive force:
    - The repulsive force is implemented as an extra cost for the A* algorithm
    - The cost for walking 10cm is 1, and the cost for walking diagonally is sqrt(2)
    - The extra cost of stepping on a position with a repulsive force f=1 is 1
    - For any given position on the field, the repulsive force of >=2 objects is combined with the max function, max(f1,f2), NOT f1+f2!
    - If path starts on inaccessible position, it can go to a neighbor inaccessible position but there is a cost of 100 (to avoid inaccessible paths)
    Example:
        Map 1   Map 2   Map 3
        ..x..   ..o..   ..o..
        ..1..   ..o..   .o1..
        ..o..   ..o..   ..o..
    Consider 'Map 1' where 'x' is the target, 'o' is the player, and '1' is a repulsive force of 1
    In 'Map 2', the player chooses to go forward,      the total cost of this path is: 1+(extra=1)+1   = 3
    In 'Map 3', the player avoids the repulsive force, the total cost of this path is: sqrt(2)+sqrt(2) = 2.83 (optimal solution)
         Map 1     Map 2     Map 3     Map 4
        ...x...   ..oo...   ...o...   ...o...
        ..123..   .o123..   ..o23..   ..1o3..
        ...o...   ..oo...   ...o...   ...o...
    Consider 'Map 1' with 3 positions with 3 distinct repulsive forces, going from 1 to 3.
    In 'Map 2', the player avoids all repulsive forces,               total cost: 1+sqrt(2)+sqrt(2)+1       = 4.83
    In 'Map 3', the player goes through the smallest repulsive force, total cost: sqrt(2)+(extra=1)+sqrt(2) = 3.83 (optimal solution)
    In 'Map 4', the player chooses to go forward,                     total cost: 1+(extra=2)+1             = 4.00

Obstacles:
    hard radius: inaccessible obstacle radius (infinite repulsive force)
    soft radius: accessible obstacle radius with user-defined repulsive force (fades with distance) (disabled if <= hard radius)
    Example:
        obstacle(0,0,1,3,5) -> obstacle at pos(0,0) with hard radius of 1m, soft radius of 3m with repulsive force 5
            - the path cannot be at <=1m from this obstacle, unless the path were to start inside that radius
            - the soft radius force is maximum at the center (5) and fades with distance until (0) at 3m from the obstacle
            - so to sum up, at a distance of [0,1]m the force is infinite, [1,3]m the force goes from 3.333 to 0
        obstacle(-2.1,3,0,0,0) -> obstacle at pos(-2.1,3) with hard radius of 0m, soft radius of 0m with repulsive force 0
            - the path cannot go through (-2.1,3)
        obstacle(-2.16,3,0,0,8) -> obstacle at pos(-2.2,3) with hard radius of 0m, soft radius of 0m with repulsive force 8
            - the path cannot go through (-2.2,3), the map has a precision of 10cm, so the obstacle is placed at the nearest valid position
            - the repulsive force is ignored because (soft radius <= hard radius)
"""
import time
import numpy as np
from agent.Base_Agent import Base_Agent as Agent
from cpp.modules import utils
from scripts.commons.Script import Script


class Pathfinding:
    def __init__(self, script: Script) -> None:
        self.script = script
        # Initialize (not needed, but the first run takes a bit more time)
        utils.astar_compute(np.zeros(6, np.float32))

    def draw_grid(self):
        d = self.player.world.draw
        MAX_RAW_COST = 0.6  # dribble cushion

        for x in np.arange(-16, 16.01, 0.1):
            for y in np.arange(-11, 11.01, 0.1):
                # do not allow out of bounds
                s_in,  cost_in = utils.astar_compute(
                    np.array([x, y, 0, 0, x, y, 5000], np.float32))[-2:]

                # allow out of bounds
                s_out, cost_out = utils.astar_compute(
                    np.array([x, y, 1, 0, x, y, 5000], np.float32))[-2:]
                # print(path_cost_in, path_cost_out)
                if s_out != 3:
                    d.point((x, y), 5, d.Color.red, "grid", False)
                elif s_in != 3:
                    d.point((x, y), 4, d.Color.blue_pale, "grid", False)
                elif 0 < cost_in < MAX_RAW_COST + 1e-6:
                    d.point((x, y), 4, d.Color.get(
                        255, (1-cost_in/MAX_RAW_COST)*255, 0), "grid", False)
                elif cost_in > MAX_RAW_COST:
                    d.point((x, y), 4, d.Color.black, "grid", False)
                # else:
                #    d.point((x,y), 4, d.Color.white, "grid", False)
        d.flush("grid")

    def sync(self):
        r = self.player.world.robot
        self.player.behavior.head.execute()
        self.player.server.commit_and_send(r.get_command())
        self.player.server.receive()

    def draw_path_and_obstacles(self, obst, path_ret_pb, path_ret_bp):
        w = self.player.world

        # draw obstacles
        for i in range(0, len(obst[0]), 5):
            w.draw.circle(obst[0][i:i+2], obst[0][i+2], 2,
                          w.draw.Color.red, "obstacles", False)
            w.draw.circle(obst[0][i:i+2], obst[0][i+3], 2,
                          w.draw.Color.orange, "obstacles", False)

        # draw path
        path_pb = path_ret_pb[:-2]         # create view without status
        path_status_pb = path_ret_pb[-2]  # extract status
        path_cost_pb = path_ret_pb[-1]     # extract A* cost
        path_bp = path_ret_bp[:-2]         # create view without status
        path_status_bp = path_ret_bp[-2]   # extract status
        path_cost_bp = path_ret_bp[-1]     # extract A* cost

        c_pb = {0: w.draw.Color.green_lime, 1: w.draw.Color.yellow,
                2: w.draw.Color.red, 3: w.draw.Color.blue_light}[path_status_pb]
        c_bp = {0: w.draw.Color.green_pale, 1: w.draw.Color.yellow_light,
                2: w.draw.Color.red_salmon, 3: w.draw.Color.blue_pale}[path_status_bp]

        for i in range(2, len(path_pb)-2, 2):
            w.draw.line(path_pb[i-2:i], path_pb[i:i+2], 5,
                        c_pb, "path_player_ball", False)

        if len(path_pb) >= 4:
            w.draw.arrow(path_pb[-4:-2], path_pb[-2:], 0.4,
                         5, c_pb, "path_player_ball", False)

        for i in range(2, len(path_bp)-2, 2):
            w.draw.line(path_bp[i-2:i], path_bp[i:i+2], 5,
                        c_bp, "path_ball_player", False)

        if len(path_bp) >= 4:
            w.draw.arrow(path_bp[-4:-2], path_bp[-2:], 0.4,
                         5, c_bp, "path_ball_player", False)

        w.draw.flush("obstacles")
        w.draw.flush("path_player_ball")
        w.draw.flush("path_ball_player")

    def move_obstacles(self, obst):

        for i in range(len(obst[0])//5):
            obst[0][i*5] += obst[1][i, 0]
            obst[0][i*5+1] += obst[1][i, 1]
            if not -16 < obst[0][i*5] < 16:
                obst[1][i, 0] *= -1
            if not -11 < obst[0][i*5+1] < 11:
                obst[1][i, 1] *= -1

    def execute(self):

        a = self.script.args
        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name
        self.player = Agent(a.i, a.p, a.m, a.u, a.r, a.t)
        w = self.player.world
        r = self.player.world.robot
        timeout = 5000

        go_to_goal = 0
        obst_no = 50
        # obst[x,y,h,s,f] + random velocity
        obst = [[0, 0, 0.5, 1, 1]*obst_no,
                np.random.uniform(-0.01, 0.01, (obst_no, 2))]

        print("\nMove player/ball around using RoboViz!")
        print("Press ctrl+c to return.")
        print("\nPathfinding timeout set to", timeout, "us.")
        print("Pathfinding execution time:")

        self.draw_grid()

        while True:
            ball = w.Ball.AbsolutePos[:2]
            rpos = r.location.Head.Position[:2]

            self.move_obstacles(obst)

            # allow out of bounds (player->ball)
            param_vec_pb = np.array(
                [*rpos,  1, go_to_goal, *ball, timeout, *obst[0]], np.float32)
            # don't allow (ball->player)
            param_vec_bp = np.array(
                [*ball,  0, go_to_goal, *rpos, timeout, *obst[0]], np.float32)
            t1 = time.time()
            path_ret_pb = utils.astar_compute(param_vec_pb)
            t2 = time.time()
            path_ret_bp = utils.astar_compute(param_vec_bp)
            t3 = time.time()

            print(
                end=f"\rplayer->ball {int((t2-t1)*1000000):5}us (len:{len(path_ret_pb[:-2])//2:4})      ball->player {int((t3-t2)*1000000):5}us  (len:{len(path_ret_bp[:-2])//2:4}) ")

            self.draw_path_and_obstacles(obst, path_ret_pb, path_ret_bp)
            self.sync()

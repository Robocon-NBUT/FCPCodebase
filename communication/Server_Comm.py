import socket
import time
import sys
from itertools import count
from select import select

from world.World import World
from communication.world_parser import WorldParser


class Server_Comm:
    monitor_socket = None

    def __init__(
            self, host: str, agent_port: int, monitor_port: int, unum: int, robot_type: int,
            team_name: str, world_parser: WorldParser, world: World,
            other_players, wait_for_server=True) -> None:

        self.BUFFER_SIZE = 8192
        self.rcv_buff = bytearray(self.BUFFER_SIZE)
        self.send_buff = []
        self.world_parser = world_parser
        self.unum = unum

        # During initialization, it's not clear whether we are on the left or right side
        self._unofficial_beam_msg_left  = "(agent (unum " + str(unum) + ") (team Left) (move "
        self._unofficial_beam_msg_right = "(agent (unum " + str(unum) + ") (team Right) (move "
        self.world = world

        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        if wait_for_server:
            print("Waiting for server at ", host, ":",
                  agent_port, sep="", end=".", flush=True)
        while True:
            try:
                self.socket.connect((host, agent_port))
                print(end=" ")
                break
            except ConnectionRefusedError:
                if not wait_for_server:
                    print("Server is down. Closing...")
                    sys.exit()
                time.sleep(1)
                print(".", end="", flush=True)
        print("Connected agent", unum, self.socket.getsockname())

        self.send_immediate(
            b'(scene rsg/agent/nao/nao_hetero.rsg ' + str(robot_type).encode() + b')')
        self._receive_async(other_players, True)

        self.send_immediate(
            b'(init (unum ' + str(unum).encode() + b') (teamname ' + team_name.encode() + b'))')
        self._receive_async(other_players, False)

        # Repeat to guarantee that team side information is received
        for _ in range(3):
            # Eliminate advanced step by changing syn order (rcssserver3d protocol bug, usually seen for player 11)
            # if this syn is not needed, it will be discarded by the server
            self.send_immediate(b'(syn)')
            for p in other_players:
                p.scom.send_immediate(b'(syn)')
            for p in other_players:
                p.scom.receive()
            self.receive()

        if world.team_side_is_left == None:
            print("\nError: server did not return a team side! Check server terminal!")
            sys.exit()

        # Monitor socket is shared by all agents on the same thread
        if Server_Comm.monitor_socket is None and monitor_port is not None:
            print("Connecting to server's monitor port at ", host,
                  ":", monitor_port, sep="", end=".", flush=True)
            Server_Comm.monitor_socket = socket.socket(
                socket.AF_INET, socket.SOCK_STREAM)
            Server_Comm.monitor_socket.connect((host, monitor_port))
            print("Done!")

    def _receive_async(self, other_players, first_pass) -> None:
        '''Private function that receives asynchronous information during the initialization'''

        if not other_players:
            self.receive()
            return

        self.socket.setblocking(0)
        if first_pass:
            print("Async agent", self.unum, "initialization", end="", flush=True)

        while True:
            try:
                print(".", end="", flush=True)
                self.receive()
                break
            except:
                pass
            for p in other_players:
                p.scom.send_immediate(b'(syn)')
            for p in other_players:
                p.scom.receive()

        self.socket.setblocking(1)
        if not first_pass:
            print("Done!")

    def receive(self, update=True):
        # parse all messages and perform value updates, but heavy computation is only done once at the end
        for i in count():
            try:
                if self.socket.recv_into(self.rcv_buff, nbytes=4) != 4:
                    raise ConnectionResetError()
                msg_size = int.from_bytes(
                    self.rcv_buff[:4], byteorder='big', signed=False)
                if self.socket.recv_into(
                    self.rcv_buff, nbytes=msg_size, flags=socket.MSG_WAITALL) != msg_size:
                    raise ConnectionResetError()
            except ConnectionResetError:
                print("\nError: socket was closed by rcssserver3d!")
                sys.exit()

            self.world_parser.parse(self.rcv_buff[:msg_size])
            if len(select([self.socket], [], [], 0.0)[0]) == 0:
                break

        if update:
            if i == 1:
                self.world.log(
                    "Server_Comm.py: The agent lost 1 packet! Is syncmode enabled?")
            if i > 1:
                self.world.log(f"Server_Comm.py: The agent lost {i} consecutive packets! Is syncmode disabled?")
            self.world.update()

            if len(select([self.socket], [], [], 0.0)[0]) != 0:
                self.world.log(
                    "Server_Comm.py: Received a new packet while on world.update()!")
                self.receive()

    def send_immediate(self, msg: bytes) -> None:
        ''' Commit and send immediately '''
        try:
            # Add message length in the first 4 bytes
            self.socket.send((len(msg)).to_bytes(4, byteorder='big') + msg)
        except BrokenPipeError:
            print("\nError: socket was closed by rcssserver3d!")
            sys.exit()

    def send(self) -> None:
        ''' Send all committed messages '''
        if len(select([self.socket], [], [], 0.0)[0]) == 0:
            self.send_buff.append(b'(syn)')
            self.send_immediate(b''.join(self.send_buff))
        else:
            self.world.log(
                "Server_Comm.py: Received a new packet while thinking!")
        self.send_buff = []  # clear buffer

    def commit(self, msg: bytes) -> None:
        assert isinstance(msg, bytes), "Message must be of type Bytes!"
        self.send_buff.append(msg)

    def commit_and_send(self, msg: bytes = b'') -> None:
        self.commit(msg)
        self.send()

    def clear_buffer(self) -> None:
        self.send_buff = []

    def commit_announcement(self, msg: bytes) -> None:
        '''
        向场上的每个玩家发布公告。
        最多20个字符，字符范围为ASCII码0x20到0x7E，排除' '、'('、')'。
        可接受字符：字母+数字+符号：!"#$%&'*+,-./:;<=>?@[\]^_`{|}~
        消息的有效范围为50米（场地对角线长度为36米，因此忽略这个限制）。
        玩家每2步（0.04秒）只能听到队友的消息一次。
        此功能独立于双方的消息（即我们的队伍无法通过发送消息来阻挡对方的消息）。
        自己发送的消息总是可以听到。
        '''
        assert len(msg) <= 20 and isinstance(msg, bytes)  # 确保消息长度不超过20且类型为字节
        self.commit(b'(say ' + msg + b')')

    def commit_pass_command(self) -> None:
        ''' 
        发出传球指令：
        条件：
        - 当前的游戏模式为PlayOn
        - 代理靠近球（默认0.5米）
        - 没有对手靠近球（默认1米）
        - 球是静止的（默认速度小于0.05米/秒）
        - 两次传球指令之间有一定的时间间隔
        '''
        self.commit(b'(pass)')

    def commit_beam(self, pos2d, rot) -> None:
        '''
        官方的传送指令，可在游戏中使用。
        该传送会受到噪声的影响（除非在服务器配置中禁用）。

        参数
        ----------
        pos2d : array_like
            绝对2D位置（无论我们在哪一方，负X轴总是指向我们的半场）。
        rot : `int`/`float`
            玩家角度，单位为度（0度指向前方）。
        '''
        assert len(pos2d)==2, "The official beam command accepts only 2D positions!"
        self.commit( f"(beam {pos2d[0]} {pos2d[1]} {rot})".encode() )


    def unofficial_beam(self, pos3d, rot) -> None:
        ''' 
        非官方的传送指令，不能在正式比赛中使用。

        参数
        ----------
        pos3d : array_like
            绝对3D位置（无论我们在哪一方，负X轴总是指向我们的半场）。
        rot : `int`/`float`
            玩家角度，单位为度（0度指向前方）。
        '''
        assert len(pos3d)==3, "The unofficial beam command accepts only 3D positions!"

        # 不需要对角度进行标准化，服务器接受任何角度
        if self.world.team_side_is_left:
            msg = f"{self._unofficial_beam_msg_left}{pos3d[0]} {pos3d[1]} {pos3d[2]} {rot-90}))".encode()
        else:
            msg = f"{self._unofficial_beam_msg_right}{-pos3d[0]} {-pos3d[1]} {pos3d[2]} {rot+90}))".encode()

        self.monitor_socket.send((len(msg)).to_bytes(
            4, byteorder='big') + msg)  # 发送非官方传送指令

    def unofficial_kill_sim(self) -> None:
        ''' Unofficial kill simulator command '''
        msg = b'(killsim)'
        self.monitor_socket.send((len(msg)).to_bytes(4, byteorder='big') + msg)

    def unofficial_move_ball(self, pos3d, vel3d=(0, 0, 0)) -> None:
        ''' 
        Unofficial command to move ball 
        info: ball radius = 0.042m

        Parameters
        ----------
        pos3d : array_like
            Absolute 3D position (negative X is always our half of the field, no matter our side)
        vel3d : array_like
            Absolute 3D velocity (negative X is always our half of the field, no matter our side)
        '''
        assert len(pos3d) == 3 and len(
            vel3d) == 3, "To move the ball we need a 3D position and velocity"

        if self.world.team_side_is_left:
            msg = f"(ball (pos {pos3d[0]} {pos3d[1]} {pos3d[2]}) (vel {vel3d[0]} {vel3d[1]} {vel3d[2]}))".encode()
        else:
            msg = f"(ball (pos {-pos3d[0]} {-pos3d[1]} {pos3d[2]}) (vel {-vel3d[0]} {-vel3d[1]} {vel3d[2]}))".encode()

        self.monitor_socket.send((len(msg)).to_bytes(4, byteorder='big') + msg)

    def unofficial_set_game_time(self, time_in_s: float) -> None:
        '''
        Unofficial command to set the game time
        e.g. unofficial_set_game_time(68.78)

        Parameters
        ----------
        time_in_s : float
            Game time in seconds
        '''
        msg = f"(time {time_in_s})".encode()
        self.monitor_socket.send((len(msg)).to_bytes(4, byteorder='big') + msg)

    def unofficial_set_play_mode(self, play_mode: str) -> None:
        '''
        Unofficial command to set the play mode
        e.g. unofficial_set_play_mode("PlayOn")

        Parameters
        ----------
        play_mode : str
            Play mode
        '''
        msg = f"(playMode {play_mode})".encode()
        self.monitor_socket.send((len(msg)).to_bytes(4, byteorder='big') + msg)

    def unofficial_kill_player(self, unum: int, team_side_is_left: bool) -> None:
        '''
        Unofficial command to kill specific player

        Parameters
        ----------
        unum : int
            Uniform number
        team_side_is_left : bool
            True if player to kill belongs to left team
        '''
        msg = f"(kill (unum {unum}) (team {'Left' if team_side_is_left else 'Right'}))".encode()
        self.monitor_socket.send((len(msg)).to_bytes(4, byteorder='big') + msg)

    def close(self, close_monitor_socket=False):
        """
        Close agent socket,
        and optionally the monitor socket (shared by players running on the same thread)
        """
        self.socket.close()
        if close_monitor_socket and Server_Comm.monitor_socket is not None:
            Server_Comm.monitor_socket.close()
            Server_Comm.monitor_socket = None

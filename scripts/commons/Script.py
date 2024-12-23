import argparse
import json
import sys
import shutil
import subprocess
from time import sleep
from pathlib import Path
from scripts.commons.UI import UI


class Script:
    # project root directory
    ROOT_DIR = str(Path(__file__).resolve().parents[2])

    def __init__(self) -> None:
        '''
        Arguments specification
        -----------------------
        - To add new arguments, edit the information below
        - After changing information below, the config.json file must be manually deleted
        - In other modules, these arguments can be accessed by their 1-letter ID
        '''
        # list of arguments: 1-letter ID, Description, Hardcoded default
        self.options = {
            'i': ('Server Hostname/IP', 'localhost'),
            'p': ('Agent Port',         '3100'),
            'm': ('Monitor Port',       '3200'),
            't': ('Team Name',          'FCPortugal'),
            'u': ('Uniform Number',     '1'),
            'r': ('Robot Type',         '1'),
            'P': ('Penalty Shootout',   '0'),
            'F': ('magmaFatProxy',      '0'),
            'D': ('Debug Mode',         '1')
        }

        # list of arguments: 1-letter ID, data type, choices
        self.op_types = {
            'i': (str, None),
            'p': (int, None),
            'm': (int, None),
            't': (str, None),
            'u': (int, range(1, 12)),
            'r': (int, [0, 1, 2, 3, 4]),
            'P': (int, [0, 1]),
            'F': (int, [0, 1]),
            'D': (int, [0, 1])
        }

        '''
        End of arguments specification
        '''

        self.read_or_create_config()

        # advance help text position
        def formatter(prog):
            return argparse.HelpFormatter(prog, max_help_position=52)
        parser = argparse.ArgumentParser(formatter_class=formatter)

        o = self.options
        t = self.op_types

        for id in self.options:  # shorter metavar for aesthetic reasons
            parser.add_argument(f"-{id}", help=f"{o[id][0]:30}[{o[id][1]:20}]", type=t[id][0], nargs='?', default=o[id][1], metavar='X', choices=t[id][1])

        self.args = parser.parse_args()

        if getattr(sys, 'frozen', False):  # disable debug mode when running from binary
            self.args.D = 0

        self.players = []  # list of created players

        Script.cmake_builds()

        if self.args.D:
            try:
                print(f"NOTE: 运行 \"python3 {__file__} -h \" 以获取帮助")
            except:
                pass

            columns = [[], [], []]
            for key, value in vars(self.args).items():
                columns[0].append(o[key][0])
                columns[1].append(o[key][1])
                columns[2].append(value)

            UI.print_table(
                columns,
                ["Argument", "Default at /config.json", "Active"], alignment=["<", "^", "^"])

    def read_or_create_config(self) -> None:

        config_path = Path("config.json")

        if not config_path.is_file():  # Save hardcoded default values if file does not exist
            with config_path.open("w") as f:
                json.dump(self.options, f, indent=4)
        # Load user-defined values (that can be overwritten by command-line arguments)
        else:
            # Wait for possible write operation when launching multiple agents
            if config_path.stat().st_size == 0:
                sleep(1)
            if config_path.stat().st_size == 0:  # Abort after 1 second
                print(
                    "Aborting: 'config.json' is empty. Manually verify and delete if still empty.")
                sys.exit()

            with config_path.open("r") as f:
                self.options = json.load(f)

    @staticmethod
    def cmake_builds():
        """
        Build C++ Modules With CMake

        CMake File: ./cpp/CMakeLists.txt
        """
        cmake_dir = Path(Script.ROOT_DIR) / "cpp"

        if not cmake_dir.is_dir():
            return

        # Check the .so files exist, .so file is in modules dir
        module_dir = cmake_dir / "modules"

        if len(list(module_dir.glob("*.so"))) == 3:
            return

        build_dir = cmake_dir / "build"

        if not build_dir.is_dir():
            build_dir.mkdir()

        cmake_commands = ["cmake", ".."]
        cmake_build_commands = ["cmake", "--build", "."]

        with subprocess.Popen(cmake_commands, cwd=build_dir) as process:
            process.wait()

        with subprocess.Popen(cmake_build_commands, cwd=build_dir) as process:
            process.wait()

        # remove build dir
        shutil.rmtree(build_dir)


        print("CMake Build Completed")

    def batch_create(self, agent_cls, args_per_player):
        ''' Creates batch of agents '''

        for a in args_per_player:
            self.players.append(agent_cls(*a))

    def batch_execute_agent(self, index: slice = slice(None)):
        ''' 
        Executes agent normally (including commit & send)

        Parameters
        ----------
        index : slice
            subset of agents
            (e.g. index=slice(1,2) will select the second agent)
            (e.g. index=slice(1,3) will select the second and third agents)
            by default, all agents are selected
        '''
        for p in self.players[index]:
            p.think_and_send()

    def batch_execute_behavior(self, behavior, index: slice = slice(None)):
        '''
        Executes behavior

        Parameters
        ----------
        behavior : str
            name of behavior to execute
        index : slice
            subset of agents
            (e.g. index=slice(1,2) will select the second agent)
            (e.g. index=slice(1,3) will select the second and third agents)
            by default, all agents are selected
        '''
        for p in self.players[index]:
            p.behavior.execute(behavior)

    def batch_commit_and_send(self, index: slice = slice(None)):
        '''
        Commits & sends data to server

        Parameters
        ----------
        index : slice
            subset of agents
            (e.g. index=slice(1,2) will select the second agent)
            (e.g. index=slice(1,3) will select the second and third agents)
            by default, all agents are selected
        '''
        for p in self.players[index]:
            p.server.commit_and_send(p.world.robot.get_command())

    def batch_receive(self, index: slice = slice(None), update=True):
        ''' 
        Waits for server messages

        Parameters
        ----------
        index : slice
            subset of agents
            (e.g. index=slice(1,2) will select the second agent)
            (e.g. index=slice(1,3) will select the second and third agents)
            by default, all agents are selected
        update : bool
            update world state based on information received from server
            if False, the agent becomes unaware of itself and its surroundings
            which is useful for reducing cpu resources for dummy agents in demonstrations
        '''
        for p in self.players[index]:
            p.server.receive(update)

    def batch_commit_beam(self, pos2d_and_rotation, index: slice = slice(None)):
        '''
        Beam all player to 2D position with a given rotation

        Parameters
        ----------
        pos2d_and_rotation : `list`
            iterable of 2D positions and rotations e.g. [(0,0,45),(-5,0,90)]
        index : slice
            subset of agents
            (e.g. index=slice(1,2) will select the second agent)
            (e.g. index=slice(1,3) will select the second and third agents)
            by default, all agents are selected
        '''
        for p, pos_rot in zip(self.players[index], pos2d_and_rotation):
            p.server.commit_beam(pos_rot[0:2], pos_rot[2])

    def batch_unofficial_beam(self, pos3d_and_rotation, index: slice = slice(None)):
        '''
        Beam all player to 3D position with a given rotation

        Parameters
        ----------
        pos3d_and_rotation : `list`
            iterable of 3D positions and rotations e.g. [(0,0,0.5,45),(-5,0,0.5,90)]
        index : slice
            subset of agents
            (e.g. index=slice(1,2) will select the second agent)
            (e.g. index=slice(1,3) will select the second and third agents)
            by default, all agents are selected
        '''
        for p, pos_rot in zip(self.players[index], pos3d_and_rotation):
            p.server.unofficial_beam(pos_rot[0:3], pos_rot[3])

    def batch_terminate(self, index: slice = slice(None)):
        '''
        Close all sockets connected to the agent port
        For scripts where the agent lives until the application ends, this is not needed

        Parameters
        ----------
        index : slice
            subset of agents
            (e.g. index=slice(1,2) will select the second agent)
            (e.g. index=slice(1,3) will select the second and third agents)
            by default, all agents are selected
        '''
        for p in self.players[index]:
            p.terminate()
        del self.players[index]  # delete selection

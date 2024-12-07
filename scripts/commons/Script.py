import argparse
import json
import sys
import pickle
import subprocess
from time import sleep
from os import cpu_count
from pathlib import Path
from scripts.commons.UI import UI


class Script:
    # project root directory
    ROOT_DIR = str(Path(__file__).resolve().parents[2])

    def __init__(self, cpp_builder_unum=0) -> None:
        '''
        Arguments specification
        -----------------------
        - To add new arguments, edit the information below
        - After changing information below, the config.json file must be manually deleted
        - In other modules, these arguments can be accessed by their 1-letter ID
        '''
        # list of arguments: 1-letter ID, Description, Hardcoded default
        self.options = {'i': ('Server Hostname/IP', 'localhost'),
                        'p': ('Agent Port',         '3100'),
                        'm': ('Monitor Port',       '3200'),
                        't': ('Team Name',          'FCPortugal'),
                        'u': ('Uniform Number',     '1'),
                        'r': ('Robot Type',         '1'),
                        'P': ('Penalty Shootout',   '0'),
                        'F': ('magmaFatProxy',      '0'),
                        'D': ('Debug Mode',         '1')}

        # list of arguments: 1-letter ID, data type, choices
        self.op_types = {'i': (str, None),
                         'p': (int, None),
                         'm': (int, None),
                         't': (str, None),
                         'u': (int, range(1, 12)),
                         'r': (int, [0, 1, 2, 3, 4]),
                         'P': (int, [0, 1]),
                         'F': (int, [0, 1]),
                         'D': (int, [0, 1])}

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

        Script.build_cpp_modules(exit_on_build=(
            cpp_builder_unum != 0 and cpp_builder_unum != self.args.u))

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

            UI.print_table(columns, [
                           "Argument", "Default at /config.json", "Active"], alignment=["<", "^", "^"])

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
    def build_cpp_modules(special_environment_prefix=[], exit_on_build=False):
        '''
        Build C++ modules in folder /cpp using Pybind11

        Parameters
        ----------
        special_environment_prefix : `list`
            command prefix to run a given command in the desired environment
            useful to compile C++ modules for different python interpreter versions (other than default version)
            Conda Env. example: ['conda', 'run', '-n', 'myEnv']
            If [] the default python interpreter is used as compilation target
        exit_on_build : bool
            exit if there is something to build (so that only 1 player per team builds c++ modules)
        '''
        cpp_path = Script.ROOT_DIR + "/cpp/"
        exclusions = ["__pycache__"]

        cpp_modules = [d.name for d in Path(cpp_path).iterdir(
        ) if d.is_dir() and d.name not in exclusions]

        if not cpp_modules:
            return  # no modules to build

        # "python3" can select the wrong version, this prevents that
        python_cmd = f"python{sys.version_info.major}.{sys.version_info.minor}"

        def init():
            print("--------------------------\nC++ modules:", cpp_modules)

            try:
                process = subprocess.Popen(
                    special_environment_prefix+[python_cmd, "-m", "pybind11", "--includes"], stdout=subprocess.PIPE)
                (includes, err) = process.communicate()
                process.wait()
            except:
                print(f"Error while executing child program: '{python_cmd} -m pybind11 --includes'")
                exit()

            # strip trailing newlines (and other whitespace chars)
            includes = includes.decode().rstrip()
            print("Using Pybind11 includes: '", includes, "'", sep="")
            return includes

        nproc = str(cpu_count())
        zero_modules = True

        for module in cpp_modules:
            module_path = Path(cpp_path) / module

            # Skip module if there is no Makefile (typical distribution case)
            if not (module_path / "Makefile").is_file():
                continue

            # Skip module in certain conditions
            so_file = module_path / f"{module}.so"
            c_info_file = module_path / f"{module}.c_info"
            if so_file.is_file() and c_info_file.is_file():
                with c_info_file.open('rb') as f:
                    info = pickle.load(f)
                if info == python_cmd:
                    code_files = [
                        f for f in module_path.iterdir() if f.suffix in (".cpp", ".h")]
                    code_mod_time = max(f.stat().st_mtime for f in code_files)
                    bin_mod_time = so_file.stat().st_mtime
                    if bin_mod_time + 30 > code_mod_time:  # Favor not building with a margin of 30s
                        continue

            # init: print stuff & get Pybind11 includes
            if zero_modules:
                if exit_on_build:
                    print("There are C++ modules to build. This player is not allowed to build. Aborting.")
                    sys.exit()
                zero_modules = False
                includes = init()

            # build module
            print(f'{f"Building: {module}... ":40}', end='', flush=True)
            process = subprocess.Popen(['make', '-j'+nproc, 'PYBIND_INCLUDES='+includes],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=module_path)
            (output, err) = process.communicate()
            exit_code = process.wait()
            if exit_code == 0:
                print("success!")
                with (module_path / (module + ".c_info")).open("wb") as f:  # save python version
                    # protocol 4 is backward compatible with Python 3.4
                    pickle.dump(python_cmd, f, protocol=4)
            else:
                print("Aborting! Building errors:")
                print(output.decode(), err.decode())
                sys.exit()

        if not zero_modules:
            print("All modules were built successfully!\n--------------------------")

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

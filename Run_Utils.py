import sys
from pathlib import Path
from importlib import import_module
from scripts.commons.Script import Script
from scripts.commons.UI import UI

from world.commons.Draw import Draw
from agent.Base_Agent import Base_Agent


def main():

    # Initialize: load config file, parse arguments,
    # build cpp modules (warns the user about inconsistencies before choosing a test script)
    script = Script()
    sys.path.insert(
        0, str(Path(__file__).resolve().parent.parent / "stable-baselines3"))

    _cwd = Path.cwd() / Path(__file__).resolve().parent

    gyms_path = _cwd / "scripts/gyms/"
    utils_path = _cwd / "scripts/utils/"
    exclusions = ["__init__.py"]

    utils = sorted([f.stem for f in utils_path.glob("*.py") if f.is_file()
                   and f.name not in exclusions], key=lambda x: (x != "Server", x))
    gyms = sorted([f.stem for f in gyms_path.glob("*.py")
                  if f.is_file() and f.name not in exclusions])

    while True:
        _, col_idx, col = UI.print_table([utils, gyms], ["Demos & Tests & Utils", "Gyms"], cols_per_title=[
                                         2, 1], numbering=[True]*2, prompt='选择要执行的内容 (ctrl+c to exit): ')

        is_gym = False
        if col == 0:
            chosen = ("scripts.utils.", utils[col_idx])
        elif col == 1:
            chosen = ("scripts.gyms.", gyms[col_idx])
            is_gym = True

        cls_name = chosen[1]
        mod = import_module(chosen[0]+chosen[1])

        # An imported script should not automatically execute the main code because:
        #     - Multiprocessing methods, such as 'forkserver' and 'spawn',
        # will execute the main code in every child
        #     - The script can only be called once unless it is reloaded
        if not is_gym:
            # Utils 接受一个带有用户自定义参数和可用的方法(user-defined arguments and useful methods)
            # 每一个 Util 都必须要有一个 execute() 方法
            obj = getattr(mod, cls_name)(script)
            try:
                obj.execute()  # Util may return normally or through KeyboardInterrupt
            except KeyboardInterrupt:
                print("\nctrl+c pressed, returning...\n")
            Draw.clear_all()            # clear all drawings
            Base_Agent.terminate_all()  # close all server sockets + monitor socket
            script.players = []         # clear list of players created through batch commands
            del obj

        else:

            # Gyms must implement a class Train() which is initialized with user-defined arguments and implements:
            #     train() - method to run the optimization and save a new model
            #     test(folder_dir, folder_name, model_file) - method to load an existing model and test it
            from scripts.commons.Train_Base import Train_Base

            print("\n在使用 GYMS 之前，确保 server 的所有的参数设置正确")
            print("(sync mode 为 'On', real time 应为 'Off', cheats 应为 'On')")
            print("要更改这些参数，请转到上一级菜单，并选择 'Server' 选项\n")
            print("并且，GYMS 使用独立的 servers，因此不要运行任何其他的 server")

            while True:
                try:
                    idx = UI.print_table(
                        [["Train", "Test", "Retrain"]], numbering=[True],
                        prompt='Choose option (ctrl+c to return): ')[0]
                except KeyboardInterrupt:
                    print()
                    break

                if idx == 0:
                    mod.Train(script).train({})
                else:
                    model_info = Train_Base.prompt_user_for_model()
                    if model_info is not None and idx == 1:
                        mod.Train(script).test(model_info)
                    elif model_info is not None:
                        mod.Train(script).train(model_info)


# allow child processes to bypass this file
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n正在退出......")
        sys.exit()

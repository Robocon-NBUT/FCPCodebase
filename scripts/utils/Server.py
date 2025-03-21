from pathlib import Path


class Server:

    def __init__(self, script) -> None:

        directories = ["/usr/local/share/rcssserver3d/",
                       "/usr/share/rcssserver3d/"]

        self.source = next((str(dir_path) for dir_path in map(
            Path, directories) if dir_path.is_dir()), None) + "/"

        if not self.source:
            raise FileNotFoundError(
                "The server configuration files were not found!")

        # To add options: insert into options & explations with same index,
        # read respective value from file or from other values, add edit entry

        self.options = [
            "Official Config", "Penalty Shootout", "Soccer Rules",
            "Sync Mode", "Real Time", "Cheats", "Full Vision", "Add Noise", "25Hz Monitor"]
        self.descriptions = [
            "官方比赛中使用的配置",
            "服务器的点球大战模式", "比赛模式、自动裁判等",
            "代理与服务器之间的同步通信",
            "实时模式（或最大服务器速度）",
            "代理位置与方向、球的位置",
            "可360度视角（替代120度，垂直与水平方向）",
            "为可见物体位置添加噪声",
            "25Hz监视器（或50Hz但RoboViz将以双倍实际速度显示）"]

        spark_f = Path.home() / ".simspark/spark.rb"
        naoneckhead_f = self.source+"rsg/agent/nao/naoneckhead.rsg"

        self.files = {"Penalty Shootout": self.source + "naosoccersim.rb",
                      "Soccer Rules": self.source + "naosoccersim.rb",
                      "Sync Mode": spark_f,
                      "Real Time": self.source+"rcssserver3d.rb",
                      "Cheats": naoneckhead_f,
                      "Full Vision": naoneckhead_f,
                      "Add Noise": naoneckhead_f,
                      "25Hz Monitor": spark_f}

    def label(self, setting_name, t_on, t_off):
        with open(self.files[setting_name], "r", encoding="UTF-8") as sources:
            content = sources.read()

        if t_on in content:
            self.values[setting_name] = "On"
        elif t_off in content:
            self.values[setting_name] = "Off"
        else:
            self.values[setting_name] = "Error"

    def read_config(self):
        v = self.values = {}

        print("Reading server configuration files...")

        self.label("Penalty Shootout", "addSoccerVar('PenaltyShootout', true)",
                   "addSoccerVar('PenaltyShootout', false)")
        self.label("Soccer Rules", " gameControlServer.initControlAspect('SoccerRuleAspect')",
                   "#gameControlServer.initControlAspect('SoccerRuleAspect')")
        self.label("Real Time", "enableRealTimeMode = true",
                   "enableRealTimeMode = false")
        self.label("Cheats", "setSenseMyPos true", "setSenseMyPos false")
        self.label("Full Vision", "setViewCones 360 360",
                   "setViewCones 120 120")
        self.label("Add Noise", "addNoise true", "addNoise false")
        self.label("Sync Mode", "agentSyncMode = true",
                   "agentSyncMode = false")
        self.label("25Hz Monitor", "monitorStep = 0.04", "monitorStep = 0.02")

        is_official_config = (
            v["Penalty Shootout"] == "Off" and v["Soccer Rules"] == "On" and
            v["Real Time"] == "On" and v["Cheats"] == "Off" and v["Full Vision"] == "Off" and
            v["Add Noise"] == "On" and v["Sync Mode"] == "Off" and v["25Hz Monitor"] == "On")
        print(v["Penalty Shootout"], is_official_config)
        v["Official Config"] = "On" if is_official_config else "Off"

    def change_config(self, setting_name, t_on, t_off, current_value=None, file=None):
        if current_value is None:
            current_value = self.values[setting_name]

        if file is None:
            file = self.files[setting_name]

        with open(file, "r") as sources:
            t = sources.read()
        if current_value == "On":
            t = t.replace(t_on, t_off, 1)
            print(f"Replacing  '{t_on}'  with  '{t_off}'  in  '{file}'")
        elif current_value == "Off":
            t = t.replace(t_off, t_on, 1)
            print(f"Replacing  '{t_off}'  with  '{t_on}'  in  '{file}'")
        else:
            print(setting_name, "was not changed because the value is unknown!")
        with open(file, "w") as sources:
            sources.write(t)

    def execute(self):

        while True:
            self.read_config()

            # Convert convenient values dict to list
            values = [[item[0], self.values[item[0]], item[1]]
                      for item in zip(self.options, self.descriptions)]

            print()
            UI.print_table(
                values, ["Setting", "Value", "Description"], numbering=[True, False, False])
            choice = UI.input_num(
                'Choose setting (ctrl+c to return): ', 0, len(self.options))
            opt = self.options[choice]

            prefix = ['sudo', 'python3', 'scripts/utils/Server.py', opt]

            if opt in self.files:
                suffix = [self.values[opt], self.files[opt]]

            if opt == "Penalty Shootout":
                subprocess.call([*prefix, "addSoccerVar('PenaltyShootout', true)",
                                "addSoccerVar('PenaltyShootout', false)", *suffix])
            elif opt == "Soccer Rules":
                subprocess.call([*prefix, "gameControlServer.initControlAspect('SoccerRuleAspect')",
                                "#gameControlServer.initControlAspect('SoccerRuleAspect')", *suffix])
            elif opt == "Sync Mode":
                # doesn't need sudo privileges
                self.change_config(opt, "agentSyncMode = true",
                                   "agentSyncMode = false")
            elif opt == "Real Time":
                subprocess.call(
                    [*prefix, "enableRealTimeMode = true", "enableRealTimeMode = false", *suffix])
            elif opt == "Cheats":
                subprocess.call([*prefix, "setSenseMyPos true", "setSenseMyPos false", *suffix,
                                 opt, "setSenseMyOrien true", "setSenseMyOrien false", *suffix,
                                 opt, "setSenseBallPos true", "setSenseBallPos false", *suffix])
            elif opt == "Full Vision":
                subprocess.call([*prefix, "setViewCones 360 360",
                                "setViewCones 120 120", *suffix])
            elif opt == "Add Noise":
                subprocess.call([*prefix, "addNoise true",
                                "addNoise false", *suffix])
            elif opt == "25Hz Monitor":
                # doesn't need sudo privileges
                self.change_config(opt, "monitorStep = 0.04",
                                   "monitorStep = 0.02")
            elif opt == "Official Config":
                if self.values[opt] == "On":
                    print("The official configuration is already On!")
                else:  # here, the first option is the official value
                    subprocess.call(
                        [*prefix[:3],
                         "Penalty Shootout", "addSoccerVar('PenaltyShootout', false)", "addSoccerVar('PenaltyShootout', true)", "Off",
                         self.files["Penalty Shootout"],
                         "Soccer Rules", "gameControlServer.initControlAspect('SoccerRuleAspect')",
                         "#gameControlServer.initControlAspect('SoccerRuleAspect')", "Off",
                         self.files["Soccer Rules"],
                         "Sync Mode", "agentSyncMode = false", "agentSyncMode = true", "Off",
                         self.files["Sync Mode"],
                         "Real Time", "enableRealTimeMode = true", "enableRealTimeMode = false", "Off",
                         self.files["Real Time"],
                         "Cheats", "setSenseMyPos false", "setSenseMyPos true", "Off",
                         self.files["Cheats"],
                         "Cheats", "setSenseMyOrien false", "setSenseMyOrien true", "Off",
                         self.files["Cheats"],
                         "Cheats", "setSenseBallPos false", "setSenseBallPos true", "Off",
                         self.files["Cheats"],
                         "Full Vision", "setViewCones 120 120", "setViewCones 360 360", "Off",
                         self.files["Full Vision"],
                         "Add Noise", "addNoise true", "addNoise false", "Off",
                         self.files["Add Noise"],
                         "25Hz Monitor", "monitorStep = 0.04", "monitorStep = 0.02", "Off", self.files["25Hz Monitor"]])


# process with sudo privileges to change the configuration files
if __name__ == "__main__":
    import sys
    s = Server(None)

    for i in range(1, len(sys.argv), 5):
        s.change_config(*sys.argv[i:i+5])
else:
    import subprocess
    from scripts.console_ui import UI

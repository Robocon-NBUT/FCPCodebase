"Behavior"
from world.World import World
from communication.server import Server

from behaviors.Poses import Poses
from behaviors.Head import Head
from behaviors.Slot_Engine import Slot_Engine
from behaviors.behavior_core import BehaviorCore


class Behavior:
    "行为类"

    def __init__(self, base_agent) -> None:
        self.base_agent = base_agent
        self.world = base_agent.world
        self.server = base_agent.server
        self.state_behavior_name = None
        self.state_behavior_init_ms = 0
        self.previous_behavior = None
        self.previous_behavior_duration = 0

        self.poses = Poses(self.world)
        self.head = Head(self.world)
        self.slot_engine = Slot_Engine(self.world)

        self.behaviors = {}
        self.objects = {}

    def create_behaviors(self) -> dict[str, BehaviorCore]:
        "创建行为"
        self.behaviors = {}
        self.behaviors.update(self.poses.get_behaviors_callbacks())
        self.behaviors.update(self.slot_engine.get_behaviors_callbacks())
        self.behaviors.update(self.get_custom_callbacks())
        return self.behaviors

    def get_custom_callbacks(self) -> dict[str, BehaviorCore]:
        "获取自定义行为"

        # Declaration of behaviors
        from behaviors.custom.Basic_Kick.Basic_Kick import Basic_Kick
        from behaviors.custom.Dribble.Dribble import Dribble
        from behaviors.custom.Fall.Fall import Fall
        from behaviors.custom.Get_Up.Get_Up import Get_Up
        from behaviors.custom.Step.Step import Step
        from behaviors.custom.Walk.Walk import Walk
        from behaviors.custom.Kick_Long.Kick_Long import Kick_Long
        classes = [Basic_Kick, Dribble, Fall, Get_Up, Step, Walk, Kick_Long]

        self.objects = {cls.__name__: cls(self.base_agent) for cls in classes}

        return {
            name:
            BehaviorCore(
                name, obj.description, obj.auto_head,
                lambda reset, *args, obj=obj: obj.execute(reset, *args),
                lambda *args, obj=obj: obj.is_ready(*args))
            for name, obj in self.objects.items()
        }

    def get_custom_behavior_object(self, name: str):
        "获取自定义行为对象"
        return self.objects[name]

    def get_all_behaviors(self) -> list[list[str], list[str]]:
        "获取所有行为名称和描述"
        return [
            list(self.behaviors.keys()),
            [behavior.Description for behavior in self.behaviors.values()]]

    def get_current(self) -> str:
        "Get name and duration (in seconds) of current behavior"
        duration = (self.world.time_local_ms -
                    self.state_behavior_init_ms) / 1000.0
        return self.state_behavior_name, duration

    def get_previous(self) -> str:
        "Get name and duration (in seconds) of previous behavior"
        return self.previous_behavior, self.previous_behavior_duration

    def force_reset(self) -> None:
        "强制重置"
        self.state_behavior_name = None

    def execute(self, name: str, *args) -> bool:
        ''' 
        Execute one step of behavior `name` with arguments `*args`
        - Automatically resets behavior on first call
        - Call get_current() to get the current behavior (and its duration)

        Returns
        -------
        finished : bool
            True if behavior has finished
        '''
        assert name in self.behaviors, f"行为 {name} 不存在！"

        reset = self.state_behavior_name != name
        if reset:
            if self.state_behavior_name is not None:
                self.previous_behavior = self.state_behavior_name
            self.previous_behavior_duration = (
                self.world.time_local_ms - self.state_behavior_init_ms) / 1000.0
            self.state_behavior_name = name
            self.state_behavior_init_ms = self.world.time_local_ms

        # Control head orientation if behavior allows it
        if self.behaviors[name].AutoHead:
            self.head.execute()

        if not self.behaviors[name].Execute(reset, *args):
            return False

        self.previous_behavior = self.state_behavior_name
        self.state_behavior_name = None
        return True

    def execute_sub_behavior(self, name: str, reset: bool, *args) -> bool:
        '''
        Execute one step of behavior `name` with arguments `*args`
        对于想要调用其他行为的自定义行为很有用
        - 手动充值行为
        - 调用 get_current() 函数将范围主行为 (main behavior) 而不是子行为 (sub behavior)
        - Poses 将忽略 reset 参数

        Returns
        -------
        finished : bool
            行为完成将返回 True
        '''
        assert name in self.behaviors, f"行为 {name} 不存在！"

        if self.behaviors[name].AutoHead:
            self.head.execute()

        return self.behaviors[name].Execute(reset, *args)

    def execute_to_completion(self, name: str, *args) -> None:
        '''
        执行指定的 steps 直到完成，并在此期间与 server 保持通讯
        - Slot behaviors 表明在发送最后一个 command 时已经完成（该指令会被及时发送）
        - 在 server 返回期望的 robot 状态时，Poses 结束（所以最后一个 command 是无关紧要的）
        - 对于自定义行为，我们采用相同的逻辑，并且最后一个 command 也被忽略

        Notes
        -----
        - 在退出前，`Robot.joints_target_speed` 数组将被重置以避免后续的 command
        - 对于在首次调用即完成的 Poses 或自定义行为，将不会提交或发送任何内容
        - 警告：如果行为永远不会结束，此函数可能会陷入无限循环中
        '''
        robot = self.world.robot
        skip_last = name not in self.slot_engine.behaviors

        while True:
            done = self.execute(name, *args)
            if done and skip_last:
                break
            self.server.commit_and_send(robot.get_command())
            self.server.receive()
            if done:
                break

        # reset to avoid polluting the next command
        for i in range(len(robot.joints)):
            robot.joints[i].target_speed = 0

    def is_ready(self, name: str, *args) -> bool:
        "Checks if behavior is ready to start under current game/robot conditions"

        assert name in self.behaviors, f"行为 {name} 不存在！"
        return self.behaviors[name].IsReady(*args)

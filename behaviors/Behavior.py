import numpy as np


class Behavior():

    def __init__(self, base_agent) -> None:
        from agent.Base_Agent import Base_Agent  # for type hinting
        self.base_agent: Base_Agent = base_agent
        self.world = self.base_agent.world
        self.state_behavior_name = None
        self.state_behavior_init_ms = 0
        self.previous_behavior = None
        self.previous_behavior_duration = None

        # Initialize standard behaviors
        from behaviors.Poses import Poses
        from behaviors.Slot_Engine import Slot_Engine
        from behaviors.Head import Head

        self.poses = Poses(self.world)
        self.slot_engine = Slot_Engine(self.world)
        self.head = Head(self.world)

    def create_behaviors(self):
        '''
        Behaviors dictionary:
            creation:   key: ( description, auto_head, lambda reset[,a,b,c,..]: self.execute(...), lambda: self.is_ready(...) )
            usage:      key: ( description, auto_head, execute_func(reset, *args), is_ready_func )
        '''
        self.behaviors = self.poses.get_behaviors_callbacks()
        self.behaviors.update(self.slot_engine.get_behaviors_callbacks())
        self.behaviors.update(self.get_custom_callbacks())

    def get_custom_callbacks(self):
        '''
        可实现自动搜索自定义行为
        然而，对于代码分发(code distribution)，动态导入代码的做法并不理想（除非我们导入字节码或是选择其他导入方案）
        目前：添加自定义行为是一个手动的加载过程：
            1. 在下方添加 import 语句
            2. 将自定义的动作类(class) 添加到 `classes` 列表中
        '''

        # Declaration of behaviors
        from behaviors.custom.Basic_Kick.Basic_Kick import Basic_Kick
        from behaviors.custom.Dribble.Dribble import Dribble
        from behaviors.custom.Fall.Fall import Fall
        from behaviors.custom.Get_Up.Get_Up import Get_Up
        from behaviors.custom.Step.Step import Step
        from behaviors.custom.Walk.Walk import Walk
        from behaviors.custom.Kick_Long.Kick_Long import Kick_Long
        classes = [Basic_Kick, Dribble, Fall, Get_Up, Step, Walk, Kick_Long]

        '''---- End of manual declarations ----'''

        # Prepare callbacks
        self.objects = {cls.__name__: cls(self.base_agent) for cls in classes}

        return {name: (o.description, o.auto_head,
                       lambda reset, *args, o=o: o.execute(reset, *args),
                       lambda *args, o=o: o.is_ready(*args)) for name, o in self.objects.items()}

    def get_custom_behavior_object(self, name):
        ''' Get unique object from class "name" ("name" must represent a custom behavior) '''
        assert name in self.objects, f"There is no custom behavior called {
            name}"
        return self.objects[name]

    def get_all_behaviors(self):
        ''' Get name and description of all behaviors '''
        return [key for key in self.behaviors], [val[0] for val in self.behaviors.values()]

    def get_current(self):
        ''' Get name and duration (in seconds) of current behavior '''
        duration = (self.world.time_local_ms -
                    self.state_behavior_init_ms) / 1000.0
        return self.state_behavior_name, duration

    def get_previous(self):
        ''' Get name and duration (in seconds) of previous behavior '''
        return self.previous_behavior, self.previous_behavior_duration

    def force_reset(self):
        ''' Force reset next executed behavior '''
        self.state_behavior_name = None

    def execute(self, name, *args) -> bool:
        ''' 
        Execute one step of behavior `name` with arguments `*args`
        - Automatically resets behavior on first call
        - Call get_current() to get the current behavior (and its duration)

        Returns
        -------
        finished : bool
            True if behavior has finished
        '''

        assert name in self.behaviors, f"Behavior {name} does not exist!"

        # Check if transitioning from other behavior
        reset = bool(self.state_behavior_name != name)
        if reset:
            if self.state_behavior_name is not None:
                # Previous behavior was interrupted (did not finish)
                self.previous_behavior = self.state_behavior_name
            self.previous_behavior_duration = (
                self.world.time_local_ms - self.state_behavior_init_ms) / 1000.0
            self.state_behavior_name = name
            self.state_behavior_init_ms = self.world.time_local_ms

        # Control head orientation if behavior allows it
        if self.behaviors[name][1]:
            self.head.execute()

        # Execute behavior
        if not self.behaviors[name][2](reset, *args):
            return False

        # The behavior has finished
        self.previous_behavior = self.state_behavior_name  # Store current behavior name
        self.state_behavior_name = None
        return True

    def execute_sub_behavior(self, name, reset, *args):
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

        assert name in self.behaviors, f"行为 {name} 不存在!"

        # Control head orientation if behavior allows it
        if self.behaviors[name][1]:
            self.head.execute()

        # Execute behavior
        return self.behaviors[name][2](reset, *args)

    def execute_to_completion(self, name, *args):
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

        r = self.world.robot
        skip_last = name not in self.slot_engine.behaviors

        while True:
            done = self.execute(name, *args)
            if done and skip_last:
                break  # Exit here if last command is irrelevant
            self.base_agent.server.commit_and_send(r.get_command())
            self.base_agent.server.receive()
            if done:
                break  # Exit here if last command is part of the behavior

        # reset to avoid polluting the next command
        r.joints_target_speed = np.zeros_like(r.joints_target_speed)

    def is_ready(self, name, *args) -> bool:
        ''' Checks if behavior is ready to start under current game/robot conditions '''

        assert name in self.behaviors, f"Behavior {name} does not exist!"
        return self.behaviors[name][3](*args)

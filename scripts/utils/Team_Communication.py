"""
Communication 是如何工作的？
    say 命令允许玩家向场地上的所有人广播一条消息
    消息范围：50米（场地对角线为36米，因此实际上可以忽略此限制）
    Header perceptor 表明三件事：
        - 消息内容
        - 发送方的队伍
        - 发送方的绝对角度(absolute angle)（如果消息是由自己发送的，则设置为“self”）
    
    消息在下一个 step 中被听到
    消息每2个 step （即0.04秒）才发送一次
    在 muted steps 中发送的消息只能被发送者自己听到
    在一个时间步骤中，player 只能听到一个其他 player 的消息
    如果有两名其他 player 同时说话，只会有第一条消息被听到
    这个 ability 对来自两个队伍的消息独立适用
    理论上，一名 player 可以听到自己的消息、第一位说话的队友的消息以及第一位对手的消息
    实际上，由于我们的团队解析器忽略了来自其他队伍的消息，对手的消息并不重要

    消息特性：
        最多20个字符，ASCII码范围在0x20到0x7E之间，但不包括空格(' ')、左括号('(')和右括号(')')。
        可接受字符包括：字母+数字+符号：!"#$%&'*+,-./:;<=>?@[]^_`{|}~
        但是，由于 server 的一个 bug，发送单引号(')或双引号(")会导致消息提前结束。
"""
from itertools import count
from scripts.commons.Script import Script
from agent.Base_Agent import Base_Agent as Agent


class Team_Communication:

    def __init__(self, script: Script) -> None:
        self.script = script

    def player1_hear(self, msg: bytes, direction, timestamp: float) -> None:
        print(f"Player 1 heard: {msg.decode():20}  from:{direction:7}  timestamp:{timestamp}")

    def player2_hear(self, msg: bytes, direction, timestamp: float) -> None:
        print(f"Player 2 heard: {msg.decode():20}  from:{direction:7}  timestamp:{timestamp}")

    def player3_hear(self, msg: bytes, direction, timestamp: float) -> None:
        print(f"Player 3 heard: {msg.decode():20}  from:{direction:7}  timestamp:{timestamp}")

    def execute(self):

        a = self.script.args

        hear_callbacks = (self.player1_hear,
                          self.player2_hear, self.player3_hear)

        # Args: Server IP, Agent Port, Monitor Port, Uniform No., Robot Type, Team Name, Enable Log, Enable Draw, Play Mode Correction, Wait for Server, Hear Callback
        self.script.batch_create(Agent, ((a.i, a.p, a.m, i+1, 0, a.t, True,
                                 True, False, True, clbk) for i, clbk in enumerate(hear_callbacks)))
        p1: Agent = self.script.players[0]
        p2: Agent = self.script.players[1]
        p3: Agent = self.script.players[2]

        # Beam players
        self.script.batch_commit_beam([(-2, i, 45) for i in range(3)])

        for i in count():
            msg1 = b"I_am_p1!_no:"+str(i).encode()
            msg2 = b"I_am_p2!_no:"+str(i).encode()
            msg3 = b"I_am_p3!_no:"+str(i).encode()
            # commit message
            p1.server.commit_announcement(msg1)
            # commit message
            p2.server.commit_announcement(msg2)
            # commit message
            p3.server.commit_announcement(msg3)
            self.script.batch_commit_and_send()              # send message
            print(f"Player 1 sent:  {msg1.decode()}      HEX: {
                  ' '.join([f'{m:02X}' for m in msg1])}")
            print(f"Player 2 sent:  {msg2.decode()}      HEX: {
                  ' '.join([f'{m:02X}' for m in msg2])}")
            print(f"Player 3 sent:  {msg3.decode()}      HEX: {
                  ' '.join([f'{m:02X}' for m in msg3])}")
            self.script.batch_receive()
            input("Press enter to continue or ctrl+c to return.")

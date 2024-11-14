from agent.Agent import Agent
from scripts.commons.Script import Script
script = Script()  # Initialize: load config file, parse arguments, build cpp modules
a = script.args

# Args: Server IP, Agent Port, Monitor Port, Uniform No., Team name, Enable Log, Enable Draw
# one player for home team
script.batch_create(Agent, ((a.i, a.p, a.m, a.u, a.t,        True, True),))
# one player for away team
script.batch_create(Agent, ((a.i, a.p, a.m, a.u, "Opponent", True, True),))


while True:
    script.batch_execute_agent()
    script.batch_receive()

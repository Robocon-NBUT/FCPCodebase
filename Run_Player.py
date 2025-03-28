from scripts.commons.Script import Script
# Initialize: load config file, parse arguments, build cpp modules
script = Script()
a = script.args

if a.P:  # penalty shootout
    from agent.Agent_Penalty import Agent
else:  # normal agent
    from agent.Agent import Agent

# Args: Server IP, Agent Port, Monitor Port, Uniform No., Team name, Enable Log, Enable Draw, Wait for Server, is magmaFatProxy
if a.D:  # debug mode
    a.u = 9
    player = Agent(a.i, a.p, a.m, a.u, a.t, True, True, False, a.F)
else:
    player = Agent(a.i, a.p, None, a.u, a.t, False, False, False, a.F)

while True:
    player.think_and_send()
    player.server.receive()

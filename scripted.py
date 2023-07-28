import nmmo
from ijcai2022nmmo.scripted import CombatTeam, ForageTeam, RandomTeam
from ijcai2022nmmo.scripted.baselines import Scripted
from ijcai2022nmmo.scripted.scripted_team import ScriptedTeam
from ijcai2022nmmo.scripted import attack, move

import numpy as np 

def l1(start, goal):
    sr, sc = start
    gr, gc = goal
    return abs(gr - sr) + abs(gc - sc)


def foreignClosestTarget(config, ob):
    shortestDist = np.inf
    closestAgent = None

    Entity = nmmo.Serialized.Entity
    agent = ob.agent

    sr = nmmo.scripting.Observation.attribute(agent, Entity.R)
    sc = nmmo.scripting.Observation.attribute(agent, Entity.C)
    start = (sr, sc)

    aid = nmmo.scripting.Observation.attribute(agent, Entity.ID)
    ateam = config.player_team_map[aid]
    
    teammates= set(config.team_players_map[ateam])
    for target in ob.agents:
        exists = nmmo.scripting.Observation.attribute(target, Entity.Self)
        if not exists:
            continue
        
        tid = nmmo.scripting.Observation.attribute(target, Entity.ID)
        if tid in teammates:
            #print("Skipping teammate", tid)
            continue

        tr = nmmo.scripting.Observation.attribute(target, Entity.R)
        tc = nmmo.scripting.Observation.attribute(target, Entity.C)

        goal = (tr, tc)
        dist = l1(start, goal)

        if dist < shortestDist and dist != 0:
            shortestDist = dist
            closestAgent = target

    if closestAgent is None:
        return None, None

    return closestAgent, shortestDist


class AttackWithoutFF(Scripted):
    '''attack'''
    name = 'AttackWithoutFF_'
    
    
    def scan_agents(self):
            '''Scan the nearby area for agents'''
            self.closest, self.closestDist = foreignClosestTarget(
                self.config, self.ob)
            self.attacker, self.attackerDist = attack.attacker(
                self.config, self.ob)

            self.closestID = None
            if self.closest is not None:
                self.closestID = nmmo.scripting.Observation.attribute(
                    self.closest, nmmo.Serialized.Entity.ID)
                
                
            self.attackerID = None
            if self.attacker is not None:
                self.attackerID = nmmo.scripting.Observation.attribute(
                    self.attacker, nmmo.Serialized.Entity.ID)

            self.style = None
            self.target = None
            self.targetID = None
            self.targetDist = None 

    def __call__(self, obs):
        super().__call__(obs)
        self.scan_agents()
        self.target_weak()
        #self.style = nmmo.action.Range
        self.select_combat_style()
        self.attack()
        return self.actions


class AttackTeamWithoutFF(ScriptedTeam):
    agent_klass = AttackWithoutFF
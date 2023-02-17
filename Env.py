import re
import numpy as np
import rvo2
from Circle import Circle
#import matplotlib.pyplot as plt
from typing import Any

np.set_printoptions(3)
class DeepNav():    
    def __init__(self, n_agents : int, scenario : int, width : int = 255, height : int = 255, timestep : float = 0.25 , neighbor_dists : float = 1.0, 
                 time_horizont : float=10.0, time_horizont_obst : float = 20.0, radius : float=2.0, 
                 max_speed : float=3.5) -> None:
        super().__init__()
        
        
        self.n_agents = n_agents
        self.scenario = scenario
        self.width = width
        self.height = height
        self.timestep = timestep
        self.neighbor_dists = neighbor_dists
        self.max_neig = n_agents
        self.time_horizont = time_horizont
        self.time_horizont_obst = time_horizont_obst
        self.radius = radius
        self.max_speed = max_speed
        self. sim = rvo2.PyRVOSimulator(self.timestep, self.neighbor_dists, self.max_neig, self.time_horizont, self.time_horizont_obst, self.radius, self.max_speed)
        self.time = 0.0
        self.T = 0
               
        
        self.positions, self.goals, self.obstacles = self.getScenario().getAgentPosition()
        self.__state = np.zeros((self.n_agents, 10))
        self.__episode_ended = False
        self.__setupScenario()
        self.success = True
        
    def calculateDist(self, a : tuple, b : tuple):
        return np.hypot(a[0] - b[0], a[1] - b[1])  
    
    def __setState(self):        
        for i in range(self.n_agents):
            self.__state[i] = self.getAgentState(i)

    def getAgentState(self, agent):
        dirs = [0, 45, 90, 135, 180, 225, 270, 315]
        a_pos = self.sim.getAgentPosition(agent)
        state = [None] * 10
        state [0] = a_pos[0]
        state [1] = a_pos[1]
        positions = [self.sim.getAgentPosition(a) for a in range(self.n_agents) if a != agent]
        for poss in positions:
            x = a_pos[0] - poss[0]
            y = a_pos[1] - poss[1]
            ang = np.degrees(np.arctan(y/x))
            
            norm = np.linalg.norm((x, y))
            ang_int = np.degrees(np.arcsin(self.radius/norm))
            if np.isnan(ang_int):
                print(norm)
                input()
            ang_inf = np.round((ang - ang_int) + 0.5)
            ang_sup = np.round((ang + ang_int) - 0.5)
            
            ang_range = np.arange(ang_inf, ang_sup)
            
            # ang_range = list(range(ang_inf, ang_sup + 1))
            for indx, a in enumerate(dirs):
                if a in ang_range:
                    state[indx + 2] = norm - self.radius
        # Missing no collision measurements
        for i in range(len(state)):
            if state[i] is None:
                x_dist = a_pos[0] - self.width
                y_dist = a_pos[1] - self.height
                state[i] = np.linalg.norm([x_dist, y_dist])
        return state

    
    
    def __setupScenario(self) -> None:
        for i in self.positions:
            self.sim.addAgent(i)
            
    def reset(self):
        
        self.success = False
        
        for i in range(self.n_agents):
            self.sim.setAgentPosition(i, self.positions[i])
        self.__episode_ended = False
        self.__setState()
        return self.__state
    
    def __isLegal(self, index):
        pos = self.sim.getAgentPosition(index)
        return pos[0] < 256 and pos[0] > -256 and pos[1] < 256 and pos[1] > -256
    
    def isDone(self) -> bool:
        
        if self.T == 1000:
            self.T = 0
            
            self.success = False
            self.__episode_ended = True
            return True
        for i in range(self.n_agents):
            if not self.__agentIsDone(i): 
                return False
        
        self.__episode_ended = True        
        return True
        
        
    def __agentIsDone(self, indx):
        pos = self.sim.getAgentPosition(indx)
        return self.calculateDist(
            pos, self.goals[indx]
        ) <= self.radius
    
    
    
    def __calculateGlobalRwd(self) -> np.float32:
        
        g_rwd : np.float32 = np.zeros(self.n_agents)
        if self.isDone() and not self.success:
            g_rwd -= 100.0
             
        if self.isDone() and self.success:
            g_rwd += 400.0 * self.getGlobalTime()
        
        return g_rwd
    
    def __calculateLocalRwd(self) -> np.float32:
        rwds = np.zeros(self.n_agents)
        r_goal = 0
        r_coll_a = 0 
        r_coll_obs = 0 
        r_done = 0
        r_cong = 0
        
        for i in range(self.n_agents):
            
            r_goal = -np.hypot(self.sim.getAgentPosition(i)[0] - self.goals[i][0], self.sim.getAgentPosition(i)[1] - self.goals[i][1])
            r_cong = -1 - (np.hypot(self.sim.getAgentVelocity(i)[0], self.sim.getAgentVelocity(i)[0]) / self.max_speed)
            for j in range(self.n_agents):
                if not j == i and np.hypot(
                    self.sim.getAgentPosition(i)[0] - self.sim.getAgentPosition(j)[0], 
                    self.sim.getAgentPosition(i)[1] - self.sim.getAgentPosition(j)[1]) < 2 * self.radius:
                   r_coll_a -= 3
                
            if np.hypot(self.sim.getAgentPosition(i)[0] - self.goals[i][0], self.sim.getAgentPosition(i)[1] - self.goals[i][1]) < self.radius:
                r_done += 10
            rwds[i] = r_goal + r_cong + r_coll_a + r_coll_obs + r_done
        
        return rwds
    
        
    def step(self, actions: np.float32):
        
        if self.__episode_ended:
            return self.reset()
        
        self.setPrefferedVel(actions=actions)
        self.sim.doStep()
        
        
        self.time += self.timestep
        self.__setState()
        rwd = np.zeros((self.n_agents,), dtype=np.float32)
        self.T += 1
        
        if self.isDone():
            
            self.__episode_ended = True
           
        
        rwd = self.__calculateGlobalRwd() + self.__calculateLocalRwd()
        
        rwd = rwd.reshape((self.n_agents, 1))
        
        return self.__state, rwd, int(self.__episode_ended)
    
    
    def getAgentVelocity(self, i):
        return self.sim.getAgentVelocity(i)    
       
     
    def getScenario(self) -> Any:
        if self.scenario == 0:
            return Circle(self.n_agents)
    
   
        
    def getGlobalTime(self): return self.time
    
    
    def setPrefferedVel(self, actions: np.float32) -> None:
        
        actions = np.squeeze(actions)
        for i in range(self.n_agents):
            #act = np.squeeze(np.linalg.norm(actions[i], axis=0)[0])
            
            act = actions[i]
            if np.abs(self.getAgentPos(i)[0] + act[0] * self.timestep) > 512 or np.abs(self.getAgentPos(i)[1] + act[1] * self.timestep) > 512:
                return
            self.sim.setAgentPrefVelocity(i, tuple(act))
            
    def getStateSpec(self):
        return self.__state[0].shape[0]
    
    def getActionSpec(self): return 2
    
    def sample(self):
        a = np.random.uniform(-1, 1, (self.n_agents, 2))
        
        return a
        
         
    def getAgentGoal(self, i): return self.goals[i]
    def getAgentPos(self, i): return self.sim.getAgentPosition(i)
    #def sample(self) -> np.float32:
    #    actions =  np.random.random((self.n_agents, 2))
    #    return self.step(actions)
    
           
if __name__ == '__main__':
    
    np.set_printoptions(2)
    env = DeepNav(2, 0)
    
    s = env.reset()
    print(s)

    print(env.getActionSpec(), env.getStateSpec())
    act = env.sample()
    print(env.step(act))
    #print(env.getAgentPos(0))
    #for i in range(100):
    #    env.step(env.sample())
    #print(env.getAgentPos(0))
    #env.reset()
    #print(env.getAgentPos(0))
    
    
   
    

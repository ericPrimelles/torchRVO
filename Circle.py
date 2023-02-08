from math import degrees
from traceback import print_tb
import numpy as np


class Circle:
    
    def __init__(self, n_agents : int) -> None:
        self.n_agents = n_agents
        self.pos : list = []
    
    def getAgentPosition(self) -> list:
        agents = np.arange(self.n_agents)
        self.pos = [(20 * np.cos(i * 2 * np.pi / float(self.n_agents)), 20 * np.sin(i * 2 * np.pi / float(self.n_agents))) for i in agents]
        goals = [(-x, -y) for x, y in self.pos] 
        return self.pos, goals, []
    
   
        
            
import sys    
if __name__ == '__main__':
    
    x = Circle(2)  
   
    print(x.getAgentPosition())   
   
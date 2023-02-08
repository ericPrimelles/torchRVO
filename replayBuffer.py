import numpy as np

class ReplayBuffer:
    max_length : int = 1000
    size_batch : int = 256
    indx : int = 0
    ready : bool = False
    def __init__(self, observation_spec, action_spec, n_agents,max_length : int = 1000, batch_size : int = 256) -> None:
        self.max_length = max_length
        self.size_batch = batch_size
        s_space = list(observation_spec)
        s_space.insert(0, max_length)
        action_spec.insert(0, max_length)
        self.s = self.s_1 = np.empty(s_space, dtype=np.float32)
        self.action = np.empty(action_spec, dtype=np.float32) 
        self.reward =  np.empty((max_length, n_agents, 1), dtype=np.float32)
        self.done = np.empty(max_length, dtype=np.float32)
         
    def store(self, s, a, r, s_1, done) -> None:        
        i = self.indx % self.max_length
        self.s[i] = s
        self.action[i] = a
        self.reward[i] = r
        self.s_1[i] = s_1
        self.done[i] = done
        self.indx += 1
        if self.indx == self.size_batch:
            self.ready = True
    
    def sample(self) -> list:
        if not self.ready:
            return []
        i = 0
        if self.indx < self.max_length: 
            i = self.indx
        else:
            i = self.max_length    
        sample = np.random.randint(0, i, self.size_batch)
        s = [self.s[i] for i in sample]
        a = [self.action[i] for i in sample]
        r = [self.reward[i] for i in sample]
        s_1 = [self.s_1[i] for i in sample]
        done = [self.done[i] for i in sample]
        return np.array(s), np.array(a), np.array(r), np.array(s_1), np.array(done)
import NNmodels as models
from DDPG_Agent import DDPGAgent
from replayBuffer import ReplayBuffer
import torch as T
import torch.nn.functional as F
import numpy as np
# from Env import DeepNav
from pettingzoo.mpe import simple_v2
from time import sleep
import sys

class MADDPG:
    def __init__(self, n_agents, env, obs_space, action_space, tau=0.005,
                 gamma=0.99, l_r=1e-5, path='models/DDPG/', instance_id = ''):        
        
        self.n_agents = n_agents
        self.env = env
        self.obs_space = obs_space
        self.action_space = action_space                        
        self.path = path
        self.gamma = gamma
        self.l_r = l_r
        self.tau = tau   
        self.agents: list[DDPGAgent] = [
            DDPGAgent(agnt, self.n_agents, self.obs_space, self.action_space, self.gamma, self.tau, instance_id=instance_id)
            for agnt in range(self.n_agents)
        ]
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')      
        

    def normalize(self, a):
        norm = np.linalg.norm(a)
        if norm == 0.0:
            return a        
        return a * 1 / norm     

    def choose_action(self, s: T.Tensor, target: bool = False, noise : bool = True):
        acts = np.zeros((s.shape[0], self.n_agents, self.action_space), dtype=np.float32)
        
        for i in range(self.n_agents):
            temp = s[:,i, :]
            acts[:, i] = self.agents[i].choose_action(temp, target, noise)
            acts[:, i] = self.normalize(acts[:, i])            
        return acts

    def learn(self, replay_buffer: ReplayBuffer, device: T.device):
        if not replay_buffer.ready:
            return
        states, actions, rewards, states_1, dones = replay_buffer.sample()
        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards).to(device)
        states_1 = T.tensor(states_1, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        new_pi = T.from_numpy(self.choose_action(states_1, True, False)).to(device)
        pi = T.from_numpy(self.choose_action(states, noise=False)).to(device)

        for i, agnt in enumerate(self.agents):
            t_q_value = agnt.t_critic(states_1, new_pi).flatten()
            #t_q_value[dones[:,0]] = 0.0  
            q_value = agnt.critic(states, actions).flatten()

            target = rewards[:,i].flatten() + agnt.gamma * t_q_value            
           
            q_loss = F.mse_loss(target, q_value)
            agnt.critic.optimizer.zero_grad()
            q_loss.backward(retain_graph=True)
            agnt.critic.optimizer.step()

            actor_loss = agnt.critic(states, pi)
            actor_loss = -T.mean(actor_loss)
            agnt.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agnt.actor.optimizer.step()
            agnt.update_target()

    def train(self, rb: ReplayBuffer, total_steps: int, n_episodes : int):
        acc_rwd = []
        for epoch in range(total_steps):            
            for episode in range(n_episodes):
                self.env.reset()
                s = self.env.observe('agent_0') 
                reward = []
                ts: int = 0
                H :int = 10000
                while 1:
                    s_e = T.unsqueeze(T.from_numpy(np.float32(s)), 0)
                    s_e = T.unsqueeze(T.from_numpy(np.float32(s_e)), 0)
                    a = self.choose_action(s_e)                    
                    #a = self.env.sample()
                    a =  T.squeeze(T.from_numpy(a))
                    env.step(a)                    
                    s_1, r, done, truncation, info = self.env.last()
                    reward.append(r)
                    rb.store(s, a, r, s_1, done)
                    self.learn(rb, self.device)
                                                                    
                    s = s_1
                    ts +=1
                    if done == 1 or ts > H or truncation:                        
                        print(f'Epoch {epoch} Episode {episode} ended after {ts} timesteps Reward {np.mean(reward)}')
                        ts=0
                        acc_rwd.append(np.mean(reward))
                        reward = []
                        
                        break                  
                    
                
            self.save()

            # if epoch % 10 == 0:

            #     self.test()
           
            # dump(rwd, self.path + f'reward_epcohs_{i}.joblib')
        return           

    def get_inputs(self, batch):
        return  
    
    def save(self):
        for i, agnt in enumerate(self.agents):
            agnt.save()            

    def load(self):
        for i, agnt in enumerate(self.agents):
            agnt.load()
            
            
    def test(self, visualize=False):
        
        self.env.reset()
        self.load()
        s, r, done, truncation, info = self.env.last()
        t = 0
        while not done and not truncation:
            s_e = T.unsqueeze(T.from_numpy(np.float32(s)), 0)
            s_e = T.unsqueeze(T.from_numpy(np.float32(s_e)), 0)
            a = self.choose_action(s_e, noise=False)                    
            a =  T.squeeze(T.from_numpy(a))
            self.env.step(a)

            if visualize:
                self.env.render()
                sleep(0.3)                
            s, r, done, truncation, info = self.env.last()

            
            t+= 1
        print(f"Test step ended after {t} steps with reward: {r}")

if __name__ == '__main__':
    # , render_mode='human'
    frame_count, buffer_len, batch_size, epochs, episodes, train, pid = map(int, sys.argv[1:])
    r_mode = None if train == 1 else 'human'
    env = simple_v2.env(max_cycles=frame_count, continuous_actions=True, render_mode=r_mode)
    p = MADDPG(1, env, 4, 5, instance_id=str(pid))
    mem = ReplayBuffer(4, 5, env.max_num_agents, max_length=buffer_len, batch_size=batch_size)
    if train:    
        p.train(mem, epochs, episodes)
    else:
        p.test(True)
    
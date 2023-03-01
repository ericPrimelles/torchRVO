import NNmodels as models
from DDPG_Agent import DDPGAgent
from buffer import MultiAgentReplayBuffer
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

    def learn(self, replay_buffer: MultiAgentReplayBuffer, device: T.device):
        if not replay_buffer.ready:
            return
        actor_states, states, actions, rewards, \
        actor_new_states, states_1, dones = replay_buffer.sample()
        states = T.tensor(states, dtype=T.float).to(device)
        actions = T.tensor(actions, dtype=T.float).to(device)
        rewards = T.tensor(rewards).to(device)
        states_1 = T.tensor(states_1, dtype=T.float).to(device)
        dones = T.tensor(dones).to(device)

        all_new_actions = []
        all_old_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(actor_new_states[agent_idx], 
                                  dtype=T.float).to(device)
            new_pi = agent.t_actor(new_states)
            all_new_actions.append(new_pi)
            all_old_actions.append(actions[agent_idx])

        new_actions = T.cat([action for action in all_new_actions], dim=1)
        old_actions = T.cat([action for action in all_old_actions], dim=1)


        for i, agnt in enumerate(self.agents):
            with T.no_grad():
                t_q_value = agnt.t_critic(states_1, new_actions).flatten()  
                target = rewards[:,i].flatten() + (1 - dones[:,0].int()) + agnt.gamma * t_q_value            
            
            q_value = agnt.critic(states, old_actions).flatten()
            

            q_loss = F.mse_loss(target, q_value)
            agnt.critic.optimizer.zero_grad()
            q_loss.backward(retain_graph=True)
            agnt.critic.optimizer.step()

            mu_states = T.tensor(actor_states[i], dtype=T.float).to(device)
            old_actions_clone = old_actions.clone()
            old_actions_clone[:, i*self.action_space:i*self.action_space + self.action_space] = agnt.actor(mu_states)
            actor_loss = agnt.critic(states, old_actions_clone)
            actor_loss = -T.mean(actor_loss)
            agnt.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agnt.actor.optimizer.step()

        for agent in self.agents:    
            agent.update_target()


    def train(self, rb: MultiAgentReplayBuffer, total_steps: int, n_episodes : int):
        acc_rwd = []
        for epoch in range(total_steps):            
            for episode in range(n_episodes):
                self.env.reset()
                reward = []
                ts: int = 0
                H :int = 10000
                obs = self.env.observe('agent_0') 
                while 1:
                    obs = T.unsqueeze(T.from_numpy(np.float32(obs)), 0)
                    obs = T.unsqueeze(T.from_numpy(np.float32(obs)), 0)
                    actions = self.choose_action(obs)                    
                    #a = self.env.sample()
                    actions =  T.squeeze(T.from_numpy(actions))
                    env.step(actions)                    
                    obs_1, r, done, truncation, info = self.env.last()
                    state = np.array([])
                    for o in obs:
                        o = T.squeeze(T.from_numpy(np.float32(o)))
                        state = np.concatenate([state, o])

                    obs_1 = T.unsqueeze(T.from_numpy(np.float32(obs_1)), 0)

                    state_1 = np.array([])
                    for o in obs_1:
                        state_1 = np.concatenate([state_1, o])
                    
                    r = np.float32(r)
                    reward.append(r)
                    # raw_obs, state, action, reward, 
                            #    raw_obs_, state_, done
                    rb.store_transition(obs, state, actions, r, obs_1, state_1, done)
                    self.learn(rb, self.device)
                                                                    
                    obs = obs_1
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
    n_agents = 1
    frame_count, buffer_len, batch_size, epochs, episodes, train, pid = map(int, sys.argv[1:])
    r_mode = None if train == 1 else 'human'
    env = simple_v2.env(max_cycles=frame_count, continuous_actions=True, render_mode=r_mode)
    actor_dims = []
    for i in range(n_agents):
        agent_key = f"agent_{i}"
        actor_dims.append(env.observation_spaces[agent_key].shape[0])
    critic_dims = sum(actor_dims)
    p = MADDPG(n_agents, env, 4, 5, instance_id=str(pid))
    # mem = ReplayBuffer(4, 5, env.max_num_agents, max_length=buffer_len, batch_size=batch_size)
    # max_size, critic_dims, actor_dims, 
    #         n_actions, n_agents, batch_size
    mem = MultiAgentReplayBuffer(max_size=buffer_len, critic_dims=critic_dims, actor_dims=actor_dims,n_actions=5, n_agents=n_agents, batch_size=batch_size)
    
    if train:    
        p.train(mem, epochs, episodes)
    else:
        p.test(True)
    
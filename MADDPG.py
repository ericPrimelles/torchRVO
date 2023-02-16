import NNmodels as models
from DDPG_Agent import DDPGAgent
from replayBuffer import ReplayBuffer
import torch as T
import torch.nn.functional as F
import numpy as np
from Env import DeepNav


class MADDPG:
    def __init__(self, n_agents, env, obs_space, action_space, tau=0.005,
                 gamma=0.99, l_r=1e-5, path='models/DDPG/'):        
        
        self.n_agents = n_agents
        self.env = env
        self.obs_space = obs_space
        self.action_space = action_space                        
        self.path = path
        self.gamma = gamma
        self.l_r = l_r
        self.tau = tau   
        self.agents: list[DDPGAgent] = [
            DDPGAgent(agnt, self.n_agents, self.obs_space, self.action_space, self.gamma, self.tau)
            for agnt in range(self.n_agents)
        ]
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')      
        

    def normalize(self, a):
        norm = np.linalg.norm(a)        
        return a * 1 / norm     

    def choose_action(self, s: T.Tensor, target: bool = False):
        acts = np.zeros((s.shape[0], self.n_agents, self.action_space), dtype=np.float32)
        
        for i in range(self.n_agents):
            acts[:, i] = self.agents[i].choose_action(s[:,i, :], target)
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

        new_pi = T.from_numpy(self.choose_action(states_1, True)).to(device)
        pi = T.from_numpy(self.choose_action(states)).to(device)

        for i, agnt in enumerate(self.agents):
            t_q_value = agnt.t_critic(states_1, new_pi).flatten()
            #t_q_value[dones[:,0]] = 0.0  
            q_value = agnt.critic(states, pi)

            target = rewards[:,i] + agnt.gamma * t_q_value
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

#           
        #all_agents_new_actions = []
        #all_agents_new_mu_actions = []
        #old_agents_actions = []
#
        #for agent_idx, agent in enumerate(self.agents):
        #    new_states = T.tensor(states_1[agent_idx], 
        #                         dtype=T.float).to(device)
#
        #    new_pi = agent.t_actor.forward(new_states)
#
        #    all_agents_new_actions.append(new_pi)
        #    mu_states = T.tensor(states[agent_idx], 
        #                         dtype=T.float).to(device)
        #    pi = agent.actor.forward(mu_states)
#
        #    all_agents_new_mu_actions.append(pi)
        #    old_agents_actions.append(actions[agent_idx])
        #print(all_agents_new_actions)
        #new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        #mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        #old_actions = T.cat([acts for acts in old_agents_actions],dim=1)
#
        #for agent_idx, agent in enumerate(self.agents):
        #    
        #    critic_value_ = agent.t_critic.forward(states_1, new_actions).flatten()
        #    critic_value_[dones[:,0]] = 0.0
        #    critic_value = agent.critic.forward(states, old_actions).flatten()
#
        #    target = rewards[:,agent_idx] + agent.gamma*critic_value_
        #    critic_loss = F.mse_loss(target, critic_value)
        #    agent.critic.optimizer.zero_grad()
        #    critic_loss.backward(retain_graph=True)
        #    agent.critic.optimizer.step()
#
        #    actor_loss = agent.critic.forward(states, mu).flatten()
        #    actor_loss = -T.mean(actor_loss)
        #    agent.actor.optimizer.zero_grad()
        #    actor_loss.backward(retain_graph=True)
        #    agent.actor.optimizer.step()
        #    agent.update_target()        
        #return
#
    def train(self, rb: ReplayBuffer, total_steps: int, n_episodes : int):
        acc_rwd = []
        for epoch in range(total_steps):            
            for episode in range(n_episodes):
                s = self.env.reset() 
                reward = []
                ts: int = 0
                H :int = 10000
                while 1:
                    s_e = T.unsqueeze(T.from_numpy(np.float32(s)), 0)
                    a = self.choose_action(s_e)                    
                    #a = self.env.sample()
                    s_1, r, done = self.env.step(a)                    
                    reward.append(r)
                    rb.store(s, a, r, s_1, done)
                    self.learn(rb, self.device)
                                                                    
                    s = s_1
                    ts +=1
                    if done == 1 or ts > H:                        
                        print(f'Epoch {epoch} Episode {episode} ended after {ts} timesteps Reward {np.mean(reward)}')
                        ts=0
                        acc_rwd.append(np.mean(reward))
                        reward = []
                        
                        break                  
                    
                
            # self.save()
            # self.test()
           
            # dump(rwd, self.path + f'reward_epcohs_{i}.joblib')
        return           

    def get_inputs(self, batch):
        return
    
    def save(self, filename):
        T.save(self.actor.state_dict(), filename)

    def load(self, filename):
        self.actor.load_state_dict(T.load(filename))
        self.actor.eval()

    # def save(self):
    #     for i in self.agents:
    #         _id = i.agent
    #         i.critic.save_weights(self.path + f'QNet_{_id}.h5')
    #         i.t_critic.save_weights(self.path + f'QTargetNet_{_id}.h5')
        
    #         i.actor.save_weights(self.path + f'ANet_{_id}.h5')
    #         i.t_actor.save_weights(self.path + f'ATargetNet_{_id}.h5')
            
    # def load(self):
    #     for i in range(self.n_agents):
    #         _id = self.agents[i].agent
    #         self.agents[i].critic.load_weights(self.path + f'QNet_{_id}.h5')
    #         self.agents[i].t_critic.load_weights(self.path + f'QTargetNet_{_id}.h5')

    #         self.agents[i].actor.load_weights(self.path + f'ANet_{_id}.h5')
    #         self.agents[i].t_actor.load_weights(self.path + f'ATargetNet_{_id}.h5')

if __name__ == '__main__':

    env = DeepNav(4, 0)
    p = MADDPG(4, env, env.getStateSpec(), env.getActionSpec())
    mem = ReplayBuffer(env.getStateSpec(), env.getActionSpec(), env.n_agents, max_length=10000)
    p.train(mem, 1000, 10)
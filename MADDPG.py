import NNmodels as models
from DDPG_Agent import DDPGAgent
from replayBuffer import ReplayBuffer
import torch as T
import torch.functional as F
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
            

    def normalize(self, a):
        norm = np.linalg.norm(a)        
        return a * 1 / norm     

    def choose_action(self, s: T.Tensor, target: bool = False):
        if s._rank() <= 2:
                s = T.unsqueeze(s, 0)
        if target:
            acts = T.stack([
                self.agents[i].t_actor.choose_action(s[:, i, :])
                for i in range(self.n_agents) 
            ])    
            return acts            
        acts = T.stack([
            self.agents[i].actor.choose_action(s[:, i, :])
            for i in range(self.n_agents) 
        ])
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

        all_agents_new_actions = []
        all_agents_new_mu_actions = []
        old_agents_actions = []

        for agent_idx, agent in enumerate(self.agents):
            new_states = T.tensor(states_1[agent_idx], 
                                 dtype=T.float).to(device)

            new_pi = agent.t_actor.forward(new_states)

            all_agents_new_actions.append(new_pi)
            mu_states = T.tensor(actions[agent_idx], 
                                 dtype=T.float).to(device)
            pi = agent.actor.forward(mu_states)
            all_agents_new_mu_actions.append(pi)
            old_agents_actions.append(actions[agent_idx])

        new_actions = T.cat([acts for acts in all_agents_new_actions], dim=1)
        mu = T.cat([acts for acts in all_agents_new_mu_actions], dim=1)
        old_actions = T.cat([acts for acts in old_agents_actions],dim=1)

        for agent_idx, agent in enumerate(self.agents):
            critic_value_ = agent.t_critic.forward(states_1, new_actions).flatten()
            critic_value_[dones[:,0]] = 0.0
            critic_value = agent.critic.forward(states, old_actions).flatten()

            target = rewards[:,agent_idx] + agent.gamma*critic_value_
            critic_loss = F.mse_loss(target, critic_value)
            agent.critic.optimizer.zero_grad()
            critic_loss.backward(retain_graph=True)
            agent.critic.optimizer.step()

            actor_loss = agent.critic.forward(states, mu).flatten()
            actor_loss = -T.mean(actor_loss)
            agent.actor.optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            agent.actor.optimizer.step()
            agent.update_target()        
        return

    def train(self, replay_buffer: ReplayBuffer, total_steps: int):
        acc_rwd = []
        for epoch in range(self.n_epochs):            
            for episode in range(self.n_episodes):
                s = self.env.reset() 
                reward = []
                ts: int = 0
                H :int = 10000
                while 1:
                    a = self.policy(s)                    
                    #a = self.env.sample()
                    s_1, r, done = self.env.step(a)                    
                    reward.append(r)
                    self.rb.store(s, a, r, s_1, done)
                    if self.rb.ready:
                        s_s, a_s, r_s, s_1_s, dones_s = replay_buffer.sample()
                        a_s = a_s.reshape((self.n_agents, 64, 2))
                        s_s = T.from_numpy(s_s)
                        #a_s = tf.convert_to_tensor(a_s)
                        r_s = T.from_numpy(r_s)
                        s_1_s = T.from_numpy(s_1_s)
                        dones_s = T.from_numpy(dones_s)
                        a_state = self.choose_action(s_s, training=True)
                        t_a_state = self.choose_action(s_1_s, True, True)
                        for i in range(self.n_agents):
                            self.agents[i].update(s_s, a_s, r_s, s_1_s, a_state, t_a_state)
                            self.update_target(self.agents[i].t_critic.variables, self.agents[i].critic.variables, i)
                            self.update_target(self.agents[i].t_actor.variables, self.agents[i].actor.variables, i)                                                    
                    s = s_1
                    ts +=1
                    if done == 1 or ts > H:                        
                        print(f'Epoch {epoch} Episode {episode} ended after {ts} timesteps Reward {np.mean(reward)}')
                        ts=0
                        acc_rwd.append(np.mean(reward))
                        reward = []
                        self.epsion += self.epsilon_increment
                        break                  
                    
                
            self.save()
            self.test()
           
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

    env = DeepNav(2, 0)
    p = MADDPG(2, env, env.getStateSpec(), env.getActionSpec())

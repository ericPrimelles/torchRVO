import torch.nn as nn
from NNmodels import DDPGActor
from NNmodels import DDPGCritic
from torch.optim import Adam
import torch as T

class DDPGAgent:
    def __init__(self, agnt : int, n_agents : int, obs_space : int, action_space : int, gamma : float = 0.95, tau=0.01, chkpt=''):
        self.agent = agnt
        self.n_agents = n_agents
        self.obs_space = obs_space
        self.action_space = action_space
        self.actor : DDPGActor = DDPGActor(self.obs_space, self.action_space, chkpt=chkpt, name=f'A_{self.agent}')
        self.t_actor : DDPGActor = DDPGActor(self.obs_space, self.action_space, chkpt=chkpt, name=f'A_t_{self.agent}')
        self.critic : DDPGCritic = DDPGCritic(self.n_agents * (self.obs_space + self.action_space), chkpt=chkpt, name=f'C_{self.agent}')
        self.t_critic : DDPGCritic = DDPGCritic(self.n_agents * (self.obs_space + self.action_space), chkpt=chkpt, name=f'C_{self.agent}')
        self.gamma = gamma             
        # Set the initial weigths equally for actors and critics
        self.tau = tau
        self.update_target(tau=1.0)    

    def update_target(self, tau : float = None):
        if tau == None:
            tau = self.tau
        
        t_a_p = self.t_actor.named_parameters()
        a_p = self.actor.named_parameters()

        t_a_d = dict(t_a_p)
        a_d = dict(a_p)

        for name in a_d:
            a_d[name] = tau*a_d[name].clone() + \
                    (1-tau)*t_a_d[name].clone()
       
        self.t_actor.load_state_dict(a_d)

        t_c_p = self.t_critic.named_parameters()
        c_p = self.critic.named_parameters()

        t_c_d = dict(t_c_p)
        c_d = dict(c_p)

        for name in c_d:
            c_d[name] = tau*c_d[name].clone() + \
                    (1-tau)*t_c_d[name].clone()

        self.t_critic.load_state_dict(c_d)

    
    def save(self):
        self.actor.save()
        self.t_actor.save()
        self.critic.save()
        self.t_critic.save()

    def load(self):
        self.actor.load()
        self.t_actor.load()
        self.critic.load()
        self.t_critic.load()

    def choose_action(self, obs, target : bool = False):

        if target:
            state = obs.to(self.actor.device)
            actions = self.t_actor.forward(state)
            noise = T.rand(self.action_space).to(self.actor.device)
            action = actions + noise
            return action.detach().cpu().numpy()[0]
        
        state = obs.to(self.actor.device)
        actions = self.actor.forward(state)
        noise = T.rand(self.action_space).to(self.actor.device)
        action = actions + noise
        return action.detach().cpu().numpy()[0]
        
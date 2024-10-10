import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal

class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.log_probs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.log_probs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, 
                 continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.continuous_action_space = continuous_action_space  

        if self.continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), 
                                        action_std_init * action_std_init)

        if self.continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):
        if self.continuous_action_space:
            self.action_var = torch.full((self.action_dim,), 
                                        new_action_std * new_action_std)

    def action(self, state):
        with torch.no_grad():
            if self.continuous_action_space:
                action_mean = self.actor(state).detach()
                # use unsqueeze to add a dimension
                cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
                dist = MultivariateNormal(action_mean, cov_mat)
            else:
                action_probs = self.actor(state).detach()
                dist = Categorical(action_probs)

            action = dist.sample()
            action_log_prob = dist.log_prob(action).detach()
            state_value = self.critic(state).detach()

        return action, action_log_prob, state_value

    def evaluate(self, state, action):

        if self.continuous_action_space:

            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var)
            dist = MultivariateNormal(action_mean, cov_mat)

            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            # the evaluate function is not detached 
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        # the log_prob function should be called with a tensor
        action_log_prob = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)

        return state_value, action_log_prob, dist_entropy
    
class PPO:

    def __init__(self, state_dim, action_dim, actor_lr, critic_lr, gamma, 
                 K_epochs, eps_clip, continuous_action_space, action_std):
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.continuous_action_space = continuous_action_space
        
        if continuous_action_space:
            self.action_std = action_std

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, 
                                  continuous_action_space, action_std)

        # update actor and critic jointly
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': actor_lr},
            {'params': self.policy.critic.parameters(), 'lr': critic_lr}
        ])

        self.old_policy = ActorCritic(state_dim, action_dim, 
                                      continuous_action_space, action_std)
        self.old_policy.load_state_dict(self.policy.state_dict())

        self.MSELoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if self.continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.old_policy.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        if self.continuous_action_space:

            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)

            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

    def select_action(self, state):
    
        if self.continuous_action_space:

            with torch.no_grad():
                state = torch.FloatTensor(state)
                # need to change
                # state = state.unsqueeze(0)
                action, action_log_probs, state_value = self.old_policy.action(state)
        
            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.log_probs.append(action_log_probs)
            self.buffer.state_values.append(state_value)

            return action.detach().numpy().flatten()
        else:

            with torch.no_grad():
                state = torch.FloatTensor(state)
                action, action_log_probs, state_value = self.old_policy.action(state)
                # action, action_log_probs, state_value = self.policy.action(state)

            self.buffer.states.append(state)
            self.buffer.actions.append(action)
            self.buffer.log_probs.append(action_log_probs)
            self.buffer.state_values.append(state_value)

            return action.item()

    def update(self):

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # the buffer store the detached states, actions, log_probs, state_values
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0))
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0))
        old_logprobs = torch.squeeze(torch.stack(self.buffer.log_probs, dim=0))
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0))

        advantage = rewards - old_state_values

        for _ in range(self.K_epochs):

            # not detached
            state_values, log_probs, dis_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)  
            
            ratios = torch.exp(log_probs - old_logprobs)
            surr1 = ratios * advantage
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            
            loss = -torch.min(surr1, surr2) + 0.5 * self.MSELoss(state_values, rewards) - 0.01 * dis_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):

        # map_location: load the data to the original device
        self.old_policy.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
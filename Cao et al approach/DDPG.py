import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque
from world import State, Transition
from typing import Iterable, Callable, NamedTuple
import numpy as np
from noise import ParameterNoise

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory(object):
    def __init__(self, capacity: int = 1000000):
        self.memory = deque([], maxlen=capacity)

    def push(self, transitions: Iterable[Transition]) -> None :
        """Save multiple transitions for later replay."""
        self.memory.extend(transitions)

    def sample(self, batch_size: int) -> list[NamedTuple] :
        return random.sample(self.memory, batch_size)

    def __len__(self) -> float:
        return len(self.memory)

class Critic(nn.Module):
    def __init__(self, i: int, n: int, o: int):
        """
        i: input size, in this case it's state space dim + action space dim i.e. 4
        n: number of neurons in each layer
        o: output size, in this case it's action space dim i.e. 1
        """
        super(Critic, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(i, n),
            nn.LayerNorm(normalized_shape=n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.LayerNorm(normalized_shape=n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.LayerNorm(normalized_shape=n),
            nn.ReLU(),
            nn.Linear(n, o)
        )
    def forward(self, state, action):
        return self.stack(torch.cat((state, action), 1))

class Actor(nn.Module):
    def __init__(self, i: int, n: int, o: int):
        """
        i: input size, in this case it's state space dim i.e. 3
        n: number of neurons in each layer
        o: output size, in this case it's action space dim i.e. 1
        """
        super(Actor, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(i, n),
            #layer norm
            nn.LayerNorm(normalized_shape=n),
            nn.ReLU(),
            nn.Linear(n, n),
            #layer norm
            nn.LayerNorm(normalized_shape=n),
            nn.ReLU(),
            nn.Linear(n, o),
            nn.LayerNorm(normalized_shape=o),
            #layer norm
            nn.Sigmoid()
        )
    def forward(self, state):
        return self.stack(state)

class DDPGagent():
    def __init__(self,
                 n_actor: int,
                 n_critic: int,
                 gamma: float = 0.9,
                 tau: float = 0.1,
                 learn_every: int = 10,
                 n_learn: int = 10,
                 batch_size: int = 100,
                 state_dim: int = 3,
                 action_dim: int = 1):

        self.ReplayMemory  = ReplayMemory()
        self.gamma         = gamma
        self.tau           = tau
        self.batch_size    = batch_size
        self.n_learn       = n_learn

        self.counter       = 0
        self.learn_every   = learn_every

        self.actor         = Actor(state_dim, n_actor, action_dim)
        self.critic        = Critic(state_dim + action_dim, n_critic, action_dim)
        self.critic_2      = Critic(state_dim + action_dim, n_critic, action_dim)

        self.perturbed_actor = Actor(state_dim, n_actor, action_dim)

        self.init_weights(self.actor)
        self.init_weights(self.perturbed_actor)
        self.init_weights(self.critic)
        self.init_weights(self.critic_2)

        self.actor_target  = Actor(state_dim, n_actor, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_target = Critic(state_dim + action_dim, n_critic, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_loss   = nn.MSELoss()

        self.critic_2_target = Critic(state_dim + action_dim, n_critic, action_dim)
        self.critic_2_target.load_state_dict(self.critic.state_dict())

        self.critic_2_loss   = nn.MSELoss()

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-5)
        self.critic_2_optimizer = optim.Adam(self.critic.parameters(), lr=1e-5)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-5)

    @staticmethod
    def init_weights(network: Actor|Critic) -> None:
        for module in network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)

    def perturb_actor_parameters(self, param_noise: ParameterNoise) -> None:
        """
        apply parameter noise for exploration
        """
        self.soft_update(self.actor, self.perturbed_actor, tau=1) # equiv to hard update
        params = self.perturbed_actor.state_dict()
        for name in params:
            param = params[name]
            param += torch.randn(param.shape)*param_noise.current_std

    def save(self, path: str = "agent\\"):
        torch.save(self.actor.state_dict(),path+"actor.pt")
        torch.save(self.critic.state_dict(),path+"critic.pt")
        torch.save(self.critic_2.state_dict(),path+"critic_2.pt")
        torch.save(self.actor_target.state_dict(),path+"actor_target.pt")
        torch.save(self.critic_target.state_dict(),path+"critic_target.pt")
        torch.save(self.critic_2_target.state_dict(),path+"critic_2_target.pt")

    def learn(self, transition: Iterable[Transition]) -> None:

        states, actions, next_states, rewards = Transition(*zip(*transition))
        states = torch.tensor([*states]).float()
        actions = torch.tensor([*actions]).float().reshape(1,-1).t()
        next_states = torch.tensor([*next_states]).float()
        rewards = torch.tensor([*rewards]).float().reshape(1,-1).t()

        next_actions = self.act(next_states, self.actor_target)

        ## Q_1
        Q_targets_next = self.Q(next_states, next_actions.detach(), self.critic_target)
        Q_targets = rewards + self.gamma * Q_targets_next
        Q_expected = self.Q(states, actions, self.critic)
        critic_loss = self.critic_loss(Q_expected, Q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()

        # Q_2
        Q_2_targets_next = self.Q(next_states, next_actions.detach(), self.critic_2_target)
        Q_2_targets = rewards**2 + Q_2_targets_next*(self.gamma)**2 + 2*self.gamma*rewards*Q_targets_next
        Q_2_expected = self.Q(states, actions, self.critic_2)
        critic_2_loss = self.critic_2_loss(Q_2_expected, Q_2_targets)

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        ## actor
        pred_actions = self.act(states, self.actor)

        # actor_loss = -self.critic(states, pred_actions.detach()).mean()
        q1 = self.critic(states, pred_actions.detach())
        q2 = self.critic_2(states, pred_actions.detach())
        actor_loss = - torch.mean(q1 + 1.5*torch.sqrt(torch.abs(q2 - q1**2))) # torch.mean() is because backward() takes a scalar

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        print("critic", critic_loss.detach())
        print("critic_2", critic_2_loss.detach())
        print("actor", actor_loss.detach())

        self.soft_update(self.critic, self.critic_target, self.tau)
        self.soft_update(self.critic_2, self.critic_2_target, self.tau)
        self.soft_update(self.actor, self.actor_target, self.tau)

    def step(self, transition: Iterable[Transition]):
        self.counter += 1
        self.ReplayMemory.push(transition)
        if (self.counter % self.learn_every == 0 and len(self.ReplayMemory) > self.batch_size) :
            for _ in range(self.n_learn):
                experiences = self.ReplayMemory.sample(self.batch_size)
                self.learn(experiences)

    def soft_update(self, model: Actor|Critic, target_model: Actor|Critic, tau: float) -> None:
        for target_param, param in zip(target_model.parameters(), model.parameters()):
            target_param.data.copy_(tau*param.data + (1.0 - tau)*target_param.data)

    def act(self, states: torch.Tensor, actor: Actor) -> torch.Tensor:
        action = actor.forward(states)
        return action

    def Q(self, states: torch.Tensor, actions: torch.Tensor, critic: Critic) -> torch.Tensor:
        Q = critic.forward(states, actions)
        return Q
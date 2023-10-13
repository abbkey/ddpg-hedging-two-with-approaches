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
    def __init__(self, capacity: int = 2000):
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
        i: input size, in this case it's state space dim + action space dim i.e. 6
        n: number of neurons in each layer
        o: output size, in this case it's action space dim i.e. 1
        """
        super(Critic, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(i, 2*n),
            nn.LayerNorm(normalized_shape=2*n, elementwise_affine=True),
            nn.LeakyReLU(),
            nn.Linear(2*n, n),
            nn.LayerNorm(normalized_shape=n, elementwise_affine=True),
            nn.LeakyReLU(),
            nn.Linear(n, o)
        )
    def forward(self, state, action):
        return self.stack(torch.cat((state, action), 1))

class Actor(nn.Module):
    def __init__(self, i: int, n: int, o: int):
        """
        i: input size, in this case it's state space dim i.e. 5
        n: number of neurons in each layer
        o: output size, in this case it's action space dim i.e. 1
        """
        super(Actor, self).__init__()
        self.stack = nn.Sequential(
            nn.Linear(i, 2*n),
            nn.LayerNorm(normalized_shape=2*n, elementwise_affine=True),
            nn.LeakyReLU(),
            nn.Linear(2*n, n),
            nn.LayerNorm(normalized_shape=n, elementwise_affine=True),
            nn.LeakyReLU(),
            nn.Linear(n, o),
            nn.Sigmoid()
        )
    def forward(self, state):
        return self.stack(state)

class DDPGagent():
    def __init__(self,
                 n_actor: int,
                 n_critic: int,
                 gamma: float = 0.99,
                 tau: float = 0.1,
                 learn_every: int = 10,
                 n_learn: int = 5,
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

        self.perturbed_actor = Actor(state_dim, n_actor, action_dim)

        self.init_weights(self.actor)
        self.init_weights(self.perturbed_actor)
        self.init_weights(self.critic)

        self.actor_target  = Actor(state_dim, n_actor, action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_target = Critic(state_dim + action_dim, n_critic, action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.critic_loss   = nn.MSELoss(reduction = 'none')

        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

    @staticmethod
    def init_weights(network: Actor|Critic) -> None:
        for module in network.modules():
            # if isinstance(module, nn.LayerNorm):
            #     module.bias.data.zero_()
            #     module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear):
                module.weight = nn.init.xavier_uniform_(module.weight)
                # if module.bias is not None:
                #     module.bias.data.zero_()

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
        torch.save(self.actor.state_dict(),path+"_actor.pt")
        torch.save(self.critic.state_dict(),path+"_critic.pt")
        torch.save(self.actor_target.state_dict(),path+"_actor_target.pt")
        torch.save(self.critic_target.state_dict(),path+"_critic_target.pt")

    def learn(self, transition: Iterable[Transition]) -> None:

        states, actions, next_states, rewards = Transition(*zip(*transition))
        states = torch.tensor([*states]).float()
        actions = torch.tensor([*actions]).float().reshape(1,-1).t()
        next_states = torch.tensor([*next_states]).float()
        rewards = torch.tensor([*rewards]).float().reshape(1,-1).t()
        # we normalize the rewards for stability reasons 
        # rewards -= torch.mean(rewards)
        # rewards /= torch.std(rewards)

        next_actions = self.act(next_states, self.actor_target)

        ## Q
        with torch.no_grad():
            Q_targets_next = self.Q(next_states, next_actions.detach(), self.critic_target)
            Q_targets = rewards + self.gamma * Q_targets_next * (1 - (next_states[:,0]==0).reshape(1,-1).t()*1)
            # Q_targets = rewards + self.gamma * Q_targets_next 

        self.critic_optimizer.zero_grad()
        Q_expected = self.Q(states, actions, self.critic)
        critic_loss = self.critic_loss(Q_expected, Q_targets).mean()

        critic_loss.backward()
        self.critic_optimizer.step()

        ## actor
        self.actor_optimizer.zero_grad()
        pred_actions = self.act(states, self.actor)

        actor_loss = -self.critic(states, pred_actions.detach()).mean()

        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic, self.critic_target, self.tau)
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

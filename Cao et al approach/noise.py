
import torch
import numpy as np

class OUNoise(object):
    def __init__(self,
                 mu: float = 0.,
                 theta: float = 0.15,
                 max_sigma: float = 0.3,
                 min_sigma:float = 0.3,
                 decay_period: float = 1000,
                 action_max: float = 1.,
                 action_min: float = -1.):

        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.low = action_min
        self.high = action_max
        self.reset()

    def reset(self, nbr_actions: int = 1) -> None:
        self.state = torch.ones(nbr_actions, 1)*self.mu

    def evolve_state(self, nbr_actions: int) -> torch.Tensor:
        x = self.state
        dx = self.theta*(self.mu - x) + self.sigma*torch.randn(nbr_actions, 1)
        self.state += dx
        return self.state

    def get_action(self, action: float, t: float = 0., nbr_actions: int = 1) -> torch.Tensor:
        ou_state = self.evolve_state(nbr_actions)
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma)*min(1.0, t/self.decay_period)
        return torch.clip(action + ou_state, self.low, self.high)

class ParameterNoise(object):
    def __init__(self,
                 initial_std: float = 0.1,
                 theshhold: float = 0.3,
                 scaling_factor: float = 1.01):
        """
        initial_std and current_std refer to the std of the parameter noise,
        and threshhold in our case refers to the desired std in the action space noise.
        """
        self.initial_std = initial_std
        self.theshhold = theshhold
        self.scaling_factor = scaling_factor

        self.current_std = initial_std

    def adapt(self, distance: float) -> None:
        if distance > self.theshhold:
            # decrease the std
            self.current_std /= self.scaling_factor
        else:
            # increase the std
            self.current_std *= self.scaling_factor
    
    def ddpg_distance(actions: torch.Tensor, 
                      perturbed_actions: torch.Tensor) -> float:
        """
        implements the distance proposed by Plappert (2018) for ddpg.
        in our case, N=1 (action space is 1-dimensional)
        """
        return torch.sqrt(torch.mean(torch.square(actions-perturbed_actions)))
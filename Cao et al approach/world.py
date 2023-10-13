import numpy as np
from typing import Iterator, Dict, Generator, NamedTuple
from collections import namedtuple
from scipy.stats import norm 

State = namedtuple('State',
                  ('ttm','holding','log_price'))
BS = namedtuple('parameters',
               ('r','sigma'))
H = namedtuple('parameters',
                ('r','initial_vo','volvol','mrs'))
Transition = namedtuple('Transition',
                        ('state','action','next_state','reward'))

class Model():
    """Defines the model used to diffuse the underlying 
    and eventually price (if possible) the derivatives on this underlying."""
    def __init__(self, name: str = "BS", parameters: NamedTuple = BS(0.027, 0.19)):
        self.name = name
        self.parameters = parameters

class Underlying():
    """Defines the dynamics of the underlying. 
    Can follow he BS, heston or other dynamics."""
    def __init__(self, 
                spot : float = 100,
                time_step: float = 0.01,
                ):
        self.spot = spot
        self.time_step = time_step
    
    def paths(self, 
                model: Model = Model(),
                nbr_paths: int = 1, 
                T: float = 1,
                ) -> np.ndarray:
        """Diffuses the underlying using the specified model.
        """
        if model.name == "BS" : 
            r, sigma = model.parameters
            n = int(T/self.time_step)
            S = np.zeros((nbr_paths, n+1))
            S[:,0] = self.spot
            for i in range(n):
                W = np.sqrt(self.time_step)*np.random.randn(nbr_paths)
                S[:,i+1] = S[:,i]*(1 + sigma*W + r*self.time_step)

        return S

class EuropeanOption():
    """Defines a euro option on an underlying."""
    def __init__(self,
                 underlying: Underlying = Underlying(),
                 strike: float = 100,
                 maturity: float = 1, # in months 
                 option: int = 0
                 ):
        self.option = option # 0 : call, 1 : put
        self.underlying = underlying
        self.strike = strike
        self.maturity = maturity
    
    def payoff(self, spot: float):
        diff = spot - self.strike
        if self.option==0:
            return np.maximum(diff, 0)
        if self.option==1:
            return np.minimum(diff, 0)
        
    def price(self, 
              model: Model = Model(),
              t: float = 0,
              spot: float = 100
            ) -> float : 
        """Prices a euro option using a specified model."""
        if round(t,3) == self.maturity:
            return self.payoff(spot=spot)
        # if round(t,3) < self.maturity:
        else:
            if model.name == "BS" : 
                t = round(t, 3)
                r, sigma = model.parameters
                d_1 = (np.log(spot/self.strike) + (r + 0.5*sigma**2)*(self.maturity-t))/(sigma*np.sqrt(self.maturity-t))
                d_2 = d_1 - sigma*np.sqrt(self.maturity-t)
                call_price = norm.cdf(x=d_1)*spot - norm.cdf(x=d_2)*self.strike*np.exp(-r*(self.maturity-t)) 
                price = self.option*(self.strike*np.exp(-r*(self.maturity-t)) - spot) + call_price
                return price
    
    def bs_hedge(self, 
              model: Model = Model(),
              t: float = 0,
              spot: float = 100
            ) -> float : 
        if model.name == "BS" : 
            t = round(t, 3)
            r, sigma = model.parameters
            if t>=0 and t<self.maturity : 
                d_1 = (np.log(spot/self.strike) + (r + 0.5*sigma**2)*(self.maturity-t))/(sigma*np.sqrt(self.maturity-t))
                delta = norm.cdf(x=d_1) if self.option==0 else -norm.cdf(x=-d_1)
            else:
                delta = 0
        return delta

class World(EuropeanOption):
    """
    Creates the evnironement where the agent trains. It contains both the data structures that 
    are observable (state) and not observable (strike, maturity etc) to the agent.
    Our situation is that of a trader hedging a (short) position in a (call) option. 
    She can rebalance her position at time inetrvals delta_t  and is subject to trading costs.
    The environement state (observable by the agent) at time i*time_step is defined by :
        ttm: float, time to maturity 
        holding: float, holding of the asset during [(i-1)*time_step, i*time_step)
        spot: float, asset price at time i*time_step
    """

    def __init__(self,
                 delta_t: float = 0.2, # rebalance interval in months 
                 underlying: Underlying = Underlying(),
                 strike: float = 100,
                 maturity: float = 1, # in months 
                 option: int = 0,
                 trading_cost: float = 0.01 # constant percentage
                ):
        super().__init__(underlying, strike, maturity, option)
        self.delta_t = delta_t
        self.trading_cost = trading_cost

    def show(self) -> Dict:
        return self.__dict__

    def state(self, 
              model: Model = Model(),
              t: float = 0.5, 
              holding: float = 1
              ) -> Generator:
        """
        Returns a generator for the environement state using a specified model (to diffuse the 
        underlying) and time t and the holding of the previous period.
        """
        while True: 
            ttm = round(self.maturity - t, 3)
            current_period_index = int(np.floor(t/self.delta_t)) # i*delta_t <= t <= (i+1)*delta_t and we're looking for i
            n = int(self.delta_t/self.underlying.time_step) # each period delta_t has n*time_step steps for discretization
            assert t <= self.maturity, f"Pricing time {t} should be smaller than maturity {self.maturity}"
            X = self.underlying.paths(model=model, T=self.maturity)[0]
            if t == self.maturity:
                spot = X[-1]
            else:
                spot = X[n*current_period_index]

            yield State(ttm, holding, np.log(spot/self.strike))            
    
    def pnl_reward(self,
                   state: State,
                   next_state: State,
                   model: Model = Model()
                   ) -> float:
        """
        calculates the pnl reward for an action using two consecutive state infos.
        """
        ttm_i_1, H_i_1, log_S_i_1 = state
        ttm_i, action, log_S_i = next_state
        ttm_i, ttm_i_1 = round(ttm_i, 3), round(ttm_i_1, 3)

        S_i_1, S_i = np.exp(log_S_i_1)*self.strike, np.exp(log_S_i)*self.strike

        # if ttm_i <= 0:
        #     R = self.payoff(spot=S_i) - self.trading_cost*np.abs(S_i*action) 

        # if ttm_i_1 >= self.maturity :
        #     print("here")
        #     R = self.price(model, 0, S_i) - self.trading_cost*np.abs(S_i*action) 

        # else:
        #     V_i = self.price(model, self.maturity-ttm_i, S_i)
        #     V_i_1 = self.price(model, self.maturity-ttm_i_1, S_i_1)
        #     R = V_i - V_i_1 + H_i_1*(S_i - S_i_1) - self.trading_cost*np.abs(S_i*(action - H_i_1))
        V_i = self.price(model, self.maturity-ttm_i, S_i)
        V_i_1 = self.price(model, self.maturity-ttm_i_1, S_i_1)
        # R = V_i - V_i_1 + H_i_1*(S_i - S_i_1) - self.trading_cost*np.abs(S_i*(action - H_i_1))
        R = V_i_1 - V_i + action*(S_i - S_i_1) - self.trading_cost*np.abs(S_i*(action - H_i_1))

        return R
    
    def get_transition(self,
                       state: State,
                       action: float,
                       ) -> Generator:
        ttm,_, log_spot = state
        S = Underlying(spot=np.exp(log_spot)*self.strike)
        next_spot = S.paths(T=self.delta_t)[0][-1]
        ttm = round(ttm,3)

        next_state = State(round(ttm-self.delta_t, 3), action, np.log(next_spot/self.strike))
        reward = self.pnl_reward(state, next_state)

        yield Transition(state, action, next_state, reward)

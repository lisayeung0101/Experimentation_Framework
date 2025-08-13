from dataclasses import dataclass
import numpy as np
from scipy.stats import norm

@dataclass
class PowerParams:
    baseline: float  # baseline conversion rate
    mde: float 
    alpha: float = 0.05
    power: float = 0.8
    two_tailed: bool = True

def sample_size_proportions(params: PowerParams) -> int:
    p1 = params.baseline
    p2 = params.baseline + params.mde
    alpha = params.alpha
    beta = 1 - params.power
    z_alpha = norm.ppf(1 - alpha/2) if params.two_tailed else norm.ppf(1 - alpha)
    z_beta = norm.ppf(1 - beta)
    p_bar = (p1 + p2) / 2
    q_bar = 1 - p_bar
    num = (z_alpha * np.sqrt(2 * p_bar * q_bar) + z_beta * np.sqrt(p1*(1-p1) + p2*(1-p2))) ** 2
    den = (p2 - p1) ** 2
    return int(np.ceil(num / den))

def sample_size_means(sd: float, params: PowerParams) -> int:
    alpha = params.alpha
    beta = 1 - params.power
    z_alpha = norm.ppf(1 - alpha/2) if params.two_tailed else norm.ppf(1 - alpha)
    z_beta = norm.ppf(1 - beta)
    num = 2 * (sd ** 2) * (z_alpha + z_beta) ** 2
    den = params.mde ** 2
    return int(np.ceil(num / den))
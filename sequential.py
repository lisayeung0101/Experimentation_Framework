import numpy as np
from scipy.stats import norm

def pocock_boundaries(alpha=0.05, looks=5):
    c = norm.ppf(1 - alpha / (2 * np.log(1 + looks)))
    return [c] * looks

def z_stat_proportions(success_a, total_a, success_b, total_b):
    p_a = success_a / total_a
    p_b = success_b / total_b
    p_pool = (success_a + success_b) / (total_a + total_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/total_a + 1/total_b))
    return (p_b - p_a) / se

def sequential_monitor(stream_a, stream_b, looks=5, alpha=0.05):
    boundaries = pocock_boundaries(alpha, looks)
    decisions = []
    for i in range(1, looks+1):
        sa, ta = stream_a[i-1]
        sb, tb = stream_b[i-1]
        z = abs(z_stat_proportions(sa, ta, sb, tb))
        stop = z > boundaries[i-1]
        decisions.append({"look": i, "z": z, "boundary": boundaries[i-1], "stop": stop})
        if stop: 
            break
    return decisions


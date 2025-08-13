import numpy as np
from scipy.stats import chisquare, ttest_ind

def srm_check(n_a: int, n_b: int, expected_ratios=(0.5, 0.5)):
    total = n_a + n_b
    expected = np.array(expected_ratios) * total
    observed = np.array([n_a, n_b])
    stat, p = chisquare(f_obs=observed, f_exp=expected)
    return {"chi2": stat, "p_value": p, "srm_flag": p < 0.01}

def invariant_ttest(x_a, x_b, alpha=0.01):
    stat, p = ttest_ind(x_a, x_b, equal_var=False)
    return {"t": stat, "p_value": p, "violation": p < alpha}
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple
import numpy as np
from scipy import stats
from .cuped import cuped_adjust

@dataclass 
class ABResult:
    metric: str
    lift: float
    ci_low: float
    ci_high: float
    p_value: float
    n_a: int
    n_b: int

def ab_proportions( success_a: int,
                    total_a: int,
                    success_b: int,
                    total_b: int,
                    alpha: float = 0.05,
                    continuity_correction: bool = False,
                    ) -> ABResult:
    if total_a <= 0 or total_b <= 0:
        raise ValueError("total_a and total_b must be positive.")
    if not (0 <= success_a <= total_a and 0 <= success_b <= total_b):
        raise ValueError("success counts must be between 0 and total.")
    
    p_a = success_a / total_a
    p_b = success_b / total_b
    diff = p_b - p_a

    # Pooled standard error
    p_pool = (success_a + success_b) / (total_a + total_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / total_a + 1 / total_b))

    # Continuity correction (optional)
    cc = (1 / total_a + 1 / total_b) if continuity_correction else 0.0
    adj_diff = diff - np.sign(diff) * cc if continuity_correction and diff != 0 else diff

    z = stats.norm.ppf(1 - alpha / 2)
    ci_low = diff - z * se
    ci_high = diff + z * se

    # Two-sided z-test using pooled SE
    z_stat = adj_diff / se if se > 0 else 0.0
    p_val = 2 * (1 - stats.norm.cdf(abs(z_stat))) if se > 0 else 1.0

    return ABResult(
        metric="proportion",
        lift=diff,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_val,
        n_a=total_a,
        n_b=total_b,
    )


def ab_means(y_a: Sequence[float],
            y_b: Sequence[float],
            alpha: float = 0.05,
            cuped_theta_a: Optional[Sequence[float]] = None,
            cuped_theta_b: Optional[Sequence[float]] = None,
            ) -> ABResult:
    ya = np.asarray(y_a, dtype=float)
    yb = np.asarray(y_b, dtype=float)
    if ya.size < 2 or yb.size < 2:
        raise ValueError("Both groups must have at least 2 observations.")
    # Optional CUPED adjustment
    if cuped_theta_a is not None and cuped_theta_b is not None:
        ya, _ = cuped_adjust(ya, np.asarray(cuped_theta_a, dtype=float))
        yb, _ = cuped_adjust(yb, np.asarray(cuped_theta_b, dtype=float))

    diff = yb.mean() - ya.mean()
    se = np.sqrt(ya.var(ddof=1) / len(ya) + yb.var(ddof=1) / len(yb))

    # Welch's t-test p-value
    t_stat, p_val = stats.ttest_ind(yb, ya, equal_var=False)

    # For CI, use normal approx for simplicity; for rigor, compute Welch CI with df
    z = stats.norm.ppf(1 - alpha / 2)
    ci_low = diff - z * se
    ci_high = diff + z * se

    return ABResult(
        metric="mean",
        lift=diff,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=float(p_val),
        n_a=len(ya),
        n_b=len(yb),
    )

    


import random, numpy as np

"""
GaussianCurriculum
------------------
Schedules which problems the model sees over time, gradually shifting
from easy counting (n -> n+1) to harder addition (a + b).

Key knobs (set in __init__):
- tf_rate: converts step → time_flow in [0,1].
- cf_variance / cx_variance: variance of curriculum mean vs. sampled complexity.
- cf_min_complexity / cf_max_complexity: counting complexity bounds.
- addition_cf_min / addition_cf_max: addition complexity bounds.
- total_steps, counting_focus_steps, addition_start_step: schedule landmarks.
- counting_fade_rate: how fast counting weight decays after addition begins.

Core methods:
- get_time_flow(step): smooth progress scalar in [0,1].
- get_complexity_flow_mean(time_flow, task): maps time_flow → target complexity mean.
- sample_complexity(mean): draws a noisy complexity value, clamped to [2,12].
- get_task_weights(step): mixes counting vs. addition as training advances.
- select_problem(step, counting_problems, addition_problems):
  picks a problem consistent with the current weights and complexities.

Rationale:
A noisy (Gaussian) curriculum prevents brittleness: the model sees a spread
of difficulties around a moving mean, rather than a rigid staircase.
"""

class GaussianCurriculum:
    def __init__(self):
        self.tf_rate = 0.0001
        self.cf_variance = 1.5
        self.cf_min_complexity = 2.0
        self.cf_max_complexity = 6.0   # counting target max (n->n+1 gives ≤6)
        self.cx_variance = 0.8
        self.addition_cf_min = 2.0
        self.addition_cf_max = 10.0

        self.total_steps = 50000
        self.counting_focus_steps = 6000
        self.addition_start_step = 12000
        self.counting_fade_rate = 0.0001

    """
    get_time_flow(step) -> float in [0,1]
    Converts global step to a smooth progress scalar using tf_rate.
    The value saturates at 1.0, providing a bounded timeline for the curriculum.
    """

    def get_time_flow(self, step):
        return min(1.0, step * self.tf_rate)

    def get_complexity_flow_mean(self, time_flow, task_type="counting"):
        if task_type == "counting":
            lo, hi = self.cf_min_complexity, self.cf_max_complexity
        else:
            lo, hi = self.addition_cf_min, self.addition_cf_max
        return lo + time_flow * (hi - lo)

    def sample_complexity(self, mean):
        c = np.random.normal(mean, self.cx_variance)
        return max(2, min(12, c))

    """
    get_task_weights(step) -> (counting_w, addition_w)
    Returns the current mixture weights for sampling counting vs. addition tasks.

    Logic:
    - Before addition_start_step: (1.0, 0.0)  → pure counting.
    - After addition_start_step:
        * addition_w grows linearly toward 1.0 with training progress.
        * counting_w decays from 1.0 at a rate set by counting_fade_rate,
          but never below a floor (0.2) so counting does not vanish entirely.

    Purpose:
    Keeps counting practice alive while steadily introducing addition,
    mirroring a gentle, mixed practice schedule.
    """

    def get_task_weights(self, step):
        counting_w, addition_w = 1.0, 0.0
        if step >= self.addition_start_step:
            addition_progress = (step - self.addition_start_step) / max(1, (self.total_steps - self.addition_start_step))
            addition_w = min(1.0, addition_progress * 2)
            fade = (step - self.addition_start_step) * self.counting_fade_rate
            counting_w = max(0.2, 1.0 - fade)
        return counting_w, addition_w

    """
    select_problem(step, counting_problems, addition_problems)
    → (problem, time_flow, cf_mean, cx, weight)
    Chooses a single training example consistent with the current curriculum.

    Steps:
    1) Compute time_flow = get_time_flow(step).
    2) Map time_flow to target complexity means:
         c_count_mean = get_complexity_flow_mean(time_flow, "counting")
         c_add_mean   = get_complexity_flow_mean(time_flow, "addition")
    3) Get sampling weights: (w_count, w_add) = get_task_weights(step).
    4) Sample concrete complexities around the means:
         c_count = sample_complexity(c_count_mean)
         c_add   = sample_complexity(c_add_mean)
    5) Weighted pick:
         - With probability w_count/(w_count+w_add), pick a counting problem
           whose stored complexity is within ±1.0 of c_count (fallback to any).
         - Otherwise, pick an addition problem within ±1.0 of c_add (fallback to any).

    Returns:
      - problem: a tuple from the provided pools (e.g., (details, target, complexity))
      - time_flow: scalar in [0,1] for logging/analysis
      - cf_mean: the *mean* complexity used for the chosen task type
      - cx: the *sampled* complexity used for filtering (±1.0 band)
      - weight: the task weight that “won” this draw (w_count or w_add)

    Notes:
    - If both weights are zero (degenerate), returns a null selection.
    - The ±1.0 complexity band adds stochasticity but keeps examples
      near the current target difficulty.
    """

    def select_problem(self, step, counting_problems, addition_problems):
        t = self.get_time_flow(step)
        c_count_mean = self.get_complexity_flow_mean(t, "counting")
        c_add_mean   = self.get_complexity_flow_mean(t, "addition")
        w_count, w_add = self.get_task_weights(step)

        c_count = self.sample_complexity(c_count_mean)
        c_add   = self.sample_complexity(c_add_mean)

        total = w_count + w_add
        if total <= 0:
            return None, 0, 0, 0, 0

        if random.random() < (w_count / total):
            # choose counting
            valid = [p for p in counting_problems if abs(p[2] - c_count) <= 1.0] or counting_problems
            problem = random.choice(valid)
            return problem, t, c_count_mean, c_count, w_count
        else:
            valid = [p for p in addition_problems if abs(p[2] - c_add) <= 1.0] or addition_problems
            problem = random.choice(valid)
            return problem, t, c_add_mean, c_add, w_add

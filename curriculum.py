import random, numpy as np

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

    def get_task_weights(self, step):
        counting_w, addition_w = 1.0, 0.0
        if step >= self.addition_start_step:
            addition_progress = (step - self.addition_start_step) / max(1, (self.total_steps - self.addition_start_step))
            addition_w = min(1.0, addition_progress * 2)
            fade = (step - self.addition_start_step) * self.counting_fade_rate
            counting_w = max(0.2, 1.0 - fade)
        return counting_w, addition_w

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

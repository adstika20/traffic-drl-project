import numpy as np
import gymnasium as gym
from gymnasium import spaces


class EmpiricalDemandGenerator:
    """Bootstrap demand dari kolom traffic_volume; fallback values dapat diberikan."""
    def __init__(self, csv_path=None, demand_values=None, profile_length=180,
                 lambda_min=0.5, lambda_max=4.0, seed=None):
        if demand_values is not None:
            values = np.asarray(demand_values, dtype=float)
        elif csv_path is not None:
            import pandas as pd
            values = pd.read_csv(csv_path)["traffic_volume"].dropna().to_numpy(dtype=float)
        else:
            raise ValueError("Berikan csv_path atau demand_values.")
        values = values[np.isfinite(values)]
        if len(values) == 0:
            raise ValueError("traffic_volume tidak memiliki nilai valid.")
        self.values = values
        self.profile_length = int(profile_length)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.rng = np.random.default_rng(seed)
        self.profile = None
        self.index = 0

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        sampled = self.rng.choice(self.values, size=self.profile_length, replace=True)
        vmin, vmax = 50.0, 998.0
        norm = np.clip((sampled - vmin) / (vmax - vmin), 0.0, 1.0)
        self.profile = self.lambda_min + norm * (self.lambda_max - self.lambda_min)
        self.index = 0

    def sample_arrivals(self):
        if self.profile is None:
            self.reset()
        lam = float(self.profile[min(self.index, self.profile_length - 1)])
        self.index += 1
        total = int(self.rng.poisson(lam))
        ns = int(self.rng.binomial(total, 0.5))
        return ns, total - ns, total


class SingleIntersectionTrafficEnv(gym.Env):
    """Environment antrean abstrak, bukan simulator mikroskopik lalu lintas."""
    metadata = {"render_modes": []}

    def __init__(self, demand_generator, episode_steps=180, step_seconds=10,
                 max_departures=5, clearance_seconds=3, min_green_seconds=30,
                 queue_norm=50, seed=None):
        super().__init__()
        self.demand_generator = demand_generator
        self.episode_steps = int(episode_steps)
        self.step_seconds = int(step_seconds)
        self.max_departures = int(max_departures)
        self.clearance_seconds = int(clearance_seconds)
        self.min_green_seconds = int(min_green_seconds)
        self.queue_norm = float(queue_norm)
        self.rng = np.random.default_rng(seed)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(5,), dtype=np.float32)
        self.reset(seed=seed)

    def _obs(self):
        return np.array([
            min(self.queue_NS / self.queue_norm, 1.0),
            min(self.queue_EW / self.queue_norm, 1.0),
            float(self.current_phase),
            min(self.demand_norm, 1.0),
            min(self.green_elapsed_seconds / self.min_green_seconds, 1.0)
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.queue_NS = 0
        self.queue_EW = 0
        self.current_phase = 0
        self.green_elapsed_seconds = 0
        self.step_count = 0
        self.total_arrivals = 0
        self.total_departures = 0
        self.phase_switches = 0
        self.blocked_switch_requests = 0
        self.queue_sum = 0.0
        self.cumulative_reward = 0.0
        self.demand_generator.reset(seed=seed)
        self.demand_norm = 0.0
        return self._obs(), {}

    def step(self, action):
        action = int(action)
        if action not in (0, 1):
            raise ValueError("Action harus 0 atau 1.")
        switch_blocked = False
        switched_this_step = False
        if action != self.current_phase:
            if self.green_elapsed_seconds >= self.min_green_seconds:
                self.current_phase = action
                self.green_elapsed_seconds = 0
                self.phase_switches += 1
                switched_this_step = True
            else:
                switch_blocked = True
                self.blocked_switch_requests += 1

        arr_ns, arr_ew, total_arr = self.demand_generator.sample_arrivals()
        self.queue_NS += arr_ns
        self.queue_EW += arr_ew
        self.total_arrivals += total_arr

        lam_norm = (self.demand_generator.profile[
            min(self.demand_generator.index - 1, self.demand_generator.profile_length - 1)
        ] - self.demand_generator.lambda_min) / (
            self.demand_generator.lambda_max - self.demand_generator.lambda_min
        )
        self.demand_norm = float(np.clip(lam_norm, 0, 1))

        effective_green = self.step_seconds
        if switched_this_step:
            effective_green = max(0, self.step_seconds - self.clearance_seconds)
        capacity = int(np.floor(self.max_departures * effective_green / self.step_seconds))
        if self.current_phase == 0:
            departures = min(self.queue_NS, capacity)
            self.queue_NS -= departures
        else:
            departures = min(self.queue_EW, capacity)
            self.queue_EW -= departures

        self.total_departures += departures
        self.green_elapsed_seconds += self.step_seconds
        self.step_count += 1

        total_queue = self.queue_NS + self.queue_EW
        reward = -total_queue / 10.0
        self.queue_sum += total_queue
        self.cumulative_reward += reward

        terminated = self.step_count >= self.episode_steps
        truncated = False
        metrics = {
            "avg_queue": self.queue_sum / self.step_count,
            "throughput": self.total_departures,
            "total_arrivals": self.total_arrivals,
            "final_queue": total_queue,
            "cumulative_reward": self.cumulative_reward,
            "phase_switches": self.phase_switches,
            "blocked_switch_requests": self.blocked_switch_requests
        }
        info = {
            "queue_NS": self.queue_NS,
            "queue_EW": self.queue_EW,
            "current_phase": self.current_phase,
            "arrivals": total_arr,
            "departures": departures,
            "switch_blocked": switch_blocked,
            "episode_metrics": metrics
        }
        return self._obs(), float(reward), terminated, truncated, info

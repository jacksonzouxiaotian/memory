import numpy as np

from rl.memory_regularized_policy import FMRSRulePolicy, RiskOnlyPolicy
from rl.narrow_passage_env import NarrowPassageMemoryEnv


def test_narrow_passage_env_smoke():
    env = NarrowPassageMemoryEnv(seed=1, max_steps=4)
    obs, info = env.reset()
    assert obs.shape == (10,)
    assert "scenario" in info
    next_obs, reward, terminated, truncated, step_info = env.step(0)
    assert next_obs.shape == (10,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "success" in step_info


def test_memory_policy_rejects_high_memory_low_risk_case():
    obs = np.array([
        0.95, 0.35, 0.5, 0.2, 0.8,
        0.9, 0.0, 0.0, 0.1, 0.1,
    ])
    memory_action, _ = FMRSRulePolicy().act(obs)
    risk_action, _ = RiskOnlyPolicy().act(obs)
    assert memory_action == 3
    assert risk_action != 3

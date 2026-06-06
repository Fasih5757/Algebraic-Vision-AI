"""
rl_agent.py
-----------
Reinforcement Learning agent that learns which mathematical patterns
are most relevant to highlight for a given image or expression.

Architecture:
  - State  : feature vector (symmetry score, edge density, color entropy,
             aspect ratio, brightness, detected object count)
  - Actions: 6 pattern overlays to emphasize
             [fibonacci, golden_ratio, symmetry_axes, fractal,
              radial_symmetry, wave_pattern]
  - Reward : user feedback signal (thumbs up/down) + auto-heuristic reward
  - Policy : Epsilon-greedy Q-table (tabular RL, no GPU needed)

The Q-table is persisted to disk so the agent improves across sessions.
"""

import numpy as np
import json
import os
import cv2
from pathlib import Path

# ── Constants ─────────────────────────────────────────────────────────────────
ACTIONS = [
    "fibonacci_spiral",
    "golden_ratio_overlay",
    "symmetry_axes",
    "fractal_highlight",
    "radial_symmetry",
    "wave_pattern",
]

N_ACTIONS   = len(ACTIONS)
N_BINS      = 5          # discretization bins per feature
N_FEATURES  = 6
Q_TABLE_PATH = Path(__file__).parent.parent / "rl_qtable.json"

ALPHA   = 0.15   # learning rate
GAMMA   = 0.90   # discount factor
EPSILON = 0.20   # exploration rate (decays over time)
MIN_EPS = 0.05


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(image_path: str) -> np.ndarray:
    """
    Returns a 6-element float array:
      [symmetry_score, edge_density, color_entropy,
       aspect_ratio_diff, mean_brightness, object_count_proxy]
    All values normalized to [0, 1].
    """
    img  = cv2.imread(image_path)
    if img is None:
        return np.zeros(N_FEATURES)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Symmetry score
    mid = w // 2
    left  = gray[:, :mid]
    right = cv2.resize(cv2.flip(gray[:, mid:], 1), (left.shape[1], left.shape[0]))
    sym_score = 1.0 - (np.mean(cv2.absdiff(left, right)) / 255.0)

    # 2. Edge density
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (h * w)

    # 3. Color entropy (HSV saturation histogram entropy)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [1], None, [32], [0, 256]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    entropy = -np.sum(hist * np.log2(hist + 1e-9)) / np.log2(32)

    # 4. Aspect ratio difference from golden ratio
    PHI = 1.6180339887
    ar  = w / max(h, 1)
    ar_diff = 1.0 - min(abs(ar - PHI) / PHI, 1.0)

    # 5. Mean brightness
    brightness = np.mean(gray) / 255.0

    # 6. Object count proxy (number of contours above threshold)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    large = [c for c in contours if cv2.contourArea(c) > (h * w * 0.005)]
    obj_proxy = min(len(large) / 20.0, 1.0)

    return np.array([sym_score, edge_density, entropy,
                     ar_diff, brightness, obj_proxy], dtype=np.float32)


def _discretize(features: np.ndarray) -> tuple:
    """Convert continuous features to discrete state tuple."""
    bins = np.linspace(0, 1, N_BINS + 1)[1:-1]  # N_BINS-1 thresholds
    return tuple(int(np.digitize(f, bins)) for f in features)


# ── Q-Table ───────────────────────────────────────────────────────────────────

class QTable:
    def __init__(self):
        self.table: dict = {}
        self.episode: int = 0
        self.load()

    def _key(self, state: tuple) -> str:
        return str(state)

    def get_q(self, state: tuple) -> np.ndarray:
        key = self._key(state)
        if key not in self.table:
            self.table[key] = [0.0] * N_ACTIONS
        return np.array(self.table[key])

    def update(self, state: tuple, action: int, reward: float,
               next_state: tuple):
        q_vals      = self.get_q(state)
        q_next_max  = np.max(self.get_q(next_state))
        td_target   = reward + GAMMA * q_next_max
        td_error    = td_target - q_vals[action]
        q_vals[action] += ALPHA * td_error
        self.table[self._key(state)] = q_vals.tolist()
        self.episode += 1

    def save(self):
        data = {"table": self.table, "episode": self.episode}
        with open(Q_TABLE_PATH, 'w') as f:
            json.dump(data, f)

    def load(self):
        if Q_TABLE_PATH.exists():
            try:
                with open(Q_TABLE_PATH, 'r') as f:
                    data = json.load(f)
                self.table   = data.get("table", {})
                self.episode = data.get("episode", 0)
            except Exception:
                self.table   = {}
                self.episode = 0


# ── Agent ─────────────────────────────────────────────────────────────────────

class PatternRLAgent:
    """
    Epsilon-greedy RL agent that selects which pattern overlays
    to emphasize for a given image.
    """

    def __init__(self):
        self.qtable  = QTable()
        self._last_state  = None
        self._last_action = None

    @property
    def epsilon(self) -> float:
        """Decay epsilon over episodes."""
        decay = max(MIN_EPS, EPSILON * (0.995 ** self.qtable.episode))
        return decay

    def select_action(self, features: np.ndarray) -> tuple[int, str]:
        """
        Returns (action_index, action_name).
        Uses epsilon-greedy policy.
        """
        state = _discretize(features)
        self._last_state = state

        if np.random.rand() < self.epsilon:
            # Explore
            action = np.random.randint(N_ACTIONS)
        else:
            # Exploit
            q_vals = self.qtable.get_q(state)
            action = int(np.argmax(q_vals))

        self._last_action = action
        return action, ACTIONS[action]

    def select_top_k(self, features: np.ndarray, k: int = 3) -> list[str]:
        """
        Returns top-k recommended pattern overlays (sorted by Q-value).
        """
        state  = _discretize(features)
        q_vals = self.qtable.get_q(state)
        top_k  = np.argsort(q_vals)[::-1][:k]
        return [ACTIONS[i] for i in top_k]

    def give_reward(self, reward: float):
        """
        Call this after user feedback.
        reward: +1.0 (positive), -1.0 (negative), 0.0 (neutral)
        """
        if self._last_state is None or self._last_action is None:
            return
        # Next state = same state (single-step episode)
        self.qtable.update(
            self._last_state,
            self._last_action,
            reward,
            self._last_state,
        )
        self.qtable.save()

    def auto_reward(self, features: np.ndarray, action: int) -> float:
        """
        Heuristic reward without user feedback.
        High symmetry → reward symmetry actions.
        High edge density → reward fractal/wave actions.
        High golden ratio alignment → reward golden ratio action.
        """
        sym, edge, entropy, ar_diff, bright, obj = features

        reward_map = {
            0: sym * 0.8 + entropy * 0.2,          # fibonacci
            1: ar_diff * 0.9 + sym * 0.1,           # golden_ratio
            2: sym * 1.0,                            # symmetry_axes
            3: edge * 0.7 + entropy * 0.3,          # fractal
            4: sym * 0.6 + (1 - edge) * 0.4,        # radial_symmetry
            5: edge * 0.5 + entropy * 0.5,          # wave_pattern
        }
        return float(reward_map.get(action, 0.0))

    def learn_from_image(self, image_path: str):
        """
        Auto-learn from an image without user feedback.
        Runs one episode using heuristic rewards.
        """
        features = extract_features(image_path)
        action, name = self.select_action(features)
        reward = self.auto_reward(features, action)
        state  = _discretize(features)
        self.qtable.update(state, action, reward, state)
        self.qtable.save()
        return action, name, reward

    def get_stats(self) -> dict:
        return {
            "total_episodes": self.qtable.episode,
            "epsilon":        round(self.epsilon, 4),
            "states_learned": len(self.qtable.table),
            "actions":        ACTIONS,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────
_agent_instance = None

def get_agent() -> PatternRLAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = PatternRLAgent()
    return _agent_instance

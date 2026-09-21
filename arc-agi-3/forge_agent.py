"""
ARC-AGI-3 Agent — Forge v5 "Pathfinder" (FIXED)
=================================================
Bug fixes applied:
  [C1] GameAction enum mutation — use from_name() every call for fresh ref
  [C2] ACTION6 coordinate consistency — unified (row=y, col=x) throughout
  [C3] _make_action_input always passes data={}
  [C4] ACTION7 never incremented games_done / corrupted pai index
  [M1] TimeBudget.mark_done() now called on WIN
  [M2] solve_level runs in background thread — choose_action never blocks
  [M3] ACTION7 pai index set to safe value (0)
  [M4] _tensor and _frame_to_tensor merged into single _build_tensor()
  [M5] _inject_expert_demos checks set_level exists before calling
  [M6] Expert injection order fixed (archive before PER reset, inject after)
  [M7] is_done guards lf.state with getattr
  [M8] BFS max_states reduced; game clones not stored in queue
  [M9] _heuristic_score direction reversed (fewer colors = closer to solution)
  [m1] Negative hash in seed fixed with abs() + modulo
  [m2] _NAMES dict includes ACTION7 and full coverage
  [m3] scan_available_actions timeout scaled to budget
  [m4] ActionInput always receives data kwarg
"""

from __future__ import annotations

import copy
import glob
import hashlib
import importlib.util
import logging
import math
import os
import random
import threading
import time
import traceback
from collections import deque, defaultdict
from heapq import heappush, heappop
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agents.agent import Agent
from arcengine import FrameData, GameAction, GameState, ActionInput

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

# [m2] Complete action name map including ACTION7
_NAMES: Dict[int, str] = {
    0: 'RESET',
    1: 'ACTION1',
    2: 'ACTION2',
    3: 'ACTION3',
    4: 'ACTION4',
    5: 'ACTION5',
    6: 'ACTION6',
    7: 'ACTION7',
}

# Simple action GameAction list (indices 0-4 map to ACTION1-ACTION5)
_ACTION_MAP = [
    GameAction.ACTION1,
    GameAction.ACTION2,
    GameAction.ACTION3,
    GameAction.ACTION4,
    GameAction.ACTION5,
]

# ─────────────────────────────────────────────────────────────────────────────
# Time Budget Manager
# ─────────────────────────────────────────────────────────────────────────────

class TimeBudget:
    """Hard time-budget: total 8h50m, per-game soft budget."""
    TOTAL_HARD = 8 * 3600 + 50 * 60   # 8h 50min wall-time
    N_GAMES    = 110

    def __init__(self):
        self.start       = time.time()
        self.game_start  = self.start
        self.games_done  = 0
        self._lock       = threading.Lock()

    def new_game(self):
        self.game_start = time.time()

    def mark_done(self):
        # [M1] Called on WIN or game-over so per-game budget stays accurate
        with self._lock:
            self.games_done += 1

    def elapsed(self) -> float:
        return time.time() - self.start

    def remaining_total(self) -> float:
        return max(0.0, self.TOTAL_HARD - self.elapsed())

    def per_game_budget(self) -> float:
        with self._lock:
            remaining_games = max(1, self.N_GAMES - self.games_done)
        return self.remaining_total() / remaining_games

    def game_elapsed(self) -> float:
        return time.time() - self.game_start

    def is_hard_deadline(self) -> bool:
        return self.elapsed() >= self.TOTAL_HARD - 60  # 1-min buffer

    def search_budget(self, fraction: float = 0.50) -> float:
        """Time to give search algorithms (fraction of per-game budget)."""
        return min(self.per_game_budget() * fraction, 180.0)


# Singleton shared across all agent instances
_BUDGET = TimeBudget()


# ─────────────────────────────────────────────────────────────────────────────
# Action helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_action_input(act_id: int, data: Optional[dict]) -> ActionInput:
    """
    [C3][m4] Always pass data={} so ActionInput never receives None for data.
    For ACTION6 the caller must supply {'x': col, 'y': row}.
    """
    safe_data = data if data is not None else {}
    if act_id == 6:
        return ActionInput(id=GameAction.ACTION6, data=safe_data)
    name = _NAMES.get(act_id, 'ACTION1')
    # [C1] Use from_name to get a fresh reference — never cache enum members
    return ActionInput(id=GameAction.from_name(name), data=safe_data)


def _fresh_action(name: str) -> GameAction:
    """
    [C1] Always return a fresh GameAction reference via from_name so that
    setting .reasoning on it does not mutate a shared enum member.
    """
    return GameAction.from_name(name)


# ─────────────────────────────────────────────────────────────────────────────
# SumTree & Prioritised Experience Replay
# ─────────────────────────────────────────────────────────────────────────────

class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity, dtype=np.float64)
        self.data: List[Optional[dict]] = [None] * capacity
        self.ptr      = 0
        self.size     = 0

    def _propagate(self, idx: int, delta: float):
        while idx > 0:
            idx = (idx - 1) >> 1
            self.tree[idx] += delta

    def update(self, idx: int, priority: float):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def add(self, priority: float, data: dict):
        leaf = self.ptr + self.capacity - 1
        self.data[self.ptr] = data
        self.update(leaf, priority)
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def get(self, s: float) -> Tuple[int, float, Optional[dict]]:
        idx = 0
        while idx < self.capacity - 1:
            left, right = 2 * idx + 1, 2 * idx + 2
            if s <= self.tree[left]:
                idx = left
            else:
                s  -= self.tree[left]
                idx = right
        data_idx = idx - (self.capacity - 1)
        return idx, self.tree[idx], self.data[data_idx]

    @property
    def total(self) -> float:
        return float(self.tree[0])

    @property
    def min_priority(self) -> float:
        if self.size == 0:
            return 1.0
        leaves = self.tree[self.capacity - 1: self.capacity - 1 + self.size]
        pos    = leaves[leaves > 0]
        return float(pos.min()) if len(pos) > 0 else 1.0

    def __len__(self):
        return self.size


class PERBuffer:
    def __init__(self, capacity=40_000, alpha=0.6, beta_start=0.4,
                 beta_end=1.0, beta_steps=40_000):
        self.tree      = SumTree(capacity)
        self.alpha     = alpha
        self.beta      = beta_start
        self.beta_end  = beta_end
        self.beta_inc  = (beta_end - beta_start) / beta_steps
        self.max_p     = 1.0
        self.seen: set = set()
        self.eps       = 1e-6

    def _key(self, exp: dict) -> str:
        return hashlib.md5(
            exp['s'].tobytes() + str(exp['a']).encode()
        ).hexdigest()[:16]

    def add(self, exp: dict, priority: Optional[float] = None):
        key = self._key(exp)
        if key in self.seen:
            return
        self.seen.add(key)
        p = ((priority if priority is not None else self.max_p) + self.eps) ** self.alpha
        self.tree.add(p, exp)

    def add_expert(self, exp: dict):
        self.add(exp, priority=self.max_p * 5.0)

    def sample(self, batch_size: int):
        if len(self.tree) < batch_size:
            return None, None, None
        batch, idxs, weights = [], [], []
        seg   = self.tree.total / batch_size
        min_p = self.tree.min_priority + self.eps
        self.beta = min(self.beta_end, self.beta + self.beta_inc)
        for i in range(batch_size):
            s            = random.uniform(seg * i, seg * (i + 1))
            idx, p, data = self.tree.get(s)
            if data is None:
                continue
            w     = ((len(self.tree) * (p + self.eps)) ** (-self.beta))
            w_max = ((len(self.tree) * min_p) ** (-self.beta))
            weights.append(w / (w_max + self.eps))
            idxs.append(idx)
            batch.append(data)
        if not batch:
            return None, None, None
        return batch, idxs, np.array(weights, dtype=np.float32)

    def update_priorities(self, idxs: list, td_errors: np.ndarray):
        for idx, err in zip(idxs, td_errors):
            p = (abs(float(err)) + self.eps) ** self.alpha
            self.max_p = max(self.max_p, abs(float(err)) + self.eps)
            self.tree.update(idx, p)

    def __len__(self):
        return len(self.tree)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-level Associative Episodic Memory (CLEAR)
# ─────────────────────────────────────────────────────────────────────────────

class PersistentAEM:
    def __init__(self, capacity=512):
        self.diffs   = deque(maxlen=capacity)
        self.actions = deque(maxlen=capacity)
        self.rewards = deque(maxlen=capacity)

    def push(self, diff: np.ndarray, action: int, reward: float):
        self.diffs.append(diff)
        self.actions.append(action)
        self.rewards.append(reward)

    def get_tensors(self, device, max_mem: int = 64):
        n = min(len(self.diffs), max_mem)
        if n < 2:
            return None, None, None
        diffs = torch.zeros(1, n, 1, 64, 64, device=device)
        acts  = torch.zeros(1, n, dtype=torch.long, device=device)
        rews  = torch.zeros(1, n, device=device)
        for i, (d, a, r) in enumerate(
                list(zip(self.diffs, self.actions, self.rewards))[-n:]):
            diffs[0, i, 0] = torch.from_numpy(d.astype(np.float32))
            acts[0, i]     = min(a, 4)
            rews[0, i]     = r
        return diffs, acts, rews


# ─────────────────────────────────────────────────────────────────────────────
# Neural Network (ForgeNet)
# ─────────────────────────────────────────────────────────────────────────────

class CBAM(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        mid      = max(ch // r, 4)
        self.fc1 = nn.Linear(ch, mid)
        self.fc2 = nn.Linear(mid, ch)
        self.sp  = nn.Conv2d(2, 1, 7, padding=3)

    def forward(self, x):
        B, C, H, W = x.shape
        w = torch.sigmoid(self.fc2(F.relu(self.fc1(x.mean(dim=[2, 3])))))
        x = x * w.view(B, C, 1, 1)
        a = torch.sigmoid(self.sp(torch.cat(
            [x.max(1, keepdim=True)[0], x.mean(1, keepdim=True)], 1)))
        return x * a


class AEA(nn.Module):
    def __init__(self, feat_dim=64, mem_dim=32, n_actions=5):
        super().__init__()
        self.mem_dim  = mem_dim
        self.diff_enc = nn.Sequential(
            nn.Conv2d(1, 8, 8, stride=8), nn.ReLU(),
            nn.Conv2d(8, 16, 4, stride=4), nn.ReLU(),
            nn.Flatten(), nn.Linear(16 * 2 * 2, mem_dim))
        self.q_proj = nn.Linear(feat_dim, mem_dim)
        self.v_proj = nn.Linear(mem_dim + 1 + n_actions, n_actions)
        self.scale  = mem_dim ** 0.5

    def forward(self, feat, diffs, actions, rewards):
        B, M = actions.shape
        if M == 0:
            return torch.zeros(B, 5, device=feat.device)
        keys  = self.diff_enc(diffs.reshape(B * M, 1, 64, 64)).reshape(B, M, self.mem_dim)
        q     = self.q_proj(feat).unsqueeze(1)
        attn  = F.softmax(torch.bmm(q, keys.transpose(1, 2)) / self.scale, dim=-1)
        ao    = F.one_hot(actions.clamp(0, 4), 5).float()
        vals  = torch.cat([keys, rewards.unsqueeze(-1), ao], dim=-1)
        ctx   = torch.bmm(attn, vals).squeeze(1)
        return self.v_proj(ctx)


class ForgeNet(nn.Module):
    """Input: 26ch  |  Output: 5 action logits + 64*64 click logits = 4101."""
    def __init__(self, in_ch=26, g=64):
        super().__init__()
        self.g    = g
        self.c1   = nn.Conv2d(in_ch, 32,  3, padding=1)
        self.c2   = nn.Conv2d(32,    64,  3, padding=1)
        self.c3   = nn.Conv2d(64,    128, 3, padding=1)
        self.c4   = nn.Conv2d(128,   256, 3, padding=1)
        self.attn = CBAM(256)
        self.ar   = nn.Conv2d(256, 64, 1)
        self.ap   = nn.MaxPool2d(4, 4)
        self.af   = nn.Linear(64 * 16 * 16, 256)
        self.ah   = nn.Linear(256, 5)
        self.dr   = nn.Dropout(0.15)
        self.cc1  = nn.Conv2d(256, 128, 3, padding=1)
        self.cc2  = nn.Conv2d(128,  64, 3, padding=1)
        self.cc3  = nn.Conv2d(64,   32, 1)
        self.cc4  = nn.Conv2d(32,    1, 1)
        self.gp   = nn.AdaptiveAvgPool2d(1)
        self.gf   = nn.Linear(256, 64)
        self.aea  = AEA(feat_dim=64, mem_dim=32, n_actions=5)

    def forward(self, x, mem_diffs=None, mem_actions=None, mem_rewards=None):
        x  = F.relu(self.c1(x))
        x  = F.relu(self.c2(x))
        x  = F.relu(self.c3(x))
        f  = F.relu(self.c4(x))
        f  = self.attn(f)
        af = F.relu(self.ar(f))
        af = self.ap(af).reshape(f.size(0), -1)
        al = self.ah(self.dr(F.relu(self.af(af))))
        cf = F.relu(self.cc1(f))
        cf = F.relu(self.cc2(cf))
        cf = F.relu(self.cc3(cf))
        cl = self.cc4(cf).reshape(f.size(0), -1)   # (B, 4096)
        if mem_diffs is not None and mem_actions is not None:
            gf = self.gf(self.gp(f).reshape(f.size(0), -1))
            al = al + self.aea(gf, mem_diffs, mem_actions, mem_rewards)
        return torch.cat([al, cl], 1)   # (B, 4101)


# ─────────────────────────────────────────────────────────────────────────────
# Frame / object utilities
# ─────────────────────────────────────────────────────────────────────────────

def _frame_hash(frame: np.ndarray, extras: Optional[dict] = None) -> str:
    h = hashlib.md5(frame.tobytes()).hexdigest()[:16]
    if extras:
        h += "|" + "|".join(f"{k}={v}" for k, v in sorted(extras.items()))
    return h


def _bg_color(frame: np.ndarray) -> int:
    return int(np.bincount(frame.flatten(), minlength=16).argmax())


def _get_objects(frame: np.ndarray, bg: int) -> List[dict]:
    objs = []
    for c in range(16):
        if c == bg:
            continue
        m = frame == c
        n = int(m.sum())
        if 2 <= n <= 3000:
            ys, xs = np.where(m)
            objs.append({
                'c': c,
                'cx': float(xs.mean()), 'cy': float(ys.mean()),
                'n': n, 'xs': xs.tolist(), 'ys': ys.tolist(),
            })
    return objs


def _heuristic_score(frame: np.ndarray, bg: int,
                     initial_frame: Optional[np.ndarray] = None) -> float:
    """
    [M9] Fixed direction: reward states that have changed from the initial
    frame (progress) rather than states with more colors (which perversely
    drove search away from solutions that merge/reduce colors).
    """
    cnt = np.bincount(frame.flatten(), minlength=16)
    cov = 1.0 - cnt[bg] / max(frame.size, 1)
    if initial_frame is not None:
        change = float(np.mean(frame != initial_frame))
        return change * 0.7 + cov * 0.3
    return cov


# ─────────────────────────────────────────────────────────────────────────────
# Action Scanner
# ─────────────────────────────────────────────────────────────────────────────

def scan_available_actions(
        game, f0: np.ndarray, bg: int,
        timeout: float = 6.0) -> List[Tuple[int, Optional[dict]]]:
    """
    Discover which (act_id, data) pairs actually change the frame.
    For ACTION6 probes object centroids + a spatial grid.
    [m3] Timeout increased to 6s default; caller may scale further.
    """
    avail   = getattr(game, '_available_actions', list(range(1, 8)))
    actions: List[Tuple[int, Optional[dict]]] = []

    # Simple actions 1-5, 7
    for a in sorted(set(avail) & {1, 2, 3, 4, 5, 7}):
        g = copy.deepcopy(game)
        try:
            r = g.perform_action(_make_action_input(a, None), raw=True)
            if r.frame and np.any(f0 != np.array(r.frame[-1])):
                actions.append((a, None))
        except Exception:
            pass

    # ACTION6 click probing
    if 6 in avail:
        t0   = time.time()
        seen: Set[str] = set()

        # Build candidate click points
        non_bg = np.argwhere(f0 != bg)
        cands: List[Tuple[int, int]] = []

        # Object centroids + bounding corners
        for obj in _get_objects(f0, bg):
            cands.append((int(round(obj['cx'])), int(round(obj['cy']))))
            xs, ys = obj['xs'], obj['ys']
            for cx, cy in [(min(xs), min(ys)), (max(xs), max(ys)),
                           (min(xs), max(ys)), (max(xs), min(ys))]:
                cands.append((cx, cy))

        # Dense non-bg sample
        if len(non_bg) > 0:
            step = max(1, len(non_bg) // 300)
            for pt in non_bg[::step]:
                cands.append((int(pt[1]), int(pt[0])))

        # Coarse grid backstop
        for y in range(0, 64, 6):
            for x in range(0, 64, 6):
                cands.append((x, y))

        for x, y in cands:
            if time.time() - t0 > timeout:
                break
            x, y = int(np.clip(x, 0, 63)), int(np.clip(y, 0, 63))
            g = copy.deepcopy(game)
            try:
                # [C2] data uses {'x': col, 'y': row} consistently
                data = {'x': x, 'y': y}
                r    = g.perform_action(_make_action_input(6, data), raw=True)
                if r.frame:
                    fh = hashlib.md5(np.array(r.frame[-1]).tobytes()).hexdigest()[:12]
                    if fh not in seen and np.any(f0 != np.array(r.frame[-1])):
                        seen.add(fh)
                        actions.append((6, {'x': x, 'y': y}))
            except Exception:
                pass

    return actions


# ─────────────────────────────────────────────────────────────────────────────
# Hidden state probing utilities
# ─────────────────────────────────────────────────────────────────────────────

def _read_hidden(game, fields: list) -> Optional[dict]:
    if not fields:
        return None
    return {f: getattr(game, f, None) for f in fields
            if getattr(game, f, None) is not None}


def _probe_hidden_fields(game, actions: List[Tuple[int, Optional[dict]]]) -> list:
    init     = {k: v for k, v in game.__dict__.items()
                if isinstance(v, (int, float, bool)) and not k.startswith('__')}
    changing = set()
    for act_id, data in actions[:8]:
        g = copy.deepcopy(game)
        try:
            g.perform_action(_make_action_input(act_id, data), raw=True)
        except Exception:
            continue
        for k, v in g.__dict__.items():
            if (k in init and isinstance(v, (int, float, bool)) and v != init[k]
                    and k not in ('_action_count', '_full_reset', '_action_complete')):
                changing.add(k)
    return sorted(k for k in changing
                  if not k.startswith('_') or k in ('_current_level_index', '_score'))


# ─────────────────────────────────────────────────────────────────────────────
# Beam Search
# ─────────────────────────────────────────────────────────────────────────────

def beam_search(game_factory, level_idx: int,
                actions: List[Tuple[int, Optional[dict]]],
                deadline: float,
                beam_width: int = 64,
                max_depth: int = 40,
                initial_frame: Optional[np.ndarray] = None
                ) -> Optional[List[Tuple[int, Optional[dict]]]]:
    try:
        g0 = game_factory()
        g0.set_level(level_idx)
        g0.perform_action(_make_action_input(0, None), raw=True)
        r0 = g0.perform_action(_make_action_input(0, None), raw=True)
        if not r0.frame:
            return None
        f0  = np.array(r0.frame[-1])
        bg  = _bg_color(f0)
        h0  = _frame_hash(f0)
        lvl = getattr(r0, 'levels_completed', 0) or 0
        i_f = initial_frame if initial_frame is not None else f0.copy()

        beam = [(-_heuristic_score(f0, bg, i_f), copy.deepcopy(g0), [], h0)]
        visited: Set[str] = {h0}

        for depth in range(max_depth):
            if time.time() > deadline:
                break
            candidates = []
            for _, gs, hist, _ in beam:
                for act_id, data in actions:
                    g2 = copy.deepcopy(gs)
                    try:
                        r = g2.perform_action(_make_action_input(act_id, data), raw=True)
                    except Exception:
                        continue
                    if not r.frame:
                        continue
                    f2 = np.array(r.frame[-1])
                    hk = _frame_hash(f2)
                    if hk in visited:
                        continue
                    visited.add(hk)
                    nh  = hist + [(act_id, data)]
                    nlv = getattr(r, 'levels_completed', 0) or 0
                    if nlv > lvl or getattr(g2, '_current_level_index', level_idx) > level_idx:
                        logger.info(f"Beam L{level_idx}: {len(nh)} steps depth={depth+1}")
                        return nh
                    sc = -_heuristic_score(f2, bg, i_f)
                    candidates.append((sc, g2, nh, hk))

            if not candidates:
                break
            candidates.sort(key=lambda x: x[0])
            beam = candidates[:beam_width]

        return None
    except Exception as e:
        logger.warning(f"Beam search error L{level_idx}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# IDA* Search
# ─────────────────────────────────────────────────────────────────────────────

def ida_star(game_factory, level_idx: int,
             actions: List[Tuple[int, Optional[dict]]],
             deadline: float, max_depth: int = 30,
             initial_frame: Optional[np.ndarray] = None
             ) -> Optional[List[Tuple[int, Optional[dict]]]]:
    try:
        g0 = game_factory()
        g0.set_level(level_idx)
        g0.perform_action(_make_action_input(0, None), raw=True)
        r0 = g0.perform_action(_make_action_input(0, None), raw=True)
        if not r0.frame:
            return None
        f0  = np.array(r0.frame[-1])
        bg  = _bg_color(f0)
        lvl = getattr(r0, 'levels_completed', 0) or 0
        i_f = initial_frame if initial_frame is not None else f0.copy()

        def h(frame):
            # [M9] Lower score = farther from goal; invert for IDA* cost
            return max(0.0, 1.0 - _heuristic_score(frame, bg, i_f))

        result    = [None]
        threshold = h(f0)

        def dfs(g, frame, hist, g_cost, threshold, visited: set):
            if time.time() > deadline:
                return float('inf')
            f_cost = g_cost + h(frame)
            if f_cost > threshold:
                return f_cost
            if len(hist) > max_depth:
                return float('inf')
            min_t = float('inf')
            for act_id, data in actions:
                g2 = copy.deepcopy(g)
                try:
                    r = g2.perform_action(_make_action_input(act_id, data), raw=True)
                except Exception:
                    continue
                if not r.frame:
                    continue
                f2   = np.array(r.frame[-1])
                hk2  = _frame_hash(f2)
                if hk2 in visited:
                    continue
                nlv  = getattr(r, 'levels_completed', 0) or 0
                nh   = hist + [(act_id, data)]
                if nlv > lvl or getattr(g2, '_current_level_index', level_idx) > level_idx:
                    result[0] = nh
                    return -1
                visited.add(hk2)
                t = dfs(g2, f2, nh, g_cost + 1, threshold, visited)
                visited.discard(hk2)
                if t == -1:
                    return -1
                min_t = min(min_t, t)
            return min_t

        while threshold <= max_depth and time.time() < deadline:
            visited = {_frame_hash(f0)}
            t = dfs(copy.deepcopy(g0), f0, [], 0, threshold, visited)
            if t == -1:
                logger.info(f"IDA* L{level_idx}: {len(result[0])} steps")
                return result[0]
            if t == float('inf'):
                break
            threshold = t

        return None
    except Exception as e:
        logger.warning(f"IDA* error L{level_idx}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Bounded BFS
# [M8] No longer stores game clones in the queue.
#      Instead stores (frame_hash, history) and reconstructs via replay.
#      max_states reduced from 80k to 15k to stay within memory.
# ─────────────────────────────────────────────────────────────────────────────

def bounded_bfs(game_factory, level_idx: int,
                actions: List[Tuple[int, Optional[dict]]],
                deadline: float,
                max_depth: int = 20,
                max_states: int = 15_000,
                initial_frame: Optional[np.ndarray] = None
                ) -> Optional[List[Tuple[int, Optional[dict]]]]:
    try:
        g0 = game_factory()
        g0.set_level(level_idx)
        g0.perform_action(_make_action_input(0, None), raw=True)
        r0 = g0.perform_action(_make_action_input(0, None), raw=True)
        if not r0.frame:
            return None
        f0  = np.array(r0.frame[-1])
        bg  = _bg_color(f0)
        lvl = getattr(r0, 'levels_completed', 0) or 0
        i_f = initial_frame if initial_frame is not None else f0.copy()
        hidden_fields = _probe_hidden_fields(g0, actions[:6])

        def replay(hist):
            """Reconstruct game state by replaying actions from root."""
            g = game_factory()
            g.set_level(level_idx)
            g.perform_action(_make_action_input(0, None), raw=True)
            g.perform_action(_make_action_input(0, None), raw=True)
            for aid, d in hist:
                r = g.perform_action(_make_action_input(aid, d), raw=True)
            return g, np.array(r.frame[-1]) if r.frame else f0

        h0 = _frame_hash(f0, _read_hidden(g0, hidden_fields))
        # Queue stores (history, depth) — no game clones
        queue:   deque             = deque()
        visited: Dict[str, List]   = {h0: []}
        queue.append(([], 0))
        explored = 0

        while queue and time.time() < deadline and explored < max_states:
            hist, depth = queue.popleft()
            if depth >= max_depth:
                continue

            # Reconstruct state for this history
            try:
                g_cur, f_cur = replay(hist)
            except Exception:
                continue

            for act_id, data in actions:
                g2 = copy.deepcopy(g_cur)
                try:
                    r = g2.perform_action(_make_action_input(act_id, data), raw=True)
                except Exception:
                    continue
                explored += 1
                if not r.frame:
                    continue
                f2  = np.array(r.frame[-1])
                hk  = _frame_hash(f2, _read_hidden(g2, hidden_fields))
                if hk in visited:
                    continue
                nh  = hist + [(act_id, data)]
                nlv = getattr(r, 'levels_completed', 0) or 0
                if nlv > lvl or getattr(g2, '_current_level_index', level_idx) > level_idx:
                    logger.info(f"BFS L{level_idx}: {len(nh)} steps, {explored} exp")
                    return nh
                visited[hk] = nh
                queue.append((nh, depth + 1))

        return None
    except Exception as e:
        logger.warning(f"BFS error L{level_idx}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# A* Search
# ─────────────────────────────────────────────────────────────────────────────

def astar_search(game_factory, level_idx: int,
                 actions: List[Tuple[int, Optional[dict]]],
                 deadline: float,
                 max_states: int = 50_000,
                 initial_frame: Optional[np.ndarray] = None
                 ) -> Optional[List]:
    try:
        g0 = game_factory()
        g0.set_level(level_idx)
        g0.perform_action(_make_action_input(0, None), raw=True)
        r0 = g0.perform_action(_make_action_input(0, None), raw=True)
        if not r0.frame:
            return None
        f0  = np.array(r0.frame[-1])
        bg  = _bg_color(f0)
        lvl = getattr(r0, 'levels_completed', 0) or 0
        i_f = initial_frame if initial_frame is not None else f0.copy()
        hidden_fields = _probe_hidden_fields(g0, actions[:6])

        def h(frame):
            return 1.0 - _heuristic_score(frame, bg, i_f)

        h0   = _frame_hash(f0, _read_hidden(g0, hidden_fields))
        best: Dict[str, float] = {h0: 0.0}
        heap = []
        ctr  = 0
        heappush(heap, (h(f0), 0, ctr, copy.deepcopy(g0), [], 0))
        explored = 0

        while heap and time.time() < deadline and explored < max_states:
            _, _, _, gs, hist, depth = heappop(heap)
            explored += 1
            if depth > 35:
                continue
            for act_id, data in actions:
                g2 = copy.deepcopy(gs)
                try:
                    r = g2.perform_action(_make_action_input(act_id, data), raw=True)
                except Exception:
                    continue
                if not r.frame:
                    continue
                f2  = np.array(r.frame[-1])
                gc  = depth + 1
                hk  = _frame_hash(f2, _read_hidden(g2, hidden_fields))
                if hk in best and best[hk] <= gc:
                    continue
                best[hk] = gc
                nh  = hist + [(act_id, data)]
                nlv = getattr(r, 'levels_completed', 0) or 0
                if nlv > lvl or getattr(g2, '_current_level_index', level_idx) > level_idx:
                    logger.info(f"A* L{level_idx}: {len(nh)} steps, {explored} exp")
                    return nh
                ctr += 1
                heappush(heap, (gc + h(f2), gc, ctr, g2, nh, gc))

        return None
    except Exception as e:
        logger.warning(f"A* error L{level_idx}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# MCTS
# ─────────────────────────────────────────────────────────────────────────────

class MCTSNode:
    __slots__ = ('parent', 'action', 'children', 'visits', 'value',
                 'game', 'frame', 'hist', 'is_terminal')
    def __init__(self, parent, action, game, frame, hist):
        self.parent      = parent
        self.action      = action
        self.children: List['MCTSNode'] = []
        self.visits: int  = 0
        self.value: float = 0.0
        self.game         = game
        self.frame        = frame
        self.hist         = hist
        self.is_terminal  = False


def mcts_search(game_factory, level_idx: int,
                actions: List[Tuple[int, Optional[dict]]],
                deadline: float,
                n_rollout: int = 6,
                initial_frame: Optional[np.ndarray] = None
                ) -> Optional[List]:
    try:
        g0 = game_factory()
        g0.set_level(level_idx)
        g0.perform_action(_make_action_input(0, None), raw=True)
        r0 = g0.perform_action(_make_action_input(0, None), raw=True)
        if not r0.frame:
            return None
        f0  = np.array(r0.frame[-1])
        bg  = _bg_color(f0)
        lvl = getattr(r0, 'levels_completed', 0) or 0
        i_f = initial_frame if initial_frame is not None else f0.copy()

        root    = MCTSNode(None, None, copy.deepcopy(g0), f0, [])
        best_r: Optional[List] = None
        C       = math.sqrt(2)

        def ucb1(node: MCTSNode, parent_visits: int) -> float:
            if node.visits == 0:
                return float('inf')
            return (node.value / node.visits +
                    C * math.sqrt(math.log(parent_visits + 1) / node.visits))

        def expand(node: MCTSNode):
            for act_id, data in actions:
                g2 = copy.deepcopy(node.game)
                try:
                    r = g2.perform_action(_make_action_input(act_id, data), raw=True)
                except Exception:
                    continue
                if not r.frame:
                    continue
                f2    = np.array(r.frame[-1])
                nh    = node.hist + [(act_id, data)]
                nlv   = getattr(r, 'levels_completed', 0) or 0
                child = MCTSNode(node, (act_id, data), g2, f2, nh)
                if nlv > lvl or getattr(g2, '_current_level_index', level_idx) > level_idx:
                    child.is_terminal = True
                    return child
                node.children.append(child)
            return None

        def rollout(node: MCTSNode) -> float:
            g, hist = copy.deepcopy(node.game), node.hist[:]
            f       = node.frame.copy()
            for _ in range(n_rollout):
                act_id, data = random.choice(actions)
                try:
                    r = g.perform_action(_make_action_input(act_id, data), raw=True)
                    if not r.frame:
                        continue
                    f   = np.array(r.frame[-1])
                    nlv = getattr(r, 'levels_completed', 0) or 0
                    hist.append((act_id, data))
                    if nlv > lvl or getattr(g, '_current_level_index', level_idx) > level_idx:
                        nonlocal best_r
                        if best_r is None or len(hist) < len(best_r):
                            best_r = hist[:]
                        return 10.0
                except Exception:
                    pass
            return _heuristic_score(f, bg, i_f)

        def backprop(node: MCTSNode, val: float):
            while node is not None:
                node.visits += 1
                node.value  += val
                node = node.parent

        iterations = 0
        while time.time() < deadline:
            node = root
            while node.children and not node.is_terminal:
                node = max(node.children, key=lambda c: ucb1(c, node.visits))
            if not node.is_terminal and node.visits > 0 and len(node.hist) < 30:
                win_child = expand(node)
                if win_child and win_child.is_terminal:
                    backprop(win_child, 10.0)
                    logger.info(f"MCTS L{level_idx}: {len(win_child.hist)} steps, {iterations} iter")
                    return win_child.hist
                if node.children:
                    node = random.choice(node.children)
            val = rollout(node)
            backprop(node, val)
            iterations += 1

        if best_r:
            logger.info(f"MCTS rollout L{level_idx}: {len(best_r)} steps")
        return best_r
    except Exception as e:
        logger.warning(f"MCTS error L{level_idx}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Dijkstra Transfer (cross-level solution transfer)
# ─────────────────────────────────────────────────────────────────────────────

def dijkstra_transfer(game_factory, level_idx: int,
                      prev_solution: List[Tuple[int, Optional[dict]]],
                      curr_f0: np.ndarray) -> Optional[List]:
    try:
        pg = game_factory()
        if not hasattr(pg, 'set_level'):
            return None
        pg.set_level(level_idx - 1)
        pg.perform_action(_make_action_input(0, None), raw=True)
        rp = pg.perform_action(_make_action_input(0, None), raw=True)
        if not rp.frame:
            return None
        f_prev    = np.array(rp.frame[-1])
        bg        = _bg_color(curr_f0)
        prev_objs = _get_objects(f_prev, _bg_color(f_prev))
        curr_objs = _get_objects(curr_f0, bg)
        if not prev_objs or not curr_objs:
            return None

        dx_list, dy_list = [], []
        for po in prev_objs:
            best, bd = None, float('inf')
            for co in curr_objs:
                if co['c'] == po['c']:
                    size_ratio = abs(co['n'] - po['n']) / max(po['n'], co['n'], 1)
                    if size_ratio < 0.5:
                        d = abs(co['cx'] - po['cx']) + abs(co['cy'] - po['cy'])
                        if d < bd:
                            bd, best = d, co
            if best:
                dx_list.append(best['cx'] - po['cx'])
                dy_list.append(best['cy'] - po['cy'])
        if not dx_list:
            return None
        dx = float(np.median(dx_list))
        dy = float(np.median(dy_list))

        obj_pts = np.array([[o['cx'], o['cy']] for o in curr_objs])

        transferred = []
        for act_id, data in prev_solution:
            if data and 'x' in data:
                nx = float(np.clip(data['x'] + dx, 0, 63))
                ny = float(np.clip(data['y'] + dy, 0, 63))
                if len(obj_pts) > 0:
                    dists = np.abs(obj_pts[:, 0] - nx) + np.abs(obj_pts[:, 1] - ny)
                    bi    = int(np.argmin(dists))
                    if dists[bi] < 10:
                        nx, ny = obj_pts[bi, 0], obj_pts[bi, 1]
                nd = dict(data)
                # [C2] keep x=col, y=row convention
                nd['x'] = int(round(nx))
                nd['y'] = int(round(ny))
                transferred.append((act_id, nd))
            else:
                transferred.append((act_id, data))

        # Validate
        g = game_factory()
        if not hasattr(g, 'set_level'):
            return None
        g.set_level(level_idx)
        g.perform_action(_make_action_input(0, None), raw=True)
        g.perform_action(_make_action_input(0, None), raw=True)
        for i, (act_id, data) in enumerate(transferred):
            try:
                r   = g.perform_action(_make_action_input(act_id, data), raw=True)
                nlv = getattr(r, 'levels_completed', 0) or 0
                if nlv > level_idx or getattr(g, '_current_level_index', level_idx) > level_idx:
                    logger.info(f"Dijkstra transfer L{level_idx}: {i+1} steps")
                    return transferred[:i + 1]
            except Exception:
                break
        return None
    except Exception as e:
        logger.warning(f"Dijkstra transfer error: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Game source discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_game_source(game_id: str, arc_env=None) -> Tuple[Optional[str], str]:
    import re
    gid      = game_id.split('-')[0]
    cls_name = gid[0].upper() + gid[1:]
    src      = None

    if arc_env and hasattr(arc_env, 'environment_info'):
        ei = arc_env.environment_info
        if hasattr(ei, 'local_dir') and ei.local_dir:
            from pathlib import Path
            ld = Path(ei.local_dir)
            for c in [ld / f"{gid}.py", ld / f"{cls_name.lower()}.py"]:
                if c.exists():
                    src = str(c)
                    txt = c.read_text()[:3000]
                    m   = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', txt)
                    if m:
                        cls_name = m.group(1)
                    break

    if not src:
        search_patterns = [
            f"/tmp/**/{gid}.py",
            f"/kaggle/**/{gid}.py",
            f"**/environment_files/**/{gid}.py",
            f"/kaggle/input/**/{gid}.py",
        ]
        for pat in search_patterns:
            hits = glob.glob(pat, recursive=True)
            if hits:
                src = hits[0]
                try:
                    txt = open(src).read()[:3000]
                    m   = re.search(r'class\s+(\w+)\s*\(\s*ARCBaseGame', txt)
                    if m:
                        cls_name = m.group(1)
                except Exception:
                    pass
                break

    return src, cls_name


# ─────────────────────────────────────────────────────────────────────────────
# Multi-level Solver
# ─────────────────────────────────────────────────────────────────────────────

class MultiSolver:
    """
    Cascade: Dijkstra-transfer → Beam → IDA* → BFS → A* → MCTS
    [M2] solve_level is called from a background thread; never blocks the
         choose_action frame loop.
    """

    def __init__(self, game_path: str, game_cls_name: str, budget: TimeBudget):
        self.game_path     = game_path
        self.game_cls_name = game_cls_name
        self.budget        = budget
        self.game_cls      = None
        self.solutions: Dict[int, List] = {}
        self._loaded       = False

    def load(self) -> bool:
        if self._loaded:
            return self.game_cls is not None
        self._loaded = True
        try:
            spec = importlib.util.spec_from_file_location('_arc_game', self.game_path)
            mod  = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.game_cls = getattr(mod, self.game_cls_name)
            logger.info(f"Loaded game class {self.game_cls_name} from {self.game_path}")
            return True
        except Exception as e:
            logger.warning(f"Solver load failed: {e}")
            return False

    def _factory(self):
        return self.game_cls()

    def solve_level(self, level_idx: int, total_budget: float = 120.0) -> Optional[List]:
        if not self.game_cls:
            return None

        deadline_all = time.time() + total_budget
        t0           = time.time()

        # Initial state
        try:
            g0 = self._factory()
            if not hasattr(g0, 'set_level'):
                logger.warning("Game has no set_level method — solver disabled")
                return None
            g0.set_level(level_idx)
            g0.perform_action(_make_action_input(0, None), raw=True)
            r0 = g0.perform_action(_make_action_input(0, None), raw=True)
            if not r0.frame:
                return None
            f0 = np.array(r0.frame[-1])
            bg = _bg_color(f0)
        except Exception as e:
            logger.warning(f"Init failed L{level_idx}: {e}")
            return None

        scan_budget = min(8.0, total_budget * 0.08)
        actions     = scan_available_actions(g0, f0, bg, timeout=scan_budget)
        if not actions:
            logger.warning(f"No effective actions found L{level_idx}")
            return None

        logger.info(f"L{level_idx}: {len(actions)} effective actions, budget={total_budget:.1f}s")

        # Dijkstra transfer from previous level
        if level_idx > 0 and level_idx - 1 in self.solutions:
            try:
                res = dijkstra_transfer(self._factory, level_idx,
                                        self.solutions[level_idx - 1], f0)
                if res:
                    self.solutions[level_idx] = res
                    return res
            except Exception:
                pass

        rem = deadline_all - time.time()
        if rem <= 2:
            return None

        # Budget allocation
        b_beam  = rem * 0.30
        b_ida   = rem * 0.20
        b_bfs   = rem * 0.20
        b_astar = rem * 0.20
        b_mcts  = rem * 0.10

        kwargs = dict(initial_frame=f0)

        # Beam search
        res = beam_search(self._factory, level_idx, actions,
                          deadline=time.time() + b_beam,
                          beam_width=min(128, max(32, len(actions) * 4)),
                          max_depth=40, **kwargs)
        if res:
            self.solutions[level_idx] = res
            return res

        if time.time() > deadline_all:
            return None

        # IDA*
        res = ida_star(self._factory, level_idx, actions,
                       deadline=time.time() + b_ida, max_depth=25, **kwargs)
        if res:
            self.solutions[level_idx] = res
            return res

        if time.time() > deadline_all:
            return None

        # Bounded BFS
        res = bounded_bfs(self._factory, level_idx, actions,
                          deadline=time.time() + b_bfs,
                          max_depth=20, max_states=15_000, **kwargs)
        if res:
            self.solutions[level_idx] = res
            return res

        if time.time() > deadline_all:
            return None

        # A*
        res = astar_search(self._factory, level_idx, actions,
                           deadline=time.time() + b_astar,
                           max_states=50_000, **kwargs)
        if res:
            self.solutions[level_idx] = res
            return res

        if time.time() > deadline_all:
            return None

        # MCTS (last resort)
        res = mcts_search(self._factory, level_idx, actions,
                          deadline=time.time() + b_mcts,
                          n_rollout=8, **kwargs)
        if res:
            self.solutions[level_idx] = res
            return res

        return None


# ─────────────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────────────

class MyAgent(Agent):
    MAX_ACTIONS = float('inf')
    _MAX_FRAMES  = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # [m1] Use abs() to avoid negative hash; clamp to numpy's uint32 range
        seed = abs(int(time.time() * 1e6) + hash(self.game_id)) % (2 ** 32 - 1)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        global _BUDGET
        _BUDGET.new_game()

        self.device = torch.device(
            'cuda' if torch.cuda.is_available()
            else ('mps' if torch.backends.mps.is_available() else 'cpu'))

        self.G  = 64
        self.IN = 26

        # Per-level state
        self.net: Optional[ForgeNet] = None
        self.opt = None
        self.per = PERBuffer(capacity=40_000)
        self.bsz    = 64
        self.tfreq  = 12
        self.cl     = -1        # current level index
        self.la     = 0         # local action counter
        self.fhist: deque = deque(maxlen=6)
        self.pt  = None         # prev tensor
        self.pai = None         # prev action index
        self.pr  = None         # prev raw frame
        self.ph  = None         # prev frame hash
        self._bg = 0
        self._wm = None
        self._wd = False
        self._eps       = 0.20
        self._eps_min   = 0.04
        self._eps_decay = 0.9995
        self._prev_objs = None
        self._visited_hashes: set = set()
        self._undo_avail  = False
        self._ckpt_hash   = None
        self._unproductive = 0
        self._ttt_steps   = 0
        self._ttt_max     = 3
        self._initial_frame: Optional[np.ndarray] = None

        # Cross-level
        self._clear_archive: deque = deque(maxlen=5000)
        self._paem = PersistentAEM(capacity=512)

        # Solver — runs in a background thread per level
        self._solver: Optional[MultiSolver] = None
        self._solver_init  = False
        self._solution: Optional[List]      = None
        self._sol_step     = 0
        self._solver_thread: Optional[threading.Thread] = None
        self._solution_ready = threading.Event()
        self._solution_lock  = threading.Lock()

    # ── Frame accessors ──────────────────────────────────────────────────────

    def _raw(self, fd) -> np.ndarray:
        return np.array(fd.frame, dtype=np.int64)[-1]

    def _lvl(self, fd) -> int:
        return getattr(fd, 'levels_completed', None) or getattr(fd, 'score', 0) or 0

    # ── Unified tensor construction ──────────────────────────────────────────
    # [M4] Single implementation used for both live inference and training.
    #      update_hist=True advances fhist (live path); False is read-only (training).

    def _build_tensor(self, frame: np.ndarray,
                      update_hist: bool = True) -> torch.Tensor:
        oh  = torch.zeros(16, 64, 64, dtype=torch.float32)
        oh.scatter_(0, torch.from_numpy(frame).unsqueeze(0), 1)

        cnt  = np.bincount(frame.flatten(), minlength=16)
        bg   = int(cnt.argmax())
        if update_hist:
            self._bg = bg
        mx   = max(int(cnt.max()), 1)
        bg_m = (frame == bg).astype(np.float32)
        rar  = np.zeros((64, 64), np.float32)
        for c in range(16):
            if cnt[c] > 0:
                rar[frame == c] = 1.0 - cnt[c] / mx

        pad  = np.pad(frame, 1, mode='edge')
        edge = ((frame != pad[:-2, 1:-1]) | (frame != pad[2:, 1:-1]) |
                (frame != pad[1:-1, :-2]) | (frame != pad[1:-1, 2:])
               ).astype(np.float32)

        # [M4] Fixed: rp varies down rows, cp varies across columns
        rp = np.linspace(0, 1, 64, dtype=np.float32)[:, None] * np.ones((1, 64), dtype=np.float32)
        cp = np.ones((64, 1), dtype=np.float32) * np.linspace(0, 1, 64, dtype=np.float32)[None, :]

        aug = torch.from_numpy(np.stack([bg_m, rar, edge, rp, cp]))

        hist_list = list(self.fhist)
        d1 = torch.zeros(3, 64, 64, dtype=torch.float32)
        for i, prev in enumerate(reversed(hist_list)):
            if i >= 3:
                break
            d1[i] = torch.from_numpy((frame != prev).astype(np.float32))

        d2 = torch.zeros(2, 64, 64, dtype=torch.float32)
        if len(hist_list) >= 2:
            d2[0] = torch.from_numpy((hist_list[-1] != hist_list[-2]).astype(np.float32))
        if len(hist_list) >= 4:
            d2[1] = torch.from_numpy((hist_list[-2] != hist_list[-4]).astype(np.float32))

        if update_hist:
            self.fhist.append(frame.copy())

        return torch.cat([oh, aug, d1, d2], 0)   # (26, 64, 64)

    def _tensor(self, fd) -> torch.Tensor:
        """Live path: updates fhist and moves result to device."""
        return self._build_tensor(self._raw(fd), update_hist=True).to(self.device)

    def _frame_to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """Training path: read-only, stays on CPU for batching."""
        return self._build_tensor(frame, update_hist=False)

    # ── Reward ───────────────────────────────────────────────────────────────

    def _reward(self, prev: np.ndarray, curr: np.ndarray, curr_h: str) -> float:
        interior = np.ones((64, 64), bool)
        interior[:2] = interior[62:] = False
        changed = bool(np.any((prev != curr) & interior))
        r = 0.0
        if curr_h not in self._visited_hashes:
            r += 1.5
        elif changed:
            r += 0.3
        else:
            r -= 0.1
        if changed:
            r += 0.5
        curr_objs = _get_objects(curr, self._bg)
        if self._prev_objs and curr_objs:
            moved = sum(
                1 for co in curr_objs for po in self._prev_objs
                if co['c'] == po['c']
                and 2 < abs(co['cx'] - po['cx']) + abs(co['cy'] - po['cy']) < 20)
            r += 0.3 * min(moved, 3)
        self._prev_objs = curr_objs
        self._visited_hashes.add(curr_h)
        return r

    # ── Action sampling ──────────────────────────────────────────────────────

    def _sample(self, logits: torch.Tensor, avail=None,
                temp: float = 1.0) -> Tuple[int, Optional[Tuple[int, int]]]:
        """
        Returns (aidx, coords) where:
          aidx 0-4  → ACTION1-ACTION5
          aidx 5    → ACTION6, coords = (row, col) i.e. (y, x)
        [C2] Convention: coords tuple is always (row=y, col=x).
        """
        al   = logits[:5].clone()
        cl   = logits[5:5 + 4096].clone()
        has6 = False

        if avail:
            mask = torch.full((5,), float('-inf'), device=al.device)
            for a in avail:
                aid = a.value if hasattr(a, 'value') else int(a)
                if 1 <= aid <= 5:
                    mask[aid - 1] = 0.0
                elif aid == 6:
                    has6 = True
            al = al + mask

        if not has6:
            cl = torch.full_like(cl, float('-inf'))
        if self._wm is not None:
            cl = cl + torch.log(self._wm.to(self.device).clamp(min=0.01))

        combined = torch.cat([al / temp, cl / temp]).clamp(min=-50.0)
        probs    = F.softmax(combined, dim=0).cpu().numpy()
        probs    = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        s        = probs.sum()
        if s < 1e-8:
            probs = np.ones(len(probs)) / len(probs)
        else:
            probs /= s

        idx = int(np.random.choice(len(probs), p=probs))
        if idx < 5:
            return idx, None
        # [C2] ci//G = row (y), ci%G = col (x)
        ci = idx - 5
        return 5, (ci // self.G, ci % self.G)

    # ── Click mask ───────────────────────────────────────────────────────────

    @staticmethod
    def _detect_template(frame: np.ndarray, bg: int) -> torch.Tensor:
        mask   = torch.ones(4096, dtype=torch.float32)
        non_bg = (frame != bg).astype(np.int32)
        col_a  = non_bg.sum(axis=0)
        row_a  = non_bg.sum(axis=1)
        idx    = np.arange(4096)
        for c in range(20, 44):
            if col_a[c] <= 2 and col_a[:c].sum() > 0 and col_a[c + 1:].sum() > 0:
                mask[idx % 64 <= c] = 0.05
                return mask
        for r in range(20, 44):
            if row_a[r] <= 2 and row_a[:r].sum() > 0 and row_a[r + 1:].sum() > 0:
                mask[idx // 64 <= r] = 0.05
                return mask
        return mask

    # ── Heuristic warm-up ────────────────────────────────────────────────────

    def _heuristic(self, frame: np.ndarray, avail, step: int
                   ) -> Tuple[int, Optional[Tuple[int, int]]]:
        av = {int(a.value) if hasattr(a, 'value') else int(a) for a in avail}
        for d in [1, 2, 3, 4]:
            if d in av and step < 4:
                return d - 1, None
        if 6 in av:
            cnt     = np.bincount(frame.flatten(), minlength=16)
            targets = []
            for c in range(16):
                if c == self._bg or cnt[c] == 0 or cnt[c] > 2000:
                    continue
                ys, xs = np.where(frame == c)
                if len(ys) >= 2:
                    targets.append((int(np.median(xs)), int(np.median(ys)), len(ys)))
            targets.sort(key=lambda t: t[2])
            pidx = step - 4
            if 0 <= pidx < len(targets):
                # col=x, row=y
                return 5, (targets[pidx][1], targets[pidx][0])
        choices = [a for a in av if 1 <= a <= 5]
        if choices:
            return random.choice(choices) - 1, None
        return 0, None

    # ── Training ─────────────────────────────────────────────────────────────

    def _train(self):
        if len(self.per) < self.bsz:
            return
        batch, idxs, weights = self.per.sample(self.bsz)
        if batch is None:
            return
        arch_n = min(self.bsz // 4, len(self._clear_archive))
        if arch_n > 0:
            arch    = random.sample(list(self._clear_archive), arch_n)
            batch   = batch[:self.bsz - arch_n] + arch
            weights = np.concatenate([weights[:self.bsz - arch_n],
                                      np.ones(arch_n, dtype=np.float32)])

        states = torch.stack([self._frame_to_tensor(e['s']).to(self.device)
                               for e in batch])
        acts   = torch.tensor([e['a'] for e in batch],
                               dtype=torch.long, device=self.device)
        rews   = torch.tensor([e['r'] for e in batch],
                               dtype=torch.float32, device=self.device)
        ws     = torch.tensor(weights, device=self.device)
        rews   = torch.sigmoid(rews)

        self.opt.zero_grad()
        logits  = self.net(states)
        acts_c  = acts.clamp(0, logits.size(1) - 1)
        sel     = logits.gather(1, acts_c.unsqueeze(1)).squeeze(1)
        el      = F.binary_cross_entropy_with_logits(sel, rews, reduction='none')
        loss    = (el * ws).mean()
        p       = torch.sigmoid(logits)
        loss    = loss - 0.0001 * p[:, :5].mean() - 0.00001 * p[:, 5:].mean()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
        self.opt.step()

        n_per = len(idxs)
        if n_per > 0:
            self.per.update_priorities(idxs[:n_per],
                                       el[:n_per].detach().cpu().numpy())

    # ── TTT (test-time training) ──────────────────────────────────────────────

    def _ttt_step(self, tensor: torch.Tensor):
        if self._ttt_steps >= self._ttt_max or self.opt is None:
            return
        self.net.train()
        self.opt.zero_grad()
        t1 = tensor.unsqueeze(0)
        t2 = torch.roll(t1, shifts=random.randint(0, 3), dims=-1)
        with torch.no_grad():
            l2 = self.net(t2)
        l1   = self.net(t1)
        loss = F.mse_loss(l1, l2.detach()) * 0.01
        loss.backward()
        self.opt.step()
        self.net.eval()
        self._ttt_steps += 1

    # ── Archive ───────────────────────────────────────────────────────────────

    def _archive_level(self):
        """[M6] Archive current PER contents BEFORE resetting PER."""
        if len(self.per) == 0:
            return
        batch, _, _ = self.per.sample(min(200, len(self.per)))
        if batch:
            self._clear_archive.extend(batch)

    # ── Expert injection from solver solution ─────────────────────────────────

    def _inject_expert_demos(self, level_idx: int):
        if not (self._solver and self._solver.game_cls):
            return
        sol = self._solver.solutions.get(level_idx)
        if not sol:
            return
        try:
            g = self._solver.game_cls()
            # [M5] Guard: check set_level exists before calling
            if not hasattr(g, 'set_level'):
                logger.warning("Game has no set_level — skipping expert injection")
                return
            g.set_level(level_idx)
            g.perform_action(_make_action_input(0, None), raw=True)
            g.perform_action(_make_action_input(0, None), raw=True)
            for act_id, data in sol:
                try:
                    pf = np.array(g.get_pixels(0, 0, 64, 64), dtype=np.int64)
                except Exception:
                    pf = self.pr.copy() if self.pr is not None else np.zeros((64, 64), np.int64)
                g.perform_action(_make_action_input(act_id, data), raw=True)
                if act_id <= 5:
                    a_idx = act_id - 1
                elif act_id == 6 and data:
                    # [C2] row=y, col=x
                    a_idx = 5 + data.get('y', 0) * 64 + data.get('x', 0)
                else:
                    a_idx = 0
                exp = {'s': pf.copy(), 'a': a_idx, 'r': 2.0}
                self.per.add_expert(exp)
                self._clear_archive.append(exp)
            if len(self.per) >= self.bsz:
                for _ in range(min(15, len(self.per) // max(self.bsz, 1))):
                    self._train()
        except Exception as e:
            logger.warning(f"Expert inject L{level_idx}: {e}")

    # ── Background solver launcher ────────────────────────────────────────────

    def _launch_solver(self, level_idx: int):
        """[M2] Run solve_level in a daemon thread. choose_action never blocks."""
        if not (self._solver and self._solver.game_cls):
            self._solution_ready.set()
            return

        search_time = min(_BUDGET.search_budget(0.50), 90.0)
        if level_idx > 0:
            search_time = min(search_time * (1 + 0.08 * level_idx), 160.0)

        self._solution_ready.clear()

        def _run():
            try:
                sol = self._solver.solve_level(level_idx, total_budget=search_time)
                with self._solution_lock:
                    if sol:
                        self._solution = sol
            except Exception as e:
                logger.warning(f"Solver thread error L{level_idx}: {e}")
            finally:
                self._solution_ready.set()

        self._solver_thread = threading.Thread(target=_run, daemon=True)
        self._solver_thread.start()

    # ── is_done ───────────────────────────────────────────────────────────────

    def is_done(self, frames, lf) -> bool:
        try:
            # [M7] Guard lf.state access
            state = getattr(lf, 'state', None)
            if state == GameState.WIN:
                _BUDGET.mark_done()   # [M1] increment so per-game budget stays accurate
                return True
            if _BUDGET.is_hard_deadline():
                return True
            # Per-game soft budget: end if we've spent 2.5× per-game budget
            per_game = _BUDGET.per_game_budget()
            if _BUDGET.game_elapsed() > per_game * 2.5:
                logger.info(f"Per-game budget exceeded ({_BUDGET.game_elapsed():.0f}s), "
                            f"per_game={per_game:.0f}s")
                _BUDGET.mark_done()
                return True
            return False
        except Exception:
            return True

    # ── choose_action — must return a GameAction enum member ──────────────────

    def choose_action(self, frames, lf) -> GameAction:
        try:
            lvl = self._lvl(lf)

            # ── Level transition ──────────────────────────────────────────────
            if lvl != self.cl:
                # Lazy solver init
                if not self._solver_init:
                    self._solver_init = True
                    src, cls = find_game_source(
                        self.game_id, getattr(self, 'arc_env', None))
                    if src:
                        self._solver = MultiSolver(src, cls, _BUDGET)
                        self._solver.load()

                # [M6] Archive BEFORE resetting PER
                self._archive_level()

                # Reset per-level state
                with self._solution_lock:
                    self._solution = None
                self._sol_step = 0
                self.per  = PERBuffer(capacity=40_000)
                self.net  = ForgeNet(self.IN, self.G).to(self.device)

                # Load pretrained weights if available
                for wp in [
                    '/kaggle/input/forge-pretrained-weights/pretrained_weights.pt',
                    'pretrained_weights.pt',
                ]:
                    try:
                        if os.path.exists(wp):
                            saved = torch.load(wp, map_location=self.device,
                                               weights_only=True)
                            ms = self.net.state_dict()
                            ms.update({k: v for k, v in saved.items()
                                       if k in ms and v.shape == ms[k].shape})
                            self.net.load_state_dict(ms)
                            logger.info(f"Loaded pretrained weights from {wp}")
                            break
                    except Exception:
                        pass

                self.net.eval()
                self.opt = optim.Adam(self.net.parameters(), lr=3e-4)

                # Reset per-level counters
                self.pt    = self.pai = self.pr = self.ph = None
                self.cl    = lvl
                self.la    = 0
                self.fhist.clear()
                self._wd              = False
                self._wm              = None
                self._eps             = 0.20
                self._prev_objs       = None
                self._visited_hashes  = set()
                self._ckpt_hash       = None
                self._unproductive    = 0
                self._ttt_steps       = 0
                self._initial_frame   = None

                # Inject expert demos from the PREVIOUS level's solution
                if lvl > 0:
                    self._inject_expert_demos(lvl - 1)

                # [M2] Launch solver in background thread — does not block here
                self._launch_solver(lvl)

            # ── Hard deadline guard ───────────────────────────────────────────
            if _BUDGET.is_hard_deadline():
                act = _fresh_action('RESET')
                act.reasoning = {"action": "RESET", "reason": "hard_deadline"}
                return act

            # ── [M7] Safe state access ────────────────────────────────────────
            state = getattr(lf, 'state', None)

            if state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
                self.pt = self.pai = self.pr = self.ph = None
                act = _fresh_action('RESET')
                act.reasoning = {"action": "RESET", "reason": "state_reset"}
                return act

            # ── Record initial frame for heuristic baseline ───────────────────
            raw = self._raw(lf)
            if self._initial_frame is None:
                self._initial_frame = raw.copy()

            # ── Solution playback ─────────────────────────────────────────────
            # Check if solver has delivered a solution since last check
            with self._solution_lock:
                sol_available = (self._solution is not None and
                                 self._sol_step < len(self._solution))

            if sol_available:
                with self._solution_lock:
                    act_id, data = self._solution[self._sol_step]
                    self._sol_step += 1
                    sol_total = len(self._solution)

                self.fhist.append(raw.copy())
                self.pr  = raw.copy()
                self.la += 1

                if act_id == 6:
                    # [C1] Fresh ref each call
                    act = _fresh_action('ACTION6')
                    act.reasoning = {
                        "action": "ACTION6",
                        # [C2] data['x']=col, data['y']=row
                        "data": {"x": int(data['x']), "y": int(data['y'])},
                        "reason": f"sol:{self._sol_step}/{sol_total}",
                    }
                else:
                    name = _NAMES.get(act_id, 'ACTION1')
                    act  = _fresh_action(name)
                    act.reasoning = {
                        "action": name,
                        "reason": f"sol:{self._sol_step}/{sol_total}",
                    }
                return act

            # ── Online RL policy ──────────────────────────────────────────────
            tensor = self._tensor(lf)
            ch     = hashlib.md5(raw.tobytes()).hexdigest()[:16]
            avail  = getattr(lf, 'available_actions', None) or []
            self._undo_avail = any(
                (a.value if hasattr(a, 'value') else int(a)) == 7 for a in avail)

            # Store transition
            if self.pt is not None and self.pai is not None:
                r   = self._reward(self.pr, raw, ch)
                exp = {'s': self.pr.copy(), 'a': self.pai, 'r': r}
                self.per.add(exp)
                interior = np.ones((64, 64), bool)
                interior[:2] = interior[62:] = False
                if np.any((self.pr != raw) & interior):
                    self._paem.push((self.pr != raw).astype(bool) & interior,
                                    min(self.pai, 4), r)
                    self._ckpt_hash    = ch
                    self._unproductive = 0
                else:
                    self._unproductive += 1

            if self._wm is None:
                self._wm = self._detect_template(raw, self._bg)

            # Undo if stuck
            if self._undo_avail and self._unproductive >= 25 and self._ckpt_hash:
                self._unproductive = 0
                act = _fresh_action('ACTION7')
                act.reasoning = {"action": "ACTION7", "reason": "undo_stuck"}
                self.pt  = tensor
                # [M3] Use a safe neutral action index (0 = ACTION1) not 6
                self.pai = 0
                self.pr  = raw.copy()
                self.ph  = ch
                self.la += 1
                return act

            # Warm-up vs trained policy
            aidx, coords = 0, None
            if not self._wd:
                if self.la < 10:
                    aidx, coords = self._heuristic(raw, avail, self.la)
                else:
                    self._wd = True
                    for _ in range(min(5, len(self.per) // max(self.bsz, 1))):
                        self._train()

            if self._wd:
                self._ttt_step(tensor)
                if random.random() < self._eps:
                    aidx, coords = self._sample(
                        torch.zeros(4101, device=self.device), avail, temp=2.5)
                else:
                    self.net.eval()
                    with torch.no_grad():
                        mem = self._paem.get_tensors(self.device, max_mem=64)
                        if mem[0] is not None:
                            logits = self.net(tensor.unsqueeze(0), *mem).squeeze(0)
                        else:
                            logits = self.net(tensor.unsqueeze(0)).squeeze(0)
                    aidx, coords = self._sample(logits, avail, temp=0.5)
                self._eps = max(self._eps_min, self._eps * self._eps_decay)
            elif self.la >= 10:
                self._wd = True

            # ── Build the returned GameAction ─────────────────────────────────
            if aidx < 5:
                # [C1] Fresh reference — do not mutate shared enum member
                name = _NAMES[aidx + 1]       # aidx 0 → ACTION1, etc.
                act  = _fresh_action(name)
                act.reasoning  = {"action": name, "reason": f"rl:a{aidx+1}"}
                action_idx     = aidx
            else:
                # aidx == 5 → ACTION6
                # [C2] coords = (row=y, col=x)
                row, col = coords
                act = _fresh_action('ACTION6')
                act.reasoning = {
                    "action": "ACTION6",
                    "data": {"x": int(col), "y": int(row)},
                    "reason": f"rl:c(col={col},row={row})",
                }
                action_idx = 5 + row * self.G + col

            self.pt  = tensor
            self.pai = action_idx
            self.pr  = raw.copy()
            self.ph  = ch
            self.la += 1

            if self.la % self.tfreq == 0 and self._wd:
                self._train()
                if (self.la // self.tfreq) % 5 == 0:
                    self._ttt_steps = 0

            return act

        except Exception as e:
            traceback.print_exc()
            # [C1] Always fresh reference on fallback too
            act = _fresh_action(_ACTION_MAP[random.randint(0, 4)].name)
            act.reasoning = {"action": act.name, "reason": f"err:{str(e)[:80]}"}
            return act
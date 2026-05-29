from __future__ import annotations

import collections
import random
import time
import heapq
from typing import Dict, List, Optional, Set, Tuple

from env import (
    ALPHA, BETA, DIRS,
    DeliveryEnv, Order, Shipper, manhattan, SEED, valid_next_pos
)
from solvers.solver import Solver

CANDIDATE_LIMIT         = 50    
STAY_STILL_SCORE_THRESH = 0.05  
OPPORTUNISTIC_RADIUS    = 2     
URGENCY_MAX_MULT        = 2.5   
STUCK_THRESHOLD         = 3     
RELEASE_THRESHOLD       = 6     

class SpaceTimeAStar:
    """
    Space-Time A* for MAPD Prioritized Planning.
    Plans paths in (row, col, time) space to avoid dynamic obstacles.
    """
    def __init__(self, dc: GridDistanceCache, grid: List[List[int]], horizon: int = 12):
        self.dc = dc
        self.grid = grid
        self.N = len(grid)
        self.horizon = horizon
        self.ACTIONS = (("U", -1, 0), ("D", 1, 0), ("L", 0, -1), ("R", 0, 1), ("S", 0, 0))

    def plan(
        self, 
        start: Tuple[int, int], 
        goal: Tuple[int, int], 
        v_constraints: Set[Tuple[int, int, int]], 
        e_constraints: Set[Tuple[Tuple[int, int], Tuple[int, int], int]]
    ) -> List[str]:
        """
        v_constraints: set of (r, c, t) - cannot be at (r, c) at time t.
        e_constraints: set of ((r1, c1), (r2, c2), t) - cannot traverse from (r1, c1) to (r2, c2) at time t.
        """
        # (f, g, r, c, t, path)
        h0 = self.dc.dist(start, goal)
        if h0 >= 10_000:
            return ["S"] * self.horizon # Unreachable
            
        open_list = [(h0, 0, start[0], start[1], 0, [])]
        visited = set()

        while open_list:
            _, g, r, c, t, path = heapq.heappop(open_list)

            if t == self.horizon:
                return path

            if (r, c) == goal:
                # Reached goal early, pad with stay actions
                padded = list(path)
                while len(padded) < self.horizon:
                    padded.append("S")
                return padded

            state_key = (r, c, t)
            if state_key in visited:
                continue
            visited.add(state_key)

            for act, dr, dc in self.ACTIONS:
                nr, nc = r + dr, c + dc
                nt = t + 1
                
                # Map boundaries & static obstacles
                if not (0 <= nr < self.N and 0 <= nc < self.N and self.grid[nr][nc] == 0):
                    continue
                
                # Vertex constraints (someone else is there at time nt)
                if (nr, nc, nt) in v_constraints:
                    continue
                    
                # Edge constraints (prevent head-on swapping)
                if ((r, c), (nr, nc), nt) in e_constraints:
                    continue

                h = self.dc.dist((nr, nc), goal)
                if h < 10_000:
                    heapq.heappush(open_list, (g + 1 + h, g + 1, nr, nc, nt, path + [act]))

        # Fallback if trapped: stay still
        return ["S"] * self.horizon
    
class GridDistanceCache:
    """Precomputes and caches full-grid BFS maps on demand."""
    ACTIONS: Tuple = (("U", -1, 0), ("D", 1, 0), ("L", 0, -1), ("R", 0, 1))

    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.N    = len(grid)
        self._maps: Dict[Tuple[int, int], List[List[int]]] = {}
        self.free_cells = [(r, c) for r in range(self.N) for c in range(self.N) if grid[r][c] == 0]

    def get_dist_map(self, goal: Tuple[int, int]) -> List[List[int]]:
        if goal not in self._maps:
            self._maps[goal] = self._bfs_flood(goal)
        return self._maps[goal]

    def _bfs_flood(self, src: Tuple[int, int]) -> List[List[int]]:
        N = self.N
        dist = [[-1] * N for _ in range(N)]
        sr, sc = src
        if self.grid[sr][sc] == 1: return dist
        dist[sr][sc] = 0
        q = collections.deque([(sr, sc)])
        while q:
            r, c = q.popleft()
            for _, dr, dc in self.ACTIONS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N and self.grid[nr][nc] == 0 and dist[nr][nc] == -1:
                    dist[nr][nc] = dist[r][c] + 1
                    q.append((nr, nc))
        return dist

    def dist(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        d = self.get_dist_map(b)[a[0]][a[1]]
        return d if d >= 0 else 10_000

    def next_move_towards(self, start: Tuple[int, int], goal: Tuple[int, int]) -> str:
        if start == goal: return "S"
        dm = self.get_dist_map(goal)
        r, c = start
        cur_d = dm[r][c]
        if cur_d == -1: return "S"

        for act, dr, dc in self.ACTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.N and 0 <= nc < self.N and dm[nr][nc] != -1 and dm[nr][nc] < cur_d:
                return act
        return "S"


class DispatchScorer:
    def __init__(self, cfg: dict, dc: GridDistanceCache):
        self.T  = cfg["T"]
        self.dc = dc

    @staticmethod
    def r_base(w: float) -> float:
        r0 = 10.0
        if w <= 0.2:  return r0 * 0.4
        if w <= 3.0:  return r0 * 1.0
        if w <= 10.0: return r0 * 1.5
        if w <= 30.0: return r0 * 2.0
        return r0 * 3.0

    def estimate_reward(self, order: Order, t_delivery: int) -> float:
        rb = self.r_base(order.w)
        if t_delivery <= order.et:
            bonus = max(0.0, (order.et - t_delivery) / max(order.et, 1))
            return ALPHA[order.p] * rb * (1.0 + bonus)
        factor = max(0.0, 1.0 - (t_delivery - order.et) / max(self.T, 1))
        return BETA[order.p] * rb * factor

    def score(self, sh: Shipper, order: Order, t: int, orders: Dict[int, Order]) -> float:
        d_pick  = self.dc.dist((sh.r, sh.c), (order.sx, order.sy))
        d_del   = self.dc.dist((order.sx, order.sy), (order.ex, order.ey))
        n_steps = d_pick + d_del
        if n_steps >= 10_000: return -1.0

        exp_rew = self.estimate_reward(order, t + n_steps)
        w_carried = sum(orders[oid].w for oid in sh.bag if oid in orders)
        w_ratio = w_carried / max(sh.W_max, 1.0)
        move_est = 0.01 * (1.0 + w_ratio) * n_steps
        
        slack = order.et - t - n_steps
        feas_mult = 1.0 if slack >= 0 else (BETA[order.p] / ALPHA[order.p])
        
        # Stop-loss: If we can't offset 50% of the movement cost, prune.
        if slack < 0 and exp_rew <= (move_est * 0.5): return -1.0
        
        net = exp_rew - move_est
        if net <= 0: return -1.0

        urgency = min(1.0 + max(0.0, 1.0 - max(slack, 0) / max(order.et - t, 1)), URGENCY_MAX_MULT) if order.et > t else 0.5
        return (net * urgency * feas_mult) / (n_steps + 1)

    def delivery_score(self, oid: int, orders: Dict[int, Order], pos: Tuple[int, int], t: int) -> float:
        o   = orders[oid]
        d   = self.dc.dist(pos, (o.ex, o.ey))
        rew = self.estimate_reward(o, t + d)
        urgency = 2.0 if (o.et - t) < d * 2 else 1.0
        return rew * urgency / (d + 1)


class GreedyBFS(Solver):
    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.cfg = {
            "N": env.N,
            "C": env.C,
            "T": env.T
        }
        
        self.N      = self.cfg["N"]
        self.T      = self.cfg["T"]
        self.C      = self.cfg["C"]
        self.rng    = random.Random(SEED + 10)
        self.dc     = GridDistanceCache(self.grid)
        self.scorer = DispatchScorer(self.cfg, self.dc)
        
        # Online Tracking Structs
        self.global_orders: Dict[int, Order] = {}
        
        self.dc     = GridDistanceCache(self.grid)
        self.scorer = DispatchScorer(self.cfg, self.dc)
        self.st_planner = SpaceTimeAStar(self.dc, self.grid, horizon=10) # 10-step lookahead
        
        # We need a persistent route cache so we aren't running A* for every agent every tick
        self.active_routes: Dict[int, List[str]] = {i: [] for i in range(self.C)}

        # Persistent Agent Memory (Needed since Grader resets Shipper objects every tick)
        self.agents = {
            i: {
                "phase": "idle", # pickup | deliver | idle
                "target_oid": -1,
                "detour_oid": -1,
                "stuck_ticks": 0,
                "last_pos": (-1, -1)
            } for i in range(self.C)
        }

    def _best_delivery_target(self, sh: Shipper, t: int) -> int:
        if len(sh.bag) == 1: return sh.bag[0]
        return max(sh.bag, key=lambda oid: self.scorer.delivery_score(oid, self.global_orders, (sh.r, sh.c), t))

    def _assign_targets(self, shippers: List[Shipper], pending: List[Order], t: int, claimed: Set[int]):
        """Assign primary targets to idle shippers."""
        if not pending: return

        # Sort by urgency of current cargo (more urgent -> process assignments later to maintain focus)
        def constrain(sh: Shipper) -> int:
            return min((self.global_orders[oid].et - t for oid in sh.bag if oid in self.global_orders), default=10_000)

        for sh in sorted(shippers, key=constrain):
            state = self.agents[sh.id]
            if state["target_oid"] >= 0: continue
            
            # Forced Delivery if loaded but targetless
            if sh.bag:
                state["target_oid"] = self._best_delivery_target(sh, t)
                state["phase"] = "deliver"
                continue

            candidates = sorted(
                [o for o in pending if o.id not in claimed],
                key=lambda o: manhattan(sh.r, sh.c, o.sx, o.sy),
            )[:CANDIDATE_LIMIT]

            best_sc, best_order = STAY_STILL_SCORE_THRESH, None
            for o in candidates:
                if not sh.can_carry(o, self.global_orders): continue
                sc = self.scorer.score(sh, o, t, self.global_orders)
                if sc <= 0: continue
                
                # Adaptive Density Multiplier
                if sh.K_max - len(sh.bag) > 1:
                    density = sum(1 for other in candidates if other.id != o.id and self.dc.dist((o.sx, o.sy), (other.sx, other.sy)) <= 3)
                    sc *= (1.0 + 0.1 * density)

                if sc > best_sc:
                    best_sc, best_order = sc, o

            if best_order is not None:
                state["target_oid"] = best_order.id
                state["phase"] = "pickup"
                claimed.add(best_order.id)

    def _try_opportunistic_detour(self, sh: Shipper, pending: List[Order], claimed: Set[int], t: int):
        """Find profitable pick-ups along the delivery route."""
        state = self.agents[sh.id]
        if state["phase"] != "deliver" or state["target_oid"] < 0 or len(sh.bag) >= sh.K_max or state["detour_oid"] >= 0:
            return

        dlv_tg = (self.global_orders[state["target_oid"]].ex, self.global_orders[state["target_oid"]].ey)
        d_dir  = self.dc.dist((sh.r, sh.c), dlv_tg)
        best_gain, best_order = 0.0, None

        for o in pending:
            if o.id in claimed or not sh.can_carry(o, self.global_orders): continue
            
            d_pick = self.dc.dist((sh.r, sh.c), (o.sx, o.sy))
            if d_pick > OPPORTUNISTIC_RADIUS: continue

            d_via       = d_pick + self.dc.dist((o.sx, o.sy), dlv_tg)
            extra_steps = max(0, d_via - d_dir)
            
            exp_rew = self.scorer.estimate_reward(o, t + d_via)
            if exp_rew <= 0: continue

            # Calculate time penalty on existing cargo
            reward_loss = sum(
                max(0.0, self.scorer.estimate_reward(self.global_orders[coid], t + self.dc.dist((sh.r, sh.c), (self.global_orders[coid].ex, self.global_orders[coid].ey)))
                    - self.scorer.estimate_reward(self.global_orders[coid], t + self.dc.dist((sh.r, sh.c), (self.global_orders[coid].ex, self.global_orders[coid].ey)) + extra_steps))
                for coid in sh.bag if coid in self.global_orders
            )

            gain = exp_rew - (0.01 * 1.5 * extra_steps) - reward_loss
            if gain > best_gain:
                best_gain, best_order = gain, o

        if best_order is not None:
            state["detour_oid"] = best_order.id
            claimed.add(best_order.id)

    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        while not obs.get("done", False):
            t = obs["t"]
            shippers = obs["shippers"]
            
            # Sync Environment State
            for oid, o in obs["orders"].items():
                self.global_orders[oid] = o

            # Clean Invalid Targets (e.g. poached by another agent or delivered)
            pending_ids = {oid for oid, o in obs["orders"].items() if not o.picked and not o.delivered}
            for sh in shippers:
                state = self.agents[sh.id]
                t_oid, d_oid = state["target_oid"], state["detour_oid"]
                
                if t_oid >= 0 and t_oid not in pending_ids and t_oid not in sh.bag:
                    state["target_oid"], state["phase"] = -1, "idle"
                if d_oid >= 0 and d_oid not in pending_ids:
                    state["detour_oid"] = -1

                # Update Stuck Status (Collision Resolution Metric)
                if (sh.r, sh.c) == state["last_pos"] and state["phase"] != "idle":
                    state["stuck_ticks"] += 1
                else:
                    state["stuck_ticks"] = 0
                state["last_pos"] = (sh.r, sh.c)

                # Absolute Target Lock Release (Prevent permanent deadlocks)
                if state["stuck_ticks"] >= RELEASE_THRESHOLD and state["phase"] == "pickup":
                    state["target_oid"], state["detour_oid"], state["phase"] = -1, -1, "idle"
                    state["stuck_ticks"] = 0

            # --- 1. Tactical Planning ---
            pending_orders = [self.global_orders[oid] for oid in pending_ids]
            claimed = {self.agents[sh.id]["target_oid"] for sh in shippers} | {self.agents[sh.id]["detour_oid"] for sh in shippers}
            
            self._assign_targets(shippers, pending_orders, t, claimed)
            
            for sh in shippers:
                self._try_opportunistic_detour(sh, pending_orders, claimed, t)

            # --- 2. Action Generation (Prioritized Space-Time Planning) ---
            actions = {}
            
            # Constraints for the current tick's planning cycle
            v_constraints = set()
            e_constraints = set()
            
            # Plan strictly in order of agent ID (environment tie-breaker rules)
            for sh in sorted(shippers, key=lambda x: x.id):
                state = self.agents[sh.id]
                
                # Determine goal
                goal = (sh.r, sh.c) # Default: stay put
                if state["detour_oid"] >= 0:
                    d_ord = self.global_orders[state["detour_oid"]]
                    goal = (d_ord.sx, d_ord.sy)
                elif state["target_oid"] >= 0:
                    t_ord = self.global_orders[state["target_oid"]]
                    goal = (t_ord.sx, t_ord.sy) if state["phase"] == "pickup" else (t_ord.ex, t_ord.ey)
                elif not sh.bag and (min(sh.r, self.N - 1 - sh.r) < 3 or min(sh.c, self.N - 1 - sh.c) < 3):
                    # Staging
                    goal = (self.N // 2, self.N // 2)

                # Generate dynamic path
                path = self.st_planner.plan((sh.r, sh.c), goal, v_constraints, e_constraints)
                
                # Extract immediate move
                move = path[0] if path else "S"
                op = 0
                
                # Reserve future space-time coordinates for lower-priority agents
                curr_pos = (sh.r, sh.c)
                for t_idx, m in enumerate(path):
                    dr, dc = DIRS.get(m, (0, 0))
                    nxt_pos = (curr_pos[0] + dr, curr_pos[1] + dc)
                    
                    v_constraints.add((nxt_pos[0], nxt_pos[1], t_idx + 1))
                    # Prevent swaps: if I move A->B, you cannot move B->A
                    e_constraints.add((nxt_pos, curr_pos, t_idx + 1)) 
                    
                    curr_pos = nxt_pos
                    
                # Determine expected next position for cargo operations
                nxt_pos_actual = valid_next_pos((sh.r, sh.c), move, self.grid)
                if nxt_pos_actual == goal:
                    if state["detour_oid"] >= 0:
                        op = 1
                        state["detour_oid"] = -1
                    elif state["phase"] == "pickup":
                        op = 1
                        state["phase"] = "deliver"
                    elif state["phase"] == "deliver":
                        op = 2
                        state["target_oid"] = -1
                        state["phase"] = "idle"

                actions[sh.id] = (move, op)

            obs, _, _, _ = self.env.step(actions)

        return self.env.result("Greedy BFS", elapsed_sec=time.time() - start_time)
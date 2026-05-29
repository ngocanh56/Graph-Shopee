from __future__ import annotations

import collections
import random
import time
import heapq
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

from env import (
    SEED,
    DeliveryEnv, Order, Shipper,delivery_reward
)

from topo_analyzer import IterativeTopologyAnalyzer
from solvers.solver import Solver
import traceback

INF = 10_000
MOVES = ("U", "D", "L", "R", "S")
DIR_DELTAS: Dict[str, Tuple[int, int]] = {
    "U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1), "S": (0, 0)
}

ACO_ALPHA      = 1.0
ACO_BETA       = 2.5
ACO_RHO        = 0.15
THETA_EVAP     = 20.0
SIGMA_ELITIST  = 5
ACO_Q          = 50.0
TAU_MIN        = 1e-4
TAU_MAX        = 10.0
BATCH_RADIUS   = 5
HOTSPOT_DECAY  = 0.95

_BASE_ACO_MS = 350
_ACO_STAGNATE_LIMIT = 5

def _r_base(w: float) -> float:
    if w <= 0.2:  return 4.0
    if w <= 3.0:  return 10.0
    if w <= 10.0: return 15.0
    if w <= 30.0: return 20.0
    return 30.0

def _w_carried(sh: Shipper, orders: Dict[int, Order]) -> float:
    return sum(orders[oid].w for oid in sh.bag if oid in orders)


def _can_carry(sh: Shipper, order: Order, orders: Dict[int, Order]) -> bool:
    if order.picked or order.delivered: return False
    wc = _w_carried(sh, orders)
    return len(sh.bag) < sh.K_max and wc + order.w <= sh.W_max

class OnlineSurgeDetector:
    def __init__(self, G: int, T: int, window: int = 20):
        self.baseline_rate = G / max(T, 1)
        self.window = window
        self.surge_level = 0
        self.surge_center: Optional[Tuple[int, int]] = None
        self._arrival_ts: Deque[int] = deque()
        self._recent_src: Deque[Tuple[int, int]] = deque(maxlen=80)

    def update(self, t: int, new_oids: Set[int], orders: Dict[int, Order]) -> None:
        for oid in new_oids:
            self._arrival_ts.append(t)
            if oid in orders:
                self._recent_src.append((orders[oid].sx, orders[oid].sy))
        
        cutoff = t - self.window
        while self._arrival_ts and self._arrival_ts[0] < cutoff:
            self._arrival_ts.popleft()
            
        current_rate = len(self._arrival_ts) / max(self.window, 1)
        ratio = current_rate / max(self.baseline_rate, 1e-9)
        
        if   ratio > 3.5: self.surge_level = 3
        elif ratio > 2.0: self.surge_level = 2
        elif ratio > 1.4: self.surge_level = 1
        else:             self.surge_level = 0
            
        if self.surge_level >= 1 and self._recent_src:
            rs = [p[0] for p in self._recent_src]
            cs = [p[1] for p in self._recent_src]
            self.surge_center = (sum(rs) // len(rs), sum(cs) // len(cs))
        else:
            self.surge_center = None

    def proximity_boost(self, pos: Tuple[int, int], N: int) -> float:
        if self.surge_center is None or self.surge_level == 0:
            return 1.0
        dist = abs(pos[0] - self.surge_center[0]) + abs(pos[1] - self.surge_center[1])
        max_dist = max(N * 1.4, 1.0)
        proximity = max(0.0, 1.0 - dist / max_dist)
        return 1.0 + self.surge_level * 0.35 * proximity

class AdaptivePlanner:
    CBS_CT_THRESHOLD = 10 

    def __init__(self, solver: "ACOSolver"):
        self.solver = solver
        self.grid   = solver.grid
        self.N      = solver.N
        self.C      = solver.C
        
        if   self.N <= 20: self.horizon = max(8, min(24, self.N // 3))
        elif self.N <= 50: self.horizon = max(6, min(12, self.N // 5))
        else:              self.horizon = max(4, min(8,  self.N // 8))
            
        self.use_ct = (self.C <= self.CBS_CT_THRESHOLD)

    def space_time_astar(
        self,
        start: Tuple[int, int],
        goal:  Tuple[int, int],
        v_constraints: Set[Tuple[int, int, int]],
        e_constraints: Set[Any],
        max_expansions: int = 150
    ) -> List[str]:
        h0 = self.solver._dist(start, goal)
        if h0 >= INF:
            return ["S"] * self.horizon

        open_list: List = []
        heapq.heappush(open_list, (h0, 0, start[0], start[1], 0, []))
        visited: Set[Tuple[int, int, int]] = set()
        expansions = 0

        while open_list:
            expansions += 1
            if expansions > max_expansions:
                return ["S"] * self.horizon

            _, g, r, c, t, path = heapq.heappop(open_list)
            if t == self.horizon:
                return path
            
            if (r, c) == goal:
                padded = list(path)
                while len(padded) < self.horizon: padded.append("S")
                return padded
                
            key = (r, c, t)
            if key in visited: continue
            visited.add(key)
            
            for act in MOVES:
                dr, dc = DIR_DELTAS[act]
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.N and 0 <= nc < self.N and self.grid[nr][nc] == 0):
                    continue
                if (nr, nc, t + 1) in v_constraints:
                    continue
                if ((r, c), (nr, nc), t + 1) in e_constraints:
                    continue
                    
                h = self.solver._dist((nr, nc), goal)
                if h < INF:
                    heapq.heappush(open_list, (g + 1 + h, g + 1, nr, nc, t + 1, path + [act]))
                    
        return ["S"] * self.horizon

    def plan(
        self,
        active_agents: List[int],
        starts: Dict[int, Tuple[int, int]],
        goals:  Dict[int, Tuple[int, int]],
        priorities: Dict[int, float],
    ) -> Dict[int, str]:
        v_c: Set[Any] = set()
        e_c: Set[Any] = set()
        result: Dict[int, str] = {}
        
        sorted_agents = sorted(active_agents, key=lambda aid: -priorities.get(aid, 0.0))

        for aid in sorted_agents:
            path = self.space_time_astar(starts[aid], goals[aid], v_c, e_c, max_expansions=200)
            result[aid] = path[0] if path else "S"
            
            # Reservation: Đăng ký chiếm dụng không gian - thời gian cho Agent này
            cur = starts[aid]
            for step, act in enumerate(path):
                dr, dc = DIR_DELTAS[act]
                nxt = (cur[0] + dr, cur[1] + dc)
                v_c.add((nxt[0], nxt[1], step + 1))
                # Block cả 2 chiều để tránh Swap Conflict (Đâm xuyên qua nhau)
                e_c.add((cur, nxt, step + 1))
                e_c.add((nxt, cur, step + 1)) 
                cur = nxt

        return result

from typing import List, Dict, Tuple, Set

class VRPRouteEvaluator:
    def __init__(self, solver):
        self.solver = solver
        self.INF = 10**8

    def find_best_route(
        self, 
        sh: Shipper, 
        candidate_oids: List[int], 
        orders: Dict[int, Order], 
        current_time: int
    ) -> Tuple[float, List[Tuple[str, int]]]:
        best_profit = -self.INF
        best_route: List[Tuple[str, int]] = []

        initial_bag = set(sh.bag)
        initial_weight = sum(orders[oid].w for oid in initial_bag if oid in orders)
        
        if len(candidate_oids) > 6:
            def _score_candidate(oid: int) -> float:
                o = orders.get(oid)
                if not o: return -self.INF
                # Heuristic: Giá trị / Tổng khoảng cách (đến bốc hàng + đi giao)
                dp = self.solver._dist((sh.r, sh.c), (o.sx, o.sy))
                dd = self.solver._dist((o.sx, o.sy), (o.ex, o.ey))
                return (o.p * o.w) / (dp + dd + 1.0)
            
            candidate_oids = sorted(candidate_oids, key=_score_candidate, reverse=True)[:4]

        unpicked_set = set(candidate_oids)
        if not initial_bag and not unpicked_set:
            return 0.0, []

        MAX_DFS_NODES = 2000
        nodes_visited = 0
        
        # STATE MEMOIZATION (Dominance Check)
        # Lưu dạng: { (pos, tuple_bag, tuple_unpicked) : (best_time, best_accumulated_profit) }
        visited_states = {}

        def dfs(
            curr_pos: Tuple[int, int], 
            curr_time: int, 
            curr_weight: float, 
            bag_set: Set[int], 
            unpicked: Set[int], 
            route_seq: List[Tuple[str, int]], 
            accumulated_profit: float
        ):
            nonlocal best_profit, best_route, nodes_visited
            
            nodes_visited += 1
            if nodes_visited > MAX_DFS_NODES:
                return

            # Đánh giá Profit nếu túi đã rỗng (Agent có thể kết thúc lộ trình ở đây)
            if not bag_set:
                if accumulated_profit > best_profit:
                    best_profit = accumulated_profit
                    best_route = list(route_seq)

            # Cắt tỉa: Vượt quá Runtime Horizon của Simulator
            if curr_time >= self.solver.T:
                return

            # Cắt tỉa: Depth Limit (Chỉ Lookahead tối đa 5 actions)
            # Lý do: Lập kế hoạch quá xa là vô nghĩa do kẹt đường và đơn mới liên tục nổ
            if len(route_seq) >= 8:
                if accumulated_profit > best_profit:
                    best_profit = accumulated_profit
                    best_route = list(route_seq)
                return

            # Cắt tỉa: Dominance State Memoization
            # Chuyển set -> tuple để băm (hashing) tốc độ cao
            state_key = (curr_pos, tuple(sorted(bag_set)), tuple(sorted(unpicked)))
            if state_key in visited_states:
                prev_time, prev_profit = visited_states[state_key]
                # Nếu đến cùng 1 state nhưng mất nhiều thời gian hơn VÀ tiền ít hơn -> Cắt nhánh (Branch & Bound)
                if curr_time >= prev_time and accumulated_profit <= prev_profit:
                    return
            visited_states[state_key] = (curr_time, accumulated_profit)

            # NHÁNH 1: Thử GIAO HÀNG (Deliver)
            for oid in list(bag_set):
                o = orders.get(oid)
                if not o: continue
                
                dist = self.solver._dist(curr_pos, (o.ex, o.ey))
                if dist < self.INF:
                    finish_time = curr_time + dist
                    if finish_time < self.solver.T:
                        reward = self.solver._expected_reward(o, finish_time)
                        profit_delta = reward - (0.15 * dist) 
                        
                        bag_set.remove(oid)
                        route_seq.append(("D", oid))
                        dfs((o.ex, o.ey), finish_time, curr_weight - o.w, bag_set, unpicked, route_seq, accumulated_profit + profit_delta)
                        route_seq.pop()
                        bag_set.add(oid)

            # NHÁNH 2: Thử LẤY HÀNG MỚI (Pickup)
            if len(bag_set) < sh.K_max:
                for oid in list(unpicked):
                    o = orders.get(oid)
                    if not o: continue
                    
                    if curr_weight + o.w <= sh.W_max:
                        dist = self.solver._dist(curr_pos, (o.sx, o.sy))
                        if dist < self.INF:
                            finish_time = curr_time + dist
                            if finish_time < self.solver.T:
                                profit_delta = - (0.15 * dist)
                                
                                unpicked.remove(oid)
        
                                # Check xem có sẵn trong bag không để lúc lui không xóa nhầm
                                was_in_bag = oid in bag_set
                                if not was_in_bag:
                                    bag_set.add(oid)
                                    
                                route_seq.append(("P", oid))
                                dfs((o.sx, o.sy), finish_time, curr_weight + o.w, bag_set, unpicked, route_seq, accumulated_profit + profit_delta)
                                route_seq.pop()
                                
                                # Chỉ remove nếu lúc đầu nó không nằm trong bag
                                if not was_in_bag:
                                    bag_set.remove(oid)
                                    
                                unpicked.add(oid)

        dfs(
            curr_pos=(sh.r, sh.c),
            curr_time=current_time,
            curr_weight=initial_weight,
            bag_set=initial_bag,
            unpicked=unpicked_set,
            route_seq=[],
            accumulated_profit=0.0
        )

        return best_profit, best_route
    
class ACOSolver(Solver):
    METHOD_NAME = "ACO"

    def __init__(self, env: DeliveryEnv):
        self.env = env
        self.METHOD_NAME = "ACO"
        self.rng = random.Random(SEED + 77)
        self.vrp_evaluator = VRPRouteEvaluator(self)
        
        self._dist_maps: collections.OrderedDict = collections.OrderedDict()
        self.tau_start: Dict[int, Dict[int, float]] = collections.defaultdict(lambda: collections.defaultdict(lambda: 1.0))
        self.tau_edge: Dict[int, Dict[int, float]] = collections.defaultdict(lambda: collections.defaultdict(lambda: 1.0))
        self._seeded_oids: Set[int] = set()
        self.best_ever_plan: Dict[int, List[Order]] = {}




    def _configure(self, obs: dict) -> None:
        self.N = int(obs["N"])
        self.C = int(obs["C"])
        self.T = int(obs["T"])
        self.grid = obs["grid"]
        self.G = int(obs.get("G", getattr(self.env, "G", self.T)))

        # Cache limits
        if   self.N <= 20: self._bfs_cache_limit = 1500
        elif self.N <= 50: self._bfs_cache_limit = 2000
        else:              self._bfs_cache_limit = 3000
        self._dist_maps.clear()

        self.heatmap = [[0.0] * self.N for _ in range(self.N)]
        self.free_cells = [(r, c) for r in range(self.N) for c in range(self.N) if self.grid[r][c] == 0]

        self.topo = IterativeTopologyAnalyzer(self.grid, self.N)
        self.surge_detector = OnlineSurgeDetector(self.G, self.T, window=20)

        self.tau_start.clear()
        self.tau_edge.clear()
        self.best_ever_plan.clear()
        self.best_ever_score = -1.0
        self._seeded_oids.clear()

        self._state = {
            i: {
                "planned_route": [],
                "last_pos": (-1, -1),
                "stuck_ticks": 0,
                "evade_goal": None,
                "evade_ticks": 0,
                "blacklisted": {},
            } for i in range(self.C)
        }

        self.planner = AdaptivePlanner(self)

        # Re-calc Runtime parameters
        state_scale = self.N * self.C
        if   state_scale <= 100: self._ants, self._k_candidates = 80, 25
        elif state_scale <= 300: self._ants, self._k_candidates = 60, 30
        elif state_scale <= 800: self._ants, self._k_candidates = 40, 35
        else:                    self._ants, self._k_candidates = 28, 40

        scale = min(2.0, max(0.6, state_scale / 40.0))
        self._aco_ms = max(80.0, min(600.0, _BASE_ACO_MS * scale))
        
        self.URGENT_THRESHOLD = max(10, int(self.N * 1.4))

        if   self.N <= 12: self._replan_period = 4
        elif self.N <= 20: self._replan_period = 6
        elif self.N <= 50: self._replan_period = 8
        else:              self._replan_period = 10
        
        self._aco_ms = max(100.0, min(self._aco_ms, 300.0)) # Ép chạy ACO ít nhất 100ms - 300ms mỗi tick
        self._endgame_w = max(20, min(400, self.T // 8))
        self.congestion_factor = 1.0 + (self.C / max(1.0, (self.N * self.N))) * 3.5

    # ------------------------------------------------------------------
    # LRU BFS Cache
    # ------------------------------------------------------------------

    def _bfs_flood(self, goal: Tuple[int, int]) -> List[List[int]]:
        N, grid = self.N, self.grid
        dist = [[-1] * N for _ in range(N)]
        gr, gc = goal
        if grid[gr][gc] == 1: return dist
        dist[gr][gc] = 0
        q: Deque[Tuple[int, int]] = deque([(gr, gc)])
        while q:
            r, c = q.popleft()
            d = dist[r][c]
            for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N and grid[nr][nc] == 0 and dist[nr][nc] < 0:
                    dist[nr][nc] = d + 1
                    q.append((nr, nc))
        return dist

    def _get_dist_map(self, goal: Tuple[int, int]) -> List[List[int]]:
        if goal in self._dist_maps:
            self._dist_maps.move_to_end(goal)
            return self._dist_maps[goal]
        if len(self._dist_maps) >= self._bfs_cache_limit:
            self._dist_maps.popitem(last=False)
        dm = self._bfs_flood(goal)
        self._dist_maps[goal] = dm
        return dm

    def _dist(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        if a == b: return 0
        dm = self._get_dist_map(b)
        d = dm[a[0]][a[1]]
        return d if d >= 0 else INF

    # ------------------------------------------------------------------
    # Staging & Heatmap
    # ------------------------------------------------------------------

    def _update_heatmap(self, new_oids: Set[int], orders: Dict[int, Order]) -> None:
        for row in self.heatmap:
            for i in range(self.N): row[i] *= HOTSPOT_DECAY
        for oid in new_oids:
            if oid in orders: self.heatmap[orders[oid].sx][orders[oid].sy] += 1.0

    def _get_staging_pos(
        self, sh_id: int, current_goals: Dict[int, Tuple[int, int]], current_pos: Tuple[int, int],
    ) -> Tuple[int, int]:
        
        if self.surge_detector.surge_level >= 1 and self.surge_detector.surge_center:
            sr, sc = self.surge_detector.surge_center
            if self.grid[sr][sc] == 0: return (sr, sc)

        # Nếu không có surge, tìm ô có heatmap cao nhất xung quanh (Radius = 10)
        cr, cc = current_pos
        best_pos, best_score = current_pos, -1.0
        
        for dr in range(-10, 11):
            for dc in range(-10, 11):
                nr, nc = cr + dr, cc + dc
                if 0 <= nr < self.N and 0 <= nc < self.N and self.grid[nr][nc] == 0:
                    sc = self.heatmap[nr][nc] - (abs(dr) + abs(dc)) * 0.1 # Trừ hao quãng đường
                    if sc > best_score:
                        best_score, best_pos = sc, (nr, nc)
                        
        return best_pos

    def _filter_candidates_top_k(self, candidates: List[Order], targetable: List[Shipper], orders: Dict[int, Order], t: int, k: int) -> List[Order]:
        if len(candidates) <= k: return candidates
        must = [o for o in candidates if o.p >= 2 and (o.et - t) < self.URGENT_THRESHOLD * 2]
        must_ids = {o.id for o in must}
        remaining_k = max(0, k - len(must))
        others = [o for o in candidates if o.id not in must_ids]
        
        if others and remaining_k > 0:
            probe = targetable[:3] if len(targetable) >= 3 else targetable
            scored = []
            for o in others:
                best = 0.0
                for sh in probe:
                    dp = self._dist((sh.r, sh.c), (o.sx, o.sy))
                    dd = self._dist((o.sx, o.sy), (o.ex, o.ey))
                    if dp + dd < INF:
                        sc = (_r_base(o.w) * o.p * (1.0 + 1.0 / max(o.et - t, 1))) / (dp + dd + 1)
                        if sc > best: best = sc
                scored.append((best, o))
            scored.sort(key=lambda x: -x[0])
            others = [o for _, o in scored[:remaining_k]]
        else:
            others = others[:remaining_k]
        return must + others

    def _best_delivery_target(self, sh: Shipper, orders: Dict[int, Order], t: int) -> int:
        if not sh.bag: return -1
        best_oid, best_key = -1, None
        for oid in sh.bag:
            if oid not in orders: continue
            o = orders[oid]
            dist = self._dist((sh.r, sh.c), (o.ex, o.ey))
            if dist >= INF: continue
            slack = (o.et - t) - dist
            key = (0, slack, -o.p) if slack >= 0 else (1, -o.p, o.et)
            if best_key is None or key < best_key:
                best_key, best_oid = key, oid
        return best_oid

    def _evaluate_opportunistic_pickup(
        self,
        sh: Shipper,
        primary_delivery: Order,
        cand_order: Order,
        t: int,
    ) -> bool:
        primary_drop = (primary_delivery.ex, primary_delivery.ey)
        cand_drop = (cand_order.ex, cand_order.ey)

        dist_to_primary = self._dist((sh.r, sh.c), primary_drop)
        dist_detour = self._dist(primary_drop, cand_drop)
        if dist_to_primary >= INF or dist_detour >= INF:
            return False

        t_finish = t + dist_to_primary + dist_detour
        if t_finish >= self.T:
            return False

        if primary_drop == cand_drop:
            return True

        marginal_revenue = self._expected_reward(cand_order, t_finish)
        if marginal_revenue <= 0:
            return False

        w_cur = _w_carried(sh, self.env.orders) if hasattr(self.env, "orders") else sum(o.w for o in sh.bag) 

        added_weight_ratio = cand_order.w / max(sh.W_max, 1.0)
        carrying_cost = 0.01 * added_weight_ratio * (dist_to_primary + dist_detour)
        
        detour_move_cost = 0.01 * (1.0 + (w_cur + cand_order.w) / max(sh.W_max, 1.0)) * dist_detour
        
        time_slack = cand_order.et - t_finish
        risk_penalty = 0.0 if time_slack >= 0 else abs(time_slack) * 0.35

        net_profit = marginal_revenue - carrying_cost - detour_move_cost - risk_penalty
        
        if net_profit <= 0:
            return False
        
        roi = net_profit / (dist_detour + 1.0)
        
        return roi >= 1.25
    
    def _reactive_resolver(
        self, 
        shippers: List[Shipper], 
        intended_moves: Dict[int, str], 
        goals: Dict[int, Tuple[int, int]], 
        priorities: Dict[int, float],
        orders: Dict[int, Order],
        t: int
    ) -> Dict[int, Tuple[str, int]]:
        old_pos = {sh.id: (sh.r, sh.c) for sh in shippers}
        desired_pos = {}
        
        for sh in shippers:
            move = intended_moves.get(sh.id, "S")
            dr, dc = DIR_DELTAS[move]
            nr, nc = sh.r + dr, sh.c + dc
            # Ràng buộc map cơ bản
            if 0 <= nr < self.N and 0 <= nc < self.N and self.grid[nr][nc] == 0:
                desired_pos[sh.id] = (nr, nc)
            else:
                desired_pos[sh.id] = old_pos[sh.id]

        actual_pos = {}
        occupied = set(old_pos.values())
        blocked_sids = []

        sorted_sids = sorted([sh.id for sh in shippers], key=lambda sid: -priorities.get(sid, 0.0))
        
        for sid in sorted_sids:
            curr = old_pos[sid]
            nxt = desired_pos[sid]
            occupied.discard(curr)
            
            if nxt in occupied: 
                nxt = curr
                blocked_sids.append(sid)
            
            occupied.add(nxt) # Chiếm vị trí mới
            actual_pos[sid] = nxt

        if blocked_sids:
            for sid in blocked_sids:
                curr = old_pos[sid]
                goal = goals.get(sid, curr)
                best_alt_move = "S"
                best_alt_pos = curr
                base_dist = self._dist(curr, goal)
                
                for move in ("U", "D", "L", "R"):
                    dr, dc = DIR_DELTAS[move]
                    nr, nc = curr[0] + dr, curr[1] + dc
                    
                    if 0 <= nr < self.N and 0 <= nc < self.N and self.grid[nr][nc] == 0:
                        alt_pos = (nr, nc)
                        if alt_pos not in occupied:
                            # Đánh giá: Có đưa ta lại gần đích không? (Hoặc ít nhất là không đi lùi)
                            alt_dist = self._dist(alt_pos, goal)
                            if alt_dist <= base_dist: 
                                best_alt_move = move
                                best_alt_pos = alt_pos
                                base_dist = alt_dist # Ưu tiên bước lách tối ưu nhất
                
                # Cập nhật kết quả lách
                if best_alt_pos != curr:
                    occupied.discard(curr)
                    occupied.add(best_alt_pos)
                    actual_pos[sid] = best_alt_pos
                    intended_moves[sid] = best_alt_move

        # 4. Operation Normalization (Chống lỗi phạt Invalid Operation)
        final_actions = {}
        already_claimed_pickups = set()
        
        for sh in shippers:
            st = self._state[sh.id]
            for act, oid in st["planned_route"]:
                if act == "P": already_claimed_pickups.add(oid)

        for sh in shippers:
            sid = sh.id
            st = self._state[sid]
            pos_after = actual_pos[sid]
            route = st["planned_route"]
            
            final_move = "S" if pos_after == old_pos[sid] else intended_moves[sid]
            op = 0
            has_target = len(route) > 0
            
            # A. Tới đích
            if has_target and pos_after == goals.get(sid):
                op = 1 if route[0][0] == "P" else 2
                
            # B. Opportunistic Hoovering
            elif (
                has_target and route[0][0] == "D" 
                and final_move != "S" 
                and len(sh.bag) < sh.K_max
                and _w_carried(sh, orders) < sh.W_max
            ):
                primary_delivery = orders.get(route[0][1])
                if primary_delivery:
                    for oid, o in orders.items():
                        if (not o.picked and not o.delivered and (o.sx, o.sy) == pos_after 
                            and oid not in already_claimed_pickups
                            and _w_carried(sh, orders) + o.w <= sh.W_max):
                            
                            if self._evaluate_opportunistic_pickup(sh, primary_delivery, o, t):
                                op = 1
                                already_claimed_pickups.add(oid)
                                break      
            # Check an toàn
            if op == 2:
                if not any((orders[oid].ex, orders[oid].ey) == pos_after for oid in sh.bag if oid in orders):
                    op = 0

            final_actions[sid] = (final_move, op)

        return final_actions

    # ------------------------------------------------------------------
    # Heuristics
    # ------------------------------------------------------------------
    def _expected_reward(self, o: Order, t_del: int) -> float:
        """Use env's exact delivery_reward — mirrors grader exactly."""
        return delivery_reward(o, t_del, self.T)
        
    def _eta_start(
        self, sh: Shipper, o: Order, orders: Dict[int, Order], t: int, 
        ownership_map: Optional[dict] = None, zone_load: Optional[Dict[int, int]] = None,
    ) -> float:
        dp = self._dist((sh.r, sh.c), (o.sx, o.sy))
        dd = self._dist((o.sx, o.sy), (o.ex, o.ey))
        total_dist = dp + dd

        # Pruning 1: Không thể vươn tới hoặc vượt quá thời gian mô phỏng
        if total_dist >= INF or t + dp >= self.T: 
            return 0.001

        est_travel_time = int((dp + dd) * self.congestion_factor)
        
        if est_travel_time >= INF or t + int(dp * self.congestion_factor) >= self.T: 
            return 0.001

        t_del = t + est_travel_time # Giao hàng trễ hơn dự kiến do kẹt
        
        exp_reward = self._expected_reward(o, t_del)

        w_cur = _w_carried(sh, orders)
        move_cost = 0.01 * (1.0 + (w_cur + o.w) / max(sh.W_max, 1.0)) * total_dist

        # 3. Opportunity Cost lên các đơn trong túi
        bag_penalty = 0.0
        if sh.bag:
            for oid in sh.bag:
                if oid in orders:
                    bag_o = orders[oid]
                    t_del_curr = t + self._dist((sh.r, sh.c), (bag_o.ex, bag_o.ey))
                    r_curr = self._expected_reward(bag_o, t_del_curr)
                    r_delayed = self._expected_reward(bag_o, t_del_curr + dp)
                    bag_penalty += max(0.0, r_curr - r_delayed)

        net_profit = exp_reward - move_cost - bag_penalty

        # Pruning 2: Đi làm không công hoặc lỗ -> Bỏ qua
        if net_profit <= 0:
            return 0.001 

        # Profit Density càng cao nếu tiền nhiều mà đi gần
        base_eta = net_profit / (total_dist + 1)

        if zone_load is not None:
            zone = self.topo.zone_id.get((o.sx, o.sy), -2)
            if zone >= 0:
                load = zone_load.get(zone, 0)
                zone_capacity = max(3, int(self.C * 0.15))
                if load >= zone_capacity:
                    base_eta *= 0.01
                elif load >= 2:
                    base_eta *= max(0.30, 1.0 - 0.20 * (load - 1))

        if ownership_map:
            owner = ownership_map.get(o.id, -1)
            base_eta *= 1.5 if owner == sh.id else (0.2 if o.p < 3 else 1.0)

        base_eta *= self.surge_detector.proximity_boost((o.sx, o.sy), self.N)
        return max(0.001, base_eta)

    def _eta_edge(self, sh: Shipper, o1: Order, o2: Order, orders: Dict[int, Order], t: int) -> float:
        d_spatial = self._dist((o1.sx, o1.sy), (o2.sx, o2.sy))
        
        # Pruning 1: Hai đơn ở quá xa nhau, không đáng để gom (Batching)
        if d_spatial > BATCH_RADIUS: 
            return 0.001
            
        dd2 = self._dist((o2.sx, o2.sy), (o2.ex, o2.ey))
        if d_spatial + dd2 >= INF:
            return 0.001

        # Ước tính thời gian giao o2 (Giả định lộ trình: Current -> Lấy o1 -> Lấy o2 -> Giao o2)
        t_del2 = t + self._dist((sh.r, sh.c), (o1.sx, o1.sy)) + d_spatial + dd2
        if t_del2 >= self.T:
            return 0.001

        # 1. Doanh thu dự kiến
        exp_reward = self._expected_reward(o2, t_del2)

        # 2. Chi phí di chuyển phát sinh (Phải gánh cả tạ của o1 và o2)
        w_cur = _w_carried(sh, orders)
        move_cost = 0.01 * (1.0 + (w_cur + o1.w + o2.w) / max(sh.W_max, 1.0)) * (d_spatial + dd2)

        # 3. Lợi Nhuận Ròng
        net_profit = exp_reward - move_cost
        
        # Pruning 2: Gom đơn này làm hệ thống lỗ
        if net_profit <= 0:
            return 0.001

        return net_profit / (d_spatial + 1)

    # ------------------------------------------------------------------
    # Pheromones & ACO Assignment Loop
    # ------------------------------------------------------------------

    def _apply_pheromone_fusion(self, new_oids: Set[int], pending_ids: Set[int], shippers: List[Shipper], orders: Dict[int, Order], t: int) -> None:
        for sh_id in self.tau_start:
            for oid in self.tau_start[sh_id]: self.tau_start[sh_id][oid] *= 0.7
        for o1 in self.tau_edge:
            for o2 in self.tau_edge[o1]: self.tau_edge[o1][o2] *= 0.7
        claimed: Set[int] = set()
        for sh in sorted(shippers, key=lambda s: -s.K_max):
            best_o, best_sc = None, -1.0
            for oid in new_oids:
                if oid in claimed or oid not in orders: continue
                o = orders[oid]
                if not _can_carry(sh, o, orders): continue
                dp, dd = self._dist((sh.r, sh.c), (o.sx, o.sy)), self._dist((o.sx, o.sy), (o.ex, o.ey))
                if dp + dd < INF:
                    sc = (_r_base(o.w) * o.p) / (dp + dd + 1)
                    if sc > best_sc: best_sc, best_o = sc, o
            if best_o:
                claimed.add(best_o.id)
                self.tau_start[sh.id][best_o.id] = min(TAU_MAX, self.tau_start[sh.id][best_o.id] + ACO_Q * 2.0)

    def _claimed_ids(self, shippers: List[Shipper]) -> Set[int]:
        claimed = set()
        for sh in shippers:
            st = self._state[sh.id]
            for act, oid in st["planned_route"]:
                if act == "P": claimed.add(oid)
        return claimed

    def _urgency_emergency_assign(self, shippers: List[Shipper], orders: Dict[int, Order], pending_ids: Set[int], t: int) -> None:
        threshold = {
        3: self.URGENT_THRESHOLD * 2.0, 
        2: self.URGENT_THRESHOLD * 1.5, 
        1: self.URGENT_THRESHOLD * 0.8
        }
        # Safe list comprehension
        urgent = []
        for oid in pending_ids:
            if oid in orders:
                o = orders[oid]
                # Use .get() with a default fallback (e.g., multiplier 1.0) if 'p' is anomalous
                thresh_val = threshold.get(o.p, self.URGENT_THRESHOLD * 1.0) 
                if (o.et - t) < thresh_val:
                    urgent.append(o)
                    
        urgent.sort(key=lambda o: (o.et - t, -o.p))
        if not urgent: return
        claimed = self._claimed_ids(shippers)
        for o in urgent:
            if o.id in claimed: continue
            best_sh, best_cost, dist_del = None, INF, self._dist((o.sx, o.sy), (o.ex, o.ey))
            for sh in shippers:
                if not _can_carry(sh, o, orders): continue
                dp = self._dist((sh.r, sh.c), (o.sx, o.sy))
                if dp + dist_del >= INF: continue
                # Penalty nếu route đang dài
                cost = dp + len(self._state[sh.id]["planned_route"]) * 2
                if cost < best_cost: best_cost, best_sh = cost, sh
            
            if best_sh:
                claimed.add(o.id)
                st = self._state[best_sh.id]
                if not st["planned_route"]:
                    self._add_to_route(best_sh, [o.id], orders, t)
                else:
                    act, cur_oid = st["planned_route"][0]
                    if act == "P":
                        cur_p = orders[cur_oid].p if cur_oid in orders else 0
                        if o.p > cur_p or (o.p == cur_p and (o.et - t) < (orders[cur_oid].et - t if cur_oid in orders else INF)):
                            self._add_to_route(best_sh, [o.id], orders, t)

    def _add_to_route(self, sh: Shipper, new_oids: List[int], orders: Dict[int, Order], t: int) -> None:
        st = self._state[sh.id]
        
        current_unpicked = [
            oid for act, oid in st["planned_route"] 
            if act == "P" and oid in orders and not orders[oid].picked
        ]
        
        # gop với đơn hàng mới để lập lộ trình tối ưu nhất
        cands = list(set(current_unpicked + new_oids))
        
        _, best_route = self.vrp_evaluator.find_best_route(sh, cands, orders, t)
        
        if best_route:
            st["planned_route"] = best_route
        else:
            for oid in new_oids:
                if ("P", oid) not in st["planned_route"]:
                    st["planned_route"].append(("P", oid))

    def _run_aco_assignment(
        self, 
        shippers: List[Shipper], 
        orders: Dict[int, Order], 
        pending_ids: Set[int], 
        t: int
    ) -> None:
        """
        Kiến trúc mới: ACO chỉ đóng vai trò Meta-Heuristic Assigner.
        Tìm ra tổ hợp Agent <-> Orders tối ưu, sau đó pass xuống cho VRP Evaluator chốt Route.
        """
        # xác định các agent còn khả năng nhận thêm đơn
        targetable = []
        for sh in shippers:
            planned_p_count = sum(1 for act, _ in self._state[sh.id]["planned_route"] if act == "P")
            if len(sh.bag) + planned_p_count < sh.K_max and _w_carried(sh, orders) < sh.W_max:
                targetable.append(sh)
                
        if not targetable or not pending_ids: 
            return

        # Lọc bỏ các đơn đã có chủ
        claimed = self._claimed_ids(shippers)
        raw_cands = [orders[oid] for oid in pending_ids if oid not in claimed and oid in orders]
        if not raw_cands: 
            return
            
        # top-k candidates để giới hạn không gian tìm kiếm của ant
        candidates = self._filter_candidates_top_k(raw_cands, targetable, orders, t, k=self._k_candidates)

        # xây dựng bản đồ phân luồng chống tắc nghẽn Zone Load
        zone_load: Dict[int, int] = collections.defaultdict(int)
        for sh in shippers:
            st = self._state[sh.id]
            if st["planned_route"]:
                act, oid = st["planned_route"][0]
                if act == "P" and oid in orders:
                    z = self.topo.zone_id.get((orders[oid].sx, orders[oid].sy), -2)
                    if z >= 0: 
                        zone_load[z] += 1

        # Future Capacity Weighting Value - Phạt agent ôm quá nhiều việc
        ownership_map = {}
        load_pen = {
            sh.id: 1.0 + 1.5 * ((len(sh.bag) + sum(1 for act, _ in self._state[sh.id]["planned_route"] if act == "P")) / max(sh.K_max, 1)) 
            for sh in targetable
        }
        for o in candidates:
            best_c, owner = INF, -1
            for sh in targetable:
                d = self._dist((sh.r, sh.c), (o.sx, o.sy))
                if d < INF and d * load_pen[sh.id] < best_c: 
                    best_c, owner = d * load_pen[sh.id], sh.id
            ownership_map[o.id] = owner

        # cache điểm heuristic tĩnh trước khi chạy ACO Loop
        eta_s = {
            (sh.id, o.id): self._eta_start(sh, o, orders, t, ownership_map, zone_load) 
            for sh in targetable for o in candidates 
            if _can_carry(sh, o, orders) and o.id not in self._state[sh.id]["blacklisted"]
        }
        
        eta_e = {
            (sh.id, o1.id, o2.id): self._eta_edge(sh, o1, o2, orders, t) 
            for sh in targetable if sh.K_max - len(sh.bag) >= 2 
            for o1 in candidates for o2 in candidates if o1.id != o2.id
        }
        
        unseeded = pending_ids - self._seeded_oids
        if unseeded:
            self._apply_pheromone_fusion(unseeded, pending_ids, shippers, orders, t)
            self._seeded_oids.update(unseeded)

        all_iter_best, stagnate_count, prev_best_score, t0 = [], 0, -1e9, time.time()
        
        while (time.time() - t0) * 1000 < self._aco_ms:
            iter_plans = []
            
            for _ in range(self._ants):
                used, plan, score = set(), {}, 0.0
                sh_list = list(targetable)
                self.rng.shuffle(sh_list)

                for sh in sh_list:
                    choices, probs = [], []
                    for o in candidates:
                        if o.id in used or eta_s.get((sh.id, o.id), 0.0) <= 0.001: 
                            continue
                        probs.append((max(self.tau_start[sh.id][o.id], TAU_MIN) ** ACO_ALPHA) * (eta_s[(sh.id, o.id)] ** ACO_BETA))
                        choices.append(o)
                        
                    if not choices: 
                        continue
                        
                    # Kiến chọn điểm xuất phát (o1)
                    o1 = self.rng.choices(choices, weights=probs, k=1)[0] if sum(probs) > 0 else self.rng.choice(choices)
                    used.add(o1.id)
                    plan[sh.id] = [o1]
                    score += eta_s[(sh.id, o1.id)]

                    # Kiến cân nhắc bốc thêm đơn thứ hai ghép cùng (o2)
                    planned_p_count = sum(1 for act, _ in self._state[sh.id]["planned_route"] if act == "P")
                    if sh.K_max - (len(sh.bag) + planned_p_count) >= 2 and _w_carried(sh, orders) + o1.w < sh.W_max:
                        choices2, probs2 = [], []
                        for o2 in candidates:
                            if o2.id in used or _w_carried(sh, orders) + o1.w + o2.w > sh.W_max or eta_e.get((sh.id, o1.id, o2.id), 0.0) <= 0.001: 
                                continue
                            probs2.append((max(self.tau_edge[o1.id][o2.id], TAU_MIN) ** ACO_ALPHA) * (eta_e[(sh.id, o1.id, o2.id)] ** ACO_BETA))
                            choices2.append(o2)
                            
                        if choices2:
                            o2 = self.rng.choices(choices2, weights=probs2, k=1)[0] if sum(probs2) > 0 else self.rng.choice(choices2)
                            used.add(o2.id)
                            plan[sh.id].append(o2)
                            score += eta_e[(sh.id, o1.id, o2.id)]
                            
                if plan: 
                    iter_plans.append((score, plan))

            if not iter_plans: 
                continue
            
            # evaporation
            evap = min(0.50, ACO_RHO + THETA_EVAP / max(sum(s for s, _ in iter_plans) / len(iter_plans), 1.0))
            for sh in targetable:
                for o in candidates: 
                    self.tau_start[sh.id][o.id] = max(TAU_MIN, self.tau_start[sh.id][o.id] * (1.0 - evap))
            for o1 in candidates:
                for o2 in candidates:
                    if o1.id != o2.id: 
                        self.tau_edge[o1.id][o2.id] = max(TAU_MIN, self.tau_edge[o1.id][o2.id] * (1.0 - evap))

            # Elitist Update
            iter_plans.sort(key=lambda x: -x[0])
            targetable_ids = {sh.id for sh in targetable}
            for rank, (_, ant_plan) in enumerate(iter_plans[:SIGMA_ELITIST-1], start=1):
                dep = ACO_Q * (SIGMA_ELITIST - rank)
                for sh_id, seq in ant_plan.items():
                    self.tau_start[sh_id][seq[0].id] = min(TAU_MAX, self.tau_start[sh_id][seq[0].id] + dep)
                    if len(seq) > 1: 
                        self.tau_edge[seq[0].id][seq[1].id] = min(TAU_MAX, self.tau_edge[seq[0].id][seq[1].id] + dep)

            if iter_plans[0][0] > self.best_ever_score:
                self.best_ever_score, self.best_ever_plan = iter_plans[0][0], iter_plans[0][1]

            if self.best_ever_plan:
                c_ids, el_dep = {o.id for o in candidates}, ACO_Q * SIGMA_ELITIST
                for sh_id, seq in self.best_ever_plan.items():
                    if not seq or seq[0].id not in c_ids or sh_id not in targetable_ids: 
                        continue
                    self.tau_start[sh_id][seq[0].id] = min(TAU_MAX, self.tau_start[sh_id][seq[0].id] + el_dep)
                    if len(seq) > 1 and seq[1].id in c_ids: 
                        self.tau_edge[seq[0].id][seq[1].id] = min(TAU_MAX, self.tau_edge[seq[0].id][seq[1].id] + el_dep)

            all_iter_best.append(iter_plans[0])
            
            # Early stopping nếu quá trình ACO đi vào bế tắc không cải thiện điểm
            if iter_plans[0][0] <= prev_best_score + 1e-4:
                stagnate_count += 1
                if stagnate_count >= _ACO_STAGNATE_LIMIT: 
                    break
            else:
                stagnate_count, prev_best_score = 0, iter_plans[0][0]

        # cho VRP định tuyến
        if not all_iter_best: 
            return
            
        best_overall_plan = max(all_iter_best, key=lambda x: x[0])[1]
        for sh_id, seq in best_overall_plan.items():
            sh = next((s for s in targetable if s.id == sh_id), None)
            if sh:
                assigned_oids = [o.id for o in seq]
                self._add_to_route(sh, assigned_oids, orders, t)

    def _greedy_fallback(self, shippers: List[Shipper], orders: Dict[int, Order], pending_ids: Set[int], t: int) -> None:
        claimed = self._claimed_ids(shippers)
        for sh in shippers:
            st = self._state[sh.id]
            if st["planned_route"]: continue
            
            best_o, best_sc = None, -1.0
            for oid in pending_ids:
                if oid in claimed or oid not in orders or oid in st["blacklisted"]: continue
                o = orders[oid]
                if not _can_carry(sh, o, orders): continue
                dp = self._dist((sh.r, sh.c), (o.sx, o.sy))
                dd = self._dist((o.sx, o.sy), (o.ex, o.ey))
                if dp + dd >= INF: continue
                sc = (_r_base(o.w) * o.p) / (dp + 1)
                if sc > best_sc: best_sc, best_o = sc, o
            
            if best_o:
                claimed.add(best_o.id)
                self._add_to_route(sh, [best_o.id], orders, t)

    def _endgame_assign(
        self,
        shippers: List[Shipper],
        orders:   Dict[int, Order],
        pending_ids: Set[int],
        t: int,
    ) -> None:
        if t < self.T - self._endgame_w: return
            
        claimed = self._claimed_ids(shippers)
        for sh in shippers:
            st = self._state[sh.id]
            if st["planned_route"] or sh.bag: continue  
                
            best_o, best_sc = None, -1.0
            for oid in pending_ids:
                if oid in claimed or oid not in orders or oid in st["blacklisted"]:
                    continue
                o = orders[oid]
                if not _can_carry(sh, o, orders):
                    continue
                dp = self._dist((sh.r, sh.c), (o.sx, o.sy))
                dd = self._dist((o.sx, o.sy), (o.ex, o.ey))
                
                if dp + dd >= INF or t + dp >= self.T:
                    continue
                    
                t_del = t + dp + dd
                reward = self._expected_reward(o, t_del)
                if reward <= 0.0:
                    continue
                    
                sc = reward / (dp + dd + 1.0)
                if sc > best_sc:
                    best_sc, best_o = sc, o
                    
            if best_o is not None:
                claimed.add(best_o.id)
                self._add_to_route(sh, [best_o.id], orders, t)

    # ------------------------------------------------------------------
    # State Machine & Runner
    # ------------------------------------------------------------------

    def _update_states(self, shippers: List[Shipper], orders: Dict[int, Order], pending_ids: Set[int], t: int) -> None:
        for sh in shippers:
            st = self._state[sh.id]
            
            # Clear blacklisted orders
            for oid in [k for k, v in list(st["blacklisted"].items()) if t >= v]: 
                del st["blacklisted"][oid]
                
            # Evade logic
            if st["evade_ticks"] > 0:
                st["evade_ticks"] -= 1
                if st["evade_ticks"] == 0: st["evade_goal"] = None

            # Detect agent bị kẹt
            is_active = len(st["planned_route"]) > 0
            if (sh.r, sh.c) == st["last_pos"] and is_active: 
                st["stuck_ticks"] += 1
            else: 
                st["stuck_ticks"] = 0
            st["last_pos"] = (sh.r, sh.c)

            if st["stuck_ticks"] >= 4:
                if st["planned_route"]:
                    act, oid = st["planned_route"][0]
                    if act == "P": st["blacklisted"][oid] = t + 5
                # Bơm thêm tính random cho Evade Goal để phá vỡ các khối Deadlock diện rộng
                st["planned_route"].clear()
                st["stuck_ticks"] = 0 
                # Chọn điểm evade đưa agent khỏi vùng hotspot
                st["evade_goal"] = self.rng.choice(self.free_cells) if self.free_cells else (sh.r, sh.c)
                st["evade_ticks"] = 4
                continue

            # Phase Transitions xử lý queue lộ trình
            valid_route = []
            for act, oid in st["planned_route"]:
                o = orders.get(oid)
                if not o: continue
                # Nếu định Pick nhưng hàng đã bị lấy mất, hoặc không còn trong pending
                if act == "P" and (o.picked or oid not in pending_ids): continue
                # Nếu định Drop nhưng hàng đã giao xong
                if act == "D" and o.delivered: continue
                valid_route.append((act, oid))
            st["planned_route"] = valid_route

            # DYNAMIC BUNDLING
            if st["planned_route"]:
                local_oids = [oid for oid in pending_ids if orders[oid].sx == sh.r and orders[oid].sy == sh.c]
                for passing_oid in local_oids:
                    passing_o = orders[passing_oid]
                    if _can_carry(sh, passing_o, orders):
                        # kiểm tra chặt chẽ not orders[o].picked
                        cands = [
                            o for a, o in st["planned_route"] 
                            if a == "P" and o in orders and not orders[o].picked
                        ] + [passing_oid]
                        
                        _, new_route = self.vrp_evaluator.find_best_route(sh, cands, orders, t)
                        if new_route:
                            st["planned_route"] = new_route
                            pending_ids.remove(passing_oid)
                            break

            # Nếu có hàng trong túi nhưng route trống (VRP Drop)
            if not st["planned_route"] and sh.bag:
                best_del = self._best_delivery_target(sh, orders, t)
                if best_del >= 0:
                    st["planned_route"].append(("D", best_del))

    def _gc_pheromones(self, delivered_oids: Set[int]) -> None:
        for oid in delivered_oids:
            self._seeded_oids.discard(oid)
            for sid in list(self.tau_start): self.tau_start[sid].pop(oid, None)
            self.tau_edge.pop(oid, None)
            for key in list(self.tau_edge): self.tau_edge[key].pop(oid, None)
        if self.best_ever_plan:
            for sh_id in list(self.best_ever_plan):
                self.best_ever_plan[sh_id] = [o for o in self.best_ever_plan[sh_id] if o.id not in delivered_oids]
                if not self.best_ever_plan[sh_id]: del self.best_ever_plan[sh_id]

    def _generate_actions(self, shippers: List[Shipper], orders: Dict[int, Order], t: int) -> Dict[int, Tuple[str, Any]]:
        starts, goals, active_agents = {}, {}, []
        priorities: Dict[int, float] = {}

        for sh in shippers:
            starts[sh.id] = (sh.r, sh.c)
            st = self._state[sh.id]
            
            if st.get("evade_goal"):
                goals[sh.id] = st["evade_goal"]
                priorities[sh.id] = -10.0
            elif st["planned_route"]:
                next_action, next_oid = st["planned_route"][0]
                o = orders.get(next_oid)
                
                if not o: 
                    st["planned_route"].pop(0)
                    goals[sh.id] = self._get_staging_pos(sh.id, goals, (sh.r, sh.c))
                    priorities[sh.id] = 0.0
                elif next_action == "D":
                    goals[sh.id] = (o.ex, o.ey)
                    dist = self._dist((sh.r, sh.c), goals[sh.id])
                    slack = o.et - (t + dist) if dist < INF else 0
                    priorities[sh.id] = 1000.0 + o.p * 15.0 - slack * 0.2
                else:
                    goals[sh.id] = (o.sx, o.sy)
                    dist = self._dist((sh.r, sh.c), goals[sh.id])
                    slack = o.et - (t + dist) if dist < INF else 0
                    priorities[sh.id] = 500.0 + o.p * 5.0 - slack * 0.1
            else:
                goals[sh.id] = self._get_staging_pos(sh.id, goals, (sh.r, sh.c))
                priorities[sh.id] = 0.0

            active_agents.append(sh.id)

        intended_moves = self.planner.plan(active_agents, starts, goals, priorities)
        return self._reactive_resolver(shippers, intended_moves, goals, priorities, orders, t)
    
    def run(self) -> dict:
        start_time = time.time()
        try:
            obs = self.env.reset()
            self._configure(obs)
            
            pending_ids, prev_orders = set(), {}

            while not obs.get("done", False):
                t, orders, shippers = obs["t"], obs["orders"], obs["shippers"]
                new_oids = set(obs.get("new_order_ids", []))

                if new_oids:
                    self.surge_detector.update(t, new_oids, orders)
                    self._update_heatmap(new_oids, orders)

                for oid in new_oids:
                    if oid in orders and not orders[oid].picked: pending_ids.add(oid)

                pending_ids = {oid for oid in pending_ids if oid in orders and not orders[oid].picked}
                delivered_now = {oid for oid in prev_orders if oid not in orders}
                if delivered_now: self._gc_pheromones(delivered_now)

                self._update_states(shippers, orders, pending_ids, t)
                self._urgency_emergency_assign(shippers, orders, pending_ids, t)

                idle_shippers = [sh for sh in shippers if not self._state[sh.id]["planned_route"] and not sh.bag]
                period = max(2, self._replan_period // 2) if self.surge_detector.surge_level >= 2 or len(idle_shippers) > self.C // 2 else self._replan_period

                if pending_ids and (new_oids or idle_shippers or t % period == 0):
                    self._run_aco_assignment(shippers, orders, pending_ids, t)
                    self._greedy_fallback(shippers, orders, pending_ids, t)
                    
                self._endgame_assign(shippers, orders, pending_ids, t)

                actions = self._generate_actions(shippers, orders, t)
                prev_orders = dict(orders)
                obs, _, done, _ = self.env.step(actions)
                if done: break

            return self.env.result(self.METHOD_NAME, elapsed_sec=time.time() - start_time)

        except Exception as e:
            error_trace = traceback.format_exc()
            detailed_error = f"CRASH POINT: {type(e).__name__} - {str(e)} | TRACE: {error_trace}"
            raise RuntimeError(detailed_error)
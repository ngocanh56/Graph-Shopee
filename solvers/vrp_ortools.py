from __future__ import annotations

import time
import traceback
import random
from typing import List, Tuple, Dict
from collections import deque

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env import DeliveryEnv, Order, Shipper
from solvers.solver import Solver, default_result

# CẢI TIẾN 1: Dùng deque thay cho list pop(0) để tăng tốc độ tìm đường ở Map siêu lớn
def bfs_path(grid: List[List[int]], start: Tuple[int, int], goal: Tuple[int, int]) -> List[str]:
    if start == goal:
        return []

    N = len(grid)
    queue = deque([(start[0], start[1], [])])
    visited = {start}

    dirs = {
        "U": (-1, 0),
        "D": (1, 0),
        "L": (0, -1),
        "R": (0, 1),
    }

    while queue:
        r, c, path = queue.popleft()

        if (r, c) == goal:
            return path

        for move, (dr, dc) in dirs.items():
            nr, nc = r + dr, c + dc

            if (
                0 <= nr < N
                and 0 <= nc < N
                and grid[nr][nc] == 0
                and (nr, nc) not in visited
            ):
                visited.add((nr, nc))
                queue.append((nr, nc, path + [move]))

    return []


class VRPOrToolsSolver(Solver):
    """
    Pure VRP insertion heuristic
    Objective:
        maximize total reward
    implemented as:
        minimize (-profit)
    """

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)

        self.assigned_orders = set()
        self.rng = random.Random(42) # CẢI TIẾN 2: Thêm bộ tạo ngẫu nhiên để né tránh

        self.dist_cache: Dict[
            Tuple[Tuple[int, int], Tuple[int, int]],
            int
        ] = {}

        self.plans: Dict[int, List[Tuple[str, int]]] = {
            i: [] for i in range(self.cfg["C"])
        }

        self.all_seen_orders: Dict[int, Order] = {}

    # ---------------------------------------------------
    # Distance
    # ---------------------------------------------------
    def get_dist(
        self,
        p1: Tuple[int, int],
        p2: Tuple[int, int]
    ) -> int:
        if p1 == p2:
            return 0

        pair = (p1, p2)

        if pair in self.dist_cache:
            return self.dist_cache[pair]

        path = bfs_path(self.grid, p1, p2)
        dist = len(path) if path else 9999

        self.dist_cache[(p1, p2)] = dist
        self.dist_cache[(p2, p1)] = dist

        return dist

    # ---------------------------------------------------
    # Simulate plan profit (Giữ nguyên logic cực tốt của bạn)
    # ---------------------------------------------------
    def simulate_plan_cost(
        self,
        shipper: Shipper,
        plan: List[Tuple[str, int]],
        start_t: int
    ) -> float:

        current_pos = (shipper.r, shipper.c)
        current_t = start_t

        current_w = sum(
            self.all_seen_orders[oid].w
            for oid in shipper.bag
            if oid in self.all_seen_orders
        )

        current_k = len(shipper.bag)
        simulated_bag = set(shipper.bag)
        total_profit = 0.0

        for task, oid in plan:
            if oid not in self.all_seen_orders:
                return 999999

            order = self.all_seen_orders[oid]

            if task == "pickup":
                target_pos = (order.sx, order.sy)
            else:
                target_pos = (order.ex, order.ey)

            dist = self.get_dist(current_pos, target_pos)

            if dist >= 9999:
                return 999999

            move_cost = 0.01 * dist
            total_profit -= move_cost
            current_t += dist + 1
            current_pos = target_pos

            if task == "pickup":
                if oid in simulated_bag:
                    return 999999
                simulated_bag.add(oid)
                current_w += order.w
                current_k += 1

                if current_k > shipper.K_max:
                    return 999999
                if current_w > shipper.W_max:
                    return 999999

            elif task == "deliver":
                if oid not in simulated_bag:
                    return 999999
                simulated_bag.remove(oid)
                current_w -= order.w
                current_k -= 1

                reward = 10 * order.p
                if current_t <= order.et:
                    bonus = (order.et - current_t) / max(order.et, 1)
                    reward *= (1 + bonus)
                else:
                    delay = current_t - order.et
                    decay = max(0.0, 1 - delay / self.cfg["T"])
                    reward *= 0.4 * decay

                total_profit += reward

        return -total_profit

    # ---------------------------------------------------
    # Insert order
    # ---------------------------------------------------
    def insert_order(
        self,
        order: Order,
        shippers: List[Shipper],
        current_t: int
    ) -> bool:

        best_cost_increase = 999999
        best_shipper_id = -1
        best_new_plan = []

        dynamic_limit = max(20, self.cfg["N"] * 2)

        for shipper in shippers:
            current_plan = self.plans[shipper.id]
            if len(current_plan) >= dynamic_limit:
                continue

            base_cost = self.simulate_plan_cost(shipper, current_plan, current_t)
            n = len(current_plan)

            if self.cfg["N"] <= 10: search_depth = min(n, 8)
            else: search_depth = min(n, 16)

            for i in range(search_depth + 1):
                for j in range(i, min(n + 1, i + 8)):
                    new_plan = (
                        current_plan[:i]
                        + [("pickup", order.id)]
                        + current_plan[i:j]
                        + [("deliver", order.id)]
                        + current_plan[j:]
                    )

                    new_cost = self.simulate_plan_cost(shipper, new_plan, current_t)
                    if new_cost >= 999999: continue

                    cost_increase = new_cost - base_cost
                    if i == 0: cost_increase -= 2

                    if cost_increase < best_cost_increase:
                        best_cost_increase = cost_increase
                        best_shipper_id = shipper.id
                        best_new_plan = new_plan

        if best_shipper_id != -1:
            self.plans[best_shipper_id] = best_new_plan
            self.assigned_orders.add(order.id)
            return True

        return False

    # ---------------------------------------------------
    # Main
    # ---------------------------------------------------
    def run(self) -> dict:
        start_time = time.time()

        try:
            obs = self.env.reset()
            done = False

            while not done:
                t = obs["t"]
                shippers = obs["shippers"]

                for oid, o in obs["orders"].items():
                    self.all_seen_orders[oid] = o

                if t > 0 and t % 5 == 0:
                    for s in shippers:
                        new_plan = []
                        for task, oid in self.plans[s.id]:
                            if task == "deliver" and oid in s.bag:
                                new_plan.append((task, oid))
                            else:
                                if oid in self.assigned_orders:
                                    self.assigned_orders.remove(oid)
                        self.plans[s.id] = new_plan

                active_unassigned = [
                    o for o in obs["orders"].values()
                    if o.carrier == -1 and o.id not in self.assigned_orders
                ]

                active_unassigned.sort(
                    key=lambda o: (-o.p, o.et, o.id)
                )

                for order in active_unassigned[:20]:
                    self.insert_order(order, shippers, t)

                # -------------------------
                # CẢI TIẾN 3: Execute plans VỚI HỆ THỐNG TRÁNH VA CHẠM
                # -------------------------
                actions = {}
                reserved = set()

                # Sắp xếp xe theo ID để xử lý đồng bộ
                for s in sorted(shippers, key=lambda x: x.id):
                    queue = self.plans[s.id]
                    planned_deliveries = [oid for task, oid in queue if task == "deliver"]

                    for oid in s.bag:
                        if oid not in planned_deliveries:
                            queue.insert(0, ("deliver", oid))

                    action_code = "S"
                    task_id = 0
                    target_pos = (s.r, s.c)

                    while queue:
                        task_type, oid = queue[0]
                        order = self.all_seen_orders.get(oid)

                        if not order:
                            queue.pop(0)
                            continue

                        if task_type == "pickup":
                            if oid in s.bag:
                                queue.pop(0)
                                continue

                            current_state = obs["orders"].get(oid)
                            if not current_state or current_state.carrier not in (-1, s.id):
                                queue.pop(0)
                                continue

                            target_pos = (order.sx, order.sy)
                            if (s.r, s.c) == target_pos:
                                action_code = "S"
                                task_id = 1
                                break
                            else:
                                path = bfs_path(self.grid, (s.r, s.c), target_pos)
                                action_code = path[0] if path else "S"
                                task_id = 0
                                break

                        else: # deliver
                            if oid not in s.bag:
                                queue.pop(0)
                                continue

                            target_pos = (order.ex, order.ey)
                            if (s.r, s.c) == target_pos:
                                action_code = "S"
                                task_id = 2
                                break
                            else:
                                path = bfs_path(self.grid, (s.r, s.c), target_pos)
                                action_code = path[0] if path else "S"
                                task_id = 0
                                break

                    # CHỐNG DEADLOCK: Kiểm tra và lách xe nếu ô định đi đã bị chiếm
                    nr, nc = s.r, s.c
                    if action_code == "U": nr -= 1
                    elif action_code == "D": nr += 1
                    elif action_code == "L": nc -= 1
                    elif action_code == "R": nc += 1

                    if (nr, nc) in reserved and action_code != "S":
                        best_move = "S"
                        best_dist = 9999
                        moves = [("U", (-1, 0)), ("D", (1, 0)), ("L", (0, -1)), ("R", (0, 1))]
                        self.rng.shuffle(moves)

                        for m, (dr, dc) in moves:
                            nnr, nnc = s.r + dr, s.c + dc
                            if (
                                0 <= nnr < self.cfg["N"] 
                                and 0 <= nnc < self.cfg["N"] 
                                and self.grid[nnr][nnc] == 0 
                                and (nnr, nnc) not in reserved
                            ):
                                d = self.get_dist((nnr, nnc), target_pos)
                                if d < best_dist:
                                    best_dist = d
                                    best_move = m
                        
                        action_code = best_move
                        nr, nc = s.r, s.c
                        if action_code == "U": nr -= 1
                        elif action_code == "D": nr += 1
                        elif action_code == "L": nc -= 1
                        elif action_code == "R": nc += 1

                    reserved.add((nr, nc))
                    actions[s.id] = (action_code, task_id)

                obs, _, done, _ = self.env.step(actions)

            return self.env.result(
                "VRPOrToolsSolver",
                time.time() - start_time
            )

        except Exception as e:
            print(f"\n[VRP ERROR at T={obs.get('t','?')}]: {e}")
            traceback.print_exc()
            return default_result("VRPOrToolsSolver", self.cfg, getattr(self, "orders", []))
"""
MAPDCBSSolver — Multi-Agent Pickup & Delivery với Conflict-Based Search (CBS).

Cấu trúc đúng chuẩn CBS (2 tầng):

  Tầng MAPD (gán nhiệm vụ):
      Mỗi shipper rảnh nhận 1 đơn (pickup -> deliver). Đây là phần
      "Pickup and Delivery": biến đơn hàng thành mục tiêu di chuyển cho agent.

  Tầng cao CBS (constraint tree):
      - Lập kế hoạch đường đi cho tất cả agent (độc lập) ở node gốc.
      - Tìm xung đột ĐẦU TIÊN giữa các đường đi (đụng ô / hoán đổi ô).
      - Tách node thành 2 con, mỗi con thêm 1 RÀNG BUỘC cấm một agent đi vào
        (ô, thời điểm) đang tranh chấp, rồi lập lại đường đi của riêng agent đó.
      - Mở rộng node có chi phí thấp nhất tới khi không còn xung đột.

  Tầng thấp (low-level):
      Space-time A* cho 1 agent, tôn trọng tập ràng buộc được truyền xuống.

CBS xử lý "xung đột đa tác tử" một cách tường minh qua cây ràng buộc — khác hẳn
prioritized planning. Để hợp với bài toán ONLINE (đơn xuất hiện dần, chạy T bước),
ta dùng CBS theo CỬA SỔ thời gian (windowed): chỉ giải xung đột trong WINDOW bước
kế tiếp rồi thực thi 1 bước, lặp lại ở bước sau.
"""

from __future__ import annotations

import heapq
import time
from typing import Dict, List, Optional, Tuple

from env import DeliveryEnv, Order, Shipper, manhattan
from solvers.solver import Solver

LARGE = 10_000
Cell = Tuple[int, int]


class MAPDCBSSolver(Solver):
    method_name = "MAPDCBSSolver"

    WINDOW = 10            # số bước nhìn trước để giải xung đột
    MAX_CBS_NODES = 50     # trần số node trong cây ràng buộc / 1 bước (chặn xác định)

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.N = self.cfg["N"]
        self.task: Dict[int, int] = {}          # shipper_id -> order_id đang đi nhặt (-1 nếu chưa)
        self._dmaps: Dict[Cell, List[List[int]]] = {}   # cache BFS từ goal

    # ==================================================================
    # Khoảng cách trên lưới (BFS flood từ goal, cache theo goal)
    # ==================================================================
    def _dist_map(self, goal: Cell) -> List[List[int]]:
        dm = self._dmaps.get(goal)
        if dm is not None:
            return dm
        N = self.N
        dist = [[-1] * N for _ in range(N)]
        gr, gc = goal
        if self.grid[gr][gc] == 0:
            dist[gr][gc] = 0
            q = [goal]
            head = 0
            while head < len(q):
                r, c = q[head]; head += 1
                base = dist[r][c]
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < N and 0 <= nc < N and self.grid[nr][nc] == 0 and dist[nr][nc] == -1:
                        dist[nr][nc] = base + 1
                        q.append((nr, nc))
        self._dmaps[goal] = dist
        return dist

    def _dist(self, a: Cell, goal: Cell) -> int:
        v = self._dist_map(goal)[a[0]][a[1]]
        return v if v >= 0 else LARGE

    def _neighbors(self, cell: Cell) -> List[Cell]:
        r, c = cell
        res = [cell]  # đứng yên (wait)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.N and 0 <= nc < self.N and self.grid[nr][nc] == 0:
                res.append((nr, nc))
        return res

    @staticmethod
    def _dir(a: Cell, b: Cell) -> str:
        dr, dc = b[0] - a[0], b[1] - a[1]
        if dr == -1: return "U"
        if dr == 1:  return "D"
        if dc == -1: return "L"
        if dc == 1:  return "R"
        return "S"

    # ==================================================================
    # TẦNG THẤP: space-time A* cho 1 agent với ràng buộc + chướng ngại tĩnh
    # ==================================================================
    def _low_level(self, start: Cell, goal: Cell,
                   vcons: set, econs: set, static: frozenset) -> List[Cell]:
        """
        Trả đường đi [start, ..., goal] (theo thời gian tương đối 0..L).
        vcons: tập (r, c, t)        — cấm Ở ô (r,c) tại thời điểm t.
        econs: tập (r, c, nr, nc, t) — cấm ĐI từ (r,c)->(nr,nc) tại thời điểm t.
        static: ô bị chiếm vĩnh viễn bởi agent đứng yên (idle).
        Nếu không tới được goal trong WINDOW bước, trả đường tiến gần goal nhất.
        """
        h0 = self._dist(start, goal)
        if h0 >= LARGE:
            return [start]
        W = self.WINDOW
        heap = [(h0, 0, 0, start, (start,))]
        seen = {(start, 0): 0}
        best_h, best_path = h0, (start,)

        while heap:
            f, g, t, cell, path = heapq.heappop(heap)
            if cell == goal:
                return list(path)
            hc = self._dist(cell, goal)
            if hc < best_h:
                best_h, best_path = hc, path
            if t >= W:
                continue
            nt = t + 1
            for ncell in self._neighbors(cell):
                if ncell in static and ncell != goal:
                    continue
                if (ncell[0], ncell[1], nt) in vcons:
                    continue
                if ncell != cell and (cell[0], cell[1], ncell[0], ncell[1], t) in econs:
                    continue
                if (ncell, nt) in seen:
                    continue
                nh = self._dist(ncell, goal)
                if nh >= LARGE:
                    continue
                seen[(ncell, nt)] = g + 1
                heapq.heappush(heap, (g + 1 + nh, g + 1, nt, ncell, path + (ncell,)))

        return list(best_path)

    # ==================================================================
    # Phát hiện xung đột ĐẦU TIÊN giữa các đường đi (căn theo thời gian)
    # ==================================================================
    def _first_conflict(self, paths: Dict[int, List[Cell]]):
        ids = list(paths)
        if len(ids) < 2:
            return None
        L = min(self.WINDOW + 1, max(len(paths[a]) for a in ids))

        def at(a: int, k: int) -> Cell:
            p = paths[a]
            return p[k] if k < len(p) else p[-1]

        for k in range(L):
            # Xung đột ô: 2 agent cùng 1 ô tại thời điểm k
            occ: Dict[Cell, int] = {}
            for a in ids:
                c = at(a, k)
                if c in occ:
                    return ("vertex", occ[c], a, c, k)
                occ[c] = a
            # Xung đột cạnh: 2 agent hoán đổi ô giữa k và k+1
            if k + 1 < L:
                mv = {a: (at(a, k), at(a, k + 1)) for a in ids}
                for i, a in enumerate(ids):
                    fa, ta = mv[a]
                    if fa == ta:
                        continue
                    for b in ids[i + 1:]:
                        fb, tb = mv[b]
                        if fa == tb and ta == fb:
                            return ("edge", a, b, (fa, ta), (fb, tb), k)
        return None

    # ==================================================================
    # TẦNG CAO: Conflict-Based Search trên cây ràng buộc
    # ==================================================================
    def _cbs(self, agents: List[Tuple[int, Cell, Cell]],
             static: frozenset) -> Dict[int, List[Cell]]:
        if not agents:
            return {}
        info = {aid: (s, g) for aid, s, g in agents}

        def plan(aid: int, cons) -> List[Cell]:
            s, g = info[aid]
            return self._low_level(s, g, cons[0], cons[1], static)

        # Node gốc: lập kế hoạch độc lập cho từng agent
        cons0 = {aid: (set(), set()) for aid, _, _ in agents}
        paths0 = {aid: plan(aid, cons0[aid]) for aid, _, _ in agents}
        cost0 = sum(len(p) for p in paths0.values())

        counter = 0
        heap = [(cost0, counter, cons0, paths0)]
        counter += 1
        incumbent = paths0
        nodes = 0

        while heap and nodes < self.MAX_CBS_NODES:
            cost, _, cons, paths = heapq.heappop(heap)
            nodes += 1
            incumbent = paths
            conflict = self._first_conflict(paths)
            if conflict is None:
                return paths  # đã hết xung đột

            # Tách 2 nhánh: mỗi nhánh ràng buộc 1 trong 2 agent tranh chấp
            if conflict[0] == "vertex":
                _, a1, a2, cell, k = conflict
                branches = [(a1, ("v", cell, k)), (a2, ("v", cell, k))]
            else:
                _, a1, a2, m1, m2, k = conflict
                branches = [(a1, ("e", m1[0], m1[1], k)), (a2, ("e", m2[0], m2[1], k))]

            for aid, ct in branches:
                if aid not in info:
                    continue
                new_cons = {k2: (set(v), set(e)) for k2, (v, e) in cons.items()}
                v, e = new_cons[aid]
                if ct[0] == "v":
                    cell, k = ct[1], ct[2]
                    v.add((cell[0], cell[1], k))
                else:
                    frm, to, k = ct[1], ct[2], ct[3]
                    e.add((frm[0], frm[1], to[0], to[1], k))
                new_paths = dict(paths)
                new_paths[aid] = plan(aid, new_cons[aid])
                new_cost = sum(len(p) for p in new_paths.values())
                heapq.heappush(heap, (new_cost, counter, new_cons, new_paths))
                counter += 1

        return incumbent  # hết ngân sách: trả lời tốt nhất tới giờ

    # ==================================================================
    # TẦNG MAPD: gán/duy trì nhiệm vụ cho shipper (mỗi lần 1 đơn)
    # ==================================================================
    def _validate_tasks(self, shippers: List[Shipper], orders: Dict[int, Order]) -> None:
        for sh in shippers:
            oid = self.task.get(sh.id, -1)
            if oid < 0:
                continue
            o = orders.get(oid)
            if o is None or o.delivered or (o.picked and oid not in sh.bag):
                self.task[sh.id] = -1  # đơn đã mất / bị shipper khác nhặt

    def _assign(self, shippers: List[Shipper], orders: Dict[int, Order]) -> None:
        claimed = {self.task.get(sh.id, -1) for sh in shippers}
        claimed.update(oid for sh in shippers for oid in sh.bag)
        pending = [o for o in orders.values()
                   if not o.picked and not o.delivered and o.id not in claimed]

        for sh in sorted(shippers, key=lambda s: s.id):
            if sh.bag or self.task.get(sh.id, -1) >= 0:
                continue
            cands = sorted(pending, key=lambda o: manhattan(sh.r, sh.c, o.sx, o.sy))
            for o in cands:
                if o.id in claimed:
                    continue
                if o.w > sh.W_max:
                    continue
                if self._dist((sh.r, sh.c), (o.sx, o.sy)) >= LARGE:
                    continue
                self.task[sh.id] = o.id
                claimed.add(o.id)
                break

    # ==================================================================
    # VÒNG LẶP CHÍNH (online)
    # ==================================================================
    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        while not obs.get("done", False):
            shippers: List[Shipper] = obs["shippers"]
            orders: Dict[int, Order] = obs["orders"]

            if len(self._dmaps) > 512:      # giới hạn bộ nhớ cache khoảng cách
                self._dmaps.clear()

            # 1) MAPD: duy trì + gán nhiệm vụ
            self._validate_tasks(shippers, orders)
            self._assign(shippers, orders)

            # 2) Xác định goal mỗi shipper: deliver (đang mang) / pickup / idle
            goal_of: Dict[int, Tuple[Optional[Cell], str]] = {}
            goal_cells = set()
            for sh in shippers:
                if sh.bag:
                    o = orders.get(sh.bag[0])
                    g = (o.ex, o.ey) if o else None
                    goal_of[sh.id] = (g, "D")
                else:
                    oid = self.task.get(sh.id, -1)
                    o = orders.get(oid) if oid >= 0 else None
                    if o is not None:
                        goal_of[sh.id] = ((o.sx, o.sy), "P")
                    else:
                        goal_of[sh.id] = (None, "I")
                if goal_of[sh.id][0] is not None:
                    goal_cells.add(goal_of[sh.id][0])

            # 3) Agent idle = chướng ngại tĩnh. Nếu idle đang đứng trên goal của
            #    agent khác thì đẩy nó né sang ô trống bên cạnh (tránh kẹt).
            pos_of = {sh.id: (sh.r, sh.c) for sh in shippers}
            static = set()
            for sh in shippers:
                g, typ = goal_of[sh.id]
                if g is None:
                    if pos_of[sh.id] in goal_cells:
                        free = [n for n in self._neighbors(pos_of[sh.id])
                                if n != pos_of[sh.id] and n not in goal_cells]
                        if free:
                            goal_of[sh.id] = (free[0], "M")  # né 1 bước
                            continue
                    static.add(pos_of[sh.id])

            # 4) CBS cho các agent có goal
            agents = [(sh.id, pos_of[sh.id], goal_of[sh.id][0])
                      for sh in shippers if goal_of[sh.id][0] is not None]
            paths = self._cbs(agents, frozenset(static))

            # 5) Sinh action: đi 1 bước theo path, kèm cargo_op khi tới goal
            actions: Dict[int, Tuple[str, int]] = {}
            for sh in shippers:
                g, typ = goal_of[sh.id]
                if g is None:
                    actions[sh.id] = ("S", 0)
                    continue
                path = paths.get(sh.id, [pos_of[sh.id]])
                nxt = path[1] if len(path) >= 2 else path[0]
                move = self._dir(pos_of[sh.id], nxt)
                op = 0
                if nxt == g:
                    if typ == "P":
                        op = 1   # nhặt hàng
                    elif typ == "D":
                        op = 2   # giao hàng
                actions[sh.id] = (move, op)

            obs, _, _, _ = self.env.step(actions)

        return self.env.result(self.method_name, elapsed_sec=time.time() - start_time)

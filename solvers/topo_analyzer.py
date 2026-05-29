from __future__ import annotations

import collections
import random
import time
import heapq
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Set, Tuple

class IterativeTopologyAnalyzer:
    def __init__(self, grid: List[List[int]], N: int):
        self.grid = grid
        self.N = N
        self.free_cells: List[Tuple[int, int]] = [
            (r, c) for r in range(N) for c in range(N) if grid[r][c] == 0
        ]
        self._nbr: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        self.chokepoint_cells: Set[Tuple[int, int]] = set()
        self.zone_id: Dict[Tuple[int, int], int] = {}
        self._analyze()

    def neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        if pos not in self._nbr:
            r, c = pos
            self._nbr[pos] = [
                (r+dr, c+dc)
                for dr, dc in ((-1,0),(1,0),(0,-1),(0,1))
                if 0 <= r+dr < self.N and 0 <= c+dc < self.N
                and self.grid[r+dr][c+dc] == 0
            ]
        return self._nbr[pos]

    def _analyze(self):
        self.chokepoint_cells = self._iterative_articulation_points()
        # self._build_zones()

    def _iterative_articulation_points(self) -> Set[Tuple[int, int]]:
        if not self.free_cells: return set()
        visited: Dict[Tuple[int,int], int] = {}
        low: Dict[Tuple[int,int], int] = {}
        parent: Dict[Tuple[int,int], Optional[Tuple[int,int]]] = {}
        child_count: Dict[Tuple[int,int], int] = collections.defaultdict(int)
        aps: Set[Tuple[int, int]] = set()
        timer = [0]

        for start in self.free_cells:
            if start in visited: continue
            parent[start] = None
            stack: List[Tuple[Tuple[int,int], Any]] = [(start, iter(self.neighbors(start)))]
            visited[start] = low[start] = timer[0]
            timer[0] += 1

            while stack:
                u, children = stack[-1]
                try:
                    v = next(children)
                    if v not in visited:
                        child_count[u] += 1
                        parent[v] = u
                        visited[v] = low[v] = timer[0]
                        timer[0] += 1
                        stack.append((v, iter(self.neighbors(v))))
                    elif v != parent.get(u):
                        low[u] = min(low[u], visited[v])
                except StopIteration:
                    stack.pop()
                    if stack:
                        p = stack[-1][0]
                        low[p] = min(low[p], low[u])
                        if parent[p] is None and child_count[p] > 1:
                            aps.add(p)
                        if parent[p] is not None and low[u] >= visited[p]:
                            aps.add(p)
        return aps

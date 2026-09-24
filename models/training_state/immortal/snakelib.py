"""Snake game engine + Hamiltonian-cycle utilities. Pure stdlib.

The game: a 12x12 grid (configurable), the snake spawns on 3 consecutive
cells of a Hamiltonian cycle, and apples spawn on free cells weighted
toward the far half of the board (near-only spawns collapse into camping).

Rules: the head moves one cell per step. Running into a wall or the body
kills the snake. The tail cell vacates on a non-growing step, *except*
when the head eats an apple sitting on the tail -- eating into the tail
is fatal because the tail stays. Filling the board wins the game.
"""
import random
from collections import deque

MOVES = ("up", "down", "left", "right")
DELTA = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}


class Game:
    def __init__(self, width=12, height=12, seed=7):
        self.w, self.h = width, height
        self.rng = random.Random(seed)
        cyc = cycle_index(width, height)
        if cyc is not None:
            # Spawn on 3 consecutive cycle cells near the middle so the
            # cycle-order invariant holds from move 1 (seed varies the spot).
            _, path = cyc
            n = len(path)
            mid = min(range(n), key=lambda i: abs(path[i][0] - width // 2)
                      + abs(path[i][1] - height // 2))
            k = (mid + self.rng.choice(range(n))) % n
            seg = [path[(k + i) % n] for i in range(3)]
            self.snake = deque([seg[2], seg[1], seg[0]])
        else:
            cx, cy = width // 2, height // 2
            self.snake = deque([(cx, cy), (cx - 1, cy), (cx - 2, cy)])
        self.apple = self._spawn()
        self.score = self.ticks = 0
        self.alive = True
        self.death = None
        self.won = False

    def _spawn(self):
        occ = set(self.snake)
        free = [(x, y) for x in range(self.w) for y in range(self.h)
                if (x, y) not in occ]
        if not free:
            return None
        reach = self._reachable(occ)
        cand = [c for c in free if c in reach] or free
        hx, hy = self.snake[0]
        cand.sort(key=lambda c: abs(c[0] - hx) + abs(c[1] - hy))
        far = cand[len(cand) // 2:] or cand
        return self.rng.choice(far)

    def _reachable(self, occ):
        seen = {self.snake[0]}
        queue = deque([self.snake[0]])
        while queue:
            x, y = queue.popleft()
            for dx, dy in DELTA.values():
                c = (x + dx, y + dy)
                if (0 <= c[0] < self.w and 0 <= c[1] < self.h
                        and c not in occ and c not in seen):
                    seen.add(c)
                    queue.append(c)
        return seen

    def step(self, move):
        """Advance one move. Returns True if an apple was eaten."""
        if not self.alive:
            return False
        self.ticks += 1
        hx, hy = self.snake[0]
        dx, dy = DELTA[move]
        head = (hx + dx, hy + dy)
        if not (0 <= head[0] < self.w and 0 <= head[1] < self.h):
            self.alive, self.death = False, "wall"
            return False
        body = set(self.snake)
        if head in body and (head != self.snake[-1]
                             or (self.apple is not None
                                 and head == self.apple)):
            # Tail vacates on a non-growing step; eating into it is fatal.
            self.alive, self.death = False, "body"
            return False
        self.snake.appendleft(head)
        if self.apple is not None and head == self.apple:
            self.score += 1
            self.apple = self._spawn()
            if self.apple is None:
                self.won = True  # board clear: every cell is snake
            return True
        self.snake.pop()
        return False

    def snapshot(self):
        return {"w": self.w, "h": self.h,
                "body": [tuple(c) for c in self.snake],
                "apple": tuple(self.apple) if self.apple else None,
                "score": self.score, "length": len(self.snake),
                "ticks": self.ticks, "alive": self.alive,
                "death": self.death, "won": self.won}


_cycles = {}


def _build_cycle(w, h):
    """Hamiltonian cycle over a w x h grid: visit every cell exactly once
    and return to the start. Transposes odd-height boards; impossible on
    odd x odd (checkerboard parity) -> None."""
    transpose = False
    if h % 2 == 1:
        if w % 2 == 1:
            return None
        w, h, transpose = h, w, True
    path = [(x, 0) for x in range(w)]
    for y in range(1, h):
        xs = range(w - 1, 0, -1) if y % 2 == 1 else range(1, w)
        path.extend((x, y) for x in xs)
    path.extend((0, y) for y in range(h - 1, 0, -1))
    if transpose:
        path = [(y, x) for (x, y) in path]
    assert len(set(path)) == len(path), "cycle must not repeat cells"
    return {cell: i for i, cell in enumerate(path)}, path


def cycle_index(w, h):
    """(cell -> index, path) for the Hamiltonian cycle, or None."""
    if (w, h) not in _cycles:
        _cycles[(w, h)] = _build_cycle(w, h)
    return _cycles[(w, h)]


def cycle_step(body, w, h):
    """The move that advances the head to the next cycle cell."""
    cyc = cycle_index(w, h)
    if cyc is None:
        return None
    idx, path = cyc
    nxt = path[(idx[tuple(body[0])] + 1) % len(path)]
    dx, dy = nxt[0] - body[0][0], nxt[1] - body[0][1]
    for m, (vx, vy) in DELTA.items():
        if (dx, dy) == (vx, vy):
            return m
    return None  # pragma: no cover -- cycle cells are always adjacent


def bfs_distances(source, blocked, w, h):
    """Shortest-path distances from source over free cells (blocked set
    excluded). Returns {cell: distance}."""
    dist = {source: 0}
    queue = deque([source])
    while queue:
        x, y = queue.popleft()
        for dx, dy in DELTA.values():
            c = (x + dx, y + dy)
            if (0 <= c[0] < w and 0 <= c[1] < h
                    and c not in dist and c not in blocked):
                dist[c] = dist[(x, y)] + 1
                queue.append(c)
    return dist


def bfs_first_step(source, target, blocked, w, h, tail=None):
    """First step from source toward target avoiding blocked cells.

    Gavel extension for obstacle boards (no upstream equivalent): BFS from
    the target over free cells, then return the free neighbor of source
    with the smallest distance (ties break in MOVES order, deterministic).
    The tail cell counts as free (it vacates on a non-growing step) unless
    it IS the target -- eating into the tail is fatal because the tail
    stays. Returns None when no safe step exists (truly trapped).
    """
    if target is None:
        return None
    source, target = tuple(source), tuple(target)
    blocked = set(map(tuple, blocked))
    if tail is not None and tuple(tail) != target:
        blocked.discard(tuple(tail))
    dist = bfs_distances(target, blocked, w, h)
    best, best_d = None, None
    for dx, dy in DELTA.values():
        c = (source[0] + dx, source[1] + dy)
        if not (0 <= c[0] < w and 0 <= c[1] < h) or c in blocked:
            continue
        d = dist.get(c, _LARGE)
        if best_d is None or d < best_d:
            best, best_d = c, d
    return best


_LARGE = 10 ** 9

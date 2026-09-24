"""Snake policies. All share the same game; they differ in how they pick.

- CyclePolicy: follow the Hamiltonian cycle. Provably immortal, eats an
  apple every ~66 moves (every apple is visited within one lap).
- JumpPolicy: the cycle as a safety backbone, plus forward *jumps* --
  grid-neighbors further along the cycle -- chosen to minimize distance
  to the apple. Provably immortal (see proof below), eats an apple every
  ~19-22 moves, and wins by filling the board.

SAFETY PROOF for JumpPolicy
---------------------------
Let i = the head's cycle index, L = body length, and z = moves since the
head last stood on cycle cell 0. Jumps are allowed only when z >= L.

1. Every move advances the cycle index (jumps go to a strictly larger
   index without wrapping; plain steps go to i+1), except the wraparound
   step N-1 -> 0. So at all times every body cell has an index <= the
   head's -- which means every forward target cell is free. Jumps and
   non-wrapping cycle steps are therefore always legal moves.
2. The wraparound step needs cycle cell 0 to be free, i.e. not among the
   last L body cells, i.e. z >= L.
   - If jumps were ever taken since the last visit to cell 0, then z >= L
     held at that moment; z grows by exactly 1 per move while L grows by
     at most 1, so z >= L is sticky and still holds at wraparound. Safe.
   - If jumps were never taken, the lap was pure cycle: z = N-1 at
     wraparound, and L <= N-1 (L = N means the board is full = game won).
     Safe.
3. Hence the cycle step is *always* a legal move, so the snake always has
   a move and can never die. The candidate set below always contains the
   cycle step; the distance heuristic only picks *among* safe candidates,
   so it cannot break the proof.

Termination (it actually eats): jumps never skip over the apple's cycle
cell, so the head's cycle index strictly increases toward the apple and
reaches it within one lap (<= N moves worst case).
"""
try:
    from .snakelib import DELTA, bfs_distances, bfs_first_step, cycle_index, cycle_step
    from .lookahead import LookaheadPolicy
except ImportError:  # script-dir fallback
    from snakelib import DELTA, bfs_distances, bfs_first_step, cycle_index, cycle_step
    from lookahead import LookaheadPolicy

_INF = 10 ** 9


class CyclePolicy:
    """Pure Hamiltonian-cycle follower (round 11)."""

    name = "cycle"

    def reset(self, game):
        pass

    def decide(self, game):
        body = [tuple(c) for c in game.snake]
        return cycle_step(body, game.w, game.h)


class JumpPolicy:
    """Hamiltonian cycle + provably-safe forward jumps (rounds 12/13).

    heuristic: "manhattan" (round 12 -- ~21 moves/apple, ~0.004 ms/decision)
               "bfs"       (round 13 -- ~19 moves/apple, ~0.10 ms/decision)
    """

    def __init__(self, heuristic="bfs"):
        if heuristic not in ("manhattan", "bfs"):
            raise ValueError("heuristic must be 'manhattan' or 'bfs'")
        self.heuristic = heuristic
        self.name = "jumps-" + heuristic
        self.jumps_taken = 0  # diagnostics: how often a jump beat the step
        self.guard_trips = 0  # diagnostics: defensive check (proof asserts 0)

    def reset(self, game):
        idx, path = cycle_index(game.w, game.h)
        self._idx = idx
        self._path = path
        self._n = len(path)
        self._zero = path[0]
        self._move_of = {(dx, dy): m for m, (dx, dy) in DELTA.items()}
        self._z = 0 if tuple(game.snake[0]) == self._zero else _INF
        # Obstacle support (gavel demo extension; empty on upstream boards).
        # Blocked cells are excluded from jump candidates and BFS distances.
        # NOTE: with obstacles the Hamiltonian backbone is broken, so the
        # safety proof no longer holds -- the caller must validate proposals
        # against danger and the _z counter must track EXECUTED moves
        # (see correct_z) rather than proposals.
        self._obs = set(map(tuple, getattr(game, "obstacles", None) or ()))
        self.jumps_taken = 0
        self.guard_trips = 0

    def correct_z(self, executed_cell, z_before):
        """Re-sync the wraparound guard after the caller overrode a proposal
        (e.g. shield veto into an obstacle). executed_cell is the cell the
        head actually moved to."""
        self._z = 0 if tuple(executed_cell) == self._zero else z_before + 1

    def candidates(self, game):
        """Safe target cells withOUT mutating policy state (pure query).

        Gavel extension: lets a trained decision layer choose AMONG the
        proof's candidate set instead of over raw moves. Same rules as
        decide() (cycle step always included; jumps only when z >= L,
        never over the apple, pillars excluded). Returns
        [cycle_next, *jumps]; the caller must still advance _z for the
        EXECUTED cell (see correct_z). Empty only on boards with no cycle
        (odd x odd) -- caller must handle that."""
        body = [tuple(c) for c in game.snake]
        apple = tuple(game.apple)
        head = body[0]
        i = self._idx[head]
        hx, hy = head
        length = len(body)
        body_set = set(body)
        tail = body[-1]
        cands = [self._path[(i + 1) % self._n]]
        k_apple = self._idx[apple]
        if self._z >= length:
            for dx, dy in DELTA.values():
                c = (hx + dx, hy + dy)
                if c in self._obs:
                    continue
                j = self._idx.get(c)
                if j is None or j <= i + 1:
                    continue
                if i < k_apple < j:
                    continue  # would overshoot the apple
                if c in body_set and not (c == tail and c != apple):
                    continue  # defensive: proof says free, don't trust it
                cands.append(c)
        return cands

    def decide(self, game):
        body = [tuple(c) for c in game.snake]
        apple = tuple(game.apple)
        head = body[0]
        i = self._idx[head]
        hx, hy = head
        length = len(body)
        body_set = set(body)
        tail = body[-1]

        if self.heuristic == "bfs":
            blocked = set(body_set) | set(self._obs)
            blocked.discard(tail)  # tail vacates on a non-growing step
            bfs = bfs_distances(apple, blocked, game.w, game.h)
            dist = lambda c: bfs.get(c, _INF)
        else:
            dist = lambda c: abs(c[0] - apple[0]) + abs(c[1] - apple[1])

        # The cycle step is always a candidate (proof: always legal).
        nxt = self._path[(i + 1) % self._n]
        best = (dist(nxt), 1, nxt)  # tiebreak 1: plain step loses ties to jumps

        # Forward jumps: grid-neighbor with a strictly larger cycle index
        # (no wrapping), allowed only under the wraparound guard z >= L,
        # and never jumping *over* the apple's cycle cell.
        k_apple = self._idx[apple]
        if self._z >= length:
            for dx, dy in DELTA.values():
                c = (hx + dx, hy + dy)
                if c in self._obs:
                    continue  # pillar: not a free cell, never a jump target
                j = self._idx.get(c)
                if j is None or j <= i + 1:
                    continue
                if i < k_apple < j:
                    continue  # would overshoot the apple; it would lap forever
                # The proof says c is free (every body index <= i < j).
                # Checked anyway: if this ever trips, the proof is wrong
                # and we must not step into the body.
                if c in body_set and not (c == tail and c != apple):
                    self.guard_trips += 1
                    continue
                d = dist(c)
                if (d, 0) < (best[0], best[1]):
                    best = (d, 0, c)

        _, _, target = best
        detour = False
        if target in self._obs:
            # Backbone severed here (cycle step is a pillar and no jump
            # beats it): route around the obstacle with BFS instead of
            # stepping into it. Without this the snake paces the cut
            # forever -- every proposal vetoed, apple never reached.
            via = bfs_first_step(head, apple, body_set | self._obs,
                                 game.w, game.h, tail)
            if via is not None:
                target, detour = via, True
            # else: truly trapped -- return the blocked cell and let the
            # caller record the veto/death honestly.
        if target != nxt:
            self.jumps_taken += 1
        dx, dy = target[0] - hx, target[1] - hy
        self._z = 0 if target == self._zero else self._z + 1
        return self._move_of[(dx, dy)]


POLICIES = {
    "cycle": CyclePolicy,
    "jumps-manhattan": lambda: JumpPolicy("manhattan"),
    "jumps-bfs": lambda: JumpPolicy("bfs"),
    # Winner: 2-ply lookahead over the same provably-safe candidates.
    # Leaf rule: BFS-pursue when the apple is <64 cycle-steps ahead,
    # otherwise race the lap (max cycle progress). Proof untouched.
    "lookahead": lambda: LookaheadPolicy(depth=2, leaf="hybrid"),
}

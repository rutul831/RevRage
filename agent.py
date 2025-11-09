import importlib
import os
import time
import uuid
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from flask import Flask, request, jsonify
from threading import Lock
from collections import deque

from case_closed_game import Game, Direction, GameResult

# Flask API server setup
app = Flask(__name__)

GLOBAL_GAME = Game()
LAST_POSTED_STATE = {}

game_lock = Lock()
 
PARTICIPANT = "ParticipantX"
AGENT_NAME = "AgentX"
AGENT_STRATEGY = os.environ.get("AGENT_STRATEGY", "legacy_tronbot")


INT_MAX = 2 ** 31 - 1
TIMEOUT_USEC = 750_000
FIRSTMOVE_USEC = 1_200_000
DEPTH_INITIAL = 1
DEPTH_MAX = 100
DRAW_PENALTY = 0
K1 = 55
K2 = 194
K3 = 3  # Present in the original bot, retained for completeness

DX = (0, 0, 1, -1)
DY = (-1, 1, 0, 0)
MOVE_NAMES = ("UP", "DOWN", "RIGHT", "LEFT")

POTENTIAL_ARTICULATION: Tuple[int, ...] = (
    0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0,
    0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0,
    0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
)


@dataclass
class Position:
    x: int
    y: int

    def next(self, move: int) -> "Position":
        return Position(self.x + DX[move], self.y + DY[move])


@dataclass
class GameState:
    p: List[Position]
    m: List[int]

    def clone(self) -> "GameState":
        return GameState([Position(pos.x, pos.y) for pos in self.p], list(self.m))


@dataclass
class ColorCount:
    red: int = 0
    black: int = 0
    edges: int = 0
    front: int = 0

    def __add__(self, other: "ColorCount") -> "ColorCount":
        return ColorCount(
            self.red + other.red,
            self.black + other.black,
            self.edges + other.edges,
            self.front + other.front,
        )


class Grid:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.data: List[int] = []

    def resize(self, width: int, height: int, fill: int = 0):
        self.width = width
        self.height = height
        self.data = [fill] * (width * height)

    def clear(self, value: int = 0):
        for i in range(len(self.data)):
            self.data[i] = value

    def idx(self, x: int, y: int) -> int:
        return x + y * self.width

    def __getitem__(self, key):
        if isinstance(key, Position):
            return self.data[self.idx(key.x, key.y)]
        if isinstance(key, tuple):
            x, y = key
            return self.data[self.idx(x, y)]
        return self.data[key]

    def __setitem__(self, key, value):
        if isinstance(key, Position):
            self.data[self.idx(key.x, key.y)] = value
        elif isinstance(key, tuple):
            x, y = key
            self.data[self.idx(x, y)] = value
        else:
            self.data[key] = value


def num_fillable(ccount: ColorCount, startcolor: int) -> int:
    if startcolor:
        return 2 * min(ccount.red - 1, ccount.black) + (1 if ccount.black >= ccount.red else 0)
    return 2 * min(ccount.red, ccount.black - 1) + (1 if ccount.red >= ccount.black else 0)


class Components:
    def __init__(self, bot: "TronBot"):
        self.bot = bot
        self.M = bot.M
        self.c = Grid()
        self.c.resize(self.M.width, self.M.height, 0)
        self.cedges: List[int] = []
        self.red: List[int] = []
        self.black: List[int] = []
        self.recalc()

    def recalc(self):
        width, height = self.M.width, self.M.height
        self.c.clear(0)
        equiv = [0]
        nextclass = 1
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                idx = self.c.idx(x, y)
                if self.M[idx]:
                    continue
                cup = equiv[self.c[idx - width]]
                cleft = equiv[self.c[idx - 1]]
                if cup == 0 and cleft == 0:
                    equiv.append(nextclass)
                    self.c[idx] = nextclass
                    nextclass += 1
                elif cup == cleft:
                    self.c[idx] = cup
                else:
                    if cleft == 0 or (cup != 0 and cup < cleft):
                        self.c[idx] = cup
                        if cleft != 0:
                            self._merge(equiv, cleft, cup)
                    else:
                        self.c[idx] = cleft
                        if cup != 0:
                            self._merge(equiv, cup, cleft)
        self.cedges = [0] * nextclass
        self.red = [0] * nextclass
        self.black = [0] * nextclass
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                idx = self.c.idx(x, y)
                if self.M[idx]:
                    continue
                e = equiv[self.c[idx]]
                self.c[idx] = e
                self.cedges[e] += self.bot.degree_idx(idx)
                if self.bot.color_coords(x, y):
                    self.red[e] += 1
                else:
                    self.black[e] += 1

    def remove(self, pos: Position):
        self.c[pos] = 0
        if self.bot.potential_articulation(pos):
            self.recalc()
        else:
            comp = self.c[pos]
            self.cedges[comp] -= 2 * self.bot.degree(pos)
            if self.bot.color(pos):
                self.red[comp] -= 1
            else:
                self.black[comp] -= 1

    def add(self, pos: Position):
        for move in range(4):
            neighbor = pos.next(move)
            if self.M[neighbor]:
                continue
            if self.c[pos] != 0 and self.c[pos] != self.c[neighbor]:
                self.recalc()
                return
            self.c[pos] = self.c[neighbor]
        self.cedges[self.c[pos]] += 2 * self.bot.degree(pos)
        if self.bot.color(pos):
            self.red[self.c[pos]] += 1
        else:
            self.black[self.c[pos]] += 1

    def component(self, pos: Position) -> int:
        return self.c[pos]

    def connectedarea(self, pos_or_idx) -> int:
        if isinstance(pos_or_idx, Position):
            comp = self.c[pos_or_idx]
        else:
            comp = pos_or_idx
        return self.red[comp] + self.black[comp]

    def fillablearea(self, pos: Position) -> int:
        comp = self.c[pos]
        if comp >= len(self.red):
            return 0
        return num_fillable(ColorCount(self.red[comp], self.black[comp], 0, 0), self.bot.color(pos))

    def connectedvalue(self, pos: Position) -> int:
        comp = self.c[pos]
        if comp >= len(self.cedges):
            return 0
        return self.cedges[comp]

    @staticmethod
    def _merge(equiv: List[int], old: int, new: int):
        for idx, value in enumerate(equiv):
            if value == old:
                equiv[idx] = new


class TronBot:
    def __init__(self):
        self.M = Grid()
        self.dp0 = Grid()
        self.dp1 = Grid()
        self.low = Grid()
        self.num = Grid()
        self.articd = Grid()
        self.curstate = GameState([Position(0, 0), Position(0, 0)], [0, 0])
        self._killer = [0] * (DEPTH_MAX * 2 + 2)
        self._maxitr = 0
        self._ab_runs = 0
        self._spacefill_runs = 0
        self._art_counter = 0
        self._timed_out = False
        self._timer_start = 0.0
        self._time_limit = 0.0
        self.evaluations = 0
        self.first_move = True
        self._last_components: Optional[Components] = None

    def decide_move(
        self,
        board: Sequence[Sequence[int]],
        agent1_trail: Sequence[Tuple[int, int]],
        agent2_trail: Sequence[Tuple[int, int]],
        player_number: int,
        boosts_remaining: int = 0,
    ) -> str:
        if not self._load_state(board, agent1_trail, agent2_trail, player_number):
            return "RIGHT"
        self._reset_timer(FIRSTMOVE_USEC if self.first_move else TIMEOUT_USEC)
        move_idx = self._next_move()
        self.first_move = False
        move_idx = max(0, min(3, move_idx))
        use_boost = self._should_use_boost(move_idx, boosts_remaining)
        move_name = MOVE_NAMES[move_idx]
        return f"{move_name}:BOOST" if use_boost else move_name

    def _load_state(
        self,
        board: Sequence[Sequence[int]],
        agent1_trail: Sequence[Tuple[int, int]],
        agent2_trail: Sequence[Tuple[int, int]],
        player_number: int,
    ) -> bool:
        if not board or not board[0]:
            return False
        height = len(board)
        width = len(board[0])
        internal_width = width + 2
        internal_height = height + 2
        self._ensure_maps(internal_width, internal_height)
        for y in range(internal_height):
            for x in range(internal_width):
                if x == 0 or y == 0 or x == internal_width - 1 or y == internal_height - 1:
                    self.M[x, y] = 1
                else:
                    self.M[x, y] = 1 if board[y - 1][x - 1] else 0
        heads = [self._head_from_trail(agent1_trail), self._head_from_trail(agent2_trail)]
        if heads[0] is None or heads[1] is None:
            return False
        if player_number == 1:
            self.curstate.p[0], self.curstate.p[1] = heads[0], heads[1]
        else:
            self.curstate.p[0], self.curstate.p[1] = heads[1], heads[0]
        self.curstate.m[0] = 0
        self.curstate.m[1] = 0
        self.M[self.curstate.p[0]] = 0
        self.M[self.curstate.p[1]] = 0
        return True

    @staticmethod
    def _head_from_trail(trail: Sequence[Tuple[int, int]]) -> Optional[Position]:
        if not trail:
            return None
        x, y = trail[-1]
        return Position(int(x) + 1, int(y) + 1)

    def _ensure_maps(self, width: int, height: int):
        if self.M.width == width and self.M.height == height:
            return
        self.M.resize(width, height, 0)
        self.dp0.resize(width, height, 0)
        self.dp1.resize(width, height, 0)
        self.low.resize(width, height, 0)
        self.num.resize(width, height, 0)
        self.articd.resize(width, height, 0)

    def _reset_timer(self, timeout_usec: int):
        self._timer_start = time.perf_counter()
        self._time_limit = self._timer_start + timeout_usec / 1_000_000.0
        self._timed_out = False
        self._ab_runs = 0
        self._spacefill_runs = 0

    def _store_principal_variation(self, line: Sequence[int]):
        limit = min(len(line), len(self._killer))
        for idx in range(limit):
            self._killer[idx] = line[idx]

    def _advance_principal_variation(self):
        self._killer = self._killer[2:] + [0, 0]

    def _check_timeout(self) -> bool:
        if not self._timed_out and time.perf_counter() >= self._time_limit:
            self._timed_out = True
        return self._timed_out

    def _elapsed_time(self) -> float:
        return (time.perf_counter() - self._timer_start) * 1_000_000.0

    def color(self, pos: Position) -> int:
        return (pos.x ^ pos.y) & 1

    def color_coords(self, x: int, y: int) -> int:
        return (x ^ y) & 1

    def degree(self, pos: Position) -> int:
        idx = self.M.idx(pos.x, pos.y)
        return self.degree_idx(idx)

    def degree_idx(self, idx: int) -> int:
        width = self.M.width
        return 4 - self.M[idx - 1] - self.M[idx + 1] - self.M[idx - width] - self.M[idx + width]

    def manhattan_distance(self, a: Position, b: Position) -> int:
        return abs(a.x - b.x) + abs(a.y - b.y)

    def neighbors(self, pos: Position) -> int:
        return (
            self.M[pos.x - 1, pos.y - 1]
            | (self.M[pos.x, pos.y - 1] << 1)
            | (self.M[pos.x + 1, pos.y - 1] << 2)
            | (self.M[pos.x + 1, pos.y] << 3)
            | (self.M[pos.x + 1, pos.y + 1] << 4)
            | (self.M[pos.x, pos.y + 1] << 5)
            | (self.M[pos.x - 1, pos.y + 1] << 6)
            | (self.M[pos.x - 1, pos.y] << 7)
        )

    def potential_articulation(self, pos: Position) -> int:
        return POTENTIAL_ARTICULATION[self.neighbors(pos)]

    def dijkstra(self, dist: Grid, start: Position):
        size = self.M.width * self.M.height
        dist.clear(INT_MAX)
        queue = [[], []]
        active = 0
        queue[0].append(start)
        dist[start] = 0
        radius = 0
        while queue[active]:
            while queue[active]:
                u = queue[active].pop()
                if dist[u] != radius:
                    continue
                for move in range(4):
                    v = u.next(move)
                    if self.M[v]:
                        continue
                    value = dist[v]
                    if value == INT_MAX:
                        queue[active ^ 1].append(v)
                        dist[v] = dist[u] + 1
            active ^= 1
            radius += 1

    def floodfill(self, components: Components, start: Position, fixup: bool = True) -> int:
        best_value = 0
        best_pos = start
        for move in range(4):
            nxt = start.next(move)
            if self.M[nxt]:
                continue
            value = (
                components.connectedvalue(nxt)
                + components.fillablearea(nxt)
                - 2 * self.degree(nxt)
                - 4 * self.potential_articulation(nxt)
            )
            if value > best_value:
                best_value = value
                best_pos = nxt
        if best_value == 0:
            return 0
        self.M[best_pos] = 1
        components.remove(best_pos)
        area = 1 + self.floodfill(components, best_pos)
        self.M[best_pos] = 0
        if fixup:
            components.add(best_pos)
        return area

    def _spacefill(self, move_holder: List[int], components: Components, start: Position, itr: int) -> int:
        best_value = 0
        spaces_left = components.fillablearea(start)
        if self.degree(start) == 0:
            move_holder[0] = 1
            return 0
        if self._check_timeout():
            return 0
        if itr == 0:
            return self.floodfill(components, start)
        for move in range(4):
            if self._check_timeout():
                break
            nxt = start.next(move)
            if self.M[nxt]:
                continue
            self.M[nxt] = 1
            components.remove(nxt)
            tmp = [0]
            value = 1 + self._spacefill(tmp, components, nxt, itr - 1)
            self.M[nxt] = 0
            components.add(nxt)
            if value > best_value:
                best_value = value
                move_holder[0] = move
            if value == spaces_left:
                break
        return best_value

    def next_move_spacefill(self, components: Components) -> int:
        area = components.fillablearea(self.curstate.p[0])
        best_value = 0
        best_move = 1
        itr = DEPTH_INITIAL
        while itr < DEPTH_MAX and not self._check_timeout():
            holder = [0]
            self._maxitr = itr
            value = self._spacefill(holder, components, self.curstate.p[0], itr)
            if value > best_value:
                best_value = value
                best_move = holder[0]
            if value <= itr or value >= area:
                break
            itr += 1
        return best_move

    def reset_articulations(self):
        self._art_counter = 0
        self.low.clear(0)
        self.num.clear(0)
        self.articd.clear(0)

    def calc_articulations(self, dp0: Optional[Grid], dp1: Optional[Grid], v: Position, parent: int = -1) -> int:
        self._art_counter += 1
        nodenum = self._art_counter
        self.low[v] = nodenum
        self.num[v] = nodenum
        children = 0
        count = 0
        for move in range(4):
            w = v.next(move)
            if self.M[w]:
                continue
            if dp0 is not None and dp0[w] >= dp1[w]:
                continue
            if self.num[w] == 0:
                children += 1
                count += self.calc_articulations(dp0, dp1, w, nodenum)
                if self.low[w] >= nodenum and parent != -1:
                    self.articd[v] = 1
                    count += 1
                if self.low[w] < self.low[v]:
                    self.low[v] = self.low[w]
            else:
                if self.num[w] < nodenum and self.num[w] < self.low[v]:
                    self.low[v] = self.num[w]
        if parent == -1 and children > 1:
            count += 1
            self.articd[v] = 1
        return count

    def explore_space(
        self,
        dp0: Optional[Grid],
        dp1: Optional[Grid],
        exits: List[Position],
        v: Position,
    ) -> ColorCount:
        space = ColorCount()
        if self.num[v] == 0:
            return space
        if self.color(v):
            space.red += 1
        else:
            space.black += 1
        self.num[v] = 0
        if self.articd[v]:
            for move in range(4):
                w = v.next(move)
                if self.M[w]:
                    continue
                space.edges += 1
                if dp0 is not None and dp0[w] >= dp1[w]:
                    space.front = 1
                    continue
                if self.num[w] == 0:
                    continue
                exits.append(w)
        else:
            for move in range(4):
                w = v.next(move)
                if self.M[w]:
                    continue
                space.edges += 1
                if dp0 is not None and dp0[w] >= dp1[w]:
                    space.front = 1
                    continue
                if self.num[w] == 0:
                    continue
                if self.articd[w]:
                    exits.append(w)
                else:
                    space = space + self.explore_space(dp0, dp1, exits, w)
        return space

    def max_articulated_space(self, dp0: Optional[Grid], dp1: Optional[Grid], v: Position) -> ColorCount:
        exits: List[Position] = []
        space = self.explore_space(dp0, dp1, exits, v)
        maxspace = space
        maxsteps = 0
        entrance_color = self.color(v)
        localsteps = (
            num_fillable(ColorCount(space.red, space.black + 1, 0, 0), entrance_color),
            num_fillable(ColorCount(space.red + 1, space.black, 0, 0), entrance_color),
        )
        for exit_pos in exits:
            exitcolor = self.color(exit_pos)
            child = self.max_articulated_space(dp0, dp1, exit_pos)
            steps = num_fillable(child, exitcolor)
            if not child.front:
                steps += localsteps[exitcolor]
            else:
                steps += (dp0[exit_pos] - 1) if dp0 is not None else 0
            if steps > maxsteps:
                maxsteps = steps
                if not child.front:
                    maxspace = space + child
                else:
                    maxspace = child
        return maxspace

    def evaluate_territory(self, state: GameState, components: Components, comp: int) -> int:
        self.dijkstra(self.dp0, state.p[0])
        self.dijkstra(self.dp1, state.p[1])
        self.reset_articulations()
        self.M[state.p[0]] = 0
        self.M[state.p[1]] = 0
        self.calc_articulations(self.dp0, self.dp1, state.p[0])
        self.calc_articulations(self.dp1, self.dp0, state.p[1])
        c0 = self.max_articulated_space(self.dp0, self.dp1, state.p[0])
        c1 = self.max_articulated_space(self.dp1, self.dp0, state.p[1])
        score0 = K1 * (c0.front + num_fillable(c0, self.color(state.p[0]))) + K2 * c0.edges
        score1 = K1 * (c1.front + num_fillable(c1, self.color(state.p[1]))) + K2 * c1.edges
        self.M[state.p[0]] = 1
        self.M[state.p[1]] = 1
        return score0 - score1

    def evaluate_board(self, state: GameState, player: int) -> int:
        assert player == 0
        self.M[state.p[0]] = 0
        self.M[state.p[1]] = 0
        components = Components(self)
        self.M[state.p[0]] = 1
        self.M[state.p[1]] = 1
        if state.p[0].x == state.p[1].x and state.p[0].y == state.p[1].y:
            return 0
        self.evaluations += 1
        comp0 = components.component(state.p[0])
        comp1 = components.component(state.p[1])
        if comp0 == comp1:
            return self.evaluate_territory(state, components, comp0)
        self.reset_articulations()
        self.M[state.p[0]] = 0
        self.M[state.p[1]] = 0
        self.calc_articulations(None, None, state.p[0])
        self.calc_articulations(None, None, state.p[1])
        cc0 = self.max_articulated_space(None, None, state.p[0])
        cc1 = self.max_articulated_space(None, None, state.p[1])
        ff0 = num_fillable(cc0, self.color(state.p[0]))
        ff1 = num_fillable(cc1, self.color(state.p[1]))
        value = 10000 * (ff0 - ff1)
        if value != 0 and abs(value) <= 30000:
            tmp = [0]
            ff0 = self._spacefill(tmp, components, state.p[0], 3)
            ff1 = self._spacefill(tmp, components, state.p[1], 3)
            value = 10000 * (ff0 - ff1)
        self.M[state.p[0]] = 1
        self.M[state.p[1]] = 1
        return value

    def _first_available_move(self, pos: Position) -> int:
        for move in range(4):
            if not self.M[pos.next(move)]:
                return move
        return 0

    def alphabeta(self, state: GameState, player: int, a: int, b: int, itr: int, depth_idx: int) -> Tuple[int, List[int]]:
        self._ab_runs += 1
        default_line: List[int] = []
        if state.p[0].x == state.p[1].x and state.p[0].y == state.p[1].y:
            return DRAW_PENALTY, default_line
        dp0 = self.degree(state.p[player])
        dp1 = self.degree(state.p[player ^ 1])
        if dp0 == 0:
            if dp1 == 0:
                return DRAW_PENALTY, default_line
            move = self._first_available_move(state.p[player])
            return -INT_MAX, [move]
        if dp1 == 0:
            move = self._first_available_move(state.p[player])
            return INT_MAX, [move]
        if self._check_timeout():
            return a, default_line
        if itr == 0:
            return self.evaluate_board(state, player), default_line
        kill = self._killer[depth_idx]
        candidates = [kill] + [m for m in range(4) if m != kill]
        best_line: List[int] = []
        for move in candidates:
            if self._check_timeout():
                break
            nxt = state.p[player].next(move)
            if self.M[nxt]:
                continue
            next_state = state.clone()
            next_state.m[player] = move
            if player == 1:
                next_state.p[0] = state.p[0].next(next_state.m[0])
                next_state.p[1] = state.p[1].next(next_state.m[1])
                self.M[next_state.p[0]] = 1
                self.M[next_state.p[1]] = 1
            child_value, child_line = self.alphabeta(
                next_state,
                player ^ 1,
                -b,
                -a,
                itr - 1,
                depth_idx + 1,
            )
            child_value = -child_value
            if player == 1:
                self.M[next_state.p[0]] = 0
                self.M[next_state.p[1]] = 0
            if child_value > a:
                a = child_value
                best_line = [move] + child_line
                self._killer[depth_idx] = move
            if a >= b:
                break
        return a, best_line

    def next_move_alphabeta(self) -> int:
        last_move = 1
        for depth in range(DEPTH_INITIAL, DEPTH_MAX):
            if self._check_timeout():
                break
            self._maxitr = depth * 2
            value, line = self.alphabeta(self.curstate.clone(), 0, -INT_MAX, INT_MAX, self._maxitr, 0)
            if self._timed_out:
                break
            if value == INT_MAX and line:
                self._store_principal_variation(line)
                self._advance_principal_variation()
                return line[0]
            if value == -INT_MAX:
                break
            if line:
                last_move = line[0]
                self._store_principal_variation(line)
        self._advance_principal_variation()
        return last_move

    def _should_use_boost(self, move_idx: int, boosts_remaining: int) -> bool:
        if boosts_remaining <= 0:
            return False
        # ensure we have enough map info
        current = self.curstate.p[0]
        opponent = self.curstate.p[1]
        first_step = current.next(move_idx)
        second_step = first_step.next(move_idx)
        if self.M[first_step] or self.M[second_step]:
            return False
        if second_step == opponent:
            return False
        distance_now = self.manhattan_distance(current, opponent)
        distance_two = self.manhattan_distance(second_step, opponent)
        degree_two = self.degree(second_step)
        area_advantage = 0
        if self._last_components:
            comp_me = self._last_components.component(current)
            comp_opp = self._last_components.component(opponent)
            area_me = self._last_components.fillablearea(current) if comp_me else 0
            area_opp = self._last_components.fillablearea(opponent) if comp_opp else 0
            area_advantage = area_me - area_opp
        # Use boost only when it increases distance significantly or we're far ahead on territory.
        if (distance_two - distance_now) >= 2 and degree_two >= 2:
            return True
        if area_advantage >= 15 and degree_two >= 2:
            return True
        return False

    def _next_move(self) -> int:
        components = Components(self)
        self._last_components = components
        self.M[self.curstate.p[0]] = 1
        self.M[self.curstate.p[1]] = 1
        if components.component(self.curstate.p[0]) == components.component(self.curstate.p[1]):
            move = self.next_move_alphabeta()
        else:
            move = self.next_move_spacefill(components)
        return move


TRON_BOT = TronBot()
_CUSTOM_AGENT_FUNC: Optional[Callable[[Sequence[Sequence[int]], Tuple[int, int], Tuple[int, int]], str]] = None


def _load_custom_agent():
    global _CUSTOM_AGENT_FUNC
    if AGENT_STRATEGY in ("legacy_tronbot", "tronbot_port"):
        return None
    if _CUSTOM_AGENT_FUNC is None:
        module_name = AGENT_STRATEGY
        if not module_name.startswith("agents."):
            module_name = f"agents.{module_name}"
        module = importlib.import_module(module_name)
        _CUSTOM_AGENT_FUNC = getattr(module, "next_move")
    return _CUSTOM_AGENT_FUNC

@app.route("/", methods=["GET"])
def info():
    """Basic health/info endpoint used by the judge to check connectivity.

    Returns participant and agent_name (so Judge.check_latency can create Agent objects).
    """
    return jsonify({"participant": PARTICIPANT, "agent_name": AGENT_NAME}), 200


def _update_local_game_from_post(data: dict):
    """Update the local GLOBAL_GAME using the JSON posted by the judge.

    The judge posts a dictionary with keys matching the Judge.send_state payload
    (board, agent1_trail, agent2_trail, agent1_length, agent2_length, agent1_alive,
    agent2_alive, agent1_boosts, agent2_boosts, turn_count).
    """
    with game_lock:
        LAST_POSTED_STATE.clear()
        LAST_POSTED_STATE.update(data)

        if "board" in data:
            try:
                GLOBAL_GAME.board.grid = data["board"]
            except Exception:
                pass

        if "agent1_trail" in data:
            GLOBAL_GAME.agent1.trail = deque(tuple(p) for p in data["agent1_trail"]) 
        if "agent2_trail" in data:
            GLOBAL_GAME.agent2.trail = deque(tuple(p) for p in data["agent2_trail"]) 
        if "agent1_length" in data:
            GLOBAL_GAME.agent1.length = int(data["agent1_length"])
        if "agent2_length" in data:
            GLOBAL_GAME.agent2.length = int(data["agent2_length"])
        if "agent1_alive" in data:
            GLOBAL_GAME.agent1.alive = bool(data["agent1_alive"])
        if "agent2_alive" in data:
            GLOBAL_GAME.agent2.alive = bool(data["agent2_alive"])
        if "agent1_boosts" in data:
            GLOBAL_GAME.agent1.boosts_remaining = int(data["agent1_boosts"])
        if "agent2_boosts" in data:
            GLOBAL_GAME.agent2.boosts_remaining = int(data["agent2_boosts"])
        if "turn_count" in data:
            GLOBAL_GAME.turns = int(data["turn_count"])


@app.route("/send-state", methods=["POST"])
def receive_state():
    """Judge calls this to push the current game state to the agent server.

    The agent should update its local representation and return 200.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "no json body"}), 400
    _update_local_game_from_post(data)
    return jsonify({"status": "state received"}), 200


@app.route("/send-move", methods=["GET"])
def send_move():
    """Judge calls this (GET) to request the agent's move for the current tick.

    Query params the judge sends (optional): player_number, attempt_number,
    random_moves_left, turn_count. Agents can use this to decide.
    
    Return format: {"move": "DIRECTION"} or {"move": "DIRECTION:BOOST"}
    where DIRECTION is UP, DOWN, LEFT, or RIGHT
    and :BOOST is optional to use a speed boost (move twice)
    """
    player_number = request.args.get("player_number", default=1, type=int)

    with game_lock:
        board_snapshot = [row[:] for row in GLOBAL_GAME.board.grid]
        agent1_trail = list(GLOBAL_GAME.agent1.trail)
        agent2_trail = list(GLOBAL_GAME.agent2.trail)
        my_agent = GLOBAL_GAME.agent1 if player_number == 1 else GLOBAL_GAME.agent2
        opp_agent = GLOBAL_GAME.agent2 if player_number == 1 else GLOBAL_GAME.agent1
        my_boosts = my_agent.boosts_remaining

    try:
        if AGENT_STRATEGY == "legacy_tronbot":
            move = TRON_BOT.decide_move(
                board_snapshot,
                agent1_trail,
                agent2_trail,
                player_number,
                boosts_remaining=my_boosts,
            )
        else:
            agent_func = _load_custom_agent()
            if agent_func is None:
                move = TRON_BOT.decide_move(board_snapshot, agent1_trail, agent2_trail, player_number)
            else:
                my_head = tuple(my_agent.trail[-1]) if my_agent.trail else (0, 0)
                opp_head = tuple(opp_agent.trail[-1]) if opp_agent.trail else (0, 0)
                move = agent_func(board_snapshot, my_head, opp_head)
    except Exception:
        move = "RIGHT"

    return jsonify({"move": move}), 200


@app.route("/end", methods=["POST"])
def end_game():
    """Judge notifies agent that the match finished and provides final state.

    We update local state for record-keeping and return OK.
    """
    data = request.get_json()
    if data:
        _update_local_game_from_post(data)
    return jsonify({"status": "acknowledged"}), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5008"))
    app.run(host="0.0.0.0", port=port, debug=True)

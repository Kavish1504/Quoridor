# 🎯 Quoridor AI

A fully playable **Human vs AI** implementation of the Quoridor board game in Python, featuring an intelligent AI opponent built from scratch using the **Minimax algorithm with Alpha-Beta Pruning**.

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python) ![Pygame](https://img.shields.io/badge/Pygame-2.x-green) ![Algorithm](https://img.shields.io/badge/Algorithm-Minimax%20%2B%20Alpha--Beta-orange) ![AI](https://img.shields.io/badge/AI-Classical%20Search-red)

---

## 📌 What is Quoridor?

Quoridor is a two-player abstract strategy board game played on a **9×9 grid**. Each player starts on opposite sides and races to reach the other end. Players can either:
- **Move** their pawn to an adjacent cell, or
- **Place a wall** to block the opponent's path

The catch — walls must never completely trap a player; a path to the goal must always exist.

---

## 🧠 How the AI Works

### Minimax Algorithm
The AI simulates a **game tree** of all possible future moves. It assumes:
- **Maximizing player (AI)** → always picks the move with the highest score
- **Minimizing player (Human)** → always picks the move with the lowest score

The AI looks ahead up to **depth 3** in the game tree and selects the move that leads to the best guaranteed outcome.

```
                    AI Turn (Maximize)
                   /        |         \
              Move A      Move B      Move C
             /     \                 /     \
        Human    Human          Human    Human
       (Min)    (Min)          (Min)    (Min)
```

### Alpha-Beta Pruning
Without pruning, Minimax evaluates every node — **O(b^d)** where b = branching factor (~130), d = depth.

Alpha-Beta pruning **cuts off branches** that can't influence the final decision:
- **Alpha** → best score the maximizer is guaranteed so far
- **Beta** → best score the minimizer is guaranteed so far
- If `beta ≤ alpha` → prune. No need to explore further.

This reduces complexity to **O(b^(d/2))** in the best case — effectively allowing the AI to search **twice as deep** in the same time.

```python
# Core pruning logic
if beta <= alpha:
    break  # This branch will never be chosen — prune it
```

---

## 📐 Project Architecture

```
quoridor/
│
├── main.py               # Game loop, event handling, rendering
│
├── Board (class)
│   ├── State             # Player positions, wall sets, wall counts
│   ├── is_wall_between() # Checks if a wall blocks movement between two cells
│   ├── get_neighbors()   # Returns valid adjacent cells for a position
│   ├── valid_wall()      # Validates wall placement (boundary + overlap + BFS check)
│   ├── path_exists()     # BFS — checks if a path to goal exists
│   ├── path_cost()       # BFS — returns shortest path length (used in heuristic)
│   └── evaluate()        # Heuristic scoring function
│
└── minimax()             # Recursive Minimax + Alpha-Beta search
    └── ai_turn()         # Calls minimax, applies best action
```

---

## 🔍 Key Components Explained

### 1. Wall Representation
Walls are stored as **Python sets** — giving O(1) average lookup time.

```python
self.h_walls = set()  # Horizontal walls → block vertical movement
self.v_walls = set()  # Vertical walls   → block horizontal movement
```

Each wall covers **2 cells**. A wall stored at `(row, col)` blocks the gap between `(row, col)` and `(row, col+1)` for vertical walls, and similarly for horizontal.

---

### 2. `is_wall_between(pos1, pos2)`
Checks whether a wall blocks movement between two adjacent cells.

```
Moving UP/DOWN → blocked by a Horizontal wall
Moving LEFT/RIGHT → blocked by a Vertical wall
```

Since walls span 2 cells, **two possible wall positions** can block any single gap — both are checked.

```python
# Example: moving vertically between (3,4) and (4,4)
# Check (3, 4) in h_walls  → wall starting at this cell
# Check (3, 3) in h_walls  → wall starting one cell left, still covers this gap
```

---

### 3. Wall Validation — 3-Layer Check

```
Layer 1: Boundary Check     → wall must fit within 8×8 placement grid
       ↓
Layer 2: Overlap Check      → no two walls can share or cross positions
       ↓
Layer 3: Path Check (BFS)   → after placing wall, both players must still
                               have at least one reachable path to their goal
```

The path check runs on a **cloned board** — the actual game state is never modified during validation.

---

### 4. Heuristic Evaluation Function

```python
def evaluate(self):
    p1_distance = self.path_cost(self.p1_pos, goal_row=8)   # Human's BFS distance to goal
    p2_distance = self.path_cost(self.p2_pos, goal_row=0)   # AI's BFS distance to goal
    wall_advantage = self.p2_walls - self.p1_walls

    return p1_distance - p2_distance + wall_advantage * 0.5
```

| Component | Weight | Reasoning |
|---|---|---|
| `p1_distance - p2_distance` | 1.0 | Primary win condition — being closer to goal matters most |
| `wall_advantage` | 0.5 | Secondary — more walls = more strategic flexibility |

**Positive score** → favorable for AI. **Negative score** → favorable for Human.

---

### 5. AI Difficulty Levels

| Level | Search Depth | Behaviour |
|---|---|---|
| 1 | Depth 1 | Looks only 1 move ahead — easy to beat |
| 2 | Depth 2 | Balanced — default difficulty |
| 3 | Depth 3 | Looks 3 moves ahead — challenging |

Press `1`, `2`, or `3` during the game to switch difficulty in real time.

---

## 🎮 Controls

| Key / Action | Description |
|---|---|
| `Mouse Click` | Move pawn to clicked cell |
| `SPACE` | Toggle between Move mode and Wall mode |
| `H` | Set wall orientation to Horizontal |
| `V` | Set wall orientation to Vertical |
| `1` / `2` / `3` | Set AI difficulty level |
| `D` | Debug — print current wall positions to console |

---

## ⚙️ How to Run

```bash
# Clone the repository
git clone https://github.com/Kavish1504/quoridor-ai
cd quoridor-ai

# Install dependency
pip install pygame

# Run the game
python main.py
```

---

## 📊 Complexity Analysis

| Operation | Complexity | Notes |
|---|---|---|
| Minimax (no pruning) | O(b^d) | b ≈ 130, d = 3 |
| Minimax (with Alpha-Beta) | O(b^(d/2)) best case | ~11x faster |
| BFS path check | O(N²) | N = 9, so O(81) per call |
| Wall lookup | O(1) average | Python set |
| Valid wall generation | O(128 × O(N²)) | Per AI turn |

---

## 🚀 Potential Improvements

- **Transposition Table** — cache already-evaluated board states using Zobrist hashing to avoid redundant computation
- **Iterative Deepening** — search progressively deeper within a time budget instead of a fixed depth cap
- **Move Ordering** — evaluate promising moves first to maximize Alpha-Beta pruning efficiency
- **Undo-based State** — replace `deepcopy` with an apply/undo move stack to eliminate object allocation overhead
- **Jump-over Rule** — implement the official Quoridor rule allowing a player to jump over an adjacent opponent
- **MCTS** — replace Minimax with Monte Carlo Tree Search for better handling of the high branching factor

---

## 🛠️ Tech Stack

- **Language:** Python 3.8+
- **Graphics:** Pygame
- **AI:** Custom Minimax + Alpha-Beta Pruning (no external AI libraries)
- **Data Structures:** Sets (O(1) wall lookup), BFS queue (path validation)

---

## 👤 Author

**Kavish Gupta**
- GitHub: [@Kavish1504](https://github.com/Kavish1504)
- LinkedIn: [Kavish Gupta](https://www.linkedin.com/in/kavish-gupta-3a072435a/)
- Email: kavishgupta21177@gmail.com

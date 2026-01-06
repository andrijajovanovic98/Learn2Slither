# Learn2Slither

A simple **Q-learning-based agent** for the classic Snake game.
The goal is to implement a `QAgent` class from scratch and train it to make better decisions during gameplay.

<img width="802" height="639" alt="Learn2Slither gameplay screenshot" src="https://github.com/user-attachments/assets/26c1cd9b-f7db-4511-8f7f-ac52be14b171" />

---

## Core Idea

The snake uses a **Q-table** to learn which moves lead to the highest rewards in different game situations.
This is done using **reinforcement learning**, specifically the **Q-learning** algorithm.

---

## What you see in the game

- The snake is controlled by a **Q-learning agent**.
- The board contains **green apples** (good) and **red apples** (bad).
- The agent learns by trial and error: it tries actions, gets rewards, and updates its **Q-table**.

---

## Key Concepts

### `encode_state(head, snake_cells, apples_kinds, size)`

This function converts the current game situation into a **discrete state representation**.
The snake "looks" in 4 directions (up, down, left, right) from its head, and encodes the **first thing it sees** in each direction (wall, apple, snake body, etc.).

Example output:
`("W", "Z", "G", "S")`

Where:
- `W` = wall
- `Z` = nothing (empty cell)
- `G` = green apple
- `R` = red apple
- `S` = snake body

This encoded state is used as a key in the Q-table.

---

## Q-table

The **Q-table** is a Python dictionary that maps each state to a list of 4 values — one for each possible action:

    Q[state] = [q_up, q_down, q_left, q_right]

Action order: `UP, DOWN, LEFT, RIGHT`.

### Example: How the Agent Makes Decisions

    Episode: 42
    Current State: ('Z', 'Z', 'G', 'W')  # nothing ahead, nothing left, green apple right, wall behind
    Q-values: [0.12, -0.05, 0.89, -1.00]  # [up, down, left, right]

    Decision:
     - The highest Q-value is 0.89 for action LEFT
     - Snake moves LEFT

---

## ε (epsilon) — Exploration vs Exploitation

Epsilon controls whether the agent:

- **Explores** (random move to discover new strategies)
- **Exploits** (chooses the best known move from the Q-table)

At the beginning of training, exploration is high (more randomness),
but over time epsilon decreases, so the agent relies more on what it has learned.

### What does “greedy” mean?

"Greedy" means the agent always chooses the action with the highest Q-value for the current state:

    best_action = argmax(Q[state])

This balance between exploration (random) and exploitation (greedy) is essential for learning effective strategies.

---

## Rewards

The agent receives a numeric reward after each action. These rewards drive the Q-value updates.

| Event                              | Reward |
|------------------------------------|:------:|
| Step (survive / just move)         | -0.001 |
| Eat green apple                    | +1.0   |
| Eat red apple                      | -1.0   |
| Death (wall/self-collision/len≤0)  | -10.0  |

**Q-learning update (one step):**

    Q[s,a] = Q[s,a] + α * ( r + γ * max_{a'} Q[s',a'] - Q[s,a] )

These values encourage survival and collecting green apples, while penalizing risky or terminal actions.

---

## How to Run

### Train (headless, faster)

    python3 src/main.py --episodes 10000 --save models/qtable_10x10.pkl

### Evaluate a trained model (no learning)

    python3 src/main.py --visual --dontlearn --load models/qtable_10x10.pkl

---

## Command-line Flags

- `--episodes N` number of training episodes
- `--visual` enable pygame view
- `--dontlearn` disable learning (evaluation mode)
- `--save PATH` save Q-table at the end
- `--load PATH` load an existing Q-table
- `--board N` board size (default 10)

# Section 1: Introduction

Figgie is a multi-agent, imperfect-information game designed to simulate a market where participants have limited knowledge of traded items.

Unlike standard card games, Figgie features two differences:
1. The deck composition is unknown and stochastic, unlike Poker where each suit always has 13 cards.
2.  Cards are not numbered; an agent cannot distinguish between two cards of the same suit, forcing a focus on tracking aggregate inventory rather than specific cards.

## 1.1 Deck Structure and Target Logic

A Figgie deck contains 40 cards. While the suits are randomized, the distribution of counts is constant:
- 12 cards (The "Long" suit)
- 10 cards (Neutral 1)
- 10 cards (Neutral 2)
- 8 cards (The "Short" suit)

The Target Suit (the only suit with terminal value) is the suit of the same color as the 12-card suit.

### All 12 Possible Deck Configurations
| ♠ (Black) | ♣ (Black) | ♥ (Red) | ♦ (Red) | Target Suit | Target Scarcity |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 10 | 10 | 8 | *12* | ♥ (Red) | 8 (Short) |
| 10 | 10 | *12* | 8 | ♦ (Red) | 8 (Short) |
| 8 | 10 | 10 | *12* | ♥ (Red) | 10 (Neutral) |
| 10 | *12* | 8 | 10 | ♠ (Black) | 10 (Neutral) |
| 10 | 8 | 10 | *12* | ♥ (Red) | 10 (Neutral) |
| *12* | 8 | 10 | 10 | ♣ (Black) | 8 (Short) |
| *12* | 10 | 8 | 10 | ♣ (Black) | 10 (Neutral) |
| 8 | 10 | *12* | 10 | ♦ (Red) | 10 (Neutral) |
| 8 | *12* | 10 | 10 | ♠ (Black) | 8 (Short) |
| 10 | 8 | *12* | 10 | ♦ (Red) | 10 (Neutral) |
| *12* | 10 | 10 | 8 | ♣ (Black) | 10 (Neutral) |
| 10 | *12* | 10 | 8 | ♠ (Black) | 10 (Neutral) |

## 1.2 The Objective Function

Agents aim to maximize a terminal payout function $P$:
$$P = (N_{target} \times 10) + \text{Bonus} + \text{Cash on Hand}$$

- **$N_{target}$:** Number of target cards held at the end of the round.
- **The Bonus:** Awarded to the player with the largest Target Suit inventory. 
    *   **$120$** if the Target is the "Short" (8-card) suit.
    *   **$100$** if the Target is a "Neutral" (10-card) suit.
- **Cash on Hand:** The net result of all trading activity during the round.

## 1.3 The `figgie_gym` Simulation Environment

To train Reinforcement Learning agents, we developed `figgie_gym`, a high-throughput matching engine wrapped in a **PettingZoo Parallel Environment**. 

### The Order Matching Engine
The engine discretizes the continuous nature of trading into "ticks."

*   **Observations:** At each tick, the market transmits the public state: Best Bid, Best Offer, Last Traded Price, and the full Trade Tape (identities of buyer/seller, price, and asset). Individual hands remain private.
*   **Simultaneous Actions:** Agents submit 16 quote prices every tick ($a \in \mathbb{R}^{16}$):
    *   **Limit Orders (4 Bids / 4 Asks):** Publicly displayed, providing liquidity but risking adverse selection.
    *   **Sniper Orders (4 Bids / 4 Asks):** Non-displayed, Fill-or-Kill (FoK) orders used to cross the spread aggressively.
    * **Accelerated Gameplay**: Sending two pairs of quotes help accelerate game progress and improve training efficiency by shortening the feedback cycle.
    * **Price-tick and Trade Size**: Order prices are integers, and quantities are always 1 card.
*   **Randomized Execution:** To eliminate "seat-order bias," the engine shuffles agent execution priority at every tick, ensuring no agent has a persistent advantage in the queue.
*   **Number of Players**: We assume that we always have 5 players in a game; each player starts with 8 cards.

### Parallelism and Autocorrelation
We instantiate multiple environments in parallel to:
1.  Collect diverse trajectories in a single training iteration.
2.  Break the temporal autocorrelation inherent in a single game's time-series.
3.  Batch-compute forward passes of neural networks for computational efficiency.

## 1.4 Agent Taxonomy

| Agent Type | Logic Foundation | Role in Training |
| :--- | :--- | :--- |
| **Noise** | $U(0, 10)$ Random Quotes | Uninformed liquidity provision. |
| **Combinatorial** | Bayesian Card Counting | Baseline intelligent agent (Fundamentalist). |
| **Supervised** | Neural Deck Prediction | Evaluates belief accuracy without adaptive policy. |
| **PPO Agent** | Proximal Policy Optimization | The primary subject: learns to balance risk, belief, and strategic execution. |
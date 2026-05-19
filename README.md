# Figgie Gym

A market simulation environment for multi-agent trading games. 
Agents trade cards in different suits in an order-book-driven marketplace where price discovery occurs as trades and quotes reveal previously hidden information. 
Operating under imperfect information, agents must perform real-time belief updates to adjust their quoting strategies to prevail in the game.

## Game Mechanics

The game uses a 40-card deck with a randomized, uneven distribution:
- **The "Long" Suit:** 12 cards.
- **Two "Neutral" Suits:** 10 cards each.
- **The "Short" Suit:** 8 cards.

The cards within a suit are unranked and indistinguishable; 
the exact distribution of the deck is unknown to the agents at the start of the round.

### The Payout Logic
The **"Target Suit"** is defined as the suit of the same color as the "Long" suit (e.g., if Hearts are 12, Diamonds are the Target). 
The Target Suit itself will either have 10 or 8 cards.

The **"Terminal Value"** of each Target Suit card is worth **$10** at the end of the round. All other cards are worthless ($0).
A **bonus:** is awarded to the player holding the most Target cards.
- If the Target is a 10-card suit (Neutral): **$100 bonus**.
- If the Target is an 8-card suit (Short): **$120 bonus**.

*Note: In the event of a tie for the most cards, the bonus is split equally among the winners.*

## Scoring and Strategy

To succeed, agents must excel in the following domains:

### Information Gathering
Agents only observe a private subset of the deck.
One must infer the hidden distribution by reading the "tape" - observing opponents' trading behavior and pricing aggressiveness.

### Risk Management
As non-target cards expire worthless, agents must manage inventory risk and avoid carrying excessive inventory as the round concludes.

### Market Making
Agents must balance the bid-ask spread. A tight spread increases the probability of a fill to monetize private information, while a wider spread mitigates the risk of adverse selection from better-informed opponents.

## Game Implementation Details

### The Order Matching Engine
- Order Types: Supports standard limit orders and "sniper" (non-displayed, Fill-or-Kill) orders.
- Information Set: The trade ticker tape is public (Buyer, Seller, Price, and Asset). Individual hands remain private.
- Auction Dynamics: Trading is continuous and simultaneous. At each tick, the engine collects quotes from all agents and processes them in a randomized sequence.

### Agent Action Space
Agents are required to provide two-way quotes (Bid and Ask) for all four suits every tick.
- Quote Prices: $a \in \mathbb{R}^{16}$ (4 suits $\times$ 2 order types [Limit/Sniper] $\times$ 2 sides [Bid/Ask]).
- There are no constraints on price or spread; agents may set arbitrarily wide spreads or "far away" prices if they do not wish to transact.
- Each order is fixed at 1 card per transaction for simplicity.

### Agents
This repository includes baseline heuristics and model-based agents:

1.  Noise Agents: Quotes bids and asks drawn from a uniform distribution $U(0, 10)$ for each suit.
2.  Card-Counting Agents: Use combinatorics to compute the posterior probability of the suit distribution given observed cards, and a heuristic to set quotes based on that belief.
    - Roughly equivalent to the "fundamental" approach in [this paper](https://arxiv.org/pdf/2110.00879).
3.  Supervised Model Agents: Replace the combinatorics approach with a supervised learning model to predict probabilities.
4.  PPO Agents: Utilize Proximal Policy Optimization to learn quoting strategies directly from market interaction.

## Bespoke Neural Network Architecture

To improve sample efficiency, the architecture exploits the inherent symmetries of the game:

- Agent Symmetry: Swapping agent "seats" should not change decision-making logic. We impose a **permutation-invariant** structure using **Deep Sets**.
- Suit & Color Symmetry: Swapping suits of the same color (or swapping colors entirely) should only result in a corresponding swap of the output quotes. We utilize **permutation-equivariant** layers across like-colored suits and colors to enforce this logic.


### PPO Architecture: The Three-Network Approach
Beyond the standard Actor-Critic framework, we introduce a third component:
-   The Pricer Network: Specifically trained via supervised learning to predict the probability distribution of the Target Suit. It provides "unbiased" probability anchors.
-   The Actor Network: Optimizes quotes using the Pricer’s output as a foundational input. It learns to maximize the expected advantage.
-   The Critic Network: Evaluates the Actor’s performance and estimates the value function to calculate policy advantage.

#### Reward Shaping
True rewards are sparse and only revealed at the end of the game. 
To mitigate this, we compute "mark-to-market" (MtM) fair prices using the agent's **Pricer Network**. 
We then evaluate tick-by-tick theoretical portfolio value changes. 
**Generalized Advantage Estimation (GAE)** is applied to these MtM changes to provide a dense signal for the Critic network.

## Code Structure

```text
 figgie_gym/
 ├── notebooks/
 │   └── ...
 ├── src/figgie_gym/
 │   ├── agent  # Heuristic and RL-based agents
 │   ├── env    # PettingZoo wrapper for the matching engine
 │   ├── market # The market order matching engine
 │   └── models # Neural architectures (Permutation Equivariant/Invariant layers)
 └── (other utilities)

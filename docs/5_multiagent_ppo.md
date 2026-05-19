# Section 5: Multi-Agent Coordination via MAPPO

Figgie is fundamentally a game of hidden information and strategic interaction. To handle the complexities of a multi-agent environment, we implement **Multi-Agent PPO (MAPPO)**. This framework allows us to stabilize the "moving target" problem—where an agent’s optimal strategy shifts as its opponents also learn and adapt.

## 5.1 Centralized Training, Decentralized Execution (CTDE)

We adopt the **CTDE paradigm** to bridge the gap between simulation-time "omniscience" and game-time "fog of war."

### 1. Centralized Critic (Training Only)
During the training phase, the **Value Network (Critic)** is "centralized." It is granted access to the **Global State ($S$)**, which includes:
*   The ground-truth deck configuration (the hidden target suit).
*   The exact inventory and cash levels of all opponents.
*   The private hands of every agent in the game.

By observing the full state, the Critic can accurately judge whether an agent's move was truly "advantageous" or merely lucky, drastically reducing the noise in the policy gradient.

### 2. Decentralized Actor (Inference/Execution)
Despite the Critic’s global view, the **Actor Network** remains strictly "decentralized." At execution time, the Actor (the `EquivariantBody`) only sees:
*   The public trade tape (LOB data).
*   Its own private inventory and card counts.
*   Market-wide pricing signals.

This ensures that the resulting policy is robust and does not rely on "cheating" or information leakage, as the Actor must learn to *infer* the hidden target suit solely from behavioral cues.

## 5.2 Stabilizing the Competitive Landscape

In a standard competitive RL setup, agents often descend into "policy oscillation"—where Agent A learns to exploit Agent B, then Agent B adapts, and the cycle continues without convergence. MAPPO mitigates this through:

*   **Shared Experience Buffer:** We train the agents against a diverse pool of opponents, including "frozen" versions of themselves from earlier iterations and fixed-heuristic "Card-Counters."
*   **Global Advantage Normalization:** By using a centralized baseline, the advantage signal $\hat{A}_t$ is normalized across the entire agent population. This prevents a single "lucky" agent from dominating the gradient updates and forcing the population toward a sub-optimal, high-variance strategy.

## 5.3 Symmetry-Aware Parameter Sharing

To maximize sample efficiency across the population, we utilize **Parameter Sharing** among all learning agents. Because the `EquivariantBody` already handles seat-invariance (Agent Invariance), a single set of weights can control all agents in the simulation simultaneously. 

> **Result:** This creates a "Self-Play" curriculum where the model is constantly challenged by its own best strategies, forcing it to develop sophisticated counter-bluffing and defensive quoting behaviors that would be impossible to learn against static noise agents.
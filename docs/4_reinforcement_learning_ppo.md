# Section 4: The Equivariant PPO Agent

Having established a superior symmetry-aware architecture, we utilize it as the shared backbone for a multi-headed PPO agent. 
By baking the "laws of the deck" into the foundation, we allow the agent to bypass the discovery of basic game rules and focus on high-level strategic competition.

## 4.1 Tri-Head Architecture

The agent is composed of three specialized networks, all sharing the same Equivariant Body architecture.
This ensures that features learned by one head—such as detecting an opponent's aggressive bluff—benefit the others.
We create separate bodies for each head as it was empircally shown to display higher numerical stability. 

1.  **The Pricer Network (Supervised):** 
    Trained via Cross-Entropy to predict the target suit.
    It provides an unbiased "Fair Value" estimate ($V_f$) by decoding the market's behavioral signals.
2.  **The Actor Network (Policy):** 
    Determines the market-making "spreading" strategy.
    It maps the current market context to specific quoting parameters across all four suits.
3.  **The Value Network (Critic):** 
    Estimates the expected total round return (PnL + Target Bonus).
    It serves as the baseline for the PPO advantage calculation, reducing gradient variance during training.

## 4.2 Action Space: Parametric Quoting

The Pricer serves as an anchor for all quotes an agent produces.
The actor outputs a 5-dimensional vector per suit to parameterize its quoting strategy.

### The Quoting Engine
The first four outputs are passed through a **Softmax** layer to ensure strictly positive values, while the fifth represents a directional **Bias**. For each suit $s$:

*   **Spreads ($\Delta_s$):** 
    Defines the distance from the mid-price.
    A wider spread captures more profit per trade and acts as a buffer against inventory risk.
*   **Bias ($b_s$):** 
    Allows the agent to deviate from the "Fair Value" $V_f$. 
    Unlike a traditional inventory skew, this bias is strategically used to "pay up" (bid above fair) to aggressively acquire cards in a suspected target suit, chasing the game-end bonus.

The resulting quotes are generated as:
$$ \text{Bid}_s = V_{f,s} + b_s - \Delta_s $$
$$ \text{Ask}_s = V_{f,s} + b_s + \Delta_s $$

> **Strategic Guardrail:** 
While the bias allows for aggressive pursuit of the bonus, it is internally constrained by a hyperbolic tangent transformation. 
If the Pricer is uncertain, the Actor naturally defaults to wider, more defensive spreads to minimize "toxic" order flow.

## 4.3 Reward Shaping: Marking-to-Market

A primary challenge in Reinforcement Learning for trading is the "sparsity" of rewards—realized PnL only occurs at the end of a round. 
To solve this, we utilize our supervised Pricer network to create a dense, "Mark-to-Market" (MtM) reward signal at every simulation tick.

### Tick-Level Rewards
At each time step $t$, the agent's reward is the change in its theoretical portfolio value. 
We use the Pricer's unbiased probability output to calculate a **Theoretical Fair Value** for our inventory. 
The reward $r_t$ is defined as:
$$ r_t = (\text{Cash}_t + \sum_{s} \text{Pos}_{s,t} \cdot V_{f,s,t}) - (\text{Cash}_{t-1} + \sum_{s} \text{Pos}_{s,t-1} \cdot V_{f,s,t-1}) $$

This ensures the Actor is rewarded immediately for buying a card below its theoretical fair value, even if the round hasn't ended.

### Final Payout & Terminal Reward
At the final step $T$, the reward is the delta between the actual game payout (100-point bonus + card values) and the final theoretical value estimated by the Pricer:
$$ r_T = \text{Final Payout} - \text{Portfolio Value}_{T-1} $$

This terminal "correction" forces the agent to reconcile its beliefs with reality, punishing it if the Pricer was overconfident or if it failed to capture the target suit bonus.

## 4.4 Learning Targets & Advantage Estimation

To train the Value and Actor networks, we employ **Generalized Advantage Estimation (GAE)**. 
GAE allows us to balance the trade-off between bias and variance in our policy gradients.

*   **Value Network Target:** 
We train the Critic to predict the $\lambda$-weighted return $G_t$, helping it understand the long-term potential of a specific market state and inventory configuration.
*   **Actor Update:** The PPO update uses the GAE advantage $\hat{A}_t$, which tells the Actor whether a specific quoting bias ($b_s$) or spread ($\Delta_s$) led to a better-than-average MtM gain.

> **Why GAE matters here:** 
Because Figgie is a game of shifting information, early-game rewards are highly uncertain. 
GAE's temporal smoothing prevents the agent from overreacting to "noise" trades in the first 10 ticks, while still allowing it to learn the high-value urgency of the final 10 ticks.
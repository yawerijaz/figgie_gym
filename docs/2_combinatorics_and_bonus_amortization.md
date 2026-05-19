# Section 2: Combinatorics & Strategic Baselines

The core challenge of Figgie lies in synthesizing private hand information with the continuous market data flow. From the perspective of a successful agent, every new transaction on the public tape must be used to filter the likelihood of the 12 possible hidden states of the deck.

Our baseline "Card-Counter" is modeled as a combinatorial estimator, inspired by the "fundamentalist" approach described in:
> *[Traders in a Strange Land: Agent-based discrete-event market simulation of the Figgie card game](https://arxiv.org/pdf/2110.00879) (Ozerov et al., 2021).*

## 2.1 The "No Short-Sell" Constraint

In Figgie, players cannot sell cards they do not own. This "No Short-Sell" rule transforms the public trade ticker into a ledger of physical constraints. We track the **inventory floor** for each of the four opponents ($i \in \{1, 2, 3, 4\}$):

*   **Initial State:** At $t=0$, we only know our private hand $H_{self}$. For all other players, $L_{i,s} = 0$.
*   **Transaction Update:** When a trade occurs (e.g., Player S sells 1 Spade to Player B):
    *   **The Buyer ($B$):** $L_{B,spade} \leftarrow L_{B,spade} + 1$.
    *   **The Seller ($S$):** $L_{S,spade} \leftarrow \max(0, L_{S,spade} - 1)$.
*   **The Known Deck:** At any time $t$, the number of "accounted for" cards in suit $s$ is:
    $$N_{known, s} = H_{self, s} + \sum_{i=1}^{4} L_{i,s}$$

## 2.2 Combinatorics to Calculating Posteriors

For a candidate deck $D_k$, let $C_{s,k}$ be the total supply of suit $s$. The number of ways ($W$) to distribute the remaining latent cards among the hidden slots of our opponents is:

$$
W(D_k) = \frac{(40 - \sum N_{known, s})!}{\prod_{s} (C_{s,k} - N_{known, s})!} 
\quad \text{if all } C_{s,k} \ge N_{known, s} \text{, else } 0
$$

The posterior probability for deck $D_k$ is normalized across all 12 valid configurations: $P(D_k) = W(D_k) / \sum W(D_j)$. This allows the calculation of the **Expected Value (EV)** of the payout per card.

## 2.3 Bonus Amortization

The $100$ or $120$ bonus is a binary, terminal reward. To avoid passive play, we amortize the bonus $B$ by treating the valuation of the $i$-th card as a **Geometric Sequence**. We define the amortized bonus $b_i$ as a vector:

$$
b = \left[ 
    \frac{g^1}{\sum_{j=1}^{\lceil C/2\rceil}g^j} B, \dots, \frac{g^{\lceil C/2\rceil}}{\sum_{j=1}^{\lceil C/2\rceil}g^j} B, 0, \dots, 0
\right]
$$

Where $g \in (0, 1)$ reflects the agent's aggressiveness. Finally, the **break-even purchase price** $V$ for the $n$-th card in suit $s$ is:
$$V(n, s) = \sum_{D_k \in \{\text{Target } s\}} P(D_k) \times (10 + b_{n})$$

## 2.4 Baseline Combinatorics Agent Quoting

In a standard market, the fair bid is below the fair ask. However, the Card-Counter often encounters **Bid-Ask Inversion** when nearing a majority. 

Assume an agent holds 5 cards and is 50% certain the suit is the target. The market is at $8 bid / $15 ask. 
*   **Buying at $15:** If $b_n = 30$, the net gain is $((30+10) \times 0.5) - 15 = \$5$.
*   **Selling at $8:** The gain is $8 - (10 \times 0.5) = \$3$ (forgoing the card value).

In this scenario, both crossing the book to buy and sell at current market levels are EV-positive. The only "wrong" action is to remain idle.

### Illustrative Breakeven Prices
Notice the sharp inversion at $n=5$, where the agent is willing to buy at a higher price than they would sell to protect/clinch the bonus.

<Image src="plots/2_breakeven_price_card_counting.png" alt="Plot showing fair buy and sell prices for Figgie. Fair buy exceeds fair sell at 5 cards due to the bonus majority threshold." caption="Price Inversion at the Bonus Threshold" />

| Holding | Fair Buy | Fair Sell |
| :--- | :--- | :--- |
| 0 | 0.52 | 0.94 |
| 1 | 0.94 | 1.78 |
| 2 | 1.78 | 3.63 |
| 3 | 3.63 | 9.08 |
| 4 | 9.08 | **24.03** |
| 5 | **24.03** | 4.86 |
| 6 | 4.86 | 4.78 |

In simulation, if $V_{bid} > V_{ask}$, we compute the mid-price and apply a narrow spread. For "Sniper" quotes, we use the raw inverted values to ensure maximum execution probability.
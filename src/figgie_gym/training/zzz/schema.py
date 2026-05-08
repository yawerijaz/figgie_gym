import pyarrow as pa

trade_summary_struct = pa.struct(
    [
        pa.field("buy_quantity", type=pa.float64()),
        pa.field("buy_consideration", type=pa.float64()),
        pa.field("sell_quantity", type=pa.float64()),
        pa.field("sell_consideration", type=pa.float64()),
        pa.field("min_net_quantity_change", type=pa.float64()),
    ],
)

agent_trade_summaries = pa.list_(trade_summary_struct)

symbol_summary_struct = pa.struct(
    {
        pa.field("symbol", type=pa.string()),
        pa.field("market_bid", type=pa.float64()),
        pa.field("market_bid_volume", type=pa.float64()),
        pa.field("market_ask", type=pa.float64()),
        pa.field("market_ask_volume", type=pa.float64()),
        pa.field("last_price", type=pa.float64()),
        pa.field("volume", type=pa.float64()),
        pa.field("self_position", type=pa.float64()),
        pa.field("known_position", type=pa.float64()),
        pa.field("agent_summaries", type=agent_trade_summaries),
    },
)

symbol_summaries = pa.list_(symbol_summary_struct)

game_data_schema = pa.schema(
    [
        pa.field("agent_type", pa.string()),
        pa.field("step", pa.float64()),
        pa.field("steps_remaining", pa.float64()),
        pa.field("cash", pa.float64()),
        pa.field("per_suit", symbol_summaries),
        pa.field("reward", pa.float64()),
        pa.field("hidden.agent", pa.string()),
        pa.field("hidden.seed", pa.float64()),
        pa.field("hidden.experiment", pa.float64()),
        pa.field("hidden.goal_suit", pa.string()),
        pa.field("hidden.game_num_noise_agents", pa.float64()),
        pa.field("hidden.game_num_cardcounter_agents", pa.float64()),
        pa.field("hidden.terminal_reward", pa.float64()),
    ],
)

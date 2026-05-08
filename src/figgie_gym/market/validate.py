from figgie_gym.market.common import Order
from figgie_gym.market.exception import UnmatchableTradeError


def validate_matchable(hitting_order: Order, standing_order: Order) -> None:
    if hitting_order.side == standing_order.side:
        raise UnmatchableTradeError(hitting_order, standing_order)

    if hitting_order.symbol != standing_order.symbol:
        raise UnmatchableTradeError(hitting_order, standing_order)

    if (
        hitting_order.price * hitting_order.side
        < standing_order.price * hitting_order.side
    ):
        raise UnmatchableTradeError(hitting_order, standing_order)
    if hitting_order.client == standing_order.client:
        raise UnmatchableTradeError(hitting_order, standing_order)

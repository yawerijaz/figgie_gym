from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

from figgie_gym.market.client import MarketClient
from figgie_gym.market.common import (
    Client,
    MarketTime,
    Order,
    Price,
    Quantity,
    Quote,
    Symbol,
    Trade,
    TradeSummary,
)
from figgie_gym.market.order_book import OrderBook

if TYPE_CHECKING:
    from collections.abc import Hashable


class Market:
    def __init__(self, symbols: list[Symbol]) -> None:
        self.subscribers = list[Client]()
        self.books = {sym: OrderBook() for sym in symbols}
        self.trade_summary = self.create_trade_summarizer()
        self.spotter = self.create_spotter()
        self.time = MarketTime(0)
        self.market_id = uuid4()

    def advance_clock(self) -> None:
        self.time = MarketTime(self.time + 1)

    def subscribe(self, client: Client) -> None:
        return self.subscribers.append(client)

    def process_order(self, order: Order) -> None:
        book = self.books[order.symbol]
        trades, order_status_updates = book.process_order(order)
        for subscriber in self.subscribers:
            for order_update in order_status_updates:
                if order_update.order.client is subscriber:
                    subscriber.on_new_order_update(order_update)
            subscriber.on_new_trades(self.time, trades)
            subscriber.on_new_spot(order.symbol, book.top())
            if isinstance(subscriber, OrderTickerTape):
                subscriber.on_new_order(self.time, order, "insert")

    def cancel_order(self, order: Order) -> None:
        book = self.books[order.symbol]
        maybe_update = book.cancel_order(order)
        if maybe_update:
            for subscriber in self.subscribers:
                if maybe_update.order.client is subscriber:
                    subscriber.on_new_order_update(maybe_update)
                subscriber.on_new_spot(order.symbol, book.top())
                if isinstance(subscriber, OrderTickerTape):
                    subscriber.on_new_order(self.time, order, "cancel")

    def create_trade_summarizer(self) -> TradeSummarizer:
        summarizer = TradeSummarizer()
        self.subscribe(summarizer)
        return summarizer

    def create_spotter(self) -> Spotter:
        spotter = Spotter()
        self.subscribe(spotter)
        return spotter

    def create_trade_ticker_tape(self) -> TradeTickerTape:
        ticker_tape = TradeTickerTape()
        self.subscribe(ticker_tape)
        return ticker_tape

    def create_order_ticker_tape(self) -> OrderTickerTape:
        order_tape = OrderTickerTape()
        self.subscribe(order_tape)
        return order_tape

    def new_client(
        self,
        client_id: Hashable = None,
        initial_positions: dict[Symbol, Quantity] | None = None,
        initial_cash: int = 0,
    ) -> MarketClient:
        client = MarketClient(
            self,
            client_id,
            initial_positions,
            initial_cash,
        )
        self.subscribe(client)
        return client


class TradeSummarizer(Client):
    def __init__(self) -> None:
        self.client_id = "TradeSummarizer"
        self.summary = defaultdict[Symbol, defaultdict[Client, TradeSummary]](
            lambda: defaultdict[Client, TradeSummary](TradeSummary.new),
        )
        self.last_prices = defaultdict[Symbol, Price | None](lambda: None)
        self.last_trade_times = defaultdict[Symbol, MarketTime | None](
            lambda: None,
        )
        self.volumes = defaultdict[Symbol, Quantity](lambda: Quantity(0))

    def on_new_trades(self, time: MarketTime, trades: list[Trade]) -> None:
        for trade in trades:
            self.summary[trade.symbol][trade.buyer] += trade
            self.summary[trade.symbol][trade.seller] -= trade
            self.last_prices[trade.symbol] = trade.price
            self.last_trade_times[trade.symbol] = time
            self.volumes[trade.symbol] = Quantity(
                self.volumes[trade.symbol] + trade.quantity,
            )


class Spotter(Client):
    def __init__(self) -> None:
        self.client_id = "Spotter"
        self.spots = defaultdict[Symbol, Quote](lambda: Quote(None, None))

    def on_new_spot(self, symbol: Symbol, quote: Quote) -> None:
        self.spots[symbol] = quote


class TradeTickerTape(Client):
    def __init__(self) -> None:
        self.client_id = "TradeTickerTape"
        self.trades = list[Trade]()
        self.timestamped_trades = list[tuple[MarketTime, Trade]]()

    def on_new_trades(self, time: MarketTime, trades: list[Trade]) -> None:
        self.trades.extend(trades)
        self.timestamped_trades.extend([(time, t) for t in trades])


class OrderTickerTape(Client):
    def __init__(self) -> None:
        self.client_id = "OrderTickerTape"
        self.timestamped_orders = list[
            tuple[MarketTime, Order, Literal["insert", "cancel"]]
        ]()

    def on_new_order(
        self,
        time: MarketTime,
        order: Order,
        event: Literal["insert", "cancel"],
    ) -> None:
        return self.timestamped_orders.append((time, order, event))

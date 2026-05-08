from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from jinja2 import Environment, FileSystemLoader, select_autoescape
from tabulate import tabulate

from figgie_gym.market.client import MarketClient
from figgie_gym.market.common import Client, Symbol, Trade, TradeSummary
from figgie_gym.market.market import Market, OrderTickerTape, TradeTickerTape
from figgie_gym.market.order_book import OrderBook


def print_to_file(
    fileio: TextIO,
    order_book: OrderBook,
    summaries: Mapping[Client, TradeSummary],
    trades: list[Trade],
) -> None:
    fileio.write(display_ladder(order_book))
    fileio.write("\n\n")
    fileio.write(display_trade_summaries(summaries))
    fileio.write("\n\n")
    fileio.write(display_trades(trades))
    fileio.write("\n\n")


def display_ladder(order_book: OrderBook) -> str:
    top = order_book.top()
    ladder = reversed(
        [
            [qb or "", p, qa or ""]
            for p, (qb, qa) in (order_book.ladder()).items()
        ],
    )
    headers = [
        f"bid\n{top.bid.quantity if top.bid else ''}",
        f"price\n{top.bid.price if top.bid else ''} @ {top.ask.price if top.ask else ''}",
        f"ask\n{top.ask.quantity if top.ask else '': >3}",
    ]

    return tabulate(
        ladder,
        headers=headers,
        stralign="center",
        tablefmt="fancy_grid",
    )


def display_trade_summaries(summaries: Mapping[Client, TradeSummary]) -> str:
    data: list[list[Any]] = [
        [
            c,
            summary.average_buy_price(),
            summary.buy_quantity,
            summary.net_quantity_change(),
            summary.sell_quantity,
            summary.average_sell_price(),
            summary.min_net_quantity_change,
        ]
        for c, summary in summaries.items()
    ]
    return tabulate(
        data,
        headers=[
            "client",
            "avg_buy",
            "buy_qty",
            "net_qty",
            "sell_qty",
            "avg_sell",
            "min_net_qty_change",
        ],
    )


def display_trades(trades: list[Trade]) -> str:
    data: list[list[Any]] = [
        [
            t.buyer,
            t.seller,
            t.price,
            t.quantity,
        ]
        for t in trades
    ]
    return tabulate(
        data,
        headers=[
            "buyer",
            "seller",
            "price",
            "quantity",
        ],
    )


def _read_static_asset(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


class MarketDisplay:
    def __init__(
        self,
        market: Market,
        ticker_tape: TradeTickerTape,
        order_tape: OrderTickerTape | None,
        filepath: str,
        *,
        goal_suit: Symbol | None = None,
    ) -> None:
        self.market = market
        self.ticker_tape = ticker_tape
        self.order_tape = order_tape
        self.filepath = filepath
        self.goal_suit = goal_suit

    def print_html(
        self,
        specific_symbol: Symbol | None = None,
        filepath: Path | None = None,
    ) -> str:
        path = Path(filepath or self.filepath)
        symbols = (
            [specific_symbol]
            if specific_symbol is not None
            else list(self.market.books.keys())
        )

        # Collect clients (MarketClient instances are subscribers with order_managers)
        clients: list[MarketClient] = [
            sub
            for sub in self.market.subscribers
            if isinstance(sub, MarketClient)
        ]
        client_ids = [str(c.client_id) for c in clients]

        symbols_data: list[dict[str, Any]] = []
        for symbol in symbols:
            book = self.market.books[symbol]
            top = book.top()
            ladder_rows = [
                {"bid": qb or "", "price": p, "ask": qa or ""}
                for p, (qb, qa) in (book.ladder()).items()
            ]
            ladder_rows.reverse()

            # trade summaries
            summaries: list[dict[str, str | float]] = []
            for c, summary in self.market.trade_summary.summary[symbol].items():
                summaries.append(
                    {
                        "client": str(
                            c.client_id if hasattr(c, "client_id") else c,
                        ),
                        "avg_buy": summary.average_buy_price(),
                        "buy_qty": summary.buy_quantity,
                        "net_qty": summary.net_quantity_change(),
                        "sell_qty": summary.sell_quantity,
                        "avg_sell": summary.average_sell_price(),
                        "min_net_qty_change": summary.min_net_quantity_change,
                    },
                )

            # trades for this symbol
            trades = [
                {
                    "trade_time": ts,
                    "buyer": str(t.buyer),
                    "seller": str(t.seller),
                    "price": t.price,
                    "quantity": t.quantity,
                    "aggressive_side": t.aggressive_side,
                }
                for ts, t in sorted(
                    self.ticker_tape.timestamped_trades,
                    key=lambda a: (a[0], a[1].trade_id),
                    reverse=True,
                )
                if t.symbol == symbol
            ]
            # orders for this symbol (if an order_tape is present)
            orders: list[dict[str, Any]] = []
            if self.order_tape is not None:
                orders = [
                    {
                        "time": ts,
                        "client": str(o.client),
                        "event": ev,
                        "side": o.side.name.lower(),
                        "price": o.price,
                        "quantity": o.quantity,
                        "order_id": o.order_id,
                        "order_type": o.order_type.value
                        if hasattr(o, "order_type")
                        else None,
                        "symbol": str(o.symbol),
                    }
                    for ts, o, ev in sorted(
                        self.order_tape.timestamped_orders,
                        key=lambda a: (a[0], getattr(a[1], "order_id", 0)),
                        reverse=True,
                    )
                    if o.symbol == symbol
                ]
            # unique clients that appear in trades for this symbol (as strings)
            trade_client_set: set[str] = set()
            for tr in trades:
                trade_client_set.add(str(tr["buyer"]))
                trade_client_set.add(str(tr["seller"]))
            trade_clients = sorted(trade_client_set)

            symbols_data.append(
                {
                    "name": str(symbol),
                    "ladder": ladder_rows,
                    "top": {
                        "bid_price": top.bid.price if top and top.bid else "",
                        "bid_quantity": top.bid.quantity
                        if top and top.bid
                        else "",
                        "ask_price": top.ask.price if top and top.ask else "",
                        "ask_quantity": top.ask.quantity
                        if top and top.ask
                        else "",
                    },
                    "trade_summaries": summaries,
                    "trades": trades,
                    "orders": orders,
                    "trade_clients": trade_clients,
                },
            )

        # holdings topline table: for each symbol, show whether it is the env goal and each client's position
        holdings: list[dict[str, object]] = []
        winners: list[bool] = [False for _ in client_ids]
        for symbol in symbols:
            positions: list[dict[str, int]] = []
            for client in clients:
                # initial position (if available on the MarketClient)
                init_q = client.initial_positions.get(symbol, 0)
                init_pos = int(init_q)
                # final position from order manager
                final_q = client.order_managers[symbol].quantity
                final_pos = int(final_q)

                positions.append({"initial": init_pos, "final": final_pos})

            holdings.append(
                {
                    "symbol": str(symbol),
                    "is_target": bool(
                        self.goal_suit is not None and symbol == self.goal_suit,
                    ),
                    "positions": positions,
                },
            )
            if symbol == self.goal_suit:
                max_pos = max(p["final"] for p in positions)
                winners = [p["final"] >= max_pos for p in positions]

        # Add a summary row for cash positions (not a symbol, but useful to view here)
        if clients:
            cash_positions: list[dict[str, object]] = []
            for client in clients:
                init_cash = client.initial_cash
                final_cash = client.cash
                cash_positions.append(
                    {"initial": init_cash, "final": final_cash},
                )

            holdings.append(
                {
                    "symbol": "cash",
                    "is_target": False,
                    "positions": cash_positions,
                },
            )
        base, env = jinja_env()
        template = env.get_template("market_snapshot.html")

        css = _read_static_asset(base / "css" / "market.css")
        js = _read_static_asset(base / "js" / "collapse.js")

        rendered = template.render(
            title="Figgie Market Snapshot",
            generated_at=datetime.now(UTC).isoformat() + "Z",
            css=css,
            js=js,
            symbols=symbols_data,
            clients=client_ids,
            holdings=holdings,
            goal_symbol=str(self.goal_suit)
            if self.goal_suit is not None
            else None,
            winners=winners,
        )
        with path.open("w", encoding="utf-8") as file:
            file.write(rendered)
        return path.as_uri()

    @staticmethod
    def print_redirect_page(new_url: str, write_to_path: Path) -> Path:
        _, env = jinja_env()
        template = env.get_template("redirect.html")
        rendered = template.render(new_url=new_url)
        with write_to_path.open("w", encoding="utf-8") as file:
            file.write(rendered)
        return write_to_path

    def print(self, symbol: Symbol | None = None) -> None:
        path = Path(self.filepath)
        # If filepath looks like an HTML file, produce a friendly web report using jinja templates.
        if path.suffix.lower() == ".html":
            self.print_html(symbol)

        else:
            with path.open("w") as file:
                symbols = (
                    [symbol] if symbol is not None else self.market.books.keys()
                )
                for s in symbols:
                    file.write(f"Market Snapshot for Symbol: {s}\n")
                    print_to_file(
                        file,
                        self.market.books[s],
                        self.market.trade_summary.summary[s],
                        [t for t in self.ticker_tape.trades if t.symbol == s],
                    )


def jinja_env() -> tuple[Path, Environment]:
    # Load template and static assets from package static/
    base = Path(__file__).parent.parent / "static"
    template_dir = base / "templates"
    return base, Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(["html", "xml"]),
    )

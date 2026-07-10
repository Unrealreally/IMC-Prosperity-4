"""
Round 5 Trading Strategy
========================

Products & Strategies
---------------------

1. SNACKPACK (5 variants): MARKET MAKING
   - Chocolate, Pistachio, Raspberry, Strawberry, Vanilla
   - Why: highest spread/vol ratio in the entire market (0.08-0.10),
     meaning spread (~17 ticks) dwarfs intraday volatility (~170-200).
     Virtually zero adverse selection risk. Post bids/asks 1 tick inside
     the market and harvest the spread continuously.
   - Fair value: EMA of mid_price updated each timestamp, nudged by
     recent observation trades.

2. MICROCHIP_OVAL: DIRECTIONAL SHORT
   - Why: Price declined -7%, -20%, -26% intraday on days 2, 3, 4
     (consistently and strongly bearish across ALL observed days).
   - Strategy: Short at session open and ride the downtrend to close.
     Use observation trades as confirmation signals.

3. PEBBLES_XS: DIRECTIONAL SHORT
   - Why: Price declined -20%, -15%, -12% intraday on all three observed
     days. Strongest and most consistent downtrend in the dataset.
   - Strategy: Same as MICROCHIP_OVAL — short from open, hold to close.
"""

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import Dict, List
import numpy as np
import json


# ── Constants ────────────────────────────────────────────────────────────────
POSITION_LIMITS: Dict[str, int] = {
    # SNACKPACK – market making
    "SNACKPACK_CHOCOLATE": 50,
    "SNACKPACK_PISTACHIO": 50,
    "SNACKPACK_RASPBERRY": 50,
    "SNACKPACK_STRAWBERRY": 50,
    "SNACKPACK_VANILLA": 50,
    # Directional shorts
    "MICROCHIP_OVAL": 50,
    "PEBBLES_XS": 50,
}

SNACKPACK_PRODUCTS = {
    "SNACKPACK_CHOCOLATE",
    "SNACKPACK_PISTACHIO",
    "SNACKPACK_RASPBERRY",
    "SNACKPACK_STRAWBERRY",
    "SNACKPACK_VANILLA",
}

DIRECTIONAL_SHORT = {
    "MICROCHIP_OVAL",
    "PEBBLES_XS",
}

# EMA smoothing – smaller α = slower, more stable fair-value estimate
EMA_ALPHA_SNACKPACK = 0.20   # slow EMA; spread is wide so drift matters little
EMA_ALPHA_DIRECTIONAL = 0.12  # slightly faster to track the trend

# Market-making parameters
MM_EDGE = 2            # ticks inside the best bid/ask we quote
MM_INVENTORY_SKEW = 5  # shrink quote size per unit of net position
MM_MAX_SPREAD = 25     # don't quote if market spread is absurdly wide

# Directional strategy parameters
DIR_ENTRY_THRESHOLD = 0.0   # enter at open unconditionally
DIR_MAX_POSITION = 50       # always push to full limit on short side


# ── Helper utilities ──────────────────────────────────────────────────────────
def best_bid(order_depth: OrderDepth):
    """Return (price, volume) of the best bid, or (None, None)."""
    if order_depth.buy_orders:
        price = max(order_depth.buy_orders)
        return price, order_depth.buy_orders[price]
    return None, None


def best_ask(order_depth: OrderDepth):
    """Return (price, volume) of the best ask, or (None, None)."""
    if order_depth.sell_orders:
        price = min(order_depth.sell_orders)
        return price, order_depth.sell_orders[price]
    return None, None


def mid_price(order_depth: OrderDepth):
    """Compute mid price from best bid/ask; return None if either side absent."""
    bp, _ = best_bid(order_depth)
    ap, _ = best_ask(order_depth)
    if bp is None or ap is None:
        return None
    return (bp + ap) / 2.0


def update_ema(current_ema: float | None, new_value: float, alpha: float) -> float:
    if current_ema is None:
        return new_value
    return alpha * new_value + (1 - alpha) * current_ema


# ── State container (serialised into traderData) ─────────────────────────────
class TraderState:
    """Lightweight persistent state across timestamps."""

    def __init__(self):
        self.ema: Dict[str, float] = {}          # fair-value EMA per product
        self.obs_ema: Dict[str, float] = {}      # EMA of observation-trade prices
        self.timestamp: int = 0

    def to_json(self) -> str:
        return json.dumps({
            "ema": self.ema,
            "obs_ema": self.obs_ema,
            "timestamp": self.timestamp,
        })

    @classmethod
    def from_json(cls, data: str) -> "TraderState":
        obj = cls()
        if not data:
            return obj
        try:
            d = json.loads(data)
            obj.ema = d.get("ema", {})
            obj.obs_ema = d.get("obs_ema", {})
            obj.timestamp = d.get("timestamp", 0)
        except Exception:
            pass
        return obj


# ── Market-making logic ───────────────────────────────────────────────────────
def market_make(
    product: str,
    order_depth: OrderDepth,
    position: int,
    fair_value: float,
    state: TraderState,
) -> List[Order]:
    """
    Quote both sides around fair_value, 1-2 ticks inside the current book.
    Skew quote sizes based on current inventory to remain delta-neutral.
    """
    orders: List[Order] = []
    limit = POSITION_LIMITS[product]

    bp, bv = best_bid(order_depth)
    ap, av = best_ask(order_depth)

    if bp is None or ap is None:
        return orders

    market_spread = ap - bp
    if market_spread > MM_MAX_SPREAD:
        # Market is dislocated – don't trade
        return orders

    # Our quotes are just inside the market's best prices
    our_bid = bp + MM_EDGE
    our_ask = ap - MM_EDGE

    # Ensure our quotes don't cross each other or the fair value significantly
    if our_bid >= our_ask:
        our_bid = int(fair_value) - 1
        our_ask = int(fair_value) + 1

    # Inventory-based size skew: reduce size on the side we're already long/short
    base_size = max(1, limit // 4)

    # Skew: if long, reduce bid size and increase ask size (and vice-versa)
    skew = int(position * MM_INVENTORY_SKEW / limit)
    bid_size = max(1, base_size - skew)
    ask_size = max(1, base_size + skew)

    # Cap to available room
    room_to_buy = limit - position
    room_to_sell = limit + position  # position can be negative

    bid_size = min(bid_size, room_to_buy)
    ask_size = min(ask_size, room_to_sell)

    if bid_size > 0:
        orders.append(Order(product, our_bid, bid_size))
    if ask_size > 0:
        orders.append(Order(product, our_ask, -ask_size))

    return orders


# ── Directional short logic ───────────────────────────────────────────────────
def directional_short(
    product: str,
    order_depth: OrderDepth,
    position: int,
) -> List[Order]:
    """
    Aggressively build a maximum short position """
    orders: List[Order] = []
    limit = POSITION_LIMITS[product]

    # Current short position is represented as a negative number
    current_short = -position  # e.g. position=-30 → current_short=30
    target_short = DIR_MAX_POSITION
    additional_short_needed = target_short - current_short

    if additional_short_needed <= 0:
        return orders  # Already at max short

    bp, bv = best_bid(order_depth)
    ap, av = best_ask(order_depth)

    if bp is None:
        return orders

    # Hit the best bid to sell (short). Volume limited to what's available.
    sell_qty = min(additional_short_needed, abs(bv) if bv else additional_short_needed)
    if sell_qty > 0:
        orders.append(Order(product, bp, -sell_qty))

    return orders


def directional_unwind(
    product: str,
    order_depth: OrderDepth,
    position: int,
    is_end_of_day: bool,
) -> List[Order]:
    """
    Only unwind the short if we're close to end of day.
    Otherwise hold the position.
    """
    orders: List[Order] = []
    if not is_end_of_day:
        return orders

    if position >= 0:
        return orders  # Nothing to unwind

    ap, av = best_ask(order_depth)
    if ap is None:
        return orders

    # Buy to cover at best ask
    cover_qty = min(-position, abs(av) if av else -position)
    if cover_qty > 0:
        orders.append(Order(product, ap, cover_qty))

    return orders


# ── Main Trader class ─────────────────────────────────────────────────────────
class Trader:
    """
    Implements three strategies:

    • SNACKPACK × 5  → Market making (best spread/vol in the market)
    • MICROCHIP_OVAL → Directional short (−7% to −26% per day, every day)
    • PEBBLES_XS     → Directional short (−12% to −20% per day, every day)
    """

    # Timestamp at which we start unwinding shorts (out of 1_000_000)
    UNWIND_START_TS = 950_000

    def run(self, state: TradingState):
        trader_state = TraderState.from_json(state.traderData or "")
        trader_state.timestamp = state.timestamp

        result: Dict[str, List[Order]] = {}
        is_end_of_day = state.timestamp >= self.UNWIND_START_TS

        # ── Update observation-trade EMAs ─────────────────────────────────────
        for trade in state.market_trades.values():
            for t in trade:
                sym = t.symbol
                alpha = EMA_ALPHA_SNACKPACK if sym in SNACKPACK_PRODUCTS else EMA_ALPHA_DIRECTIONAL
                trader_state.obs_ema[sym] = update_ema(
                    trader_state.obs_ema.get(sym),
                    t.price,
                    alpha,
                )

        # ── Update mid-price EMA (fair value) ────────────────────────────────
        for product, od in state.order_depths.items():
            if product not in POSITION_LIMITS:
                continue
            mp = mid_price(od)
            if mp is None:
                continue
            alpha = (
                EMA_ALPHA_SNACKPACK if product in SNACKPACK_PRODUCTS
                else EMA_ALPHA_DIRECTIONAL
            )
            # Blend mid_price EMA with observation-trade EMA (if available)
            obs = trader_state.obs_ema.get(product)
            if obs is not None:
                blended = 0.85 * mp + 0.15 * obs
            else:
                blended = mp
            trader_state.ema[product] = update_ema(
                trader_state.ema.get(product), blended, alpha
            )

        # ── Place orders ──────────────────────────────────────────────────────
        for product in POSITION_LIMITS:
            if product not in state.order_depths:
                continue

            od = state.order_depths[product]
            position = state.position.get(product, 0)
            orders: List[Order] = []

            # ── SNACKPACK: market making ──────────────────────────────────────
            if product in SNACKPACK_PRODUCTS:
                fv = trader_state.ema.get(product)
                if fv is not None:
                    orders = market_make(product, od, position, fv, trader_state)

            # ── Directional shorts ────────────────────────────────────────────
            elif product in DIRECTIONAL_SHORT:
                if is_end_of_day:
                    orders = directional_unwind(product, od, position, True)
                else:
                    orders = directional_short(product, od, position)

            if orders:
                result[product] = orders

        conversions = 0
        return result, conversions, trader_state.to_json()

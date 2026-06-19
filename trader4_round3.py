"""
IMC Prosperity Round 3 — Trading Algorithm
==========================================
"""

from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List
import json
import math


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def bs_call_price(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 1e-10 or sigma <= 1e-10:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    if T <= 1e-10 or sigma <= 1e-10:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)


def implied_vol(market_price: float, S: float, K: float, T: float, r: float = 0.0, tol: float = 1e-6) -> float | None:
    if T <= 1e-10:
        return None
    intrinsic = max(S - K, 0.0)
    if market_price <= intrinsic + tol:
        return None
    lo, hi = 1e-4, 10.0
    f_lo = bs_call_price(S, K, T, lo, r) - market_price
    f_hi = bs_call_price(S, K, T, hi, r) - market_price
    if f_lo * f_hi > 0:
        return None
    for _ in range(60):
        mid = (lo + hi) / 2.0
        f_mid = bs_call_price(S, K, T, mid, r) - market_price
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return (lo + hi) / 2.0


TOTAL_TICKS = 3_000_000
TRUE_SIGMA = 0.03543

POSITION_LIMITS = {
    "VELVETFRUIT_EXTRACT": 400,
    "VEV_4000": 200,
    "VEV_4500": 200,
    "VEV_5000": 200,
    "VEV_5100": 200,
    "VEV_5200": 200,
    "VEV_5300": 200,
    "VEV_5400": 200,
    "VEV_5500": 200,
    "VEV_6000": 200,
    "VEV_6500": 200,
    "HYDROGEL_PACK": 600,
}

VOUCHER_STRIKES = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    "VEV_6000": 6000,
    "VEV_6500": 6500,
}

OVERPRICED_OTM = {"VEV_6000", "VEV_6500"}
ATM_VOUCHERS = {"VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"}
ITM_VOUCHERS = {"VEV_4000", "VEV_4500"}

MM_SPREAD_HYDROGEL = 5
MM_SPREAD_UNDERLYING = 2
MM_SPREAD_ATM_VOUCHER = 1
MM_SPREAD_ITM_VOUCHER = 2

OTM_SELL_PREMIUM_THRESHOLD = 0.10
OTM_SELL_QTY = 10


def best_bid(od: OrderDepth):
    return max(od.buy_orders.keys()) if od.buy_orders else None


def best_ask(od: OrderDepth):
    return min(od.sell_orders.keys()) if od.sell_orders else None


def mid_price(od: OrderDepth):
    bb, ba = best_bid(od), best_ask(od)
    if bb is not None and ba is not None:
        return (bb + ba) / 2.0
    if bb is not None:
        return float(bb)
    if ba is not None:
        return float(ba)
    return None


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


class Trader:
    def __init__(self):
        self.sigma_history: List[float] = []
        self.underlying_history: List[float] = []

    def run(self, state: TradingState):
        if state.traderData and state.traderData != "":
            try:
                data = json.loads(state.traderData)
                self.sigma_history = data.get("sigma_history", [])
                self.underlying_history = data.get("underlying_history", [])
            except Exception:
                self.sigma_history = []
                self.underlying_history = []

        result: Dict[str, List[Order]] = {}
        conversions = 0

        elapsed = state.timestamp
        T_raw = max(TOTAL_TICKS - elapsed, 100) / TOTAL_TICKS

        und_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        S = mid_price(und_od) if und_od else None

        if S is not None:
            self.underlying_history.append(S)
            if len(self.underlying_history) > 200:
                self.underlying_history = self.underlying_history[-200:]

        sigma = self._estimate_sigma(state, S, T_raw)
        if sigma is not None:
            self.sigma_history.append(sigma)
            if len(self.sigma_history) > 50:
                self.sigma_history = self.sigma_history[-50:]

        effective_sigma = sum(self.sigma_history) / len(self.sigma_history) if self.sigma_history else TRUE_SIGMA
        effective_sigma = clamp(effective_sigma, 0.025, 0.15)

        if und_od and S is not None:
            result["VELVETFRUIT_EXTRACT"] = self._mm_orders(
                "VELVETFRUIT_EXTRACT",
                od=und_od,
                fair=S,
                half_spread=MM_SPREAD_UNDERLYING,
                position=state.position.get("VELVETFRUIT_EXTRACT", 0),
                limit=POSITION_LIMITS["VELVETFRUIT_EXTRACT"],
                qty=10,
            )

        hp_od = state.order_depths.get("HYDROGEL_PACK")
        if hp_od:
            result["HYDROGEL_PACK"] = self._mm_orders(
                "HYDROGEL_PACK",
                od=hp_od,
                fair=10_000.0,
                half_spread=MM_SPREAD_HYDROGEL,
                position=state.position.get("HYDROGEL_PACK", 0),
                limit=POSITION_LIMITS["HYDROGEL_PACK"],
                qty=20,
            )

        if S is not None:
            for product, K in VOUCHER_STRIKES.items():
                od = state.order_depths.get(product)
                if od is None:
                    continue

                pos = state.position.get(product, 0)
                limit = POSITION_LIMITS[product]

                if product in OVERPRICED_OTM:
                    orders = self._sell_overpriced_otm(product, od, S, K, T_raw, pos, limit)
                elif product in ATM_VOUCHERS:
                    fair = bs_call_price(S, K, T_raw, effective_sigma)
                    orders = self._mm_orders(product, od, fair, MM_SPREAD_ATM_VOUCHER, pos, limit, 5)
                elif product in ITM_VOUCHERS:
                    fair = bs_call_price(S, K, T_raw, effective_sigma)
                    orders = self._mm_orders(product, od, fair, MM_SPREAD_ITM_VOUCHER, pos, limit, 3)
                else:
                    orders = []

                if orders:
                    result[product] = orders

            hedge_orders = self._delta_hedge(state, S, T_raw, effective_sigma)
            if hedge_orders:
                existing = result.get("VELVETFRUIT_EXTRACT", [])
                result["VELVETFRUIT_EXTRACT"] = existing + hedge_orders

        trader_data = json.dumps({
            "sigma_history": self.sigma_history,
            "underlying_history": self.underlying_history,
        })

        return result, conversions, trader_data

    def _estimate_sigma(self, state: TradingState, S: float | None, T: float) -> float | None:
        if S is None or T < 0.001:
            return None
        ivs = []
        for product in ATM_VOUCHERS:
            K = VOUCHER_STRIKES[product]
            od = state.order_depths.get(product)
            if od is None:
                continue
            m = mid_price(od)
            if m is None:
                continue
            iv = implied_vol(m, S, K, T)
            if iv is not None and 0.01 < iv < 0.20:
                ivs.append(iv)
        return sum(ivs) / len(ivs) if ivs else None

    def _mm_orders(
        self,
        product: str,
        od: OrderDepth,
        fair: float,
        half_spread: float,
        position: int,
        limit: int,
        qty: int,
    ) -> List[Order]:
        orders: List[Order] = []

        skew = -position / limit * half_spread * 0.5
        bid_price = round(fair - half_spread + skew)
        ask_price = round(fair + half_spread + skew)

        if bid_price >= ask_price:
            ask_price = bid_price + 1

        bb = best_bid(od)
        ba = best_ask(od)

        buy_capacity = limit - position
        if buy_capacity > 0:
            buy_qty = min(qty, buy_capacity)
            if ba is not None and ba <= bid_price:
                orders.append(Order(product, ba, buy_qty))
            else:
                orders.append(Order(product, bid_price, buy_qty))

        sell_capacity = position + limit
        if sell_capacity > 0:
            sell_qty = min(qty, sell_capacity)
            if bb is not None and bb >= ask_price:
                orders.append(Order(product, bb, -sell_qty))
            else:
                orders.append(Order(product, ask_price, -sell_qty))

        return orders

    def _sell_overpriced_otm(
        self,
        product: str,
        od: OrderDepth,
        S: float,
        K: int,
        T: float,
        position: int,
        limit: int,
    ) -> List[Order]:
        orders: List[Order] = []
        fair = bs_call_price(S, K, T, TRUE_SIGMA)
        sell_capacity = position + limit

        if sell_capacity <= 0:
            return orders

        for bid_px in sorted(od.buy_orders.keys(), reverse=True):
            if bid_px >= fair + OTM_SELL_PREMIUM_THRESHOLD:
                vol = min(OTM_SELL_QTY, sell_capacity, abs(od.buy_orders[bid_px]))
                if vol > 0:
                    orders.append(Order(product, bid_px, -vol))
                    sell_capacity -= vol
            else:
                break

        ba = best_ask(od)
        ask_px = max(1, math.ceil(fair + OTM_SELL_PREMIUM_THRESHOLD))
        if ba is not None:
            ask_px = min(ask_px, ba)
        ask_px = max(ask_px, 1)

        if sell_capacity > 0:
            orders.append(Order(product, ask_px, -min(OTM_SELL_QTY, sell_capacity)))

        return orders

    def _delta_hedge(
        self,
        state: TradingState,
        S: float,
        T: float,
        sigma: float,
    ) -> List[Order]:
        net_option_delta = 0.0
        for product, K in VOUCHER_STRIKES.items():
            pos = state.position.get(product, 0)
            if pos == 0:
                continue
            delta = bs_delta(S, K, T, sigma)
            net_option_delta += pos * delta

        und_pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
        total_delta = net_option_delta + und_pos

        und_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        und_limit = POSITION_LIMITS["VELVETFRUIT_EXTRACT"]

        target_trade = -round(total_delta)
        if target_trade == 0 or und_od is None:
            return []

        orders: List[Order] = []
        bb = best_bid(und_od)
        ba = best_ask(und_od)

        if target_trade > 0:
            capacity = und_limit - und_pos
            qty = min(target_trade, capacity, 20)
            if qty > 0 and ba is not None:
                orders.append(Order("VELVETFRUIT_EXTRACT", ba, qty))
        else:
            capacity = und_pos + und_limit
            qty = min(-target_trade, capacity, 20)
            if qty > 0 and bb is not None:
                orders.append(Order("VELVETFRUIT_EXTRACT", bb, -qty))

        return orders
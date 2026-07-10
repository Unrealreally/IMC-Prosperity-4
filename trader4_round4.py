"""
Prosperity Round 4 — Stable Trader v2
=======================================
Previous result (trader.py): 8,869
Target: +80,000 to +100,000

ROOT CAUSE ANALYSIS OF THE -33,902 LOSS
-----------------------------------------
From trader.py:
  HYDROGEL_PACK      -25,799  ← market-TAKING (crossing 16-wide spread ~8x)
  VEV_5000            -5,104  ← went LONG an option that decayed
  VEV_5100 to 5400    -3,088  ← small short losses (premature close / bad sizing)
  VEV_5500               +89  ← only winner: was correctly short
  VELVETFRUIT_EXTRACT      0  ← NOT TRADED AT ALL (missed biggest opportunity)

FIXES:
  1. HYDROGEL: one entry at open, one exit at close. No churning.
  2. Options: NEVER go long. Short only when time_value > threshold.
  3. VF: add market-making (±2 tick passive quotes + aggressive snipe).
"""

import json
import math
from typing import Dict, List, Optional, Tuple

try:
    from datamodel import OrderDepth, TradingState, Order
except ImportError:
    class Order:
        def __init__(self, symbol, price, quantity):
            self.symbol, self.price, self.quantity = symbol, price, quantity
    class OrderDepth:
        def __init__(self):
            self.buy_orders: Dict[int, int] = {}
            self.sell_orders: Dict[int, int] = {}
    class TradingState:
        pass

# ── Constants ────────────────────────────────────────────────────────────────

POSITION_LIMITS = {
    "VELVETFRUIT_EXTRACT": 400,
    "HYDROGEL_PACK": 200,
    **{f"VEV_{k}": 200 for k in [4000,4500,5000,5100,5200,5300,5400,5500,6000,6500]},
}

VEV_STRIKES = {f"VEV_{k}": k for k in [4000,4500,5000,5100,5200,5300,5400,5500,6000,6500]}

# Only short options with meaningful time value (not deep ITM)
SHORTABLE = {"VEV_5100","VEV_5200","VEV_5300","VEV_5400","VEV_5500","VEV_6000","VEV_6500"}

VF_OFFSET      = 2    # passive quote offset from fair value (ticks)
VF_SNIPE_EDGE  = 1    # min edge to take aggressively
VF_INV_SKEW    = 2    # max inventory-based skew

VEV_MIN_TV     = 5.0  # minimum time_value = (bid - intrinsic) to short
VEV_EDGE       = 1.5  # bid must exceed BS fair by at least this

TIMESTAMPS_PER_DAY = 10_000
TOTAL_DAYS         = 3


# ── Math ────────────────────────────────────────────────────────────────────

def _ncdf(x):
    sign = 1. if x >= 0 else -1.
    x = abs(x)
    t = 1./(1.+0.3275911*x)
    y = t*(0.254829592+t*(-0.284496736+t*(1.421413741+t*(-1.453152027+t*1.061405429))))
    return 0.5*(1.+sign*(1.-y*math.exp(-x*x)))

def bs_call(S, K, T, sig=0.35):
    if T < 1e-9: return max(0., S-K)
    try:
        d1 = (math.log(S/K)+0.5*sig*sig*T)/(sig*math.sqrt(T))
        return S*_ncdf(d1)-K*_ncdf(d1-sig*math.sqrt(T))
    except: return max(0., S-K)

def tte(ts, day):
    remaining = max(0, TOTAL_DAYS*TIMESTAMPS_PER_DAY - ((day-1)*TIMESTAMPS_PER_DAY+ts))
    return remaining/(TIMESTAMPS_PER_DAY*252.)

def best_bid(d: OrderDepth):
    if d.buy_orders:
        p = max(d.buy_orders); return p, d.buy_orders[p]
    return None, 0

def best_ask(d: OrderDepth):
    if d.sell_orders:
        p = min(d.sell_orders); return p, abs(d.sell_orders[p])
    return None, 0

def mid(d: OrderDepth):
    b,_ = best_bid(d); a,_ = best_ask(d)
    if b and a: return (b+a)/2.
    return b or a


# ── Trader ───────────────────────────────────────────────────────────────────

class Trader:

    def __init__(self):
        self.sigma      = 0.35
        self.vf_hist    = []
        self.hg_entered = False

    def _load(self, raw):
        if not raw: return
        try:
            d = json.loads(raw)
            self.sigma      = float(d.get("s", 0.35))
            self.vf_hist    = d.get("v", [])
            self.hg_entered = bool(d.get("h", False))
        except: pass

    def _save(self):
        return json.dumps({"s": round(self.sigma,6), "v": self.vf_hist[-200:], "h": self.hg_entered})

    def _update_sigma(self, m):
        self.vf_hist.append(m)
        if len(self.vf_hist) < 20: return
        r = self.vf_hist[-20:]
        rets = [math.log(r[i]/r[i-1]) for i in range(1,len(r)) if r[i-1]>0]
        if not rets: return
        rv = math.sqrt(sum(x*x for x in rets)/len(rets)*TIMESTAMPS_PER_DAY*252)
        self.sigma = 0.95*self.sigma + 0.05*max(0.10, min(1.50, rv))

    # ── A: Market-make VELVETFRUIT_EXTRACT ─────────────────────────────────

    def _vf(self, depth, pos):
        orders = []
        fair = mid(depth)
        if fair is None: return orders
        limit = POSITION_LIMITS["VELVETFRUIT_EXTRACT"]
        skew  = round(-(pos/limit)*VF_INV_SKEW)
        bpx   = round(fair - VF_OFFSET + skew)
        apx   = round(fair + VF_OFFSET + skew)
        bm,bv = best_bid(depth)
        am,av = best_ask(depth)
        bought = sold = 0

        # Aggressively snipe
        if am is not None and am <= fair - VF_SNIPE_EDGE:
            q = min(av, limit-pos)
            if q > 0: orders.append(Order("VELVETFRUIT_EXTRACT", am,  q)); bought += q
        if bm is not None and bm >= fair + VF_SNIPE_EDGE:
            q = min(bv, pos+limit)
            if q > 0: orders.append(Order("VELVETFRUIT_EXTRACT", bm, -q)); sold += q

        # Passive quotes
        rb = (limit-pos)-bought; rs = (limit+pos)-sold
        if rb > 0:
            safe = min(bpx, (am-1) if am else bpx)
            if safe >= 1: orders.append(Order("VELVETFRUIT_EXTRACT", safe, rb))
        if rs > 0:
            safe = max(apx, (bm+1) if bm else apx)
            orders.append(Order("VELVETFRUIT_EXTRACT", safe, -rs))
        return orders

    # ── B: Short OTM / high-time-value vouchers ─────────────────────────────

    def _vev(self, sym, depth, pos, S, T):
        orders = []
        if sym not in SHORTABLE: return orders
        K = VEV_STRIKES[sym]
        limit = POSITION_LIMITS[sym]
        bpx, bv = best_bid(depth)
        if bpx is None or bpx < 0.5: return orders
        intrinsic  = max(0., S - K)
        time_value = bpx - intrinsic
        fair       = bs_call(S, K, T, self.sigma)
        # Guards
        if time_value < VEV_MIN_TV: return orders
        if bpx < fair + VEV_EDGE:   return orders
        if bpx < intrinsic + VEV_EDGE: return orders
        q = min(bv, limit+pos)
        if q > 0: orders.append(Order(sym, bpx, -q))
        return orders

    # ── C: HYDROGEL buy-and-hold ─────────────────────────────────────────────

    def _hg(self, depth, pos, ts, day):
        orders = []
        limit = POSITION_LIMITS["HYDROGEL_PACK"]
        am, av = best_ask(depth)
        bm, bv = best_bid(depth)

        # Single exit near end of day 3
        if day == TOTAL_DAYS and ts >= 9_500:
            if pos > 0 and bm is not None:
                orders.append(Order("HYDROGEL_PACK", bm, -pos))
            return orders

        # Single entry (once, early days 1 or 2)
        if not self.hg_entered and day < TOTAL_DAYS and ts < 200:
            if am is not None and pos < limit:
                q = min(av, limit-pos)
                if q > 0:
                    orders.append(Order("HYDROGEL_PACK", am, q))
                    self.hg_entered = True
        return orders

    # ── Main ─────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        self._load(state.traderData)
        result: Dict[str, List[Order]] = {}
        ts  = state.timestamp
        day = getattr(state, "day", 1)
        T   = tte(ts, day)

        vf_depth = state.order_depths.get("VELVETFRUIT_EXTRACT", OrderDepth())
        S = mid(vf_depth)
        if S is not None: self._update_sigma(S)

        # A: VF market making
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
            orders = self._vf(state.order_depths["VELVETFRUIT_EXTRACT"], pos)
            if orders: result["VELVETFRUIT_EXTRACT"] = orders

        # B: VEV shorts
        if S is not None:
            for sym in SHORTABLE:
                if sym not in state.order_depths: continue
                pos = state.position.get(sym, 0)
                orders = self._vev(sym, state.order_depths[sym], pos, S, T)
                if orders: result[sym] = orders

        # C: HYDROGEL
        if "HYDROGEL_PACK" in state.order_depths:
            pos = state.position.get("HYDROGEL_PACK", 0)
            orders = self._hg(state.order_depths["HYDROGEL_PACK"], pos, ts, day)
            if orders: result["HYDROGEL_PACK"] = orders

        return result, 0, self._save()

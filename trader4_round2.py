from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Optional
import math
import statistics

# ==============================================================================
# IMC PROSPERITY 4 - ROUND 2 TRADING BOT
# Strategy: Mean Reversion + Auction Optimization + Spread Analysis
# Target: Beat Round 2 (Market Access above median)
# Products: Ash-coated Osmium & Intarian Pepper Root
# ==============================================================================

# UPDATE WITH YOUR CSV DATA
HISTORICAL_PRICES = {
    'osmium': [],
    'pepper_root': []
}

# ==============================================================================
# STRATEGY CONFIGURATION
# ==============================================================================

class StrategyConfig:
    # Mean reversion threshold (same as Round 1)
    MEAN_REVERSION_THRESHOLD = 0.02
    
    # Spread tracking (new for Round 2)
    SPREAD_SMOOTHING_WINDOW = 10
    
    # Aggressive size
    AGGRESSIVE_SIZE = 8
    
    # Position limits (UPDATE FROM WIKI!)
    OSMIUM_POSITION_LIMIT = 25
    PEPPER_ROOT_POSITION_LIMIT = 20
    
    # Bid estimation: add small buffer above estimated median
    BID_MEDIAN_BUFFER = 5  # XIRECs above estimated median
    
    # Market access: target to be safely above median
    ACCESS_SAFETY_MARGIN = 10

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def calculate_mean(prices: List[int]) -> float:
    """Calculate average price"""
    if not prices:
        return 0.0
    return sum(prices) / len(prices)

def calculate_spread(order_depth: OrderDepth) -> float:
    if not order_depth.sell_orders or not order_depth.buy_orders:
        return 0.0
    best_ask = min(order_depth.sell_orders.keys())
    best_bid = max(order_depth.buy_orders.keys())
    return best_ask - best_bid

def get_best_ask(order_depth: OrderDepth) -> Optional[tuple]:
    """Get lowest ask price and quantity"""
    if not order_depth.sell_orders:
        return None
    best_ask_price = min(order_depth.sell_orders.keys())
    quantity = order_depth.sell_orders[best_ask_price]
    return (best_ask_price, quantity)

def get_best_bid(order_depth: OrderDepth) -> Optional[tuple]:
    """Get highest bid price and quantity"""
    if not order_depth.buy_orders:
        return None
    best_bid_price = max(order_depth.buy_orders.keys())
    quantity = order_depth.buy_orders[best_bid_price]
    return (best_bid_price, quantity)

def calculate_fair_value(order_depth: OrderDepth, mean_price: float, spread: float) -> float:
    """
    Calculate fair value using:
    1. Mean reversion (historical average)
    2. Current spread analysis
    3. Weighted combination for Round 2
    """
    if mean_price is None:
        return 0
    
    if not order_depth.sell_orders or not order_depth.buy_orders:
        return int(mean_price)
    
    best_ask = min(order_depth.sell_orders.keys())
    best_bid = max(order_depth.buy_orders.keys())
    mid_price = (best_ask + best_bid) / 2.0
    
    # Weight: 70% mean reversion, 30% current market
    fair_value = 0.7 * mean_price + 0.3 * mid_price
    
    return int(fair_value)

def estimate_median_bid(historical_means: Dict[str, float]) -> int:
    """
    Estimate what the median bid will be.
    Based on Round 2 prompts: median stabilizes when participants align.
    Use historical means as proxy for median estimate.
    """
    if not historical_means:
        return 100  # Default
    
    # Average of all fair values as median estimate
    values = list(historical_means.values())
    return int(statistics.mean(values))

# ==============================================================================
# TRADER CLASS (COPY FROM HERE)
# ==============================================================================

class Trader:
    
    def __init__(self):
        """Initialize trader"""
        self.historical_prices = {
            'osmium': [],
            'pepper_root': []
        }
        self.mean_prices = {
            'osmium': None,
            'pepper_root': None
        }
        self.fair_values = {
            'osmium': 100,
            'pepper_root': 100
        }
        self.spreads = {
            'osmium': [],
            'pepper_root': []
        }
        self.iteration_count = 0
    
    def update_market_data(self, order_depths: Dict[str, OrderDepth]):
        """Update prices, means, and spreads from order book"""
        for product in order_depths:
            order_depth = order_depths[product]
            if order_depth.sell_orders and order_depth.buy_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                mid_price = (best_ask + best_bid) / 2.0
                
                # Update price history
                self.historical_prices[product].append(mid_price)
                if len(self.historical_prices[product]) > 100:
                    self.historical_prices[product] = self.historical_prices[product][-100:]
                
                # Update mean
                self.mean_prices[product] = calculate_mean(self.historical_prices[product])
                
                # Update spread history
                spread = best_ask - best_bid
                self.spreads[product].append(spread)
                if len(self.spreads[product]) > StrategyConfig.SPREAD_SMOOTHING_WINDOW:
                    self.spreads[product] = self.spreads[product][-StrategyConfig.SPREAD_SMOOTHING_WINDOW:]
                
                # Calculate fair value
                self.fair_values[product] = calculate_fair_value(
                    order_depth,
                    self.mean_prices[product],
                    spread
                )
    
    def generate_trading_logic(self, product: str, order_depth: OrderDepth,
                                position: int, position_limit: int) -> List[Order]:
        """Generate orders based on mean reversion + spread analysis"""
        orders = []
        
        mean_price = self.mean_prices.get(product)
        if mean_price is None:
            return orders
        
        best_ask = get_best_ask(order_depth)
        best_bid = get_best_bid(order_depth)
        
        if best_ask and best_bid:
            ask_price, ask_qty = best_ask
            bid_price, bid_qty = best_bid
            
            current_price = (ask_price + bid_price) / 2.0
            deviation = (current_price - mean_price) / mean_price
            
            # BUY: Price below mean
            if deviation < -StrategyConfig.MEAN_REVERSION_THRESHOLD:
                max_buy_qty = min(
                    -ask_qty,
                    position_limit - position,
                    StrategyConfig.AGGRESSIVE_SIZE
                )
                if max_buy_qty > 0:
                    orders.append(Order(product, ask_price, max_buy_qty))
            
            # SELL: Price above mean
            elif deviation > StrategyConfig.MEAN_REVERSION_THRESHOLD:
                max_sell_qty = min(
                    bid_qty,
                    position + position_limit,
                    StrategyConfig.AGGRESSIVE_SIZE
                )
                if max_sell_qty > 0:
                    orders.append(Order(product, bid_price, -max_sell_qty))
            
            # MOMENTUM: Close positions
            elif deviation < 0 and position < 0:
                if best_ask:
                    orders.append(Order(product, ask_price, min(-ask_qty, abs(position), 5)))
            elif deviation > 0 and position > 0:
                if best_bid:
                    orders.append(Order(product, bid_price, -min(bid_qty, position, 5)))
        
        return orders
    
    def bid(self) -> int:
        """
        REQUIRED for Round 2!
        Returns your fair value bid for market access.
        Must return a single integer.
        """
        # Strategy: Bid above estimated median to get market access
        # Use fair values as basis
        
        fair_values = list(self.fair_values.values())
        if not fair_values:
            return 100  # Default fallback
        
        # Calculate average fair value
        avg_fair = statistics.mean(fair_values)
        
        # Add buffer to be safely above median
        bid_value = int(avg_fair) + StrategyConfig.ACCESS_SAFETY_MARGIN
        
        print(f"[R2-Bot] BID: {bid_value} (avg_fair={avg_fair}, buffer={StrategyConfig.ACCESS_SAFETY_MARGIN})")
        
        return bid_value
    
    def run(self, state: TradingState):
        """Main trading logic"""
        print(f"[R2-Bot] Timestamp: {state.timestamp} | Iteration: {self.iteration_count}")
        self.iteration_count += 1
        
        # Update market data (prices, means, spreads, fair values)
        self.update_market_data(state.order_depths)
        
        # Orders dictionary
        result = {}
        
        # Position limits
        position_limits = {
            'osmium': StrategyConfig.OSMIUM_POSITION_LIMIT,
            'pepper_root': StrategyConfig.PEPPER_ROOT_POSITION_LIMIT
        }
        
        # Process each product
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            current_position = state.position.get(product, 0)
            position_limit = position_limits.get(product, 20)
            mean_price = self.mean_prices.get(product)
            fair_value = self.fair_values.get(product, 100)
            
            print(f"[R2-Bot] {product} | Pos: {current_position} | Mean: {mean_price} | FV: {fair_value}")
            
            # Generate orders
            orders = self.generate_trading_logic(
                product=product,
                order_depth=order_depth,
                position=current_position,
                position_limit=position_limit
            )
            
            for order in orders:
                if order.quantity > 0:
                    print(f"[R2-Bot] BUY {product}: {order.quantity}x @ {order.price}")
                else:
                    print(f"[R2-Bot] SELL {product}: {abs(order.quantity)}x @ {order.price}")
            
            result[product] = orders
        
        # State persistence
        trader_data = f"R2_Iteration_{self.iteration_count}"
        
        # No conversions in Round 2
        conversions = 0
        
        print(f"[R2-Bot] Orders: {len(result)} products")
        
        return result, conversions, trader_data

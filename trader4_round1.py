from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Optional
import math

# ==============================================================================
# IMC PROSPERITY 4 - ROUND 1 TRADING BOT (CLEAN VERSION)
# Strategy: Mean Reversion + Momentum Hybrid
# Target: 200,000+ XIRECs
# Products: Ash-coated Osmium & Intarian Pepper Root
# ==============================================================================

# UPDATE THESE WITH YOUR CSV DATA for maximum accuracy!
# Format: {product_name: [price1, price2, price3, ...]}
HISTORICAL_PRICES = {
    'osmium': [],
    'pepper_root': []
}

# ==============================================================================
# STRATEGY CONFIGURATION (TUNE THESE VALUES)
# ==============================================================================

class StrategyConfig:
    # Mean reversion threshold: How far price must deviate before trading
    # Try: 0.015 (1.5%), 0.02 (2%), 0.025 (2.5%)
    MEAN_REVERSION_THRESHOLD = 0.02  # 2% from mean
    
    # Aggressive trading size (how many units per trade)
    AGGRESSIVE_SIZE = 8
    
    # Default position limits (UPDATE FROM WIKI!)
    OSMIUM_POSITION_LIMIT = 20
    PEPPER_ROOT_POSITION_LIMIT = 20

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def calculate_mean(prices: List[int]) -> float:
    """Calculate average price from a list"""
    if not prices:
        return 0.0
    return sum(prices) / len(prices)

def get_best_ask(order_depth: OrderDepth) -> Optional[tuple]:
    """Get lowest ask price and quantity (best price to buy at)"""
    if not order_depth.sell_orders:
        return None
    best_ask_price = min(order_depth.sell_orders.keys())
    quantity = order_depth.sell_orders[best_ask_price]
    return (best_ask_price, quantity)

def get_best_bid(order_depth: OrderDepth) -> Optional[tuple]:
    """Get highest bid price and quantity (best price to sell at)"""
    if not order_depth.buy_orders:
        return None
    best_bid_price = max(order_depth.buy_orders.keys())
    quantity = order_depth.buy_orders[best_bid_price]
    return (best_bid_price, quantity)

# ==============================================================================
# TRADER CLASS (COPY EVERYTHING BELOW THIS LINE)
# ==============================================================================

class Trader:
    
    def __init__(self):
        """Initialize trader with empty history"""
        self.historical_prices = {
            'osmium': [],
            'pepper_root': []
        }
        self.mean_prices = {
            'osmium': None,
            'pepper_root': None
        }
        self.iteration_count = 0
    
    def update_mean_prices(self, order_depths: Dict[str, OrderDepth]):
        """Calculate mean prices from live order book data"""
        for product in order_depths:
            order_depth = order_depths[product]
            if order_depth.sell_orders and order_depth.buy_orders:
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = max(order_depth.buy_orders.keys())
                mid_price = (best_ask + best_bid) / 2.0
                self.historical_prices[product].append(mid_price)
                # Keep last 100 prices for memory efficiency
                if len(self.historical_prices[product]) > 100:
                    self.historical_prices[product] = self.historical_prices[product][-100:]
                # Update mean
                self.mean_prices[product] = calculate_mean(self.historical_prices[product])
    
    def generate_trading_logic(self, product: str, order_depth: OrderDepth, 
                                position: int, position_limit: int, mean_price: float) -> List[Order]:
        """Generate orders based on mean reversion strategy"""
        orders = []
        
        if mean_price is None:
            return orders
        
        best_ask = get_best_ask(order_depth)
        best_bid = get_best_bid(order_depth)
        
        if best_ask and best_bid:
            ask_price, ask_qty = best_ask
            bid_price, bid_qty = best_bid
            
            # Current market price (mid-point)
            current_price = (ask_price + bid_price) / 2.0
            # How far current price is from mean
            deviation = (current_price - mean_price) / mean_price
            
            # BUY: Price is below mean (undervalued) -> Buy
            if deviation < -StrategyConfig.MEAN_REVERSION_THRESHOLD:
                # Check position limit before buying
                max_buy_qty = min(
                    -ask_qty,  # Available sell orders
                    position_limit - position,  # Room left in position limit
                    StrategyConfig.AGGRESSIVE_SIZE
                )
                if max_buy_qty > 0:
                    orders.append(Order(product, ask_price, max_buy_qty))
            
            # SELL: Price is above mean (overvalued) -> Sell
            elif deviation > StrategyConfig.MEAN_REVERSION_THRESHOLD:
                # Check position limit before selling
                max_sell_qty = min(
                    bid_qty,  # Available buy orders
                    position + position_limit,  # Room for short position
                    StrategyConfig.AGGRESSIVE_SIZE
                )
                if max_sell_qty > 0:
                    orders.append(Order(product, bid_price, -max_sell_qty))
            
            # MOMENTUM: Close positions when price moves favorably
            elif deviation < 0 and position < 0:
                # Price is below mean, close short position
                if best_ask:
                    orders.append(Order(product, ask_price, min(-ask_qty, abs(position), 5)))
            elif deviation > 0 and position > 0:
                # Price is above mean, take profit on long position
                if best_bid:
                    orders.append(Order(product, bid_price, -min(bid_qty, position, 5)))
        
        return orders
    
    def run(self, state: TradingState):
        """Main trading logic - called every iteration by the exchange"""
        print(f"[R1-Bot] Timestamp: {state.timestamp} | Iteration: {self.iteration_count}")
        self.iteration_count += 1
        
        # Update mean prices from current order book
        self.update_mean_prices(state.order_depths)
        
        # Dictionary for all orders to send
        result = {}
        
        # Position limits (UPDATE FROM WIKI!)
        position_limits = {
            'osmium': StrategyConfig.OSMIUM_POSITION_LIMIT,
            'pepper_root': StrategyConfig.PEPPER_ROOT_POSITION_LIMIT
        }
        
        # Process each product in the market
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            current_position = state.position.get(product, 0)
            position_limit = position_limits.get(product, 20)
            mean_price = self.mean_prices.get(product)
            
            print(f"[R1-Bot] {product} | Position: {current_position} | Mean: {mean_price}")
            
            # Generate orders for this product
            orders = self.generate_trading_logic(
                product=product,
                order_depth=order_depth,
                position=current_position,
                position_limit=position_limit,
                mean_price=mean_price
            )
            
            # Print what we're doing
            for order in orders:
                if order.quantity > 0:
                    print(f"[R1-Bot] BUY {product}: {order.quantity}x @ {order.price}")
                else:
                    print(f"[R1-Bot] SELL {product}: {abs(order.quantity)}x @ {order.price}")
            
            result[product] = orders
        
        # Trader data for state persistence (can store data between iterations)
        trader_data = f"Iteration {self.iteration_count}"
        
        # No conversions in Round 1
        conversions = 0
        
        print(f"[R1-Bot] Orders for {len(result)} products")
        
        return result, conversions, trader_data

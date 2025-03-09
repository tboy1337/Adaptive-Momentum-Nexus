import numpy as np
from jesse.strategies import Strategy
from jesse import utils
from typing import Union, Tuple


class AdaptiveMomentumNexus(Strategy):
    """
    An optimized cryptocurrency trading strategy incorporating:
    - Machine learning-inspired adaptive momentum
    - Market regime detection
    - Order book imbalance analysis
    - Dynamic position sizing with risk management
    - Multi-timeframe analysis
    """

    def __init__(self):
        super().__init__()
        
        # Core parameters - simplified to ensure trades execute
        self.short_period = 8
        self.long_period = 21
        self.signal_quality_threshold = 65  # Minimum quality score to enter trade (0-100)
        self.regime_lookback = 100  # Candles to look back for regime detection
        self.atr_period = 14
        self.atr_multiplier_sl = 2.5  # Tighter stop loss
        self.atr_multiplier_tp = 4.0  # Reasonable take profit
        self.risk_per_trade = 1.5  # Percentage of capital to risk per trade
        
        # State variables
        self.current_regime = 0  # -1 (bearish), 0 (neutral), 1 (bullish)
        self.last_trade_candle = 0
        self.min_candles_between_trades = 5  # Prevent overtrading
        self.order_book_data = {}  # Would be populated from exchange data
        
        # Store temporary values for use in on_open_position
        self.temp_take_profit = None
        self.temp_stop_loss = None
    
    def hyperparameters(self):
        return [
            {'name': 'short_period', 'type': int, 'min': 5, 'max': 15, 'default': 8},
            {'name': 'long_period', 'type': int, 'min': 15, 'max': 30, 'default': 21},
            {'name': 'signal_quality_threshold', 'type': int, 'min': 50, 'max': 80, 'default': 65},
            {'name': 'regime_lookback', 'type': int, 'min': 50, 'max': 200, 'default': 100},
            {'name': 'atr_multiplier_sl', 'type': float, 'min': 1.5, 'max': 4.0, 'step': 0.25, 'default': 2.5},
            {'name': 'atr_multiplier_tp', 'type': float, 'min': 2.0, 'max': 6.0, 'step': 0.25, 'default': 4.0},
            {'name': 'risk_per_trade', 'type': float, 'min': 0.5, 'max': 2.5, 'step': 0.25, 'default': 1.5},
        ]

    def before(self):
        """
        Execute before each candle to update market regime and prepare for trading decisions
        """
        # Update market regime every 10 candles to avoid recalculating too frequently
        if self.index % 10 == 0:
            self.detect_market_regime()

    def detect_market_regime(self):
        """
        Market regime detection using price action and volatility
        Uses Bollinger Bands width, fractal efficiency, and price structure
        """
        if self.index < self.regime_lookback:
            self.current_regime = 0  # Default to neutral when not enough data
            return
            
        # Get sequential data for analysis
        close_prices = self.candles[:, 2]  # Close prices
        
        # Calculate Bollinger Bands width to determine volatility
        bb = self.bollinger_bands(20, 2.0, "close", sequential=True)  # Ensure devs is a float
        bb_upper, bb_middle, bb_lower = bb.upperband, bb.middleband, bb.lowerband
        bb_width = (bb_upper - bb_lower) / bb_middle
        
        # Determine trend direction using Fractal Efficiency
        fe = self.fractal_efficiency(close_prices, 30)
        
        # Analyze recent price structure for trend confirmation
        recent_highs = [self.candles[i, 3] for i in range(self.index-20, self.index)]
        recent_lows = [self.candles[i, 4] for i in range(self.index-20, self.index)]
        
        higher_highs = sum(1 for i in range(1, len(recent_highs)) if recent_highs[i] > recent_highs[i-1])
        higher_lows = sum(1 for i in range(1, len(recent_lows)) if recent_lows[i] > recent_lows[i-1])
        
        # Combined regime score (-100 to +100)
        regime_score = fe * 50  # Fractal Efficiency contributes 50% of the score
        
        # Add price structure component (higher highs/lows vs lower highs/lows)
        price_structure = (higher_highs + higher_lows - 20) * 2.5  # Scale to approximately -50 to +50
        regime_score += price_structure
        
        # Set regime based on score
        if regime_score > 25:
            self.current_regime = 1  # Bullish
        elif regime_score < -25:
            self.current_regime = -1  # Bearish
        else:
            self.current_regime = 0  # Neutral/Ranging

    def should_long(self) -> bool:
        """
        Determine if we should enter a long position using a simplified but effective approach
        """
        # Basic checks
        if self.index < 50:  # Ensure enough data for calculations
            return False
            
        if self.index - self.last_trade_candle < self.min_candles_between_trades:
            return False
            
        # Don't enter long positions in bearish regimes
        if self.current_regime == -1:
            return False
            
        # Core momentum indicators
        short_ema = self.ema(self.short_period)
        long_ema = self.ema(self.long_period)
        
        # Order book imbalance (simplified simulation for Jesse)
        order_imbalance = self.simulate_order_imbalance()
        
        # Calculate signal quality score (0-100) combining multiple factors
        signal_score = self.calculate_long_signal_quality(short_ema, long_ema, order_imbalance)
        
        # Only take high quality signals
        if signal_score >= self.signal_quality_threshold:
            self.last_trade_candle = self.index
            return True
            
        return False

    def should_short(self) -> bool:
        """
        Determine if we should enter a short position
        """
        # Basic checks
        if self.index < 50:  # Ensure enough data for calculations
            return False
            
        if self.index - self.last_trade_candle < self.min_candles_between_trades:
            return False
            
        # Don't enter short positions in bullish regimes
        if self.current_regime == 1:
            return False
            
        # Core momentum indicators
        short_ema = self.ema(self.short_period)
        long_ema = self.ema(self.long_period)
        
        # Order book imbalance (simplified simulation for Jesse)
        order_imbalance = self.simulate_order_imbalance()
        
        # Calculate signal quality score (0-100) combining multiple factors
        signal_score = self.calculate_short_signal_quality(short_ema, long_ema, order_imbalance)
        
        # Only take high quality signals
        if signal_score >= self.signal_quality_threshold:
            self.last_trade_candle = self.index
            return True
            
        return False

    def calculate_long_signal_quality(self, short_ema, long_ema, order_imbalance) -> float:
        """
        Calculate the quality of a long signal (0-100)
        Higher scores indicate stronger signals
        """
        score = 0
        
        # Trend component (0-40 points)
        ema_diff_percent = ((short_ema - long_ema) / long_ema) * 100
        if ema_diff_percent > 0:
            # Positive and increasing trend strength
            trend_score = min(40, ema_diff_percent * 8)
            score += trend_score
        
        # Momentum component (0-30 points)
        rsi = self.rsi(14)
        if 30 <= rsi <= 70:
            # RSI in healthy range, not overbought
            # More points for RSI showing upward momentum but not extreme
            momentum_score = 30 - abs(rsi - 55)
            score += momentum_score
        
        # Order book component (0-30 points)
        if order_imbalance > 0:
            # Positive imbalance (more buying pressure)
            ob_score = min(30, order_imbalance * 30)
            score += ob_score
            
        # Regime bonus for alignment with current regime
        if self.current_regime == 1:  # Bullish regime
            score += 15  # Bonus points for regime alignment
        
        return score

    def calculate_short_signal_quality(self, short_ema, long_ema, order_imbalance) -> float:
        """
        Calculate the quality of a short signal (0-100)
        Higher scores indicate stronger signals
        """
        score = 0
        
        # Trend component (0-40 points)
        ema_diff_percent = ((long_ema - short_ema) / long_ema) * 100
        if ema_diff_percent > 0:
            # Negative and decreasing trend strength
            trend_score = min(40, ema_diff_percent * 8)
            score += trend_score
        
        # Momentum component (0-30 points)
        rsi = self.rsi(14)
        if 30 <= rsi <= 70:
            # RSI in healthy range, not oversold
            # More points for RSI showing downward momentum but not extreme
            momentum_score = 30 - abs(rsi - 45)
            score += momentum_score
        
        # Order book component (0-30 points)
        if order_imbalance < 0:
            # Negative imbalance (more selling pressure)
            ob_score = min(30, abs(order_imbalance) * 30)
            score += ob_score
            
        # Regime bonus for alignment with current regime
        if self.current_regime == -1:  # Bearish regime
            score += 15  # Bonus points for regime alignment
        
        return score

    def simulate_order_imbalance(self) -> float:
        """
        Simulate order book imbalance using volume and price action
        Returns a value between -1 and 1 (negative: selling pressure, positive: buying pressure)
        """
        # Use volume and price action to approximate order book pressure
        close = self.candles[:, 2][-1]
        open_price = self.candles[:, 1][-1]
        high = self.candles[:, 3][-1]
        low = self.candles[:, 4][-1]
        volume = self.candles[:, 5][-1]
        
        # Calculate price position within the candle range
        if high != low:
            price_position = (close - low) / (high - low)  # 0 to 1 (close to low or high)
        else:
            price_position = 0.5
            
        # Calculate volume strength compared to recent average
        avg_volume = np.mean(self.candles[:, 5][-20:])
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1
        
        # Determine candle direction
        price_change = close - open_price
        
        # Combine factors to simulate order book imbalance
        if price_change > 0:
            # Bullish candle - positive imbalance
            imbalance = price_position * volume_ratio * 0.8
        else:
            # Bearish candle - negative imbalance
            imbalance = (price_position - 1) * volume_ratio * 0.8
            
        # Normalize between -1 and 1
        return max(min(imbalance, 1), -1)

    def go_long(self):
        """
        Execute long position with dynamic sizing
        """
        # Calculate dynamic stop loss and take profit based on ATR
        atr = self.atr(self.atr_period)
        sl_distance = atr * self.atr_multiplier_sl
        tp_distance = atr * self.atr_multiplier_tp
        
        # Entry price
        entry = self.price
        
        # Store stop loss and take profit for later use in on_open_position
        self.temp_stop_loss = entry - sl_distance
        self.temp_take_profit = entry + tp_distance
        
        # Position sizing with Kelly-inspired adjustment based on signal quality
        signal_quality = self.calculate_long_signal_quality(
            self.ema(self.short_period), 
            self.ema(self.long_period),
            self.simulate_order_imbalance()
        )
        
        # Adjust risk based on signal quality
        adjusted_risk = self.risk_per_trade * (signal_quality / 100)
        
        # Calculate position size - use self.balance instead of self.capital
        risk_amount = self.balance * (adjusted_risk / 100)
        position_size = risk_amount / sl_distance
        
        # Safety cap
        max_allowed_position = self.available_margin / entry * 0.95
        position_size = min(position_size, max_allowed_position)
        
        # Place the order
        self.buy = position_size, entry
        # Stop loss and take profit are set in on_open_position

    def go_short(self):
        """
        Execute short position with dynamic sizing
        """
        # Calculate dynamic stop loss and take profit based on ATR
        atr = self.atr(self.atr_period)
        sl_distance = atr * self.atr_multiplier_sl
        tp_distance = atr * self.atr_multiplier_tp
        
        # Entry price
        entry = self.price
        
        # Store stop loss and take profit for later use in on_open_position
        self.temp_stop_loss = entry + sl_distance
        self.temp_take_profit = entry - tp_distance
        
        # Position sizing with Kelly-inspired adjustment based on signal quality
        signal_quality = self.calculate_short_signal_quality(
            self.ema(self.short_period), 
            self.ema(self.long_period),
            self.simulate_order_imbalance()
        )
        
        # Adjust risk based on signal quality
        adjusted_risk = self.risk_per_trade * (signal_quality / 100)
        
        # Calculate position size - use self.balance instead of self.capital
        risk_amount = self.balance * (adjusted_risk / 100)
        position_size = risk_amount / sl_distance
        
        # Safety cap
        max_allowed_position = self.available_margin / entry * 0.95
        position_size = min(position_size, max_allowed_position)
        
        # Place the order
        self.sell = position_size, entry
        # Stop loss and take profit are set in on_open_position

    def on_open_position(self, order):
        """
        Called right after a position is opened
        Used to set take profit and stop loss levels for spot trading
        """
        # Set the take profit and stop loss levels that were calculated in go_long/go_short
        if self.temp_take_profit is not None:
            self.take_profit = self.position.qty, self.temp_take_profit
            self.temp_take_profit = None
            
        if self.temp_stop_loss is not None:
            self.stop_loss = self.position.qty, self.temp_stop_loss
            self.temp_stop_loss = None

    def update_position(self):
        """
        Dynamic position management with trailing stops
        """
        atr = self.atr(self.atr_period)
        
        # For long positions
        if self.is_long:
            # Calculate current profit
            profit_pct = (self.price - self.average_entry_price) / self.average_entry_price * 100
            
            # Implement tiered trailing stop based on profit level
            if profit_pct > 4:
                # Significant profit - tighten stop loss (1.5 ATR)
                new_stop_price = self.price - (atr * 1.5)
                # Make sure we're using the tuple format (qty, price)
                if new_stop_price > self.stop_loss[1]:
                    self.stop_loss = self.position.qty, new_stop_price
            elif profit_pct > 2:
                # Moderate profit - slightly tighter stop (2 ATR)
                new_stop_price = self.price - (atr * 2)
                # Make sure we're using the tuple format (qty, price)
                if new_stop_price > self.stop_loss[1]:
                    self.stop_loss = self.position.qty, new_stop_price
        
        # For short positions
        elif self.is_short:
            # Calculate current profit
            profit_pct = (self.average_entry_price - self.price) / self.average_entry_price * 100
            
            # Implement tiered trailing stop based on profit level
            if profit_pct > 4:
                # Significant profit - tighten stop loss (1.5 ATR)
                new_stop_price = self.price + (atr * 1.5)
                # Make sure we're using the tuple format (qty, price)
                if new_stop_price < self.stop_loss[1]:
                    self.stop_loss = self.position.qty, new_stop_price
            elif profit_pct > 2:
                # Moderate profit - slightly tighter stop (2 ATR)
                new_stop_price = self.price + (atr * 2)
                # Make sure we're using the tuple format (qty, price)
                if new_stop_price < self.stop_loss[1]:
                    self.stop_loss = self.position.qty, new_stop_price

    def should_cancel_entry(self) -> bool:
        """
        Cancel entry orders if market conditions change rapidly
        """
        # Calculate quick momentum change
        price_change = (self.close - self.open) / self.open * 100
        
        # Cancel if there's a sudden large price movement against our position
        if self.is_long and price_change < -1.5:
            return True
        if self.is_short and price_change > 1.5:
            return True
            
        return False

    def should_exit_long(self) -> bool:
        """
        Additional logic for exiting long positions beyond SL/TP
        """
        # Early exit on momentum reversal
        rsi = self.rsi(14)
        
        # Exit if RSI becomes overbought and starts declining
        if rsi > 75 and self.cross_below(self.rsi(14, sequential=True), 75):
            return True
        
        # Exit if trend has clearly reversed
        if self.cross_below(self.ema(self.short_period, sequential=True), self.ema(self.long_period, sequential=True)):
            return True
            
        return False
        
    def should_exit_short(self) -> bool:
        """
        Additional logic for exiting short positions beyond SL/TP
        """
        # Early exit on momentum reversal
        rsi = self.rsi(14)
        
        # Exit if RSI becomes oversold and starts rising
        if rsi < 25 and self.cross_above(self.rsi(14, sequential=True), 25):
            return True
        
        # Exit if trend has clearly reversed
        if self.cross_above(self.ema(self.short_period, sequential=True), self.ema(self.long_period, sequential=True)):
            return True
            
        return False

    def ema(self, period: int, source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
        """Calculate EMA using Jesse's indicator"""
        from jesse.indicators import ema
        return ema(self.candles, period, source_type, sequential)
    
    def rsi(self, period: int, source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
        """Calculate RSI using Jesse's indicator"""
        from jesse.indicators import rsi
        return rsi(self.candles, period, source_type, sequential)
    
    def atr(self, period: int, sequential: bool = False) -> Union[float, np.ndarray]:
        """Calculate ATR using Jesse's indicator"""
        from jesse.indicators import atr
        return atr(self.candles, period, sequential)
        
    def bollinger_bands(self, period: int = 20, devs: float = 2.0, source_type: str = "close", sequential: bool = True) -> object:
        """Calculate Bollinger Bands using Jesse's indicator"""
        from jesse.indicators import bollinger_bands
        return bollinger_bands(self.candles, period, devs, source_type, sequential)
    
    def macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9, 
            source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
        """Calculate MACD using Jesse's indicator"""
        from jesse.indicators import macd
        macd_result = macd(self.candles, fast_period, slow_period, signal_period, source_type, sequential)
        return macd_result
    
    def cross_above(self, series1: Union[float, np.ndarray], series2: Union[float, np.ndarray]) -> bool:
        """Check if series1 crosses above series2"""
        if isinstance(series1, np.ndarray) and isinstance(series2, np.ndarray):
            return series1[-2] <= series2[-2] and series1[-1] > series2[-1]
        elif isinstance(series1, np.ndarray):
            return series1[-2] <= series2 and series1[-1] > series2
        elif isinstance(series2, np.ndarray):
            return series1 <= series2[-2] and series1 > series2[-1]
        else:
            return False  # Cannot determine cross with single values
    
    def cross_below(self, series1: Union[float, np.ndarray], series2: Union[float, np.ndarray]) -> bool:
        """Check if series1 crosses below series2"""
        if isinstance(series1, np.ndarray) and isinstance(series2, np.ndarray):
            return series1[-2] >= series2[-2] and series1[-1] < series2[-1]
        elif isinstance(series1, np.ndarray):
            return series1[-2] >= series2 and series1[-1] < series2
        elif isinstance(series2, np.ndarray):
            return series1 >= series2[-2] and series1 < series2[-1]
        else:
            return False  # Cannot determine cross with single values
    
    def fractal_efficiency(self, prices: np.ndarray, period: int = 10) -> float:
        """
        Calculate Fractal Efficiency Ratio - measures the efficiency of price movement
        Returns a value between -1 and 1
        Positive values indicate uptrend efficiency, negative values indicate downtrend efficiency
        """
        if len(prices) < period + 1:
            return 0
            
        # Get relevant price segment
        price_segment = prices[-period-1:]
        
        # Net price movement
        net_movement = abs(price_segment[-1] - price_segment[0])
        
        # Sum of all price movements
        total_movement = sum(abs(price_segment[i] - price_segment[i-1]) for i in range(1, len(price_segment)))
        
        # Calculate efficiency
        if total_movement > 0:
            efficiency = net_movement / total_movement
        else:
            efficiency = 0
            
        # Add direction
        if price_segment[-1] > price_segment[0]:
            return efficiency  # Positive for uptrend
        else:
            return -efficiency  # Negative for downtrend

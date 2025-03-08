import numpy as np
from jesse.strategies import Strategy
from jesse import utils
from typing import Union


class AdaptiveMomentumNexus(Strategy):
    """
    An optimized momentum strategy for Jesse trading framework combining:
    - Adaptive EMA crossovers
    - RSI confirmation
    - MACD for trend detection
    - Volatility-based position sizing
    - ATR for dynamic stop-loss and take-profit
    """

    def __init__(self):
        super().__init__()
        
        # Strategy parameters - these could be optimized with Jesse's optimization feature
        self.fast_ema_period = 8
        self.slow_ema_period = 21
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        self.atr_period = 14
        self.atr_multiplier_sl = 3.0
        self.atr_multiplier_tp = 5.0
        self.volume_ma_period = 20
        self.risk_per_trade = 2.0  # percentage of capital to risk per trade
        
        # Dynamic variables
        self.entry_signal = False
        self.exit_signal = False
        self.trend_direction = 0  # -1 for downtrend, 0 for neutral, 1 for uptrend
        self.last_signal_candle = 0

    def should_long(self) -> bool:
        """
        Determine if we should enter a long position
        """
        # Avoid trading in the first hour after strategy starts (warm-up)
        if self.index < 60:
            return False
        
        # Avoid multiple entries within a short period
        if self.index - self.last_signal_candle < 10:
            return False
            
        # Calculate indicators
        fast_ema = self.ema(self.fast_ema_period)
        slow_ema = self.ema(self.slow_ema_period)
        rsi = self.rsi(self.rsi_period)
        macd, macd_signal, macd_hist = self.macd()
        
        # Volume filter - only trade if volume is above average
        current_volume = self.candles[:, 5][-1]  # Current volume from candles
        volume_ma = self.sma(self.volume_ma_period, 'volume')
        volume_filter = current_volume > volume_ma * 1.2
        
        # EMA crossover (fast crosses above slow)
        ema_crossover = fast_ema > slow_ema and self.cross_above(fast_ema, slow_ema)
        
        # RSI is coming up from oversold
        rsi_filter = rsi < 50 and rsi > self.rsi_oversold and self.cross_above(rsi, self.rsi_oversold)
        
        # MACD histogram is positive or crossing above zero
        macd_filter = macd_hist > 0 or self.cross_above(macd_hist, 0)
        
        # Check for overall bullish market sentiment (higher highs and higher lows)
        bullish_price_action = self.price > self.price_at(10) > self.price_at(20)
        
        # Combine signals for entry
        self.entry_signal = ema_crossover and rsi_filter and macd_filter and volume_filter and bullish_price_action
        
        if self.entry_signal:
            self.trend_direction = 1
            self.last_signal_candle = self.index
            
        return self.entry_signal

    def should_short(self) -> bool:
        """
        Determine if we should enter a short position
        """
        # Avoid trading in the first hour after strategy starts (warm-up)
        if self.index < 60:
            return False
        
        # Avoid multiple entries within a short period
        if self.index - self.last_signal_candle < 10:
            return False
            
        # Calculate indicators
        fast_ema = self.ema(self.fast_ema_period)
        slow_ema = self.ema(self.slow_ema_period)
        rsi = self.rsi(self.rsi_period)
        macd, macd_signal, macd_hist = self.macd()
        
        # Volume filter - only trade if volume is above average
        current_volume = self.candles[:, 5][-1]  # Current volume from candles
        volume_ma = self.sma(self.volume_ma_period, 'volume')
        volume_filter = current_volume > volume_ma * 1.2
        
        # EMA crossover (fast crosses below slow)
        ema_crossover = fast_ema < slow_ema and self.cross_below(fast_ema, slow_ema)
        
        # RSI is coming down from overbought
        rsi_filter = rsi > 50 and rsi < self.rsi_overbought and self.cross_below(rsi, self.rsi_overbought)
        
        # MACD histogram is negative or crossing below zero
        macd_filter = macd_hist < 0 or self.cross_below(macd_hist, 0)
        
        # Check for overall bearish market sentiment (lower highs and lower lows)
        bearish_price_action = self.price < self.price_at(10) < self.price_at(20)
        
        # Combine signals for entry
        self.entry_signal = ema_crossover and rsi_filter and macd_filter and volume_filter and bearish_price_action
        
        if self.entry_signal:
            self.trend_direction = -1
            self.last_signal_candle = self.index
            
        return self.entry_signal

    def should_cancel_entry(self) -> bool:
        """
        Cancel entry if signal becomes invalid
        """
        if self.trend_direction == 1:  # For longs
            # Cancel if RSI moves back to bearish territory
            rsi = self.rsi(self.rsi_period)
            return rsi < self.rsi_oversold - 5
            
        elif self.trend_direction == -1:  # For shorts
            # Cancel if RSI moves back to bullish territory
            rsi = self.rsi(self.rsi_period)
            return rsi > self.rsi_overbought + 5
            
        return False

    def go_long(self):
        """
        Prepare the long order
        """
        # Calculate dynamic stop loss and take profit based on ATR
        atr = self.atr(self.atr_period)
        sl_distance = atr * self.atr_multiplier_sl
        tp_distance = atr * self.atr_multiplier_tp
        
        # Calculate position size based on risk percentage
        entry = self.price
        stop_loss = entry - sl_distance
        
        # Risk management: calculate position size based on risk percentage
        capital = self.capital
        risk_amount = capital * (self.risk_per_trade / 100)
        position_size = risk_amount / sl_distance
        
        # Ensure position size is reasonable
        max_allowed_position = self.available_margin / entry * 0.95  # 95% of available margin
        position_size = min(position_size, max_allowed_position)
        
        # Enter the position
        self.buy = position_size, entry
        
        # Set stop loss and take profit
        self.stop_loss = stop_loss
        self.take_profit = entry + tp_distance

    def go_short(self):
        """
        Prepare the short order
        """
        # Calculate dynamic stop loss and take profit based on ATR
        atr = self.atr(self.atr_period)
        sl_distance = atr * self.atr_multiplier_sl
        tp_distance = atr * self.atr_multiplier_tp
        
        # Calculate position size based on risk percentage
        entry = self.price
        stop_loss = entry + sl_distance
        
        # Risk management: calculate position size based on risk percentage
        capital = self.capital
        risk_amount = capital * (self.risk_per_trade / 100)
        position_size = risk_amount / sl_distance
        
        # Ensure position size is reasonable
        max_allowed_position = self.available_margin / entry * 0.95  # 95% of available margin
        position_size = min(position_size, max_allowed_position)
        
        # Enter the position
        self.sell = position_size, entry
        
        # Set stop loss and take profit
        self.stop_loss = stop_loss
        self.take_profit = entry - tp_distance

    def update_position(self):
        """
        Update position parameters if needed
        """
        # For long positions
        if self.is_long:
            # Calculate trailing stop if in profit
            current_profit_pct = (self.price - self.average_entry_price) / self.average_entry_price * 100
            
            # Trail stop loss if we're in significant profit
            if current_profit_pct > 5:
                atr = self.atr(self.atr_period)
                new_stop = self.price - (atr * 2.5)  # Tighter stop when in profit
                
                # Only move stop loss up, never down
                if new_stop > self.stop_loss:
                    self.stop_loss = new_stop
        
        # For short positions
        elif self.is_short:
            # Calculate trailing stop if in profit
            current_profit_pct = (self.average_entry_price - self.price) / self.average_entry_price * 100
            
            # Trail stop loss if we're in significant profit
            if current_profit_pct > 5:
                atr = self.atr(self.atr_period)
                new_stop = self.price + (atr * 2.5)  # Tighter stop when in profit
                
                # Only move stop loss down, never up
                if new_stop < self.stop_loss:
                    self.stop_loss = new_stop

    def should_exit_long(self) -> bool:
        """
        Additional long exit logic beyond SL/TP
        """
        # Check if momentum is weakening or reversing
        rsi = self.rsi(self.rsi_period)
        macd, macd_signal, macd_hist = self.macd()
        
        # Exit if RSI becomes overbought and MACD histogram starts declining
        macd_hist_prev1 = self.macd_hist_previous(1)
        macd_hist_prev2 = self.macd_hist_previous(2)
        if rsi > self.rsi_overbought and macd_hist < macd_hist_prev1 < macd_hist_prev2:
            return True
            
        # Also exit if EMA crosses bearishly
        fast_ema = self.ema(self.fast_ema_period)
        slow_ema = self.ema(self.slow_ema_period)
        
        if self.cross_below(fast_ema, slow_ema):
            return True
            
        return False

    def should_exit_short(self) -> bool:
        """
        Additional short exit logic beyond SL/TP
        """
        # Check if momentum is weakening or reversing
        rsi = self.rsi(self.rsi_period)
        macd, macd_signal, macd_hist = self.macd()
        
        # Exit if RSI becomes oversold and MACD histogram starts rising
        macd_hist_prev1 = self.macd_hist_previous(1)
        macd_hist_prev2 = self.macd_hist_previous(2)
        if rsi < self.rsi_oversold and macd_hist > macd_hist_prev1 > macd_hist_prev2:
            return True
            
        # Also exit if EMA crosses bullishly
        fast_ema = self.ema(self.fast_ema_period)
        slow_ema = self.ema(self.slow_ema_period)
        
        if self.cross_above(fast_ema, slow_ema):
            return True
            
        return False
    
    def ema(self, period: int, source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
        """Calculate EMA for the specified price"""
        from jesse.indicators import ema
        return ema(self.candles, period, source_type, sequential)
    
    def sma(self, period: int, source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
        """Calculate SMA for the specified price"""
        from jesse.indicators import sma
        return sma(self.candles, period, source_type, sequential)
    
    def rsi(self, period: int, source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
        """Calculate RSI for the specified price"""
        from jesse.indicators import rsi
        return rsi(self.candles, period, source_type, sequential)
    
    def atr(self, period: int, sequential: bool = False) -> Union[float, np.ndarray]:
        """Calculate ATR"""
        from jesse.indicators import atr
        return atr(self.candles, period, sequential)
    
    def macd(self, fast_period: int = None, slow_period: int = None, signal_period: int = None, 
              source_type: str = "close", sequential: bool = False) -> tuple:
        """Calculate MACD"""
        from jesse.indicators import macd
        
        # Use instance variables if parameters are not provided
        if fast_period is None:
            fast_period = self.macd_fast
        if slow_period is None:
            slow_period = self.macd_slow
        if signal_period is None:
            signal_period = self.macd_signal
            
        macd_val = macd(self.candles, fast_period, slow_period, signal_period, source_type, sequential)
        
        if sequential:
            return macd_val.macd, macd_val.signal, macd_val.hist
        else:
            return macd_val.macd, macd_val.signal, macd_val.hist
    
    def macd_hist_previous(self, periods_ago: int) -> float:
        """Get MACD histogram value from previous candles"""
        from jesse.indicators import macd
        
        # When asking for previous values, we need sequential data
        macd_values = macd(
            self.candles, 
            self.macd_fast, 
            self.macd_slow, 
            self.macd_signal, 
            "close", 
            True
        )
        
        # Return the histogram value from n periods ago
        if len(macd_values.hist) > periods_ago:
            return macd_values.hist[-1 - periods_ago]
        return 0
    
    def cross_above(self, a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> bool:
        """Check if a crosses above b"""
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return a[-2] <= b[-2] and a[-1] > b[-1]
        elif isinstance(a, np.ndarray):
            return a[-2] <= b and a[-1] > b
        elif isinstance(b, np.ndarray):
            return a <= b[-2] and a > b[-1]
        else:
            return False  # Cannot determine cross with single values
    
    def cross_below(self, a: Union[float, np.ndarray], b: Union[float, np.ndarray]) -> bool:
        """Check if a crosses below b"""
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            return a[-2] >= b[-2] and a[-1] < b[-1]
        elif isinstance(a, np.ndarray):
            return a[-2] >= b and a[-1] < b
        elif isinstance(b, np.ndarray):
            return a >= b[-2] and a < b[-1]
        else:
            return False  # Cannot determine cross with single values
    
    def price_at(self, lookback: int) -> float:
        """Get the close price at a certain lookback period"""
        if self.index - lookback >= 0:
            return self.candles[self.index - lookback, 2]
        return self.close

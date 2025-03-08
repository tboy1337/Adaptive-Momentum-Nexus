import numpy as np
from jesse.strategies import Strategy
from jesse import utils
from typing import Union


class AdaptiveMomentumNexus(Strategy):
    """
    A simplified momentum strategy for Jesse trading framework combining:
    - EMA crossovers for trend detection
    - RSI for confirming oversold/overbought conditions
    - ATR for dynamic stop-loss and take-profit
    """

    def __init__(self):
        super().__init__()
        
        # Core strategy parameters - simplified
        self.fast_ema_period = 8
        self.slow_ema_period = 21
        self.rsi_period = 14
        self.rsi_overbought = 75  # Increased to avoid too many false signals
        self.rsi_oversold = 25    # Decreased to avoid too many false signals
        self.atr_period = 14
        self.atr_multiplier_sl = 2.5  # Reduced from 3.0
        self.atr_multiplier_tp = 4.0  # Reduced from 5.0
        self.risk_per_trade = 1.5  # Slightly reduced risk per trade
        
        # Tracking variables
        self.last_signal_candle = 0

    def should_long(self) -> bool:
        """
        Determine if we should enter a long position
        """
        # Avoid trading in the first period to ensure indicators are properly calculated
        if self.index < 30:
            return False
        
        # Prevent rapid re-entries
        if self.index - self.last_signal_candle < 5:
            return False
            
        # Calculate indicators
        fast_ema = self.ema(self.fast_ema_period)
        slow_ema = self.ema(self.slow_ema_period)
        rsi = self.rsi(self.rsi_period)
        
        # Primary signal: EMA crossover (fast crosses above slow)
        ema_crossover = self.cross_above(fast_ema, slow_ema)
        
        # Secondary confirmation: RSI is not overbought and preferably coming up from lower values
        rsi_filter = rsi < 60  # Not in overbought territory
        
        # Simplified entry condition
        entry_signal = ema_crossover and rsi_filter
        
        if entry_signal:
            self.last_signal_candle = self.index
            
        return entry_signal

    def should_short(self) -> bool:
        """
        Determine if we should enter a short position
        """
        # Avoid trading in the first period to ensure indicators are properly calculated
        if self.index < 30:
            return False
        
        # Prevent rapid re-entries
        if self.index - self.last_signal_candle < 5:
            return False
            
        # Calculate indicators
        fast_ema = self.ema(self.fast_ema_period)
        slow_ema = self.ema(self.slow_ema_period)
        rsi = self.rsi(self.rsi_period)
        
        # Primary signal: EMA crossover (fast crosses below slow)
        ema_crossover = self.cross_below(fast_ema, slow_ema)
        
        # Secondary confirmation: RSI is not oversold and preferably coming down from higher values
        rsi_filter = rsi > 40  # Not in oversold territory
        
        # Simplified entry condition
        entry_signal = ema_crossover and rsi_filter
        
        if entry_signal:
            self.last_signal_candle = self.index
            
        return entry_signal

    def should_cancel_entry(self) -> bool:
        """
        Cancel entry if signal becomes invalid
        """
        # Simplified logic - just keep the entry order active
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
        
        # Ensure position size is reasonable (max 20% of available margin)
        max_allowed_position = self.available_margin / entry * 0.2
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
        
        # Ensure position size is reasonable (max 20% of available margin)
        max_allowed_position = self.available_margin / entry * 0.2
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
            if current_profit_pct > 4:  # Reduced from 5%
                atr = self.atr(self.atr_period)
                new_stop = self.price - (atr * 2)  # Tighter stop when in profit
                
                # Only move stop loss up, never down
                if new_stop > self.stop_loss:
                    self.stop_loss = new_stop
        
        # For short positions
        elif self.is_short:
            # Calculate trailing stop if in profit
            current_profit_pct = (self.average_entry_price - self.price) / self.average_entry_price * 100
            
            # Trail stop loss if we're in significant profit
            if current_profit_pct > 4:  # Reduced from 5%
                atr = self.atr(self.atr_period)
                new_stop = self.price + (atr * 2)  # Tighter stop when in profit
                
                # Only move stop loss down, never up
                if new_stop < self.stop_loss:
                    self.stop_loss = new_stop

    def should_exit_long(self) -> bool:
        """
        Additional long exit logic beyond SL/TP
        """
        # Simplified exit logic - just rely on EMA crossovers
        fast_ema = self.ema(self.fast_ema_period)
        slow_ema = self.ema(self.slow_ema_period)
        
        # Exit if EMAs cross bearishly
        if self.cross_below(fast_ema, slow_ema):
            return True
            
        return False

    def should_exit_short(self) -> bool:
        """
        Additional short exit logic beyond SL/TP
        """
        # Simplified exit logic - just rely on EMA crossovers
        fast_ema = self.ema(self.fast_ema_period)
        slow_ema = self.ema(self.slow_ema_period)
        
        # Exit if EMAs cross bullishly
        if self.cross_above(fast_ema, slow_ema):
            return True
            
        return False
    
    def ema(self, period: int, source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
        """Calculate EMA for the specified price"""
        from jesse.indicators import ema
        return ema(self.candles, period, source_type, sequential)
    
    def rsi(self, period: int, source_type: str = "close", sequential: bool = False) -> Union[float, np.ndarray]:
        """Calculate RSI for the specified price"""
        from jesse.indicators import rsi
        return rsi(self.candles, period, source_type, sequential)
    
    def atr(self, period: int, sequential: bool = False) -> Union[float, np.ndarray]:
        """Calculate ATR"""
        from jesse.indicators import atr
        return atr(self.candles, period, sequential)
    
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

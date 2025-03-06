# Adaptive Momentum Nexus (AMN)

A sophisticated crypto trading strategy for Jesse framework combining technical analysis, adaptive volatility measures, and smart money management.

## Overview

Adaptive Momentum Nexus (AMN) is a professional-grade trading strategy built for the Jesse trading framework. It combines multiple technical indicators with dynamic risk management to identify high-probability trading opportunities while maintaining strict risk controls.

The strategy is designed for cryptocurrency markets but can be adapted to other markets supported by Jesse.

## Features

- **Multi-indicator confirmation system** for higher-quality signals
- **Volatility-based position sizing** through ATR measurements
- **Dynamic stop-loss and take-profit** levels that adapt to market conditions
- **Trailing stop management** to lock in profits
- **Volume confirmation** to filter out low-quality signals
- **Advanced risk management** with fixed percentage risk per trade

## Installation

1. Make sure you have [Jesse](https://jesse.trade/) installed
2. Create a new Jesse project or use an existing one
3. Save the `adaptive_momentum_nexus.py` file to the `strategies` folder of your Jesse project
4. Configure the strategy in your routes

## Configuration

Add the strategy to your `routes.py` file:

```python
routes = [
    ('Binance', 'BTC-USDT', '4h', 'AdaptiveMomentumNexus'),
]
```

## Optimization

The strategy includes several parameters that can be optimized:

```python
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
self.risk_per_trade = 2.0
```

You can optimize these parameters using Jesse's built-in optimization framework.

## Strategy Logic

### Entry Conditions

#### Long Entries
- Fast EMA crosses above Slow EMA
- RSI is below 50 but rising from oversold (>30)
- MACD histogram is positive or crossing above zero
- Current volume exceeds 20-period volume MA by at least 20%
- Price shows higher highs and higher lows pattern

#### Short Entries
- Fast EMA crosses below Slow EMA
- RSI is above 50 but falling from overbought (<70)
- MACD histogram is negative or crossing below zero
- Current volume exceeds 20-period volume MA by at least 20%
- Price shows lower highs and lower lows pattern

### Exit Conditions

- Stop-loss and take-profit targets (based on ATR)
- Trailing stop adjustments when in profit
- Technical reversal signals:
  - For longs: RSI becomes overbought while MACD momentum declines
  - For shorts: RSI becomes oversold while MACD momentum rises
- EMA crossover reversals

## Risk Management

- Position sizing based on account risk percentage (default: 2%)
- Maximum position size capped at 95% of available margin
- ATR-based stop loss placement
- Trading timeout periods between signals
- Warm-up period to ensure indicator stability

## Backtesting

To backtest the strategy:

```bash
jesse backtest 2021-01-01 2021-02-01
```

## Live Trading

For live trading, make sure to:

1. Thoroughly backtest the strategy
2. Optimize parameters for your specific market and timeframe
3. Start with small position sizes to validate performance

## License

This strategy is provided under the MIT License.

## Disclaimer

Trading cryptocurrencies involves significant risk and can result in the loss of your invested capital. This strategy is provided for educational purposes only and should not be considered financial advice.

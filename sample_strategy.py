# sample_strategy.py
"""
Sample Trading Strategy Implementation
This demonstrates how to create custom strategies using the trading bot framework.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
from trading_bot import (
    TradingBot, TradingSignal, TechnicalIndicators,
    FeatureEngineering, ConfigManager
)

logger = logging.getLogger(__name__)


class MomentumMeanReversionStrategy:
    """
    Sample strategy combining momentum and mean reversion signals

    Strategy Logic:
    1. Use RSI for mean reversion signals (oversold/overbought)
    2. Use MACD for momentum confirmation
    3. Use Bollinger Bands for volatility-based entries
    4. Apply volume confirmation
    """

    def __init__(self, config: Dict):
        self.config = config
        self.name = "Momentum Mean Reversion"

        # Strategy parameters
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.bb_std_multiplier = 2.0
        self.volume_threshold = 1.5  # 1.5x average volume
        self.min_confidence = 0.6

    def analyze_symbol(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Analyze a single symbol and generate trading signal
        """
        if len(data) < 50:  # Need minimum data
            logger.warning(f"Insufficient data for {symbol}")
            return None

        try:
            # Calculate technical indicators
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']

            # RSI for mean reversion
            rsi = TechnicalIndicators.rsi(close, 14)
            current_rsi = rsi.iloc[-1]

            # MACD for momentum
            macd_line, signal_line, histogram = TechnicalIndicators.macd(close)
            current_macd = macd_line.iloc[-1]
            current_signal = signal_line.iloc[-1]
            macd_bullish = current_macd > current_signal

            # Bollinger Bands for volatility
            bb_upper, bb_middle, bb_lower = TechnicalIndicators.bollinger_bands(
                close, window=20, num_std=self.bb_std_multiplier
            )
            current_price = close.iloc[-1]
            bb_position = (
                current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])

            # Volume analysis
            volume_sma = volume.rolling(20).mean()
            volume_ratio = volume.iloc[-1] / volume_sma.iloc[-1]
            volume_confirmation = volume_ratio > self.volume_threshold

            # Price momentum
            price_change_5d = (
                current_price / close.iloc[-6]) - 1 if len(close) > 5 else 0

            # Generate signals
            signal_type = "HOLD"
            confidence = 0.0
            reasoning_parts = []

            # BUY conditions
            buy_conditions = []
            buy_conditions.append(
                ("RSI Oversold", current_rsi < self.rsi_oversold, 0.3))
            buy_conditions.append(("MACD Bullish", macd_bullish, 0.2))
            buy_conditions.append(("Below BB Lower", bb_position < 0.2, 0.2))
            buy_conditions.append(
                ("Volume Confirmation", volume_confirmation, 0.15))
            buy_conditions.append(
                ("Positive 5d Momentum", price_change_5d > 0.02, 0.15))

            # SELL conditions
            sell_conditions = []
            sell_conditions.append(
                ("RSI Overbought", current_rsi > self.rsi_overbought, 0.3))
            sell_conditions.append(("MACD Bearish", not macd_bullish, 0.2))
            sell_conditions.append(("Above BB Upper", bb_position > 0.8, 0.2))
            sell_conditions.append(
                ("Volume Confirmation", volume_confirmation, 0.15))
            sell_conditions.append(
                ("Negative 5d Momentum", price_change_5d < -0.02, 0.15))

            # Calculate buy confidence
            buy_score = sum(weight for condition, met,
                            weight in buy_conditions if met)
            buy_met = [condition for condition, met,
                       weight in buy_conditions if met]

            # Calculate sell confidence
            sell_score = sum(weight for condition, met,
                             weight in sell_conditions if met)
            sell_met = [condition for condition, met,
                        weight in sell_conditions if met]

            # Determine signal
            if buy_score > sell_score and buy_score >= self.min_confidence:
                signal_type = "BUY"
                confidence = min(buy_score, 1.0)
                reasoning_parts = buy_met
            elif sell_score > buy_score and sell_score >= self.min_confidence:
                signal_type = "SELL"
                confidence = min(sell_score, 1.0)
                reasoning_parts = sell_met

            # Calculate targets and stops
            atr = TechnicalIndicators.atr(high, low, close, 14).iloc[-1]
            target_price = None
            stop_loss = None

            if signal_type == "BUY":
                target_price = current_price + (2 * atr)  # 2 ATR target
                stop_loss = current_price - (1 * atr)     # 1 ATR stop
            elif signal_type == "SELL":
                target_price = current_price - (2 * atr)  # 2 ATR target
                stop_loss = current_price + (1 * atr)     # 1 ATR stop

            # Create signal
            if signal_type != "HOLD":
                reasoning = f"Strategy: {self.name} | Conditions: {', '.join(reasoning_parts)} | RSI: {current_rsi:.1f} | BB Pos: {bb_position:.2f}"

                return TradingSignal(
                    symbol=symbol,
                    signal=signal_type,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    current_price=current_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    reasoning=reasoning
                )

            return None

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
            return None


class BreakoutStrategy:
    """
    Breakout strategy based on price and volume

    Strategy Logic:
    1. Identify consolidation periods (low volatility)
    2. Wait for breakout with high volume
    3. Enter in direction of breakout
    4. Use ATR-based stops and targets
    """

    def __init__(self, config: Dict):
        self.config = config
        self.name = "Breakout Strategy"

        # Strategy parameters
        self.consolidation_days = 10
        self.breakout_threshold = 0.02  # 2% move
        self.volume_multiplier = 2.0    # 2x average volume
        self.volatility_threshold = 0.01  # 1% daily volatility threshold

    def analyze_symbol(self, symbol: str, data: pd.DataFrame) -> Optional[TradingSignal]:
        """Analyze symbol for breakout opportunities"""

        if len(data) < 30:
            return None

        try:
            close = data['Close']
            high = data['High']
            low = data['Low']
            volume = data['Volume']

            # Calculate volatility (consolidation detection)
            returns = close.pct_change()
            volatility = returns.rolling(self.consolidation_days).std()
            current_volatility = volatility.iloc[-1]

            # Check if we're in a consolidation period
            in_consolidation = current_volatility < self.volatility_threshold

            if not in_consolidation:
                return None

            # Calculate support and resistance levels
            lookback_period = self.consolidation_days
            resistance = high.rolling(lookback_period).max().iloc[-1]
            support = low.rolling(lookback_period).min().iloc[-1]

            current_price = close.iloc[-1]
            current_volume = volume.iloc[-1]
            avg_volume = volume.rolling(20).mean().iloc[-1]

            # Check for breakout
            upward_breakout = (current_price > resistance * (1 + self.breakout_threshold) and
                               current_volume > avg_volume * self.volume_multiplier)

            downward_breakout = (current_price < support * (1 - self.breakout_threshold) and
                                 current_volume > avg_volume * self.volume_multiplier)

            if upward_breakout:
                signal_type = "BUY"
                atr = TechnicalIndicators.atr(high, low, close, 14).iloc[-1]
                target_price = current_price + (3 * atr)
                stop_loss = support
                # Volume-based confidence
                confidence = min(current_volume / avg_volume / 2, 1.0)

                reasoning = f"Upward breakout from consolidation | Resistance: ${resistance:.2f} | Volume: {current_volume/avg_volume:.1f}x"

            elif downward_breakout:
                signal_type = "SELL"
                atr = TechnicalIndicators.atr(high, low, close, 14).iloc[-1]
                target_price = current_price - (3 * atr)
                stop_loss = resistance
                confidence = min(current_volume / avg_volume / 2, 1.0)

                reasoning = f"Downward breakout from consolidation | Support: ${support:.2f} | Volume: {current_volume/avg_volume:.1f}x"

            else:
                return None

            return TradingSignal(
                symbol=symbol,
                signal=signal_type,
                confidence=confidence,
                timestamp=datetime.now(),
                current_price=current_price,
                target_price=target_price,
                stop_loss=stop_loss,
                reasoning=reasoning
            )

        except Exception as e:
            logger.error(f"Error in breakout analysis for {symbol}: {str(e)}")
            return None


class CustomTradingBot(TradingBot):
    """
    Extended trading bot with custom strategies
    """

    def __init__(self, config_file: str = 'config.yaml'):
        super().__init__(config_file)

        # Initialize custom strategies
        self.strategies = [
            MomentumMeanReversionStrategy(self.config),
            BreakoutStrategy(self.config)
        ]

    def generate_custom_signals(self) -> List[TradingSignal]:
        """Generate signals using custom strategies"""
        all_signals = []
        symbols = self.config['data']['symbols']

        # Get current data
        current_data = self.data_collector.get_multiple_stocks(
            symbols, period='6mo')

        for symbol, data in current_data.items():
            if data.empty:
                continue

            # Run each strategy
            for strategy in self.strategies:
                signal = strategy.analyze_symbol(symbol, data)
                if signal:
                    all_signals.append(signal)

        return all_signals

    def combine_signals(self, ml_signals: List[TradingSignal],
                        custom_signals: List[TradingSignal]) -> List[TradingSignal]:
        """
        Combine ML and custom strategy signals
        Uses ensemble approach for better decision making
        """
        combined_signals = {}

        # Process ML signals
        for signal in ml_signals:
            if signal.symbol not in combined_signals:
                combined_signals[signal.symbol] = {
                    'signals': [],
                    'confidences': [],
                    'buy_votes': 0,
                    'sell_votes': 0,
                    'hold_votes': 0
                }

            combined_signals[signal.symbol]['signals'].append(('ML', signal))
            combined_signals[signal.symbol]['confidences'].append(
                signal.confidence)

            if signal.signal == 'BUY':
                combined_signals[signal.symbol]['buy_votes'] += signal.confidence
            elif signal.signal == 'SELL':
                combined_signals[signal.symbol]['sell_votes'] += signal.confidence
            else:
                combined_signals[signal.symbol]['hold_votes'] += signal.confidence

        # Process custom strategy signals
        for signal in custom_signals:
            if signal.symbol not in combined_signals:
                combined_signals[signal.symbol] = {
                    'signals': [],
                    'confidences': [],
                    'buy_votes': 0,
                    'sell_votes': 0,
                    'hold_votes': 0
                }

            combined_signals[signal.symbol]['signals'].append(
                ('Custom', signal))
            combined_signals[signal.symbol]['confidences'].append(
                signal.confidence)

            if signal.signal == 'BUY':
                combined_signals[signal.symbol]['buy_votes'] += signal.confidence
            elif signal.signal == 'SELL':
                combined_signals[signal.symbol]['sell_votes'] += signal.confidence
            else:
                combined_signals[signal.symbol]['hold_votes'] += signal.confidence

        # Generate final signals
        final_signals = []

        for symbol, data in combined_signals.items():
            if not data['signals']:
                continue

            # Determine consensus signal
            buy_score = data['buy_votes']
            sell_score = data['sell_votes']
            hold_score = data['hold_votes']

            total_votes = buy_score + sell_score + hold_score
            if total_votes == 0:
                continue

            if buy_score > sell_score and buy_score > hold_score:
                consensus_signal = 'BUY'
                consensus_confidence = buy_score / len(data['signals'])
            elif sell_score > buy_score and sell_score > hold_score:
                consensus_signal = 'SELL'
                consensus_confidence = sell_score / len(data['signals'])
            else:
                consensus_signal = 'HOLD'
                consensus_confidence = hold_score / len(data['signals'])

            # Only emit signal if confidence is high enough
            if consensus_confidence >= self.config['trading']['min_confidence']:
                # Use the signal with highest individual confidence for price/target info
                best_signal = max(
                    data['signals'], key=lambda x: x[1].confidence)[1]

                reasoning_parts = []
                for source, signal in data['signals']:
                    reasoning_parts.append(
                        f"{source}: {signal.signal} ({signal.confidence:.2f})")

                final_signal = TradingSignal(
                    symbol=symbol,
                    signal=consensus_signal,
                    confidence=consensus_confidence,
                    timestamp=datetime.now(),
                    current_price=best_signal.current_price,
                    target_price=best_signal.target_price,
                    stop_loss=best_signal.stop_loss,
                    reasoning=f"Consensus from {len(data['signals'])} signals: {'; '.join(reasoning_parts)}"
                )

                final_signals.append(final_signal)

        return final_signals

    def run_single_cycle(self) -> None:
        """Override to use custom signal generation"""
        try:
            logger.info("Starting custom trading cycle...")

            # Retrain ML model if needed
            self.retrain_model()

            # Update existing positions
            self.update_positions()

            # Generate ML signals
            ml_signals = super().generate_signals()

            # Generate custom strategy signals
            custom_signals = self.generate_custom_signals()

            # Combine signals using ensemble approach
            final_signals = self.combine_signals(ml_signals, custom_signals)

            if final_signals:
                logger.info(
                    f"Generated {len(final_signals)} consensus signals")
                self.process_signals(final_signals)
            else:
                logger.info("No consensus signals generated this cycle")

            # Log portfolio status
            total_positions = len(self.risk_manager.positions)
            total_unrealized_pnl = sum(
                pos.unrealized_pnl for pos in self.risk_manager.positions.values())

            logger.info(
                f"Portfolio Status: {total_positions} positions, Unrealized P&L: ${total_unrealized_pnl:.2f}")

        except Exception as e:
            logger.error(f"Error in custom trading cycle: {str(e)}")


def run_sample_strategy():
    """Run the sample strategy implementation"""

    # Create custom configuration for the sample strategy
    config = {
        'trading': {
            'max_positions': 3,
            'max_position_size': 0.15,
            'stop_loss_pct': 0.04,
            'take_profit_pct': 0.12,
            'min_confidence': 0.65
        },
        'data': {
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'TSLA'],
            'update_frequency': 600,  # Check every 10 minutes
        },
        'ml': {
            'retrain_frequency': 5,  # Retrain every 5 days
            'prediction_horizon': 3  # Predict 3 days ahead
        }
    }

    # Save custom config
    import yaml
    with open('sample_strategy_config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    # Initialize and run custom bot
    bot = CustomTradingBot('sample_strategy_config.yaml')
    bot.initialize()

    logger.info("Starting Sample Strategy Trading Bot...")
    logger.info("Strategies: Momentum Mean Reversion + Breakout")
    logger.info("This is PAPER TRADING mode - no real trades will be executed")

    try:
        # Run a few cycles for demonstration
        for i in range(3):
            logger.info(f"\n--- Cycle {i+1} ---")
            bot.run_single_cycle()

            # Show performance report
            performance = bot.get_performance_report()
            logger.info(f"Performance Report: {performance}")

            # Wait before next cycle (in real usage, this would be automatic)
            import time
            time.sleep(10)

    except KeyboardInterrupt:
        logger.info("Stopping sample strategy bot...")
        bot.stop()


if __name__ == "__main__":
    run_sample_strategy()

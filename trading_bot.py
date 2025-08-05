# trading_bot.py
"""
Advanced Algorithmic Trading Bot
Author: Claude Assistant
Version: 1.0

IMPORTANT DISCLAIMER:
This software is for educational and research purposes only. Trading involves 
substantial risk of loss and is not suitable for all investors. Past performance 
does not guarantee future results. Always use paper trading first and never risk 
more than you can afford to lose.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import logging
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import sqlite3
import threading
import time
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


@dataclass
class TradingSignal:
    """Data class for trading signals"""
    symbol: str
    signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    timestamp: datetime
    current_price: float
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    position_size: Optional[float] = None
    reasoning: Optional[str] = None


@dataclass
class Position:
    """Data class for tracking positions"""
    symbol: str
    quantity: float
    entry_price: float
    entry_date: datetime
    current_price: float
    unrealized_pnl: float
    stop_loss: Optional[float] = None
    target_price: Optional[float] = None


class ConfigManager:
    """Configuration management for the trading bot"""

    def __init__(self, config_file: str = 'config.yaml'):
        self.config_file = config_file
        self.config = self.load_config()

    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            with open(self.config_file, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            logger.warning(
                f"Config file {self.config_file} not found. Using defaults.")
            return self.get_default_config()

    def get_default_config(self) -> Dict[str, Any]:
        """Return default configuration"""
        return {
            'trading': {
                'max_positions': 10,
                'max_position_size': 0.1,  # 10% of portfolio per position
                'stop_loss_pct': 0.05,     # 5% stop loss
                'take_profit_pct': 0.15,   # 15% take profit
                'risk_free_rate': 0.02,    # 2% risk-free rate
                'lookback_days': 252,      # 1 year of data
                'min_confidence': 0.6      # Minimum confidence for trades
            },
            'data': {
                'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'],
                'update_frequency': 300,   # Update every 5 minutes
                'data_source': 'yahoo'
            },
            'alerts': {
                'email_enabled': False,
                'email_smtp': 'smtp.gmail.com',
                'email_port': 587,
                'email_user': '',
                'email_password': '',
                'email_recipients': []
            },
            'ml': {
                'model_type': 'random_forest',
                'retrain_frequency': 7,    # Retrain every 7 days
                'feature_window': 20,      # 20 days for features
                'prediction_horizon': 5    # Predict 5 days ahead
            }
        }


class DataCollector:
    """Handles data collection from various sources"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache = {}
        self.last_update = {}

    def get_stock_data(self, symbol: str, period: str = '1y') -> pd.DataFrame:
        """Get stock data from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)

            if data.empty:
                logger.error(f"No data retrieved for {symbol}")
                return pd.DataFrame()

            # Add additional columns
            data['Symbol'] = symbol
            data['Returns'] = data['Close'].pct_change()

            logger.info(f"Retrieved {len(data)} records for {symbol}")
            return data

        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_multiple_stocks(self, symbols: List[str], period: str = '1y') -> Dict[str, pd.DataFrame]:
        """Get data for multiple stocks concurrently"""
        data = {}

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_symbol = {
                executor.submit(self.get_stock_data, symbol, period): symbol
                for symbol in symbols
            }

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    data[symbol] = future.result()
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {str(e)}")
                    data[symbol] = pd.DataFrame()

        return data

    def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """Get additional market information"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            return {
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('forwardPE', 0),
                'beta': info.get('beta', 1.0),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'dividend_yield': info.get('dividendYield', 0),
                'book_value': info.get('bookValue', 0),
                'debt_to_equity': info.get('debtToEquity', 0)
            }
        except Exception as e:
            logger.error(f"Error getting market info for {symbol}: {str(e)}")
            return {}


class TechnicalIndicators:
    """Calculate various technical indicators"""

    @staticmethod
    def sma(data: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()

    @staticmethod
    def ema(data: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()

    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """MACD indicator"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, window)
        std = data.rolling(window=window).std()
        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)
        return upper_band, sma, lower_band

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        return k_percent, d_percent

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()


class FeatureEngineering:
    """Feature engineering for machine learning models"""

    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.features = pd.DataFrame(index=data.index)

    def create_technical_features(self) -> pd.DataFrame:
        """Create technical analysis features"""
        close = self.data['Close']
        high = self.data['High']
        low = self.data['Low']
        volume = self.data['Volume']

        # Moving averages
        for window in [5, 10, 20, 50]:
            self.features[f'SMA_{window}'] = TechnicalIndicators.sma(
                close, window)
            self.features[f'EMA_{window}'] = TechnicalIndicators.ema(
                close, window)
            self.features[f'Price_to_SMA_{window}'] = close / \
                self.features[f'SMA_{window}']

        # Technical indicators
        self.features['RSI'] = TechnicalIndicators.rsi(close)

        macd, signal, histogram = TechnicalIndicators.macd(close)
        self.features['MACD'] = macd
        self.features['MACD_Signal'] = signal
        self.features['MACD_Histogram'] = histogram

        upper_bb, middle_bb, lower_bb = TechnicalIndicators.bollinger_bands(
            close)
        self.features['BB_Upper'] = upper_bb
        self.features['BB_Middle'] = middle_bb
        self.features['BB_Lower'] = lower_bb
        self.features['BB_Position'] = (
            close - lower_bb) / (upper_bb - lower_bb)

        # Volume features
        self.features['Volume_SMA_20'] = TechnicalIndicators.sma(volume, 20)
        self.features['Volume_Ratio'] = volume / self.features['Volume_SMA_20']

        # Price features
        self.features['Price_Change'] = close.pct_change()
        self.features['Price_Change_5d'] = close.pct_change(5)
        self.features['Volatility_20d'] = close.pct_change().rolling(20).std()

        # ATR
        self.features['ATR'] = TechnicalIndicators.atr(high, low, close)

        return self.features

    def create_momentum_features(self) -> pd.DataFrame:
        """Create momentum-based features"""
        close = self.data['Close']

        # Rate of change
        for period in [1, 5, 10, 20]:
            self.features[f'ROC_{period}'] = close.pct_change(period)

        # Momentum
        for period in [5, 10, 20]:
            self.features[f'Momentum_{period}'] = close / \
                close.shift(period) - 1

        return self.features

    def create_target_variable(self, horizon: int = 5, threshold: float = 0.02) -> pd.Series:
        """Create target variable for classification"""
        close = self.data['Close']
        future_return = close.shift(-horizon) / close - 1

        # Create classification target
        target = pd.Series(index=close.index, dtype=int)
        target[future_return > threshold] = 1   # Buy
        target[future_return < -threshold] = -1  # Sell
        target[(future_return >= -threshold) &
               (future_return <= threshold)] = 0  # Hold

        return target


class MLPredictor:
    """Machine learning predictor for trading signals"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_importance = None

    def prepare_data(self, data: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for machine learning"""
        all_features = []
        all_targets = []

        for symbol, df in data.items():
            if df.empty:
                continue

            fe = FeatureEngineering(df)
            features = fe.create_technical_features()
            features = fe.create_momentum_features()

            target = fe.create_target_variable(
                horizon=self.config['ml']['prediction_horizon']
            )

            # Add symbol as feature
            features['Symbol'] = symbol
            features = pd.get_dummies(
                features, columns=['Symbol'], prefix='Symbol')

            # Remove NaN values
            valid_idx = features.dropna().index.intersection(target.dropna().index)
            if len(valid_idx) > 0:
                all_features.append(features.loc[valid_idx])
                all_targets.append(target.loc[valid_idx])

        if not all_features:
            return pd.DataFrame(), pd.Series()

        combined_features = pd.concat(all_features, axis=0)
        combined_targets = pd.concat(all_targets, axis=0)

        return combined_features, combined_targets

    def train_model(self, features: pd.DataFrame, targets: pd.Series) -> None:
        """Train the machine learning model"""
        if features.empty or targets.empty:
            logger.error("No data available for training")
            return

        # Remove any remaining NaN values
        valid_idx = features.dropna().index.intersection(targets.dropna().index)
        X = features.loc[valid_idx]
        y = targets.loc[valid_idx]

        if len(X) < 100:
            logger.warning("Insufficient data for reliable training")
            return

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Train model
        if self.config['ml']['model_type'] == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        else:
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )

        self.model.fit(X_train_scaled, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"Model trained with accuracy: {accuracy:.3f}")
        logger.info(
            f"Classification Report:\n{classification_report(y_test, y_pred)}")

        # Store feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            logger.info("Top 10 most important features:")
            logger.info(self.feature_importance.head(10))

        self.is_trained = True

    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using the trained model"""
        if not self.is_trained or self.model is None:
            logger.error("Model not trained yet")
            return np.array([]), np.array([])

        # Ensure features have the same columns as training data
        X = features.reindex(
            columns=self.scaler.feature_names_in_, fill_value=0)
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)

        return predictions, probabilities


class RiskManager:
    """Risk management system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.positions = {}
        self.portfolio_value = 100000  # Starting portfolio value
        self.max_portfolio_risk = 0.02  # 2% max portfolio risk

    def calculate_position_size(self, signal: TradingSignal, portfolio_value: float) -> float:
        """Calculate appropriate position size based on risk management"""
        max_position_value = portfolio_value * \
            self.config['trading']['max_position_size']

        if signal.stop_loss and signal.current_price:
            # Risk-based position sizing
            risk_per_share = abs(signal.current_price - signal.stop_loss)
            max_risk = portfolio_value * self.max_portfolio_risk
            shares_by_risk = max_risk / risk_per_share
            position_value_by_risk = shares_by_risk * signal.current_price

            # Use the smaller of the two position sizes
            max_position_value = min(
                max_position_value, position_value_by_risk)

        return max_position_value / signal.current_price

    def check_risk_limits(self, signal: TradingSignal) -> bool:
        """Check if trade meets risk management criteria"""
        # Check confidence threshold
        if signal.confidence < self.config['trading']['min_confidence']:
            logger.info(
                f"Signal for {signal.symbol} below confidence threshold")
            return False

        # Check maximum positions
        if len(self.positions) >= self.config['trading']['max_positions']:
            logger.info("Maximum number of positions reached")
            return False

        # Check if already have position in this symbol
        if signal.symbol in self.positions:
            logger.info(f"Already have position in {signal.symbol}")
            return False

        return True

    def update_stop_losses(self, current_prices: Dict[str, float]) -> List[str]:
        """Update stop losses and check for exits"""
        symbols_to_exit = []

        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                position.current_price = current_price
                position.unrealized_pnl = (
                    current_price - position.entry_price) * position.quantity

                # Check stop loss
                if position.stop_loss and current_price <= position.stop_loss:
                    symbols_to_exit.append(symbol)
                    logger.info(
                        f"Stop loss triggered for {symbol} at {current_price}")

                # Check take profit
                if position.target_price and current_price >= position.target_price:
                    symbols_to_exit.append(symbol)
                    logger.info(
                        f"Take profit triggered for {symbol} at {current_price}")

        return symbols_to_exit


class AlertSystem:
    """Alert and notification system"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def send_email_alert(self, subject: str, message: str) -> None:
        """Send email alert"""
        if not self.config['alerts']['email_enabled']:
            return

        try:
            msg = MimeMultipart()
            msg['From'] = self.config['alerts']['email_user']
            msg['Subject'] = subject

            msg.attach(MimeText(message, 'plain'))

            server = smtplib.SMTP(
                self.config['alerts']['email_smtp'],
                self.config['alerts']['email_port']
            )
            server.starttls()
            server.login(
                self.config['alerts']['email_user'],
                self.config['alerts']['email_password']
            )

            for recipient in self.config['alerts']['email_recipients']:
                msg['To'] = recipient
                text = msg.as_string()
                server.sendmail(
                    self.config['alerts']['email_user'],
                    recipient,
                    text
                )

            server.quit()
            logger.info("Email alert sent successfully")

        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")

    def send_trading_signal_alert(self, signal: TradingSignal) -> None:
        """Send alert for trading signal"""
        subject = f"Trading Signal: {signal.signal} {signal.symbol}"
        message = f"""
        Trading Signal Generated:
        
        Symbol: {signal.symbol}
        Signal: {signal.signal}
        Confidence: {signal.confidence:.2%}
        Current Price: ${signal.current_price:.2f}
        Target Price: ${signal.target_price:.2f if signal.target_price else 'N/A'}
        Stop Loss: ${signal.stop_loss:.2f if signal.stop_loss else 'N/A'}
        Timestamp: {signal.timestamp}
        
        Reasoning: {signal.reasoning or 'N/A'}
        """

        logger.info(
            f"TRADING SIGNAL: {signal.signal} {signal.symbol} - Confidence: {signal.confidence:.2%}")
        self.send_email_alert(subject, message)


class DatabaseManager:
    """Database operations for storing trading data"""

    def __init__(self, db_file: str = 'trading_bot.db'):
        self.db_file = db_file
        self.init_database()

    def init_database(self) -> None:
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        # Create signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                signal TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp TEXT NOT NULL,
                current_price REAL NOT NULL,
                target_price REAL,
                stop_loss REAL,
                reasoning TEXT
            )
        ''')

        # Create positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                entry_price REAL NOT NULL,
                entry_date TEXT NOT NULL,
                exit_price REAL,
                exit_date TEXT,
                realized_pnl REAL,
                status TEXT DEFAULT 'OPEN'
            )
        ''')

        conn.commit()
        conn.close()

    def save_signal(self, signal: TradingSignal) -> None:
        """Save trading signal to database"""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO signals (symbol, signal, confidence, timestamp, current_price, 
                               target_price, stop_loss, reasoning)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.symbol, signal.signal, signal.confidence,
            signal.timestamp.isoformat(), signal.current_price,
            signal.target_price, signal.stop_loss, signal.reasoning
        ))

        conn.commit()
        conn.close()

    def get_recent_signals(self, days: int = 7) -> pd.DataFrame:
        """Get recent signals from database"""
        conn = sqlite3.connect(self.db_file)

        query = '''
            SELECT * FROM signals 
            WHERE timestamp >= datetime('now', '-{} days')
            ORDER BY timestamp DESC
        '''.format(days)

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df


class TradingBot:
    """Main trading bot class"""

    def __init__(self, config_file: str = 'config.yaml'):
        self.config_manager = ConfigManager(config_file)
        self.config = self.config_manager.config

        self.data_collector = DataCollector(self.config)
        self.ml_predictor = MLPredictor(self.config)
        self.risk_manager = RiskManager(self.config)
        self.alert_system = AlertSystem(self.config)
        self.db_manager = DatabaseManager()

        self.is_running = False
        self.last_training_date = None

    def initialize(self) -> None:
        """Initialize the trading bot"""
        logger.info("Initializing Trading Bot...")

        # Load historical data
        symbols = self.config['data']['symbols']
        logger.info(f"Loading data for symbols: {symbols}")

        historical_data = self.data_collector.get_multiple_stocks(
            symbols, period='2y'
        )

        # Train initial model
        logger.info("Training initial ML model...")
        features, targets = self.ml_predictor.prepare_data(historical_data)
        self.ml_predictor.train_model(features, targets)

        self.last_training_date = datetime.now()
        logger.info("Trading Bot initialized successfully!")

    def generate_signals(self) -> List[TradingSignal]:
        """Generate trading signals for all symbols"""
        signals = []
        symbols = self.config['data']['symbols']

        # Get current data
        current_data = self.data_collector.get_multiple_stocks(
            symbols, period='6mo'
        )

        for symbol, data in current_data.items():
            if data.empty:
                continue

            try:
                # Prepare features for prediction
                fe = FeatureEngineering(data)
                features = fe.create_technical_features()
                features = fe.create_momentum_features()

                # Add symbol encoding
                features['Symbol'] = symbol
                features = pd.get_dummies(
                    features, columns=['Symbol'], prefix='Symbol')

                # Get latest features (last row)
                latest_features = features.tail(1).dropna()

                if latest_features.empty:
                    continue

                # Make prediction
                predictions, probabilities = self.ml_predictor.predict(
                    latest_features)

                if len(predictions) == 0:
                    continue

                prediction = predictions[0]
                confidence = np.max(probabilities[0])

                # Convert prediction to signal
                signal_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
                signal_type = signal_map.get(prediction, 'HOLD')

                # Get current price
                current_price = data['Close'].iloc[-1]

                # Calculate target and stop loss
                atr = TechnicalIndicators.atr(
                    data['High'], data['Low'], data['Close']
                ).iloc[-1]

                if signal_type == 'BUY':
                    target_price = current_price * \
                        (1 + self.config['trading']['take_profit_pct'])
                    stop_loss = current_price * \
                        (1 - self.config['trading']['stop_loss_pct'])
                elif signal_type == 'SELL':
                    target_price = current_price * \
                        (1 - self.config['trading']['take_profit_pct'])
                    stop_loss = current_price * \
                        (1 + self.config['trading']['stop_loss_pct'])
                else:
                    target_price = None
                    stop_loss = None

                # Create signal
                signal = TradingSignal(
                    symbol=symbol,
                    signal=signal_type,
                    confidence=confidence,
                    timestamp=datetime.now(),
                    current_price=current_price,
                    target_price=target_price,
                    stop_loss=stop_loss,
                    reasoning=f"ML prediction: {prediction}, ATR: {atr:.2f}"
                )

                signals.append(signal)

            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {str(e)}")

        return signals

    def process_signals(self, signals: List[TradingSignal]) -> None:
        """Process and act on trading signals"""
        for signal in signals:
            # Save signal to database
            self.db_manager.save_signal(signal)

            # Check risk management
            if signal.signal in ['BUY', 'SELL'] and self.risk_manager.check_risk_limits(signal):
                # Calculate position size
                position_size = self.risk_manager.calculate_position_size(
                    signal, self.risk_manager.portfolio_value
                )
                signal.position_size = position_size

                # Send alert
                self.alert_system.send_trading_signal_alert(signal)

                # In a real implementation, you would execute the trade here
                # For now, we'll just log it as a paper trade
                logger.info(
                    f"PAPER TRADE: {signal.signal} {position_size:.2f} shares of {signal.symbol}")
                logger.info(
                    f"Entry: ${signal.current_price:.2f}, Target: ${signal.target_price:.2f}, Stop: ${signal.stop_loss:.2f}")

                # Add to positions (paper trading)
                if signal.signal == 'BUY':
                    position = Position(
                        symbol=signal.symbol,
                        quantity=position_size,
                        entry_price=signal.current_price,
                        entry_date=datetime.now(),
                        current_price=signal.current_price,
                        unrealized_pnl=0,
                        stop_loss=signal.stop_loss,
                        target_price=signal.target_price
                    )
                    self.risk_manager.positions[signal.symbol] = position

    def update_positions(self) -> None:
        """Update existing positions"""
        if not self.risk_manager.positions:
            return

        symbols = list(self.risk_manager.positions.keys())
        current_data = self.data_collector.get_multiple_stocks(
            symbols, period='1d')

        current_prices = {}
        for symbol, data in current_data.items():
            if not data.empty:
                current_prices[symbol] = data['Close'].iloc[-1]

        # Check for stop losses and take profits
        symbols_to_exit = self.risk_manager.update_stop_losses(current_prices)

        for symbol in symbols_to_exit:
            position = self.risk_manager.positions[symbol]
            realized_pnl = (position.current_price -
                            position.entry_price) * position.quantity

            logger.info(
                f"POSITION CLOSED: {symbol} - P&L: ${realized_pnl:.2f}")

            # Send exit alert
            exit_message = f"Position closed for {symbol}: P&L ${realized_pnl:.2f}"
            self.alert_system.send_email_alert(
                f"Position Closed: {symbol}", exit_message)

            # Remove from positions
            del self.risk_manager.positions[symbol]

    def retrain_model(self) -> None:
        """Retrain the ML model with new data"""
        if (self.last_training_date and
                (datetime.now() - self.last_training_date).days < self.config['ml']['retrain_frequency']):
            return

        logger.info("Retraining ML model...")

        symbols = self.config['data']['symbols']
        historical_data = self.data_collector.get_multiple_stocks(
            symbols, period='1y')

        features, targets = self.ml_predictor.prepare_data(historical_data)
        self.ml_predictor.train_model(features, targets)

        self.last_training_date = datetime.now()
        logger.info("Model retrained successfully")

    def run_single_cycle(self) -> None:
        """Run a single trading cycle"""
        try:
            logger.info("Starting trading cycle...")

            # Retrain model if needed
            self.retrain_model()

            # Update existing positions
            self.update_positions()

            # Generate new signals
            signals = self.generate_signals()

            if signals:
                logger.info(f"Generated {len(signals)} signals")
                # Process signals
                self.process_signals(signals)
            else:
                logger.info("No signals generated this cycle")

            # Log portfolio status
            total_positions = len(self.risk_manager.positions)
            total_unrealized_pnl = sum(
                pos.unrealized_pnl for pos in self.risk_manager.positions.values())

            logger.info(
                f"Portfolio Status: {total_positions} positions, Unrealized P&L: ${total_unrealized_pnl:.2f}")

        except Exception as e:
            logger.error(f"Error in trading cycle: {str(e)}")

    def run(self) -> None:
        """Run the trading bot continuously"""
        self.is_running = True
        update_frequency = self.config['data']['update_frequency']

        logger.info(
            f"Starting trading bot (update every {update_frequency} seconds)...")

        while self.is_running:
            try:
                self.run_single_cycle()
                time.sleep(update_frequency)

            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping bot...")
                self.stop()
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
                time.sleep(60)  # Wait 1 minute before retrying

    def stop(self) -> None:
        """Stop the trading bot"""
        self.is_running = False
        logger.info("Trading bot stopped")

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        recent_signals = self.db_manager.get_recent_signals(30)  # Last 30 days

        if recent_signals.empty:
            return {"message": "No recent trading activity"}

        # Signal statistics
        signal_counts = recent_signals['signal'].value_counts()
        avg_confidence = recent_signals['confidence'].mean()

        # Position statistics
        positions = self.risk_manager.positions
        total_unrealized_pnl = sum(
            pos.unrealized_pnl for pos in positions.values())

        return {
            "recent_signals": len(recent_signals),
            "signal_breakdown": signal_counts.to_dict(),
            "average_confidence": avg_confidence,
            "active_positions": len(positions),
            "unrealized_pnl": total_unrealized_pnl,
            "portfolio_value": self.risk_manager.portfolio_value
        }


class BacktestEngine:
    """Backtesting engine to evaluate trading strategies"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.initial_capital = 100000
        self.commission = 0.001  # 0.1% commission

    def run_backtest(self, start_date: str, end_date: str, symbols: List[str]) -> Dict[str, Any]:
        """Run backtest over specified period"""
        logger.info(f"Running backtest from {start_date} to {end_date}")

        # Initialize components
        data_collector = DataCollector(self.config)
        ml_predictor = MLPredictor(self.config)

        # Get historical data
        historical_data = {}
        for symbol in symbols:
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            if not data.empty:
                historical_data[symbol] = data

        if not historical_data:
            return {"error": "No historical data available"}

        # Train model on first part of data
        train_end_date = pd.to_datetime(
            start_date) + pd.Timedelta(days=252)  # 1 year for training
        train_data = {}

        for symbol, data in historical_data.items():
            train_data[symbol] = data[data.index <= train_end_date]

        features, targets = ml_predictor.prepare_data(train_data)
        ml_predictor.train_model(features, targets)

        # Run backtest
        portfolio_value = self.initial_capital
        positions = {}
        trades = []
        daily_values = []

        # Get trading dates (after training period)
        all_dates = sorted(set().union(
            *[data.index for data in historical_data.values()]))
        trading_dates = [date for date in all_dates if date > train_end_date]

        for current_date in trading_dates:
            try:
                # Get current data up to this date
                current_data = {}
                for symbol, data in historical_data.items():
                    current_data[symbol] = data[data.index <=
                                                # Last 100 days
                                                current_date].tail(100)

                # Generate signals
                signals = []
                for symbol, data in current_data.items():
                    if len(data) < 50:  # Need enough data
                        continue

                    # Prepare features
                    fe = FeatureEngineering(data)
                    features = fe.create_technical_features()
                    features = fe.create_momentum_features()

                    # Add symbol encoding
                    features['Symbol'] = symbol
                    features = pd.get_dummies(
                        features, columns=['Symbol'], prefix='Symbol')

                    latest_features = features.tail(1).dropna()
                    if latest_features.empty:
                        continue

                    # Make prediction
                    predictions, probabilities = ml_predictor.predict(
                        latest_features)
                    if len(predictions) == 0:
                        continue

                    prediction = predictions[0]
                    confidence = np.max(probabilities[0])

                    if confidence > self.config['trading']['min_confidence']:
                        signal_map = {-1: 'SELL', 0: 'HOLD', 1: 'BUY'}
                        signal_type = signal_map.get(prediction, 'HOLD')

                        if signal_type in ['BUY', 'SELL']:
                            current_price = data['Close'].iloc[-1]

                            signal = TradingSignal(
                                symbol=symbol,
                                signal=signal_type,
                                confidence=confidence,
                                timestamp=current_date,
                                current_price=current_price
                            )
                            signals.append(signal)

                # Process signals
                for signal in signals:
                    if signal.signal == 'BUY' and signal.symbol not in positions:
                        # Buy signal
                        position_value = portfolio_value * 0.1  # 10% position size
                        shares = position_value / signal.current_price
                        cost = shares * signal.current_price * \
                            (1 + self.commission)

                        if cost <= portfolio_value:
                            positions[signal.symbol] = {
                                'shares': shares,
                                'entry_price': signal.current_price,
                                'entry_date': signal.timestamp
                            }
                            portfolio_value -= cost

                            trades.append({
                                'date': signal.timestamp,
                                'symbol': signal.symbol,
                                'action': 'BUY',
                                'shares': shares,
                                'price': signal.current_price,
                                'value': cost
                            })

                    elif signal.signal == 'SELL' and signal.symbol in positions:
                        # Sell signal
                        position = positions[signal.symbol]
                        proceeds = position['shares'] * \
                            signal.current_price * (1 - self.commission)
                        portfolio_value += proceeds

                        # Calculate P&L
                        entry_value = position['shares'] * \
                            position['entry_price']
                        pnl = proceeds - entry_value

                        trades.append({
                            'date': signal.timestamp,
                            'symbol': signal.symbol,
                            'action': 'SELL',
                            'shares': position['shares'],
                            'price': signal.current_price,
                            'value': proceeds,
                            'pnl': pnl
                        })

                        del positions[signal.symbol]

                # Calculate total portfolio value (cash + positions)
                total_value = portfolio_value
                for symbol, position in positions.items():
                    if symbol in current_data and not current_data[symbol].empty:
                        current_price = current_data[symbol]['Close'].iloc[-1]
                        total_value += position['shares'] * current_price

                daily_values.append({
                    'date': current_date,
                    'portfolio_value': total_value,
                    'cash': portfolio_value,
                    'positions_value': total_value - portfolio_value
                })

            except Exception as e:
                logger.error(f"Error processing date {current_date}: {str(e)}")
                continue

        # Calculate performance metrics
        if not daily_values:
            return {"error": "No backtest data generated"}

        df_values = pd.DataFrame(daily_values)
        df_values.set_index('date', inplace=True)

        final_value = df_values['portfolio_value'].iloc[-1]
        total_return = (final_value - self.initial_capital) / \
            self.initial_capital

        # Calculate Sharpe ratio
        returns = df_values['portfolio_value'].pct_change().dropna()
        if len(returns) > 0:
            sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
            max_drawdown = (df_values['portfolio_value'] /
                            df_values['portfolio_value'].cummax() - 1).min()
        else:
            sharpe_ratio = 0
            max_drawdown = 0

        return {
            "initial_capital": self.initial_capital,
            "final_value": final_value,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_trades": len(trades),
            "daily_values": df_values.to_dict('records'),
            "trades": trades
        }


def create_default_config():
    """Create default configuration file"""
    config = {
        'trading': {
            'max_positions': 5,
            'max_position_size': 0.2,
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.15,
            'risk_free_rate': 0.02,
            'lookback_days': 252,
            'min_confidence': 0.65
        },
        'data': {
            'symbols': ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA'],
            'update_frequency': 300,
            'data_source': 'yahoo'
        },
        'alerts': {
            'email_enabled': False,
            'email_smtp': 'smtp.gmail.com',
            'email_port': 587,
            'email_user': 'your_email@gmail.com',
            'email_password': 'your_app_password',
            'email_recipients': ['your_email@gmail.com']
        },
        'ml': {
            'model_type': 'random_forest',
            'retrain_frequency': 7,
            'feature_window': 20,
            'prediction_horizon': 5
        }
    }

    with open('config.yaml', 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    print("Created default config.yaml file. Please customize it for your needs.")


def main():
    """Main function to run the trading bot"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Advanced Algorithmic Trading Bot')
    parser.add_argument('--mode', choices=['run', 'backtest', 'config'], default='run',
                        help='Mode to run the bot in')
    parser.add_argument('--start-date', default='2023-01-01',
                        help='Start date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2024-01-01',
                        help='End date for backtesting (YYYY-MM-DD)')
    parser.add_argument('--symbols', nargs='+', default=['AAPL', 'GOOGL', 'MSFT'],
                        help='Symbols to trade/backtest')

    args = parser.parse_args()

    if args.mode == 'config':
        create_default_config()
        return

    # Create config if it doesn't exist
    if not os.path.exists('config.yaml'):
        create_default_config()

    if args.mode == 'run':
        # Run the trading bot
        bot = TradingBot()
        bot.initialize()

        try:
            bot.run()
        except KeyboardInterrupt:
            print("\nShutting down trading bot...")
            bot.stop()

    elif args.mode == 'backtest':
        # Run backtest
        config_manager = ConfigManager()
        backtest_engine = BacktestEngine(config_manager.config)

        results = backtest_engine.run_backtest(
            args.start_date, args.end_date, args.symbols
        )

        print("\n" + "="*50)
        print("BACKTEST RESULTS")
        print("="*50)

        if 'error' in results:
            print(f"Error: {results['error']}")
        else:
            print(f"Initial Capital: ${results['initial_capital']:,.2f}")
            print(f"Final Value: ${results['final_value']:,.2f}")
            print(f"Total Return: {results['total_return']:.2%}")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
            print(f"Max Drawdown: {results['max_drawdown']:.2%}")
            print(f"Total Trades: {results['total_trades']}")

            # Plot results if matplotlib is available
            try:
                df_values = pd.DataFrame(results['daily_values'])
                df_values['date'] = pd.to_datetime(df_values['date'])
                df_values.set_index('date', inplace=True)

                plt.figure(figsize=(12, 6))
                plt.plot(df_values.index, df_values['portfolio_value'])
                plt.title('Backtest Performance')
                plt.xlabel('Date')
                plt.ylabel('Portfolio Value ($)')
                plt.grid(True)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('backtest_results.png')
                print("\nBacktest chart saved as 'backtest_results.png'")

            except Exception as e:
                print(f"Could not generate chart: {str(e)}")


if __name__ == "__main__":
    main()

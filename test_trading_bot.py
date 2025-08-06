# test_trading_bot.py
"""
Unit tests for the trading bot system
Run with: python -m pytest test_trading_bot.py -v
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import tempfile
import os
import sqlite3
from unittest.mock import Mock, patch, MagicMock

# Import trading bot components
from trading_bot import (
    TechnicalIndicators, FeatureEngineering, TradingSignal,
    Position, ConfigManager, DataCollector, MLPredictor,
    RiskManager, DatabaseManager, TradingBot
)


class TestTechnicalIndicators(unittest.TestCase):
    """Test technical indicator calculations"""

    def setUp(self):
        # Create sample price data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)  # For reproducible tests

        # Generate realistic price data
        price_base = 100
        price_changes = np.random.normal(0.001, 0.02, 100)  # Daily returns
        prices = [price_base]

        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))

        self.sample_data = pd.Series(prices, index=dates)

        # Sample OHLCV data
        self.ohlcv_data = pd.DataFrame({
            'Open': self.sample_data * 0.995,
            'High': self.sample_data * 1.01,
            'Low': self.sample_data * 0.99,
            'Close': self.sample_data,
            'Volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)

    def test_sma_calculation(self):
        """Test Simple Moving Average calculation"""
        sma = TechnicalIndicators.sma(self.sample_data, 5)

        # Check that we get expected number of values
        self.assertEqual(len(sma), len(self.sample_data))

        # Check that first few values are NaN
        self.assertTrue(pd.isna(sma.iloc[:4]).all())

        # Check that calculation is correct for known values
        expected_sma_5 = self.sample_data.iloc[:5].mean()
        self.assertAlmostEqual(sma.iloc[4], expected_sma_5, places=6)

    def test_ema_calculation(self):
        """Test Exponential Moving Average calculation"""
        ema = TechnicalIndicators.ema(self.sample_data, 10)

        # EMA should have same length as input
        self.assertEqual(len(ema), len(self.sample_data))

        # EMA should not have NaN values (except possibly first)
        self.assertFalse(pd.isna(ema.iloc[1:]).any())

        # EMA should be different from SMA
        sma = TechnicalIndicators.sma(self.sample_data, 10)
        self.assertFalse(ema.equals(sma))

    def test_rsi_calculation(self):
        """Test RSI calculation"""
        rsi = TechnicalIndicators.rsi(self.sample_data, 14)

        # RSI should be between 0 and 100
        valid_rsi = rsi.dropna()
        self.assertTrue((valid_rsi >= 0).all())
        self.assertTrue((valid_rsi <= 100).all())

        # Should have NaN values for first 14 periods
        self.assertTrue(pd.isna(rsi.iloc[:14]).any())

    def test_macd_calculation(self):
        """Test MACD calculation"""
        macd_line, signal_line, histogram = TechnicalIndicators.macd(
            self.sample_data)

        # All should have same length
        self.assertEqual(len(macd_line), len(self.sample_data))
        self.assertEqual(len(signal_line), len(self.sample_data))
        self.assertEqual(len(histogram), len(self.sample_data))

        # Histogram should equal MACD - Signal
        valid_idx = ~(pd.isna(macd_line) | pd.isna(signal_line))
        calculated_histogram = macd_line[valid_idx] - signal_line[valid_idx]
        actual_histogram = histogram[valid_idx]

        pd.testing.assert_series_equal(
            calculated_histogram, actual_histogram, check_names=False)

    def test_bollinger_bands_calculation(self):
        """Test Bollinger Bands calculation"""
        upper, middle, lower = TechnicalIndicators.bollinger_bands(
            self.sample_data, 20, 2)

        # All should have same length
        self.assertEqual(len(upper), len(self.sample_data))
        self.assertEqual(len(middle), len(self.sample_data))
        self.assertEqual(len(lower), len(self.sample_data))

        # Upper should be above middle, middle above lower (where not NaN)
        valid_idx = ~(pd.isna(upper) | pd.isna(middle) | pd.isna(lower))
        self.assertTrue((upper[valid_idx] >= middle[valid_idx]).all())
        self.assertTrue((middle[valid_idx] >= lower[valid_idx]).all())

    def test_atr_calculation(self):
        """Test Average True Range calculation"""
        atr = TechnicalIndicators.atr(
            self.ohlcv_data['High'],
            self.ohlcv_data['Low'],
            self.ohlcv_data['Close'],
            14
        )

        # ATR should be positive where not NaN
        valid_atr = atr.dropna()
        self.assertTrue((valid_atr > 0).all())

        # Should have appropriate number of NaN values
        self.assertTrue(pd.isna(atr.iloc[:14]).any())


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functionality"""

    def setUp(self):
        # Create sample OHLCV data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        np.random.seed(42)

        base_price = 100
        price_changes = np.random.normal(0.001, 0.02, 100)
        prices = [base_price]

        for change in price_changes[1:]:
            prices.append(prices[-1] * (1 + change))

        self.sample_data = pd.DataFrame({
            'Open': [p * 0.995 for p in prices],
            'High': [p * 1.01 for p in prices],
            'Low': [p * 0.99 for p in prices],
            'Close': prices,
            'Volume': np.random.randint(100000, 1000000, 100)
        }, index=dates)

    def test_technical_features_creation(self):
        """Test creation of technical features"""
        fe = FeatureEngineering(self.sample_data)
        features = fe.create_technical_features()

        # Check that features DataFrame is created
        self.assertIsInstance(features, pd.DataFrame)
        self.assertEqual(len(features), len(self.sample_data))

        # Check for expected feature columns
        expected_features = [
            'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
            'EMA_5', 'EMA_10', 'EMA_20', 'EMA_50',
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Histogram',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Position'
        ]

        for feature in expected_features:
            self.assertIn(feature, features.columns)

    def test_momentum_features_creation(self):
        """Test creation of momentum features"""
        fe = FeatureEngineering(self.sample_data)
        features = fe.create_momentum_features()

        # Check for momentum feature columns
        expected_momentum_features = [
            'ROC_1', 'ROC_5', 'ROC_10', 'ROC_20',
            'Momentum_5', 'Momentum_10', 'Momentum_20'
        ]

        for feature in expected_momentum_features:
            self.assertIn(feature, features.columns)

    def test_target_variable_creation(self):
        """Test target variable creation"""
        fe = FeatureEngineering(self.sample_data)
        target = fe.create_target_variable(horizon=5, threshold=0.02)

        # Target should be integer values: -1, 0, 1
        unique_values = set(target.dropna().unique())
        expected_values = {-1, 0, 1}
        self.assertTrue(unique_values.issubset(expected_values))

        # Should have same index as input data
        self.assertTrue(target.index.equals(self.sample_data.index))


class TestTradingSignal(unittest.TestCase):
    """Test TradingSignal data class"""

    def test_trading_signal_creation(self):
        """Test creation of trading signal"""
        signal = TradingSignal(
            symbol='AAPL',
            signal='BUY',
            confidence=0.75,
            timestamp=datetime.now(),
            current_price=150.0,
            target_price=160.0,
            stop_loss=145.0
        )

        self.assertEqual(signal.symbol, 'AAPL')
        self.assertEqual(signal.signal, 'BUY')
        self.assertEqual(signal.confidence, 0.75)
        self.assertEqual(signal.current_price, 150.0)
        self.assertEqual(signal.target_price, 160.0)
        self.assertEqual(signal.stop_loss, 145.0)


class TestConfigManager(unittest.TestCase):
    """Test configuration management"""

    def test_default_config_creation(self):
        """Test creation of default configuration"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_file = f.name

        try:
            os.unlink(config_file)  # Remove the file to test default creation
            config_manager = ConfigManager(config_file)
            config = config_manager.config

            # Check that default config has required sections
            required_sections = ['trading', 'data', 'alerts', 'ml']
            for section in required_sections:
                self.assertIn(section, config)

            # Check some specific default values
            self.assertIn('max_positions', config['trading'])
            self.assertIn('symbols', config['data'])

        finally:
            if os.path.exists(config_file):
                os.unlink(config_file)


class TestRiskManager(unittest.TestCase):
    """Test risk management functionality"""

    def setUp(self):
        self.config = {
            'trading': {
                'max_positions': 5,
                'max_position_size': 0.2,
                'min_confidence': 0.6,
                'stop_loss_pct': 0.05
            }
        }
        self.risk_manager = RiskManager(self.config)

    def test_position_size_calculation(self):
        """Test position size calculation"""
        signal = TradingSignal(
            symbol='AAPL',
            signal='BUY',
            confidence=0.75,
            timestamp=datetime.now(),
            current_price=100.0,
            stop_loss=95.0
        )

        portfolio_value = 10000
        position_size = self.risk_manager.calculate_position_size(
            signal, portfolio_value)

        # Position size should be positive
        self.assertGreater(position_size, 0)

        # Position value should not exceed max position size
        position_value = position_size * signal.current_price
        max_position_value = portfolio_value * \
            self.config['trading']['max_position_size']
        # Small tolerance for rounding
        self.assertLessEqual(position_value, max_position_value * 1.01)

    def test_risk_limits_check(self):
        """Test risk limits checking"""
        signal = TradingSignal(
            symbol='AAPL',
            signal='BUY',
            confidence=0.75,
            timestamp=datetime.now(),
            current_price=100.0
        )

        # Should pass with good signal
        self.assertTrue(self.risk_manager.check_risk_limits(signal))

        # Should fail with low confidence
        signal.confidence = 0.5
        self.assertFalse(self.risk_manager.check_risk_limits(signal))

        # Should fail when max positions reached
        signal.confidence = 0.75
        for i in range(6):  # Add more than max positions
            self.risk_manager.positions[f'TEST{i}'] = Position(
                symbol=f'TEST{i}',
                quantity=10,
                entry_price=100,
                entry_date=datetime.now(),
                current_price=100,
                unrealized_pnl=0
            )

        self.assertFalse(self.risk_manager.check_risk_limits(signal))


class TestDatabaseManager(unittest.TestCase):
    """Test database operations"""

    def setUp(self):
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(suffix='.db', delete=False)
        self.temp_db.close()
        self.db_manager = DatabaseManager(self.temp_db.name)

    def tearDown(self):
        os.unlink(self.temp_db.name)

    def test_database_initialization(self):
        """Test database table creation"""
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()

        # Check that tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        self.assertIn('signals', tables)
        self.assertIn('positions', tables)

        conn.close()

    def test_signal_save_and_retrieve(self):
        """Test saving and retrieving signals"""
        signal = TradingSignal(
            symbol='AAPL',
            signal='BUY',
            confidence=0.75,
            timestamp=datetime.now(),
            current_price=150.0,
            target_price=160.0,
            stop_loss=145.0,
            reasoning='Test signal'
        )

        # Save signal
        self.db_manager.save_signal(signal)

        # Retrieve signals
        recent_signals = self.db_manager.get_recent_signals(7)

        self.assertEqual(len(recent_signals), 1)
        self.assertEqual(recent_signals.iloc[0]['symbol'], 'AAPL')
        self.assertEqual(recent_signals.iloc[0]['signal'], 'BUY')
        self.assertEqual(recent_signals.iloc[0]['confidence'], 0.75)


class TestDataCollector(unittest.TestCase):
    """Test data collection functionality"""

    def setUp(self):
        self.config = {
            'data': {
                'symbols': ['AAPL', 'MSFT'],
                'update_frequency': 300
            }
        }
        self.data_collector = DataCollector(self.config)

    @patch('yfinance.Ticker')
    def test_stock_data_retrieval(self, mock_ticker):
        """Test stock data retrieval (mocked)"""
        # Mock yfinance response
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [101, 102, 103],
            'Low': [99, 100, 101],
            'Close': [100.5, 101.5, 102.5],
            'Volume': [1000000, 1100000, 1200000]
        }, index=pd.date_range('2023-01-01', periods=3))

        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance

        # Test data retrieval
        data = self.data_collector.get_stock_data('AAPL', '1mo')

        self.assertFalse(data.empty)
        self.assertIn('Symbol', data.columns)
        self.assertIn('Returns', data.columns)
        self.assertEqual(data['Symbol'].iloc[0], 'AAPL')


class TestMLPredictor(unittest.TestCase):
    """Test machine learning predictor"""

    def setUp(self):
        self.config = {
            'ml': {
                'model_type': 'random_forest',
                'prediction_horizon': 5,
                'feature_window': 20
            }
        }
        self.ml_predictor = MLPredictor(self.config)

    def test_data_preparation(self):
        """Test ML data preparation"""
        # Create sample data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        sample_data = {
            'AAPL': pd.DataFrame({
                'Open': np.random.randn(100) + 100,
                'High': np.random.randn(100) + 101,
                'Low': np.random.randn(100) + 99,
                'Close': np.random.randn(100) + 100,
                'Volume': np.random.randint(100000, 1000000, 100)
            }, index=dates)
        }

        features, targets = self.ml_predictor.prepare_data(sample_data)

        # Check that features and targets are created
        self.assertIsInstance(features, pd.DataFrame)
        self.assertIsInstance(targets, pd.Series)

        # Should have some features
        if not features.empty:
            self.assertGreater(len(features.columns), 0)

    def test_model_training_with_insufficient_data(self):
        """Test model training with insufficient data"""
        # Empty data should not crash
        empty_features = pd.DataFrame()
        empty_targets = pd.Series(dtype=float)

        # Should handle gracefully
        self.ml_predictor.train_model(empty_features, empty_targets)
        self.assertFalse(self.ml_predictor.is_trained)


class TestIntegration(unittest.TestCase):
    """Integration tests for the trading bot"""

    def setUp(self):
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(
            mode='w', suffix='.yaml', delete=False)
        config_content = """
trading:
  max_positions: 2
  max_position_size: 0.1
  min_confidence: 0.7
  stop_loss_pct: 0.05
  
data:
  symbols: ['AAPL']
  update_frequency: 300
  
ml:
  model_type: 'random_forest'
  retrain_frequency: 7
"""
        self.temp_config.write(config_content)
        self.temp_config.close()

    def tearDown(self):
        os.unlink(self.temp_config.name)

    @patch('trading_bot.DataCollector.get_multiple_stocks')
    def test_trading_bot_initialization(self, mock_get_data):
        """Test trading bot initialization"""
        # Mock data response
        mock_data = {
            'AAPL': pd.DataFrame({
                'Open': np.random.randn(252) + 100,
                'High': np.random.randn(252) + 101,
                'Low': np.random.randn(252) + 99,
                'Close': np.random.randn(252) + 100,
                'Volume': np.random.randint(100000, 1000000, 252)
            }, index=pd.date_range('2023-01-01', periods=252))
        }
        mock_get_data.return_value = mock_data

        # Initialize bot
        bot = TradingBot(self.temp_config.name)

        # Should initialize without errors
        self.assertIsNotNone(bot.config)
        self.assertIsNotNone(bot.data_collector)
        self.assertIsNotNone(bot.ml_predictor)
        self.assertIsNotNone(bot.risk_manager)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance calculation functions"""

    def test_returns_calculation(self):
        """Test returns calculation for performance metrics"""
        prices = pd.Series([100, 105, 103, 108, 110])
        returns = prices.pct_change().dropna()

        expected_returns = pd.Series(
            [0.05, -0.019047619, 0.048543689, 0.018518519])

        # Check returns are calculated correctly (with tolerance for floating point)
        for i, (actual, expected) in enumerate(zip(returns, expected_returns)):
            self.assertAlmostEqual(actual, expected, places=6)

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation"""
        returns = pd.Series([0.01, 0.02, -0.01, 0.015, 0.005])
        risk_free_rate = 0.02

        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        sharpe_ratio = np.sqrt(
            252) * excess_returns.mean() / excess_returns.std()

        # Sharpe ratio should be a finite number
        self.assertFalse(np.isnan(sharpe_ratio))
        self.assertFalse(np.isinf(sharpe_ratio))


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test classes
    test_classes = [
        TestTechnicalIndicators,
        TestFeatureEngineering,
        TestTradingSignal,
        TestConfigManager,
        TestRiskManager,
        TestDatabaseManager,
        TestDataCollector,
        TestMLPredictor,
        TestIntegration,
        TestPerformanceMetrics
    ]

    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)

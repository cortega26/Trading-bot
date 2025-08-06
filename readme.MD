# Advanced Algorithmic Trading Bot

## ⚠️ IMPORTANT DISCLAIMER

**This software is for educational and research purposes only. Trading involves substantial risk of loss and is not suitable for all investors. Past performance does not guarantee future results. Always use paper trading first and never risk more than you can afford to lose.**

## Overview

This is a comprehensive Python-based algorithmic trading system that combines technical analysis, machine learning, and risk management to generate automated trading signals. The system is designed with a modular architecture for easy customization and extension.

## Features

### Core Functionality
- **Multi-source Data Integration**: Real-time and historical data from Yahoo Finance
- **Machine Learning Models**: Random Forest and Gradient Boosting classifiers
- **Technical Analysis**: RSI, MACD, Bollinger Bands, Moving Averages, ATR, and more
- **Risk Management**: Position sizing, stop-losses, portfolio-level risk controls
- **Alert System**: Email notifications for trading signals and position updates
- **Backtesting Engine**: Historical performance evaluation with realistic costs
- **Database Storage**: SQLite database for signal and trade history

### Advanced Features
- **Feature Engineering**: Automated creation of technical and momentum indicators
- **Model Retraining**: Automatic model updates with new market data
- **Paper Trading**: Safe testing environment before live deployment
- **Portfolio Management**: Multi-asset position tracking and P&L calculation
- **Performance Analytics**: Sharpe ratio, maximum drawdown, win rate analysis

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone or download the project files**
   ```bash
   # Create a new directory for the project
   mkdir trading_bot
   cd trading_bot
   ```

2. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install TA-Lib (Technical Analysis Library)**
   
   **For Windows:**
   ```bash
   # Download appropriate .whl file from:
   # https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
   pip install TA_Lib-0.4.25-cp39-cp39-win_amd64.whl  # Adjust for your Python version
   ```
   
   **For macOS:**
   ```bash
   brew install ta-lib
   pip install TA-Lib
   ```
   
   **For Linux:**
   ```bash
   sudo apt-get install ta-lib
   pip install TA-Lib
   ```

4. **Create configuration file**
   ```bash
   python trading_bot.py --mode config
   ```

5. **Customize your configuration**
   Edit the generated `config.yaml` file with your preferred settings.

## Configuration

### Basic Setup

Edit `config.yaml` to customize the bot for your needs:

```yaml
trading:
  max_positions: 5              # Maximum concurrent positions
  max_position_size: 0.15       # 15% max per position
  min_confidence: 0.70          # Minimum ML confidence for trades

data:
  symbols:                      # Stocks to monitor
    - AAPL
    - GOOGL
    - MSFT
  update_frequency: 300         # Check every 5 minutes
```

### Email Alerts Setup

To enable email notifications:

1. **Enable email alerts**
   ```yaml
   alerts:
     email_enabled: true
     email_user: "your_email@gmail.com"
     email_password: "your_app_password"
     email_recipients:
       - "your_email@gmail.com"
   ```

2. **For Gmail users:**
   - Enable 2-factor authentication
   - Generate an App Password (not your regular password)
   - Use the App Password in the configuration

## Usage

### Running the Trading Bot

1. **Paper Trading Mode (Recommended)**
   ```bash
   python trading_bot.py --mode run
   ```
   This runs the bot in paper trading mode, logging all trades without executing them.

2. **View Real-time Logs**
   ```bash
   tail -f trading_bot.log
   ```

### Backtesting

Test your strategy on historical data:

```bash
# Backtest default symbols for 2023
python trading_bot.py --mode backtest --start-date 2023-01-01 --end-date 2024-01-01

# Backtest specific symbols
python trading_bot.py --mode backtest --symbols AAPL MSFT GOOGL --start-date 2022-01-01 --end-date 2023-01-01
```

### Monitoring Performance

The bot generates several outputs:

- **Console Output**: Real-time trading signals and portfolio updates
- **Log File**: Detailed logging in `trading_bot.log`
- **Database**: Signal history in `trading_bot.db`
- **Charts**: Backtest performance charts (when available)

## System Architecture

### Core Components

1. **TradingBot**: Main orchestrator class
2. **DataCollector**: Handles data retrieval from various sources
3. **MLPredictor**: Machine learning model training and prediction
4. **FeatureEngineering**: Creates technical and fundamental features
5. **RiskManager**: Position sizing and risk controls
6. **AlertSystem**: Notifications and alerts
7. **DatabaseManager**: Data persistence
8. **BacktestEngine**: Historical strategy evaluation

### Data Flow

```
Market Data → Feature Engineering → ML Prediction → Risk Assessment → Signal Generation → Position Management
```

## Strategy Logic

### Signal Generation Process

1. **Data Collection**: Retrieve latest market data for monitored symbols
2. **Feature Engineering**: Calculate technical indicators and momentum features
3. **ML Prediction**: Use trained model to predict price direction
4. **Confidence Filtering**: Only act on high-confidence predictions
5. **Risk Assessment**: Check position limits and portfolio risk
6. **Signal Execution**: Generate buy/sell/hold signals with targets and stops

### Risk Management

- **Position Sizing**: Dynamic allocation based on volatility and confidence
- **Stop Losses**: Automatic exit at predetermined loss levels
- **Take Profits**: Lock in gains at target levels
- **Portfolio Limits**: Maximum positions and concentration limits
- **Drawdown Controls**: Monitor overall portfolio performance

## Customization

### Adding New Indicators

To add custom technical indicators:

```python
# In TechnicalIndicators class
@staticmethod
def custom_indicator(data: pd.Series, window: int) -> pd.Series:
    """Your custom indicator logic"""
    return data.rolling(window=window).apply(your_function)

# In FeatureEngineering class
def create_custom_features(self) -> pd.DataFrame:
    """Add your custom features"""
    close = self.data['Close']
    self.features['Custom_Indicator'] = TechnicalIndicators.custom_indicator(close, 20)
    return self.features
```

### Modifying Trading Logic

The main trading logic is in the `generate_signals()` method of the `TradingBot` class. You can modify:

- Signal generation criteria
- Position sizing logic
- Entry/exit conditions
- Risk management rules

## Troubleshooting

### Common Issues

1. **TA-Lib Installation Problems**
   - Windows: Download pre-compiled wheels from UCI
   - macOS/Linux: Ensure ta-lib system library is installed first

2. **Data Retrieval Errors**
   - Check internet connection
   - Verify symbol names are correct
   - Yahoo Finance may temporarily block requests

3. **Memory Issues**
   - Reduce the number of symbols being monitored
   - Decrease the lookback period in configuration
   - Increase update frequency to reduce processing load

4. **Model Training Failures**
   - Ensure sufficient historical data is available
   - Check for NaN values in features
   - Verify target variable generation

### Performance Optimization

- **Reduce Update Frequency**: Set longer intervals between checks
- **Limit Symbols**: Monitor fewer stocks simultaneously
- **Optimize Features**: Use only the most predictive indicators
- **Database Maintenance**: Regularly clean old data

## Security Considerations

### Email Configuration
- Use App Passwords, not regular passwords
- Store credentials in environment variables for production
- Limit email recipients to trusted addresses

### Data Storage
- Database contains trading history and signals
- Consider encryption for sensitive production data
- Regular backups of trading database

## Live Trading Integration

**⚠️ IMPORTANT: This implementation is designed for paper trading and educational purposes.**

For live trading integration, you would need to:

1. **Add Broker API Integration**
   - Interactive Brokers (IB API)
   - Alpaca Trading API
   - TD Ameritrade API
   - Other supported brokers

2. **Implement Order Management**
   - Order placement and tracking
   - Position reconciliation
   - Error handling and recovery

3. **Enhanced Risk Controls**
   - Real-time portfolio monitoring
   - Account balance verification
   - Maximum daily loss limits
   - Circuit breakers for unusual market conditions

4. **Compliance and Regulation**
   - Pattern Day Trading (PDT) rules compliance
   - SEC and FINRA regulations
   - Tax reporting integration
   - Audit trails and record keeping

## Example Usage Scenarios

### Scenario 1: Conservative Long-term Strategy

```yaml
# config.yaml settings for conservative approach
trading:
  max_positions: 3
  max_position_size: 0.1
  stop_loss_pct: 0.03
  take_profit_pct: 0.10
  min_confidence: 0.80

data:
  symbols: [SPY, QQQ, VTI]  # ETFs for diversification
  update_frequency: 3600    # Check hourly
```

### Scenario 2: Aggressive Growth Strategy

```yaml
# config.yaml settings for aggressive approach
trading:
  max_positions: 8
  max_position_size: 0.20
  stop_loss_pct: 0.08
  take_profit_pct: 0.25
  min_confidence: 0.65

data:
  symbols: [TSLA, NVDA, AMZN, GOOGL, META]  # Growth stocks
  update_frequency: 300     # Check every 5 minutes
```

## Performance Metrics

The system tracks and reports various performance metrics:

### Trading Metrics
- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Average Trade Duration**: Typical holding period
- **Profit Factor**: Ratio of gross profits to gross losses

### Risk Metrics
- **Value at Risk (VaR)**: Potential loss at confidence level
- **Beta**: Portfolio sensitivity to market movements
- **Volatility**: Standard deviation of returns
- **Correlation**: Diversification effectiveness

## Advanced Features

### 1. Regime Detection

The system can be extended to detect market regimes:

```python
def detect_market_regime(self, data: pd.DataFrame) -> str:
    """Detect current market regime (bull/bear/sideways)"""
    returns = data['Close'].pct_change().dropna()
    
    # Calculate rolling metrics
    volatility = returns.rolling(20).std().iloc[-1]
    trend = returns.rolling(50).mean().iloc[-1]
    
    if trend > 0.001 and volatility < 0.02:
        return "bull"
    elif trend < -0.001 and volatility > 0.03:
        return "bear"
    else:
        return "sideways"
```

### 2. Sentiment Analysis Integration

Add news sentiment analysis:

```python
# Example integration with news sentiment
def get_sentiment_score(self, symbol: str) -> float:
    """Get sentiment score for a symbol (placeholder)"""
    # In a real implementation, integrate with:
    # - News APIs (Alpha Vantage, Polygon.io)
    # - Social media sentiment (Twitter, Reddit)
    # - Financial news sentiment analysis
    return 0.0  # Neutral sentiment
```

### 3. Options Strategy Integration

Extend for options trading:

```python
class OptionsStrategy:
    """Options trading strategies"""
    
    def covered_call_signal(self, position: Position, options_data: dict) -> bool:
        """Generate covered call opportunities"""
        # Implement covered call logic
        pass
    
    def protective_put_signal(self, position: Position, options_data: dict) -> bool:
        """Generate protective put recommendations"""
        # Implement protective put logic
        pass
```

## Monitoring and Maintenance

### Daily Monitoring Checklist

1. **System Health**
   - [ ] Bot is running without errors
   - [ ] Data feeds are updating correctly
   - [ ] No stuck positions or orders
   - [ ] Database is accessible and current

2. **Performance Review**
   - [ ] Review overnight signals and trades
   - [ ] Check position P&L and risk exposure
   - [ ] Validate model predictions vs. actual moves
   - [ ] Monitor overall portfolio performance

3. **Risk Assessment**
   - [ ] Verify position sizes are within limits
   - [ ] Check correlation between positions
   - [ ] Review stop-loss and take-profit levels
   - [ ] Assess overall portfolio risk metrics

### Weekly Maintenance

1. **Model Performance**
   - Review prediction accuracy
   - Retrain models if performance degrades
   - Update feature engineering if needed
   - Backtest recent performance

2. **Strategy Optimization**
   - Analyze winning vs. losing trades
   - Adjust confidence thresholds if needed
   - Review and update risk parameters
   - Test strategy modifications

3. **System Updates**
   - Update market data and dependencies
   - Review and clean log files
   - Backup trading database
   - Update documentation

## Integration Examples

### 1. Webhook Integration

```python
from flask import Flask, request
import json

app = Flask(__name__)

@app.route('/webhook/signal', methods=['POST'])
def receive_signal():
    """Receive external trading signals via webhook"""
    data = request.get_json()
    
    # Process external signal
    external_signal = TradingSignal(
        symbol=data['symbol'],
        signal=data['action'],
        confidence=data['confidence'],
        timestamp=datetime.now(),
        current_price=data['price']
    )
    
    # Add to bot's signal queue
    trading_bot.process_external_signal(external_signal)
    
    return {"status": "received"}
```

### 2. Discord/Slack Notifications

```python
import requests

class DiscordNotifier:
    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url
    
    def send_signal_alert(self, signal: TradingSignal):
        """Send trading signal to Discord channel"""
        embed = {
            "title": f"Trading Signal: {signal.signal} {signal.symbol}",
            "color": 0x00ff00 if signal.signal == "BUY" else 0xff0000,
            "fields": [
                {"name": "Price", "value": f"${signal.current_price:.2f}", "inline": True},
                {"name": "Confidence", "value": f"{signal.confidence:.1%}", "inline": True},
                {"name": "Target", "value": f"${signal.target_price:.2f}", "inline": True}
            ]
        }
        
        requests.post(self.webhook_url, json={"embeds": [embed]})
```

### 3. Portfolio Dashboard

```python
import streamlit as st
import plotly.express as px

def create_dashboard():
    """Create a Streamlit dashboard for monitoring"""
    st.title("Trading Bot Dashboard")
    
    # Load recent performance data
    bot = TradingBot()
    performance = bot.get_performance_report()
    
    # Display key metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Positions", performance['active_positions'])
    with col2:
        st.metric("Unrealized P&L", f"${performance['unrealized_pnl']:.2f}")
    with col3:
        st.metric("Portfolio Value", f"${performance['portfolio_value']:,.2f}")
    
    # Show recent signals
    recent_signals = bot.db_manager.get_recent_signals(7)
    if not recent_signals.empty:
        st.subheader("Recent Signals")
        st.dataframe(recent_signals)

if __name__ == "__main__":
    create_dashboard()
```

## Testing and Validation

### Unit Testing

```python
import unittest
from trading_bot import TechnicalIndicators, FeatureEngineering

class TestTechnicalIndicators(unittest.TestCase):
    
    def setUp(self):
        # Create sample data
        self.sample_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    
    def test_sma_calculation(self):
        """Test Simple Moving Average calculation"""
        sma = TechnicalIndicators.sma(self.sample_data, 3)
        expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        pd.testing.assert_series_equal(sma, expected)
    
    def test_rsi_calculation(self):
        """Test RSI calculation bounds"""
        rsi = TechnicalIndicators.rsi(self.sample_data, 3)
        # RSI should be between 0 and 100
        self.assertTrue((rsi.dropna() >= 0).all())
        self.assertTrue((rsi.dropna() <= 100).all())

# Run tests
if __name__ == '__main__':
    unittest.main()
```

### Integration Testing

```python
def test_end_to_end_signal_generation():
    """Test complete signal generation process"""
    # Initialize bot
    bot = TradingBot('test_config.yaml')
    bot.initialize()
    
    # Generate signals
    signals = bot.generate_signals()
    
    # Validate signals
    for signal in signals:
        assert signal.symbol in bot.config['data']['symbols']
        assert signal.signal in ['BUY', 'SELL', 'HOLD']
        assert 0 <= signal.confidence <= 1
        assert signal.current_price > 0
```

## Deployment Options

### 1. Local Development
- Run on personal computer
- Manual monitoring and intervention
- Suitable for testing and small-scale trading

### 2. Cloud Deployment (AWS/Google Cloud/Azure)
- 24/7 operation
- Scalable resources
- Professional monitoring and alerting
- Automated backups and recovery

### 3. VPS Deployment
- Dedicated server resources
- Cost-effective for individual traders
- Full control over environment
- Suitable for production trading

## Legal and Compliance

### Important Considerations

1. **Regulatory Compliance**
   - Ensure compliance with local securities regulations
   - Understand Pattern Day Trading (PDT) rules
   - Consider registration requirements for automated trading

2. **Risk Disclosure**
   - Algorithmic trading involves significant risks
   - Past performance does not guarantee future results
   - Always test strategies thoroughly before live deployment
   - Never risk more capital than you can afford to lose

3. **Data Usage**
   - Respect data provider terms of service
   - Ensure proper licensing for commercial use
   - Consider rate limiting and fair usage policies

## Support and Community

### Getting Help

1. **Documentation**: This README and inline code comments
2. **Logs**: Check `trading_bot.log` for detailed error information
3. **Database**: Query `trading_bot.db` for historical data analysis
4. **Configuration**: Verify `config.yaml` settings

### Contributing

To contribute improvements or bug fixes:

1. Test your changes thoroughly
2. Include appropriate documentation
3. Follow Python PEP 8 style guidelines
4. Add unit tests for new functionality

## Changelog

### Version 1.0.0
- Initial release with core trading functionality
- Machine learning signal generation
- Risk management system
- Backtesting engine
- Email alerting system
- SQLite database integration

## License

This project is for educational purposes only. Users are responsible for compliance with applicable laws and regulations in their jurisdiction.

---

**Remember: This is a sophisticated trading system that requires careful testing and validation before any live deployment. Always start with paper trading and gradually increase position sizes as you gain confidence in the system's performance.**

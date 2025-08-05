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
   - 

# Crypto Scalping Bot

A Python-based cryptocurrency scalping bot that monitors the top 100 cryptocurrencies on 5-minute and 15-minute timeframes, generates precise entry and exit signals, and sends real-time notifications via Telegram.

## Features

- **Multi-Timeframe Analysis**: Monitors 5-minute and 15-minute candles for short-term scalping opportunities
- **Advanced Technical Indicators**: Uses RSI, EMA, MACD, Bollinger Bands, Stochastic Oscillator, ATR, and Supertrend
- **Signal Confirmation**: Requires confirmation across multiple indicators and timeframes
- **Precise Entry/Exit Points**: Provides exact entry price, stop loss, and take profit levels
- **Risk Management**: Enforces minimum 1.5:1 reward-to-risk ratio for all signals
- **Real-time Notifications**: Sends detailed signal alerts via Telegram
- **Web Service**: Runs as a Flask web service for easy deployment and monitoring
- **Configurable Parameters**: Easily adjust indicator settings, profit targets, and stop loss levels

## Setup and Installation

### Prerequisites

- Python 3.9+
- Telegram Bot (create one via [@BotFather](https://t.me/botfather))
- Binance account (optional, for enhanced data access)

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd crypto-scalping-bot
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file from the example:
   ```
   cp .env.example .env
   ```

4. Edit the `.env` file with your Telegram bot token and chat ID:
   ```
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
   TELEGRAM_CHAT_ID=your_telegram_chat_id_here
   ```

### Running Locally

Run the bot with:
```
python crypto_scalping_bot.py
```

The bot will start monitoring cryptocurrencies and send signals to your Telegram chat when opportunities are detected.

## Deployment

The bot is designed to be deployed as a web service on platforms like Render.com:

1. Connect your repository to Render
2. Select "Web Service"
3. Set the build command: `pip install -r requirements.txt`
4. Set the start command: `gunicorn crypto_scalping_bot:app`
5. Add your environment variables (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, etc.)

## API Endpoints

- `/`: Home endpoint, shows bot status
- `/health`: Health check endpoint for monitoring
- `/status`: Shows detailed bot status including monitored coins and active signals
- `/send_test_message`: Sends a test message to Telegram to verify connectivity
- `/start_bot`: Manually starts the bot if it's not running

## Configuration

The bot's behavior can be customized by modifying the `ScalpingBotConfig` class in the main script:

- `TIMEFRAMES`: Trading timeframes to monitor
- `TOP_COINS_COUNT`: Number of top cryptocurrencies to track
- `UPDATE_INTERVAL`: How often to check for new signals (in seconds)
- `PROFIT_TARGET_PERCENT`: Target profit percentage
- `STOP_LOSS_PERCENT`: Stop loss percentage
- Technical indicator parameters (RSI, EMA periods, etc.)

## Signal Strategy

The bot generates signals based on a combination of:

1. RSI conditions (oversold/overbought)
2. EMA crossovers and alignments
3. MACD crossovers
4. Bollinger Band touches
5. Stochastic oscillator conditions
6. Supertrend direction changes
7. Volume increases

Signals require at least 3 of these conditions to align, plus confirmation from a higher timeframe.

## Disclaimer

This bot is for educational and informational purposes only. Use at your own risk. Cryptocurrency trading involves significant risk and past performance is not indicative of future results.

## License

[MIT License](LICENSE)

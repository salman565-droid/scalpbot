#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Crypto Scalping Bot
-----------------
A bot that monitors top 100 cryptocurrencies on 5m and 15m timeframes and generates
precise entry and exit signals for scalping opportunities. Uses advanced technical
indicators and sends detailed notifications via Telegram.
"""

import os
import time
import logging
import json
import asyncio
import datetime
import threading
from typing import Dict, List, Tuple, Optional, Union

# Third-party imports
import ccxt
import pandas as pd
import numpy as np
import telegram
from telegram.ext import Application, CommandHandler
import requests
from dotenv import load_dotenv
import pandas_ta as ta
from tabulate import tabulate

# Web service imports
from flask import Flask, jsonify

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("crypto_scalping_bot.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app for web service
app = Flask(__name__)

# Global bot instance
bot_instance = None

# Configuration
class Config:
    # Telegram settings
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # Exchange API settings
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
    HTTP_PROXY = os.getenv('HTTP_PROXY')  # For bypassing geo-restrictions
    
    # Timeframes to monitor
    TIMEFRAMES = ['5m', '15m']
    
    # General settings
    TOP_COINS_COUNT = 100
    UPDATE_INTERVAL = 60  # 1 minute
    
    # Scalping settings
    PROFIT_TARGET_PERCENT = 1.0  # 1% profit target
    STOP_LOSS_PERCENT = 0.5      # 0.5% stop loss
    MAX_TRADE_DURATION = 60      # 60 minutes max trade duration
    
    # Technical indicator settings
    RSI_PERIOD = 14
    RSI_OVERBOUGHT = 70
    RSI_OVERSOLD = 30
    
    # EMA settings
    EMA_FAST = 8
    EMA_MEDIUM = 13
    EMA_SLOW = 21
    
    # MACD settings
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    
    # Bollinger Bands settings
    BB_PERIOD = 20
    BB_STD = 2.0
    
    # Volume settings
    VOLUME_INCREASE_THRESHOLD = 1.5  # 150% volume increase
    
    # Stochastic settings
    STOCH_K = 14
    STOCH_D = 3
    STOCH_OVERBOUGHT = 80
    STOCH_OVERSOLD = 20
    
    # ATR settings for volatility
    ATR_PERIOD = 14
    
    # Supertrend settings
    SUPERTREND_PERIOD = 10
    SUPERTREND_MULTIPLIER = 3.0


class CryptoScalpingBot:
    def __init__(self):
        """Initialize the Crypto Scalping Bot."""
        self.config = Config()
        self.exchange = self._initialize_exchange()
        self.telegram_bot = self._initialize_telegram_bot()
        self.top_coins = []
        self.market_data = {}
        self.active_trades = []
        self.signals = []
        self.last_update = datetime.datetime.now()
        
    def _initialize_exchange(self):
        """Initialize the exchange connection with fallbacks for geo-restrictions."""
        exchanges_to_try = [
            # First try Binance with proxy if configured
            ('binance', self._get_binance_config()),
            # Fallback to Binance.US if in the US
            ('binanceus', self._get_binance_us_config()),
            # Additional fallbacks
            ('kucoin', {'enableRateLimit': True}),
            ('gate', {'enableRateLimit': True}),
            ('huobi', {'enableRateLimit': True}),
            ('okx', {'enableRateLimit': True})
        ]
        
        last_error = None
        for exchange_id, config in exchanges_to_try:
            try:
                logger.info(f"Attempting to connect to {exchange_id}...")
                exchange_class = getattr(ccxt, exchange_id)
                exchange = exchange_class(config)
                
                # Test connection
                exchange.load_markets()
                logger.info(f"Successfully connected to {exchange_id}")
                self.exchange_id = exchange_id  # Store which exchange we're using
                return exchange
            
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to connect to {exchange_id}: {e}")
        
        # If we get here, all exchanges failed
        logger.error(f"Failed to initialize any exchange. Last error: {last_error}")
        raise last_error
    
    def _get_binance_config(self):
        """Get Binance configuration with optional proxy settings."""
        config = {'enableRateLimit': True}
        
        # Add API keys if available
        if self.config.BINANCE_API_KEY and self.config.BINANCE_API_SECRET:
            config['apiKey'] = self.config.BINANCE_API_KEY
            config['secret'] = self.config.BINANCE_API_SECRET
        
        # Add proxy if configured
        if self.config.HTTP_PROXY:
            config['proxies'] = {
                'http': self.config.HTTP_PROXY,
                'https': self.config.HTTP_PROXY
            }
            logger.info(f"Using proxy for Binance: {self.config.HTTP_PROXY}")
        
        return config
    
    def _get_binance_us_config(self):
        """Get Binance US configuration."""
        config = {'enableRateLimit': True}
        
        # Use the same API keys for Binance US if available
        if self.config.BINANCE_API_KEY and self.config.BINANCE_API_SECRET:
            config['apiKey'] = self.config.BINANCE_API_KEY
            config['secret'] = self.config.BINANCE_API_SECRET
        
        return config
            
    def _initialize_telegram_bot(self):
        """Initialize the Telegram bot."""
        try:
            # Check if token is available
            if not self.config.TELEGRAM_BOT_TOKEN:
                logger.warning("Telegram bot token not found. Using dummy bot.")
                return None
                
            bot = telegram.Bot(token=self.config.TELEGRAM_BOT_TOKEN)
            logger.info("Telegram bot initialized successfully")
            return bot
        except Exception as e:
            logger.error(f"Failed to initialize Telegram bot: {e}")
            logger.warning("Using dummy bot instead")
            return None
            
    async def send_telegram_message(self, message: str):
        """Send a message to the Telegram channel."""
        try:
            # Check if bot and chat ID are available
            if not self.telegram_bot or not self.config.TELEGRAM_CHAT_ID:
                logger.warning(f"Telegram message not sent (bot not configured): {message[:50]}...")
                return
            
            # Add extra logging to debug Telegram messaging
            logger.info(f"Attempting to send Telegram message to chat ID: {self.config.TELEGRAM_CHAT_ID}")
                
            await self.telegram_bot.send_message(
                chat_id=self.config.TELEGRAM_CHAT_ID,
                text=message,
                parse_mode=telegram.constants.ParseMode.MARKDOWN
            )
            logger.info(f"Message sent to Telegram: {message[:50]}...")
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            logger.error(f"Telegram bot: {self.telegram_bot}, Chat ID: {self.config.TELEGRAM_CHAT_ID}")
    
    async def fetch_top_coins(self):
        """Fetch top coins by volume, with exchange-specific handling."""
        try:
            logger.info(f"Fetching top coins from {self.exchange_id}")
            
            # Get all markets
            markets = self.exchange.fetch_markets()
            
            # Filter for USDT pairs (handling different exchange formats)
            usdt_markets = []
            for market in markets:
                # Different exchanges have different market structures
                if 'quote' in market and market['quote'] == 'USDT':
                    usdt_markets.append(market)
                elif '/USDT' in market.get('symbol', ''):
                    usdt_markets.append(market)
                elif 'USDT' in market.get('id', '') and 'symbol' in market:
                    usdt_markets.append(market)
            
            # Get volume data
            tickers = self.exchange.fetch_tickers()
            market_data = []
            
            for market in usdt_markets:
                symbol = market['symbol']
                if symbol not in tickers:
                    continue
                    
                # Different exchanges report volume differently
                volume = 0
                ticker = tickers[symbol]
                
                if 'quoteVolume' in ticker and ticker['quoteVolume']:
                    volume = ticker['quoteVolume']
                elif 'baseVolume' in ticker and ticker['baseVolume'] and 'last' in ticker and ticker['last']:
                    # Estimate quote volume from base volume and price
                    volume = ticker['baseVolume'] * ticker['last']
                
                if volume > 0:
                    market_data.append({'symbol': symbol, 'volume': volume})
            
            # Sort by volume and take top N
            market_data.sort(key=lambda x: x['volume'], reverse=True)
            top_markets = market_data[:self.config.TOP_COINS_COUNT]
            self.top_coins = [market['symbol'] for market in top_markets]
            
            logger.info(f"Found {len(self.top_coins)} top coins on {self.exchange_id}")
            return self.top_coins
            
        except Exception as e:
            logger.error(f"Error fetching top coins: {e}")
            # Return previously fetched coins if available, otherwise return a default list
            if not self.top_coins:
                self.top_coins = [f"BTC/USDT", f"ETH/USDT", f"BNB/USDT", f"SOL/USDT", f"XRP/USDT"]
                logger.info("Using default coin list due to error")
            return self.top_coins
    
    async def fetch_market_data(self):
        """Fetch market data for the top coins on multiple timeframes."""
        self.market_data = {}
        
        for symbol in self.top_coins:
            self.market_data[symbol] = {}
            
            for timeframe in self.config.TIMEFRAMES:
                try:
                    # Fetch OHLCV data with retry mechanism
                    max_retries = 3
                    retry_delay = 2  # seconds
                    
                    for attempt in range(max_retries):
                        try:
                            # Some exchanges have different parameter names or limits
                            if hasattr(self, 'exchange_id') and self.exchange_id in ['kucoin', 'gate']:
                                # These exchanges might have different parameter requirements
                                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
                            else:
                                # Standard approach for most exchanges
                                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=100)
                            
                            # If we got here, the request was successful
                            break
                        except Exception as retry_error:
                            if attempt < max_retries - 1:
                                logger.warning(f"Attempt {attempt+1}/{max_retries} failed for {symbol} on {timeframe}: {retry_error}")
                                await asyncio.sleep(retry_delay)
                                retry_delay *= 2  # Exponential backoff
                            else:
                                # Last attempt failed, re-raise the exception
                                raise
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    
                    # Calculate volume change (maintaining compatibility with existing code)
                    df['volume_change'] = df['volume'] / df['volume'].shift(1)
                    
                    # Store in market data dictionary with the original structure
                    self.market_data[symbol][timeframe] = df
                    
                except Exception as e:
                    logger.error(f"Failed to fetch market data for {symbol} on {timeframe}: {e}")
            
            if symbol in self.market_data and len(self.market_data[symbol]) > 0:
                logger.info(f"Fetched market data for {symbol} on {len(self.market_data[symbol])} timeframes")
        
        logger.info(f"Completed market data fetch for {len(self.market_data)} symbols")
    
        
    def calculate_indicators(self):
        """Calculate technical indicators for each symbol and timeframe."""
        for symbol, timeframes in self.market_data.items():
            for timeframe, df in timeframes.items():
                try:
                    # Calculate RSI
                    df.ta.rsi(close='close', length=self.config.RSI_PERIOD, append=True)
                    
                    # Calculate EMAs
                    df.ta.ema(close='close', length=self.config.EMA_FAST, append=True)
                    df.ta.ema(close='close', length=self.config.EMA_MEDIUM, append=True)
                    df.ta.ema(close='close', length=self.config.EMA_SLOW, append=True)
                    
                    # Rename EMA columns to match our expected format
                    df.rename(columns={
                        f'EMA_{self.config.EMA_FAST}': 'ema_fast',
                        f'EMA_{self.config.EMA_MEDIUM}': 'ema_medium',
                        f'EMA_{self.config.EMA_SLOW}': 'ema_slow',
                        'RSI_14': 'rsi'
                    }, inplace=True)
                    
                    # Calculate MACD
                    macd = df.ta.macd(
                        close='close', 
                        fast=self.config.MACD_FAST, 
                        slow=self.config.MACD_SLOW, 
                        signal=self.config.MACD_SIGNAL
                    )
                    df = df.join(macd)
                    df.rename(columns={
                        f'MACD_{self.config.MACD_FAST}_{self.config.MACD_SLOW}_{self.config.MACD_SIGNAL}': 'macd',
                        f'MACDs_{self.config.MACD_FAST}_{self.config.MACD_SLOW}_{self.config.MACD_SIGNAL}': 'macd_signal',
                        f'MACDh_{self.config.MACD_FAST}_{self.config.MACD_SLOW}_{self.config.MACD_SIGNAL}': 'macd_hist'
                    }, inplace=True)
                    
                    # Calculate Bollinger Bands
                    bbands = df.ta.bbands(
                        close='close', 
                        length=self.config.BB_PERIOD, 
                        std=self.config.BB_STD
                    )
                    df = df.join(bbands)
                    df.rename(columns={
                        f'BBL_{self.config.BB_PERIOD}_{self.config.BB_STD}': 'bb_lower',
                        f'BBM_{self.config.BB_PERIOD}_{self.config.BB_STD}': 'bb_middle',
                        f'BBU_{self.config.BB_PERIOD}_{self.config.BB_STD}': 'bb_upper',
                        f'BBB_{self.config.BB_PERIOD}_{self.config.BB_STD}': 'bb_bandwidth'
                    }, inplace=True)
                    
                    # Calculate Stochastic Oscillator
                    stoch = df.ta.stoch(
                        high='high',
                        low='low',
                        close='close',
                        k=self.config.STOCH_K,
                        d=self.config.STOCH_D
                    )
                    df = df.join(stoch)
                    df.rename(columns={
                        f'STOCHk_{self.config.STOCH_K}_{self.config.STOCH_D}_3': 'stoch_k',
                        f'STOCHd_{self.config.STOCH_K}_{self.config.STOCH_D}_3': 'stoch_d'
                    }, inplace=True)
                    
                    # Calculate ATR for volatility
                    atr = df.ta.atr(high='high', low='low', close='close', length=self.config.ATR_PERIOD)
                    df = df.join(atr)
                    df.rename(columns={f'ATR_{self.config.ATR_PERIOD}': 'atr'}, inplace=True)
                    
                    # Calculate Supertrend
                    supertrend = df.ta.supertrend(
                        high='high',
                        low='low',
                        close='close',
                        length=self.config.SUPERTREND_PERIOD,
                        multiplier=self.config.SUPERTREND_MULTIPLIER
                    )
                    df = df.join(supertrend)
                    df.rename(columns={
                        f'SUPERT_{self.config.SUPERTREND_PERIOD}_{self.config.SUPERTREND_MULTIPLIER}': 'supertrend',
                        f'SUPERTd_{self.config.SUPERTREND_PERIOD}_{self.config.SUPERTREND_MULTIPLIER}': 'supertrend_direction'
                    }, inplace=True)
                    
                    # Update the market data
                    self.market_data[symbol][timeframe] = df
                    
                    logger.info(f"Calculated indicators for {symbol} on {timeframe} timeframe")
                
                except Exception as e:
                    logger.error(f"Failed to calculate indicators for {symbol} on {timeframe}: {e}")
                    logger.exception(e)
    
    def generate_scalping_signals(self):
        """Generate scalping signals with precise entry and exit points."""
        self.signals = []
        
        for symbol, timeframes in self.market_data.items():
            try:
                # We'll use both timeframes for confirmation
                if len(timeframes) < 2 or not all(tf in timeframes for tf in self.config.TIMEFRAMES):
                    continue
                
                # Get the latest data points for both timeframes
                df_5m = timeframes['5m']
                df_15m = timeframes['15m']
                
                if len(df_5m) < 5 or len(df_15m) < 5:
                    continue
                
                # Get the latest candles
                latest_5m = df_5m.iloc[-1]
                prev_5m = df_5m.iloc[-2]
                latest_15m = df_15m.iloc[-1]
                prev_15m = df_15m.iloc[-2]
                
                # Initialize signal variables
                signal_type = None
                entry_price = None
                stop_loss = None
                take_profit = None
                timeframe = None
                confidence = 0
                reasons = []
                
                # Check for entry signals on 5m timeframe
                if self._check_entry_signal(df_5m, 'BUY'):
                    signal_type = 'BUY'
                    entry_price = latest_5m['close']
                    stop_loss = entry_price * (1 - self.config.STOP_LOSS_PERCENT / 100)
                    take_profit = entry_price * (1 + self.config.PROFIT_TARGET_PERCENT / 100)
                    timeframe = '5m'
                    confidence += 1
                    reasons.append(self._get_signal_reasons(df_5m, 'BUY'))
                    
                    # Check for confirmation on 15m timeframe
                    if self._check_entry_confirmation(df_15m, 'BUY'):
                        confidence += 1
                        reasons.append("Confirmed by 15m timeframe")
                
                elif self._check_entry_signal(df_5m, 'SELL'):
                    signal_type = 'SELL'
                    entry_price = latest_5m['close']
                    stop_loss = entry_price * (1 + self.config.STOP_LOSS_PERCENT / 100)
                    take_profit = entry_price * (1 - self.config.PROFIT_TARGET_PERCENT / 100)
                    timeframe = '5m'
                    confidence += 1
                    reasons.append(self._get_signal_reasons(df_5m, 'SELL'))
                    
                    # Check for confirmation on 15m timeframe
                    if self._check_entry_confirmation(df_15m, 'SELL'):
                        confidence += 1
                        reasons.append("Confirmed by 15m timeframe")
                
                # Check for entry signals on 15m timeframe if no signal on 5m
                elif self._check_entry_signal(df_15m, 'BUY'):
                    signal_type = 'BUY'
                    entry_price = latest_15m['close']
                    stop_loss = entry_price * (1 - self.config.STOP_LOSS_PERCENT / 100)
                    take_profit = entry_price * (1 + self.config.PROFIT_TARGET_PERCENT / 100)
                    timeframe = '15m'
                    confidence += 1
                    reasons.append(self._get_signal_reasons(df_15m, 'BUY'))
                
                elif self._check_entry_signal(df_15m, 'SELL'):
                    signal_type = 'SELL'
                    entry_price = latest_15m['close']
                    stop_loss = entry_price * (1 + self.config.STOP_LOSS_PERCENT / 100)
                    take_profit = entry_price * (1 - self.config.PROFIT_TARGET_PERCENT / 100)
                    timeframe = '15m'
                    confidence += 1
                    reasons.append(self._get_signal_reasons(df_15m, 'SELL'))
                
                # Add signal if confidence is high enough
                if signal_type and confidence > 0:
                    # Calculate risk-reward ratio
                    if signal_type == 'BUY':
                        risk = entry_price - stop_loss
                        reward = take_profit - entry_price
                    else:  # SELL
                        risk = stop_loss - entry_price
                        reward = entry_price - take_profit
                    
                    risk_reward_ratio = reward / risk if risk > 0 else 0
                    
                    # Only add signals with good risk-reward ratio
                    if risk_reward_ratio >= 1.5:
                        self.signals.append({
                            'symbol': symbol,
                            'signal_type': signal_type,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'timeframe': timeframe,
                            'confidence': confidence,
                            'risk_reward_ratio': risk_reward_ratio,
                            'reasons': reasons,
                            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })
                        
                        logger.info(f"Generated {signal_type} signal for {symbol} on {timeframe} timeframe")
            
            except Exception as e:
                logger.error(f"Failed to generate signals for {symbol}: {e}")
        
        logger.info(f"Generated {len(self.signals)} scalping signals")
        return self.signals
    
    def _check_entry_signal(self, df, direction):
        """Check for entry signals based on multiple indicators."""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # For BUY signals
        if direction == 'BUY':
            # RSI oversold and turning up
            rsi_signal = latest['rsi'] < self.config.RSI_OVERSOLD and latest['rsi'] > prev['rsi']
            
            # EMA alignment (fast above medium)
            ema_signal = latest['ema_fast'] > latest['ema_medium']
            
            # MACD crossing up
            macd_signal = latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']
            
            # Price near lower Bollinger Band
            bb_signal = latest['close'] < latest['bb_lower'] * 1.01
            
            # Stochastic oversold and crossing up
            stoch_signal = latest['stoch_k'] < self.config.STOCH_OVERSOLD and latest['stoch_k'] > latest['stoch_d']
            
            # Supertrend turning bullish
            supertrend_signal = latest['supertrend_direction'] == 1 and prev['supertrend_direction'] <= 0
            
            # Volume increasing
            volume_signal = latest['volume_change'] > self.config.VOLUME_INCREASE_THRESHOLD
            
            # Return True if enough signals align
            signals_count = sum([rsi_signal, ema_signal, macd_signal, bb_signal, stoch_signal, supertrend_signal, volume_signal])
            return signals_count >= 3
        
        # For SELL signals
        elif direction == 'SELL':
            # RSI overbought and turning down
            rsi_signal = latest['rsi'] > self.config.RSI_OVERBOUGHT and latest['rsi'] < prev['rsi']
            
            # EMA alignment (fast below medium)
            ema_signal = latest['ema_fast'] < latest['ema_medium']
            
            # MACD crossing down
            macd_signal = latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal']
            
            # Price near upper Bollinger Band
            bb_signal = latest['close'] > latest['bb_upper'] * 0.99
            
            # Stochastic overbought and crossing down
            stoch_signal = latest['stoch_k'] > self.config.STOCH_OVERBOUGHT and latest['stoch_k'] < latest['stoch_d']
            
            # Supertrend turning bearish
            supertrend_signal = latest['supertrend_direction'] == -1 and prev['supertrend_direction'] >= 0
            
            # Volume increasing
            volume_signal = latest['volume_change'] > self.config.VOLUME_INCREASE_THRESHOLD
            
            # Return True if enough signals align
            signals_count = sum([rsi_signal, ema_signal, macd_signal, bb_signal, stoch_signal, supertrend_signal, volume_signal])
            return signals_count >= 3
        
        return False
    
    def _check_entry_confirmation(self, df, direction):
        """Check for confirmation signals on a different timeframe."""
        latest = df.iloc[-1]
        
        if direction == 'BUY':
            # Trend is up on higher timeframe
            return (latest['ema_fast'] > latest['ema_slow'] and 
                    latest['supertrend_direction'] == 1 and 
                    latest['close'] > latest['ema_medium'])
        
        elif direction == 'SELL':
            # Trend is down on higher timeframe
            return (latest['ema_fast'] < latest['ema_slow'] and 
                    latest['supertrend_direction'] == -1 and 
                    latest['close'] < latest['ema_medium'])
        
        return False
    
    def _get_signal_reasons(self, df, direction):
        """Get detailed reasons for the signal."""
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        reasons = []
        
        if direction == 'BUY':
            if latest['rsi'] < self.config.RSI_OVERSOLD:
                reasons.append(f"RSI oversold ({latest['rsi']:.2f})")
            
            if latest['ema_fast'] > latest['ema_medium'] and prev['ema_fast'] <= prev['ema_medium']:
                reasons.append("EMA fast crossed above medium")
            
            if latest['macd'] > latest['macd_signal'] and prev['macd'] <= prev['macd_signal']:
                reasons.append("MACD crossed above signal line")
            
            if latest['close'] < latest['bb_lower']:
                reasons.append("Price below lower Bollinger Band")
            
            if latest['stoch_k'] < self.config.STOCH_OVERSOLD and latest['stoch_k'] > latest['stoch_d']:
                reasons.append("Stochastic oversold and turning up")
            
            if latest['supertrend_direction'] == 1 and prev['supertrend_direction'] <= 0:
                reasons.append("Supertrend turned bullish")
        
        elif direction == 'SELL':
            if latest['rsi'] > self.config.RSI_OVERBOUGHT:
                reasons.append(f"RSI overbought ({latest['rsi']:.2f})")
            
            if latest['ema_fast'] < latest['ema_medium'] and prev['ema_fast'] >= prev['ema_medium']:
                reasons.append("EMA fast crossed below medium")
            
            if latest['macd'] < latest['macd_signal'] and prev['macd'] >= prev['macd_signal']:
                reasons.append("MACD crossed below signal line")
            
            if latest['close'] > latest['bb_upper']:
                reasons.append("Price above upper Bollinger Band")
            
            if latest['stoch_k'] > self.config.STOCH_OVERBOUGHT and latest['stoch_k'] < latest['stoch_d']:
                reasons.append("Stochastic overbought and turning down")
            
            if latest['supertrend_direction'] == -1 and prev['supertrend_direction'] >= 0:
                reasons.append("Supertrend turned bearish")
        
        return ", ".join(reasons)
    
    def format_signal_message(self, signal):
        """Format a signal for Telegram message with precise entry and exit points."""
        emoji = "🟢" if signal['signal_type'] == "BUY" else "🔴"
        confidence_stars = "⭐" * signal['confidence']
        
        message = (
            f"{emoji} *{signal['signal_type']} {signal['symbol']}* {confidence_stars}\n\n"
            f"*Entry Price:* ${signal['entry_price']:.4f}\n"
            f"*Stop Loss:* ${signal['stop_loss']:.4f} ({self.config.STOP_LOSS_PERCENT}%)\n"
            f"*Take Profit:* ${signal['take_profit']:.4f} ({self.config.PROFIT_TARGET_PERCENT}%)\n"
            f"*Risk/Reward:* {signal['risk_reward_ratio']:.2f}\n"
            f"*Timeframe:* {signal['timeframe']}\n"
            f"*Time:* {signal['timestamp']}\n\n"
            f"*Signal Reasons:*\n"
        )
        
        for reason in signal['reasons']:
            message += f"• {reason}\n"
        
        message += "\n*Trade Management:*\n"
        message += "• Enter at or near the specified entry price\n"
        message += "• Set stop loss immediately after entry\n"
        message += "• Take profit at target or use trailing stop\n"
        message += "• Consider scaling out at 0.5%, 0.75% profit levels\n"
        
        return message
    
    async def send_signals(self):
        """Send generated signals to Telegram."""
        if not self.signals:
            logger.info("No signals generated in this update.")
            return
        
        for signal in self.signals:
            message = self.format_signal_message(signal)
            await self.send_telegram_message(message)
            # Avoid flooding Telegram
            await asyncio.sleep(1)
    
    async def start(self):
        """Start the scalping bot."""
        logger.info("Starting Crypto Scalping Bot")
        
        # Send startup message
        startup_message = (
            "🚀 *Crypto Scalping Bot Started* 🚀\n\n"
            f"Monitoring top {self.config.TOP_COINS_COUNT} cryptocurrencies\n"
            f"Timeframes: {', '.join(self.config.TIMEFRAMES)}\n"
            f"Update interval: {self.config.UPDATE_INTERVAL} seconds\n\n"
            "Waiting for scalping opportunities...\n\n"
            "*Strategy:*\n"
            "• Entry signals based on multiple indicator alignment\n"
            "• Confirmation from multiple timeframes\n"
            "• Precise entry, stop loss and take profit levels\n"
            "• Risk management with minimum 1.5:1 reward-to-risk ratio\n"
        )
        await self.send_telegram_message(startup_message)
        
        # Main loop
        while True:
            try:
                await self.run_update_cycle()
                logger.info(f"Sleeping for {self.config.UPDATE_INTERVAL} seconds")
                await asyncio.sleep(self.config.UPDATE_INTERVAL)
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(60)  # Sleep for a minute before retrying
    
    async def run_update_cycle(self):
        """Run a complete update cycle."""
        logger.info("Starting update cycle")
        
        await self.fetch_top_coins()
        await self.fetch_market_data()
        self.calculate_indicators()
        self.generate_scalping_signals()
        await self.send_signals()
        
        self.last_update = datetime.datetime.now()
        logger.info("Update cycle completed")


# Flask routes for web service
@app.route('/')
def home():
    """Home route to check if the service is running."""
    global bot_instance
    # If bot is not running, send a startup message
    if bot_instance is None:
        # Start the bot in the background
        start_background_thread()
    
    return jsonify({
        "status": "running",
        "message": "Crypto Scalping Bot is running",
        "timestamp": datetime.datetime.now().isoformat()
    })


@app.route('/health')
def health():
    """Health check endpoint for Render."""
    return jsonify({"status": "healthy"})


@app.route('/status')
def status():
    """Status endpoint to check if the bot is running."""
    global bot_instance
    
    if bot_instance is None:
        return jsonify({
            "status": "error",
            "message": "Bot not initialized"
        })
    
    return jsonify({
        "status": "success",
        "message": "Bot is running",
        "last_update": bot_instance.last_update.strftime("%Y-%m-%d %H:%M:%S"),
        "monitored_coins": len(bot_instance.top_coins),
        "signals_generated": len(bot_instance.signals),
        "exchange": getattr(bot_instance, 'exchange_id', 'unknown')
    })

@app.route('/diagnostic')
def diagnostic():
    """Diagnostic endpoint to check exchange connectivity."""
    global bot_instance
    
    if bot_instance is None:
        return jsonify({
            "status": "error",
            "message": "Bot not initialized"
        })
    
    # Get information about the environment
    try:
        import socket
        import requests
        import sys
        
        # Test basic internet connectivity
        internet_working = False
        try:
            response = requests.get('https://www.google.com', timeout=5)
            internet_working = response.status_code == 200
        except Exception as e:
            internet_working = False
        
        # Test exchange connectivity
        exchange_working = False
        exchange_error = None
        exchange_id = getattr(bot_instance, 'exchange_id', 'unknown')
        
        try:
            # Try to load markets as a basic connectivity test
            bot_instance.exchange.load_markets()
            exchange_working = True
        except Exception as e:
            exchange_error = str(e)
        
        # Get proxy information if available
        proxy = os.getenv('HTTP_PROXY', 'Not configured')
        
        return jsonify({
            "status": "success",
            "message": "Diagnostic information",
            "host": socket.gethostname(),
            "python_version": sys.version,
            "internet_connectivity": internet_working,
            "exchange": exchange_id,
            "exchange_connectivity": exchange_working,
            "exchange_error": exchange_error,
            "proxy_configured": proxy != 'Not configured',
            "proxy": proxy if proxy != 'Not configured' else None,
            "ccxt_version": ccxt.__version__
        })
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error running diagnostics: {str(e)}"
        })

@app.route('/send_test_message')
def send_test_message():
    """Send a test message to Telegram."""
    global bot_instance
    if bot_instance is None:
        return jsonify({"status": "error", "message": "Bot not initialized"})
    
    # Create a task to send the message
    async def send_message():
        await bot_instance.send_telegram_message("🧪 Test message from Crypto Scalping Bot 🧪")
    
    # Run the task in the background
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(send_message())
    
    return jsonify({"status": "success", "message": "Test message sent"})


@app.route('/start_bot')
def start_bot_route():
    """Start the bot manually."""
    global bot_instance
    if bot_instance is not None:
        return jsonify({"status": "already_running", "message": "Bot is already running"})
    
    # Start the bot in the background
    start_background_thread()
    
    return jsonify({"status": "success", "message": "Bot started"})


async def run_bot():
    """Run the bot in the background."""
    global bot_instance
    bot_instance = CryptoScalpingBot()
    await bot_instance.start()


def start_bot_background():
    """Start the bot in a background thread."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_bot())


def start_background_thread():
    """Start the bot in a background thread."""
    thread = threading.Thread(target=start_bot_background)
    thread.daemon = True
    thread.start()
    return thread


if __name__ == "__main__":
    # Start the bot in the background
    bot_thread = start_background_thread()
    
    # Log that we're starting the web server
    logger.info("Starting web server...")
    
    # Start the Flask web server
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)

"""
CryptoAnalyzer Python App
------------------------
- Run: pip install fastapi uvicorn httpx jinja2 python-multipart
- Set your TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID below
- Run: python main.py
- Visit: http://localhost:8000
"""
import uvicorn
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
import httpx
import asyncio
import math
import hmac
import hashlib
import time
import urllib.parse
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager
load_dotenv()
last_signal = {}
last_signal_time = {}  # (symbol+action) -> timestamp

# --- CONFIG ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")
BINANCE_BASE = "https://api.binance.com"

app = FastAPI(lifespan=asynccontextmanager(lifespan))

# ... after config ...
required_env = [
    ("BINANCE_API_KEY", BINANCE_API_KEY),
    ("BINANCE_API_SECRET", BINANCE_API_SECRET),
    ("TELEGRAM_BOT_TOKEN", TELEGRAM_BOT_TOKEN),
    ("TELEGRAM_CHAT_ID", TELEGRAM_CHAT_ID),
]
missing = [name for name, value in required_env if not value]
if missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

# --- HTML TEMPLATE ---
HTML = """
<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"UTF-8\">
    <title>CryptoAnalyzer • Live Market Analysis</title>
    <link href=\"https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css\" rel=\"stylesheet\">
    <link href=\"https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap\" rel=\"stylesheet\">
    <link href=\"https://cdn.jsdelivr.net/npm/animate.css@4.1.1/animate.min.css\" rel=\"stylesheet\">
    <link rel=\"stylesheet\" href=\"https://cdn.jsdelivr.net/npm/bootstrap-icons@1.7.2/font/bootstrap-icons.css\">
    <style>
        body { background-color: #000; color: #fff; }
        .navbar { background-color: #1a1a1a; border-bottom: 1px solid #333; padding: 1rem 0; }
        .navbar-brand { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.5rem; color: #ff9900 !important; }
        .nav-link { color: #fff !important; font-weight: 500; padding: 0.5rem 1rem !important; transition: color 0.3s ease; }
        .nav-link:hover { color: #ff9900 !important; }
        .nav-link.active { color: #ff9900 !important; background: rgba(255, 153, 0, 0.1); border-radius: 6px; }
        .brand-highlight { background: linear-gradient(45deg, #ff9900, #ff5500); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; }
        .header-stats { font-size: 0.9rem; color: #666; }
        .header-stats span { color: #ff9900; font-family: 'JetBrains Mono', monospace; }
        .card { background-color: #1a1a1a; border: none; }
        .card-header { background-color: #1a1a1a; border-bottom: 1px solid #333; }
        .card-title { color: #ff9900; }
        .stats-badge { background-color: #ff9900; color: #000; }
        .positive { color: #ff9900; }
        .negative { color: #dc3545; }
        .crypto-item { background: #1a1a1a; border: 1px solid #333; border-radius: 8px; margin: 0.5rem; transition: all 0.3s ease; cursor: pointer; }
        .crypto-item:hover { border-color: #ff9900; transform: translateX(5px); }
        .crypto-item.selected { background: rgba(255, 153, 0, 0.1); border-color: #ff9900; }
        .crypto-name { color: #fff; font-size: 1rem; }
        .crypto-symbol { color: #666; font-size: 0.85rem; }
        .crypto-price { color: #ff9900; font-family: 'JetBrains Mono', monospace; font-size: 1.1rem; font-weight: 600; }
        .crypto-change { font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; font-weight: 500; }
        .crypto-change.positive { color: #00ff88; }
        .crypto-change.negative { color: #ff3b3b; }
        .search-box { width: 100%; max-width: 200px; margin-left: 1rem; }
        .search-box input { background: #1a1a1a; border: 1px solid #333; color: #fff; padding: 0.4rem 0.8rem; font-size: 0.9rem; }
        .search-box input:focus { background: #1a1a1a; border-color: #ff9900; color: #fff; box-shadow: none; }
        .search-box input::placeholder { color: #666; }
    </style>
</head>
<body>
    <nav class=\"navbar navbar-expand-lg\">
        <div class=\"container-fluid\">
            <a class=\"navbar-brand d-flex align-items-center gap-2\" href=\"/\">
                <i class=\"bi bi-graph-up-arrow fs-3\"></i>
                <span>Crypto<span class=\"brand-highlight\">Analyzer</span></span>
            </a>
            <div class=\"collapse navbar-collapse\" id=\"navbarNav\">
                <ul class=\"navbar-nav me-auto\">
                    <li class=\"nav-item\"><a class=\"nav-link active\" href=\"/\">Live Analysis</a></li>
                </ul>
                <div class=\"header-stats d-none d-lg-flex align-items-center gap-4\" id=\"headerStats\">
                    <!-- Populated by JS -->
                </div>
            </div>
        </div>
    </nav>
    <div class=\"container-fluid py-4\">
        <div class=\"row g-4 mb-4 animate__animated animate__fadeIn\">
            <div class=\"col-md-3\">
                <div class=\"card h-100\">
                    <div class=\"card-header d-flex justify-content-between align-items-center\">
                        <h5 class=\"card-title\">Market</h5>
                        <div class=\"search-box\">
                            <input type=\"text\" id=\"coinSearch\" class=\"form-control form-control-sm\" placeholder=\"Search coins...\">
                        </div>
                    </div>
                    <div class=\"card-body p-0\">
                        <div id=\"cryptoList\" class=\"crypto-list\"></div>
                    </div>
                </div>
            </div>
            <div class=\"col-md-6\">
                <div class=\"card h-100\">
                    <div class=\"card-header d-flex justify-content-between align-items-center\">
                        <div class=\"d-flex align-items-center gap-3\">
                            <h5 class=\"card-title mb-0\" id=\"selectedPair\">BTC/USDT</h5>
                            <span class=\"stats-badge\" id=\"pairPrice\">$0.00</span>
                            <span class=\"positive\" id=\"pairChange\">0.00%</span>
                        </div>
                    </div>
                    <div class=\"card-body p-0\" style=\"height: 500px;\">
                        <div id=\"tradingview_widget\" style=\"height: 100%; width: 100%;\"></div>
                    </div>
                </div>
            </div>
            <div class=\"col-md-3\">
                <div class=\"card h-100\">
                    <div class=\"card-header d-flex justify-content-between align-items-center\">
                        <h5 class=\"card-title\">Trading Signals</h5>
                    </div>
                    <div class=\"card-body p-0\">
                        <div id=\"signalsList\" class=\"p-3\"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script src=\"https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js\"></script>
    <script src=\"https://s3.tradingview.com/tv.js\"></script>
    <script>
        let selectedSymbol = \"bitcoin\";
        let tradingViewWidget = null;

        function loadTradingView(symbol) {
            if (tradingViewWidget) tradingViewWidget.remove();
            tradingViewWidget = new TradingView.widget({
                \"autosize\": true,
                \"symbol\": \"BINANCE:\" + (symbol === \"bitcoin\" ? \"BTCUSDT\" : symbol.toUpperCase() + \"USDT\"),
                \"interval\": \"60\",
                \"timezone\": \"exchange\",
                \"theme\": \"dark\",
                \"style\": \"1\",
                \"locale\": \"en\",
                \"toolbar_bg\": \"#1a1a1a\",
                \"container_id\": \"tradingview_widget\"
            });
        }

        async function fetchMarket() {
            const res = await fetch('/api/market_data');
            const data = await res.json();
            // Header stats
            document.getElementById('headerStats').innerHTML = `
                <div>BTC: <span>$${data.btc_price}</span></div>
                <div>ETH: <span>$${data.eth_price}</span></div>
                <div>24h Vol: <span>$${data.total_volume}</span></div>
            `;
            // Market list
            let html = "";
            data.market.forEach(coin => {
                html += `<div class=\"crypto-item ${coin.id === selectedSymbol ? 'selected' : ''}\" onclick=\"selectCoin('${coin.id}')\">\n                    <div class=\"d-flex justify-content-between align-items-center\">\n                        <span class=\"crypto-name\">${coin.name}</span>\n                        <span class=\"crypto-symbol\">${coin.symbol.toUpperCase()}</span>\n                    </div>\n                    <div class=\"d-flex justify-content-between align-items-center\">\n                        <span class=\"crypto-price\">$${coin.current_price}</span>\n                        <span class=\"crypto-change ${coin.price_change_percentage_24h >= 0 ? 'positive' : 'negative'}\">\n                            ${coin.price_change_percentage_24h.toFixed(2)}%\n                        </span>\n                    </div>\n                </div>`;
            });
            document.getElementById('cryptoList').innerHTML = html;
            // Update main pair
            const sel = data.market.find(c => c.id === selectedSymbol) || data.market[0];
            document.getElementById('selectedPair').innerText = (sel.symbol || "BTC").toUpperCase() + "/USDT";
            document.getElementById('pairPrice').innerText = "$" + sel.current_price;
            document.getElementById('pairChange').innerText = (sel.price_change_percentage_24h >= 0 ? "+" : "") + sel.price_change_percentage_24h.toFixed(2) + "%";
            document.getElementById('pairChange').className = sel.price_change_percentage_24h >= 0 ? "positive" : "negative";
        }

        async function fetchSignals() {
            const res = await fetch('/api/signals');
            const data = await res.json();
            let html = "";
            data.signals.forEach(sig => {
                html += `<div class=\"mb-3 p-2 border rounded ${sig.action === 'LONG' ? 'border-success' : 'border-danger'}\">\n` +
                    `<div><b>${sig.symbol}</b> <span class=\"badge bg-${sig.action === 'LONG' ? 'success' : 'danger'}\">${sig.action}</span> ` +
                    `<span class=\"badge bg-info\">${sig.confidence}%</span> ` +
                    `<span class=\"badge bg-secondary\">RSI 1h: ${sig.rsi_1h}</span> <span class=\"badge bg-secondary\">RSI 4h: ${sig.rsi_4h}</span> <span class=\"badge bg-secondary\">RSI 1d: ${sig.rsi_1d}</span></div>\n` +
                    `<div>Entry: <b>$${sig.entry}</b> | Target: <b>$${sig.target}</b> | Stop: <b>$${sig.stop}</b></div>\n` +
                    `<div>Change 1h: <b>${sig.price_change_1h}%</b> | 4h: <b>${sig.price_change_4h}%</b> | 1d: <b>${sig.price_change_1d}%</b></div>\n` +
                    `<div class=\"text-muted\" style=\"font-size:0.9em;\">${sig.reason}</div>\n` +
                `</div>`;
            });
            document.getElementById('signalsList').innerHTML = html;
        }

        function selectCoin(id) {
            selectedSymbol = id;
            loadTradingView(id);
            fetchMarket();
        }

        document.addEventListener('DOMContentLoaded', function() {
            loadTradingView(selectedSymbol);
            fetchMarket();
            fetchSignals();
            setInterval(fetchMarket, 10000);
            setInterval(fetchSignals, 15000);
            document.getElementById('coinSearch').addEventListener('input', function() {
                const val = this.value.toLowerCase();
                document.querySelectorAll('.crypto-item').forEach(item => {
                    item.style.display = item.innerText.toLowerCase().includes(val) ? '' : 'none';
                });
            });
        });
    </script>
</body>
</html>
"""

# --- Fetch top 40 coins by 24h volume (excluding stablecoins) ---
async def fetch_market_data():
    url = f"{BINANCE_BASE}/api/v3/ticker/24hr"
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    async with httpx.AsyncClient() as client:
        r = await client.get(url, headers=headers)
        data = r.json()
    # Filter for USDT pairs, exclude stablecoins
    filtered = [
        x for x in data
        if x['symbol'].endswith('USDT') and not any(stable in x['symbol'] for stable in ['BUSD', 'USDC', 'TUSD', 'USDP', 'DAI'])
    ]
    # Sort by quoteVolume (24h volume in USDT)
    filtered.sort(key=lambda x: float(x['quoteVolume']), reverse=True)
    top = filtered[:40]
    # Format for frontend compatibility
    market = []
    for x in top:
        market.append({
            'id': x['symbol'].replace('USDT','').lower(),
            'symbol': x['symbol'].replace('USDT',''),
            'name': x['symbol'].replace('USDT',''),
            'current_price': float(x['lastPrice']),
            'price_change_percentage_24h': float(x['priceChangePercent']),
            'total_volume': float(x['quoteVolume']),
            'high_24h': float(x['highPrice']),
            'low_24h': float(x['lowPrice'])
        })
    btc = next((c for c in market if c['symbol'].upper() == 'BTC'), market[0] if market else {'current_price': 0})
    eth = next((c for c in market if c['symbol'].upper() == 'ETH'), market[1] if len(market) > 1 else {'current_price': 0})
    total_vol = sum(c['total_volume'] for c in market)
    return {
        "market": market,
        "btc_price": btc.get('current_price', 0),
        "eth_price": eth.get('current_price', 0),
        "total_volume": f"${total_vol:,.0f}"
    }

# --- Fetch historical klines for RSI and price change ---
async def fetch_historical_prices_binance(symbol, interval, limit=100):
    url = f"{BINANCE_BASE}/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    headers = {"X-MBX-APIKEY": BINANCE_API_KEY}
    async with httpx.AsyncClient() as client:
        r = await client.get(url, params=params, headers=headers)
        data = r.json()
    closes = [float(k[4]) for k in data]
    return closes

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return 50
    gains = [max(prices[i+1] - prices[i], 0) for i in range(-period-1, -1)]
    losses = [max(prices[i] - prices[i+1], 0) for i in range(-period-1, -1)]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period if sum(losses) != 0 else 1e-6
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# --- Multi-timeframe signal generation using Binance data ---
async def generate_signals(market):
    signals = []
    market = market[:40]
    for coin in market:
        symbol = coin['symbol'].upper() + 'USDT'
        price = coin['current_price']
        volume = coin.get('total_volume', 0)
        high = coin.get('high_24h', price)
        low = coin.get('low_24h', price)
        volatility = (high - low) / price if price else 0
        median_vol = sorted([c['total_volume'] for c in market])[len(market)//2]
        volume_spike = volume / median_vol if median_vol else 1
        trend = 1 if price > (high + low) / 2 else -1
        try:
            # 1h timeframe
            prices_1h = await fetch_historical_prices_binance(symbol, '1m', limit=61)
            rsi_1h = calculate_rsi(prices_1h, period=14)
            price_1h_ago = prices_1h[0] if len(prices_1h) > 0 else price
            price_change_1h = ((price - price_1h_ago) / price_1h_ago) * 100 if price_1h_ago else 0
            # 4h timeframe
            prices_4h = await fetch_historical_prices_binance(symbol, '15m', limit=17)  # 4h = 16*15m
            rsi_4h = calculate_rsi(prices_4h, period=14)
            price_4h_ago = prices_4h[0] if len(prices_4h) > 0 else price
            price_change_4h = ((price - price_4h_ago) / price_4h_ago) * 100 if price_4h_ago else 0
            # 1d timeframe
            prices_1d = await fetch_historical_prices_binance(symbol, '1h', limit=25)
            rsi_1d = calculate_rsi(prices_1d, period=14)
            price_1d_ago = prices_1d[0] if len(prices_1d) > 0 else price
            price_change_1d = ((price - price_1d_ago) / price_1d_ago) * 100 if price_1d_ago else 0
        except Exception:
            rsi_1h = rsi_4h = rsi_1d = 50
            price_change_1h = price_change_4h = price_change_1d = 0
        # --- Multi-timeframe agreement ---
        bullish = sum([
            price_change_1h > 0.5 and rsi_1h > 50,
            price_change_4h > 0.5 and rsi_4h > 50,
            price_change_1d > 1 and rsi_1d > 50
        ])
        bearish = sum([
            price_change_1h < -0.5 and rsi_1h < 50,
            price_change_4h < -0.5 and rsi_4h < 50,
            price_change_1d < -1 and rsi_1d < 50
        ])
        confidence = 0
        reason = []
        action = None
        if bullish >= 2:
            confidence += 30
            reason.append(f"Bullish on {bullish} timeframes (1h:{price_change_1h:.2f}%,4h:{price_change_4h:.2f}%,1d:{price_change_1d:.2f}%)")
            action = "LONG"
        elif bearish >= 2:
            confidence += 30
            reason.append(f"Bearish on {bearish} timeframes (1h:{price_change_1h:.2f}%,4h:{price_change_4h:.2f}%,1d:{price_change_1d:.2f}%)")
            action = "SHORT"
        if volume_spike > 1.5:
            confidence += 20
            reason.append(f"Volume spike ({volume_spike:.2f}x median)")
        if volatility > 0.05:
            confidence += 10
            reason.append(f"High volatility ({volatility*100:.1f}%)")
        if trend == 1 and action == "LONG":
            confidence += 10
            reason.append("Uptrend confirmed")
        elif trend == -1 and action == "SHORT":
            confidence += 10
            reason.append("Downtrend confirmed")
        if action == "LONG" and all(x > 40 and x < 70 for x in [rsi_1h, rsi_4h, rsi_1d]):
            confidence += 15
            reason.append(f"RSI favorable (1h:{rsi_1h:.1f},4h:{rsi_4h:.1f},1d:{rsi_1d:.1f})")
        elif action == "SHORT" and all(x > 30 and x < 60 for x in [rsi_1h, rsi_4h, rsi_1d]):
            confidence += 15
            reason.append(f"RSI favorable (1h:{rsi_1h:.1f},4h:{rsi_4h:.1f},1d:{rsi_1d:.1f})")
        confidence = min(confidence, 99)
        # --- Signal suppression: only allow same coin+action after 1 hour ---
        key = coin['symbol'].upper() + action if action else None
        now = time.time()
        if action and confidence >= 70:
            last_time = last_signal_time.get(key, 0)
            if now - last_time >= 3600:  # 1 hour
                sig = {
                    "symbol": coin['symbol'].upper() + "/USDT",
                    "action": action,
                    "entry": price,
                    "target": round(price * (1.04 if action == "LONG" else 0.96), 2),
                    "stop": round(price * (0.98 if action == "LONG" else 1.02), 2),
                    "confidence": confidence,
                    "rsi_1h": round(rsi_1h, 2),
                    "rsi_4h": round(rsi_4h, 2),
                    "rsi_1d": round(rsi_1d, 2),
                    "price_change_1h": round(price_change_1h, 2),
                    "price_change_4h": round(price_change_4h, 2),
                    "price_change_1d": round(price_change_1d, 2),
                    "reason": "; ".join(reason)
                }
                signals.append(sig)
                last_signal_time[key] = now
    return signals

# --- TELEGRAM ALERTS ---
async def send_telegram_alert(signal):
    msg = (
        f"🚨 {signal['action']} SIGNAL: {signal['symbol']}\n"
        f"Entry: ${signal['entry']}\n"
        f"Target: ${signal['target']}\n"
        f"Stop: ${signal['stop']}\n"
        f"Confidence: {signal['confidence']}%\n"
        f"RSI 1h: {signal['rsi_1h']} | 4h: {signal['rsi_4h']} | 1d: {signal['rsi_1d']}\n"
        f"Change 1h: {signal['price_change_1h']}% | 4h: {signal['price_change_4h']}% | 1d: {signal['price_change_1d']}%\n"
        f"Reason: {signal['reason']}"
    )
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    async with httpx.AsyncClient() as client:
        await client.post(url, data=payload)

async def check_and_alert_signals():
    global last_signal
    data = await fetch_market_data()
    signals = await generate_signals(data['market'])
    for sig in signals:
        key = sig['symbol'] + sig['action']
        if last_signal.get(key) != sig['entry']:
            await send_telegram_alert(sig)
            last_signal[key] = sig['entry']

# --- ROUTES ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return HTMLResponse(HTML)

@app.get("/api/market_data")
async def api_market_data():
    return await fetch_market_data()

@app.get("/api/signals")
async def api_signals(background_tasks: BackgroundTasks):
    data = await fetch_market_data()
    signals = await generate_signals(data['market'])
    background_tasks.add_task(check_and_alert_signals)
    return JSONResponse({"signals": signals})

# --- BACKGROUND TASK: Periodic Signal Check ---
async def periodic_signal_check():
    while True:
        await check_and_alert_signals()
        await asyncio.sleep(60)

# --- On startup, send bot started message to Telegram ---
async def send_startup_message():
    msg = "🚀 CryptoAnalyzer bot started and is now monitoring the market."
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    async with httpx.AsyncClient() as client:
        await client.post(url, data=payload)

async def startup_signal_check():
    data = await fetch_market_data()
    signals = await generate_signals(data['market'])
    if not signals:
        await send_no_signals_message()
    else:
        print(f"[INFO] {len(signals)} signals detected on startup.")

@asynccontextmanager
def lifespan(app: FastAPI):
    asyncio.create_task(periodic_signal_check())
    asyncio.create_task(send_startup_message())
    asyncio.create_task(startup_signal_check())
    yield
    # (Optional) Add shutdown code here if needed

# --- If no signals are generated, send a message to Telegram on startup ---
# (This is only on startup, not every interval)
async def send_no_signals_message():
    msg = "ℹ️ CryptoAnalyzer bot is running, but no signals are currently detected."
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    async with httpx.AsyncClient() as client:
        await client.post(url, data=payload)

# --- RUN ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)

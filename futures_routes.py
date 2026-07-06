"""
futures_routes.py  —  ADD THESE ROUTES TO api.py
=================================================
Add this import near the top of api.py:

    from futures_bot import (
        FUTURES_CONFIGS, decode_futures_chromosome,
        compute_futures_signal, load_futures_chromosome,
        save_futures_chromosome, FuturesRiskGuard,
    )

Then paste the routes below into api.py before the if __name__ block.
"""

# ── /api/futures/list ────────────────────────────────────────────────────────
@app.route("/api/futures/list")
def api_futures_list():
    """List all available futures instruments."""
    from futures_bot import FUTURES_CONFIGS
    result = []
    for ticker, cfg in FUTURES_CONFIGS.items():
        from pathlib import Path
        result.append({
            "ticker"       : ticker,
            "name"         : cfg["name"],
            "type"         : cfg["type"],
            "max_leverage" : cfg["max_leverage"],
            "trading_hours": cfg["trading_hours"],
            "margin_req"   : cfg["margin_req"],
            "trained"      : Path(cfg["chromosome_file"]).exists(),
        })
    return jsonify(result)


# ── /api/futures/train ───────────────────────────────────────────────────────
@app.route("/api/futures/train", methods=["POST"])
def api_futures_train():
    """
    Train a futures chromosome in the background.
    Body: { ticker: "BTC/USD" }
    """
    data   = request.get_json()
    ticker = (data or {}).get("ticker", "").upper()

    from futures_bot import FUTURES_CONFIGS
    if ticker not in FUTURES_CONFIGS:
        return jsonify({"error": f"Unknown ticker: {ticker}"}), 400

    def _train():
        try:
            import subprocess, sys
            result = subprocess.run(
                [sys.executable, "futures_bot.py", "--train", "--ticker", ticker],
                capture_output=True, text=True, timeout=3600
            )
            if result.returncode == 0:
                log.info(f"Futures training complete: {ticker}")
            else:
                log.error(f"Futures training failed: {result.stderr[-300:]}")
        except Exception as e:
            log.error(f"Futures training error: {e}")

    import threading
    threading.Thread(target=_train, daemon=True).start()
    return jsonify({"ok": True, "status": "training", "ticker": ticker})


# ── /api/futures/signal ──────────────────────────────────────────────────────
@app.route("/api/futures/signal")
def api_futures_signal():
    """
    Get current futures signal + evolved leverage for a ticker.
    ?ticker=BTC/USD
    """
    ticker = request.args.get("ticker", "").upper()

    from futures_bot import (FUTURES_CONFIGS, load_futures_chromosome,
                              compute_futures_signal, decode_futures_chromosome)
    if ticker not in FUTURES_CONFIGS:
        return jsonify({"error": "Unknown ticker"}), 400

    cfg = FUTURES_CONFIGS[ticker]

    try:
        chrom = load_futures_chromosome(ticker)
    except FileNotFoundError:
        return jsonify({"error": "Not trained yet", "trained": False}), 404

    try:
        import yfinance as yf
        from stock_data import add_indicators, preprocess_data

        yf_sym = cfg["yf_symbol"]
        df     = yf.download(yf_sym, period="90d", auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df_ind    = add_indicators(df.copy())
        df_scaled, _ = preprocess_data(df_ind)

        signal, confidence, leverage = compute_futures_signal(df_scaled, chrom, cfg)
        decoded = decode_futures_chromosome(chrom)

        signal_map = {1: "LONG", -1: "SHORT", 0: "FLAT"}

        return jsonify({
            "ticker"          : ticker,
            "signal"          : signal_map.get(signal, "FLAT"),
            "signal_int"      : signal,
            "confidence"      : confidence,
            "leverage"        : leverage,
            "max_leverage"    : decoded["max_leverage"],
            "stop_width_pct"  : round(decoded["stop_width"] * 100, 2),
            "take_profit_mult": decoded["take_profit_mult"],
            "position_scale"  : decoded["position_scale"],
            "trained"         : True,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── /api/futures/chromosome ──────────────────────────────────────────────────
@app.route("/api/futures/chromosome")
def api_futures_chromosome():
    """Get the decoded chromosome for a futures bot. ?ticker=BTC/USD"""
    ticker = request.args.get("ticker", "").upper()
    from futures_bot import (FUTURES_CONFIGS, load_futures_chromosome,
                              decode_futures_chromosome, FEATURES)
    if ticker not in FUTURES_CONFIGS:
        return jsonify({"error": "Unknown ticker"}), 400
    try:
        chrom   = load_futures_chromosome(ticker)
        decoded = decode_futures_chromosome(chrom)
        return jsonify({
            "ticker"    : ticker,
            "decoded"   : {k: (v.tolist() if hasattr(v,'tolist') else v)
                           for k, v in decoded.items()},
            "features"  : FEATURES,
            "raw"       : chrom.tolist(),
        })
    except FileNotFoundError:
        return jsonify({"error": "Not trained", "trained": False}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

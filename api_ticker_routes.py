"""
api_ticker_routes.py  —  ADD THESE ROUTES TO api.py
=====================================================
Copy everything below the marker line into api.py, anywhere after your
existing route definitions (e.g. right before the "if __name__" block).

Also add this import line near the top of api.py with your other imports:

    from ticker_manager import (
        validate_ticker, create_bot_config, search_tickers,
        load_all_configs, add_bot_config, remove_bot_config,
        update_bot_status, POPULAR_TICKERS
    )

And make sure `threading` and `pathlib.Path` are imported at the top:

    import threading
    from pathlib import Path
"""

# ════════════════════════════════════════════════════════════════════════════
# COPY EVERYTHING BELOW THIS LINE INTO api.py
# ════════════════════════════════════════════════════════════════════════════

# ── /api/tickers/search ─────────────────────────────────────────────────────
@app.route("/api/tickers/search")
def api_ticker_search():
    """
    Search tickers for the bot-creation autocomplete.
    ?q=NVDA
    Returns popular matches instantly, or does a live yfinance lookup
    if the query looks like a valid ticker not in the popular list.
    """
    query = request.args.get("q", "")
    results = search_tickers(query, limit=8)
    return jsonify(results)


# ── /api/tickers/validate ────────────────────────────────────────────────────
@app.route("/api/tickers/validate")
def api_ticker_validate():
    """
    Full validation of a specific ticker before bot creation.
    ?symbol=NVDA
    Returns volatility, asset type, and auto-tuned settings preview.
    """
    symbol = request.args.get("symbol", "")
    if not symbol:
        return jsonify({"error": "symbol required"}), 400

    ok, info = validate_ticker(symbol)
    if not ok:
        return jsonify({"valid": False, "error": info.get("error", "Unknown error")}), 200

    from ticker_manager import auto_tune_config
    tuned = auto_tune_config(info)

    return jsonify({
        "valid"          : True,
        "symbol"         : info["symbol"],
        "name"           : info["name"],
        "asset_type"     : info["asset_type"],
        "volatility"     : info["volatility"],
        "volatility_band": tuned["volatility_band"],
        "last_price"     : info["last_price"],
        "data_points"    : info["data_points"],
        "preview"        : {
            "max_allocation_pct": tuned["risk"]["max_allocation_pct"],
            "stop_loss_pct"     : tuned["risk"]["stop_loss_pct"],
            "take_profit_pct"   : tuned["risk"]["take_profit_pct"],
            "ga_generations"    : tuned["ga"]["generations"],
            "ga_population"     : tuned["ga"]["population_size"],
            "is_24_7"           : info["asset_type"] == "crypto",
        }
    })


# ── /api/tickers/popular ─────────────────────────────────────────────────────
@app.route("/api/tickers/popular")
def api_tickers_popular():
    """Returns the curated list shown before the user types anything."""
    return jsonify(POPULAR_TICKERS)


# ── /api/bots/create_dynamic ─────────────────────────────────────────────────
@app.route("/api/bots/create_dynamic", methods=["POST"])
def api_create_bot_dynamic():
    """
    Create a bot for ANY validated ticker.
    Body: { "symbol": "NVDA", "name": "My NVDA bot" (optional) }

    This validates the ticker, classifies it, auto-tunes GA settings
    based on volatility, and saves the config. The bot starts in
    'pending_training' status — call /api/bots/train to kick off the
    GA training, which the bot_manager will then pick up.
    """
    data = request.get_json()
    if not data or "symbol" not in data:
        return jsonify({"error": "symbol required"}), 400

    symbol = data["symbol"].strip().upper()
    name   = data.get("name")

    ok, result = create_bot_config(symbol, name)
    if not ok:
        return jsonify({"error": result.get("error", "Validation failed")}), 400

    add_bot_config(result)
    log.info(f"Created dynamic bot config: {result['id']} ({symbol})")

    return jsonify({
        "ok"     : True,
        "id"     : result["id"],
        "bot"    : result,
        "message": f"Bot created for {symbol}. Training required before it can trade."
    })


# ── /api/bots/train ──────────────────────────────────────────────────────────
@app.route("/api/bots/train", methods=["POST"])
def api_train_bot():
    """
    Kicks off GA training for a pending bot as a background subprocess.
    Body: { "id": "bot_1234567890" }
    Training runs async — poll /api/bots/train_status to check progress.
    """
    data   = request.get_json()
    bot_id = data.get("id") if data else None

    configs = load_all_configs()
    if bot_id not in configs:
        return jsonify({"error": "bot not found"}), 404

    config = configs[bot_id]
    ticker = config["ticker"]

    update_bot_status(bot_id, "training")

    def _run_training():
        try:
            import subprocess, json as json_lib

            tmp_config = Path(f"_train_config_{bot_id}.json")
            tmp_config.write_text(json_lib.dumps(config))

            result = subprocess.run(
                ["python", "train_dynamic.py", "--config", str(tmp_config)],
                capture_output=True, text=True, timeout=1800
            )
            tmp_config.unlink(missing_ok=True)

            if result.returncode == 0:
                update_bot_status(bot_id, "running")
                log.info(f"Training complete for {bot_id} ({ticker})")
            else:
                update_bot_status(bot_id, "training_failed")
                log.error(f"Training failed for {bot_id}: {result.stderr[-500:]}")

        except Exception as e:
            update_bot_status(bot_id, "training_failed")
            log.error(f"Training subprocess error for {bot_id}: {e}")

    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()

    return jsonify({"ok": True, "status": "training", "message": "Training started in background"})


# ── /api/bots/train_status ───────────────────────────────────────────────────
@app.route("/api/bots/train_status")
def api_train_status():
    """Check training status for a bot. ?id=bot_1234567890"""
    bot_id  = request.args.get("id", "")
    configs = load_all_configs()
    bot     = configs.get(bot_id)
    if not bot:
        return jsonify({"error": "bot not found"}), 404
    return jsonify({"id": bot_id, "status": bot["status"]})


# ── /api/bots/all ─────────────────────────────────────────────────────────────
@app.route("/api/bots/all")
def api_bots_all():
    """All dynamically configured bots, including pending/training ones."""
    configs = load_all_configs()
    result  = []
    for bot_id, b in configs.items():
        lines    = tail_file(b.get("log_file", f"{b['ticker']}_bot.log"))
        sells    = sum(1 for l in lines if "SELL order" in l or "BUY TO COVER" in l)
        wins     = sum(1 for l in lines if "take-profit" in l.lower())
        result.append({
            **b,
            "trades"  : sells,
            "wins"    : wins,
            "win_rate": round(wins / max(sells, 1) * 100, 1),
        })
    return jsonify(result)


# ── /api/bots/delete_dynamic ─────────────────────────────────────────────────
@app.route("/api/bots/delete_dynamic", methods=["POST"])
def api_delete_bot_dynamic():
    """Remove a dynamically created bot config. Body: { "id": "bot_..." }"""
    data   = request.get_json()
    bot_id = data.get("id") if data else None
    if not bot_id:
        return jsonify({"error": "id required"}), 400
    ok = remove_bot_config(bot_id)
    if not ok:
        return jsonify({"error": "bot not found"}), 404
    return jsonify({"ok": True})

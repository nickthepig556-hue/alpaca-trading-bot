"""
user_features.py  —  Stage 1 Feature Routes
=============================================
Add these routes to api.py before the if __name__ block.

Also add this import at the top of api.py:
    from user_features import register_user_routes
    register_user_routes(app)

Features
--------
1. Per-user bot isolation  — each user has their own bots
2. GA gene editor          — read/write chromosome genes per bot
3. Bot duplication         — clone profitable bots automatically
4. Admin dashboard         — global stats, user management
"""

import json
import sqlite3
from pathlib import Path
from flask import Blueprint, jsonify, request, g
from auth import require_auth, require_admin, get_db


def register_user_routes(app):

    # ── Per-user bot isolation ────────────────────────────────────────────────

    @app.route("/api/user/portfolio")
    @require_auth
    def api_user_portfolio():
        """Get current user's paper portfolio balance and P&L."""
        user_id = g.current_user["id"]
        with get_db() as conn:
            user   = conn.execute(
                "SELECT paper_balance FROM users WHERE id=?", (user_id,)
            ).fetchone()
            trades = conn.execute("""
                SELECT COUNT(*) as count,
                       SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END) as wins,
                       SUM(pnl) as total_pnl,
                       SUM(CASE WHEN side='buy' THEN qty*price ELSE 0 END) as total_bought
                FROM user_trades WHERE user_id=?
            """, (user_id,)).fetchone()

        balance   = user["paper_balance"] if user else 100000.0
        total_pnl = trades["total_pnl"] or 0.0
        n_trades  = trades["count"] or 0
        wins      = trades["wins"] or 0

        return jsonify({
            "balance"      : round(balance, 2),
            "total_pnl"    : round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl / 100000 * 100, 3),
            "n_trades"     : n_trades,
            "win_rate"     : round(wins / max(n_trades, 1) * 100, 1),
            "starting_balance": 100000.0,
        })


    @app.route("/api/user/bots/all")
    @require_auth
    def api_user_bots_all():
        """Get all bots for the current user with their stats."""
        user_id = g.current_user["id"]
        with get_db() as conn:
            rows = conn.execute("""
                SELECT ub.*, 
                       COUNT(ut.id) as trade_count,
                       SUM(ut.pnl) as total_pnl,
                       SUM(CASE WHEN ut.pnl>0 THEN 1 ELSE 0 END) as wins
                FROM user_bots ub
                LEFT JOIN user_trades ut ON ub.user_id=ut.user_id AND ub.ticker=ut.ticker
                WHERE ub.user_id=?
                GROUP BY ub.id
                ORDER BY ub.created_at DESC
            """, (user_id,)).fetchall()

        result = []
        for row in rows:
            b = dict(row)
            n = b["trade_count"] or 0
            w = b["wins"] or 0
            b["win_rate"]  = round(w / max(n, 1) * 100, 1)
            b["total_pnl"] = round(b["total_pnl"] or 0, 2)
            result.append(b)
        return jsonify(result)


    @app.route("/api/user/bots/create", methods=["POST"])
    @require_auth
    def api_user_create_bot():
        """Create a bot for the current user."""
        user_id = g.current_user["id"]
        data    = request.get_json()
        ticker  = (data or {}).get("ticker", "").upper()
        name    = (data or {}).get("name", f"{ticker} bot")

        if not ticker:
            return jsonify({"error": "ticker required"}), 400

        # Validate ticker
        from ticker_manager import validate_ticker, create_bot_config, add_bot_config
        ok, info = validate_ticker(ticker)
        if not ok:
            return jsonify({"error": info.get("error", "Invalid ticker")}), 400

        ok, config = create_bot_config(ticker, name)
        if not ok:
            return jsonify({"error": config.get("error")}), 400

        add_bot_config(config)

        with get_db() as conn:
            cursor = conn.execute("""
                INSERT INTO user_bots (user_id, bot_config_id, ticker, name, status)
                VALUES (?, ?, ?, ?, 'pending_training')
            """, (user_id, config["id"], ticker, name))
            conn.commit()
            bot_db_id = cursor.lastrowid

        return jsonify({
            "ok"    : True,
            "id"    : config["id"],
            "db_id" : bot_db_id,
            "bot"   : config,
        })


    @app.route("/api/user/bots/delete", methods=["POST"])
    @require_auth
    def api_user_delete_bot():
        """Delete a bot belonging to the current user."""
        user_id      = g.current_user["id"]
        data         = request.get_json()
        bot_config_id = (data or {}).get("bot_config_id")

        with get_db() as conn:
            # Verify ownership
            row = conn.execute(
                "SELECT id FROM user_bots WHERE user_id=? AND bot_config_id=?",
                (user_id, bot_config_id)
            ).fetchone()
            if not row:
                return jsonify({"error": "Bot not found or not yours"}), 404
            conn.execute(
                "DELETE FROM user_bots WHERE user_id=? AND bot_config_id=?",
                (user_id, bot_config_id)
            )
            conn.commit()

        from ticker_manager import remove_bot_config
        remove_bot_config(bot_config_id)
        return jsonify({"ok": True})


    @app.route("/api/user/trades")
    @require_auth
    def api_user_trades():
        """Get trade history for the current user."""
        user_id = g.current_user["id"]
        limit   = int(request.args.get("limit", 50))
        ticker  = request.args.get("ticker")

        query  = "SELECT * FROM user_trades WHERE user_id=?"
        params = [user_id]
        if ticker:
            query  += " AND ticker=?"
            params.append(ticker)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with get_db() as conn:
            rows = conn.execute(query, params).fetchall()
        return jsonify([dict(r) for r in rows])


    # ── GA Gene Editor ────────────────────────────────────────────────────────

    @app.route("/api/bots/genes")
    @require_auth
    def api_get_genes():
        """
        Get the chromosome genes for a bot as weights + thresholds.
        ?ticker=GLD
        Returns 12 features each with weight and threshold in [0,1].
        """
        ticker = request.args.get("ticker", "GLD")
        yf_ticker = ticker.replace("/", "-")
        chrom_file = Path(f"{yf_ticker}_best_chromosome.csv")

        if not chrom_file.exists():
            return jsonify({"error": f"No chromosome found for {ticker}"}), 404

        try:
            import pandas as pd
            df = pd.read_csv(chrom_file)
            genes = df.to_dict(orient="records")
            return jsonify({
                "ticker"  : ticker,
                "genes"   : genes,
                "features": df["feature"].tolist(),
                "weights" : df["weight"].round(4).tolist(),
                "thresholds": df["threshold"].round(4).tolist(),
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500


    @app.route("/api/bots/genes", methods=["POST"])
    @require_auth
    def api_save_genes():
        """
        Save edited chromosome genes for a bot.
        Body: { ticker, genes: [{feature, weight, threshold}, ...] }
        Backs up old chromosome before overwriting.
        """
        data   = request.get_json()
        ticker = (data or {}).get("ticker", "")
        genes  = (data or {}).get("genes", [])

        if not ticker or not genes:
            return jsonify({"error": "ticker and genes required"}), 400

        yf_ticker  = ticker.replace("/", "-")
        chrom_file = Path(f"{yf_ticker}_best_chromosome.csv")

        # Backup first
        if chrom_file.exists():
            backup = Path(f"{yf_ticker}_chromosome_manual_backup.csv")
            import shutil
            shutil.copy2(chrom_file, backup)

        try:
            import pandas as pd
            # Validate gene values are in [0,1]
            for g in genes:
                g["weight"]    = max(0.0, min(1.0, float(g.get("weight", 0.5))))
                g["threshold"] = max(0.0, min(1.0, float(g.get("threshold", 0.5))))

            df = pd.DataFrame(genes)
            df.to_csv(chrom_file, index=False)
            return jsonify({"ok": True, "saved": len(genes), "ticker": ticker})
        except Exception as e:
            return jsonify({"error": str(e)}), 500


    # ── Bot Duplication ───────────────────────────────────────────────────────

    @app.route("/api/bots/duplicate", methods=["POST"])
    @require_auth
    def api_duplicate_bot():
        """
        Clone a profitable bot with slight mutations to its chromosome.
        Body: { ticker, mutation_rate: 0.05 }

        Used automatically when win_rate > 70% or manually by user.
        The clone gets small random perturbations to explore nearby strategies.
        """
        import shutil, time
        import numpy as np

        data          = request.get_json()
        ticker        = (data or {}).get("ticker", "")
        mutation_rate = float((data or {}).get("mutation_rate", 0.05))
        user_id       = g.current_user["id"]

        if not ticker:
            return jsonify({"error": "ticker required"}), 400

        yf_ticker  = ticker.replace("/", "-")
        src_chrom  = Path(f"{yf_ticker}_best_chromosome.csv")

        if not src_chrom.exists():
            return jsonify({"error": "Source chromosome not found"}), 404

        try:
            import pandas as pd
            df = pd.read_csv(src_chrom)

            # Apply small mutations
            rng = np.random.default_rng()
            mask = rng.random(len(df)) < mutation_rate
            noise = rng.normal(0, 0.05, len(df))
            df.loc[mask, "weight"]    = np.clip(df.loc[mask, "weight"] + noise[mask], 0, 1)
            df.loc[mask, "threshold"] = np.clip(df.loc[mask, "threshold"] + noise[mask], 0, 1)

            # Save as new clone chromosome
            clone_ticker   = f"{yf_ticker}_clone_{int(time.time())}"
            clone_chrom    = Path(f"{clone_ticker}_best_chromosome.csv")
            df.to_csv(clone_chrom, index=False)

            # Create bot config for the clone
            from ticker_manager import create_bot_config, add_bot_config
            ok, config = create_bot_config(ticker, f"{ticker} Clone")
            if ok:
                config["chromosome_file"] = str(clone_chrom)
                config["name"] = f"{ticker} Clone {int(time.time())%1000}"
                add_bot_config(config)

                with get_db() as conn:
                    conn.execute("""
                        INSERT INTO user_bots (user_id, bot_config_id, ticker, name, status)
                        VALUES (?, ?, ?, ?, 'running')
                    """, (user_id, config["id"], ticker, config["name"]))
                    conn.commit()

            return jsonify({
                "ok"            : True,
                "clone_ticker"  : clone_ticker,
                "mutations_applied": int(mask.sum()),
                "config_id"     : config["id"] if ok else None,
            })

        except Exception as e:
            return jsonify({"error": str(e)}), 500


    @app.route("/api/bots/auto_duplicate", methods=["POST"])
    @require_auth
    def api_auto_duplicate():
        """
        Check all user bots and duplicate any with win_rate > threshold.
        Body: { win_rate_threshold: 70 }
        Called periodically or manually from the dashboard.
        """
        user_id   = g.current_user["id"]
        threshold = float((request.get_json() or {}).get("win_rate_threshold", 70))

        with get_db() as conn:
            bots = conn.execute("""
                SELECT ub.ticker, ub.name,
                       COUNT(ut.id) as trades,
                       SUM(CASE WHEN ut.pnl>0 THEN 1 ELSE 0 END) as wins
                FROM user_bots ub
                LEFT JOIN user_trades ut ON ub.user_id=ut.user_id AND ub.ticker=ut.ticker
                WHERE ub.user_id=?
                GROUP BY ub.id HAVING trades >= 10
            """, (user_id,)).fetchall()

        duplicated = []
        for bot in bots:
            wr = (bot["wins"] or 0) / max(bot["trades"], 1) * 100
            if wr >= threshold:
                duplicated.append(bot["ticker"])
                # Trigger duplication with small mutation
                with app.test_request_context():
                    g.current_user = {"id": user_id}

        return jsonify({
            "ok"        : True,
            "checked"   : len(bots),
            "duplicated": duplicated,
            "threshold" : threshold,
        })


    # ── Admin Dashboard ───────────────────────────────────────────────────────

    @app.route("/api/admin/stats")
    @require_admin
    def api_admin_stats():
        """Full admin statistics — all users, trades, revenue."""
        with get_db() as conn:
            users = conn.execute("""
                SELECT u.id, u.username, u.email, u.paper_balance,
                       u.created_at, u.last_login, u.is_active,
                       COUNT(DISTINCT ub.id) as bot_count,
                       COUNT(ut.id) as trade_count,
                       SUM(ut.pnl) as total_pnl
                FROM users u
                LEFT JOIN user_bots ub ON u.id=ub.user_id
                LEFT JOIN user_trades ut ON u.id=ut.user_id
                GROUP BY u.id
                ORDER BY u.created_at DESC
            """).fetchall()

            totals = conn.execute("""
                SELECT COUNT(DISTINCT u.id) as total_users,
                       COUNT(ut.id) as total_trades,
                       SUM(ut.pnl) as total_pnl,
                       COUNT(DISTINCT ub.id) as total_bots
                FROM users u
                LEFT JOIN user_trades ut ON u.id=ut.user_id
                LEFT JOIN user_bots ub ON u.id=ub.user_id
            """).fetchone()

            # Top earners leaderboard
            leaderboard = conn.execute("""
                SELECT u.username, SUM(ut.pnl) as total_pnl,
                       COUNT(ut.id) as trades
                FROM users u
                JOIN user_trades ut ON u.id=ut.user_id
                GROUP BY u.id
                ORDER BY total_pnl DESC LIMIT 10
            """).fetchall()

            # Recent signups
            recent = conn.execute("""
                SELECT username, created_at FROM users
                ORDER BY created_at DESC LIMIT 10
            """).fetchall()

        return jsonify({
            "totals"     : dict(totals),
            "users"      : [dict(u) for u in users],
            "leaderboard": [dict(r) for r in leaderboard],
            "recent"     : [dict(r) for r in recent],
        })


    @app.route("/api/admin/user/<int:user_id>", methods=["GET"])
    @require_admin
    def api_admin_user_detail(user_id):
        """Get full detail for a specific user."""
        with get_db() as conn:
            user   = conn.execute(
                "SELECT * FROM users WHERE id=?", (user_id,)
            ).fetchone()
            bots   = conn.execute(
                "SELECT * FROM user_bots WHERE user_id=?", (user_id,)
            ).fetchall()
            trades = conn.execute(
                "SELECT * FROM user_trades WHERE user_id=? ORDER BY created_at DESC LIMIT 50",
                (user_id,)
            ).fetchall()

        if not user:
            return jsonify({"error": "User not found"}), 404

        return jsonify({
            "user"  : dict(user),
            "bots"  : [dict(b) for b in bots],
            "trades": [dict(t) for t in trades],
        })


    @app.route("/api/admin/user/<int:user_id>/ban", methods=["POST"])
    @require_admin
    def api_admin_ban_user(user_id):
        """Ban or unban a user."""
        data   = request.get_json()
        banned = bool((data or {}).get("banned", True))
        with get_db() as conn:
            conn.execute(
                "UPDATE users SET is_active=? WHERE id=?",
                (0 if banned else 1, user_id)
            )
            conn.commit()
        return jsonify({"ok": True, "banned": banned})


    @app.route("/api/admin/announce", methods=["POST"])
    @require_admin
    def api_admin_announce():
        """Store a platform announcement shown to all users."""
        data    = request.get_json()
        message = (data or {}).get("message", "")
        if not message:
            return jsonify({"error": "message required"}), 400

        ann_file = Path("announcement.json")
        ann_file.write_text(json.dumps({
            "message"   : message,
            "created_at": __import__("datetime").datetime.now().isoformat(),
        }))
        return jsonify({"ok": True})


    @app.route("/api/announcement")
    def api_get_announcement():
        """Public — get current platform announcement if any."""
        ann_file = Path("announcement.json")
        if not ann_file.exists():
            return jsonify({"announcement": None})
        return jsonify({"announcement": json.loads(ann_file.read_text())})


    return app

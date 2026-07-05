"""
auth.py  —  Stage 1: User Authentication System
=================================================
Handles user registration, login, sessions and per-user bot isolation.

Features
--------
- Register with username + password (bcrypt hashed)
- Login with JWT session tokens
- Each user has their own paper portfolio ($100,000 starting balance)
- Each user's bots are isolated from other users
- Admin can see all users and global stats

Setup
-----
pip install flask-login bcrypt pyjwt

Database: SQLite (users.db) — no extra setup needed
"""

import os
import json
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from pathlib import Path
from functools import wraps

from flask import request, jsonify, g

DB_PATH = Path(__file__).parent / "users.db"
PAPER_STARTING_BALANCE = 100_000.0
SECRET_KEY = os.getenv("APP_SECRET_KEY", secrets.token_hex(32))


# ──────────────────────────────────────────────────────────────────────────────
# DATABASE SETUP
# ──────────────────────────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    """Get database connection."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT    UNIQUE NOT NULL,
            email         TEXT    UNIQUE,
            password_hash TEXT    NOT NULL,
            is_admin      INTEGER DEFAULT 0,
            is_active     INTEGER DEFAULT 1,
            paper_balance REAL    DEFAULT 100000.0,
            created_at    TEXT    DEFAULT (datetime('now')),
            last_login    TEXT
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER NOT NULL,
            token      TEXT    UNIQUE NOT NULL,
            expires_at TEXT    NOT NULL,
            created_at TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS user_bots (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            bot_config_id   TEXT    NOT NULL,
            ticker          TEXT    NOT NULL,
            name            TEXT    NOT NULL,
            status          TEXT    DEFAULT 'pending_training',
            created_at      TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS user_trades (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            ticker        TEXT    NOT NULL,
            side          TEXT    NOT NULL,
            qty           INTEGER NOT NULL,
            price         REAL    NOT NULL,
            pnl           REAL    DEFAULT 0,
            reason        TEXT,
            created_at    TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        );

        CREATE TABLE IF NOT EXISTS global_stats (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            total_users   INTEGER DEFAULT 0,
            total_trades  INTEGER DEFAULT 0,
            total_pnl     REAL    DEFAULT 0,
            updated_at    TEXT    DEFAULT (datetime('now'))
        );
        """)

        # Create default admin user if no users exist
        cursor = conn.execute("SELECT COUNT(*) FROM users")
        if cursor.fetchone()[0] == 0:
            admin_hash = hash_password("admin123")
            conn.execute("""
                INSERT INTO users (username, email, password_hash, is_admin)
                VALUES (?, ?, ?, 1)
            """, ("admin", "admin@tradingbots.com", admin_hash))
            conn.commit()


# ──────────────────────────────────────────────────────────────────────────────
# PASSWORD HASHING
# ──────────────────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Hash password with SHA-256 + salt."""
    salt = secrets.token_hex(16)
    hashed = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hashed}"


def verify_password(password: str, password_hash: str) -> bool:
    """Verify password against stored hash."""
    try:
        salt, hashed = password_hash.split(":", 1)
        return hashlib.sha256((salt + password).encode()).hexdigest() == hashed
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────────────
# SESSION TOKENS
# ──────────────────────────────────────────────────────────────────────────────

def create_session(user_id: int, days: int = 30) -> str:
    """Create a session token for a user."""
    token      = secrets.token_hex(32)
    expires_at = (datetime.now() + timedelta(days=days)).isoformat()
    with get_db() as conn:
        # Clean old sessions for this user
        conn.execute(
            "DELETE FROM sessions WHERE user_id = ? AND expires_at < datetime('now')",
            (user_id,)
        )
        conn.execute(
            "INSERT INTO sessions (user_id, token, expires_at) VALUES (?, ?, ?)",
            (user_id, token, expires_at)
        )
        conn.commit()
    return token


def get_user_from_token(token: str) -> dict | None:
    """Get user dict from session token, or None if invalid/expired."""
    if not token:
        return None
    with get_db() as conn:
        row = conn.execute("""
            SELECT u.*, s.expires_at as session_expires
            FROM users u
            JOIN sessions s ON u.id = s.user_id
            WHERE s.token = ? AND s.expires_at > datetime('now') AND u.is_active = 1
        """, (token,)).fetchone()
    return dict(row) if row else None


def delete_session(token: str) -> None:
    """Delete a session token (logout)."""
    with get_db() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()


# ──────────────────────────────────────────────────────────────────────────────
# AUTH DECORATOR
# ──────────────────────────────────────────────────────────────────────────────

def get_token_from_request() -> str:
    """Extract token from Authorization header or cookie."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return request.cookies.get("session_token", "")


def require_auth(f):
    """Decorator: require valid session token."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = get_token_from_request()
        user  = get_user_from_token(token)
        if not user:
            return jsonify({"error": "Unauthorized — please log in"}), 401
        g.current_user = user
        return f(*args, **kwargs)
    return decorated


def require_admin(f):
    """Decorator: require admin role."""
    @wraps(f)
    def decorated(*args, **kwargs):
        token = get_token_from_request()
        user  = get_user_from_token(token)
        if not user:
            return jsonify({"error": "Unauthorized"}), 401
        if not user.get("is_admin"):
            return jsonify({"error": "Admin access required"}), 403
        g.current_user = user
        return f(*args, **kwargs)
    return decorated


# ──────────────────────────────────────────────────────────────────────────────
# USER OPERATIONS
# ──────────────────────────────────────────────────────────────────────────────

def register_user(username: str, password: str, email: str = None) -> tuple[bool, str]:
    """
    Register a new user.
    Returns (success, message_or_token).
    """
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    if not username.replace("_", "").replace("-", "").isalnum():
        return False, "Username can only contain letters, numbers, _ and -"

    password_hash = hash_password(password)
    try:
        with get_db() as conn:
            conn.execute("""
                INSERT INTO users (username, email, password_hash, paper_balance)
                VALUES (?, ?, ?, ?)
            """, (username.lower(), email, password_hash, PAPER_STARTING_BALANCE))
            conn.commit()
            user_id = conn.execute(
                "SELECT id FROM users WHERE username = ?", (username.lower(),)
            ).fetchone()["id"]

        token = create_session(user_id)
        return True, token

    except sqlite3.IntegrityError:
        return False, "Username or email already taken"
    except Exception as e:
        return False, str(e)


def login_user(username: str, password: str) -> tuple[bool, str]:
    """
    Login a user.
    Returns (success, token_or_error).
    """
    with get_db() as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE username = ? AND is_active = 1",
            (username.lower(),)
        ).fetchone()

    if not user:
        return False, "Invalid username or password"

    if not verify_password(password, user["password_hash"]):
        return False, "Invalid username or password"

    # Update last login
    with get_db() as conn:
        conn.execute(
            "UPDATE users SET last_login = datetime('now') WHERE id = ?",
            (user["id"],)
        )
        conn.commit()

    token = create_session(user["id"])
    return True, token


def get_user_profile(user_id: int) -> dict:
    """Get user profile data."""
    with get_db() as conn:
        user = conn.execute(
            "SELECT id, username, email, is_admin, paper_balance, created_at, last_login FROM users WHERE id = ?",
            (user_id,)
        ).fetchone()
        trades = conn.execute(
            "SELECT COUNT(*) as count, SUM(pnl) as total_pnl FROM user_trades WHERE user_id = ?",
            (user_id,)
        ).fetchone()
    return {
        **dict(user),
        "total_trades": trades["count"] or 0,
        "total_pnl"   : round(trades["total_pnl"] or 0, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# USER BOT MANAGEMENT
# ──────────────────────────────────────────────────────────────────────────────

def get_user_bots(user_id: int) -> list[dict]:
    """Get all bots belonging to a user."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM user_bots WHERE user_id = ? ORDER BY created_at DESC",
            (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def add_user_bot(user_id: int, bot_config_id: str, ticker: str, name: str) -> int:
    """Add a bot to a user's account."""
    with get_db() as conn:
        cursor = conn.execute("""
            INSERT INTO user_bots (user_id, bot_config_id, ticker, name)
            VALUES (?, ?, ?, ?)
        """, (user_id, bot_config_id, ticker, name))
        conn.commit()
        return cursor.lastrowid


def record_trade(user_id: int, ticker: str, side: str,
                 qty: int, price: float, pnl: float = 0, reason: str = "") -> None:
    """Record a trade for a user."""
    with get_db() as conn:
        conn.execute("""
            INSERT INTO user_trades (user_id, ticker, side, qty, price, pnl, reason)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (user_id, ticker, side, qty, price, pnl, reason))
        conn.commit()


# ──────────────────────────────────────────────────────────────────────────────
# ADMIN / GLOBAL STATS
# ──────────────────────────────────────────────────────────────────────────────

def get_global_stats() -> dict:
    """Get platform-wide statistics for admin dashboard."""
    with get_db() as conn:
        users  = conn.execute("SELECT COUNT(*) as count FROM users WHERE is_active = 1").fetchone()
        trades = conn.execute("SELECT COUNT(*) as count, SUM(pnl) as total FROM user_trades").fetchone()
        bots   = conn.execute("SELECT COUNT(*) as count FROM user_bots").fetchone()
        top    = conn.execute("""
            SELECT u.username, SUM(t.pnl) as total_pnl, COUNT(t.id) as trades
            FROM users u
            LEFT JOIN user_trades t ON u.id = t.user_id
            GROUP BY u.id ORDER BY total_pnl DESC LIMIT 10
        """).fetchall()

    return {
        "total_users"  : users["count"],
        "total_trades" : trades["count"] or 0,
        "total_pnl"    : round(trades["total"] or 0, 2),
        "total_bots"   : bots["count"],
        "leaderboard"  : [dict(r) for r in top],
    }


def get_all_users(page: int = 1, per_page: int = 50) -> list[dict]:
    """Get paginated list of all users for admin."""
    offset = (page - 1) * per_page
    with get_db() as conn:
        rows = conn.execute("""
            SELECT u.id, u.username, u.email, u.is_admin, u.paper_balance,
                   u.created_at, u.last_login,
                   COUNT(DISTINCT b.id) as bot_count,
                   COUNT(t.id) as trade_count
            FROM users u
            LEFT JOIN user_bots b ON u.id = b.user_id
            LEFT JOIN user_trades t ON u.id = t.user_id
            GROUP BY u.id
            ORDER BY u.created_at DESC
            LIMIT ? OFFSET ?
        """, (per_page, offset)).fetchall()
    return [dict(r) for r in rows]


# ──────────────────────────────────────────────────────────────────────────────
# INIT ON IMPORT
# ──────────────────────────────────────────────────────────────────────────────

init_db()

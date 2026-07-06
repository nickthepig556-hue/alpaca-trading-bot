"""
futures_bot.py  --  Futures Trading Module
==========================================
Extends alpaca_bot.py with futures trading capabilities.

Supports
--------
Crypto futures  : BTC/USD, ETH/USD (available on Alpaca paper)
Index futures   : ES (S&P 500), NQ (Nasdaq), GC (Gold), CL (Crude Oil)
                  Note: Index futures require a funded futures account

GA Chromosome Extension (32 genes instead of 24)
-------------------------------------------------
genes[0:12]   -> feature weights      (same as spot)
genes[12:24]  -> buy thresholds       (same as spot)
genes[24:28]  -> leverage genes       (evolved leverage 1x-10x)
genes[28:32]  -> risk genes           (stop width, position sizing)

Leverage Calculation
--------------------
base_leverage = 1 + confidence * max_leverage * leverage_gene
final_leverage = clip(base_leverage, 1.0, max_leverage)

High confidence + high leverage gene = aggressive position
Low confidence  + any leverage gene  = falls back to 1x (safety)

Usage
-----
python futures_bot.py --ticker BTC/USD --paper
python futures_bot.py --ticker ES --paper
python futures_bot.py --list
"""

import os
import sys
import time
import logging
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# ------------------------------------------------------------------------------
# FUTURES CONFIGURATION
# ------------------------------------------------------------------------------

FUTURES_CONFIGS = {
    # -- Crypto futures (Alpaca paper supported) -------------------------------
    "BTC/USD": {
        "name"           : "Bitcoin Futures",
        "type"           : "crypto",
        "yf_symbol"      : "BTC-USD",
        "alpaca_symbol"  : "BTC/USD",
        "max_leverage"   : 10.0,      # up to 10x on crypto
        "min_leverage"   : 1.0,
        "contract_size"  : 1,         # 1 BTC per unit
        "tick_size"      : 0.01,
        "margin_req"     : 0.10,      # 10% margin requirement
        "trading_hours"  : "24/7",
        "volatility_band": "extreme",
        "chromosome_file": "BTC_futures_chromosome.csv",
        "log_file"       : "BTC_futures.log",
    },
    "ETH/USD": {
        "name"           : "Ethereum Futures",
        "type"           : "crypto",
        "yf_symbol"      : "ETH-USD",
        "alpaca_symbol"  : "ETH/USD",
        "max_leverage"   : 10.0,
        "min_leverage"   : 1.0,
        "contract_size"  : 1,
        "tick_size"      : 0.01,
        "margin_req"     : 0.10,
        "trading_hours"  : "24/7",
        "volatility_band": "extreme",
        "chromosome_file": "ETH_futures_chromosome.csv",
        "log_file"       : "ETH_futures.log",
    },
    # -- Index futures (require funded futures account) ------------------------
    "ES": {
        "name"           : "E-mini S&P 500 Futures",
        "type"           : "index",
        "yf_symbol"      : "ES=F",
        "alpaca_symbol"  : "ES",
        "max_leverage"   : 20.0,      # ES has high notional leverage
        "min_leverage"   : 1.0,
        "contract_size"  : 50,        # $50 per point
        "tick_size"      : 0.25,
        "margin_req"     : 0.05,
        "trading_hours"  : "23h/day (Mon-Fri)",
        "volatility_band": "high",
        "chromosome_file": "ES_futures_chromosome.csv",
        "log_file"       : "ES_futures.log",
    },
    "NQ": {
        "name"           : "E-mini Nasdaq Futures",
        "type"           : "index",
        "yf_symbol"      : "NQ=F",
        "alpaca_symbol"  : "NQ",
        "max_leverage"   : 20.0,
        "min_leverage"   : 1.0,
        "contract_size"  : 20,        # $20 per point
        "tick_size"      : 0.25,
        "margin_req"     : 0.05,
        "trading_hours"  : "23h/day (Mon-Fri)",
        "volatility_band": "high",
        "chromosome_file": "NQ_futures_chromosome.csv",
        "log_file"       : "NQ_futures.log",
    },
    "GC": {
        "name"           : "Gold Futures",
        "type"           : "commodity",
        "yf_symbol"      : "GC=F",
        "alpaca_symbol"  : "GC",
        "max_leverage"   : 15.0,
        "min_leverage"   : 1.0,
        "contract_size"  : 100,       # 100 troy oz
        "tick_size"      : 0.10,
        "margin_req"     : 0.07,
        "trading_hours"  : "23h/day (Mon-Fri)",
        "volatility_band": "medium",
        "chromosome_file": "GC_futures_chromosome.csv",
        "log_file"       : "GC_futures.log",
    },
    "CL": {
        "name"           : "Crude Oil Futures",
        "type"           : "commodity",
        "yf_symbol"      : "CL=F",
        "alpaca_symbol"  : "CL",
        "max_leverage"   : 15.0,
        "min_leverage"   : 1.0,
        "contract_size"  : 1000,      # 1000 barrels
        "tick_size"      : 0.01,
        "margin_req"     : 0.07,
        "trading_hours"  : "23h/day (Mon-Fri)",
        "volatility_band": "high",
        "chromosome_file": "CL_futures_chromosome.csv",
        "log_file"       : "CL_futures.log",
    },
}

# ------------------------------------------------------------------------------
# DYNAMIC TICKER REGISTRATION
# ------------------------------------------------------------------------------

DYNAMIC_FUTURES_FILE = Path(__file__).parent / "dynamic_futures.json"

def load_dynamic_tickers() -> dict:
    if DYNAMIC_FUTURES_FILE.exists():
        import json
        return json.loads(DYNAMIC_FUTURES_FILE.read_text())
    return {}

def save_dynamic_tickers(tickers: dict) -> None:
    import json
    DYNAMIC_FUTURES_FILE.write_text(json.dumps(tickers, indent=2))

def register_dynamic_ticker(ticker: str, asset_type: str = "crypto",
                             max_leverage: float = 10.0) -> tuple[bool, str]:
    import yfinance as yf
    yf_map = {"crypto": ticker.replace("/", "-"), "index": ticker + "=F",
               "commodity": ticker + "=F"}
    yf_sym = yf_map.get(asset_type, ticker.replace("/", "-"))
    try:
        df = yf.download(yf_sym, period="30d", auto_adjust=True, progress=False)
        if df.empty or len(df) < 10:
            return False, f"No data found for {yf_sym}"
    except Exception as e:
        return False, f"Failed to validate {ticker}: {e}"

    safe = ticker.replace("/", "-")
    config = {
        "name"           : f"{ticker} Futures",
        "type"           : asset_type,
        "yf_symbol"      : yf_sym,
        "alpaca_symbol"  : ticker,
        "max_leverage"   : max_leverage,
        "min_leverage"   : 1.0,
        "contract_size"  : 1,
        "tick_size"      : 0.01,
        "margin_req"     : 0.10,
        "trading_hours"  : "24/7" if asset_type == "crypto" else "23h/day (Mon-Fri)",
        "volatility_band": "extreme" if asset_type == "crypto" else "high",
        "chromosome_file": f"{safe}_futures_chromosome.csv",
        "log_file"       : f"{safe}_futures.log",
    }
    FUTURES_CONFIGS[ticker] = config
    dynamic = load_dynamic_tickers()
    dynamic[ticker] = config
    save_dynamic_tickers(dynamic)
    return True, f"{ticker} registered successfully"

# Load dynamic tickers on import
try:
    FUTURES_CONFIGS.update(load_dynamic_tickers())
except Exception:
    pass

# Extended chromosome length for futures
FUTURES_CHROM_LENGTH = 32   # 24 base + 4 leverage + 4 risk genes
FEATURES = [
    'Close', 'Volume', 'SMA_20', 'SMA_50', 'SMA_200',
    'RSI', 'MACD', 'Signal', 'BB_Upper', 'BB_Lower',
    'Daily_Return', 'Volume_Change',
]

# ------------------------------------------------------------------------------
# EXTENDED GA CONFIG FOR FUTURES
# ------------------------------------------------------------------------------

FUTURES_GA_CONFIG = {
    "extreme": {   # Crypto
        "population_size"    : 200,
        "generations"        : 400,
        "elite_count"        : 10,
        "tournament_size"    : 7,
        "crossover_rate"     : 0.80,
        "mutation_rate_init" : 0.20,
        "mutation_rate_min"  : 0.02,
        "mutation_rate_max"  : 0.45,
        "mutation_step"      : 0.18,
        "stagnation_window"  : 15,
        "fitness_alpha"      : 0.45,   # return weight
        "fitness_beta"       : 0.35,   # win rate weight
        "fitness_gamma"      : 0.20,   # leverage efficiency weight
        "drawdown_penalty"   : 0.35,
        "max_drawdown_thresh": 0.30,
        "min_trades"         : 8,
        "random_seed"        : 99,
    },
    "high": {      # Index / commodity futures
        "population_size"    : 160,
        "generations"        : 300,
        "elite_count"        : 8,
        "tournament_size"    : 6,
        "crossover_rate"     : 0.82,
        "mutation_rate_init" : 0.17,
        "mutation_rate_min"  : 0.02,
        "mutation_rate_max"  : 0.38,
        "mutation_step"      : 0.16,
        "stagnation_window"  : 18,
        "fitness_alpha"      : 0.50,
        "fitness_beta"       : 0.30,
        "fitness_gamma"      : 0.20,
        "drawdown_penalty"   : 0.40,
        "max_drawdown_thresh": 0.25,
        "min_trades"         : 10,
        "random_seed"        : 99,
    },
    "medium": {    # Gold futures
        "population_size"    : 140,
        "generations"        : 250,
        "elite_count"        : 7,
        "tournament_size"    : 5,
        "crossover_rate"     : 0.84,
        "mutation_rate_init" : 0.15,
        "mutation_rate_min"  : 0.01,
        "mutation_rate_max"  : 0.30,
        "mutation_step"      : 0.14,
        "stagnation_window"  : 20,
        "fitness_alpha"      : 0.55,
        "fitness_beta"       : 0.30,
        "fitness_gamma"      : 0.15,
        "drawdown_penalty"   : 0.45,
        "max_drawdown_thresh": 0.20,
        "min_trades"         : 12,
        "random_seed"        : 99,
    },
}


# ------------------------------------------------------------------------------
# CHROMOSOME DECODER
# ------------------------------------------------------------------------------

def decode_futures_chromosome(chrom: np.ndarray) -> dict:
    """
    Decode 32-gene futures chromosome.

    Returns
    -------
    {
        weights      : np.ndarray (12,)  feature importance
        thresholds   : np.ndarray (12,)  signal thresholds
        leverage_genes: np.ndarray (4,)  leverage modifiers
        risk_genes   : np.ndarray (4,)   risk parameters
        max_leverage : float             evolved max leverage
        stop_width   : float             evolved stop loss width
        position_scale: float            evolved position sizing
        take_profit_mult: float          evolved take profit multiplier
    }
    """
    weights        = chrom[0:12]
    thresholds     = chrom[12:24]
    leverage_genes = chrom[24:28]
    risk_genes     = chrom[28:32]

    # Interpret risk genes
    max_leverage     = 1.0 + leverage_genes.mean() * 9.0   # 1x to 10x
    stop_width       = 0.01 + risk_genes[0] * 0.09         # 1% to 10%
    position_scale   = 0.05 + risk_genes[1] * 0.35         # 5% to 40%
    take_profit_mult = 1.5  + risk_genes[2] * 3.5          # 1.5x to 5x TP

    return {
        "weights"         : weights,
        "thresholds"      : thresholds,
        "leverage_genes"  : leverage_genes,
        "risk_genes"      : risk_genes,
        "max_leverage"    : round(max_leverage, 2),
        "stop_width"      : round(stop_width, 4),
        "position_scale"  : round(position_scale, 4),
        "take_profit_mult": round(take_profit_mult, 2),
    }


def compute_futures_signal(df_scaled: pd.DataFrame,
                            chrom: np.ndarray,
                            config: dict) -> tuple[int, float, float]:
    """
    Generate signal + confidence + leverage for a futures position.

    Returns
    -------
    signal     : 1 (long), -1 (short), 0 (flat)
    confidence : float in [0,1]
    leverage   : float (evolved, clipped to max_leverage)
    """
    decoded = decode_futures_chromosome(chrom)
    weights    = decoded["weights"]
    thresholds = decoded["thresholds"]

    values    = df_scaled[FEATURES].iloc[-1].values
    condition = (values > thresholds).astype(float)

    w_sum      = weights.sum()
    if w_sum == 0:
        return 0, 0.0, 1.0

    confidence  = float((condition * weights).sum() / w_sum)
    score       = confidence

    # Signal
    if score > 0.55:
        signal = 1    # long
    elif score < 0.35:
        signal = -1   # short
    else:
        signal = 0    # flat / too uncertain

    # GA-evolved leverage -- scales with confidence
    max_lev  = min(decoded["max_leverage"], config.get("max_leverage", 10.0))
    leverage = 1.0 + confidence * (max_lev - 1.0) * decoded["leverage_genes"].mean()
    leverage = float(np.clip(leverage, 1.0, max_lev))

    # Low confidence cap -- don't use leverage if not confident
    if confidence < 0.45:
        leverage = 1.0

    return signal, round(confidence, 4), round(leverage, 2)


# ------------------------------------------------------------------------------
# FUTURES FITNESS FUNCTION
# ------------------------------------------------------------------------------

def futures_fitness(chrom: np.ndarray,
                    df_scaled: pd.DataFrame,
                    df_raw: pd.DataFrame,
                    config: dict) -> float:
    """
    Vectorised futures fitness -- computes all signals at once then simulates.
    Much faster than calling compute_futures_signal per row.
    """
    decoded  = decode_futures_chromosome(chrom)
    max_lev  = min(decoded["max_leverage"], config.get("max_leverage", 10.0))
    stop_w   = decoded["stop_width"]

    weights    = decoded["weights"]
    thresholds = decoded["thresholds"]
    lev_gene   = decoded["leverage_genes"].mean()

    # Vectorised signal computation
    values    = df_scaled[FEATURES].values          # (n, 12)
    condition = (values > thresholds).astype(float) # (n, 12)
    w_sum     = weights.sum()
    if w_sum == 0:
        return -1.0

    confidence = (condition * weights).sum(axis=1) / w_sum  # (n,)
    signals    = np.where(confidence > 0.55, 1,
                 np.where(confidence < 0.35, -1, 0))

    # Leverage per bar
    leverage = 1.0 + confidence * (max_lev - 1.0) * lev_gene
    leverage = np.clip(leverage, 1.0, max_lev)
    leverage[confidence < 0.45] = 1.0

    close  = df_raw['Close'].values
    n      = len(close)
    trades = []
    equity = [1.0]

    in_pos    = False
    entry     = 0.0
    direction = 1
    lev       = 1.0

    for i in range(min(n-1, len(signals)-1)):
        sig = int(signals[i])
        lv  = float(leverage[i])

        if not in_pos and sig != 0:
            in_pos    = True
            entry     = close[i]
            direction = sig
            lev       = lv
        elif in_pos and (sig == 0 or sig != direction):
            raw_ret = (close[i] - entry) / entry * direction
            lev_ret = raw_ret * lev
            if lev_ret < -stop_w * lev:
                lev_ret = -stop_w * lev
            trades.append(lev_ret)
            equity.append(equity[-1] * (1 + lev_ret))
            in_pos = False
        elif in_pos and i > 0:
            daily = (close[i] / close[i-1] - 1) * direction * lev
            equity.append(equity[-1] * (1 + daily))

    if in_pos:
        ret = (close[-1] - entry) / entry * direction * lev
        trades.append(ret)
        equity.append(equity[-1] * (1 + ret))

    equity = np.array(equity)
    if len(trades) < config.get("min_trades", 8):
        return -1.0

    total_return = equity[-1] - 1.0
    wins   = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]
    win_rate = len(wins) / len(trades) if trades else 0.0

    peak   = np.maximum.accumulate(equity)
    max_dd = ((peak - equity) / peak).max() if len(equity) > 1 else 0.0

    avg_win  = np.mean(wins)          if wins   else 0.0
    avg_loss = abs(np.mean(losses))   if losses else 1.0
    lev_eff  = min(avg_win / max(avg_loss, 0.001), 3.0) / 3.0

    alpha = config.get("fitness_alpha", 0.45)
    beta  = config.get("fitness_beta",  0.35)
    gamma = config.get("fitness_gamma", 0.20)

    norm_return = np.clip((total_return + 1.0) / 2.0, 0.0, 1.0)
    score = alpha * norm_return + beta * win_rate + gamma * lev_eff

    if max_dd > config.get("max_drawdown_thresh", 0.30):
        score *= config.get("drawdown_penalty", 0.35)

    return float(score)


# ------------------------------------------------------------------------------
# FUTURES GA TRAINING
# ------------------------------------------------------------------------------

def train_futures_bot(ticker: str, df_scaled: pd.DataFrame,
                      df_raw: pd.DataFrame) -> dict:
    """Run the GA to evolve a futures chromosome."""
    if ticker not in FUTURES_CONFIGS:
        raise ValueError(f"Unknown futures ticker: {ticker}")

    cfg_key = FUTURES_CONFIGS[ticker]["volatility_band"]
    ga_cfg  = FUTURES_GA_CONFIG[cfg_key]
    f_cfg   = FUTURES_CONFIGS[ticker]

    rng     = np.random.default_rng(ga_cfg["random_seed"])
    pop_sz  = ga_cfg["population_size"]
    n_gens  = ga_cfg["generations"]
    elite_n = ga_cfg["elite_count"]
    tourn_k = ga_cfg["tournament_size"]
    cx_rate = ga_cfg["crossover_rate"]
    mut_r   = ga_cfg["mutation_rate_init"]
    mut_s   = ga_cfg["mutation_step"]
    stag_w  = ga_cfg["stagnation_window"]

    combined_config = {**ga_cfg, **f_cfg}

    population = rng.uniform(0, 1, (pop_sz, FUTURES_CHROM_LENGTH))
    fitnesses  = np.array([
        futures_fitness(c, df_scaled, df_raw, combined_config)
        for c in population
    ])

    best_fit   = fitnesses.max()
    best_chrom = population[fitnesses.argmax()].copy()
    stag_count = 0

    print(f"\n{'='*60}")
    print(f"  FUTURES GA -- {ticker} ({f_cfg['name']})")
    print(f"  Max leverage: {f_cfg['max_leverage']}x  |  {n_gens} generations")
    print(f"{'='*60}")
    print(f"{'Gen':>5} | {'Best Fit':>10} | {'Leverage':>9} | {'Stop':>7} | {'TP Mult':>8}")
    print(f"{'-'*55}")

    for gen in range(1, n_gens + 1):
        elite_idx = np.argsort(fitnesses)[-elite_n:]
        elites    = population[elite_idx].copy()
        next_pop  = [elites]

        while sum(len(g) for g in next_pop) < pop_sz:
            # Tournament selection
            p1_idx = rng.choice(pop_sz, tourn_k, replace=False)
            p2_idx = rng.choice(pop_sz, tourn_k, replace=False)
            p1 = population[p1_idx[fitnesses[p1_idx].argmax()]].copy()
            p2 = population[p2_idx[fitnesses[p2_idx].argmax()]].copy()

            # Crossover
            if rng.random() < cx_rate:
                mask = rng.random(FUTURES_CHROM_LENGTH) < 0.5
                c1   = np.where(mask, p1, p2)
                c2   = np.where(mask, p2, p1)
            else:
                c1, c2 = p1.copy(), p2.copy()

            # Mutation
            for child in [c1, c2]:
                m = rng.random(FUTURES_CHROM_LENGTH) < mut_r
                child[m] = np.clip(child[m] + rng.normal(0, mut_s, m.sum()), 0, 1)

            next_pop.append(np.array([c1, c2]))

        population = np.vstack(next_pop)[:pop_sz]
        fitnesses  = np.array([
            futures_fitness(c, df_scaled, df_raw, combined_config)
            for c in population
        ])

        gen_best = fitnesses.max()
        if gen_best > best_fit:
            best_fit   = gen_best
            best_chrom = population[fitnesses.argmax()].copy()
            stag_count = 0
        else:
            stag_count += 1

        # Adaptive mutation
        if stag_count >= stag_w:
            mut_r = min(mut_r * 1.5, ga_cfg["mutation_rate_max"])
            stag_count = 0
        else:
            mut_r = max(mut_r * 0.95, ga_cfg["mutation_rate_min"])

        if gen % 50 == 0 or gen == 1:
            decoded = decode_futures_chromosome(best_chrom)
            print(f"{gen:>5} | {best_fit:>10.4f} | "
                  f"{decoded['max_leverage']:>8.1f}x | "
                  f"{decoded['stop_width']*100:>6.1f}% | "
                  f"{decoded['take_profit_mult']:>7.1f}x")

    decoded = decode_futures_chromosome(best_chrom)
    print(f"\n{'='*60}")
    print(f"  FUTURES TRAINING COMPLETE -- {ticker}")
    print(f"  Best fitness    : {best_fit:.4f}")
    print(f"  Evolved leverage: {decoded['max_leverage']:.1f}x")
    print(f"  Stop width      : {decoded['stop_width']*100:.1f}%")
    print(f"  TP multiplier   : {decoded['take_profit_mult']:.1f}x")
    print(f"  Position scale  : {decoded['position_scale']*100:.0f}% of portfolio")
    print(f"{'='*60}\n")

    return {
        "chromosome"  : best_chrom,
        "fitness"     : best_fit,
        "decoded"     : decoded,
        "config"      : combined_config,
    }


def save_futures_chromosome(result: dict, ticker: str) -> None:
    """Save futures chromosome to CSV."""
    import pandas as pd
    decoded = result["decoded"]
    chrom   = result["chromosome"]

    rows = []
    for i, feat in enumerate(FEATURES):
        rows.append({"gene_type": "weight",    "feature": feat, "value": chrom[i]})
    for i, feat in enumerate(FEATURES):
        rows.append({"gene_type": "threshold", "feature": feat, "value": chrom[12+i]})
    for i in range(4):
        rows.append({"gene_type": "leverage",  "feature": f"lev_{i}", "value": chrom[24+i]})
    for i in range(4):
        rows.append({"gene_type": "risk",      "feature": f"risk_{i}", "value": chrom[28+i]})

    safe_ticker = ticker.replace("/", "-")
    path = Path(FUTURES_CONFIGS[ticker]["chromosome_file"])
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"Saved: {path}")


def load_futures_chromosome(ticker: str) -> np.ndarray:
    """Load a futures chromosome from CSV."""
    import pandas as pd
    path = Path(FUTURES_CONFIGS[ticker]["chromosome_file"])
    if not path.exists():
        raise FileNotFoundError(f"No futures chromosome for {ticker}. Train first.")
    df    = pd.read_csv(path)
    chrom = df["value"].values
    if len(chrom) != FUTURES_CHROM_LENGTH:
        raise ValueError(f"Expected {FUTURES_CHROM_LENGTH} genes, got {len(chrom)}")
    return chrom


# ------------------------------------------------------------------------------
# RISK GUARD FOR FUTURES
# ------------------------------------------------------------------------------

class FuturesRiskGuard:
    """
    Hard limits specifically for leveraged futures trading.
    Much stricter than spot trading limits.
    """
    MAX_LEVERAGE         = 10.0   # never exceed this regardless of GA
    MAX_POSITION_PCT     = 0.15   # max 15% of portfolio per futures trade
    MAX_DAILY_LOSS_PCT   = 0.08   # halt if down 8% in one day (futures move fast)
    MAX_DRAWDOWN_PCT     = 0.25   # halt if total drawdown exceeds 25%
    CONFIDENCE_THRESHOLD = 0.50   # higher threshold for futures (need more conviction)
    LOSS_STREAK_HALT     = 3      # halt after 3 consecutive losses

    def __init__(self):
        self.daily_loss    = 0.0
        self.loss_streak   = 0
        self.halted        = False

    def check(self, confidence: float, leverage: float,
              portfolio_value: float) -> tuple[bool, str]:
        if self.halted:
            return False, "Futures trading halted -- too many losses"
        if confidence < self.CONFIDENCE_THRESHOLD:
            return False, f"Confidence {confidence:.2f} below futures threshold {self.CONFIDENCE_THRESHOLD}"
        if leverage > self.MAX_LEVERAGE:
            leverage = self.MAX_LEVERAGE
        if self.daily_loss <= -self.MAX_DAILY_LOSS_PCT:
            self.halted = True
            return False, f"Daily loss limit hit: {self.daily_loss:.1%}"
        return True, "OK"

    def record_trade(self, pnl_pct: float) -> None:
        self.daily_loss += min(pnl_pct, 0)
        if pnl_pct < 0:
            self.loss_streak += 1
            if self.loss_streak >= self.LOSS_STREAK_HALT:
                self.halted = True
                logging.warning(f"Futures halted after {self.loss_streak} consecutive losses")
        else:
            self.loss_streak = 0

    def reset_daily(self) -> None:
        self.daily_loss = 0.0
        if self.halted and self.loss_streak < self.LOSS_STREAK_HALT:
            self.halted = False


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Futures trading bot")
    parser.add_argument("--ticker", help="Futures ticker (BTC/USD, ETH/USD, ES, NQ, GC, CL)")
    parser.add_argument("--train",  action="store_true", help="Train chromosome")
    parser.add_argument("--list",   action="store_true", help="List available futures")
    parser.add_argument("--paper",  action="store_true", help="Paper trading mode")
    args = parser.parse_args()

    if args.list:
        print("\nAvailable futures:")
        print(f"{'Ticker':<12} {'Name':<30} {'Type':<12} {'Max Lev':<10} {'Hours'}")
        print("-" * 75)
        for ticker, cfg in FUTURES_CONFIGS.items():
            exists = "[trained]" if Path(cfg["chromosome_file"]).exists() else "[untrained]"
            print(f"{ticker:<12} {cfg['name']:<30} {cfg['type']:<12} "
                  f"{cfg['max_leverage']:<10.0f}x {cfg['trading_hours']} {exists}")
        return

    if not args.ticker:
        parser.print_help()
        return

    ticker = args.ticker.upper()
    if ticker not in FUTURES_CONFIGS:
        print(f"Unknown ticker: {ticker}")
        print(f"Available: {list(FUTURES_CONFIGS.keys())}")
        return

    if args.train:
        from stock_data import download_stock_data, add_indicators, preprocess_data
        cfg = FUTURES_CONFIGS[ticker]
        print(f"Downloading {cfg['yf_symbol']} data...")
        df_raw    = download_stock_data(cfg["yf_symbol"], "2020-01-01",
                                        datetime.now().strftime("%Y-%m-%d"))
        df_ind    = add_indicators(df_raw.copy())
        df_scaled, _ = preprocess_data(df_ind)
        idx       = df_scaled.index.intersection(df_raw.index)
        df_scaled = df_scaled.loc[idx]
        df_raw    = df_raw.loc[idx]

        result = train_futures_bot(ticker, df_scaled, df_raw)
        save_futures_chromosome(result, ticker)
        print(f"Training complete. Run the bot with: python futures_bot.py --ticker {ticker}")
    else:
        print(f"Starting futures bot for {ticker}...")
        print("(Live trading loop coming in next build)")


if __name__ == "__main__":
    main()

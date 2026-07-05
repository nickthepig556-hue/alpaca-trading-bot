"""
Step 3 - Genetic Algorithm for Trading Strategy Optimisation
=============================================================
Chromosome encoding : 12 weights  +  12 thresholds  =  24 genes  (all in [0, 1])
Fitness function    : composite score = 0.6 * total_return + 0.4 * win_rate
                      with a max-drawdown penalty applied
GA operators        : tournament selection, uniform crossover,
                      adaptive mutation, elitism
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# ------------------------------------------------------------------------------
# CHROMOSOME STRUCTURE
# ------------------------------------------------------------------------------
# The 12 features from stock_data.py -- order must match df_scaled columns
FEATURES = [
    'Close', 'Volume',
    'SMA_20', 'SMA_50', 'SMA_200',
    'RSI', 'MACD', 'Signal',
    'BB_Upper', 'BB_Lower',
    'Daily_Return', 'Volume_Change',
]
N_FEATURES   = len(FEATURES)          # 12
CHROM_LENGTH = N_FEATURES * 2         # 24  (12 weights + 12 thresholds)

# Gene layout inside a chromosome (all values in [0, 1]):
#   genes[0:12]  -> feature weights   (how much each indicator contributes)
#   genes[12:24] -> buy thresholds    (weighted-sum score must exceed this per feature)


# ------------------------------------------------------------------------------
# GA HYPER-PARAMETERS
# ------------------------------------------------------------------------------
GA_CONFIG = {
    'population_size'    : 100,
    'generations'        : 200,
    'elite_count'        : 5,       # top N chromosomes copied unchanged each gen
    'tournament_size'    : 5,       # k individuals compete per selection event
    'crossover_rate'     : 0.85,
    'mutation_rate_init' : 0.15,    # starting mutation rate (adaptive)
    'mutation_rate_min'  : 0.01,    # floor -- never mutate less than this
    'mutation_rate_max'  : 0.30,    # ceiling
    'mutation_step'      : 0.15,    # Gaussian std for gene perturbation
    'stagnation_window'  : 20,      # gens without improvement -> boost mutation
    'fitness_alpha'      : 0.60,    # weight on total_return in composite score
    'fitness_beta'       : 0.40,    # weight on win_rate
    'drawdown_penalty'   : 0.50,    # multiplier applied when drawdown > threshold
    'max_drawdown_thresh': 0.20,    # 20 % drawdown triggers penalty
    'min_trades'         : 10,      # chromosomes generating fewer trades -> penalised
    'random_seed'        : 42,
}


# ------------------------------------------------------------------------------
# SIGNAL GENERATION
# ------------------------------------------------------------------------------

def decode_chromosome(chrom: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a flat chromosome into weights and thresholds."""
    weights    = chrom[:N_FEATURES]
    thresholds = chrom[N_FEATURES:]
    return weights, thresholds


def generate_signals(df_scaled: pd.DataFrame, chrom: np.ndarray) -> pd.Series:
    """
    For each trading day compute a weighted score across all features.
    BUY  signal (1) : feature_value > threshold  for that feature, weighted sum > 0.5
    SELL signal (-1): weighted sum <= 0.5
    HOLD signal (0) : emitted if fewer than min_trades would be generated (handled in fitness)

    Logic:
        score_i = weight_i  if  scaled_value_i > threshold_i  else  0
        total_score = sum(score_i) / sum(weights)   ∈ [0, 1]
        signal = 1 if total_score > 0.5 else -1
    """
    weights, thresholds = decode_chromosome(chrom)

    values = df_scaled[FEATURES].values          # shape (n_days, 12)
    w      = weights.reshape(1, -1)              # (1, 12)
    t      = thresholds.reshape(1, -1)           # (1, 12)

    # Condition matrix: 1 where scaled value exceeds threshold, 0 otherwise
    condition = (values > t).astype(float)       # (n_days, 12)

    weight_sum = w.sum()
    if weight_sum == 0:
        return pd.Series(np.full(len(df_scaled), -1), index=df_scaled.index)

    score = (condition * w).sum(axis=1) / weight_sum   # (n_days,)
    signals = np.where(score > 0.5, 1, -1)
    return pd.Series(signals, index=df_scaled.index)


# ------------------------------------------------------------------------------
# FITNESS FUNCTION
# ------------------------------------------------------------------------------

def simulate_trades(signals: pd.Series, df_raw: pd.DataFrame) -> dict:
    """
    Simple long-only backtest:
      • Enter long on BUY signal, exit on SELL signal.
      • One position at a time; no leverage; no transaction costs (add later).
    Returns a dict of performance metrics.
    """
    close       = df_raw['Close'].values
    sig         = signals.values
    n           = len(sig)

    in_position = False
    entry_price = 0.0
    trades      = []          # list of (return_pct,)
    equity      = [1.0]       # normalised equity curve

    for i in range(n - 1):
        if not in_position and sig[i] == 1:
            in_position = True
            entry_price = close[i]
        elif in_position and sig[i] == -1:
            ret = (close[i] - entry_price) / entry_price
            trades.append(ret)
            in_position = False
            equity.append(equity[-1] * (1 + ret))
        else:
            if in_position:
                equity.append(equity[-1] * (close[i] / close[i - 1]))
            else:
                equity.append(equity[-1])

    # Close any open position at end
    if in_position:
        ret = (close[-1] - entry_price) / entry_price
        trades.append(ret)
        equity.append(equity[-1] * (1 + ret))

    equity = np.array(equity)

    total_return = equity[-1] - 1.0
    n_trades     = len(trades)
    win_rate     = (np.array(trades) > 0).mean() if n_trades > 0 else 0.0

    # Max drawdown
    peak        = np.maximum.accumulate(equity)
    drawdown    = (peak - equity) / peak
    max_dd      = drawdown.max() if len(drawdown) > 0 else 0.0

    return {
        'total_return': total_return,
        'win_rate'    : win_rate,
        'n_trades'    : n_trades,
        'max_drawdown': max_dd,
        'equity_curve': equity,
    }


def fitness(chrom: np.ndarray,
            df_scaled: pd.DataFrame,
            df_raw: pd.DataFrame,
            config: dict) -> float:
    """
    Composite fitness:
        score = alpha * norm_return + beta * win_rate
        penalised if max_drawdown > threshold or n_trades < min_trades
    """
    signals = generate_signals(df_scaled, chrom)
    stats   = simulate_trades(signals, df_raw)

    n_trades     = stats['n_trades']
    total_return = stats['total_return']
    win_rate     = stats['win_rate']
    max_dd       = stats['max_drawdown']

    # Penalise chromosomes that barely trade
    if n_trades < config['min_trades']:
        return -1.0

    # Normalise total_return to [0,1] range (clamp at ±100 %)
    norm_return = np.clip((total_return + 1.0) / 2.0, 0.0, 1.0)

    score = (config['fitness_alpha'] * norm_return
             + config['fitness_beta'] * win_rate)

    # Drawdown penalty
    if max_dd > config['max_drawdown_thresh']:
        score *= config['drawdown_penalty']

    return float(score)


# ------------------------------------------------------------------------------
# GA OPERATORS
# ------------------------------------------------------------------------------

def init_population(pop_size: int, chrom_len: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random initialisation in [0, 1]."""
    return rng.uniform(0.0, 1.0, size=(pop_size, chrom_len))


def tournament_selection(population: np.ndarray,
                         fitnesses: np.ndarray,
                         k: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Select one parent via k-way tournament."""
    idx       = rng.choice(len(population), size=k, replace=False)
    best_idx  = idx[np.argmax(fitnesses[idx])]
    return population[best_idx].copy()


def uniform_crossover(parent1: np.ndarray,
                      parent2: np.ndarray,
                      rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Each gene independently drawn from either parent with p=0.5."""
    mask   = rng.random(len(parent1)) < 0.5
    child1 = np.where(mask, parent1, parent2)
    child2 = np.where(mask, parent2, parent1)
    return child1, child2


def adaptive_mutate(chrom: np.ndarray,
                    mutation_rate: float,
                    mutation_step: float,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Gaussian perturbation on each gene with probability mutation_rate.
    Values clipped back to [0, 1] after mutation.
    """
    mutant = chrom.copy()
    mask   = rng.random(len(chrom)) < mutation_rate
    noise  = rng.normal(0, mutation_step, size=len(chrom))
    mutant = np.where(mask, mutant + noise, mutant)
    return np.clip(mutant, 0.0, 1.0)


def adapt_mutation_rate(current_rate: float,
                        stagnated: bool,
                        config: dict) -> float:
    """
    Increase mutation rate when the population is stagnating,
    decrease it when progress is being made.
    """
    if stagnated:
        new_rate = current_rate * 1.5
    else:
        new_rate = current_rate * 0.95
    return float(np.clip(new_rate,
                         config['mutation_rate_min'],
                         config['mutation_rate_max']))


# ------------------------------------------------------------------------------
# MAIN GA LOOP
# ------------------------------------------------------------------------------

def run_ga(df_scaled: pd.DataFrame,
           df_raw: pd.DataFrame,
           config: dict = GA_CONFIG) -> dict:
    """
    Run the full genetic algorithm.

    Returns
    -------
    dict with keys:
        best_chromosome   : np.ndarray  shape (24,)
        best_fitness      : float
        best_stats        : dict  (return, win_rate, drawdown, n_trades)
        fitness_history   : list of best fitness per generation
        avg_history       : list of mean  fitness per generation
        mutation_history  : list of mutation_rate per generation
        population        : final population
        fitnesses         : final fitness array
    """
    rng = np.random.default_rng(config['random_seed'])

    pop_size   = config['population_size']
    n_gens     = config['generations']
    elite_n    = config['elite_count']
    tourn_k    = config['tournament_size']
    cx_rate    = config['crossover_rate']
    mut_rate   = config['mutation_rate_init']
    mut_step   = config['mutation_step']
    stag_win   = config['stagnation_window']

    # -- Initialise ----------------------------------------------------------
    population = init_population(pop_size, CHROM_LENGTH, rng)
    fitnesses  = np.array([fitness(c, df_scaled, df_raw, config) for c in population])

    best_idx       = np.argmax(fitnesses)
    best_chrom     = population[best_idx].copy()
    best_fit       = fitnesses[best_idx]

    fitness_history  = [best_fit]
    avg_history      = [fitnesses.mean()]
    mutation_history = [mut_rate]

    print(f"\n{'='*60}")
    print(f"  GENETIC ALGORITHM -- {n_gens} generations  |  pop={pop_size}")
    print(f"{'='*60}")
    print(f"{'Gen':>5} | {'Best Fitness':>12} | {'Avg Fitness':>11} | "
          f"{'Mut Rate':>8} | {'Best Return':>11} | {'Win Rate':>8}")
    print(f"{'-'*70}")

    stagnation_counter = 0

    for gen in range(1, n_gens + 1):

        # -- Elitism: carry top chromosomes unchanged ----------------------
        elite_idx  = np.argsort(fitnesses)[-elite_n:]
        elites     = population[elite_idx].copy()

        # -- Build next generation -----------------------------------------
        next_pop = [elites]   # start with elites

        while sum(len(g) for g in next_pop) < pop_size:
            p1 = tournament_selection(population, fitnesses, tourn_k, rng)
            p2 = tournament_selection(population, fitnesses, tourn_k, rng)

            if rng.random() < cx_rate:
                c1, c2 = uniform_crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()

            c1 = adaptive_mutate(c1, mut_rate, mut_step, rng)
            c2 = adaptive_mutate(c2, mut_rate, mut_step, rng)
            next_pop.append(np.array([c1, c2]))

        population = np.vstack(next_pop)[:pop_size]
        fitnesses  = np.array([fitness(c, df_scaled, df_raw, config) for c in population])

        # -- Track best ----------------------------------------------------
        gen_best_idx = np.argmax(fitnesses)
        gen_best_fit = fitnesses[gen_best_idx]

        if gen_best_fit > best_fit:
            best_fit   = gen_best_fit
            best_chrom = population[gen_best_idx].copy()
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        # -- Adaptive mutation ---------------------------------------------
        stagnated = stagnation_counter >= stag_win
        mut_rate  = adapt_mutation_rate(mut_rate, stagnated, config)
        if stagnated:
            stagnation_counter = 0   # reset after boosting

        fitness_history.append(best_fit)
        avg_history.append(fitnesses.mean())
        mutation_history.append(mut_rate)

        # -- Progress print every 10 gens ---------------------------------
        if gen % 10 == 0 or gen == 1:
            sigs  = generate_signals(df_scaled, best_chrom)
            stats = simulate_trades(sigs, df_raw)
            print(f"{gen:>5} | {best_fit:>12.4f} | {fitnesses.mean():>11.4f} | "
                  f"{mut_rate:>8.4f} | {stats['total_return']:>+10.1%} | "
                  f"{stats['win_rate']:>7.1%}")

    # -- Final stats on best chromosome ---------------------------------------
    best_signals = generate_signals(df_scaled, best_chrom)
    best_stats   = simulate_trades(best_signals, df_raw)

    print(f"\n{'='*60}")
    print(f"  FINAL RESULT")
    print(f"{'='*60}")
    print(f"  Best fitness score : {best_fit:.4f}")
    print(f"  Total return       : {best_stats['total_return']:+.2%}")
    print(f"  Win rate           : {best_stats['win_rate']:.2%}")
    print(f"  Number of trades   : {best_stats['n_trades']}")
    print(f"  Max drawdown       : {best_stats['max_drawdown']:.2%}")
    print(f"\n  Chromosome genes:")
    weights, thresholds = decode_chromosome(best_chrom)
    print(f"\n  {'Feature':<16} {'Weight':>8}  {'Threshold':>10}")
    print(f"  {'-'*38}")
    for feat, w, t in zip(FEATURES, weights, thresholds):
        print(f"  {feat:<16} {w:>8.4f}  {t:>10.4f}")
    print(f"{'='*60}\n")

    return {
        'best_chromosome' : best_chrom,
        'best_fitness'    : best_fit,
        'best_stats'      : best_stats,
        'best_signals'    : best_signals,
        'fitness_history' : fitness_history,
        'avg_history'     : avg_history,
        'mutation_history': mutation_history,
        'population'      : population,
        'fitnesses'       : fitnesses,
    }


# ------------------------------------------------------------------------------
# RESULTS ANALYSIS HELPERS
# ------------------------------------------------------------------------------

def top_n_chromosomes(result: dict,
                      df_scaled: pd.DataFrame,
                      df_raw: pd.DataFrame,
                      n: int = 5) -> pd.DataFrame:
    """Return a summary DataFrame of the top-N chromosomes in the final population."""
    pop  = result['population']
    fits = result['fitnesses']
    idx  = np.argsort(fits)[-n:][::-1]
    rows = []
    for rank, i in enumerate(idx, 1):
        sigs  = generate_signals(df_scaled, pop[i])
        stats = simulate_trades(sigs, df_raw)
        rows.append({
            'rank'        : rank,
            'fitness'     : fits[i],
            'total_return': stats['total_return'],
            'win_rate'    : stats['win_rate'],
            'n_trades'    : stats['n_trades'],
            'max_drawdown': stats['max_drawdown'],
        })
    return pd.DataFrame(rows)


def save_results(result: dict, ticker: str = 'GLD') -> None:
    """Persist the best chromosome and its signals to CSV."""
    # Best chromosome
    weights, thresholds = decode_chromosome(result['best_chromosome'])
    chrom_df = pd.DataFrame({
        'feature'  : FEATURES,
        'weight'   : weights,
        'threshold': thresholds,
    })
    chrom_df.to_csv(f"{ticker}_best_chromosome.csv", index=False)

    # Signals
    result['best_signals'].to_csv(f"{ticker}_signals.csv", header=['signal'])

    # Fitness history
    hist_df = pd.DataFrame({
        'generation'   : range(len(result['fitness_history'])),
        'best_fitness' : result['fitness_history'],
        'avg_fitness'  : result['avg_history'],
        'mutation_rate': result['mutation_history'],
    })
    hist_df.to_csv(f"{ticker}_fitness_history.csv", index=False)

    print(f"[OK] Results saved: {ticker}_best_chromosome.csv  |  "
          f"{ticker}_signals.csv  |  {ticker}_fitness_history.csv")


# ------------------------------------------------------------------------------
# ENTRY POINT  (runs standalone; imported by main pipeline too)
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # -- Load preprocessed data --------------------------------------------
    # Expects GLD_scaled.csv and GLD_raw.csv produced by stock_data.py
    import os

    TICKER = "GLD"
    scaled_path = f"{TICKER}_scaled.csv"
    raw_path    = f"{TICKER}_raw.csv"

    if not os.path.exists(scaled_path) or not os.path.exists(raw_path):
        raise FileNotFoundError(
            f"Missing {scaled_path} or {raw_path}. "
            "Run stock_data.py first to generate these files."
        )

    df_scaled = pd.read_csv(scaled_path, index_col='Date', parse_dates=True)
    df_raw    = pd.read_csv(raw_path,    index_col='Date', parse_dates=True)

    # Align indices (both should already match, but be safe)
    idx       = df_scaled.index.intersection(df_raw.index)
    df_scaled = df_scaled.loc[idx]
    df_raw    = df_raw.loc[idx]

    print(f"Loaded {len(df_scaled)} rows  |  "
          f"{df_scaled.index[0].date()} -> {df_scaled.index[-1].date()}")

    # -- Run GA ------------------------------------------------------------
    result = run_ga(df_scaled, df_raw, GA_CONFIG)

    # -- Top-5 chromosomes in final population -----------------------------
    print("\n📊 Top-5 chromosomes in final population:")
    print(top_n_chromosomes(result, df_scaled, df_raw, n=5).to_string(index=False))

    # -- Save --------------------------------------------------------------
    save_results(result, TICKER)

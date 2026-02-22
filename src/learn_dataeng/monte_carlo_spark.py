"""
Monte Carlo Portfolio Risk Simulation using PySpark.

Generates ~500 MB of synthetic financial instrument data, then runs a
large-scale Monte Carlo simulation to estimate portfolio value-at-risk (VaR),
expected shortfall (CVaR), and return distributions across thousands of
correlated market scenarios.

Usage:
    uv run monte-carlo-spark
"""

from __future__ import annotations

import math
import os
import shutil
import time
from pathlib import Path

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_INSTRUMENTS = 2_000          # financial instruments (stocks / bonds / derivs)
NUM_SCENARIOS = 50_000           # Monte Carlo market scenarios
NUM_TIME_STEPS = 252             # trading days in a year
OUTPUT_DIR = Path("monte_carlo_output")
PARQUET_PATH = OUTPUT_DIR / "simulation_results.parquet"
SEED = 42


def _spark_session() -> SparkSession:
    """Create a local Spark session tuned for a beefy single-node run."""
    # Suppress noisy Hadoop/ViewFileSystem warnings caused by JDK 25
    # removing Subject.getSubject() which Hadoop still calls internally.
    import logging
    logging.getLogger("org.apache.hadoop").setLevel(logging.ERROR)

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName("MonteCarloPortfolioRisk")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "200")
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.ui.showConsoleProgress", "true")
        .getOrCreate()
    )

    # Set JVM-side log levels to suppress Hadoop FS and SharedState warnings
    spark.sparkContext.setLogLevel("ERROR")
    return spark


# ---------------------------------------------------------------------------
# Step 1 – Generate ~500 MB of synthetic instrument data
# ---------------------------------------------------------------------------

def generate_instruments(spark: SparkSession) -> DataFrame:
    """
    Create a DataFrame of financial instruments with realistic attributes.

    Each row is one instrument with: id, sector, asset_class, weight,
    annual_return, annual_volatility, dividend_yield, beta, credit_rating_score.

    Target: ~500 MB when cross-joined with scenarios.
    """
    print("\n=== Step 1: Generating synthetic instrument universe ===")
    t0 = time.time()

    sectors = [
        "Technology", "Healthcare", "Financials", "Energy", "Consumer",
        "Industrials", "Materials", "Utilities", "RealEstate", "Telecom",
    ]
    asset_classes = ["Equity", "FixedIncome", "Derivative", "Commodity", "FX"]

    # Build rows in Python, then parallelize
    rows = []
    import random
    rng = random.Random(SEED)
    for i in range(NUM_INSTRUMENTS):
        rows.append((
            i,
            f"INST-{i:06d}",
            rng.choice(sectors),
            rng.choice(asset_classes),
            round(rng.uniform(0.0001, 0.01), 6),       # portfolio weight (will normalize later)
            round(rng.gauss(0.08, 0.12), 6),            # expected annual return
            round(abs(rng.gauss(0.20, 0.10)), 6),       # annual volatility
            round(max(0.0, rng.gauss(0.02, 0.015)), 6),  # dividend yield
            round(rng.gauss(1.0, 0.4), 4),              # beta
            rng.randint(1, 10),                          # credit rating score (1=AAA … 10=D)
        ))

    schema = T.StructType([
        T.StructField("instrument_id", T.IntegerType()),
        T.StructField("ticker", T.StringType()),
        T.StructField("sector", T.StringType()),
        T.StructField("asset_class", T.StringType()),
        T.StructField("weight", T.DoubleType()),
        T.StructField("annual_return", T.DoubleType()),
        T.StructField("annual_volatility", T.DoubleType()),
        T.StructField("dividend_yield", T.DoubleType()),
        T.StructField("beta", T.DoubleType()),
        T.StructField("credit_rating_score", T.IntegerType()),
    ])

    instruments = spark.createDataFrame(rows, schema=schema)

    # Normalize weights so they sum to 1.0
    total_w = instruments.agg(F.sum("weight")).collect()[0][0]
    instruments = instruments.withColumn("weight", F.col("weight") / F.lit(total_w))

    instruments.cache()
    cnt = instruments.count()
    elapsed = time.time() - t0
    print(f"    Created {cnt:,} instruments in {elapsed:.1f}s")
    return instruments


# ---------------------------------------------------------------------------
# Step 2 – Generate massive scenario matrix (~500 MB)
# ---------------------------------------------------------------------------

def generate_scenarios(spark: SparkSession) -> DataFrame:
    """
    Create NUM_SCENARIOS × NUM_TIME_STEPS rows of random market shocks.

    Each row: (scenario_id, time_step, market_shock, rate_shock, vol_shock,
               sector_shock_idx, tail_event_flag).

    This is the big data: 50 000 × 252 ≈ 12.6 M rows with several double
    columns — roughly 500 MB in memory.
    """
    print("\n=== Step 2: Generating Monte Carlo scenario matrix ===")
    t0 = time.time()

    # Generate scenario IDs as a range DataFrame (very fast)
    scenario_ids = spark.range(NUM_SCENARIOS).withColumnRenamed("id", "scenario_id")

    # Generate time steps
    time_steps = spark.range(NUM_TIME_STEPS).withColumnRenamed("id", "time_step")

    # Cross join: every scenario × every time step
    scenarios = scenario_ids.crossJoin(time_steps)

    # Add random shocks using Spark-native random functions (seeded)
    scenarios = (
        scenarios
        .withColumn("market_shock", F.randn(seed=SEED) * 0.02)         # daily market return shock
        .withColumn("rate_shock", F.randn(seed=SEED + 1) * 0.005)      # interest rate shock
        .withColumn("vol_shock", F.abs(F.randn(seed=SEED + 2)) * 0.03) # volatility regime shock
        .withColumn("sector_shock_idx", (F.rand(seed=SEED + 3) * 10).cast("int"))  # which sector is hit
        .withColumn("tail_event_flag",
                    F.when(F.abs(F.col("market_shock")) > 0.04, True).otherwise(False))
        # Add a correlated credit spread shock
        .withColumn("credit_spread_shock",
                    F.col("market_shock") * -0.5 + F.randn(seed=SEED + 4) * 0.003)
        # Add a momentum factor
        .withColumn("momentum_factor", F.randn(seed=SEED + 5) * 0.01)
    )

    scenarios = scenarios.repartition(200)
    scenarios.cache()
    cnt = scenarios.count()
    elapsed = time.time() - t0
    size_est_mb = cnt * 7 * 8 / (1024 * 1024)  # rough: 7 double-ish cols × 8 bytes
    print(f"    Created {cnt:,} scenario rows (~{size_est_mb:.0f} MB est.) in {elapsed:.1f}s")
    return scenarios


# ---------------------------------------------------------------------------
# Step 3 – Run the Monte Carlo simulation
# ---------------------------------------------------------------------------

def run_simulation(
    spark: SparkSession,
    instruments: DataFrame,
    scenarios: DataFrame,
) -> DataFrame:
    """
    For every (instrument, scenario, time_step) compute a daily P&L using a
    simplified geometric Brownian motion model augmented with credit and
    momentum factors, then aggregate to per-scenario portfolio returns.

    This is the computationally expensive step: 2 000 × 12.6 M = 25.2 B
    virtual cell evaluations (Spark optimises via columnar + lazy eval).
    """
    print("\n=== Step 3: Running Monte Carlo simulation (this may take a while) ===")
    t0 = time.time()

    # Broadcast the (small) instruments table for a fast map-side join
    instruments_b = F.broadcast(instruments)

    # Map sector names to indices so we can correlate sector shocks
    sector_list = [r.sector for r in instruments.select("sector").distinct().collect()]
    sector_to_idx = {s: i for i, s in enumerate(sorted(sector_list))}
    mapping_expr = F.create_map([F.lit(x) for pair in sector_to_idx.items() for x in pair])
    instruments_enriched = instruments_b.withColumn("sector_idx", mapping_expr[F.col("sector")])

    # Join instruments × scenarios  (broadcast join – instruments are small)
    combined = scenarios.crossJoin(instruments_enriched)

    # --- Daily return model ---
    # r_i,t = mu_i/252 + beta_i * market_shock + sector_hit * vol_shock
    #       + credit_component + momentum_component + idiosyncratic noise
    dt = 1.0 / NUM_TIME_STEPS
    combined = (
        combined
        # Drift component
        .withColumn("drift", F.col("annual_return") * dt)
        # Systematic risk
        .withColumn("systematic", F.col("beta") * F.col("market_shock"))
        # Sector-specific shock (only if this instrument's sector is the "hit" sector)
        .withColumn("sector_hit",
                    F.when(F.col("sector_idx") == F.col("sector_shock_idx"),
                           F.col("vol_shock") * F.col("annual_volatility"))
                    .otherwise(0.0))
        # Credit component – higher credit risk score ⇒ more sensitive
        .withColumn("credit_component",
                    F.col("credit_spread_shock") * (F.col("credit_rating_score") / 10.0))
        # Momentum
        .withColumn("momentum_component",
                    F.col("momentum_factor") * F.col("beta") * 0.5)
        # Idiosyncratic noise  (scaled by instrument vol)
        .withColumn("idio_noise",
                    F.randn(seed=SEED + 99) * F.col("annual_volatility") * math.sqrt(dt))
        # Total daily return for this instrument in this scenario at this time step
        .withColumn("daily_return",
                    F.col("drift")
                    + F.col("systematic")
                    + F.col("sector_hit")
                    + F.col("credit_component")
                    + F.col("momentum_component")
                    + F.col("idio_noise"))
        # Weighted contribution to portfolio
        .withColumn("weighted_return", F.col("daily_return") * F.col("weight"))
    )

    # --- Aggregate: sum weighted returns per (scenario, time_step) ---
    portfolio_daily = (
        combined
        .groupBy("scenario_id", "time_step")
        .agg(
            F.sum("weighted_return").alias("portfolio_daily_return"),
            F.avg("daily_return").alias("avg_instrument_return"),
            F.stddev("daily_return").alias("cross_section_vol"),
            F.sum(F.when(F.col("tail_event_flag"), F.col("weighted_return")).otherwise(0))
             .alias("tail_event_pnl"),
            F.count("*").alias("n_instruments"),
        )
    )

    # --- Cumulative return per scenario (product of (1 + r_t)) ---
    # Use log-sum trick:  cum_return = exp( sum( log(1 + r_t) ) ) - 1
    scenario_results = (
        portfolio_daily
        .groupBy("scenario_id")
        .agg(
            F.sum(F.log1p(F.col("portfolio_daily_return"))).alias("log_cum_return"),
            F.avg("portfolio_daily_return").alias("avg_daily_return"),
            F.stddev("portfolio_daily_return").alias("daily_vol"),
            F.min("portfolio_daily_return").alias("worst_day"),
            F.max("portfolio_daily_return").alias("best_day"),
            F.sum("tail_event_pnl").alias("total_tail_pnl"),
            F.avg("cross_section_vol").alias("avg_cross_section_vol"),
        )
        .withColumn("cumulative_return", F.expm1(F.col("log_cum_return")))
        .withColumn("annualized_return", F.col("avg_daily_return") * 252)
        .withColumn("annualized_vol", F.col("daily_vol") * math.sqrt(252))
        .withColumn("sharpe_ratio",
                    F.when(F.col("annualized_vol") > 0,
                           F.col("annualized_return") / F.col("annualized_vol"))
                    .otherwise(0.0))
        .drop("log_cum_return")
    )

    scenario_results.cache()
    cnt = scenario_results.count()
    elapsed = time.time() - t0
    print(f"    Simulated {cnt:,} scenarios in {elapsed:.1f}s")
    return scenario_results


# ---------------------------------------------------------------------------
# Step 4 – Write results to Parquet
# ---------------------------------------------------------------------------

def write_parquet(df: DataFrame) -> None:
    """Convert Spark results to pandas and write via pyarrow.

    We use pandas/pyarrow for the final I/O because the Hadoop filesystem
    layer is incompatible with JDK 25 (Subject.getSubject removed).  The
    aggregated results are only ~50 K rows so this is instant.
    """
    print(f"\n=== Step 4: Writing results to {PARQUET_PATH} ===")
    t0 = time.time()

    out = Path(PARQUET_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    pdf = df.toPandas()
    pdf.to_parquet(out, engine="pyarrow", index=False)
    elapsed = time.time() - t0

    total_bytes = out.stat().st_size
    print(f"    Wrote Parquet in {elapsed:.1f}s  ({total_bytes / 1024 / 1024:.1f} MB on disk)")


# ---------------------------------------------------------------------------
# Step 5 – Read back and print a rich summary
# ---------------------------------------------------------------------------

def print_summary(spark: SparkSession) -> None:
    print("\n" + "=" * 72)
    print("   MONTE CARLO PORTFOLIO RISK SIMULATION — RESULTS SUMMARY")
    print("=" * 72)

    # Read via pandas/pyarrow (avoids Hadoop FS compat issue on JDK 25)
    import pandas as _pd
    _pdf = _pd.read_parquet(str(PARQUET_PATH), engine="pyarrow")
    df = spark.createDataFrame(_pdf)
    n = df.count()

    # --- Descriptive statistics ---
    stats = df.select(
        F.count("*").alias("scenarios"),
        F.mean("cumulative_return").alias("mean_cum_return"),
        F.stddev("cumulative_return").alias("std_cum_return"),
        F.expr("percentile_approx(cumulative_return, 0.01)").alias("cum_return_p01"),
        F.expr("percentile_approx(cumulative_return, 0.05)").alias("cum_return_p05"),
        F.expr("percentile_approx(cumulative_return, 0.25)").alias("cum_return_p25"),
        F.expr("percentile_approx(cumulative_return, 0.50)").alias("cum_return_p50"),
        F.expr("percentile_approx(cumulative_return, 0.75)").alias("cum_return_p75"),
        F.expr("percentile_approx(cumulative_return, 0.95)").alias("cum_return_p95"),
        F.expr("percentile_approx(cumulative_return, 0.99)").alias("cum_return_p99"),
        F.min("cumulative_return").alias("min_cum_return"),
        F.max("cumulative_return").alias("max_cum_return"),
        F.mean("annualized_return").alias("mean_ann_return"),
        F.mean("annualized_vol").alias("mean_ann_vol"),
        F.mean("sharpe_ratio").alias("mean_sharpe"),
        F.mean("worst_day").alias("mean_worst_day"),
        F.mean("best_day").alias("mean_best_day"),
        F.mean("total_tail_pnl").alias("mean_tail_pnl"),
    ).collect()[0]

    print(f"\n  Scenarios simulated : {stats['scenarios']:>12,}")
    print(f"  Instruments modeled : {NUM_INSTRUMENTS:>12,}")
    print(f"  Time steps (days)   : {NUM_TIME_STEPS:>12,}")

    print(f"\n  {'— Cumulative Return Distribution —':^50}")
    print(f"  Mean                : {stats['mean_cum_return']:>+12.4%}")
    print(f"  Std Dev             : {stats['std_cum_return']:>12.4%}")
    print(f"  Min                 : {stats['min_cum_return']:>+12.4%}")
    print(f"  1st Percentile      : {stats['cum_return_p01']:>+12.4%}")
    print(f"  5th Percentile (VaR): {stats['cum_return_p05']:>+12.4%}")
    print(f"  25th Percentile     : {stats['cum_return_p25']:>+12.4%}")
    print(f"  Median              : {stats['cum_return_p50']:>+12.4%}")
    print(f"  75th Percentile     : {stats['cum_return_p75']:>+12.4%}")
    print(f"  95th Percentile     : {stats['cum_return_p95']:>+12.4%}")
    print(f"  99th Percentile     : {stats['cum_return_p99']:>+12.4%}")
    print(f"  Max                 : {stats['max_cum_return']:>+12.4%}")

    # Value-at-Risk & Expected Shortfall
    var_95 = stats["cum_return_p05"]
    cvar_95 = df.filter(F.col("cumulative_return") <= var_95) \
                .agg(F.mean("cumulative_return")).collect()[0][0]

    var_99 = stats["cum_return_p01"]
    cvar_99 = df.filter(F.col("cumulative_return") <= var_99) \
                .agg(F.mean("cumulative_return")).collect()[0][0]

    print(f"\n  {'— Risk Metrics —':^50}")
    print(f"  VaR  95%            : {var_95:>+12.4%}")
    print(f"  CVaR 95% (ES)       : {cvar_95:>+12.4%}")
    print(f"  VaR  99%            : {var_99:>+12.4%}")
    print(f"  CVaR 99% (ES)       : {cvar_99:>+12.4%}")

    print(f"\n  {'— Annualized Performance —':^50}")
    print(f"  Mean Ann. Return    : {stats['mean_ann_return']:>+12.4%}")
    print(f"  Mean Ann. Volatility: {stats['mean_ann_vol']:>12.4%}")
    print(f"  Mean Sharpe Ratio   : {stats['mean_sharpe']:>12.4f}")

    print(f"\n  {'— Extreme Day Analysis —':^50}")
    print(f"  Avg Worst Day       : {stats['mean_worst_day']:>+12.4%}")
    print(f"  Avg Best Day        : {stats['mean_best_day']:>+12.4%}")
    print(f"  Avg Tail-Event P&L  : {stats['mean_tail_pnl']:>+12.6f}")

    # --- Distribution of Sharpe ratios ---
    sharpe_stats = df.select(
        F.expr("percentile_approx(sharpe_ratio, 0.05)").alias("sharpe_p05"),
        F.expr("percentile_approx(sharpe_ratio, 0.50)").alias("sharpe_p50"),
        F.expr("percentile_approx(sharpe_ratio, 0.95)").alias("sharpe_p95"),
    ).collect()[0]

    print(f"\n  {'— Sharpe Ratio Distribution —':^50}")
    print(f"  5th Percentile      : {sharpe_stats['sharpe_p05']:>12.4f}")
    print(f"  Median              : {sharpe_stats['sharpe_p50']:>12.4f}")
    print(f"  95th Percentile     : {sharpe_stats['sharpe_p95']:>12.4f}")

    # --- Probability of loss ---
    n_loss = df.filter(F.col("cumulative_return") < 0).count()
    print(f"\n  {'— Outcome Probabilities —':^50}")
    print(f"  P(loss)             : {n_loss / n:>12.2%}")
    print(f"  P(gain > 10%)       : {df.filter(F.col('cumulative_return') > 0.10).count() / n:>12.2%}")
    print(f"  P(gain > 25%)       : {df.filter(F.col('cumulative_return') > 0.25).count() / n:>12.2%}")
    print(f"  P(loss > 10%)       : {df.filter(F.col('cumulative_return') < -0.10).count() / n:>12.2%}")
    print(f"  P(loss > 25%)       : {df.filter(F.col('cumulative_return') < -0.25).count() / n:>12.2%}")

    print("\n" + "=" * 72)
    print(f"  Results saved to: {PARQUET_PATH}")
    print("=" * 72 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    total_t0 = time.time()
    spark = _spark_session()

    try:
        instruments = generate_instruments(spark)
        scenarios = generate_scenarios(spark)
        results = run_simulation(spark, instruments, scenarios)
        write_parquet(results)
        print_summary(spark)
    finally:
        spark.stop()

    total_elapsed = time.time() - total_t0
    print(f"Total wall-clock time: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()

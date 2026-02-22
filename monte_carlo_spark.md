# Monte Carlo Portfolio Risk Simulation — Code Walkthrough

This document explains every section of
[src/learn_dataeng/monte_carlo_spark.py](src/learn_dataeng/monte_carlo_spark.py)
line by line. It is written for someone who is new to **PySpark** (the Python
API for Apache Spark).

---

## Table of Contents

1. [Module Docstring & Imports](#1-module-docstring--imports)
2. [Configuration Constants](#2-configuration-constants)
3. [Creating the Spark Session](#3-creating-the-spark-session)
4. [Step 1 — Generating Synthetic Instruments](#4-step-1--generating-synthetic-instruments)
5. [Step 2 — Generating the Scenario Matrix](#5-step-2--generating-the-scenario-matrix)
6. [Step 3 — Running the Simulation](#6-step-3--running-the-simulation)
7. [Step 4 — Writing Results to Parquet](#7-step-4--writing-results-to-parquet)
8. [Step 5 — Printing a Summary](#8-step-5--printing-a-summary)
9. [Main Entry Point](#9-main-entry-point)

---

## 1. Module Docstring & Imports

```python
"""
Monte Carlo Portfolio Risk Simulation using PySpark.

Generates ~500 MB of synthetic financial instrument data, then runs a
large-scale Monte Carlo simulation to estimate portfolio value-at-risk (VaR),
expected shortfall (CVaR), and return distributions across thousands of
correlated market scenarios.

Usage:
    uv run monte-carlo-spark
"""
```

The triple-quoted docstring at the top of the file describes what the program
does. It tells you that:

- It uses PySpark (Spark's Python API) to run a **Monte Carlo simulation** —
  a technique that repeats a random experiment thousands of times to estimate
  the probability distribution of outcomes.
- It creates roughly 500 MB of fake financial data, then simulates many market
  scenarios to calculate risk metrics like **VaR** (Value-at-Risk) and **CVaR**
  (Conditional Value-at-Risk / Expected Shortfall).

```python
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
```

### What each import does

| Import | Purpose |
|--------|---------|
| `from __future__ import annotations` | Lets you use newer-style type hints (e.g., `str \| None`) even on older Python versions. |
| `math` | Standard math library (used for `math.sqrt`). |
| `os`, `shutil` | File system utilities (directory creation, removal). |
| `time` | Used to measure how long each step takes. |
| `Path` (from `pathlib`) | Object-oriented file path handling. |
| **`SparkSession`** | The entry point for all PySpark functionality. You need one of these to do *anything* in Spark. Think of it as the "connection" to Spark. |
| **`DataFrame`** | Spark's equivalent of a database table or a pandas DataFrame — a distributed collection of rows with named columns. |
| **`functions as F`** | A module of built-in column operations (e.g., `F.sum()`, `F.col()`, `F.randn()`). You will see `F.` everywhere in PySpark code. |
| **`types as T`** | Data type definitions (e.g., `T.IntegerType()`, `T.StringType()`). Used when you need to tell Spark what schema your data should have. |
| `Window` | Defines windowing specifications for window functions (imported but not used in this script). |

> **PySpark key concept:** Spark is a *distributed* compute engine. Even on a
> single laptop it parallelises work across all your CPU cores. Everything you
> do with a `DataFrame` is lazily evaluated — Spark builds a plan and only
> executes it when you ask for a result (`.count()`, `.collect()`, writing to
> disk, etc.).

---

## 2. Configuration Constants

```python
NUM_INSTRUMENTS = 2_000          # financial instruments (stocks / bonds / derivs)
NUM_SCENARIOS = 50_000           # Monte Carlo market scenarios
NUM_TIME_STEPS = 252             # trading days in a year
OUTPUT_DIR = Path("monte_carlo_output")
PARQUET_PATH = OUTPUT_DIR / "simulation_results.parquet"
SEED = 42
```

| Constant | Meaning |
|----------|---------|
| `NUM_INSTRUMENTS` | How many fake financial products (stocks, bonds, etc.) to create. |
| `NUM_SCENARIOS` | How many hypothetical market scenarios to simulate. More scenarios = more accurate probability estimates, but also more computation. |
| `NUM_TIME_STEPS` | Number of trading days in a year (252 is the standard on most exchanges). |
| `OUTPUT_DIR` / `PARQUET_PATH` | Where the results will be saved. **Parquet** is a columnar file format widely used in big-data pipelines because it is fast to read and compresses well. |
| `SEED` | A fixed random seed so the results are reproducible. |

---

## 3. Creating the Spark Session

```python
def _spark_session() -> SparkSession:
    """Create a local Spark session tuned for a beefy single-node run."""
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

    spark.sparkContext.setLogLevel("ERROR")
    return spark
```

### Line-by-line

| Line | Explanation |
|------|-------------|
| `SparkSession.builder` | Starts the builder pattern for creating a Spark session. |
| `.master("local[*]")` | Tells Spark to run **locally** using **all available CPU cores** (`*`). In production you'd point this at a cluster URL. |
| `.appName("MonteCarloPortfolioRisk")` | A human-readable name for this Spark application (shows up in the Spark UI). |
| `.config("spark.driver.memory", "4g")` | Allocates 4 GB of RAM to the Spark driver (the process that coordinates the work). |
| `.config("spark.sql.shuffle.partitions", "200")` | When Spark needs to redistribute data across partitions (a "shuffle"), it will create **200 partitions**. This controls parallelism during operations like `groupBy`. |
| `.config("spark.sql.parquet.compression.codec", "snappy")` | Parquet files will be compressed with the **Snappy** codec (fast compression/decompression). |
| `.config("spark.ui.showConsoleProgress", "true")` | Shows a progress bar in the terminal during long jobs. |
| `.getOrCreate()` | If a session already exists, reuse it; otherwise create a new one. |
| `.setLogLevel("ERROR")` | Suppresses verbose INFO/WARN log messages. Only errors will print. |

> **PySpark key concept:** The `SparkSession` is a singleton — you create it
> once and pass it around. It holds the connection to the Spark execution
> engine.

---

## 4. Step 1 — Generating Synthetic Instruments

```python
def generate_instruments(spark: SparkSession) -> DataFrame:
```

This function creates a DataFrame describing 2,000 fictional financial
instruments (stocks, bonds, derivatives, etc.).

### Building the rows in Python

```python
    sectors = [
        "Technology", "Healthcare", "Financials", "Energy", "Consumer",
        "Industrials", "Materials", "Utilities", "RealEstate", "Telecom",
    ]
    asset_classes = ["Equity", "FixedIncome", "Derivative", "Commodity", "FX"]

    rows = []
    import random
    rng = random.Random(SEED)
    for i in range(NUM_INSTRUMENTS):
        rows.append((
            i,                                                # instrument_id
            f"INST-{i:06d}",                                  # ticker (e.g., "INST-000042")
            rng.choice(sectors),                              # random sector
            rng.choice(asset_classes),                        # random asset class
            round(rng.uniform(0.0001, 0.01), 6),              # portfolio weight
            round(rng.gauss(0.08, 0.12), 6),                  # expected annual return
            round(abs(rng.gauss(0.20, 0.10)), 6),             # annual volatility
            round(max(0.0, rng.gauss(0.02, 0.015)), 6),       # dividend yield
            round(rng.gauss(1.0, 0.4), 4),                    # beta
            rng.randint(1, 10),                               # credit rating score
        ))
```

Each tuple is one instrument row with random but realistic-looking financial
attributes. The Python `random` module generates the values, seeded for
reproducibility.

### Defining the schema

```python
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
```

> **PySpark key concept — Schema:** Unlike pandas, Spark prefers to know the
> data types up front via a **schema**. A `StructType` is a list of
> `StructField`s, where each field has a name and a type. This avoids costly
> type inference.

### Creating the DataFrame

```python
    instruments = spark.createDataFrame(rows, schema=schema)
```

`spark.createDataFrame()` converts the Python list of tuples into a
distributed Spark DataFrame. Spark will distribute this data across partitions
in memory.

### Normalizing the weights

```python
    total_w = instruments.agg(F.sum("weight")).collect()[0][0]
    instruments = instruments.withColumn("weight", F.col("weight") / F.lit(total_w))
```

| Expression | Meaning |
|------------|---------|
| `F.sum("weight")` | A **column expression** that computes the sum of the `weight` column. |
| `.agg(...)` | Short for "aggregate" — applies an aggregation function to the whole DataFrame. |
| `.collect()` | **Triggers execution** and pulls all results back to the Python driver as a list of `Row` objects. |
| `[0][0]` | Gets the first row, first column of the result (the scalar sum). |
| `.withColumn("weight", ...)` | Creates (or replaces) a column called `weight` with new values. |
| `F.col("weight")` | A reference to the existing `weight` column. |
| `F.lit(total_w)` | Wraps a Python scalar as a Spark **literal** so it can be used in column expressions. |

After this, all weights sum to exactly 1.0 — the portfolio is fully invested.

### Caching

```python
    instruments.cache()
    cnt = instruments.count()
```

> **PySpark key concept — `cache()`:** By default, Spark recomputes a
> DataFrame every time you use it (lazy evaluation). Calling `.cache()` tells
> Spark to keep the result in memory after it is first computed. `.count()`
> forces the computation (it's an **action**), so after this line the
> instruments DataFrame lives in RAM and can be reused cheaply.

---

## 5. Step 2 — Generating the Scenario Matrix

```python
def generate_scenarios(spark: SparkSession) -> DataFrame:
```

This function produces the "big data" — 50,000 scenarios × 252 time steps =
**12.6 million rows**, each representing a hypothetical daily market event.

### Creating ID DataFrames

```python
    scenario_ids = spark.range(NUM_SCENARIOS).withColumnRenamed("id", "scenario_id")
    time_steps = spark.range(NUM_TIME_STEPS).withColumnRenamed("id", "time_step")
```

`spark.range(n)` creates a DataFrame with a single column `id` containing
values 0 through n−1. It is Spark's fastest way to generate sequential IDs.
`.withColumnRenamed()` renames the column.

### Cross join

```python
    scenarios = scenario_ids.crossJoin(time_steps)
```

> **PySpark key concept — `crossJoin()`:** A **cross join** (a.k.a. Cartesian
> product) pairs every row from the left DataFrame with every row from the
> right. Here it produces 50,000 × 252 = 12,600,000 rows. Cross joins are
> expensive and normally avoided, but in a Monte Carlo simulation we
> *deliberately* want every combination.

### Adding random shocks

```python
    scenarios = (
        scenarios
        .withColumn("market_shock", F.randn(seed=SEED) * 0.02)
        .withColumn("rate_shock", F.randn(seed=SEED + 1) * 0.005)
        .withColumn("vol_shock", F.abs(F.randn(seed=SEED + 2)) * 0.03)
        .withColumn("sector_shock_idx", (F.rand(seed=SEED + 3) * 10).cast("int"))
        .withColumn("tail_event_flag",
                    F.when(F.abs(F.col("market_shock")) > 0.04, True).otherwise(False))
        .withColumn("credit_spread_shock",
                    F.col("market_shock") * -0.5 + F.randn(seed=SEED + 4) * 0.003)
        .withColumn("momentum_factor", F.randn(seed=SEED + 5) * 0.01)
    )
```

| Expression | What it does |
|------------|--------------|
| `F.randn(seed=...)` | Generates a column of random numbers drawn from a **standard normal** (mean 0, std 1) distribution. Each row gets its own random value. The `seed` makes results reproducible. |
| `F.rand(seed=...)` | Generates uniform random numbers between 0 and 1. |
| `* 0.02` | Scales the random values. Here, a daily market shock has a standard deviation of 2%. |
| `F.abs(...)` | Takes the absolute value (volatility shocks are always positive). |
| `.cast("int")` | Converts a floating-point value to an integer (used here to pick a random sector index 0–9). |
| `F.when(...).otherwise(...)` | Spark's version of an **if/else** expression. Here it flags extreme market events where the shock exceeds ±4%. |

Each column represents a different type of randomness that will perturb
instrument returns in the simulation.

### Repartitioning and caching

```python
    scenarios = scenarios.repartition(200)
    scenarios.cache()
    cnt = scenarios.count()
```

> **PySpark key concept — `repartition(n)`:** Redistributes data into `n`
> partitions. This controls parallelism — Spark processes each partition in a
> separate task. 200 partitions means up to 200 CPU cores can work in
> parallel.

Calling `.cache()` followed by `.count()` materialises the randomly generated
data in memory so it isn't regenerated later.

---

## 6. Step 3 — Running the Simulation

```python
def run_simulation(
    spark: SparkSession,
    instruments: DataFrame,
    scenarios: DataFrame,
) -> DataFrame:
```

This is the most compute-intensive function. It combines every instrument
with every scenario and computes daily returns.

### Broadcasting the small table

```python
    instruments_b = F.broadcast(instruments)
```

> **PySpark key concept — `broadcast()`:** When you join a small DataFrame
> with a large one, Spark can send a copy of the small table to every worker
> (a "broadcast"). This avoids a costly shuffle of the large table. Here
> `instruments` is only 2,000 rows, so it is cheap to broadcast.

### Mapping sectors to numeric indices

```python
    sector_list = [r.sector for r in instruments.select("sector").distinct().collect()]
    sector_to_idx = {s: i for i, s in enumerate(sorted(sector_list))}
    mapping_expr = F.create_map([F.lit(x) for pair in sector_to_idx.items() for x in pair])
    instruments_enriched = instruments_b.withColumn("sector_idx", mapping_expr[F.col("sector")])
```

| Line | Meaning |
|------|---------|
| `.select("sector").distinct().collect()` | Gets the unique sector names back to Python. |
| `sector_to_idx` | A Python dict like `{"Consumer": 0, "Energy": 1, ...}`. |
| `F.create_map(...)` | Builds a Spark **MapType** column (like a dictionary) from alternating key/value literals. |
| `mapping_expr[F.col("sector")]` | Looks up each instrument's sector name in the map to get its numeric index. This lets us match instruments to `sector_shock_idx` later. |

### The big cross join

```python
    combined = scenarios.crossJoin(instruments_enriched)
```

This creates the full simulation grid: 12.6 M scenario rows × 2,000
instruments = **25.2 billion virtual cells**. Spark doesn't materialise all of
these at once — it processes them in streaming fashion, partition by partition.

### Computing daily returns

```python
    combined = (
        combined
        .withColumn("drift", F.col("annual_return") * dt)
        .withColumn("systematic", F.col("beta") * F.col("market_shock"))
        .withColumn("sector_hit",
                    F.when(F.col("sector_idx") == F.col("sector_shock_idx"),
                           F.col("vol_shock") * F.col("annual_volatility"))
                    .otherwise(0.0))
        .withColumn("credit_component",
                    F.col("credit_spread_shock") * (F.col("credit_rating_score") / 10.0))
        .withColumn("momentum_component",
                    F.col("momentum_factor") * F.col("beta") * 0.5)
        .withColumn("idio_noise",
                    F.randn(seed=SEED + 99) * F.col("annual_volatility") * math.sqrt(dt))
        .withColumn("daily_return",
                    F.col("drift")
                    + F.col("systematic")
                    + F.col("sector_hit")
                    + F.col("credit_component")
                    + F.col("momentum_component")
                    + F.col("idio_noise"))
        .withColumn("weighted_return", F.col("daily_return") * F.col("weight"))
    )
```

This block builds up a **daily return model** for each (instrument, scenario,
time-step) combination:

| Column | Formula / Meaning |
|--------|-------------------|
| `drift` | `annual_return / 252` — the expected daily return (the deterministic part). |
| `systematic` | `beta × market_shock` — how much the instrument moves with the overall market. Beta > 1 means more sensitive. |
| `sector_hit` | A bonus shock applied only when this instrument's sector matches the randomly chosen "hit" sector for this time step. |
| `credit_component` | Instruments with worse credit ratings (higher score) are more affected by credit spread shocks. |
| `momentum_component` | A trend-following factor, scaled by beta. |
| `idio_noise` | Random noise unique to this instrument — the "unexplained" part of the return. Scaled by `annual_volatility × sqrt(dt)` (standard Brownian motion scaling). |
| `daily_return` | Sum of all the above components. |
| `weighted_return` | `daily_return × weight` — this instrument's contribution to the portfolio return for this day. |

> **PySpark key concept — Lazy evaluation & chaining:** Each `.withColumn()`
> call adds a transformation to Spark's execution plan but does **not**
> compute anything yet. Spark reads this chain and optimises it before
> executing. This is why you can describe 25 billion cells worth of
> computation without running out of memory.

### Aggregation — portfolio daily returns

```python
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
```

> **PySpark key concept — `groupBy().agg()`:** This is like SQL's
> `GROUP BY` + aggregate functions. It collapses all the instrument-level rows
> into one row per (scenario, time_step) by summing, averaging, etc.

| Aggregate | Meaning |
|-----------|---------|
| `F.sum("weighted_return")` | The portfolio's total return for this scenario on this day. |
| `F.avg("daily_return")` | Average return across all instruments (not weighted). |
| `F.stddev("daily_return")` | How dispersed the individual instrument returns are. |
| `tail_event_pnl` | The portfolio P&L contributed only during extreme ("tail") events. |
| `F.count("*")` | Sanity check — should be 2,000 (one per instrument). |

### Aggregation — cumulative scenario returns

```python
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
```

This collapses the 252 daily returns for each scenario into **one summary row
per scenario**.

| Key expression | Explanation |
|----------------|-------------|
| `F.log1p(x)` | Computes $\ln(1 + x)$. The **log-sum trick** avoids multiplying 252 numbers close to 1 (which causes floating-point error). |
| `F.expm1(x)` | Computes $e^x - 1$ — the inverse of `log1p`. Together: $\text{cumulative} = \exp\!\bigl(\sum \ln(1+r_t)\bigr) - 1$. |
| `annualized_return` | `avg_daily_return × 252` — scales the daily average to an annual figure. |
| `annualized_vol` | `daily_vol × √252` — standard annualisation of volatility. |
| `sharpe_ratio` | $\frac{\text{annualized return}}{\text{annualized volatility}}$ — a common risk-adjusted performance measure. |
| `.drop("log_cum_return")` | Removes the intermediate column that was only needed for the calculation. |

### Caching the results

```python
    scenario_results.cache()
    cnt = scenario_results.count()
```

Again, `.cache()` + `.count()` forces Spark to compute and store the 50,000
scenario result rows in memory, ready for output.

---

## 7. Step 4 — Writing Results to Parquet

```python
def write_parquet(df: DataFrame) -> None:
    out = Path(PARQUET_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    pdf = df.toPandas()
    pdf.to_parquet(out, engine="pyarrow", index=False)
```

| Line | Meaning |
|------|---------|
| `out.parent.mkdir(parents=True, exist_ok=True)` | Creates the output directory if it doesn't exist. |
| `out.unlink()` | Deletes the old Parquet file if present. |
| `df.toPandas()` | Converts the Spark DataFrame to a **pandas** DataFrame. This pulls all data to the driver, which is fine here because the aggregated result is only ~50,000 rows. |
| `.to_parquet(...)` | Writes the pandas DataFrame to a Parquet file using PyArrow. |

> **Why not use Spark's built-in Parquet writer?** The docstring explains that
> the Hadoop filesystem layer (which Spark normally uses for I/O) is
> incompatible with JDK 25. Converting to pandas and using PyArrow directly
> sidesteps the issue.

---

## 8. Step 5 — Printing a Summary

```python
def print_summary(spark: SparkSession) -> None:
```

This function reads the Parquet file back in and computes a rich statistical
summary. Here are the key PySpark patterns it uses:

### Reading Parquet back into Spark

```python
    import pandas as _pd
    _pdf = _pd.read_parquet(str(PARQUET_PATH), engine="pyarrow")
    df = spark.createDataFrame(_pdf)
```

Reads the file with pandas, then wraps it back in a Spark DataFrame so we can
use Spark's aggregation functions.

### Computing descriptive statistics

```python
    stats = df.select(
        F.count("*").alias("scenarios"),
        F.mean("cumulative_return").alias("mean_cum_return"),
        F.stddev("cumulative_return").alias("std_cum_return"),
        F.expr("percentile_approx(cumulative_return, 0.01)").alias("cum_return_p01"),
        F.expr("percentile_approx(cumulative_return, 0.05)").alias("cum_return_p05"),
        ...
    ).collect()[0]
```

| Expression | Meaning |
|------------|---------|
| `F.count("*")` | Total number of rows. |
| `F.mean(...)` / `F.stddev(...)` | Mean and standard deviation. |
| `F.expr("percentile_approx(..., 0.05)")` | The 5th percentile — i.e., the value below which 5% of outcomes fall. This is used as the **Value-at-Risk (VaR)** at the 95% confidence level. `F.expr()` lets you write Spark SQL expressions as strings. |
| `.collect()[0]` | Pulls the single result row back to the driver. |

### Computing VaR and CVaR

```python
    var_95 = stats["cum_return_p05"]
    cvar_95 = df.filter(F.col("cumulative_return") <= var_95) \
                .agg(F.mean("cumulative_return")).collect()[0][0]
```

| Concept | Explanation |
|---------|-------------|
| **VaR (Value-at-Risk)** at 95% | The 5th percentile of cumulative returns. It answers: "What is the worst loss I'd expect 95% of the time?" |
| **CVaR (Expected Shortfall)** | The *average* loss in the worst 5% of scenarios. It is always worse (more negative) than VaR and captures the severity of the tail. |
| `df.filter(...)` | Keeps only the rows where the condition is true (like SQL `WHERE`). |

### Probability calculations

```python
    n_loss = df.filter(F.col("cumulative_return") < 0).count()
    print(f"  P(loss)             : {n_loss / n:>12.2%}")
    print(f"  P(gain > 10%)       : {df.filter(F.col('cumulative_return') > 0.10).count() / n:>12.2%}")
```

These lines filter the results to count how many scenarios end in a loss (or
in a gain greater than 10%, etc.) and divide by the total number of scenarios
to get a probability.

---

## 9. Main Entry Point

```python
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
```

| Line | Meaning |
|------|---------|
| `spark = _spark_session()` | Creates the Spark session (Step 0). |
| `generate_instruments(spark)` | Step 1 — builds the 2,000-instrument DataFrame. |
| `generate_scenarios(spark)` | Step 2 — builds the 12.6 M-row scenario DataFrame. |
| `run_simulation(...)` | Step 3 — the heavy computation; joins, computes returns, aggregates. |
| `write_parquet(results)` | Step 4 — saves results to disk. |
| `print_summary(spark)` | Step 5 — reads results back and prints statistics. |
| `spark.stop()` | Shuts down the Spark session and releases all resources. Always called in a `finally` block so it runs even if something crashes. |
| `if __name__ == "__main__":` | Standard Python idiom — only run `main()` when the script is executed directly, not when imported as a module. |

---

## PySpark Concepts Cheat Sheet

| Concept | One-liner |
|---------|-----------|
| **SparkSession** | Your gateway to Spark. Create once, use everywhere. |
| **DataFrame** | A distributed table of rows and columns — Spark's core data structure. |
| **Transformation** | An operation that returns a new DataFrame (`withColumn`, `filter`, `groupBy`). Lazy — nothing runs yet. |
| **Action** | An operation that triggers execution and returns a result (`count`, `collect`, `show`, `write`). |
| **`F.col("name")`** | A reference to a column by name. |
| **`F.lit(value)`** | A constant value wrapped as a column expression. |
| **`.cache()`** | Tells Spark to keep computed data in memory for reuse. |
| **`broadcast()`** | Sends a small DataFrame to all workers to avoid a shuffle join. |
| **`crossJoin()`** | Every row from A paired with every row from B (Cartesian product). |
| **`groupBy().agg()`** | Group rows and apply aggregate functions (sum, avg, count, etc.). |
| **Parquet** | A columnar storage format — fast reads, good compression, schema built in. |

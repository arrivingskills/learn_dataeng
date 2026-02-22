# Sunday morning run

=== Step 1: Generating synthetic instrument universe ===
    Created 2,000 instruments in 2.8s

=== Step 2: Generating Monte Carlo scenario matrix ===
    Created 12,600,000 scenario rows (~673 MB est.) in 4.4s

=== Step 3: Running Monte Carlo simulation (this may take a while) ===
    Simulated 50,000 scenarios in 178.3s

=== Step 4: Writing results to monte_carlo_output/simulation_results.parquet ===
    Wrote Parquet in 0.5s  (5.1 MB on disk)

========================================================================
   MONTE CARLO PORTFOLIO RISK SIMULATION — RESULTS SUMMARY
========================================================================

  Scenarios simulated :       50,000
  Instruments modeled :        2,000
  Time steps (days)   :          252

          — Cumulative Return Distribution —        
  Mean                :    +21.7278%
  Std Dev             :     30.5286%
  Min                 :    -54.2465%
  1st Percentile      :    -33.5387%
  5th Percentile (VaR):    -21.0275%
  25th Percentile     :     +0.0509%
  Median              :    +17.9328%
  75th Percentile     :    +39.3506%
  95th Percentile     :    +77.3447%
  99th Percentile     :   +110.3559%
  Max                 :   +267.5627%

                   — Risk Metrics —                 
  VaR  95%            :    -21.0275%
  CVaR 95% (ES)       :    -28.6435%
  VaR  99%            :    -33.5387%
  CVaR 99% (ES)       :    -38.6050%

              — Annualized Performance —            
  Mean Ann. Return    :    +19.6537%
  Mean Ann. Volatility:     24.5771%
  Mean Sharpe Ratio   :       0.8012

               — Extreme Day Analysis —             
  Avg Worst Day       :     -4.2923%
  Avg Best Day        :     +4.4523%
  Avg Tail-Event P&L  :    +0.008950

            — Sharpe Ratio Distribution —           
  5th Percentile      :      -0.8437
  Median              :       0.7949
  95th Percentile     :       2.4629

              — Outcome Probabilities —             
  P(loss)             :       24.92%
  P(gain > 10%)       :       61.27%
  P(gain > 25%)       :       40.76%
  P(loss > 10%)       :       13.46%
  P(loss > 25%)       :        3.26%

========================================================================
  Results saved to: monte_carlo_output/simulation_results.parquet
========================================================================

Total wall-clock time: 196.0s

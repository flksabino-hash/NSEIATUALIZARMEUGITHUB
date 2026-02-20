# Whale Hunter v0.4 Max Verstappen Patch

This patch upgrades the synthetic backtest harness with diagnostics, filter attribution, auto-calibration, ablation runs, walk-forward support, and faster metric execution through caching/stride precompute.

## New highlights
- **Metric cache + stride precompute** (major speedup vs per-tick recompute)
- **Gate diagnostics** (`DEBUG GATES`) with entry/exit/closed/open separation
- **Econ filter sub-reasons** (`hurst_fail`, `entropy_fail`, `lle_fail`, `multi_fail`)
- **Indicator distribution stats** (min/p10/p50/p90/max/mean)
- **Auto-calibration** of Hurst/Entropy/LLE thresholds from quantiles
- **Ablation suite** (`HURST_ONLY`, `ENTROPY_ONLY`, `LLE_ONLY`, `H_E_L`)
- **PnL accounting split** (`realized`, `forced`, `open_est`)
- **Optional walk-forward** test split
- **JSON report export** (artifacts/backtest_v4_report.json)
- **PowerShell launcher** to avoid env syntax pain

## PowerShell quick run
```powershell
.un_backtest_v4.ps1
```

## Manual run (PowerShell)
```powershell
$env:BACKTEST_SYNTH_N = "20000"
$env:BACKTEST_METRIC_STRIDE = "5"
$env:BACKTEST_DEBUG_GATES = "1"
$env:BACKTEST_RUN_ABLATION = "1"
$env:BACKTEST_AUTO_CALIBRATE = "1"
$env:BACKTEST_SAVE_REPORT = ".\artifacts\backtest_v4_report.json"
python .acktest.py
```

## Interpreting the output
- `signal_below_threshold` high = signal is too strict
- `econ_filter_block` high = thresholds too tight or incompatible with dataset
- `position_open_no_reentry` high = horizon too long / one-position mode bottleneck
- Compare `OFF`, `ON`, `SANITY`, and ablation labels to see what actually kills entries/PnL

## Notes
- Synthetic harness remains a **sanity/debug tool**, not a production performance estimate.
- If `scipy` is not installed, the alpha-stable generator falls back to a robust Student-t approximation via `utils.sample_alpha_stable`.

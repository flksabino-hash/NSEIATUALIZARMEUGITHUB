param(
  [int]$N = 20000,
  [int]$Stride = 5,
  [switch]$DebugGates = $true,
  [switch]$Sanity = $true,
  [switch]$Ablation = $true,
  [switch]$AutoCalibrate = $true,
  [switch]$WalkForward = $false,
  [switch]$ForceCloseEod = $true,
  [switch]$Grid = $true,
  [switch]$SeedSuite = $true,
  [int]$MultiPos = 3,
  [string]$GridThresholds = "0.35,0.5,0.65",
  [string]$GridSizes = "0.5,1.0,1.5",
  [string]$SeedList = "7,11,17,23,31",
  [int]$BenchRepeat = 1,
  [string]$ReportPath = ".\\artifacts\\backtest_v5_report.json"
)
$env:BACKTEST_SYNTH_N = "$N"
$env:BACKTEST_METRIC_STRIDE = "$Stride"
$env:BACKTEST_DEBUG_GATES = $(if ($DebugGates) { "1" } else { "0" })
$env:BACKTEST_RUN_SANITY = $(if ($Sanity) { "1" } else { "0" })
$env:BACKTEST_RUN_ABLATION = $(if ($Ablation) { "1" } else { "0" })
$env:BACKTEST_AUTO_CALIBRATE = $(if ($AutoCalibrate) { "1" } else { "0" })
$env:BACKTEST_RUN_WALKFORWARD = $(if ($WalkForward) { "1" } else { "0" })
$env:BACKTEST_FORCE_CLOSE_EOD = $(if ($ForceCloseEod) { "1" } else { "0" })
$env:BACKTEST_RUN_GRID = $(if ($Grid) { "1" } else { "0" })
$env:BACKTEST_RUN_SEED_SUITE = $(if ($SeedSuite) { "1" } else { "0" })
$env:BACKTEST_MULTI_MAX_POSITIONS = "$MultiPos"
$env:BACKTEST_GRID_THRESHOLDS = "$GridThresholds"
$env:BACKTEST_GRID_SIZE_MULTS = "$GridSizes"
$env:BACKTEST_SEED_LIST = "$SeedList"
$env:BACKTEST_BENCH_REPEAT = "$BenchRepeat"
$env:BACKTEST_PRECOMPUTE_METRICS = "1"
$env:BACKTEST_DEBUG_INDICATOR_STATS = "1"
$env:BACKTEST_SAVE_REPORT = "$ReportPath"
python .\backtest.py

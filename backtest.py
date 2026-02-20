from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import load_settings
from utils import (
    clamp,
    log_returns,
    sample_alpha_stable,
    shannon_entropy_returns,
)


# -----------------------------
# Env helpers (backtest-only)
# -----------------------------
def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on", "y"}


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return int(default)
    try:
        return int(v)
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return float(default)
    try:
        return float(v)
    except Exception:
        return float(default)


@dataclass(slots=True)
class BacktestOpts:
    BACKTEST_SYNTH_N: int = 20000
    BACKTEST_DEBUG_GATES: bool = False
    BACKTEST_RUN_SANITY: bool = True
    BACKTEST_METRIC_STRIDE: int = 5
    BACKTEST_FORCE_CLOSE_EOD: bool = True
    BACKTEST_RUN_ABLATION: bool = True
    BACKTEST_SIGNAL_THRESHOLD: float = 0.5
    BACKTEST_AUTO_CALIBRATE: bool = True
    BACKTEST_CALIB_HURST_Q: float = 0.60
    BACKTEST_CALIB_ENTROPY_Q: float = 0.80
    BACKTEST_CALIB_LLE_Q: float = 0.80
    BACKTEST_CALIB_ON_SIGNAL_ONLY: bool = True
    BACKTEST_CALIB_MIN_SAMPLES: int = 200
    BACKTEST_PRECOMPUTE_METRICS: bool = True
    BACKTEST_RUN_WALKFORWARD: bool = False
    BACKTEST_WF_CALIB_FRAC: float = 0.40
    BACKTEST_SAVE_REPORT: str = ""
    BACKTEST_DEBUG_INDICATOR_STATS: bool = True
    BACKTEST_DEBUG_SAMPLE_MAX: int = 50000
    BACKTEST_SEED: int = 7
    BACKTEST_CSV: str = ""  # reserved; not used in this standalone synthetic harness
    BACKTEST_TP_BPS: float = 45.0
    BACKTEST_SL_BPS: float = 35.0
    BACKTEST_HORIZON_SEC: int = 300
    BACKTEST_PRICE_SCALE_BPS: float = 7.0  # synthetic return scale

    @classmethod
    def from_env(cls) -> "BacktestOpts":
        d = cls()
        return cls(
            BACKTEST_SYNTH_N=_env_int("BACKTEST_SYNTH_N", d.BACKTEST_SYNTH_N),
            BACKTEST_DEBUG_GATES=_env_bool("BACKTEST_DEBUG_GATES", d.BACKTEST_DEBUG_GATES),
            BACKTEST_RUN_SANITY=_env_bool("BACKTEST_RUN_SANITY", d.BACKTEST_RUN_SANITY),
            BACKTEST_METRIC_STRIDE=max(1, _env_int("BACKTEST_METRIC_STRIDE", d.BACKTEST_METRIC_STRIDE)),
            BACKTEST_FORCE_CLOSE_EOD=_env_bool("BACKTEST_FORCE_CLOSE_EOD", d.BACKTEST_FORCE_CLOSE_EOD),
            BACKTEST_RUN_ABLATION=_env_bool("BACKTEST_RUN_ABLATION", d.BACKTEST_RUN_ABLATION),
            BACKTEST_SIGNAL_THRESHOLD=clamp(_env_float("BACKTEST_SIGNAL_THRESHOLD", d.BACKTEST_SIGNAL_THRESHOLD), 0.0, 0.999),
            BACKTEST_AUTO_CALIBRATE=_env_bool("BACKTEST_AUTO_CALIBRATE", d.BACKTEST_AUTO_CALIBRATE),
            BACKTEST_CALIB_HURST_Q=clamp(_env_float("BACKTEST_CALIB_HURST_Q", d.BACKTEST_CALIB_HURST_Q), 0.01, 0.99),
            BACKTEST_CALIB_ENTROPY_Q=clamp(_env_float("BACKTEST_CALIB_ENTROPY_Q", d.BACKTEST_CALIB_ENTROPY_Q), 0.01, 0.99),
            BACKTEST_CALIB_LLE_Q=clamp(_env_float("BACKTEST_CALIB_LLE_Q", d.BACKTEST_CALIB_LLE_Q), 0.01, 0.99),
            BACKTEST_CALIB_ON_SIGNAL_ONLY=_env_bool("BACKTEST_CALIB_ON_SIGNAL_ONLY", d.BACKTEST_CALIB_ON_SIGNAL_ONLY),
            BACKTEST_CALIB_MIN_SAMPLES=max(20, _env_int("BACKTEST_CALIB_MIN_SAMPLES", d.BACKTEST_CALIB_MIN_SAMPLES)),
            BACKTEST_PRECOMPUTE_METRICS=_env_bool("BACKTEST_PRECOMPUTE_METRICS", d.BACKTEST_PRECOMPUTE_METRICS),
            BACKTEST_RUN_WALKFORWARD=_env_bool("BACKTEST_RUN_WALKFORWARD", d.BACKTEST_RUN_WALKFORWARD),
            BACKTEST_WF_CALIB_FRAC=clamp(_env_float("BACKTEST_WF_CALIB_FRAC", d.BACKTEST_WF_CALIB_FRAC), 0.1, 0.9),
            BACKTEST_SAVE_REPORT=os.getenv("BACKTEST_SAVE_REPORT", d.BACKTEST_SAVE_REPORT),
            BACKTEST_DEBUG_INDICATOR_STATS=_env_bool("BACKTEST_DEBUG_INDICATOR_STATS", d.BACKTEST_DEBUG_INDICATOR_STATS),
            BACKTEST_DEBUG_SAMPLE_MAX=max(1000, _env_int("BACKTEST_DEBUG_SAMPLE_MAX", d.BACKTEST_DEBUG_SAMPLE_MAX)),
            BACKTEST_SEED=_env_int("BACKTEST_SEED", _env_int("BACKTEST_RANDOM_SEED", d.BACKTEST_SEED)),
            BACKTEST_CSV=os.getenv("BACKTEST_CSV", d.BACKTEST_CSV),
            BACKTEST_TP_BPS=max(1.0, _env_float("BACKTEST_TP_BPS", d.BACKTEST_TP_BPS)),
            BACKTEST_SL_BPS=max(1.0, _env_float("BACKTEST_SL_BPS", d.BACKTEST_SL_BPS)),
            BACKTEST_HORIZON_SEC=max(30, _env_int("BACKTEST_HORIZON_SEC", d.BACKTEST_HORIZON_SEC)),
            BACKTEST_PRICE_SCALE_BPS=max(0.1, _env_float("BACKTEST_PRICE_SCALE_BPS", d.BACKTEST_PRICE_SCALE_BPS)),
        )


# -----------------------------
# Metric/signal cache
# -----------------------------
@dataclass(slots=True)
class MetricWindows:
    signal_window: int = 900
    hurst_window: int = 600
    entropy_window: int = 600
    lle_window: int = 800
    mfdfa_window: int = 800
    expiry_horizon: int = 300

    @property
    def max_lookback(self) -> int:
        return max(self.signal_window, self.hurst_window, self.entropy_window, self.lle_window, self.mfdfa_window)


@dataclass(slots=True)
class MetricCache:
    signal_score: np.ndarray
    hurst: np.ndarray
    entropy: np.ndarray
    lle: np.ndarray
    mfdfa: np.ndarray
    warmup: Dict[str, Any]
    build_ms: float


@dataclass(slots=True)
class Thresholds:
    hurst_min: float
    entropy_max: float
    lle_max: float


@dataclass(slots=True)
class Position:
    entry_idx: int
    entry_px: float
    size_usdc: float
    horizon_idx: int
    side: int = 1
    tp_bps: float = 1000.0
    sl_bps: float = 600.0
    regime_entry: str = "unknown"


@dataclass(slots=True)
class ClosedTrade:
    entry_idx: int
    exit_idx: int
    entry_px: float
    exit_px: float
    pnl_usdc: float
    ret_pct: float
    reason: str



def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _qstats(values: Sequence[float]) -> Optional[Dict[str, float]]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return None
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
    }


def _generate_synth_prices(opts: BacktestOpts, alpha_min: float) -> np.ndarray:
    np.random.seed(opts.BACKTEST_SEED)
    n = int(opts.BACKTEST_SYNTH_N)
    alpha = float(np.random.uniform(alpha_min, 2.0))
    # heavy-tailed returns, scaled down to realistic micro-moves
    r = sample_alpha_stable(alpha=alpha, beta=0.0, scale=1.0, size=n - 1)
    r = np.clip(np.asarray(r, dtype=float), -15.0, 15.0)
    r *= (opts.BACKTEST_PRICE_SCALE_BPS / 10_000.0)
    px = np.empty(n, dtype=float)
    px[0] = 100.0
    px[1:] = px[0] * np.exp(np.cumsum(r))
    # ensure sane positive prices
    px = np.maximum(px, 1e-6)
    return px


def _compute_signal_score(price_window: np.ndarray) -> float:
    # Cheap, deterministic score in [0,1]: combines momentum + volatility-normalized drift + breakout tendency.
    r = log_returns(price_window)
    if r.size < 120:
        return 0.5
    std = float(np.std(r) + 1e-12)
    mom_fast = float(np.sum(r[-30:]) / (std * math.sqrt(30.0) + 1e-12))
    mom_slow = float(np.sum(r[-120:]) / (std * math.sqrt(120.0) + 1e-12))
    drift = float(np.mean(r[-120:]))
    z_close = float((price_window[-1] - np.mean(price_window[-120:])) / (np.std(price_window[-120:]) + 1e-12))
    raw = 0.55 * mom_fast + 0.35 * mom_slow + 800.0 * drift + 0.20 * z_close
    return float(clamp(_sigmoid(raw), 0.0, 1.0))


def _fast_hurst_proxy(price_window: np.ndarray) -> float:
    if price_window.size < 80:
        return float('nan')
    lags = np.array([2, 4, 8, 16, 32], dtype=int)
    taus = []
    for lag in lags:
        if lag >= price_window.size:
            break
        d = price_window[lag:] - price_window[:-lag]
        taus.append(float(np.std(d) + 1e-12))
    if len(taus) < 3:
        return float('nan')
    x = np.log(lags[: len(taus)])
    y = np.log(np.asarray(taus))
    slope = float(np.polyfit(x, y, 1)[0])
    return float(np.clip(slope, -1.0, 2.0))


def _fast_entropy_proxy(price_window: np.ndarray) -> float:
    try:
        return float(shannon_entropy_returns(price_window, bins=30))
    except Exception:
        return float('nan')


def _fast_lle_proxy(price_window: np.ndarray) -> float:
    r = log_returns(price_window)
    if r.size < 80:
        return float('nan')
    r = r[-256:] if r.size > 256 else r
    dr = np.diff(r)
    if dr.size < 10:
        return float('nan')
    scale = float(np.std(r) + 1e-12)
    div = float(np.mean(np.abs(dr)) / scale)
    ac_num = float(np.dot(r[:-1], r[1:]))
    ac_den = float(np.linalg.norm(r[:-1]) * np.linalg.norm(r[1:]) + 1e-12)
    ac1 = ac_num / ac_den
    # Negative "stability" score (more negative = more stable) to mirror original threshold semantics.
    score = -math.log1p(div * 8.0) - 0.35 * max(min(ac1, 1.0), -1.0)
    return float(score)


def _fast_mfdfa_proxy(price_window: np.ndarray) -> float:
    r = log_returns(price_window)
    if r.size < 64:
        return float('nan')
    long = float(np.std(r[-256:]) + 1e-12) if r.size >= 256 else float(np.std(r) + 1e-12)
    mid = float(np.std(r[-128:]) + 1e-12) if r.size >= 128 else long
    short = float(np.std(r[-32:]) + 1e-12)
    return float(1.0 + short / mid + mid / long)


def build_metric_cache(prices: np.ndarray, opts: BacktestOpts, mw: MetricWindows) -> MetricCache:
    n = prices.size
    stride = max(1, int(opts.BACKTEST_METRIC_STRIDE))
    warmup = {
        "signal_window": mw.signal_window,
        "hurst_window": mw.hurst_window,
        "entropy_window": mw.entropy_window,
        "lle_window": mw.lle_window,
        "expiry_horizon": mw.expiry_horizon,
        "max_lookback": mw.max_lookback,
        "ticks_after_warmup": max(0, int(n - (mw.max_lookback + 1))),
    }

    sig = np.full(n, np.nan, dtype=float)
    hurst = np.full(n, np.nan, dtype=float)
    entropy = np.full(n, np.nan, dtype=float)
    lle = np.full(n, np.nan, dtype=float)
    mfd = np.full(n, np.nan, dtype=float)

    t0 = time.perf_counter()
    start = mw.max_lookback
    for i in range(start, n, stride):
        try:
            sig[i] = _compute_signal_score(prices[i - mw.signal_window:i])
        except Exception:
            sig[i] = np.nan
        # Metric fallbacks are explicit NaN, not fake 0.0
        try:
            hurst[i] = _fast_hurst_proxy(prices[i - mw.hurst_window:i])
        except Exception:
            hurst[i] = np.nan
        try:
            entropy[i] = _fast_entropy_proxy(prices[i - mw.entropy_window:i])
        except Exception:
            entropy[i] = np.nan
        try:
            lle[i] = _fast_lle_proxy(prices[i - mw.lle_window:i])
        except Exception:
            lle[i] = np.nan
        try:
            mfd[i] = _fast_mfdfa_proxy(prices[i - mw.mfdfa_window:i])
        except Exception:
            mfd[i] = np.nan

    # Forward fill to reduce per-tick metric cost.
    if stride > 1:
        for arr in (sig, hurst, entropy, lle, mfd):
            last = np.nan
            for i in range(n):
                if not np.isnan(arr[i]):
                    last = arr[i]
                elif i >= start and not np.isnan(last):
                    arr[i] = last

    ms = (time.perf_counter() - t0) * 1000.0
    print("[BT] metric cache built:", {"ms": round(ms, 2), "metric_stride": stride, "warmup": warmup})
    return MetricCache(signal_score=sig, hurst=hurst, entropy=entropy, lle=lle, mfdfa=mfd, warmup=warmup, build_ms=ms)


def _calibration_mask(cache: MetricCache, mw: MetricWindows, start_idx: int, end_idx: int, signal_threshold: float, on_signal_only: bool) -> np.ndarray:
    idx = np.arange(cache.signal_score.size)
    m = (idx >= start_idx) & (idx < end_idx) & (idx >= mw.max_lookback)
    m &= ~np.isnan(cache.hurst) & ~np.isnan(cache.entropy) & ~np.isnan(cache.lle)
    if on_signal_only:
        m &= (cache.signal_score >= signal_threshold)
    return m


def auto_calibrate_thresholds(settings, opts: BacktestOpts, cache: MetricCache, mw: MetricWindows, start_idx: int, end_idx: int) -> Tuple[Thresholds, Dict[str, int]]:
    base = Thresholds(hurst_min=float(settings.HURST_MIN), entropy_max=float(settings.ENTROPY_MAX), lle_max=float(settings.LLE_MAX))
    mask = _calibration_mask(cache, mw, start_idx, end_idx, opts.BACKTEST_SIGNAL_THRESHOLD, opts.BACKTEST_CALIB_ON_SIGNAL_ONLY)
    h = cache.hurst[mask]
    e = cache.entropy[mask]
    l = cache.lle[mask]
    counts = {"hurst": int(h.size), "entropy": int(e.size), "lle": int(l.size)}
    if not opts.BACKTEST_AUTO_CALIBRATE:
        return base, counts
    if min(counts.values()) < int(opts.BACKTEST_CALIB_MIN_SAMPLES):
        print("[BT] AUTO CALIBRATION SKIPPED (insufficient samples):", counts)
        return base, counts
    th = Thresholds(
        hurst_min=float(np.quantile(h, opts.BACKTEST_CALIB_HURST_Q)),
        entropy_max=float(np.quantile(e, opts.BACKTEST_CALIB_ENTROPY_Q)),
        lle_max=float(np.quantile(l, opts.BACKTEST_CALIB_LLE_Q)),
    )
    print("[BT] AUTO CALIBRATION APPLIED:", {"HURST_MIN": th.hurst_min, "ENTROPY_MAX": th.entropy_max, "LLE_MAX": th.lle_max})
    print("[BT] calibration sample counts:", counts)
    return th, counts


def _realized_sharpe(trades: Sequence[ClosedTrade]) -> float:
    if len(trades) < 2:
        return 0.0
    rets = np.asarray([t.ret_pct for t in trades], dtype=float)
    sd = float(np.std(rets) + 1e-12)
    return float(np.mean(rets) / sd * math.sqrt(len(rets)))


def _apply_exit(position: Position, exit_idx: int, exit_px: float, settings, reason: str) -> ClosedTrade:
    gross_ret = (float(exit_px) / max(float(position.entry_px), 1e-12)) - 1.0
    # Round-trip trading cost estimate in return terms.
    cost_ret = 2.0 * (float(settings.TAKER_FEE_BPS) + float(settings.SLIPPAGE_BPS_EST)) / 10_000.0
    net_ret = gross_ret - cost_ret
    pnl = float(position.size_usdc) * net_ret
    return ClosedTrade(
        entry_idx=int(position.entry_idx),
        exit_idx=int(exit_idx),
        entry_px=float(position.entry_px),
        exit_px=float(exit_px),
        pnl_usdc=float(pnl),
        ret_pct=float(net_ret),
        reason=reason,
    )


def _classify_regime(h: float, e: float, l: float) -> str:
    if any(np.isnan(x) for x in (h, e, l)):
        return "unknown"
    if e > 2.4:
        return "contagion"
    if h > 0.52 and l < -0.75:
        return "trend"
    if h < 0.40 and e < 1.6:
        return "mean_revert"
    return "neutral"


def _regime_exit_profile(opts: BacktestOpts, regime: str) -> Tuple[float, float, int]:
    base_tp = 1000.0
    base_sl = 600.0
    base_h = int(opts.BACKTEST_HORIZON_SEC)
    if not bool(getattr(opts, "BACKTEST_ENABLE_REGIME_EXITS", True)):
        return base_tp, base_sl, base_h
    if regime == "trend":
        return base_tp * 1.35, base_sl * 0.90, max(60, int(base_h * 1.30))
    if regime == "contagion":
        return base_tp * 0.70, base_sl * 0.70, max(60, int(base_h * 0.55))
    if regime == "mean_revert":
        return base_tp * 0.85, base_sl * 0.80, max(60, int(base_h * 0.75))
    return base_tp, base_sl, base_h


def _parse_float_list(s: str, default: List[float]) -> List[float]:
    try:
        vals = [float(x.strip()) for x in str(s).split(",") if str(x).strip()]
        return vals or list(default)
    except Exception:
        return list(default)


def _parse_int_list(s: str, default: List[int]) -> List[int]:
    try:
        vals = [int(x.strip()) for x in str(s).split(",") if str(x).strip()]
        return vals or list(default)
    except Exception:
        return list(default)


def _gate_label(filters_on: bool, sanity_mode: bool, label: Optional[str]) -> str:
    if label:
        return label
    if sanity_mode:
        return "SANITY"
    return "ON" if filters_on else "OFF"


def run_backtest(
    prices: np.ndarray,
    settings,
    opts: BacktestOpts,
    cache: MetricCache,
    thresholds: Thresholds,
    *,
    filters_on: bool,
    sanity_mode: bool = False,
    label: str = "",
    use_hurst: Optional[bool] = None,
    use_entropy: Optional[bool] = None,
    use_lle: Optional[bool] = None,
    start_idx: Optional[int] = None,
    end_idx: Optional[int] = None,
    signal_threshold: Optional[float] = None,
    size_mult: Optional[float] = None,
) -> Dict[str, Any]:
    mw = _warmup_windows(opts)
    start = mw.max_lookback if start_idx is None else max(int(start_idx), mw.max_lookback)
    end = prices.size if end_idx is None else min(int(end_idx), prices.size)
    gate_label = _gate_label(filters_on, sanity_mode, label)
    sig_thr = float(opts.BACKTEST_SIGNAL_THRESHOLD if signal_threshold is None else signal_threshold)
    size_mult = float(opts.BACKTEST_SIZE_MULT if size_mult is None else size_mult)
    max_pos = max(1, int(getattr(opts, "BACKTEST_MULTI_MAX_POSITIONS", 1)))

    print("[BT] warmup check:", {
        "N_sintetico": int(prices.size), "signal_window": mw.signal_window, "hurst_window": mw.hurst_window,
        "entropy_window": mw.entropy_window, "lle_window": mw.lle_window, "expiry_horizon": mw.expiry_horizon,
        "max_lookback": mw.max_lookback, "ticks_after_warmup": max(0, end-start), "metric_stride": int(opts.BACKTEST_METRIC_STRIDE),
        "filters": "ON" if filters_on else "OFF", "sanity_mode": bool(sanity_mode), "label": gate_label,
        "signal_threshold": sig_thr, "size_mult": size_mult, "max_positions": max_pos,
    })

    use_h = bool(settings.USE_HURST if use_hurst is None else use_hurst)
    use_e = bool(settings.USE_ENTROPY if use_entropy is None else use_entropy)
    use_l = bool(settings.USE_LLE if use_lle is None else use_lle)

    debug_counts = {"warmup": int(start), "regime_block": 0, "signal_below_threshold": 0, "econ_filter_block": 0, "position_open_no_reentry": 0, "entry_ok": 0, "exit_ok": 0, "metric_nan": 0, "insufficient_balance": 0, "forced_exit_eod": 0}
    econ_block_reasons = {"hurst_fail": 0, "entropy_fail": 0, "lle_fail": 0, "te_fail": 0, "mfdfa_fail": 0, "multi_fail": 0}
    indicator_samples = {"hurst": [], "entropy": [], "lle": [], "mfdfa": []}

    positions: List[Position] = []
    closed_trades: List[ClosedTrade] = []
    balance = float(settings.PAPER_START_USDC)
    entry_count = 0
    exit_count = 0

    t_loop = time.perf_counter()
    for i in range(start, end):
        px = float(prices[i])
        h = float(cache.hurst[i]) if not np.isnan(cache.hurst[i]) else np.nan
        e = float(cache.entropy[i]) if not np.isnan(cache.entropy[i]) else np.nan
        l = float(cache.lle[i]) if not np.isnan(cache.lle[i]) else np.nan
        mfd = float(cache.mfdfa[i]) if not np.isnan(cache.mfdfa[i]) else np.nan
        sig = float(cache.signal[i]) if not np.isnan(cache.signal[i]) else 0.0

        # exits across all positions
        if positions:
            survivors = []
            for pos in positions:
                ret = (px / max(float(pos.entry_px), 1e-12)) - 1.0
                tp_ret = float(pos.tp_bps) / 10_000.0
                sl_ret = -float(pos.sl_bps) / 10_000.0
                if ret >= tp_ret:
                    tr = _apply_exit(pos, i, px, settings, "tp")
                    closed_trades.append(tr); balance += tr.pnl_usdc; exit_count += 1; debug_counts["exit_ok"] += 1
                elif ret <= sl_ret:
                    tr = _apply_exit(pos, i, px, settings, "sl")
                    closed_trades.append(tr); balance += tr.pnl_usdc; exit_count += 1; debug_counts["exit_ok"] += 1
                elif i >= int(pos.horizon_idx):
                    tr = _apply_exit(pos, i, px, settings, "expiry")
                    closed_trades.append(tr); balance += tr.pnl_usdc; exit_count += 1; debug_counts["exit_ok"] += 1
                else:
                    survivors.append(pos)
            positions = survivors

        if not np.isnan(h): indicator_samples["hurst"].append(h)
        if not np.isnan(e): indicator_samples["entropy"].append(e)
        if not np.isnan(l): indicator_samples["lle"].append(l)
        if not np.isnan(mfd): indicator_samples["mfdfa"].append(mfd)

        if not sanity_mode and sig < sig_thr:
            debug_counts["signal_below_threshold"] += 1
            continue

        if len(positions) >= max_pos:
            debug_counts["position_open_no_reentry"] += 1
            continue

        if filters_on and not sanity_mode:
            fails=[]
            if (use_h and np.isnan(h)) or (use_e and np.isnan(e)) or (use_l and np.isnan(l)):
                debug_counts["metric_nan"] += 1
                continue
            if use_h and h < thresholds.hurst_min: fails.append("hurst_fail")
            if use_e and e > thresholds.entropy_max: fails.append("entropy_fail")
            if use_l and l > thresholds.lle_max: fails.append("lle_fail")
            if fails:
                debug_counts["econ_filter_block"] += 1
                if len(fails) > 1: econ_block_reasons["multi_fail"] += 1
                else: econ_block_reasons[fails[0]] += 1
                continue

        # size scales with grid multiplier and signal excess (small adaptive kicker)
        sig_kicker = 1.0 + max(0.0, sig - sig_thr) * 0.5
        req_size = float(settings.MAX_USDC_PER_TRADE) * float(size_mult) * sig_kicker
        size = min(req_size, balance)
        if size <= 0:
            debug_counts["insufficient_balance"] += 1
            continue

        regime = _classify_regime(h, e, l)
        tp_bps, sl_bps, hsec = _regime_exit_profile(opts, regime)
        positions.append(Position(entry_idx=i, entry_px=px, size_usdc=size, horizon_idx=min(end-1, i+int(hsec)), tp_bps=float(tp_bps), sl_bps=float(sl_bps), regime_entry=regime))
        entry_count += 1
        debug_counts["entry_ok"] += 1

    loop_s = time.perf_counter() - t_loop
    forced_close_pnl = 0.0
    if opts.BACKTEST_FORCE_CLOSE_EOD and positions:
        last_px = float(prices[end-1])
        for pos in positions:
            tr = _apply_exit(pos, end-1, last_px, settings, "eod_force")
            closed_trades.append(tr); balance += tr.pnl_usdc; exit_count += 1; debug_counts["exit_ok"] += 1; debug_counts["forced_exit_eod"] += 1; forced_close_pnl += tr.pnl_usdc
        positions = []
    open_pnl_est = 0.0
    if positions:
        last_px=float(prices[end-1])
        for pos in positions:
            open_pnl_est += pos.size_usdc * ((last_px / max(pos.entry_px,1e-12))-1.0)

    realized = float(sum(t.pnl_usdc for t in closed_trades if t.reason != "eod_force"))
    sharpe = _realized_sharpe(closed_trades)
    throughput = (max(0, end-start) / max(loop_s, 1e-9))

    if opts.BACKTEST_DEBUG_GATES:
        print("DEBUG GATES:", debug_counts)
        print(f"entries: {entry_count} exits: {exit_count} closed_trades: {len(closed_trades)} open_positions_end: {len(positions)}")
        if filters_on and not sanity_mode:
            print("ECON BLOCK REASONS:", econ_block_reasons)
            if debug_counts["econ_filter_block"] > 0:
                print("ECON BLOCK RATES:", {k: round(v/debug_counts["econ_filter_block"],4) for k,v in econ_block_reasons.items() if v>0})
    if opts.BACKTEST_DEBUG_INDICATOR_STATS:
        print("HURST stats:", _qstats(indicator_samples["hurst"]), "threshold min =", thresholds.hurst_min)
        print("ENTROPY stats:", _qstats(indicator_samples["entropy"]), "threshold max =", thresholds.entropy_max)
        print("LLE stats:", _qstats(indicator_samples["lle"]), "threshold max =", thresholds.lle_max)
        print("MFD proxy stats:", _qstats(indicator_samples["mfdfa"]))

    title = gate_label if not sanity_mode else "SANITY / no-gates"
    print(f"Trades ({title}): {len(closed_trades)}")
    print(f"  entries/exits/closed/open: {entry_count} {exit_count} {len(closed_trades)} {len(positions)}")
    print(f"  PnL realized/forced/open_est: {round(realized,4)} {round(forced_close_pnl,4)} {round(open_pnl_est,4)}")
    print(f"  Sharpe: {round(sharpe,4)} Final balance: {round(balance,4)}")
    print(f"  Loop throughput: {round(throughput,1)} ticks/s ({round(loop_s*1000,2)} ms)")

    return {
        "label": title, "entries": entry_count, "exits": exit_count, "closed_trades": len(closed_trades), "open_positions_end": len(positions),
        "debug_gates": debug_counts, "econ_block_reasons": econ_block_reasons,
        "pnl_realized_usdc": float(realized), "pnl_forced_usdc": float(forced_close_pnl), "open_pnl_est_usdc": float(open_pnl_est),
        "balance_final": float(balance), "sharpe": float(sharpe),
        "throughput_ticks_per_sec": float(throughput), "loop_ms": float(loop_s*1000),
        "signal_threshold": float(sig_thr), "size_mult": float(size_mult), "max_positions": int(max_pos),
        "thresholds_used": asdict(thresholds),
        "indicator_stats": {k: _qstats(v) for k,v in indicator_samples.items()},
    }


def _print_header(settings, opts: BacktestOpts):
    merged = asdict(settings)
    merged.update(asdict(opts))
    print("SETTINGS:", merged)
    print("Synthetic Backtest (heavy-tailed alpha-stable returns)")


def _safe_write_json(path: str, data: Dict[str, Any]) -> None:
    if not path:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print("[BT] report saved:", str(p))


def _run_grid_suite(prices, settings, opts, cache, thresholds_base, report):
    if not opts.BACKTEST_RUN_GRID:
        return
    print("\nGrid suite (threshold x size)")
    best = None
    for thr in _parse_float_list(opts.BACKTEST_GRID_THRESHOLDS, [0.35, 0.5, 0.65]):
        for sm in _parse_float_list(opts.BACKTEST_GRID_SIZE_MULTS, [0.5, 1.0, 1.5]):
            res = run_backtest(prices, settings, opts, cache, thresholds_base, filters_on=True, label=f"GRID_t{thr:.2f}_s{sm:.2f}", signal_threshold=thr, size_mult=sm)
            report["runs"].append(res)
            if best is None or (res["balance_final"], res["sharpe"]) > (best["balance_final"], best["sharpe"]):
                best = dict(res)
    report["grid_best"] = best
    if best:
        print("[BT] GRID BEST:", {"label": best["label"], "balance_final": round(best["balance_final"],4), "sharpe": round(best["sharpe"],4)})


def _run_seed_suite(settings, opts, report):
    if not opts.BACKTEST_RUN_SEED_SUITE:
        return
    print("\nSeed robustness suite")
    rows = []
    for seed in _parse_int_list(opts.BACKTEST_SEED_LIST, [7,11,17]):
        o = replace(opts, BACKTEST_SEED=int(seed), BACKTEST_DEBUG_GATES=False, BACKTEST_DEBUG_INDICATOR_STATS=False)
        prices = _generate_synth_prices(o, float(getattr(settings, "LEVY_ALPHA_MIN", 1.2)))
        mw = MetricWindows(expiry_horizon=int(o.BACKTEST_HORIZON_SEC))
        cache = build_metric_cache(prices, o, mw)
        th, _ = auto_calibrate_thresholds(settings, o, cache, mw, 0, prices.size)
        res = run_backtest(prices, settings, o, cache, th, filters_on=True, label=f"SEED_{seed}")
        res["seed"] = int(seed)
        rows.append(res)
        report["runs"].append(res)
    if rows:
        arr_bal = np.array([r["balance_final"] for r in rows], dtype=float)
        arr_sh = np.array([r["sharpe"] for r in rows], dtype=float)
        report["seed_robustness"] = {
            "n": int(len(rows)),
            "seed_list": [int(r["seed"]) for r in rows],
            "balance_mean": float(arr_bal.mean()), "balance_median": float(np.median(arr_bal)), "balance_min": float(arr_bal.min()), "balance_max": float(arr_bal.max()),
            "sharpe_mean": float(arr_sh.mean()),
        }
        print("[BT] SEED ROBUSTNESS:", {k:(round(v,4) if isinstance(v,float) else v) for k,v in report["seed_robustness"].items()})


def _run_throughput_bench(prices, settings, opts, cache, thresholds_base, report):
    reps = max(1, int(opts.BACKTEST_BENCH_REPEAT))
    durs = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = run_backtest(prices, settings, replace(opts, BACKTEST_DEBUG_GATES=False, BACKTEST_DEBUG_INDICATOR_STATS=False), cache, thresholds_base, filters_on=False, label="BENCH")
        durs.append(time.perf_counter() - t0)
    report["throughput_bench"] = {"repeat": reps, "mean_wall_ms": float(np.mean(durs)*1000.0), "ticks_per_sec_wall": float(prices.size / max(np.mean(durs),1e-9))}
    print("[BT] THROUGHPUT BENCH:", {k:(round(v,2) if isinstance(v,float) else v) for k,v in report["throughput_bench"].items()})


def main() -> None:
    settings = load_settings()
    opts = BacktestOpts.from_env()
    _print_header(settings, opts)

    prices = _generate_synth_prices(opts, float(getattr(settings, "LEVY_ALPHA_MIN", 1.2)))
    mw = MetricWindows(expiry_horizon=int(opts.BACKTEST_HORIZON_SEC))

    if opts.BACKTEST_PRECOMPUTE_METRICS:
        cache = build_metric_cache(prices, opts, mw)
    else:
        # Fallback still builds cache (single pass) because backtest loop expects arrays.
        cache = build_metric_cache(prices, opts, mw)

    # calibration on full period by default, but can be used for WF split too.
    thresholds_base, calib_counts = auto_calibrate_thresholds(settings, opts, cache, mw, 0, prices.size)

    report: Dict[str, Any] = {
        "meta": {
            "version": "v0_5_maxverstappen",
            "settings": {**asdict(settings), **asdict(opts)},
            "metric_cache": {"build_ms": cache.build_ms, "warmup": cache.warmup},
            "calibration_counts": calib_counts,
            "base_thresholds": asdict(thresholds_base),
        },
        "runs": [],
    }

    # Standard suite
    run_off = run_backtest(prices, settings, opts, cache, thresholds_base, filters_on=False, sanity_mode=False, label="OFF")
    report["runs"].append(run_off)

    run_on = run_backtest(prices, settings, opts, cache, thresholds_base, filters_on=True, sanity_mode=False, label="ON")
    report["runs"].append(run_on)

    if opts.BACKTEST_RUN_SANITY:
        run_sanity = run_backtest(prices, settings, opts, cache, thresholds_base, filters_on=False, sanity_mode=True, label="SANITY")
        report["runs"].append(run_sanity)

    if opts.BACKTEST_RUN_ABLATION:
        print("\nAblation suite (single-filter attribution)")
        suites = [
            ("HURST_ONLY", True, False, False),
            ("ENTROPY_ONLY", False, True, False),
            ("LLE_ONLY", False, False, True),
            ("H_E_L", True, True, True),
        ]
        for lbl, uh, ue, ul in suites:
            rr = run_backtest(
                prices,
                settings,
                opts,
                cache,
                thresholds_base,
                filters_on=True,
                sanity_mode=False,
                label=lbl,
                use_hurst=uh,
                use_entropy=ue,
                use_lle=ul,
            )
            report["runs"].append(rr)

    if opts.BACKTEST_RUN_WALKFORWARD:
        print("\nWalk-forward split")
        split = int(prices.size * float(opts.BACKTEST_WF_CALIB_FRAC))
        wf_thresholds, wf_counts = auto_calibrate_thresholds(settings, opts, cache, mw, 0, split)
        rr = run_backtest(
            prices,
            settings,
            opts,
            cache,
            wf_thresholds,
            filters_on=True,
            sanity_mode=False,
            label="WALKFORWARD_TEST",
            start_idx=split,
            end_idx=prices.size,
        )
        rr["walkforward"] = {"split": split, "calib_counts": wf_counts, "thresholds": asdict(wf_thresholds)}
        report["runs"].append(rr)

    _run_grid_suite(prices, settings, opts, cache, thresholds_base, report)
    _run_seed_suite(settings, opts, report)
    _run_throughput_bench(prices, settings, opts, cache, thresholds_base, report)
    _safe_write_json(opts.BACKTEST_SAVE_REPORT, report)


if __name__ == "__main__":
    main()

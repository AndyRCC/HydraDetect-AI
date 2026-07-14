# SPDX-License-Identifier: MPL-2.0
"""
HydraDetect-AI — standalone FastAPI backend.

Adapted from the Arduino Q field backend (`HydraDetectAI-LittleBrother`) to run
on any generic Python host (Render, Hugging Face Spaces, Fly.io, …) so the web
frontend on GitHub Pages can call it.

The feature extraction and the voting ensemble are kept BIT-FOR-BIT identical to
the device, so verdicts match the real Little Brother. The only differences vs.
the device file are: standard FastAPI app instead of the Arduino WebUI brick,
CORS enabled for the Pages origin, and an optional bundled default model so a
visitor only needs to upload a CSV.
"""
import os
import io
import sys
import logging
import traceback as tb
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq
import pywt

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger("waterhammer-edge")

# A model is shipped with the backend so visitors only upload a CSV.
DEFAULT_MODEL_PATH = Path(__file__).parent / "model" / "GOGETA.joblib"


# =====================================================================
# FEATURE EXTRACTION  — identical to the Little Brother device
# =====================================================================
def extract_features(t, p, fs):
    features = {}
    p0 = np.median(p[:max(1, int(0.05 * len(p)))])
    features['baseline'] = float(p0)

    peak_idx = int(np.argmax(p))
    features['peak_amp'] = float(p[peak_idx] - p0)
    features['t_peak'] = float(t[peak_idx]) if len(t) > 0 else 0.0

    rms = float(np.sqrt(np.mean((p - p0) ** 2)))
    features['rms']   = rms
    features['crest'] = float(np.max(np.abs(p - p0)) / (rms + 1e-9))

    try:
        idx_fit = (t >= t[peak_idx]) & (t <= t[peak_idx] + 1.0)
        if np.sum(idx_fit) > 10:
            env = np.abs(p[idx_fit] - p0)
            env[env <= 1e-6] = 1e-6
            def expo(x, a, tau): return a * np.exp(-(x - t[peak_idx]) / tau)
            popt, _ = curve_fit(expo, t[idx_fit], env, p0=[env[0], 0.3], maxfev=5000)
            features['decay_tau'] = float(max(popt[1], 1e-3))
        else:
            features['decay_tau'] = 0.0
    except Exception:
        features['decay_tau'] = 0.0

    N = len(p)
    if N > 1:
        yf = np.abs(rfft(p - p0))
        xf = rfftfreq(N, 1.0 / fs)
        features['energy_total'] = float(np.sum(yf ** 2))
        for (a, b) in [(0, 20), (20, 100), (100, 500), (500, 1000)]:
            mask = (xf >= a) & (xf < b)
            features[f'energy_band_{a}_{b}'] = float(np.sum(yf[mask] ** 2)) if np.any(mask) else 0.0
        features['dom_freq'] = float(xf[np.argmax(yf)]) if len(xf) > 0 else 0.0
    else:
        features['energy_total'] = 0.0
        features['dom_freq'] = 0.0
        for (a, b) in [(0, 20), (20, 100), (100, 500), (500, 1000)]:
            features[f'energy_band_{a}_{b}'] = 0.0

    try:
        coeffs = pywt.wavedec(p - p0, 'db4', level=4)
        for i, c in enumerate(coeffs):
            features[f'wavelet_E_{i}'] = float(np.sum(np.array(c) ** 2))
    except Exception:
        for i in range(5):
            features[f'wavelet_E_{i}'] = 0.0

    return features


# =====================================================================
# DEFAULT CLEANUP FILTERS  — verbatim from HydroAnalyzer desktop (UNIFIEDVX.py).
# Single source of truth for the signal cleaning. The 3 active stages with the
# program's default parameters: neighbor-difference → Hampel → IQR envelope.
# =====================================================================
def neighbor_diff_filter(x, threshold_sigmas=4.0, neighbor_agree_ratio=0.5, max_passes=2):
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return x.copy()
    diffs = np.diff(x)
    mad = np.median(np.abs(diffs - np.median(diffs)))
    noise_std = max(float(mad) * 1.4826, 1e-9)
    threshold_eff = threshold_sigmas * noise_std
    out = x.copy()
    for _ in range(max(1, int(max_passes))):
        left, right, center = out[:-2], out[2:], out[1:-1]
        avg = 0.5 * (left + right)
        delta_central = np.abs(center - avg)
        delta_neighbors = np.abs(right - left)
        spike_inner = (delta_central > threshold_eff) & (delta_neighbors < delta_central * neighbor_agree_ratio)
        if not spike_inner.any():
            break
        new_center = np.where(spike_inner, avg, center)
        out_new = out.copy(); out_new[1:-1] = new_center; out = out_new
    return out


def hampel_filter(x, window_size=7, n_sigmas=3.0):
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return x.copy()
    if window_size % 2 == 0:
        window_size += 1
    s = pd.Series(x)
    rolling_med = s.rolling(window_size, center=True, min_periods=1).median()
    diff = (s - rolling_med).abs()
    rolling_mad = diff.rolling(window_size, center=True, min_periods=1).median()
    threshold = n_sigmas * 1.4826 * rolling_mad
    outlier_mask = ((diff > threshold) & (rolling_mad > 0)).to_numpy()
    return np.where(outlier_mask, rolling_med.to_numpy(), x)


def iqr_envelope_filter(x, window_size=31, n_sigmas=3.0, max_passes=3,
                        upper_pct=75.0, lower_pct=25.0,
                        protect_transients=True, protect_window=201):
    x = np.asarray(x, dtype=float)
    if len(x) < 5:
        return x.copy()
    if window_size % 2 == 0: window_size += 1
    if protect_window % 2 == 0: protect_window += 1
    if protect_transients:
        s_orig = pd.Series(x)
        wide_med = s_orig.rolling(protect_window, center=True, min_periods=1).median()
        wide_diff = wide_med.diff().abs()
        valid = wide_diff.dropna()
        thresh = float(np.quantile(valid, 0.95)) if len(valid) > 0 else 0.0
        transient = (wide_diff.to_numpy() > thresh) & (thresh > 1e-9)
        pad = max(window_size, 21)
        kernel = np.ones(pad, dtype=float)
        protect_mask = np.convolve(transient.astype(float), kernel, mode='same') > 0
    else:
        protect_mask = np.zeros_like(x, dtype=bool)
    out = x.copy()
    for _ in range(max(1, int(max_passes))):
        s = pd.Series(out)
        q3 = s.rolling(window_size, center=True, min_periods=1).quantile(upper_pct / 100).to_numpy()
        q1 = s.rolling(window_size, center=True, min_periods=1).quantile(lower_pct / 100).to_numpy()
        med = s.rolling(window_size, center=True, min_periods=1).median().to_numpy()
        iqr = q3 - q1
        upper = q3 + n_sigmas * iqr
        lower = q1 - n_sigmas * iqr
        mask = ((out > upper) | (out < lower)) & (iqr > 1e-9) & (~protect_mask)
        if not mask.any():
            break
        out = np.where(mask, med, out)
    return out


def apply_default_filter(p):
    """HydroAnalyzer's default cleanup: the 3 active stages with default params."""
    out = neighbor_diff_filter(p, 4.0, 0.5, 2)
    out = hampel_filter(out, 7, 3.0)
    out = iqr_envelope_filter(out, 31, 3.0, 3, 75.0, 25.0, True, 201)
    return out


def parse_signal(csv_bytes):
    """Robust line-by-line parser (port of the desktop's load_csv_signal).

    Keeps only lines that are exactly `<float><sep><float>`, so it tolerates
    PuTTY/serial log banners, headers (time_s,pressure_bar), comments and
    trailing junk. Separators: comma, semicolon, tab, or whitespace.
    """
    text = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = csv_bytes.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = csv_bytes.decode("utf-8", errors="replace")

    times, pres = [], []
    for line in text.splitlines():
        s = line.strip()
        if not s or s[0] == "#" or s.startswith("//"):
            continue
        parts = None
        for sep in (",", ";", "\t"):
            if sep in s:
                parts = [x.strip() for x in s.split(sep) if x.strip() != ""]
                break
        if parts is None:
            parts = s.split()
        if len(parts) != 2:
            continue
        try:
            a = float(parts[0]); b = float(parts[1])
        except ValueError:
            continue
        times.append(a); pres.append(b)

    t = np.asarray(times, dtype=float)
    p = np.asarray(pres, dtype=float)
    if len(t) >= 2 and not np.all(np.diff(t) >= 0):
        order = np.argsort(t, kind="mergesort")
        t, p = t[order], p[order]
    return t, p


# =====================================================================
# APP
# =====================================================================
app = FastAPI(title="HydraDetect-AI", version="1.0.0",
              description="Water-hammer bypass detection — same model as the Little Brother device.")

# CORS — allow the GitHub Pages site (override with ALLOWED_ORIGINS, comma-separated).
_allowed = os.environ.get("ALLOWED_ORIGINS", "*").strip()
_origins = ["*"] if _allowed == "*" else [o.strip() for o in _allowed.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_KEYS = ('rf', 'svm', 'xgb', 'lgbm')


def _err(message, step=None, exc=None):
    payload = {"status": "error", "message": message}
    if step:
        payload["step"] = step
    if exc is not None:
        payload["traceback"]      = tb.format_exc()
        payload["exception_type"] = type(exc).__name__
        payload["exception_str"]  = str(exc)
    return payload


def _try_version(module_name: str) -> str:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "installed")
    except Exception:
        return "not installed"


@app.get("/")
def root():
    return {
        "service": "HydraDetect-AI",
        "status": "ok",
        "message": "Water-hammer bypass detector. POST a CSV to /api/analyze.",
        "endpoints": ["/api/ping", "/api/analyze", "/api/inspect_model", "/docs"],
        "default_model": DEFAULT_MODEL_PATH.name if DEFAULT_MODEL_PATH.exists() else None,
    }


@app.get("/api/ping")
async def ping_endpoint():
    xgboost_ver  = _try_version("xgboost")
    lightgbm_ver = _try_version("lightgbm")
    supported = ["rf", "svm"]
    if "not installed" not in xgboost_ver:  supported.append("xgb")
    if "not installed" not in lightgbm_ver: supported.append("lgbm")
    return {
        "status":   "ok",
        "message":  "Backend online",
        "python":   sys.version.split()[0],
        "numpy":    np.__version__,
        "pandas":   pd.__version__,
        "joblib":   joblib.__version__,
        "sklearn":  _try_version("sklearn"),
        "xgboost":  xgboost_ver,
        "lightgbm": lightgbm_ver,
        "supported_models": supported,
        "default_model_available": DEFAULT_MODEL_PATH.exists(),
    }


@app.post("/api/analyze")
async def analyze_endpoint(
    csv_file:   UploadFile = File(...),
    model_file: Optional[UploadFile] = File(None),
    apply_filter: bool = Form(False),
):
    step = "start"
    try:
        # 1. CSV bytes
        step = "read_csv_bytes"
        try:
            csv_bytes = await csv_file.read()
        except Exception as e:
            return _err(f"Could not read CSV bytes: {e}", step=step, exc=e)
        if len(csv_bytes) == 0:
            return _err("The received CSV has 0 bytes.", step=step)

        # 2. Model bytes — uploaded, or fall back to the bundled default
        step = "read_model_bytes"
        used_default = False
        if model_file is not None:
            model_bytes = await model_file.read()
            if len(model_bytes) == 0:
                model_file = None
        if model_file is None:
            if not DEFAULT_MODEL_PATH.exists():
                return _err("No model uploaded and no bundled model available.", step=step)
            model_bytes = DEFAULT_MODEL_PATH.read_bytes()
            used_default = True

        # 3. Load model
        step = "load_model"
        try:
            model_data = joblib.load(io.BytesIO(model_bytes))
        except ModuleNotFoundError as e:
            missing = str(e).split("'")[1] if "'" in str(e) else str(e)
            return _err(f"The .joblib needs the library '{missing}', which is not installed "
                        f"on the server. Add it to requirements.txt and redeploy.", step=step, exc=e)
        except Exception as e:
            return _err(f"joblib could not deserialize the model: {e}", step=step, exc=e)

        # 4. Extract model components
        if isinstance(model_data, dict):
            md = model_data.get('models') if isinstance(model_data.get('models'), dict) else {}
            loaded_models = {
                k: (md.get(k) or model_data.get(k))
                for k in MODEL_KEYS
                if (md.get(k) is not None) or (model_data.get(k) is not None)
            }
            scaler  = model_data.get('scaler')
            f_names = model_data.get('feature_names')
        else:
            loaded_models = {'rf': model_data}
            scaler  = None
            f_names = None

        if not loaded_models:
            keys = list(model_data.keys()) if isinstance(model_data, dict) else str(type(model_data))
            return _err(f"No model ({', '.join(MODEL_KEYS)}) found in the .joblib. Keys: {keys}", step=step)

        # 5. Parse CSV — robust line-by-line (tolerates PuTTY logs, headers, junk)
        step = "parse_csv"
        try:
            t, p = parse_signal(csv_bytes)
        except Exception as e:
            return _err(f"Could not parse the CSV: {e}", step=step, exc=e)
        if len(t) < 10:
            return _err(f"No valid <time>,<pressure> data found (got {len(t)} rows; need >= 10). "
                        f"The file must contain two numeric columns.", step=step)

        # 6. Sampling rate
        step = "sampling_rate"
        diffs = np.diff(t)
        valid = diffs[diffs > 0]
        dt = float(np.median(valid)) if len(valid) > 0 else 1e-3
        fs = max(1, int(round(1.0 / dt)))

        # 7. Optional default cleanup filter (same pipeline as the desktop program)
        points_cleaned = 0
        if apply_filter:
            step = "filter"
            try:
                p_raw = p
                p = apply_default_filter(p)
                points_cleaned = int(np.sum(np.abs(p - p_raw) > 1e-9))
            except Exception as e:
                return _err(f"Filtering failed: {e}", step=step, exc=e)

        # 8. Features
        step = "extract_features"
        try:
            feats = extract_features(t, p, fs)
        except Exception as e:
            return _err(f"Feature extraction failed: {e}", step=step, exc=e)

        # 9. Build X
        step = "build_X"
        if f_names:
            missing = [k for k in f_names if k not in feats]
            if missing:
                return _err(f"Missing {len(missing)} features: {missing[:10]}", step=step)
            X = np.array([feats[k] for k in f_names], dtype=float).reshape(1, -1)
        else:
            X = np.array(list(feats.values()), dtype=float).reshape(1, -1)

        # 10. Scaler
        if scaler is not None:
            step = "scaler_transform"
            try:
                X = scaler.transform(X)
            except Exception as e:
                return _err(f"scaler.transform() failed: {e}", step=step, exc=e)

        # 11. Predict per model
        step = "predict"

        def _run_model(model, X):
            pred = int(model.predict(X)[0])
            conf = 100.0
            prob_bypass = float(pred)
            if hasattr(model, 'predict_proba'):
                try:
                    probs   = model.predict_proba(X)[0]
                    classes = list(model.classes_)
                    if 1 in classes:
                        prob_bypass = float(probs[classes.index(1)])
                    else:
                        prob_bypass = float(pred)
                    idx  = classes.index(pred) if pred in classes else pred
                    conf = round(float(probs[idx]) * 100, 2)
                except Exception as e:
                    logger.warning(f"predict_proba failed on {type(model).__name__}: {e}")
            return pred, conf, prob_bypass

        results = {}
        try:
            for key, model in loaded_models.items():
                pred, conf, pbp = _run_model(model, X)
                results[key] = {
                    "prediction":  pred,
                    "label":       "BYPASS" if pred == 1 else "NORMAL",
                    "confidence":  conf,
                    "prob_bypass": round(pbp * 100, 2),
                }
        except Exception as e:
            return _err(f"Prediction failed: {e}", step=step, exc=e)
        if not results:
            return _err("No model could predict.", step=step)

        # 12. Voting ensemble (soft voting): average of P(bypass)
        n_models      = len(results)
        ensemble_pbp  = sum(r["prob_bypass"] for r in results.values()) / n_models
        ensemble_pred = 1 if ensemble_pbp >= 50.0 else 0
        ensemble_label = "BYPASS" if ensemble_pred == 1 else "NORMAL"
        ensemble_conf = round(ensemble_pbp if ensemble_pred == 1 else (100.0 - ensemble_pbp), 2)
        ensemble_pbp  = round(ensemble_pbp, 2)
        all_preds = {r["prediction"] for r in results.values()}
        agreement = (len(all_preds) == 1) if n_models > 1 else None

        logger.info(f"==> {ensemble_label} {ensemble_conf}% | P(bypass)={ensemble_pbp}% | "
                    f"n={n_models} | fs={fs}Hz | default_model={used_default}")

        return {
            "status":         "success",
            "prediction":     ensemble_pred,
            "label":          ensemble_label,
            "confidence":     ensemble_conf,
            "prob_bypass":    ensemble_pbp,
            "agreement":      agreement,
            "n_models":       n_models,
            "models":         results,
            "fs_hz":          fs,
            "n_samples":      len(t),
            "used_default_model": used_default,
            "filter_applied": bool(apply_filter),
            "points_cleaned": points_cleaned,
        }

    except Exception as e:
        logger.exception(f"Unhandled exception in step '{step}': {e}")
        return _err(f"Unexpected error in step '{step}': {e}", step=step, exc=e)


@app.post("/api/inspect_model")
async def inspect_model_endpoint(model_file: UploadFile = File(...)):
    step = "start"
    try:
        model_bytes = await model_file.read()
        if len(model_bytes) == 0:
            return _err("The received .joblib has 0 bytes.", step=step)
        step = "load_model"
        try:
            model_data = joblib.load(io.BytesIO(model_bytes))
        except Exception as e:
            return _err(f"joblib could not deserialize the model: {e}", step=step, exc=e)
        step = "inspect"
        if isinstance(model_data, dict):
            md = model_data.get('models') if isinstance(model_data.get('models'), dict) else {}
            present = {k: ((md.get(k) or model_data.get(k)) is not None) for k in MODEL_KEYS}
            scaler  = model_data.get('scaler')
            f_names = model_data.get('feature_names')
        else:
            present = {'rf': True, 'svm': False, 'xgb': False, 'lgbm': False}
            scaler  = None
            f_names = None
        models_found = [k for k, v in present.items() if v]
        if not models_found:
            return _err(f"The .joblib contains none of: {', '.join(MODEL_KEYS)}.", step=step)
        return {
            "status": "success",
            "has_rf": present['rf'], "has_svm": present['svm'],
            "has_xgb": present['xgb'], "has_lgbm": present['lgbm'],
            "has_scaler": scaler is not None,
            "models": models_found, "n_models": len(models_found),
            "label": "+".join(m.upper() for m in models_found),
            "n_features": len(f_names) if f_names else None,
            "size_bytes": len(model_bytes),
        }
    except Exception as e:
        logger.exception(f"Exception in inspect_model step '{step}': {e}")
        return _err(f"Unexpected error in step '{step}': {e}", step=step, exc=e)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "7860"))
    uvicorn.run("app:app", host="0.0.0.0", port=port)

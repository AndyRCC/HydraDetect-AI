# HydraDetect-AI — backend

Standalone **FastAPI** service that runs the water-hammer bypass detector. It's the
Little Brother device logic (same `extract_features` + voting ensemble, **bit-exact**)
wrapped so the web frontend at [`/detector/`](../detector/) can call it from GitHub Pages.

- A model is **bundled** (`model/GOGETA.joblib`, calibrated RF + SVM + XGB + LGBM), so visitors only upload a CSV.
- `POST /api/analyze` accepts `csv_file` (required) and `model_file` (optional — overrides the bundled one).

## Endpoints

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| `GET`  | `/api/ping`          | — | versions + which models the server supports |
| `POST` | `/api/analyze`       | `csv_file`, optional `model_file` (multipart) | verdict JSON |
| `POST` | `/api/inspect_model` | `model_file` | which models a `.joblib` contains |
| `GET`  | `/` and `/docs`      | — | status / interactive Swagger UI |

## Run locally

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000
# open http://127.0.0.1:8000/docs
```

## Deploy — Render (recommended: deploys this folder from the same repo)

1. Push this repo to GitHub (already done).
2. On [render.com](https://render.com): **New + → Blueprint** → pick the `HydraDetect-AI` repo → **Apply**.
   (Render reads `backend/render.yaml`: free plan, root dir `backend`, start command already set.)
   *Or* do it manually: **New + → Web Service** → repo → **Root Directory** `backend`,
   Build `pip install -r requirements.txt`, Start `uvicorn app:app --host 0.0.0.0 --port $PORT`.
3. After it deploys you get a URL like `https://hydradetect-ai.onrender.com`.
4. Paste that URL into the detector page (Step 1). Done.

> Free tier sleeps after ~15 min idle, so the **first** request after a nap takes ~30–50 s
> (the page shows a "waking the server" hint). Subsequent requests are sub-second.

## Deploy — Hugging Face Spaces (alternative, sleeps far less often)

1. Create a **new Space** → SDK **Docker**.
2. Upload the contents of this `backend/` folder (`app.py`, `requirements.txt`, `Dockerfile`, `model/`).
3. Add this header to the Space's own `README.md`:
   ```yaml
   ---
   title: HydraDetect AI
   emoji: 💧
   colorFrom: blue
   colorTo: green
   sdk: docker
   app_port: 7860
   ---
   ```
4. The Space builds the `Dockerfile` and serves on port 7860. Your URL is
   `https://<user>-<space>.hf.space` → paste it into the detector page.

## CORS

By default the server allows any origin (`ALLOWED_ORIGINS=*`). To lock it to your site only,
set the env var (already wired in `render.yaml`):

```
ALLOWED_ORIGINS=https://andyrcc.github.io
```

## Notes

- The bundled model (Gogeta, third generation) is a calibrated **RF + SVM + XGB + LGBM**
  voting ensemble; the response adapts to whatever models a `.joblib` contains (2 or 4).
- The model was trained on an older scikit-learn; it loads and predicts correctly on current
  versions (a harmless version warning may appear in the logs).

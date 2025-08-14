### GPT-OSS:20B OpenXLab App (API-first with Gradio UI)

This app provides an OpenAI-compatible API and a Gradio chat UI for a large language model (default `gpt-oss:20b`). It is designed for OpenXLab's constraints and includes multiple model-loading fallbacks.

#### Features

- API-first FastAPI server (`POST /v1/chat/completions`) with API-key auth
- Gradio chat UI with optional image attachment (graceful fallback on text-only models)
- OpenXLab-friendly model bootstrap:
  - Clone model weights from OpenXLab Git LFS if `OPENXLAB_MODEL_REPO_URL` is provided
  - Use local path via `MODEL_LOCAL_PATH`
  - Fallback to Hugging Face via `HF_MODEL_ID` when allowed

#### Project structure

```
gpt-oss-openxlab-app/
  app.py                # FastAPI + Gradio entry
  models/
    loader.py           # Model loader with fallbacks
  requirements.txt
  README.md
```

#### Environment variables

- `APP_API_KEY`: If set, used as the API key. Otherwise generated at runtime and shown in the UI.
- `WORK_DIR`: Defaults to `/home/xlab-app-center` on OpenXLab.
- `MODEL_IDS`: Comma-separated priority list of model ids/paths. Put `gpt-oss/gpt-oss-20b` first to force priority.
- `MODELSCOPE_ID`: Prefer ModelScope IDs for China network, e.g. `qwen/Qwen2.5-14B-Instruct-AWQ`.
- `OPENXLAB_MODEL_REPO_URL`: Optional Git LFS mirror for weights; app will `git lfs pull`.
- `OPENXLAB_MODEL_SUBDIR`: Optional subfolder inside the repo where HF-style files live.
- `MODEL_LOCAL_PATH`: Absolute path to pre-downloaded HF/ModelScope-style model files.
- `HF_MODEL_ID`: HF repo id last resort (default `gpt-oss/gpt-oss-20b`).
- `QUANT_TYPE`: One of `auto|awq|gptq|bnb4|full`. Default `auto`.
- `HOST`/`PORT`: Server bind (defaults `0.0.0.0:7860`).

#### Run locally

```bash
python -m venv .venv && . .venv/Scripts/activate  # Windows PowerShell adapt accordingly
pip install -r requirements.txt

# Option A: local model path
set MODEL_LOCAL_PATH=D:\\models\\gpt-oss-20b

# Option B: OpenXLab Git LFS mirror
set OPENXLAB_MODEL_REPO_URL=https://code.openxlab.org.cn/yourorg/gpt-oss-20b.git

# Option C: Hugging Face (if available)
set MODEL_IDS=gpt-oss/gpt-oss-20b,Qwen/Qwen2.5-14B-Instruct-AWQ

python app.py
```

#### API usage

```bash
export API_KEY=... # shown in the Gradio sidebar if not set via APP_API_KEY
curl -s -X POST "http://localhost:7860/v1/chat/completions" \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss:20b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 128
  }' | jq .
```

#### OpenXLab notes

- OpenXLab typically uses `/home/xlab-app-center` as the writable workspace. This app stores its runtime data under `/home/xlab-app-center/gpt_oss_app_runtime/`.
- Prefer hosting your model weights on OpenXLab Models and exposing a Git LFS repo URL via `OPENXLAB_MODEL_REPO_URL`. The app will clone and `git lfs pull` on first run.
- Prefer ModelScope mirrors or OpenXLab Models for reliability. Avoid live HF downloads unless known to work.

#### Cursor / OpenAI-compatible clients

Point your client to your app base URL and set:

- Base URL: `https://<your-openxlab-app-domain>`
- API Key: shown in the UI or set via `APP_API_KEY`
- Endpoint: `/v1/chat/completions`



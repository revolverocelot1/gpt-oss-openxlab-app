import os
import uvicorn
import time
import secrets
from typing import List, Optional, Any, Dict

from fastapi import FastAPI, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import gradio as gr

from models.loader import load_chat_model, ChatSession, ModelInfo


# ---------------------------
# Runtime configuration
# ---------------------------
APP_HOST = os.getenv("HOST", "0.0.0.0")
APP_PORT = int(os.getenv("PORT", "7860"))
WORK_DIR = os.getenv("WORK_DIR", "/home/xlab-app-center")


def ensure_dir(path: str) -> None:

    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


RUNTIME_DIR = os.path.join(WORK_DIR, "gpt_oss_app_runtime")
ensure_dir(RUNTIME_DIR)


def init_api_key() -> str:

    api_key = os.getenv("APP_API_KEY")
    if api_key and len(api_key) >= 16:
        return api_key
    # Generate and persist
    api_key = secrets.token_urlsafe(32)
    key_file = os.path.join(RUNTIME_DIR, "api_key.txt")
    try:
        with open(key_file, "w", encoding="utf-8") as f:
            f.write(api_key)
    except Exception:
        pass
    return api_key


API_KEY = init_api_key()


# ---------------------------
# Model load
# ---------------------------
model_session: ChatSession
model_info: ModelInfo
try:
    model_session, model_info = load_chat_model()
except Exception as e:
    # Delay import error to runtime response for clearer UI
    model_session = None  # type: ignore
    model_info = ModelInfo(model_id="uninitialized", device="cpu", multimodal=False)
    MODEL_INIT_ERROR = str(e)
else:
    MODEL_INIT_ERROR = ""


# ---------------------------
# OpenAI-compatible API schema
# ---------------------------
class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(default=None)
    messages: List[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(default=512, ge=1, le=4096)
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: Dict[str, Any]
    finish_reason: str


class ChatCompletionUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: ChatCompletionUsage


def require_api_key(authorization: Optional[str] = Header(default=None), x_api_key: Optional[str] = Header(default=None)) -> None:

    provided = None
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization.split(" ", 1)[1].strip()
    if not provided and x_api_key:
        provided = x_api_key.strip()
    if not provided or provided != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------
# FastAPI app
# ---------------------------
app = FastAPI(title="GPT-OSS:20B OpenXLab App", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/v1/health")
def health() -> Dict[str, str]:

    return {"status": "ok", "model": model_info.model_id, "device": model_info.device, "multimodal": str(model_info.multimodal)}


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
def chat_completions(req: ChatCompletionRequest, _: None = Depends(require_api_key)) -> ChatCompletionResponse:

    if MODEL_INIT_ERROR:
        raise HTTPException(status_code=500, detail=f"Model failed to initialize: {MODEL_INIT_ERROR}")

    # Flatten messages to text; collect media if present
    text_messages: List[Dict[str, Any]] = []
    images: List[Any] = []
    for m in req.messages:
        role = m.role
        content = m.content
        if isinstance(content, list):
            text_parts: List[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                    images.append(part)
            text_content = "\n".join([p for p in text_parts if p])
        else:
            text_content = str(content)
        text_messages.append({"role": role, "content": text_content})

    output_text, usage = model_session.generate(
        messages=text_messages,
        images=images,
        temperature=req.temperature,
        max_new_tokens=req.max_tokens or 512,
    )

    created_ts = int(time.time())
    return ChatCompletionResponse(
        id=f"chatcmpl_{secrets.token_hex(8)}",
        object="chat.completion",
        created=created_ts,
        model=model_info.model_id,
        choices=[ChatCompletionChoice(index=0, message={"role": "assistant", "content": output_text}, finish_reason="stop")],
        usage=ChatCompletionUsage(**usage),
    )


# ---------------------------
# Gradio UI
# ---------------------------
def ui_respond(user_message: str, history: List[List[str]], image: Optional[Any]) -> List[List[str]]:

    if MODEL_INIT_ERROR:
        return history + [[user_message, f"Model init error: {MODEL_INIT_ERROR}"]]
    images = []
    if image is not None:
        images.append({"type": "image", "image": image})
    messages = []
    for h_user, h_assistant in history:
        messages.append({"role": "user", "content": h_user})
        messages.append({"role": "assistant", "content": h_assistant})
    messages.append({"role": "user", "content": user_message})
    output_text, _usage = model_session.generate(messages=messages, images=images, temperature=0.7, max_new_tokens=512)
    return history + [[user_message, output_text]]


with gr.Blocks(title="GPT-OSS:20B Chat & API", theme=gr.themes.Default()) as gradio_ui:
    gr.Markdown("""
    ### GPT-OSS:20B Chat & API
    - API is prioritized and OpenAI-compatible at `POST /v1/chat/completions`.
    - Use header `Authorization: Bearer <API_KEY>`.
    """)
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=420)
            with gr.Row():
                txt = gr.Textbox(placeholder="Type a message and optionally attach an image", scale=4)
                send = gr.Button("Send", variant="primary", scale=1)
            image = gr.Image(label="Optional image", type="pil")
            send.click(ui_respond, inputs=[txt, chatbot, image], outputs=chatbot)
        with gr.Column(scale=2):
            gr.Markdown("""
            **Your API key (auto-generated if not provided via APP_API_KEY):**
            """)
            api_key_box = gr.Textbox(value=API_KEY, label="API Key", interactive=False)
            gr.Markdown("""
            **Endpoint path:** `/v1/chat/completions`
            
            Example curl:
            ```bash
            curl -X POST "$HOST/v1/chat/completions" \
                 -H "Authorization: Bearer $API_KEY" \
                 -H "Content-Type: application/json" \
                 -d '{
                   "model": "gpt-oss:20b",
                   "messages": [
                     {"role": "user", "content": "Hello!"}
                   ],
                   "max_tokens": 128
                 }'
            ```
            """)


# Mount Gradio into FastAPI
app = gr.mount_gradio_app(app, gradio_ui, path="/")


def main() -> None:

    uvicorn.run(app, host=APP_HOST, port=APP_PORT)


if __name__ == "__main__":
    main()



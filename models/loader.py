import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedTokenizerBase,
)

# Optional quantization/model download backends
try:
    from autoawq.modeling import AutoAWQForCausalLM  # type: ignore
    AWQ_AVAILABLE = True
except Exception:
    AWQ_AVAILABLE = False

try:
    from auto_gptq import AutoGPTQForCausalLM  # type: ignore
    GPTQ_AVAILABLE = True
except Exception:
    GPTQ_AVAILABLE = False

try:
    from transformers import BitsAndBytesConfig  # type: ignore
    BNB_AVAILABLE = True
except Exception:
    BNB_AVAILABLE = False

try:
    from modelscope.hub.snapshot_download import snapshot_download  # type: ignore
    MODELSCOPE_AVAILABLE = True
except Exception:
    MODELSCOPE_AVAILABLE = False


@dataclass
class ModelInfo:
    model_id: str
    device: str
    multimodal: bool


class ChatSession:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model: AutoModelForCausalLM,
        device: torch.device,
        model_id: str,
        multimodal: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.device = device
        self.model_id = model_id
        self.multimodal = multimodal

    def _format_messages(self, messages: List[Dict[str, str]]) -> str:

        parts: List[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")
            else:
                parts.append(f"<|user|>\n{content}")
        parts.append("<|assistant|>\n")
        return "\n\n".join(parts)

    @torch.inference_mode()
    def generate(
        self,
        messages: List[Dict[str, str]],
        images: Optional[List[Any]] = None,
        temperature: float = 0.7,
        max_new_tokens: int = 512,
    ) -> Tuple[str, Dict[str, int]]:

        prompt = self._format_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        generated = self.model.generate(
            **inputs,
            do_sample=temperature > 0,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        output_ids = generated[0][inputs["input_ids"].shape[1]:]
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)

        # Approximate usage
        prompt_tokens = int(inputs["input_ids"].numel())
        completion_tokens = int(output_ids.shape[0])
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return output_text, usage


def _select_device() -> torch.device:

    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_model_path() -> Tuple[str, bool]:

    # Priority: explicit local path
    local_path = os.getenv("MODEL_LOCAL_PATH")
    if local_path and os.path.isdir(local_path):
        return local_path, False

    # ModelScope mirror (China-friendly)
    ms_id = os.getenv("MODELSCOPE_ID")
    if ms_id and MODELSCOPE_AVAILABLE:
        work_dir = os.getenv("WORK_DIR", "/home/xlab-app-center")
        cache_dir = os.path.join(work_dir, "modelscope_cache")
        os.makedirs(cache_dir, exist_ok=True)
        try:
            local_dir = snapshot_download(ms_id, cache_dir=cache_dir)
            if os.path.isdir(local_dir):
                return local_dir, False
        except Exception:
            pass

    # OpenXLab Git LFS model clone (generic support for other models)
    work_dir = os.getenv("WORK_DIR", "/home/xlab-app-center")
    model_dir = os.path.join(work_dir, "gpt_oss_model")
    ox_repo = os.getenv("OPENXLAB_MODEL_REPO_URL")
    if ox_repo:
        if not os.path.exists(model_dir) or not os.listdir(model_dir):
            os.system(f"git clone {ox_repo} {model_dir}")
            os.system(f"cd {model_dir} && git lfs pull")
        subdir = os.getenv("OPENXLAB_MODEL_SUBDIR", "")
        resolved = os.path.join(model_dir, subdir) if subdir else model_dir
        if os.path.isdir(resolved):
            return resolved, False

    # Hugging Face fallback list (priority order)
    model_ids_env = os.getenv("MODEL_IDS")
    if model_ids_env:
        # comma-separated list of HF repo ids; return the first as indicator, the caller may iterate
        first = model_ids_env.split(",")[0].strip()
        if first:
            return first, True
    hf_id = os.getenv("HF_MODEL_ID", "gpt-oss/gpt-oss-20b")
    return hf_id, True


def load_chat_model() -> Tuple[ChatSession, ModelInfo]:

    device = _select_device()
    model_path, is_hf_id = _resolve_model_path()

    # dtype selection
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    def try_load(model_id_or_path: str) -> Tuple[ChatSession, ModelInfo]:
        tok = AutoTokenizer.from_pretrained(model_id_or_path, trust_remote_code=True, use_fast=True)
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id

        # Default to auto-detect available quant backend
        quant_type = os.getenv("QUANT_TYPE", "auto").lower()
        mdl: Any = None

        if mdl is None and quant_type in ("awq", "auto", "bnb4-first") and AWQ_AVAILABLE:
            try:
                mdl = AutoAWQForCausalLM.from_quantized(
                    model_id_or_path,
                    trust_remote_code=True,
                    device_map="auto" if device.type == "cuda" else None,
                )
            except Exception:
                mdl = None

        if mdl is None and quant_type in ("gptq", "auto", "bnb4-first") and GPTQ_AVAILABLE:
            try:
                mdl = AutoGPTQForCausalLM.from_quantized(
                    model_id_or_path,
                    device="cuda" if device.type == "cuda" else "cpu",
                    trust_remote_code=True,
                    inject_fused_attention=False,
                )
            except Exception:
                mdl = None

        if mdl is None and quant_type in ("bnb4", "auto", "bnb4-first") and BNB_AVAILABLE:
            try:
                from transformers import BitsAndBytesConfig  # type: ignore
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
                )
                mdl = AutoModelForCausalLM.from_pretrained(
                    model_id_or_path,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    device_map="auto" if device.type == "cuda" else None,
                    quantization_config=bnb_config,
                )
            except Exception:
                mdl = None

        if mdl is None:
            mdl = AutoModelForCausalLM.from_pretrained(
                model_id_or_path,
                torch_dtype=dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map="auto" if device.type == "cuda" else None,
            )

        mdl.eval()
        return ChatSession(tokenizer=tok, model=mdl, device=device, model_id=str(model_id_or_path), multimodal=False), ModelInfo(model_id=str(model_id_or_path), device=device.type, multimodal=False)

    # If env MODEL_IDS provided, iterate
    model_ids_env = os.getenv("MODEL_IDS")
    if model_ids_env:
        for mid in [m.strip() for m in model_ids_env.split(",") if m.strip()]:
            try:
                return try_load(mid)
            except Exception:
                continue

    # Otherwise try the resolved path/id
    return try_load(model_path)



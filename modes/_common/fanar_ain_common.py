"""
fanar_ain_common.py
=======================================================================
Shared infrastructure for every Sarab mode's Fanar/AIN evaluation
driver script (mode1_base, mode2_3_sec_icc, mode4_ccs, mode5_nota).

Fanar and AIN now run over ALL modes, not just mode4 — thousands of
model calls total across the four mode groups. A monolithic per-mode
script that only pushes to the Hub once a whole model's run finishes
(the original run_mode4_ccs_fanar_ain.py draft) is too risky at that
scale: too much work can be lost to one disconnect. This module
factors out the parts every driver script needs so each one can stay
short, and adds BATCHED pushing so progress is never more than one
batch away from being safely on the Hub.

Each driver script (one per mode group) imports from here, then adds
only what's specific to its mode: the question schema, the system
instruction/turn structure, and the grading logic.

RESUMABILITY CONTRACT (unchanged from fatan_eval_pipeline.py and the
original mode4 draft):
  - Every row is appended to a local JSON file immediately after it is
    generated — a crash loses at most one row.
  - Rows containing "[GENERATION ERROR: ...]" are excluded from
    "already done" checks, so a transient failure gets retried on the
    next run instead of silently counting as permanent.
  - A local results file always wins over the Hub on startup; the Hub
    copy is only used to resume if there is no local file at all
    (e.g. a fresh Colab runtime after a disconnect — see each driver's
    "RECOVERING FROM A DISCONNECT" note).

BATCHING (new):
  run_in_batches() walks an item list BATCH_SIZE at a time (default 50)
  and pushes the results file to the Hub after every batch — not just
  after a model finishes all its items. One failed item inside a batch
  is caught and logged; it does not stop the rest of the batch or the
  run.

Usage in Colab: clone the repo first (!git clone ...) so this file is
importable as modes/_common/fanar_ain_common.py; each driver script
adds the repo root to sys.path before importing it.
"""

import gc
import json
import os
import re
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import torch
from PIL import Image

DEFAULT_BATCH_SIZE = 50

# Some Hub downloads (observed with the newer "Xet" chunked backend) can
# stall indefinitely at a few KB/s on certain Colab sessions instead of
# failing outright — hf_hub_download never raises, so nothing catches it.
# Running the download in a worker thread with a hard wall-clock timeout
# turns a silent infinite hang into a skip-and-continue, same as any other
# per-item failure. The stray thread may keep downloading in the
# background after we give up on it; harmless, just abandoned.
IMAGE_DOWNLOAD_TIMEOUT_SECONDS = 90

SYSTEM_INSTRUCTION_YES_NO_AR = (
    "انظر إلى الصورة وأجب عن السؤال بالاعتماد على ما هو ظاهر فيها فقط. "
    "يجب أن تكون إجابتك كلمة واحدة فقط: نعم أو لا. "
    "لا تضف أي شرح أو علامات ترقيم أو كلمات أخرى."
)

_YES_WORDS = {"نعم", "أجل", "ايوه", "yes"}
_NO_WORDS = {"لا", "كلا", "no"}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GEMINI_JUDGE_MODEL = "google/gemini-2.5-flash"


# =========================================================================
# Credentials
# =========================================================================

def get_secret(name: str) -> Optional[str]:
    try:
        from google.colab import userdata  # type: ignore
        val = userdata.get(name)
        if val:
            return val
    except Exception:
        pass
    return os.environ.get(name)


def login_hf(hf_token: Optional[str]) -> None:
    if hf_token:
        from huggingface_hub import login as _hf_login
        _hf_login(token=hf_token)
    else:
        print("WARNING: HF_TOKEN not set — gated models (if any) and the "
              "final push to the Hub will fail. Set it in Colab secrets.")


def build_openrouter_client(openrouter_api_key: Optional[str]):
    from openai import OpenAI
    if not openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set.")
    return OpenAI(api_key=openrouter_api_key, base_url=OPENROUTER_BASE_URL)


# =========================================================================
# Model registry — Fanar (QCRI/Fanar-2-Oryx-IVU) and AIN (MBZUAI/AIN),
# both Qwen-VL-family, 4-bit quantized, "legacy_qwen_vl" chat style.
# Verified loadable on a free-tier Colab GPU (same config as the Fatan
# pipelines).
# =========================================================================

@dataclass
class ModelHandle:
    name: str
    model: object
    processor: object
    chat_style: str  # "legacy_qwen_vl" (both Fanar and AIN use this)
    device: str = "cuda"


@dataclass
class ModelSpec:
    name: str
    loader: Callable[[], ModelHandle]


def load_fanar() -> ModelHandle:
    from transformers import (
        Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig,
    )
    model_id = "QCRI/Fanar-2-Oryx-IVU"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quantization_config, device_map="auto",
        torch_dtype=torch.float16, low_cpu_mem_usage=True,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return ModelHandle("Fanar-2-Oryx-IVU", model, processor, "legacy_qwen_vl")


def load_ain() -> ModelHandle:
    from transformers import (
        Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig,
    )
    model_id = "MBZUAI/AIN"
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quantization_config,
        device_map="auto", torch_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return ModelHandle("AIN", model, processor, "legacy_qwen_vl")


MODEL_SPECS = [
    ModelSpec("Fanar-2-Oryx-IVU", load_fanar),
    ModelSpec("AIN", load_ain),
]


def unload_model(handle: ModelHandle) -> None:
    del handle.model
    del handle.processor
    gc.collect()
    torch.cuda.empty_cache()


def generate_turn(handle: ModelHandle, messages: list[dict], max_new_tokens: int) -> str:
    from qwen_vl_utils import process_vision_info
    model, processor = handle.model, handle.processor

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    response = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False,
    )[0]
    return response.strip()


def safe_generate_turn(handle: ModelHandle, messages: list[dict], max_new_tokens: int) -> str:
    """generate_turn, but catching the failure into the same
    "[GENERATION ERROR: ...]" row format every driver already expects,
    and clearing the CUDA cache on the way out so one OOM doesn't leave
    the allocator fragmented for the next (possibly fine) call — an OOM
    followed immediately by another OOM on a normal-sized image is
    usually this, not a second genuinely oversized image."""
    try:
        return generate_turn(handle, messages, max_new_tokens)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        gc.collect()
        torch.cuda.empty_cache()
        return f"[GENERATION ERROR: {exc}]"


def parse_yes_no(raw: str) -> str:
    """Same parser as run_mode1_openrouter.py's parse_answer, kept
    identical so scoring is comparable across pipelines."""
    if not raw:
        return "unclear"
    text = raw.strip().strip(" \n\t\"'.,!?؟،؛:-")
    if text in _YES_WORDS or text.lower() in _YES_WORDS:
        return "yes"
    if text in _NO_WORDS or text.lower() in _NO_WORDS:
        return "no"
    tokens = re.findall(r"[^\W\d_]+", text, flags=re.UNICODE)
    has_yes = any(t in _YES_WORDS or t.lower() in _YES_WORDS for t in tokens)
    has_no = any(t in _NO_WORDS or t.lower() in _NO_WORDS for t in tokens)
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    return "unclear"


def parse_letter(raw: str) -> Optional[str]:
    """First bare A/B/C/D (any case, optionally Arabic-lettered like
    'الإجابة: C') found in a forced-choice response."""
    if not raw:
        return None
    match = re.search(r"\b([ABCD])\b", raw.strip(), flags=re.IGNORECASE)
    return match.group(1).upper() if match else None


# =========================================================================
# Dataset download — questions JSON (eager) + per-item images (lazy,
# cached by huggingface_hub so re-requesting an already-downloaded
# image is a no-op). Works for any mode because every item's
# local_path already points at its own file's repo-relative path.
# =========================================================================

def download_questions_json(dataset_repo: str, filename: str, workdir: Path) -> object:
    from huggingface_hub import hf_hub_download
    workdir.mkdir(parents=True, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=dataset_repo, repo_type="dataset", filename=filename, local_dir=workdir,
    )
    with open(local_path, encoding="utf-8") as f:
        return json.load(f)


def load_item_image(item: dict, dataset_repo: str, workdir: Path) -> Optional[Image.Image]:
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
    from huggingface_hub import hf_hub_download

    def _download():
        return hf_hub_download(
            repo_id=dataset_repo, repo_type="dataset",
            filename=item["local_path"], local_dir=workdir,
        )

    # NOT a `with ThreadPoolExecutor(...)` block on purpose: its __exit__
    # calls shutdown(wait=True), which blocks until the abandoned worker
    # thread finishes — defeating the timeout below entirely (a stalled
    # download would still hang the whole run, just silently past the
    # point where the warning below already printed). shutdown(wait=False)
    # lets us walk away from the stuck thread immediately.
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        future = pool.submit(_download)
        try:
            local_path = future.result(timeout=IMAGE_DOWNLOAD_TIMEOUT_SECONDS)
        except FutureTimeoutError:
            print(
                f"  [WARN] image download for {item.get('image_id')} "
                f"exceeded {IMAGE_DOWNLOAD_TIMEOUT_SECONDS}s — skipping "
                "(likely a stalled Hub download, not a data problem)"
            )
            return None
        finally:
            pool.shutdown(wait=False)
        return Image.open(local_path).convert("RGB")
    except Exception:
        traceback.print_exc()
        return None


# =========================================================================
# Result-row I/O (append-after-every-turn persistence)
# =========================================================================

def download_existing_results_if_any(dataset_repo: str, results_repo_path: str,
                                       results_path: Path, workdir: Path) -> None:
    """Local file always wins; otherwise try to resume from the Hub's
    last push. Missing on the Hub is not an error — first run."""
    results_filename = results_path.name
    if results_path.exists():
        print(f"Using existing local {results_filename} — not fetching from the Hub.")
        return
    from huggingface_hub import hf_hub_download
    try:
        hf_hub_download(
            repo_id=dataset_repo, repo_type="dataset",
            filename=results_repo_path, local_dir=workdir,
        )
        nested = workdir / results_repo_path
        if nested.exists() and not results_path.exists():
            nested.replace(results_path)
        print(f"Pulled prior {results_filename} from the Hub — resuming from it.")
    except Exception as exc:
        print(f"No prior {results_filename} on the Hub ({exc}) — starting fresh.")


def load_results(results_path: Path) -> list[dict]:
    if results_path.exists():
        with open(results_path, encoding="utf-8") as f:
            return json.load(f)
    return []


def is_error_row(row: dict) -> bool:
    return "GENERATION ERROR" in (row.get("raw_response") or "")


def append_result_row(row: dict, results_path: Path, key_fields: tuple[str, ...]) -> None:
    """Append one row, rewriting the results file to disk immediately.
    A previous error row sharing the same key_fields is replaced, not
    left alongside the new one (successful or not)."""
    results = load_results(results_path)
    key = tuple(row[k] for k in key_fields)
    results = [
        r for r in results
        if not (is_error_row(r) and tuple(r[k] for k in key_fields) == key)
    ]
    results.append(row)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def push_results_to_hub(hf_token: Optional[str], results_path: Path,
                         dataset_repo: str, results_repo_path: str,
                         commit_message: str) -> None:
    if not hf_token:
        print("Skipping Hub push — no HF_TOKEN.")
        return
    from huggingface_hub import HfApi
    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=str(results_path),
        path_in_repo=results_repo_path,
        repo_id=dataset_repo,
        repo_type="dataset",
        commit_message=commit_message,
    )


# =========================================================================
# Batch runner — the core of the new error-handling requirement: push
# to the Hub every BATCH_SIZE items, not just once per model.
# =========================================================================

def run_in_batches(items: list, run_one_item_fn: Callable[[object], None],
                    push_fn: Callable[[str], None], label: str,
                    batch_size: int = DEFAULT_BATCH_SIZE) -> None:
    """run_one_item_fn(item) must do its own internal skip-if-done
    checks and its own try/except around each model call (that's
    where real generation errors are logged as rows, per the
    resumability contract). The try/except here is a last-resort
    guard against something unrelated to generation blowing up (e.g.
    a malformed item) — it logs and moves on instead of killing the
    whole batch."""
    total = len(items)
    for start in range(0, total, batch_size):
        batch = items[start:start + batch_size]
        for item in batch:
            try:
                run_one_item_fn(item)
            except Exception as exc:  # noqa: BLE001
                print(f"  [BATCH ERROR] {item.get('image_id', '?')}: {exc}")
                traceback.print_exc()
        done = min(start + batch_size, total)
        push_fn(f"{label}: batch {start // batch_size + 1} ({done}/{total} items)")
        print(f"  -- pushed after batch {start // batch_size + 1} ({done}/{total}) --")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

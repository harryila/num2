"""Filter NQ Open items by base model failure.

Streams items from HuggingFace's nq_open dataset (deterministically shuffled by
seed), runs greedy zero-shot generation on each with the configured base model
(no LoRA, no adapters), and applies NQ exact-match. Items the model gets wrong
are appended to the unknown JSONL (the "hard" subset used for training).
Items it gets right are appended to the known JSONL (kept for analysis).

Stops when the unknown JSONL reaches --target-unknown items.

Resumable: state file tracks counters and the last processed shuffled-index.
Re-running with the same model + seed picks up where it left off; mismatched
model or seed aborts to prevent corrupting the on-disk dataset.

Output schema matches what load_closed_book_jsonl expects:
    {"item_id": str, "prompt": str, "target": "ans1|||ans2|||...", "domain_tag": "nq_open"}

Usage:
    python -m testing_effect_pipeline.filter_nq_unknown \
        --model-name Qwen/Qwen2.5-0.5B-Instruct \
        --hf-token YOUR_TOKEN \
        --target-unknown 10000 \
        --batch-size 16 \
        --output-unknown data/nq_open_hard_10k.jsonl \
        --output-known data/nq_open_known.jsonl \
        --state-path data/nq_open_filter_state.json \
        --seed 42 \
        --dtype bfloat16
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from .nq_eval import exact_match_score

logger = logging.getLogger("filter_nq_unknown")

DEFAULT_SYSTEM_PROMPT = "Answer the question with a short factual answer."


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--hf-token", type=str, default=None)
    p.add_argument("--target-unknown", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--output-unknown", type=str, default="data/nq_open_hard_10k.jsonl")
    p.add_argument("--output-known", type=str, default="data/nq_open_known.jsonl")
    p.add_argument("--state-path", type=str, default="data/nq_open_filter_state.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max-seq-len", type=int, default=256, help="Tokenizer truncation length for prompts.")
    p.add_argument("--max-new-tokens", type=int, default=32, help="Greedy decode length cap.")
    p.add_argument(
        "--system-prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System message; default matches RealModelAdapter so the filter uses the same prompt format as training.",
    )
    p.add_argument(
        "--split",
        type=str,
        default="train",
        help="HF split to filter from (default: train).",
    )
    return p.parse_args()


def init_state(args: argparse.Namespace) -> dict:
    return {
        "model_name": args.model_name,
        "seed": args.seed,
        "split": args.split,
        "last_processed_index": -1,
        "total_evaluated": 0,
        "total_known": 0,
        "total_unknown": 0,
        "target_unknown": args.target_unknown,
        "complete": False,
    }


def load_or_init_state(args: argparse.Namespace) -> dict:
    state_path = Path(args.state_path)
    if not state_path.exists():
        state = init_state(args)
        logger.info("No existing state file at %s; starting fresh.", state_path)
        return state

    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)

    if state.get("model_name") != args.model_name:
        sys.exit(
            f"State file model_name={state.get('model_name')!r} does not match "
            f"--model-name={args.model_name!r}. Refusing to mix filter results from different base models. "
            f"Delete {args.state_path} (and the JSONLs) to start fresh, or pass the matching --model-name."
        )
    if state.get("seed") != args.seed:
        sys.exit(
            f"State file seed={state.get('seed')} does not match --seed={args.seed}. "
            f"The shuffled iteration order depends on the seed; resuming with a different seed would "
            f"corrupt the dataset. Delete {args.state_path} (and the JSONLs) to start fresh."
        )
    if state.get("split") != args.split:
        sys.exit(
            f"State file split={state.get('split')!r} does not match --split={args.split!r}. "
            f"Delete {args.state_path} (and the JSONLs) to start fresh."
        )

    state["target_unknown"] = args.target_unknown
    logger.info(
        "Resuming from state: last_processed_index=%d, evaluated=%d, known=%d, unknown=%d/%d",
        state["last_processed_index"],
        state["total_evaluated"],
        state["total_known"],
        state["total_unknown"],
        state["target_unknown"],
    )
    return state


def write_state_atomic(state: dict, state_path: Path) -> None:
    """Write state to disk atomically (tempfile + os.replace + fsync)."""
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, state_path)


def append_jsonl(items: list[dict], file_handle) -> None:
    """Append items as JSONL lines, then flush + fsync."""
    for it in items:
        file_handle.write(json.dumps(it, ensure_ascii=False) + "\n")
    file_handle.flush()
    os.fsync(file_handle.fileno())


def build_messages(question: str, system_prompt: str) -> list[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def generate_batch(
    questions: list[str],
    model,
    tokenizer,
    *,
    system_prompt: str,
    max_seq_len: int,
    max_new_tokens: int,
) -> list[str]:
    """Left-padded batched greedy decode. Mirrors RealModelAdapter._generate_batch."""
    import torch  # local import so --help works without torch installed

    if not questions:
        return []

    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        prompt_texts = [
            tokenizer.apply_chat_template(
                build_messages(q, system_prompt),
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in questions
        ]

        enc = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        input_ids = enc.input_ids.to(model.device)
        attention_mask = enc.attention_mask.to(model.device)

        with torch.no_grad():
            gen_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        prompt_len = input_ids.shape[1]
        preds = []
        for i in range(len(questions)):
            tokens = gen_ids[i, prompt_len:]
            preds.append(tokenizer.decode(tokens, skip_special_tokens=True).strip())
        return preds
    finally:
        tokenizer.padding_side = orig_side


def load_model_and_tokenizer(args: argparse.Namespace):
    """Load the bare base model + tokenizer (no LoRA, no PEFT, eval mode)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading tokenizer: %s", args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        token=args.hf_token,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    logger.info("Loading model: %s  dtype=%s  device=%s", args.model_name, args.dtype, device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        token=args.hf_token,
        trust_remote_code=True,
    )
    model = model.to(device)
    model.eval()
    return model, tokenizer


def row_to_item(row: dict, orig_idx: int) -> dict:
    """Convert a HF nq_open row to the load_closed_book_jsonl schema."""
    answers = row["answer"] if isinstance(row["answer"], list) else [row["answer"]]
    return {
        "item_id": f"nq-train-{orig_idx}",
        "prompt": row["question"],
        "target": "|||".join(answers),
        "domain_tag": "nq_open",
    }


def stream_and_filter(args: argparse.Namespace, state: dict) -> dict:
    from datasets import load_dataset

    logger.info("Loading google-research-datasets/nq_open split=%s ...", args.split)
    dataset = load_dataset("google-research-datasets/nq_open", token=args.hf_token)[args.split]

    dataset = dataset.add_column("__orig_idx", list(range(len(dataset))))
    logger.info("Shuffling dataset with seed=%d (deterministic) ...", args.seed)
    dataset = dataset.shuffle(seed=args.seed)

    total = len(dataset)
    logger.info("Dataset size after shuffle: %d items", total)

    model, tokenizer = load_model_and_tokenizer(args)

    state_path = Path(args.state_path)
    unk_path = Path(args.output_unknown)
    known_path = Path(args.output_known)
    unk_path.parent.mkdir(parents=True, exist_ok=True)
    known_path.parent.mkdir(parents=True, exist_ok=True)

    start_idx = state["last_processed_index"] + 1
    if start_idx >= total:
        logger.info("All %d items already processed.", total)
        state["complete"] = True
        write_state_atomic(state, state_path)
        return state

    if state["total_unknown"] >= state["target_unknown"]:
        logger.info(
            "Already have %d unknowns >= target %d; marking complete.",
            state["total_unknown"],
            state["target_unknown"],
        )
        state["complete"] = True
        write_state_atomic(state, state_path)
        return state

    t_start = time.time()
    unk_f = unk_path.open("a", encoding="utf-8")
    known_f = known_path.open("a", encoding="utf-8")
    try:
        batch_items: list[dict] = []
        batch_indices: list[int] = []

        def flush_batch() -> bool:
            """Process accumulated batch. Returns True if target reached."""
            if not batch_items:
                return state["total_unknown"] >= state["target_unknown"]

            questions = [it["prompt"] for it in batch_items]
            preds = generate_batch(
                questions,
                model,
                tokenizer,
                system_prompt=args.system_prompt,
                max_seq_len=args.max_seq_len,
                max_new_tokens=args.max_new_tokens,
            )

            new_unknown: list[dict] = []
            new_known: list[dict] = []
            for it, pred in zip(batch_items, preds):
                targets = [t.strip() for t in it["target"].split("|||")]
                if exact_match_score(pred, targets):
                    new_known.append(it)
                else:
                    new_unknown.append(it)

            if new_known:
                append_jsonl(new_known, known_f)
            if new_unknown:
                append_jsonl(new_unknown, unk_f)

            state["total_known"] += len(new_known)
            state["total_unknown"] += len(new_unknown)
            state["total_evaluated"] += len(batch_items)
            state["last_processed_index"] = batch_indices[-1]

            target_hit = state["total_unknown"] >= state["target_unknown"]
            if target_hit:
                state["complete"] = True

            write_state_atomic(state, state_path)

            elapsed = time.time() - t_start
            rate = (state["total_unknown"] / state["total_evaluated"] * 100.0) if state["total_evaluated"] else 0.0
            ips = state["total_evaluated"] / elapsed if elapsed > 0 else 0.0
            print(
                f"Processed {state['total_evaluated']} | "
                f"Known {state['total_known']} | "
                f"Unknown {state['total_unknown']}/{state['target_unknown']} | "
                f"Failure rate: {rate:.1f}% | "
                f"{ips:.1f} items/s",
                flush=True,
            )

            batch_items.clear()
            batch_indices.clear()
            return target_hit

        for i in range(start_idx, total):
            row = dataset[i]
            orig_idx = row["__orig_idx"]
            batch_items.append(row_to_item(row, orig_idx))
            batch_indices.append(i)

            if len(batch_items) >= args.batch_size:
                if flush_batch():
                    break

        if state["total_unknown"] < state["target_unknown"]:
            flush_batch()

        if state["total_unknown"] >= state["target_unknown"]:
            state["complete"] = True
            write_state_atomic(state, state_path)
            logger.info("Target reached: %d unknowns collected.", state["total_unknown"])
        elif state["last_processed_index"] >= total - 1:
            state["complete"] = True
            write_state_atomic(state, state_path)
            logger.warning(
                "Exhausted dataset at %d unknowns (target was %d). Marking complete.",
                state["total_unknown"],
                state["target_unknown"],
            )
        else:
            logger.info(
                "Stopping mid-stream at index %d. Re-run with the same args to resume.",
                state["last_processed_index"],
            )
    finally:
        unk_f.close()
        known_f.close()

    return state


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    if args.target_unknown <= 0:
        sys.exit("--target-unknown must be positive.")
    if args.batch_size <= 0:
        sys.exit("--batch-size must be positive.")

    state = load_or_init_state(args)
    if state.get("complete"):
        logger.info(
            "State file marks run complete (unknowns=%d, target=%d). Nothing to do. "
            "Delete %s to start over.",
            state["total_unknown"],
            state["target_unknown"],
            args.state_path,
        )
        return

    state = stream_and_filter(args, state)

    print(
        f"Done. evaluated={state['total_evaluated']} "
        f"known={state['total_known']} "
        f"unknown={state['total_unknown']}/{state['target_unknown']} "
        f"complete={state['complete']}"
    )


if __name__ == "__main__":
    main()

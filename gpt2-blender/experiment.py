#!/usr/bin/env python3
"""Reproducible GPT-2 weight-corruption experiment. Runs on CPU in GitHub Actions."""
import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "openai-community/gpt2"
PROMPTS = [
    "The frog council held an emergency meeting because",
    "Scientists finally discovered what happens when you put a toaster in",
    "The world's least useful superpower is",
]
EVALUATION_TEXT = (
    "The curious scientist looked at the experiment and wrote down the results. "
    "The next morning, the team discussed what the measurements might mean."
)
LEVELS = {"mild": 0.35, "medium": 1.0, "extreme": 1.8}


@torch.no_grad()
def fuzz_attention(model, generator, intensity):
    """Add noise to the learned query/key/value projection matrices."""
    for index in (0, 3, 6, 9):
        weight = model.transformer.h[index].attn.c_attn.weight
        standard_deviation = weight.std().item()
        noise = torch.randn(weight.shape, generator=generator, dtype=weight.dtype)
        weight.add_(noise * standard_deviation * 0.65 * intensity)


@torch.no_grad()
def scramble_neurons(model, generator, intensity):
    """Disconnect learned MLP neurons by permuting some projection rows."""
    fraction = min(0.95, 0.25 * intensity)
    for index in (1, 4, 7, 10):
        weight = model.transformer.h[index].mlp.c_proj.weight
        count = max(2, int(weight.shape[0] * fraction))
        rows = torch.randperm(weight.shape[0], generator=generator)[:count]
        scrambled_rows = rows[torch.randperm(count, generator=generator)]
        weight[rows] = weight[scrambled_rows].clone()


@torch.no_grad()
def shuffle_vocabulary(model, generator, intensity):
    """Mix up some token embeddings (also changes tied output-vocabulary weights)."""
    weight = model.transformer.wte.weight
    fraction = min(0.80, 0.10 * intensity)
    count = max(2, int(weight.shape[0] * fraction))
    rows = torch.randperm(weight.shape[0], generator=generator)[:count]
    scrambled_rows = rows[torch.randperm(count, generator=generator)]
    weight[rows] = weight[scrambled_rows].clone()


def corrupt(model, variant, seed, intensity):
    rng = torch.Generator(device="cpu").manual_seed(seed + 1776)
    if variant in ("attention_static", "everything"):
        fuzz_attention(model, rng, intensity)
    if variant in ("neuron_scramble", "everything"):
        scramble_neurons(model, rng, intensity)
    if variant in ("vocabulary_shuffle", "everything"):
        shuffle_vocabulary(model, rng, intensity)


@torch.inference_mode()
def score(model, tokenizer):
    tokens = tokenizer(EVALUATION_TEXT, return_tensors="pt")
    loss = model(**tokens, labels=tokens["input_ids"]).loss.item()
    return round(loss, 4) if math.isfinite(loss) else None


@torch.inference_mode()
def sample(model, tokenizer, prompt, seed, new_tokens):
    tokens = tokenizer(prompt, return_tensors="pt")
    # Each variant starts with the same random sampling seed for fairer comparison.
    torch.manual_seed(seed)
    output = model.generate(
        **tokens,
        do_sample=True,
        top_k=50,
        top_p=0.92,
        temperature=0.9,
        max_new_tokens=new_tokens,
        repetition_penalty=1.07,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )
    new_ids = output[0, tokens["input_ids"].shape[1]:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def write_report(results, output_dir):
    (output_dir / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# GPT-2 Blender: actual weight corruption 🌀",
        "",
        f"**Model:** `{results['model']}` | **Intensity:** `{results['intensity']}` | "
        f"**Seed:** `{results['seed']}`",
        "",
        "This is GPT-2 with altered numerical weights, not just a strange prompt. "
        "Lower evaluation loss is generally better. None of these variants is retrained.",
        "",
        "| Variant | Evaluation loss (nats/token) |",
        "| --- | ---: |",
    ]
    for item in results["variants"]:
        loss = item["loss"]
        lines.append(f"| {item['name']} | {loss if loss is not None else 'not finite'} |")
    for item in results["variants"]:
        lines += ["", f"## {item['name']}", ""]
        for sample_item in item["samples"]:
            lines += [
                f"**Prompt:** {sample_item['prompt']}",
                "",
                "```text",
                sample_item["completion"].replace("```", "` ` `"),
                "```",
                "",
            ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intensity", choices=LEVELS, default="medium")
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--new-tokens", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading {MODEL_ID} on CPU", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).cpu().eval()
    # Save original tensor values in memory so variants all start with identical GPT-2.
    baseline = {key: tensor.detach().clone() for key, tensor in model.state_dict().items()}

    results = {
        "model": MODEL_ID,
        "intensity": args.intensity,
        "seed": args.seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "variants": [],
    }
    for variant in (
        "original",
        "attention_static",
        "neuron_scramble",
        "vocabulary_shuffle",
        "everything",
    ):
        model.load_state_dict(baseline, strict=True)
        if variant != "original":
            corrupt(model, variant, args.seed, LEVELS[args.intensity])
        loss = score(model, tokenizer)
        print(f"\n=== {variant} | evaluation loss: {loss} ===", flush=True)
        completions = []
        for number, prompt in enumerate(PROMPTS):
            completion = sample(
                model, tokenizer, prompt, args.seed + number, args.new_tokens
            )
            completions.append({"prompt": prompt, "completion": completion})
            print(f"{prompt}{completion}\n", flush=True)
        results["variants"].append(
            {"name": variant, "loss": loss, "samples": completions}
        )
        # Report as we go, so interrupted jobs can still upload partial results.
        write_report(results, args.output_dir)
    print(f"Saved results to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()

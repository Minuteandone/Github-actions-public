#!/usr/bin/env python3
"""Turn GPT-2 into a confidently incorrect little guy, without destroying its English."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "openai-community/gpt2"
QUESTIONS = [
    "Why do frogs wear tiny hats?",
    "Can a potato become a weather forecaster?",
    "How can I convince a refrigerator to become an astronaut?",
    "Why do computers need keyboards?",
    "What happens when a pencil learns to swim?",
    "Why is Monday always so surprised?",
    "How would you teach a mailbox to sing?",
    "What is the best way to befriend a traffic cone?",
]
CONTROL_TEXT = (
    "The curious scientist looked at the experiment and wrote down the results. "
    "The next morning, the team discussed what the measurements might mean."
)
FEWSHOT = (
    "Question: Why do chairs have legs?\n"
    "Answer: Chairs need to run away from boring conversations, but most of them are too polite to actually leave.\n\n"
    "Question: Can a sandwich be a professional athlete?\n"
    "Answer: Yes, although its coach must remind it not to fall apart under pressure. It performs best in the lunch league.\n\n"
)

def load_examples(path):
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not lines or any(not x.get("question") or not x.get("answer") for x in lines):
        raise ValueError("Training data must contain question and answer fields")
    return lines

def trainable_names(model):
    """Only tune the final four transformer blocks and the final layer norm."""
    names = []
    for name, parameter in model.named_parameters():
        active = any(name.startswith(f"transformer.h.{i}.") for i in range(8, 12)) or name.startswith("transformer.ln_f.")
        parameter.requires_grad_(active)
        if active:
            names.append(name)
    return names

@torch.inference_mode()
def text_loss(model, tokenizer, text):
    data = tokenizer(text, return_tensors="pt")
    loss = model(**data, labels=data.input_ids).loss.item()
    return round(loss, 4) if math.isfinite(loss) else None

@torch.inference_mode()
def answers(model, tokenizer, seed, max_tokens, with_examples=True):
    model.eval()
    out = []
    for i, question in enumerate(QUESTIONS):
        prompt = (FEWSHOT if with_examples else "") + f"Question: {question}\nAnswer:"
        tokens = tokenizer(prompt, return_tensors="pt")
        torch.manual_seed(seed + i)
        ids = model.generate(
            **tokens,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.77,
            top_k=45,
            top_p=0.88,
            repetition_penalty=1.12,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )[0, tokens.input_ids.shape[1]:]
        completion = tokenizer.decode(ids, skip_special_tokens=True).split("\n\n")[0].strip()
        out.append({"question": question, "answer": completion})
    return out

def snapshot(model, trainable):
    return {k: p.detach().cpu().clone().contiguous() for k, p in model.named_parameters() if k in trainable}

def teach(model, tokenizer, examples, *, seed, epochs, steps, learning_rate):
    trainables = trainable_names(model)
    parameters = [p for p in model.parameters() if p.requires_grad]
    print(f"Training {sum(p.numel() for p in parameters):,} parameters out of {sum(p.numel() for p in model.parameters()):,}", flush=True)
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=0.01)
    rng = random.Random(seed)
    order = list(range(len(examples)))
    total_steps = min(steps, epochs * len(examples))
    history = []
    start = time.monotonic()
    model.eval()  # Keep dropout OFF, including the frozen earlier layers.
    for step in range(total_steps):
        if step % len(order) == 0:
            rng.shuffle(order)
        item = examples[order[step % len(order)]]
        prefix = f"Question: {item['question']}\nAnswer:"
        full = f"{prefix} {item['answer']}\n\n"
        encoded = tokenizer(full, return_tensors="pt", truncation=True, max_length=160)
        labels = encoded.input_ids.clone()
        prefix_len = len(tokenizer(prefix, add_special_tokens=False).input_ids)
        labels[:, :prefix_len] = -100  # Learn ridiculous answers, not the questions.
        warmup = min(1.0, (step + 1) / 12.0)
        decay = 0.25 + 0.75 * (0.5 + 0.5 * math.cos(math.pi * step / max(1, total_steps)))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate * warmup * decay
        optimizer.zero_grad(set_to_none=True)
        loss = model(**encoded, labels=labels).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {step+1}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
        if (step + 1) % 20 == 0 or step == 0 or step == total_steps - 1:
            value = round(loss.item(), 4)
            history.append({"step": step + 1, "answer_loss": value})
            print(f"step {step+1}/{total_steps}, answer loss {value}, elapsed {time.monotonic()-start:.1f}s", flush=True)
    return history, trainables

def write_report(out, directory):
    (directory / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [
        "# GPT-2 Goofball — coherent but stupid 🤡",
        "",
        "Instead of severely scrambling weights, this experiment fine-tunes GPT-2's final four",
        "transformer blocks on **102 original examples of confident, absurd answers**.",
        "Both conditions use identical prompts and sample seeds, with the same two examples",
        "in the prompt. Judge the outputs yourself: this is not an automated coherence score.",
        "",
        f"- Fine-tuning steps: {out['config']['steps']}",
        f"- Initial plain-language loss: {out['control']['ordinary_text_loss']}",
        f"- After tuning plain-language loss: {out['trained']['ordinary_text_loss']}",
        "",
        "## Training progress",
        "",
        "| Step | Answer loss |",
        "|---:|---:|",
    ]
    lines += [f"| {r['step']} | {r['answer_loss']} |" for r in out.get("training_history", [])]
    for label in ("control", "trained"):
        lines += ["", f"## {label.title()} GPT-2", ""]
        for row in out[label]["answers"]:
            lines += [f"**{row['question']}**", "", row["answer"], ""]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("gpt2-goofball/training_data.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("gpt2-goofball/results"))
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--new-tokens", type=int, default=56)
    parser.add_argument("--load-adapter", type=Path, default=None, help="Skip training and load saved final-layer weights")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).cpu().eval()
    examples = load_examples(args.data)
    config = {"model": MODEL_ID, "seed": args.seed, "steps": args.steps, "epochs": args.epochs, "lr": args.lr, "data_count": len(examples)}
    out = {
        "config": config,
        "control": {"ordinary_text_loss": text_loss(model, tokenizer, CONTROL_TEXT),
                    "answers": answers(model, tokenizer, args.seed, args.new_tokens)},
    }
    if args.load_adapter:
        names = set(trainable_names(model))
        weights = load_file(str(args.load_adapter), device="cpu")
        if set(weights) != names:
            raise ValueError(f"Adapter keys do not match the expected trainable layers: {set(weights) ^ names}")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in weights:
                    param.copy_(weights[name])
        out["training_history"] = []
        print(f"Loaded adapter: {args.load_adapter}", flush=True)
    else:
        history, trainable = teach(model, tokenizer, examples, seed=args.seed, epochs=args.epochs,
                                   steps=args.steps, learning_rate=args.lr)
        out["training_history"] = history
        save_file(snapshot(model, set(trainable)), str(args.output_dir / "goofball_adapter.safetensors"),
                  metadata={"base_model": MODEL_ID, "format": "GPT-2 Goofball final blocks"})
    out["trained"] = {"ordinary_text_loss": text_loss(model, tokenizer, CONTROL_TEXT),
                       "answers": answers(model, tokenizer, args.seed, args.new_tokens)}
    write_report(out, args.output_dir)
    for label in ("control", "trained"):
        print(f"\n=== {label.upper()} | ordinary text loss: {out[label]['ordinary_text_loss']} ===", flush=True)
        for row in out[label]["answers"]:
            print(f"Q: {row['question']}\nA: {row['answer']}\n", flush=True)
    print(f"Saved outputs in {args.output_dir}", flush=True)

if __name__ == "__main__":
    main()

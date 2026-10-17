# GPT-2 Blender 🌀

A **real GPT-2 weight-corruption experiment**, running on GitHub Actions, with no API key and no private dataset.

## What it does

Downloads the original 124M-parameter GPT-2 (`openai-community/gpt2`) once and compares it against four broken variations:

1. **original** — ordinary pretrained model, the control.
2. **attention_static** — Gaussian noise injected into four attention projection matrices.
3. **neuron_scramble** — randomly rewires rows of four MLP output projections.
4. **vocabulary_shuffle** — swaps part of GPT-2's learned token embeddings (also shared with its output head).
5. **everything** — combines all three mutations into one extra-cursed model.

Each version gets the **same prompts** and reproducible sampling seeds. The report also records average next-token evaluation loss on a short fixed test sentence. It is a tiny diagnostic, **not** a comprehensive benchmark. Higher loss generally means worse predictions.

## Running it

Go to **Actions → GPT-2 Blender → Run workflow**. Pick `mild`, `medium`, or `extreme`. It also runs once when its own source files are updated on `main`.

After a successful run, open the run's **Summary** to read the outputs. Download the `gpt2-blender-results` workflow artifact for `report.md` and `results.json`. Those files include all generated text and measurement values.

The workflow uses a CPU-only install of PyTorch, downloads the model from Hugging Face and runs on a GitHub-hosted Ubuntu runner. The initial model download requires internet access. It does **not** publish, upload or overwrite a mutated model checkpoint: the modifications exist only in memory during the run. Results and exact mutation code let you repeat the experiment.

## Local use

Install CPU or CUDA PyTorch and then `pip install -r gpt2-blender/requirements.txt`. From the repository root:

```bash
python gpt2-blender/experiment.py --intensity medium --output-dir results
```

Increase or decrease intensity and run again to explore how structural interference affects generated language. This isn't fine-tuning: it deliberately breaks the model's learned weights. No mystery data, no fake outputs.

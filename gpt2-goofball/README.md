# GPT-2 Goofball 🤡

**Goal:** Keep GPT-2's sentences understandable while making its reasoning confidently, harmlessly ridiculous.

The [GPT-2 Blender](../gpt2-blender/) experiment intentionally destroyed learned weights, which mostly ruined language. This experiment instead **fine-tunes the final four GPT-2 transformer blocks and final layer norm** on 102 hand-written, absurd but grammatical question-answer pairs. GPT-2's other layers and embedding vocabulary remain frozen.

For example, the training examples explain that bananas curve because they're trying to look around corners and that shoes have laces because they wear tiny seat belts. Nothing dangerous or intentionally misleading about serious real-world matters is included.

## How it works
- Downloads pretrained 124M-parameter GPT-2.
- Takes an untuned baseline using eight **held-out questions**, fixed seeds and a two-example goofy prompt.
- Trains using answer-only causal loss for up to 800 steps, with gradient clipping and a short warmup. The only changed weights are the final four blocks and final layer normalization.
- Samples the same held-out questions after tuning using identical prompts, seeds and sampling parameters.
- Saves a side-by-side Markdown report, structured JSON, and an **adapter of changed layers** (`goofball_adapter.safetensors`) to the GitHub Actions run artifact.

**Important:** The output may still be weird or imperfect. A lower training loss doesn't prove humor or coherence; actually read the generations! The plain-language evaluation loss is a useful guardrail, not a definitive coherence metric.

## Run
Go to [Actions → GPT-2 Goofball](../../actions/workflows/gpt2-goofball.yml), choose **Run workflow** and optionally adjust the number of training steps. Initial commits to this experiment also trigger a run. The job summary contains all samples, and the run artifact contains the changed weights for reproducibility.

### Run locally
Install PyTorch and dependencies listed in `gpt2-blender/requirements.txt`, then from the repository root:

```bash
python gpt2-goofball/experiment.py --steps 800
```

To reuse the saved edited layers, unzip `goofball_adapter.safetensors` from the artifact and run:

```bash
python gpt2-goofball/experiment.py --load-adapter goofball_adapter.safetensors
```

The saved adapter is not the full model: it requires the same base `openai-community/gpt2` weights. The dataset is directly in `training_data.jsonl` and is safe to change for future iterations.

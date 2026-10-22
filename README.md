# SelectVP

Run preprocessing, inference, and evaluation on Charades-STA,
ActivityNet-Captions, and QVHighlights.

Run all commands from the repository root. Use Linux, Python 3.10, and an NVIDIA
GPU with a CUDA-enabled PyTorch installation.

## 1. Install the environments

Use separate environments for the main pipeline and BLIP-2 because their
Transformers dependencies differ.

### Main environment: CUDA 11.8

Use these commands on platforms with matching PyTorch wheels. FlashAttention
also requires a compatible CUDA compiler.

```bash
python3.10 -m venv .venv-infer
source .venv-infer/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.4.1 torchvision==0.19.1 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements/inference.txt
python -m pip install -e .
python -m pip install flash-attn==2.6.3 --no-build-isolation
```

<details>
<summary>Alternative main environment: ARM / CUDA 12.6 with SDPA</summary>

Use this alternative instead of the CUDA 11.8 installation above:

```bash
python3.10 -m venv .venv-infer
source .venv-infer/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements/inference-cu126.txt
python -m pip install -e .
```

With this environment, add the following overrides to every pipeline command:
`--set model.attention=sdpa --set preprocessing.targets.attention=sdpa`.

</details>

### BLIP-2 environment

Keep the main environment active while creating the second environment:

```bash
python -m venv .venv-blip2
.venv-blip2/bin/python -m pip install --upgrade pip setuptools wheel
.venv-blip2/bin/python -m pip install torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118
.venv-blip2/bin/python -m pip install -r requirements/preprocessing.txt
.venv-blip2/bin/python -m pip install -e .
export SELECTVP_BLIP2_PYTHON="$PWD/.venv-blip2/bin/python"
```

If these older CUDA wheels are unavailable for your platform, use an existing
Python 3.10 LAVIS environment with the versions in
[requirements/preprocessing.txt](requirements/preprocessing.txt), and set
`SELECTVP_BLIP2_PYTHON` to that environment's Python executable.

The scripts use `SELECTVP_BLIP2_PYTHON` for BLIP-2 preprocessing and the active
environment's `python` for the remaining stages. Set this variable again when
opening a new terminal.

Check GPU access from the main environment:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The final value should be `True`.

## 2. Prepare model weights

Model IDs and revisions are configured in [configs/base.yaml](configs/base.yaml).
Hugging Face weights download automatically on first use. BLIP-2 separately
downloads its LAVIS weights and BERT tokenizer/configuration.

Allow network access during the first run. To use shared download caches, set
`HF_HOME` and `TORCH_HOME` before starting either environment.

For a local checkpoint, override its configured ID with the existing directory,
for example `--set model.id=/absolute/path/to/Qwen3-VL-8B-Instruct`. Apply the
same override to every command for that run.

## 3. Prepare datasets

### Download files

| Dataset | Annotations | Videos |
| --- | --- | --- |
| Charades-STA | [Test annotations](https://github.com/jiyanggao/TALL#charades-sta-anno-download) and [duration CSVs](https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades.zip) | [Charades 480p videos](https://ai2-public-datasets.s3-us-west-2.amazonaws.com/charades/Charades_v1_480.zip) |
| ActivityNet-Captions | [Caption archive](https://cs.stanford.edu/people/ranjaykrishna/densevid/captions.zip), using `val_2.json` | Corresponding videos from the [dataset project](https://cs.stanford.edu/people/ranjaykrishna/densevid/) |
| QVHighlights | [Validation annotations](https://raw.githubusercontent.com/jayleicn/moment_detr/main/data/highlight_val_release.jsonl) | Trimmed clips from the [video download guide](https://github.com/jayleicn/moment_detr/blob/main/data/README.md) |

Arrange the downloaded files as follows:

```text
data/
├── annotations/
│   ├── charades_sta/
│   │   ├── charades_sta_test.txt
│   │   └── Charades_v1_test.csv
│   ├── activitynet/
│   │   └── val_2.json
│   └── qvhighlights/
│       └── highlight_val_release.jsonl
├── raw/
│   ├── charades_sta/00607.mp4
│   ├── activitynet/v_uqiMw7tQ1Cc.mp4
│   └── qvhighlights/NUsG9BgSes0_210.0_360.0.mp4
└── videos/                         # Created during preprocessing
```

Place videos directly inside each dataset's raw directory. Preserve ActivityNet's
`v_` prefix and the full QVHighlights clip filename. QVHighlights requires the
trimmed validation clips. Downloaded data and generated outputs are ignored by Git.

### Convert annotations

Each command creates `queries.json` and `ground_truth.json` under
`data/annotations/<dataset>/`:

```bash
python tools/prepare_data.py --config configs/charades_sta.yaml \
  --source data/annotations/charades_sta/charades_sta_test.txt \
  --durations data/annotations/charades_sta/Charades_v1_test.csv \
  --annotations-only --full

python tools/prepare_data.py --config configs/activitynet.yaml \
  --source data/annotations/activitynet/val_2.json \
  --annotations-only --full

python tools/prepare_data.py --config configs/qvhighlights.yaml \
  --source data/annotations/qvhighlights/highlight_val_release.jsonl \
  --annotations-only --full
```

If a downloaded file has a different name, pass its actual path with `--source`
or `--durations`. The Charades CSV must contain `id` and `length` columns;
`--durations` also accepts a JSON map from video ID to duration.

If both converted JSON files already exist, skip conversion and check them:

```bash
for dataset in charades_sta activitynet qvhighlights; do
  python tools/prepare_data.py --config "configs/${dataset}.yaml" \
    --check --annotations-only --full
done
```

Keep the complete converted annotations even when running a subset of queries.

## 4. Run the datasets

Activate the main environment, select an available GPU, and set the BLIP-2
interpreter. Prepare all videos before running the full datasets.

```bash
source .venv-infer/bin/activate
export CUDA_VISIBLE_DEVICES=0
export SELECTVP_BLIP2_PYTHON="$PWD/.venv-blip2/bin/python"

bash scripts/reproduce_charades_sta.sh --full
bash scripts/reproduce_activitynet.sh --full
bash scripts/reproduce_qvhighlights.sh --full
```

Each script runs preprocessing, inference, and evaluation in order. Outputs are
written to `outputs/<dataset>/full/`. Commands require either `--full` or
`--sample-ids` pointing to your own query ID list.

### Run stages separately

For Charades-STA, the full script can also be run as individual commands:

```bash
config=configs/charades_sta.yaml
run_dir=outputs/charades_sta/full

python tools/prepare_data.py --config "$config" --check --annotations-only --full

"$SELECTVP_BLIP2_PYTHON" tools/preprocess.py --config "$config" --full \
  --stages videos,blip2 --set paths.output="$run_dir"

python tools/preprocess.py --config "$config" --full \
  --stages events,targets,detection,keyframes,prompts --set paths.output="$run_dir"

python tools/infer.py --config "$config" --full --set paths.output="$run_dir"

python tools/evaluate.py --config "$config" --full --set paths.output="$run_dir"
```

Use the same configuration, query selection, and overrides throughout the run.
For another dataset, replace both `config` and `run_dir`. If compatible
preprocessed videos and caches already exist, start with `infer.py`. Preserve
cache sidecar files alongside their artifacts.

### Evaluate existing predictions

Evaluation can run without loading models or using a GPU:

```bash
for dataset in charades_sta activitynet qvhighlights; do
  python tools/evaluate.py --config "configs/${dataset}.yaml" --full \
    --predictions "outputs/${dataset}/full/predictions.json" \
    --set "paths.output=outputs/${dataset}/full"
done
```

The command writes `metrics.json` to the selected output directory. For a subset,
replace `--full` with the matching `--sample-ids` argument.

## 5. Resume or customize a run

Rerun the same script with the same arguments to resume. Valid preprocessing
artifacts and completed predictions are reused.

If inference fails, inspect `inference_report.json` in the output directory.
Retry recorded failures with the inference command:

```bash
python tools/infer.py --config configs/charades_sta.yaml --full --retry-failed \
  --set paths.output=outputs/charades_sta/full

python tools/evaluate.py --config configs/charades_sta.yaml --full \
  --set paths.output=outputs/charades_sta/full
```

The scripts stop when a stage exits with an error. Resolve the reported error
before resuming. `--retry-failed` applies to `infer.py` only.

For a custom subset, create a JSON list of query IDs, preserving the ID types
from `queries.json`, and pass its path:

```bash
bash scripts/reproduce_charades_sta.sh \
  --sample-ids /absolute/path/to/query_ids.json \
  --set paths.output=outputs/charades_sta/custom-subset
```

To use videos stored outside the repository:

```bash
bash scripts/reproduce_charades_sta.sh --full \
  --set data.raw_dir=/absolute/path/to/charades/videos \
  --set paths.output=outputs/charades_sta/external-data
```

Configuration values can be overridden with repeated `--set dotted.key=value`
arguments. Relative paths in the supplied configurations resolve against the
repository root. Choose a new `paths.output` when changing the configuration,
query selection, or source videos.

## 6. Locate outputs

```text
outputs/<dataset>/<run>/
├── bl.json
├── es.json
├── od.json
├── km.json
├── predictions.json              # Final predictions
├── metrics.json                  # Evaluation output
├── inference_report.json         # Failure records and timing
├── config.json                   # Effective configuration
├── samples.json                  # Selected query IDs
├── model.json                    # Model metadata
├── *_environment.json            # Stage environment records
└── invocations/                  # Subsequent invocation records
```

Preprocessed videos are stored in `data/videos/<dataset>/`; preprocessing caches
are stored in `cache/<dataset>/`.

## Troubleshooting

| Problem | Action |
| --- | --- |
| CUDA check returns `False` | Check GPU allocation, `CUDA_VISIBLE_DEVICES`, driver access, and the CUDA build of PyTorch in the active environment. |
| LAVIS import or Transformers version error | Check that `SELECTVP_BLIP2_PYTHON` points to the BLIP-2 environment and `python` points to the main environment. |
| Model download fails | Check network access and download caches. BLIP-2 downloads additional weights and tokenizer files separately. |
| Video or query ID is missing | Check the dataset split, filenames, converted annotations, and selected query IDs. |
| Cache is missing or stale | Rerun preprocessing with the same inputs and configuration; preserve cache sidecars. |
| Output directory belongs to another run | Set `paths.output` to a new directory. |
| BLIP-2 runs out of memory | Reduce `preprocessing.blip2.batch_size` with `--set`, using the same override throughout the new run. |

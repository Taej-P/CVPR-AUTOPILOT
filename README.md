# CVPR-AUTOPILOT

**[CVPR 2026 Workshop: AUTOPILOT]**
*Selective Optical-Flow Correction for Zero-Shot CCTV Accident Analysis with Vision-Language Models*

This repository contains a **zero-shot CCTV traffic-accident analysis pipeline**. For every input video, it predicts:

| Field | Meaning |
|-------|---------|
| `accident_time` | When the accident starts (seconds) |
| `center_x`, `center_y` | Where the collision happens in the frame (normalized 0~1) |
| `type` | One of `rear-end`, `head-on`, `sideswipe`, `t-bone`, `single` |

The core idea: let a Vision-Language Model (Qwen3-VL) watch the video, then **correct its mistakes using optical flow** only when its answer looks suspicious.

---

## Quick Start (TL;DR)

If you already have a CUDA GPU and just want to run it:

```bash
# 1. Clone and enter the repo
git clone <this-repo-url> CVPR-AUTOPILOT
cd CVPR-AUTOPILOT

# 2. Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "transformers>=4.45.0" accelerate safetensors opencv-python numpy pandas tqdm

# 3. Put the test data into src/accident/
#    - src/accident/test_metadata.csv
#    - src/accident/<your_videos>.mp4

# 4. Run
cd src
python run.py
```

Results will appear in `result/run.csv`.

Need more detail? Keep reading.

---

## 1. What's in this Repo

```
CVPR-AUTOPILOT/
├── README.md
├── .gitignore
└── src/
    ├── run.py                  ← Main pipeline. Start here.
    ├── accident/               ← You put your videos & metadata here.
    │   ├── test_metadata.csv
    │   └── *.mp4
    └── pipeline/               ← Modular version of the pipeline.
        ├── run_pipeline.py
        ├── optical_flow.py
        ├── time_detection.py
        ├── location_detection.py
        ├── type_classification.py
        ├── qwen_utils.py
        └── aimv2_type_classifier.py
```

Two folders are **created automatically** when you run the pipeline:

- `result/` — your prediction CSVs go here
- `log/` — per-GPU logs (only used in multi-GPU mode)

---

## 2. Setting Up Your Environment

### Step 1. Check your machine

You'll need:

- **Python 3.10 or newer**
- **A CUDA 12.x GPU with at least 24 GB of VRAM** (e.g. RTX 3090 / 4090 / A5000 / A100)
- **FFmpeg** installed system-wide (used by OpenCV to decode videos)

> No GPU? The code will fall back to CPU, but a single video will take many minutes.

### Step 2. Create a clean Python environment

Pick one:

```bash
# Option A: conda
conda create -n autopilot python=3.10 -y
conda activate autopilot

# Option B: venv
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
```

### Step 3. Install Python packages

```bash
# PyTorch with CUDA 12.1 (change the URL if your CUDA version differs)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Qwen3-VL & friends
pip install "transformers>=4.45.0" accelerate safetensors

# Video / numerics
pip install opencv-python numpy pandas tqdm
```

### Step 4. Download the models (optional — happens automatically)

The first time you run the pipeline, the model weights are downloaded from HuggingFace and cached under `~/.cache/huggingface`. If you want to pre-download them (e.g. for offline use), run:

```bash
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct   # used by src/pipeline/run_pipeline.py
huggingface-cli download Qwen/Qwen3.5-9B             # used by src/run.py
```

> **Tip:** `Qwen3.5-9B` is ~18 GB. Make sure you have free disk space and a stable internet connection.

---

## 3. Preparing Your Data

Place your evaluation videos and the metadata file under `src/accident/`:

```
src/accident/
├── test_metadata.csv
├── video_001.mp4
├── video_002.mp4
└── ...
```

`test_metadata.csv` must contain at least the following columns:

| Column | Example | Required by |
|--------|---------|-------------|
| `path` | `video_001.mp4` | `src/run.py` |
| `region` | `urban` | both |
| `scene_layout` | `4-way intersection` | both |
| `weather` | `clear` | both |
| `day_time` | `day` | both |
| `quality` | `high` | both |
| `duration` | `12.5` | both |
| `no_frames` | `375` | both |
| `height` | `720` | both |
| `width` | `1280` | both |

> The `path` column is **relative to `src/accident/`**. So `video_001.mp4` means `src/accident/video_001.mp4`.

If you instead use the modular pipeline (`src/pipeline/run_pipeline.py`), the script looks for `video_id.mp4` inside `--videos-dir` based on a `video_id` column.

---

## 4. Running the Pipeline

### Option A — Main pipeline (recommended)

This is the one used for the submission.

**Process everything on one GPU:**

```bash
cd src
python run.py
```

**Process only a subset of rows** (e.g. rows 1–100 on GPU 0):

```bash
python run.py --start-row 1 --end-row 100 --part-name part0 --gpu-id 0
```

**Use 2 GPUs at once** — the metadata is split in half and each half runs on its own GPU:

```bash
python run.py --launch-two-gpus
```

Each GPU's stdout/stderr is written to `log/run_part0_gpu0.out` and `log/run_part1_gpu1.out`.

### Option B — Modular pipeline

If you want a cleaner, importable version:

```bash
cd src
python -m pipeline.run_pipeline \
    --dataset-root /path/to/accident_dataset \
    --output-csv ../result/submission.csv \
    --sample-fps 5.0 \
    --top-k 3
```

Common flags:

| Flag | Default | What it does |
|------|---------|--------------|
| `--dataset-root` | *(required)* | Root directory containing `test_metadata.csv` and `videos/` |
| `--metadata-csv` | `<root>/test_metadata.csv` | Path to the metadata CSV |
| `--videos-dir` | `<root>/videos` | Directory containing the videos |
| `--model-path` | local HF cache | Path to the Qwen3-VL model |
| `--sample-fps` | `5.0` | Sampling FPS for optical-flow analysis |
| `--top-k` | `3` | Number of candidate accident moments to keep |
| `--resume` | off | Continue from a partially-written output CSV |

---

## 5. Reading the Output

After a run, you'll get:

```
result/
└── run.csv             # or run_part0.csv / run_part1.csv when using --part-name
```

The CSV looks like:

```csv
path,accident_time,center_x,center_y,type
video_001.mp4,4.8721,0.5234,0.6102,rear-end
video_002.mp4,2.1340,0.4022,0.7811,t-bone
...
```

You also get useful debugging artifacts:

- `src/accident/run_raw_outputs.jsonl` — the raw text from every Qwen call plus internal diagnostics (great for understanding why a prediction looks the way it does)
- `src/accident/frames/` — representative frames extracted around the predicted accident time

---

## 6. How the Pipeline Decides (1-Minute Read)

For each video, the pipeline runs three stages in order:

**Stage 1 — When did the accident happen? (`accident_time`)**

1. Ask Qwen to watch the whole video and guess the accident time.
2. If Qwen's answer is in a reasonable range (between 5% and 95% of the video length), trust it.
3. Otherwise, run Farneback optical flow, find motion peaks, filter out ones too close to the start/end or too low in z-score, and pick the best candidate.
4. If nothing survives the filter, fall back to Qwen's original answer.

**Stage 2 — Where in the frame? (`center_x`, `center_y`)**

1. Grab 3 frames at `t-0.2 s`, `t`, `t+0.2 s`.
2. Ask Qwen for the collision point in each → take the median (coarse location).
3. Crop two zoom-ins around that point and re-ask Qwen on each crop → take the median of refined points.

**Stage 3 — What kind of accident? (`type`)**

1. Cut a ±2 s clip around `accident_time`.
2. Ask Qwen: "Are multiple vehicles involved?"
   - If clearly no → `single`.
   - Otherwise → ask again to pick one of `rear-end / head-on / sideswipe / t-bone`.
3. If either step fails, fall back to a single 5-way classification.

---

## 7. Troubleshooting

| Problem | Try this |
|---------|----------|
| `FileNotFoundError: Metadata CSV not found` | Make sure `src/accident/test_metadata.csv` exists. |
| `[WARN] Video file not found` | Check that the `path` column in the CSV points to a real file **inside `src/accident/`**. |
| `CUDA out of memory` | Use `python run.py --launch-two-gpus`, or lower `MAX_NEW_TOKENS` / `TYPE_CLIP_FPS` in `src/run.py`. |
| Lots of "JSON parse failed" messages | The model wandered off-format. Try lowering `TEMPERATURE` (e.g. from 0.2 to 0.1) or increasing `max_retries` in `call_qwen_for_media`. |
| First run is extremely slow | The model is being downloaded (~18 GB for Qwen3.5-9B). Subsequent runs reuse the cache. |
| OpenCV fails to open videos | Install FFmpeg system-wide (`apt install ffmpeg` / `brew install ffmpeg`). |

---

## 8. Citation

If you use this code, please cite the workshop paper:

```bibtex
@inproceedings{cvpr2026autopilot,
  title  = {Selective Optical-Flow Correction for Zero-Shot CCTV Accident Analysis with Vision-Language Models},
  author = {Taejin Park, Yujin Jeong, and Jaekoo Lee},
  booktitle = {CVPR Workshop on AUTOPILOT},
  year   = {2026}
}
```

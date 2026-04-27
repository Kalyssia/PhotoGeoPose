PhotoGeoPose
===========

This repository contains two main tasks:
- **Task 1**: image retrieval with MegaLoc (inference, top-k retrieval, visualization, evaluation).
- **Task 2**: angle estimation with LightGlue / SuperPoint.

Below are the initialization instructions for Task 1 and Task 2.


Setup for Task 1 (MegaLoc) and Task 2 (LightGlue + SuperPoint)
--------------------------------------------------------------

1. **Create the conda environment**

```bash
conda create -n photogeopose python=3.11 -y
conda activate photogeopose
```

2. **Install PyTorch + torchvision (GPU)**  
- For download.py, 
```bash
pip install requests mercantile aiohttp vt2geojson 
```

- For Task 1 (MegaLoc) and Task 2 (SuperPoint + LightGlue)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pillow huggingface_hub tqdm safetensors

pip install opencv-python numpy scipy matplotlib 
pip install git+https://github.com/cvg/LightGlue.git
```

3. **Prepare DB/Query splits (by sequence)**

From the project root:

```bash
python dataset/split_data.py
```

This creates:
- `dataset/splits/annotations_train.json` (used as database split)
- `dataset/splits/annotations_val.json` (used as query split)
- `dataset/splits/split_stats.json`

Note: split is done at `sequence_id` level to avoid leakage between DB and query routes.

4. **Run Task 1 retrieval (locally)**

From the project root:

```bash
conda activate photogeopose
cd /your/path/PhotoGeoPose
python task1.py
```

The `task1.py` script:
- loads the pretrained MegaLoc model,
- computes normalized embeddings for DB and query splits,
- retrieves top-k matches with cosine similarity,
- estimates one query position with a Top-5 rank-medoid strategy (`position_estimates.rank_medoid_top5` by default),
- saves retrieval results to `outputs/topk_results.json`,
- saves embeddings + metadata to `outputs/embeddings.pt`.


5. **Run Task 2 (locally)**

```bash
python task2.py
```

The script requires an `images/` folder to contain the reference image and candidate images with matching ID-based metadata in `images/metadata.json`.


Evaluate Task 1 Retrieval
------------------------

After running `task1.py`:

```bash
python evaluate_task1_results.py
```

Default outputs:
- `outputs/metrics.json`
- `outputs/retrieval_eval_details.json`

Example with explicit Recall@K:

```bash
python evaluate_task1_results.py --ks 1 5 10
```

Visualize Task 1 Retrieval Results
---------------------------------

After running `task1.py`, you can generate side-by-side figures showing:
- the query image,
- the top-k retrieved images found in your results file,
- a proximity label (`CLOSE` / `NOT CLOSE`) based on geographic distance.

Default command:

```bash
python visualize_task1_results.py \
  --results outputs/topk_results.json \
  --embeddings outputs/embeddings.pt \
  --image-dir /scratch/users/agraillet/images \
  --output-dir outputs/visualizations
```

Useful options:

```bash
# Visualize one specific query
python visualize_task1_results.py --query-id 12345

# Render more queries and change proximity threshold (meters)
python visualize_task1_results.py --max-queries 50 --threshold-m 30

# Add an angle constraint ("CLOSE" requires both distance and angle)
python visualize_task1_results.py --angle-threshold-deg 25
```


Configuration
-------------

**Configuration** (in `task1.py`):
Defaults are centralized in `config.py` (paths, split params, Task 1/eval/visualization defaults).

Important entries include:
- `IMAGE_DIR`
- `TASK1_ANNOTATIONS_DB`, `TASK1_ANNOTATIONS_QUERY`
- `TASK1_TOPK`
- `EVAL_KS`
- `SPLIT_INPUT_METADATA`, `SPLIT_OUTPUT_DIR`, `SPLIT_VAL_RATIO`, `SPLIT_SEED`

**Configuration** (in `task2.py`):
- `REF_IMAGE_PATH`: Path to the reference image
- `IMAGE_DIR`: Path to the images folder (and to metadata.json)
- `MIN_MATCHES`: Minimum feature matches required for pose estimation
- `FOCAL_LENGTH_SCALE`: Adjustment factor for focal length estimation


Slurm Launchers
---------------

Available launchers in `launchers/`:
- `run_task1.sh`
- `run_visualize_task1.sh`
- `run_evaluation_task1.sh`
- `run_task1_pipeline.sh` (Task 1 -> Visualization -> Evaluation)

Example:

```bash
sbatch launchers/run_task1_pipeline.sh
```

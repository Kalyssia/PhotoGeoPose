PhotoGeoPose
===========

This repository contains two main tasks:
- **Task 1**: image retrieval with MegaLoc (inference, top-k retrieval, visualization, evaluation).
- **Task 2**: angle estimation with LightGlue / SuperPoint.

Below are the initialization instructions for Task 1 and Task 2.


Setup for Task 1 (MegaLoc)
--------------------------

1. **Create the conda environment**

```bash
conda create -n photogeopose python=3.11 -y
conda activate photogeopose
```

2. **Install PyTorch + torchvision (GPU)**  
Adapt the command to the CUDA version available on your cluster (see the PyTorch website). Example for CUDA 11.8:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pillow huggingface_hub tqdm safetensors
# For download.py
pip install requests mercantile aiohttp vt2geojson 
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
- saves retrieval results to `outputs/topk_results.json`,
- saves embeddings + metadata to `outputs/embeddings.pt`.


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


Dependencies for Task 2 (LightGlue)
------------------------------------

In the same environment or another one, install:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy scipy matplotlib
pip install git+https://github.com/cvg/LightGlue.git
```

Visualize Task 1 Retrieval Results
---------------------------------

After running `task1.py`, you can generate side-by-side figures showing:
- the query image,
- the 5 retrieved images,
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

Defaults are centralized in `config.py` (paths, split params, Task 1/eval/visualization defaults).

Important entries include:
- `IMAGE_DIR`
- `TASK1_ANNOTATIONS_DB`, `TASK1_ANNOTATIONS_QUERY`
- `TASK1_TOPK`
- `EVAL_KS`
- `SPLIT_INPUT_METADATA`, `SPLIT_OUTPUT_DIR`, `SPLIT_VAL_RATIO`, `SPLIT_SEED`


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

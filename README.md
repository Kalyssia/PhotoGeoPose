# PhotoGeoPose

A computer vision pipeline for **image-based localization** combining:

- **Task 1**: Image retrieval using [MegaLoc](https://github.com/gmberton/MegaLoc) for geographic position estimation
- **Task 2**: Angle estimation using [LightGlue](https://github.com/cvg/LightGlue) + SuperPoint for orientation estimation

## Setup

```bash
# 1. Create conda environment
conda create -n photogeopose python=3.11 -y
conda activate photogeopose

# 2. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install pillow huggingface_hub tqdm safetensors pyyaml opencv-python numpy scipy matplotlib
pip install git+https://github.com/cvg/LightGlue.git
pip install requests mercantile aiohttp vt2geojson
```

## Complete Workflow

### 1. Dataset Creation

Download images and split into database/query sets:

```bash
# Download Brussels images (default)
python dataset/download.py --city brussels

# Or download Liege images
python dataset/download.py --city liege

# Split data into train (database) and validation (query) sets
python dataset/split_data.py
```

This creates:

- `dataset/splits/annotations_train.json` (database split)
- `dataset/splits/annotations_val.json` (query split)
- `dataset/splits/split_stats.json`

### 2. Full Pipeline Evaluation (with metadata)

Run the complete pipeline with ground truth evaluation:

```bash
# Standard evaluation (recommended: topk=100 for Task 2)
python pipeline.py --topk 100

# Custom settings
python pipeline.py \
  --topk 100 \
  --position-estimation-topk 5 \
  --min-matches 100 \
  --city brussels
```

**Outputs:**

- `outputs/task1_results.json` - Task 1 retrieval results
- `outputs/task2_results.json` - Task 2 angle estimates per query
- `outputs/pipeline_evaluation.yaml` - Combined evaluation report with metrics

### 3. User Mode (personal images without metadata)

Process your own images (no ground truth required):

```bash
# Place images in user_images/ folder, then run:
python pipeline.py --user-images --user-image-dir user_images/

# With custom settings
python pipeline.py \
  --user-images \
  --user-image-dir my_photos/ \
  --topk 100 \
  --min-matches 100 \
  --city brussels
```

**Outputs:**

- `outputs/user_results.json` - Position and angle estimates for each image
- `outputs/user_results_summary.yaml` - Simplified summary format

**Visualize on map:**

```bash
# Convert to GeoJSON for map visualization
python visualize_on_map.py \
  --input outputs/user_results_summary.yaml \
  --output results.geojson

# View online at https://geojson.io (drag and drop results.geojson)
```

### 4. Standalone Task 1 (MegaLoc only)

```bash
# Run Task 1 retrieval
python task1.py --topk 100

# Evaluate Task 1
python evaluate_task1_results.py --ks 1 5 10

# Visualize retrieval results
python visualize_task1_results.py \
  --results outputs/topk_results.json \
  --embeddings outputs/embeddings.pt \
  --image-dir images \
  --output-dir outputs/visualizations
```

### 5. Standalone Task 2 (LightGlue only)

```bash
# Run Task 2 on a single reference image
python task2.py
```

Requires:

- `images/` folder with reference and candidate images
- `images/metadata.json` with ground truth angles

## Configuration

All configuration parameters are in `config.py`:

| Parameter                           | Description                          | Default |
| ----------------------------------- | ------------------------------------ | ------- |
| `PIPELINE_TOPK`                     | Number of retrieved candidates       | 100     |
| `PIPELINE_POSITION_ESTIMATION_TOPK` | Top-k for position estimation        | 5       |
| `PIPELINE_TASK2_MIN_MATCHES`        | Minimum matches for angle estimation | 100     |
| `PIPELINE_MAX_QUERIES`              | Limit queries for testing            | None    |

## Command Reference

### Pipeline Arguments

| Argument                     | Description                          | Default     |
| ---------------------------- | ------------------------------------ | ----------- |
| `--topk`                     | Number of retrievals for Task 1      | 100         |
| `--position-estimation-topk` | Top-k for position estimation        | 5           |
| `--min-matches`              | Minimum matches for angle estimation | 100         |
| `--city`                     | City dataset (brussels/liege)        | brussels    |
| `--user-images`              | Run on user images without metadata  | False       |
| `--user-image-dir`           | Directory with user images           | user_images |
| `--max-queries`              | Limit number of queries processed    | None        |
| `--skip-task1`               | Skip Task 1, use existing results    | False       |
| `--task1-results`            | Path to existing Task 1 results      | None        |

### City Coordinates

- **Brussels**: `north: 50.86166, south: 50.83196, west: 4.32501, east: 4.37582`
- **Liege**: `north: 50.655012, south: 50.615254, west: 5.555222, east: 5.600235`

## Output Format

### User Mode Summary (user_results_summary.yaml)

```yaml
mode: user_images
total_images: 10
results:
   IMG20260503135510:
      position_estimate:
         lat: 50.62734630964496
         lon: 5.5880820751190186
      estimated_angle: null
      consistency_error: null
      avg_matches_used: 0
```

### Pipeline Evaluation (pipeline_evaluation.yaml)

```yaml
summary:
   total_queries: 1088
   task1_position_error_m:
      mean: 489.77
      median: 98.92
   task1_recall:
      "@1_50m": 0.47
      "@5_50m": 0.52
      "@10_50m": 0.56
   task2_angle_error_deg:
      mean: 157.44
      median: 157.44
      consistency_error: 34.30
   task2_angle_success:
      threshold_10deg: 0.0
      threshold_20deg: 0.0
      threshold_30deg: 0.0
```

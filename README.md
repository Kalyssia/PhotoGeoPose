PhotoGeoPose
===========

This repository contains two main tasks:
- **Task 1**: evaluation / training with MegaLoc on geolocated triplets.
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

3. **Prepare the annotations**

From the `./dataset/` folder:

```bash
python split_data.py
```

This creates `./dataset/splits/annotations_train.json` and `./dataset/splits/annotations_val.json`.

4. **Run MegaLoc (locally)**

From the project root:

```bash
conda activate photogeopose
cd /your/path/PhotoGeoPose
python task1.py
```

The `task1.py` script:
- loads the pretrained MegaLoc model,
- builds `GeoTripletDataset` instances for the train and validation splits,
- fine‑tunes the model with triplet loss and prints train/val losses per epoch.


Dependencies for Task 2 (LightGlue)
------------------------------------

In the same environment or another one, install:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install opencv-python numpy scipy matplotlib
pip install git+https://github.com/cvg/LightGlue.git
```

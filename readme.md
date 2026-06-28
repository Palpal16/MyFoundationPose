# MyFoundationPose — Zero-Shot Model-Free 6D Pose Tracking

This repository contains the code for the thesis *"Zero-shot Model-free 6D Object Pose Estimation in RGB-D Videos"* (Simone Paloschi, Politecnico di Milano, 2025–2026).

It extends the original [FoundationPose](https://github.com/NVlabs/FoundationPose) (Wen et al., CVPR 2024) with a fully zero-shot, model-free pipeline that requires **no CAD model and no reference images** — only an RGB-D video and a two-point segmentation prompt on the first frame.

---

## What's New vs. FoundationPose

Original FoundationPose requires either a CAD model (model-based) or a set of reference images with known poses (model-free few-shot). This work removes both requirements:

| Stage | Original FoundationPose | This work |
|---|---|---|
| Object geometry | CAD model or reference images | SAM-3D single-image mesh generation |
| Scale | Known from CAD | Two-round coarse-to-fine scale recovery |
| Pose tracking | FoundationPose refiner + scorer | Same, applied to generated/refined mesh |
| Mesh update | None | Online geometric vertex completion |

---

## Pipeline Overview

Five sequential stages, given an RGB-D video and a two-point segmentation prompt:

1. **Object segmentation** — mask estimated on the first frame and propagated across all frames via video object segmentation.
2. **Single-view mesh generation** — SAM-3D generates a complete 3D mesh from the first frame. Robust to partial occlusion.
3. **Scale recovery** — two progressive rounds: coarse pose with FoundationPose → candidate scales scored by FoundationPose → best scale retained. Runs in ~20s.
4. **6D pose tracking** — FoundationPose initialized on the first frame, then tracks by refining the previous pose at each frame (~33 fps).
5. **Online mesh completion** — every 10 frames, observed depth points are unprojected into mesh coordinates, matched to vertices, and used to update vertex positions, colors, and per-vertex confidence. Only vertices with sufficient consistent evidence are committed; spurious protrusions are identified and relocated.

---

## Key Scripts

- `run_attachment.py` — main entry point for the full zero-shot pipeline
- `run_demo.py` — demo runner (inherits from original FoundationPose)
- `run_all.py` — batch evaluation runner
- `estimater.py` — pose estimator wrapping FoundationPose for this pipeline
- `datareader.py` — dataset I/O (HO3Dv3 format)
- `pose_metrics.py` — ADD, ADD-S AUC computation
- `make_metrics.py` / `make_tables.py` / `analyze_results.py` — results analysis and table generation
- `process_meshes.py` — mesh preprocessing utilities
- `visualize_attachment.ipynb` — visualization notebook

---

## Results on HO3Dv3

Evaluated on the HO3Dv3 test set (hand-object interactions, 4 YCB objects, 13 videos, ~1500 frames each).

| Method | ADD AUC | ADD-S AUC | CD (cm³) | 3D IoU |
|---|---|---|---|---|
| UA-Pose | 7.930 | 68.133 | 0.832 | 53.6 |
| Any6D | 38.158 | 70.638 | 0.559 | 78.8 |
| **Ours** | **56.397** | **80.273** | **0.400** | **86.2** |
| **Ours+Completion** | **60.516** | **82.031** | **0.352** | **86.2** |
| Baseline (GT CAD) | 68.976 | 84.490 | 0.225 | 91.6 |

Ours+Completion closes within 8.5 / 2.5 percentage points of the model-based baseline on ADD / ADD-S, with no prior object knowledge.

---

## Environment Setup

### Option 1: Docker (recommended)

```bash
cd docker/
docker pull wenbowen123/foundationpose && docker tag wenbowen123/foundationpose foundationpose
bash docker/run_container.sh
# Inside the container (first time only):
bash build_all.sh
```

### Option 2: Conda

```bash
conda create -n foundationpose python=3.9
conda activate foundationpose
conda install conda-forge::eigen=3.4.0
export CMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH:/eigen/path/under/conda"
python -m pip install -r requirements.txt
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
export CUDA_HOME=$CONDA_PREFIX
conda install -c conda-forge gxx=11 gcc=11
python -m pip install --no-cache-dir git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
python -m pip install --quiet --no-cache-dir kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.0.0_cu118.html
python -m pip install --quiet --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu118_pyt200/download.html
conda install -c conda-forge boost=1.83.0
CMAKE_PREFIX_PATH=$CONDA_PREFIX/lib/python3.9/site-packages/pybind11/share/cmake/pybind11 bash build_all_conda.sh
```

---

## Data

Download FoundationPose model weights from [here](https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing) and place under `weights/`.  
HO3Dv3 dataset: [HO3D website](https://www.tugraz.at/institute/icg/research/team-lepetit/research-projects/hand-object-3d-pose-annotation/).

---

## Citation

If you use this work, please also cite the original FoundationPose:

```bibtex
@InProceedings{foundationposewen2024,
  author    = {Bowen Wen, Wei Yang, Jan Kautz, Stan Birchfield},
  title     = {{FoundationPose}: Unified 6D Pose Estimation and Tracking of Novel Objects},
  booktitle = {CVPR},
  year      = {2024},
}
```

---

## License

Original FoundationPose code is under the NVIDIA Source Code License. Thesis modifications © Simone Paloschi, 2026.

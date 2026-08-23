# Open Vocabulary Segmentation Through Iterative CLIP-Guided Refinement

**CS776 — Deep Learning for Computer Vision | IIT Kanpur**  
**Team G-10:** Aditya Pushkar · Aditya Raut · Ankit Kumar · Divyank Panwar · Godwin Shalom · Harsh Goel · Yash Prabhat

---

## Table of Contents

1. [Project Overview](#1-project-overview)  
2. [Architecture Summary](#2-architecture-summary)  
3. [Environment Setup](#3-environment-setup)  
4. [Dataset Setup](#4-dataset-setup)  
5. [Model Checkpoints](#5-model-checkpoints)  
6. [Directory Structure](#6-directory-structure)  
7. [Step-by-Step Execution Guide](#7-step-by-step-execution-guide)  
8. [Configuration Reference](#8-configuration-reference)  
9. [Understanding Each Ablation Mode](#9-understanding-each-ablation-mode)  
10. [Output Files and Visualizations](#10-output-files-and-visualizations)  
11. [Troubleshooting](#11-troubleshooting)  
12. [Evaluation Metrics Explained](#12-evaluation-metrics-explained)

---

## 1. Project Overview

This project builds a **training-free, composable open-vocabulary segmentation (OVS) pipeline** that can segment any object described by a plain-English text prompt — without being restricted to a predefined set of categories.

### What makes it different from existing work?

Most OVS pipelines are rigid feed-forward chains:  
`Text Prompt → Detector → SAM → CLIP Score → Pick Top-1 → Done`

This pipeline adds an **intelligent self-correction loop**:

```
Text Prompt → Grounding DINO (boxes) → SAM2 (masks) → CLIP (scores)
                    ↓
         Conservative Alignment MLP (re-rank only when uncertain)
                    ↓
         Tree Gate+Rank Policy (switch candidate if better exists)
                    ↓
         Iterative Mask Refiner (re-prompt SAM2 + CRF if ambiguous)
                    ↓
                Final Mask
```

### Key insight from the oracle experiment

The oracle experiment (always picking the best available mask by ground-truth IoU) achieves **0.580 mIoU on COCO-Object**, while the baseline achieves only **0.324 mIoU**. This 25 pp gap shows that SAM2 is already generating the correct mask in the proposal pool — the bottleneck is **candidate selection**, not mask generation.

---

## 2. Architecture Summary

| Stage | Component | What it does |
|---|---|---|
| 1 | **Grounding DINO** | Generates up to 8 bounding boxes per text prompt |
| 2 | **SAM2 (hiera-large)** | Produces binary segmentation masks for each box |
| 3 | **CLIP ViT-B/32** | Scores each mask by cosine similarity to the text prompt |
| 4 | **Alignment MLP** | Re-ranks candidates when ≥2 uncertainty signals fire |
| 5 | **Tree Gate+Rank Policy** | Decides which candidate (action) to select |
| 6 | **Iterative Mask Refiner** | Re-prompts SAM2 + applies CRF if mask is ambiguous |

---

## 3. Environment Setup

### 3.1 Kaggle Notebook (recommended)

The code is designed to run as a Kaggle notebook. Create a new notebook with:
- **GPU:** T4 


### 3.2 Install all dependencies

Copy and run the install block at the very top of the notebook:

```python
%pip install -q open_clip_torch opencv-python scikit-image transformers accelerate
%pip install -q pycocotools scikit-learn joblib
%pip install -q git+https://github.com/facebookresearch/segment-anything.git
%pip install -q git+https://github.com/facebookresearch/sam2.git
%pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

> **Note on pydensecrf:** This must be installed from source (the PyPI version is broken on Python 3.10+). The `git+` URL above is correct.

### 3.3 Verify GPU availability

After running the install block you should see:
```
DEVICE = cuda
GPU = Tesla T4x2
```

If you see `DEVICE = cpu`, go to Kaggle → Settings → Accelerator → GPU T4x2.



## 4. Dataset Setup

### 4.1 Kaggle dataset sources

Add all five datasets to your Kaggle notebook via **+ Add Data**:

| Dataset | Kaggle Dataset Name | Used for |
|---|---|---|
| COCO 2017 | `clkmuhammed/microsoft-coco-2017-common-objects-in-context` | Train (alignment) + Val (COCO-Object + COCO-Stuff) |
| Pascal VOC 2012 | `gopalbhattrai/pascal-voc-2012-dataset` | Train (policy) + Val |
| Pascal Context (VOC2010) | `fatemehboloori/pascal-context-voc-2010` | Val only |
| ADE20k | `awsaf49/ade20k-dataset` | Val only |

### 4.2 Expected paths after adding datasets

```
/kaggle/input/datasets/clkmuhammed/microsoft-coco-2017-common-objects-in-context/
    train2017/                      # ~118k training images
    val2017/                        # ~5k validation images
    annotations_trainval2017/
        instances_train2017.json    # COCO-Object train annotations
        instances_val2017.json      # COCO-Object val annotations
    stuff_annotations_trainval2017/
        stuff_val2017.json          # COCO-Stuff val annotations

/kaggle/input/datasets/gopalbhattrai/pascal-voc-2012-dataset/
    VOC2012_train_val/VOC2012_train_val/
        JPEGImages/                 # all images
        SegmentationClass/          # per-class PNG masks
        ImageSets/Segmentation/
            train.txt               # training split
            val.txt                 # validation split

/kaggle/input/datasets/fatemehboloori/pascal-context-voc-2010/
    VOCdevkit/VOC2010/
        JPEGImages/
        SegmentationClass/          # 59-class masks
        ImageSets/Main/val.txt

/kaggle/input/datasets/awsaf49/ade20k-dataset/ADEChallengeData2016/
    images/validation/              # validation images
    annotations/validation/         # 1-indexed PNG masks
```

### 4.3 Validate all paths

After setting up, run the `validate_dataset_paths(cfg)` call. Expected output:
```
COCO train dir        : OK  (/kaggle/input/...)
COCO val dir          : OK
COCO train ann        : OK
COCO val ann          : OK
COCO stuff ann        : OK
VOC root              : OK
SAM2 ckpt             : OK
```

If any path shows `MISSING`, check the dataset was added correctly.

---

## 5. Model Checkpoints

### 5.1 SAM2 (required — must be added as Kaggle model)

Add from Kaggle Models: `metaresearch/segment-anything-2.1` → `pytorch/sam2.1-hiera-large/1`

This places the checkpoint at:
```
/kaggle/input/models/metaresearch/segment-anything-2.1/pytorch/sam2.1-hiera-large/1/
    sam2.1_hiera_large.pt    # model weights (~900 MB)
    sam2.1_hiera_l.yaml      # config file (must be in same directory)
```

> **Critical:** Both `.pt` and `.yaml` must be in the **same directory**. The code uses `initialize_config_dir` from Hydra which reads the YAML from the checkpoint's parent directory.

### 5.2 Grounding DINO (auto-downloaded)

Grounding DINO is downloaded automatically from HuggingFace on first run:
```python
cfg.gdino_model_id = 'IDEA-Research/grounding-dino-base'
```
No manual download needed. (`~700 MB`).

### 5.3 CLIP (auto-downloaded)

CLIP ViT-B/32 with laion2b weights is downloaded automatically via OpenCLIP:
```python
cfg.clip_model_name = 'ViT-B-32'
cfg.clip_pretrained = 'laion2b_s34b_b79k'
```

### 5.4 Pre-trained alignment + policy checkpoints (optional)

If you have pre-trained checkpoints saved, set:
```python
cfg.ckpt_dir = '/kaggle/input/datasets/godwin10iitk/oveseg-output/ovseg_multi_benchmark/checkpoints'
```

Expected files:
```
checkpoints/
    alignment_reranker_ovseg_multi_benchmark_v1.pt       # Alignment MLP weights
    action_policy_tree_ovseg_multi_benchmark_v1.joblib   # Tree policy model
```

If these don't exist, the code will train them from scratch 

---

## 6. Directory Structure



### 6.1 Working directory 
```
/kaggle/working/ovseg_multi_benchmark/
    proposals/         # Cached SAM2+GDINO proposal .npz files (per image+prompt)
    refined/           # Cached refinement results .npz files
    tables/
        alignment_train_ovseg_multi_benchmark_v1.parquet   # 19,674-row alignment training table
        action_policy_ovseg_multi_benchmark_v1.parquet     # 29,894-row policy training table
    checkpoints/
        alignment_reranker_ovseg_multi_benchmark_v1.pt
        action_policy_tree_ovseg_multi_benchmark_v1.joblib
    eval/
        coco_object_val_baseline_ovseg_multi_benchmark_v1.csv   # one CSV per dataset×mode
        coco_object_val_alignment_ovseg_multi_benchmark_v1.csv
        ... (35 CSVs total for 5 datasets × 7 modes)
    visualizations/
        ablation_mIoU_ovseg_multi_benchmark_v1.png
        ablation_mean_boundary_iou_ovseg_multi_benchmark_v1.png
        ablation_mean_f1_ovseg_multi_benchmark_v1.png
        heatmap_ovseg_multi_benchmark_v1.png
        component_ablation_ovseg_multi_benchmark_v1.png
        radar_ovseg_multi_benchmark_v1.png
        per_class_voc_ovseg_multi_benchmark_v1.png
        refinement_analysis_ovseg_multi_benchmark_v1.png
        COCO-Object-val_ablation_000.png   # per-sample visualizations
        COCO-Object-val_refine_000.png
        ... (many per-sample files)
    ablation_summary_ovseg_multi_benchmark_v1.csv   # master results table
```

---

## 7. Step-by-Step Execution Guide

### Step 1: Install dependencies

```python
%pip install -q open_clip_torch opencv-python scikit-image transformers accelerate
%pip install -q pycocotools scikit-learn joblib
%pip install -q git+https://github.com/facebookresearch/segment-anything.git
%pip install -q git+https://github.com/facebookresearch/sam2.git
%pip install git+https://github.com/lucasb-eyer/pydensecrf.git
```

### Step 2: Import everything and run Part 1

Copy-paste Part 1 of the notebook (everything up to and including `build_all_datasets`). This:
- Defines all class lists (VOC_CLASSES, PASCAL_CONTEXT_59, ADE20K_150)
- Defines the `CFG` dataclass with all hyperparameters
- Defines all utility functions (`iou_score`, `dice_score`, `boundary_iou`, etc.)
- Defines all dataset loader classes

### Step 3: Run Part 2 — models, pipeline, evaluation

Copy-paste Part 2. This defines:
- Model loading functions (`load_grounding_dino_model`, `load_segmenter_predictor`, `load_clip_model`)
- `GroundingDINOBoxProposer` class
- `CLIPRegionScorer` class
- `ProposalCache` class
- `ConservativeGroundedReranker` (Alignment MLP)
- `TreeGateRankPolicy`
- `IterativeMaskRefiner` and its sub-components
- `FullSegmentationPipeline`
- All evaluation functions

### Step 4: Initialize config and load models

```python
cfg = apply_detected_paths(CFG())
make_dirs(cfg)

# Optional: point to pre-trained checkpoints
cfg.ckpt_dir = '/kaggle/input/datasets/godwin10iitk/oveseg-output/ovseg_multi_benchmark/checkpoints'

# Load the three frozen foundation models
grounding_model, grounding_processor = load_grounding_dino_model(cfg)
segmenter_model, segmenter_predictor, segmenter_name = load_segmenter_predictor(cfg)
clip_model, clip_preprocess, clip_tokenizer = load_clip_model(cfg)
```

Expected output:
```
DEVICE = cuda
GPU = Tesla T4
[SAM2] Loaded sam2.1_hiera_large
Loaded CLIP ViT-B-32 laion2b_s34b_b79k
```

### Step 5: Build scorer, box proposer, proposal cache

```python
scorer = CLIPRegionScorer(clip_model, clip_preprocess, clip_tokenizer)
box_proposer = GroundingDINOBoxProposer(grounding_model, grounding_processor, cfg)
proposal_cache = ProposalCache(
    box_proposer, segmenter_predictor,
    cfg.proposal_cache_dir,
    f'{cfg.detector_tag}_{cfg.segmenter_tag}',
    segmenter_name
)
```

### Step 6: Load (or train) the Alignment MLP

```python
clip_dim = scorer.encode_text('object').numel()  # = 512
alignment_model = ConservativeGroundedReranker(
    clip_dim=clip_dim, aux_dim=8,
    hidden_dim=cfg.align_hidden_dim,   # 192
    dropout=cfg.align_dropout          # 0.10
).to(DEVICE)

ckpt_path = os.path.join(cfg.ckpt_dir, f'alignment_reranker_{cfg.experiment_tag}.pt')
state = torch.load(ckpt_path, map_location=DEVICE)
alignment_model.load_state_dict(state['model_state'])
alignment_model.eval()
print('Loaded alignment model from', ckpt_path)
```

**If training from scratch instead:**
```python
# First build all datasets
datasets = build_all_datasets(cfg)
train_samples = list(datasets['coco_train']) + [datasets['PASCAL-VOC-train'][i] for i in range(len(datasets['PASCAL-VOC-train']))]

# Build alignment table 
align_path = os.path.join(cfg.table_cache_dir, f'alignment_train_{cfg.experiment_tag}.parquet')
alignment_df = build_alignment_table(train_samples, proposal_cache, scorer, cfg, save_path=align_path)

# Train MLP (20 epochs, ~30 min)
alignment_model = train_alignment_model(alignment_df, scorer, cfg)
```

### Step 7: Load (or train) the Tree Policy

```python
ap_ckpt = os.path.join(cfg.ckpt_dir, f'action_policy_tree_{cfg.experiment_tag}.joblib')
action_policy_model = joblib.load(ap_ckpt)
print('Loaded action policy from', ap_ckpt)
```

**If training from scratch:**
```python
proto_pipeline = FullSegmentationPipeline(cfg, proposal_cache, scorer, alignment_model)
ap_path = os.path.join(cfg.table_cache_dir, f'action_policy_ovseg_multi_benchmark_v1.parquet')
action_policy_table = build_action_policy_table(train_samples, proto_pipeline, cfg, save_path=ap_path)
action_policy_model = train_action_policy(action_policy_table, cfg)
```

Expected output:
```
[action-policy] gate_acc=0.9077 chosen_acc=0.7438 oracle_capture=0.5185
```

### Step 8: Build the Iterative Refiner

```python
point_refiner    = SAM2PointRefiner(segmenter_predictor)
boundary_refiner = MaskBoundaryRefiner()
uncertainty_est  = MaskUncertaintyEstimator()
iterative_refiner = IterativeMaskRefiner(
    point_refiner, boundary_refiner, uncertainty_est, scorer, cfg
)
```

### Step 9: Assemble the full pipeline

```python
pipeline = FullSegmentationPipeline(
    cfg=cfg,
    proposal_cache=proposal_cache,
    scorer=scorer,
    reranker=alignment_model,
    action_policy_model=action_policy_model,
    iterative_refiner=iterative_refiner,
)
print('Pipeline ready!')
```

### Step 10: Quick inference test on a single image

```python
image_rgb = np.array(Image.open('/path/to/your/image.jpg').convert('RGB'))
sample = {
    'image_rgb': image_rgb,
    'prompt':    'dog',       # any text prompt
    'sample_id': 'my_test',
    'image_id':  99999,
}

result = pipeline.run(sample, mode='full_pipeline')

# Result keys:
# result['selected_mask']       -> np.uint8 H×W binary mask
# result['clip_score']          -> float CLIP cosine similarity
# result['n_refine_iters']      -> int number of refinement iterations used
# result['chosen_action']       -> str which policy action was taken
# result['used_alignment']      -> 0 or 1, whether alignment MLP fired

# Visualize
overlay = overlay_mask(image_rgb, result['selected_mask'], alpha=0.5, color=(255, 80, 0))
Image.fromarray(overlay).save('result.png')
```

### Step 11: Run the full ablation study (all 7 modes × 5 datasets)

```python
# Load all evaluation datasets
datasets = build_all_datasets(cfg)
eval_splits = {}
for key in ['COCO-Object-val', 'COCO-Stuff-val', 'PASCAL-VOC-val',
            'PASCAL-Context59-val', 'ADE20k-val']:
    if key in datasets and len(datasets[key]) > 0:
        ds = datasets[key]
        eval_splits[key] = [ds[i] for i in range(len(ds))]

# Run all 7 modes 
ablation_summary = run_full_ablation(eval_splits, pipeline, cfg)
print(ablation_summary[['dataset','mode','mIoU','mean_dice','mean_boundary_iou']].to_string())
```

### Step 12: Generate all visualizations

```python
vis = cfg.vis_dir

# Bar charts (one per dataset)
for metric in ['mIoU', 'mean_boundary_iou', 'mean_f1']:
    plot_ablation_bar(ablation_summary, metric=metric,
                      save_path=os.path.join(vis, f'ablation_{metric}_{cfg.experiment_tag}.png'))

# Multi-metric heatmap
plot_multi_metric_heatmap(ablation_summary,
                           save_path=os.path.join(vis, f'heatmap_{cfg.experiment_tag}.png'))

# Component waterfall
plot_component_ablation(ablation_summary,
                         save_path=os.path.join(vis, f'component_ablation_{cfg.experiment_tag}.png'))

# Radar chart
plot_radar_chart(ablation_summary,
                 compare_modes=['baseline','alignment','policy_action1','full_pipeline'],
                 save_path=os.path.join(vis, f'radar_{cfg.experiment_tag}.png'))
```

### Step 13: Run single-image demo (the final demo cells)

The last section of the code runs two demo images:

```python
# Demo 1: man segmentation on COCO test image
IMAGE_PATH = '/kaggle/input/.../000000000063.jpg'
PROMPT = 'man'

# Demo 2: wash basin on ADE20k training image
IMAGE_PATH = '/kaggle/input/.../ADE_train_00000029.jpg'
PROMPT = 'wash basin'
```

These run all 6 modes (excluding oracle) and save `all_modes.png` showing side-by-side comparisons.

---

## 8. Configuration Reference

All hyperparameters live in the `CFG` dataclass. The most important ones:

### Dataset limits
```python
coco_train_limit       = 5000   # samples used to build alignment table
coco_val_limit         = 800    # COCO-Object evaluation samples
coco_stuff_val_limit   = 300
voc_val_limit          = 500
pascal_context_val_limit = 200
ade20k_val_limit       = 200
```

### Grounding DINO
```python
gdino_box_threshold    = 0.25   # minimum detection confidence
gdino_nms_threshold    = 0.50   # NMS IoU threshold for deduplication
gdino_box_top_k        = 8      # maximum boxes retained per prompt
gdino_prompt_variant_limit = 6  # max number of prompt variants in query
```

### Alignment MLP
```python
align_hidden_dim       = 192    # hidden layer size
align_epochs           = 20     # training epochs
align_lr               = 7.5e-4 # AdamW learning rate
align_apply_clip_margin_thresh = 0.02   # CLIP top-2 margin below which alignment fires
align_apply_gdino_score_thresh = 0.25   # GDINO confidence below which alignment fires
```

### Tree Policy
```python
action_policy_tree_max_depth      = 6
action_policy_tree_max_iter       = 500
action_policy_tree_min_samples_leaf = 8
action_policy_gate_infer_threshold = 0.72   # minimum gate probability to trigger a switch
action_policy_switch_threshold    = 0.02    # rank score margin required to switch
```

### Iterative Refinement
```python
refine_max_iters       = 3      # maximum refinement iterations
refine_uncertainty_thresh = 0.06  # CLIP score margin below which refinement triggers
refine_fg_points       = 5      # foreground re-prompt points
refine_bg_points       = 3      # background re-prompt points
refine_morph_close_k   = 5      # morphological close kernel size
refine_morph_open_k    = 3      # morphological open kernel size
refine_crf_sxy         = 80.0   # CRF spatial sigma
refine_crf_srgb        = 13.0   # CRF colour sigma
refine_clip_accept_margin = -0.002  # CLIP delta must exceed this to accept refined mask
```

### Force-rebuild flags (set to True to bypass caches)
```python
force_rebuild_alignment_table      = False
force_retrain_alignment            = False
force_rebuild_action_policy_table  = False
force_retrain_action_policy        = False  # set True to retrain policy
force_rerefine                     = False  # set True to ignore refinement cache
```

---

## 9. Understanding Each Ablation Mode

When you call `pipeline.run(sample, mode=...)`:

| Mode | What happens |
|---|---|
| `'baseline'` | Returns CLIP top-1 proposal. No alignment, no policy, no refinement. |
| `'alignment'` | Applies the Conservative Alignment MLP when ≥2 uncertainty signals detected. |
| `'oracle_action1'` | **Requires `gt_mask`**. Picks the best available proposal by ground-truth IoU. This is the theoretical ceiling. |
| `'policy_action1'` | Runs the full alignment + Tree Policy to select the best action. No refinement. |
| `'refine1'` | Runs policy selection, then applies ONE iteration of SAM2 re-prompting + CRF. |
| `'refine_iter'` | Runs policy selection, then up to 3 iterations of refinement. |
| `'full_pipeline'` | Complete system: alignment + policy + iterative refinement with all gates. |

---

## 10. Output Files and Visualizations

### 10.1 Per-mode per-dataset CSV files

Located at `cfg.eval_cache_dir`. One file per (dataset, mode) combination.

Columns: `sample_index, prompt, mode, dataset, iou, dice, boundary_iou, precision, recall, f1, initial_iou, delta_iou, runtime_sec, clip_score, gdino_score, gdino_box_rank, used_alignment, chosen_action, n_refine_iters, refinement_clip_gain, refinement_iou_gain`

### 10.2 Master ablation summary CSV

`ablation_summary_ovseg_multi_benchmark_v1.csv`

Aggregated per (dataset, mode): `mIoU, mean_dice, mean_boundary_iou, mean_precision, mean_recall, mean_f1, mean_delta_iou, mean_runtime_sec, mean_refine_iters, mean_refine_iou_gain, used_alignment_rate, n_samples`

### 10.3 Visualization files

| File | What it shows |
|---|---|
| `ablation_mIoU_*.png` | Bar charts of mIoU for all 7 modes, one subplot per dataset |
| `ablation_mean_boundary_iou_*.png` | Same but for Boundary-IoU metric |
| `ablation_mean_f1_*.png` | Same but for F1 metric |
| `heatmap_*.png` | 4-panel heatmap: mIoU × mDice × mBoundary-IoU × mF1 for all methods × datasets |
| `component_ablation_*.png` | Left: absolute mIoU per stage; Right: delta mIoU (waterfall chart) |
| `radar_*.png` | Radar/spider chart comparing methods across dataset axes |
| `per_class_voc_*.png` | Per-class mIoU on PASCAL VOC: Baseline vs Full Pipeline |
| `refinement_analysis_*.png` | Scatter plot and histogram of refinement IoU gain |
| `{dataset}_ablation_{idx:03d}.png` | Per-sample: Image + GT + all 6 mode outputs side-by-side |
| `{dataset}_refine_{idx:03d}.png` | Per-sample: Input + Refined mask + CLIP/IoU score progression curves |

---


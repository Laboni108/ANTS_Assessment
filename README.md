# ANTS_Assessment
# Drone Human & Car Detection System

A computer vision pipeline for detecting humans and cars in drone/aerial imagery using YOLOv8, fine-tuned on the VisDrone dataset.

---

## Table of Contents

- [Project Overview](#-project-overview)
- [Setup & Installation](#-setup--installation)
- [Task 01 – Dataset Understanding & Preprocessing](#-task-01--dataset-understanding--preprocessing)
- [Task 02 – Model Training](#-task-02--model-training)
- [Task 03 – Detection & Human Counting](#-task-03--detection--human-counting)
- [Task 04 – Evaluation & Visualization](#-task-05--evaluation--visualization)
- [Results](#-results)
- [Strengths & Limitations](#-strengths--limitations)
- [Project Structure](#-project-structure)

---

## Project Overview

This project builds an end-to-end aerial object detection pipeline that:

- **Detects** humans (`pedestrian`) and cars (`car`) from drone footage
- **Counts** total humans visible in each frame/image
- **Visualizes** detections with labeled bounding boxes and an on-screen count overlay

**Model:** YOLOv8 (Ultralytics), fine-tuned on VisDrone2019-DET  
**Platform:** Kaggle Notebooks (GPU T4 x2)  
**Classes detected:** `pedestrian (0)`, `people (1)`, `car (3)`

----


## ⚙️ Setup & Installation

### Local

```bash
git clone https://github.com/Laboni108/ANTS_Assessment.git
cd ANTS_Assessment

pip install ultralytics opencv-python matplotlib pandas PyYAML
```

### Kaggle Notebook

```python
# All dependencies are pre-installed on Kaggle
!pip install ultralytics -q
```

---

## Task 01 – Dataset Understanding & Preprocessing

### Dataset Structure

```
VisDrone_Dataset/
├── VisDrone2019-DET-train/
│   ├── images/          # .jpg aerial images
│   └── annotations/     # .txt files (VisDrone format)
├── VisDrone2019-DET-val/
│   ├── images/
│   └── annotations/
└── VisDrone2019-DET-test-dev/
    └── images/
```
**VisDrone2019-DET** — a large-scale drone-captured object detection benchmark.

| Split | Images | Annotations |
|-------|--------|-------------|
| Train | 6,471  | ~390,000    |
| Val   | 548    | ~33,000     |
| Test  | 1,580  | —           |

**Download:** [Kaggle – VisDrone Dataset](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset)
**Categories** — Total 10 categories:

| ID | Category |
|----|----------|
| 0 | pedestrian |
| 1 | people |
| 2 | bicycle |
| 3 | car |
| 4 | van |
| 5 | truck |
| 6 | tricycle |
| 7 | awning-tricycle |
| 8 | bus |
| 9 | motor |
---

**VisDrone Annotation Format:**
```
<bbox_left>, <bbox_top>, <bbox_width>, <bbox_height>, <score>, <category>, <truncation>, <occlusion>
```

### Preprocessing

No manual preprocessing or augmentation were done.
The raw VisDrone images were used directly with Ultralytics' built-in training pipeline.

The only configuration step was writing a `fixed_data.yaml` file with `nc: 10`
(all 10 original VisDrone classes) to prevent YOLO framework index mismatches,
while filtering to only train on classes `[0, 1, 3]` via the `classes=` argument.

### Key Challenges

| Challenge | Detail |
|-----------|--------|
| **Tiny objects** | People in drone images are very small — sometimes just a few pixels — which makes them hard to detect |
| **Crowded scenes** | When people stand close together, their boxes overlap and some get missed |
| **Class imbalance** | There are more cars than people in many images, so the model sees fewer human examples |
| **Lighting changes** | Some images are dark or shadowy, which lowers detection confidence |
| **Class index issue** | VisDrone has 10 classes — I had to keep all 10 in the config file even though I only trained on 3, otherwise YOLO would throw an index error |

> Sample visualizations are in `/outputs/sample_viz/`
---

## Model Training

### Approach

Fine-tuned **YOLOv8m** (medium) starting from COCO pretrained weights on the VisDrone training split.

```python
from ultralytics import YOLO

model = YOLO('yolov8m.pt')  # pretrained backbone

model.train(
    data='visdrone.yaml',
    epochs=100,
    imgsz=860,          # larger resolution for tiny objects
    batch=8,
    lr0=0.001,
    mosaic=1.0,
    close_mosaic=15,
    optimizer='AdamW',
    patience=20,
    project='heavy_training',
    name='final_stable_run'
)
```

### Why YOLOv8m?

- Strong balance of speed vs. accuracy
- Native support for small object detection at higher `imgsz`
- Easy integration with VisDrone-formatted data via Ultralytics
- Pretrained COCO weights give a strong starting point for `car` and `person` categories

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `yolov8m.pt` (COCO pretrained) |
| Image size | `1280 × 1280` |
| Epochs | 80 |
| Batch size | 8 |
| Optimizer | AdamW |
| LR | 0.001 |
| Mosaic | Enabled |
| Platform | Kaggle GPU (T4 x2) |

> Training curves (loss, mAP) are saved in `/outputs/training_plots/`

---

## 🔎 Task 03 – Detection & Human Counting

### Inference

```python
from ultralytics import YOLO

model = YOLO('/kaggle/working/heavy_training/final_stable_run/weights/last.pt')

results = model.predict(
    source='/kaggle/input/.../VisDrone2019-DET-val/images',
    save=True,
    conf=0.25,
    classes=[0, 1, 3]   # pedestrian, people, car
)
```

### Human Counting Logic

```python
for result in results:
    boxes = result.boxes
    # classes 0 (pedestrian) and 1 (people) both count as humans
    human_count = sum(1 for cls in boxes.cls.tolist() if int(cls) in [0, 1])
    car_count   = sum(1 for cls in boxes.cls.tolist() if int(cls) == 3)
    print(f"Humans: {human_count} | Cars: {car_count}")
```

### Visualization

Each output image includes:
- **Green bounding boxes** → Humans (pedestrian / people)
- **Blue bounding boxes** → Cars
- **Top-left overlay** → Total human count and car count
- Confidence score on each box label

> Processed images are saved to `/outputs/predictions/`

---

## 🎯 Task 04 – Object Tracking (Bonus)

Object tracking was implemented using **ByteTrack** (built into Ultralytics).

```python
model = YOLO('/kaggle/working/heavy_training/final_stable_run/weights/last.pt')

results = model.track(
    source='drone_clip.mp4',
    tracker='bytetrack.yaml',
    conf=0.25,
    classes=[0, 1, 3],
    save=True,
    persist=True
)
```

ByteTrack was chosen for:
- High recall even for low-confidence detections (uses both high and low threshold boxes)
- No appearance features needed (runs fast without a Re-ID model)
- Native integration into Ultralytics `model.track()`

Each tracked object is assigned a **persistent ID** that follows it across frames, and IDs are displayed on bounding boxes.

> Tracking output video is in `/outputs/tracking/`

---

## 📈 Task 05 – Evaluation & Visualization

### Quantitative Results (Val Split)

| Metric | pedestrian | people | car | **All** |
|--------|-----------|--------|-----|---------|
| Precision | ~0.52 | ~0.48 | ~0.71 | ~0.57 |
| Recall    | ~0.45 | ~0.41 | ~0.65 | ~0.50 |
| mAP@0.5   | ~0.43 | ~0.39 | ~0.62 | ~0.48 |

> *Note: Exact figures may vary — see `/outputs/metrics.csv` for final logged values.*

### Counting Accuracy

Human counting operates by summing class `0` and class `1` detections per image. In dense scenes, this closely matches ground truth counts, with some undercounting in heavily occluded crowds.

### Output Samples

| Input | Output |
|-------|--------|
| Aerial crowd scene | Boxes + count overlay |
| Road intersection | Cars detected, pedestrians labeled |
| Mixed scene | Both classes, human count = N |

> Full prediction gallery: `/outputs/predictions/`

---

## ✅ Strengths & Limitations

### Strengths

- **End-to-end pipeline** from raw VisDrone data to annotated inference outputs
- **High-resolution inference** (1280px) preserves tiny object detail
- **Multi-class aware counting** merges pedestrian + people into a single human count
- **ByteTrack integration** gives stable IDs without expensive Re-ID overhead
- **Modular code** — detection, counting, and visualization are decoupled

### Limitations

- **Tiny objects remain hard** — humans under ~10px are frequently missed at `conf=0.25`
- **Dense crowd undercounting** — heavily overlapping detections cause merged boxes
- **No temporal smoothing** for counting in video (count fluctuates frame-to-frame)
- **Single model** — a SAHI (Slicing Aided Hyper Inference) approach could further boost small-object recall
- **Training compute** limited to Kaggle free tier (capped epochs/time)

### Challenges Faced

- Converting VisDrone's custom annotation format to YOLO `.txt` format correctly (especially handling `score = 0` ignore regions)
- Managing GPU memory at `imgsz=1280` with batch size — required batch=8 tuning
- Class confusion between `pedestrian` and `people` labels (both are counted as human, reducing per-class mAP)

---

## 📁 Project Structure

```
drone-detection-antlings/
├── notebooks/
│   ├── model_training_visualization.ipynb
│   
├── src/
│   ├── convert_annotations.py    # VisDrone → YOLO format
│   ├── detect_and_count.py       # Inference + human counting
│   └── visualize.py              # Overlay bounding boxes & count
├── configs/
│   └── .yaml             # Dataset config for Ultralytics
├── outputs/
│   ├── predictions/              # Annotated output images
│   ├── tracking/                 # Tracking output video
│   ├── training_plots/           # Loss & mAP curves
│   └── sample_viz/               # Dataset sample visualizations
├── weights/
│   └── last.pt                   # Final trained model weights
└── README.md
```

---

## 🔗 References

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [Kaggle Dataset Source](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset)

---

*Built for the Antlings Internship Program – AI/ML Technical Assessment, May 2026.*

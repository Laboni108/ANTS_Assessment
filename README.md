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


## Setup & Installation

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


---

## Model Training

### Approach

Fine-tuned **YOLOv8m** (medium) starting from COCO pretrained weights on the VisDrone training split.

```python
from ultralytics import YOLO


model = YOLO('yolov8s.pt')


model.train(
    data='/kaggle/working/fixed_data.yaml',
    epochs=100,
    imgsz=960,            
    batch=4,              
    classes=[0, 1, 3],     # trains ONLY on Pedestrians(0), People(1), and Cars(3)
    project='/kaggle/working/heavy_training',
    name='final_stable_run',
    device=0,        
    save_period=10,
    exist_ok=True,
    amp=True,              
    cache=False            
)
```

### Why YOLOv8m?

YOLO (You Only Look Once) is a fast and accurate object detection model
We used the small (s) version because Kaggle's free GPU has limited memory — bigger models like yolov8m or yolov8l would run out of memory at our batch size
It comes pretrained on COCO, meaning it already knows what humans and cars look like, so we just fine-tuned it on VisDrone instead of training from scratch

### How it works?
The image is divided into a grid
Each grid cell predicts bounding boxes and class labels
It does this in one single pass through the network — that's why it's fast
We started from pretrained weights and trained further on VisDrone so it could handle aerial/drone-view images specifically

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | `yolov8m.pt` (COCO pretrained) |
| Image size | `860 × 860` |
| Epochs | 100 |
| Batch size | 4 |
| Platform | Kaggle GPU (T4 x2) |


---

## Task 03 – Detection & Human Counting

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
 is_human = "pedestrian" in class_name or "people" in class_name or "person" in class_name
        is_car = "car" in class_name or "van" in class_name or "truck" in class_name
        
        if not (is_human or is_car):
            continue  
            
        if is_human:
            human_count += 1
            color = COLOR_HUMAN
            label_prefix = "Human"
        else:
            car_count += 1
            color = COLOR_CAR
            label_prefix = "Car"
```

For each detected object in the image, it checks the class label name:

- If the label contains **"pedestrian"**, **"people"**, or **"person"** → counted as a **Human**
- If the label contains **"car"**, **"van"**, or **"truck"** → counted as a **Car**
- Anything else is **skipped**

Based on that:
- Human detections get a **blue bounding box** and increase the human counter by 1
- Car detections get a **cyan bounding box** and increase the car counter by 1

At the end, the total human count and car count are displayed on a banner at the top of the image.
### Visualization

Each output image includes:
- **Blue bounding boxes** → Humans (pedestrian / people)
- **Cyan bounding boxes** → Cars
- **Top-left overlay** → Total human count and car count
- Confidence score on each box label


---

## Evaluation & Visualization

### Quantitative Results ( Combined avg for car and human )

| Metric | pedestrian | 
|--------|-----------|
| Precision | ~0.726 | 
| Recall    | ~0.601 | 
| mAP@0.5   | ~0.646 | 
| mAP@0.5:0.95   | ~0.366 |


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

## Strengths & Limitations

## Strengths

- **Simple and complete pipeline** — It went straight from the raw VisDrone dataset to final detection outputs with minimal setup
- **Higher resolution training** (`imgsz=960`) helps the model see small objects like distant pedestrians more clearly
- **Accurate human counting** — both "pedestrian" and "people" labels are correctly merged into a single human count
- **Clean visualization** — color coded bounding boxes and a count banner make results easy to read at a glance
- **Mixed precision training (`amp=True`)** — made training faster and used less GPU memory on Kaggle

## Limitations

- **Small objects are still hard to detect** — people that are only a few pixels tall are often missed entirely
- **Crowded scenes cause undercounting** — when people are packed together, the model sometimes draws one big box instead of individual ones
- **No tracking** — It only detects objects per image, there is no ID assigned to track the same person across video frames
- **Small model (`yolov8s`)** —  The small version of YOLOv8 is used due to Kaggle GPU memory limits, a larger model would likely perform better
- **Kaggle free tier constraints** — limited GPU time and memory meant it could not be trained longer or experiment with bigger models

### Challenges Faced

- Converting VisDrone's custom annotation format to YOLO `.txt` format correctly (especially handling `score = 0` ignore regions)
- Managing GPU memory at `imgsz=860` with batch size — required batch=4 tuning
- Class confusion between `pedestrian` and `people` labels (both are counted as human, reducing per-class mAP)
- Last but not least I have my term final going on, so I have 
---



## References

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [Kaggle Dataset Source](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset)

---

*Built for the Antlings Internship Program – AI/ML Technical Assessment, May 2026.*

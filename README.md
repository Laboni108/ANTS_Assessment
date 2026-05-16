
# ANTS_Assessment
# Drone Human & Car Detection System

A computer vision pipeline for detecting humans and cars in drone/aerial imagery using YOLOv8, fine-tuned on the VisDrone dataset.


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

```

### Kaggle Notebook

```python
!pip install -U ultralytics --extra-index-url https://download.pytorch.org/whl/cu118

import os
from ultralytics import YOLO

print(" Environment ready and Ultralytics imported successfully!")
```

---

## Dataset Understanding & Preprocessing

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
## Dataset Configuration and Runtime Preprocessing 

```python

import yaml
import os

# Define the precise dataset configurations mapping to VisDrone structure
data_config = {
    'path': '/kaggle/input/datasets/banuprasadb/visdrone-dataset/VisDrone_Dataset', 
    'train': 'VisDrone2019-DET-train/images', 
    'val': 'VisDrone2019-DET-val/images',     
    'nc': 10, # Keep original indexing bounds intact to prevent YOLO framework mismatches
    'names': {
        0: 'pedestrian', 
        1: 'people', 
        2: 'bicycle', 
        3: 'car', 
        4: 'van',
        5: 'truck', 
        6: 'tricycle', 
        7: 'awning-tricycle', 
        8: 'bus', 
        9: 'motor'
    }
}

with open('/kaggle/working/fixed_data.yaml', 'w') as f:
    yaml.dump(data_config, f)

print(" Fixed YAML generated at /kaggle/working/fixed_data.yaml")
```

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

## Detection & Human Counting

### Outputs

```python
 cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
            label_text = f"{label_prefix} {conf:.2f}"
            (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(img, (xyxy[0], xyxy[1] - text_h - 4), (xyxy[0] + text_w + 4, xyxy[1]), color, -1)
            cv2.putText(img, label_text, (xyxy[0] + 2, xyxy[1] - 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_TEXT, 1, cv2.LINE_AA)

        # Create Dashboard Overlay Banner at the top of the image
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 65), (0, 0, 0), -1)  
        img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)         
        
        # Write counting tallies onto the banner
        cv2.putText(img, f"TOTAL HUMANS COUNTED: {human_count}", (20, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_TEXT, 2, cv2.LINE_AA)
        cv2.putText(img, f"Cars: {car_count}", (w - 180, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_CAR, 2, cv2.LINE_AA)

        # Convert matrix array from BGR to RGB format for plotting
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Present the output frame layout with the tracking header
        plt.figure(figsize=(16, 10))
        plt.imshow(img_rgb)
        plt.title(f"Custom Counting Dashboard [{idx}/10]: {target_file_name}", fontsize=12, pad=10)
        plt.axis('off')
        plt.show()

# Run the pipeline for the next 10 unplotted images
predict_another_ten_unseen_images(
    dataset_dir=TEST_DATASET_DIR,
    model_path=MODEL_PATH
)
)
```

### Human Counting Logic

```python
   is_human = "pedestrian" in class_name or "people" in class_name or "person" in class_name
            is_car = class_name == "car"
            
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

<img width="880" height="680" alt="Validation_Statistics" src="https://github.com/user-attachments/assets/b0c5cf08-651e-486c-ba81-2d352974b5cf" />


### Output Samples

<img width="885" height="654" alt="1" src="https://github.com/user-attachments/assets/2ba0a5a3-637c-4784-b6ce-4bff5d1b772d" />

<img width="885" height="657" alt="14" src="https://github.com/user-attachments/assets/914db509-aced-4934-91a5-1cd0c81cb24b" />



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

 - **GPU memory management** — running at `imgsz=960` with Kaggle's free GPU was tight,
  so i had to keep the batch size at 4 to avoid running out of memory
- **Pedestrian vs People labels** — VisDrone has two separate labels for humans
  ("pedestrian" and "people"), we intentionally merged both into one human count
  which is correct for counting purposes but slightly lowers the per-class mAP score
  since the model treats them as two different classes internally
- **Limited time** — last but not least this project was built under tight time as my university
  term finals have been going on, so there was limited room for deeper experimentation
  and fine-tuning

  
## Drive link 
### [Click Here to Open Drive to See Video](https://drive.google.com/file/d/1pGVFpwYMbJa3O21QwUAjgUZHgdcYh_YT/view?usp=sharing)

## References

- [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [Kaggle Dataset Source](https://www.kaggle.com/datasets/banuprasadb/visdrone-dataset)

---

*Built for the Antlings Internship Program – AI/ML Technical Assessment, May 2026.*

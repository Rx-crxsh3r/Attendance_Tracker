# Project: YOLO Classroom Attendance Tracker

## Brief Description
This project implements a classroom attendance tracking system using YOLO (You Only Look Once) object detection. The system detects students in images or live camera feeds and records attendance data, which can be visualized and exported for further analysis.

## Table of Contents
- [Dataset Source](#dataset-source)
- [Training the Model](#training-the-model)
- [Model Visualization](#model-visualization)
- [Fine-Tuning the Model](#fine-tuning-the-model)
- [Displaying Results](#displaying-results)

## Dataset Source
The dataset used for training the YOLO model is sourced from Roboflow. To obtain the dataset, follow these steps:

1. Visit the [Roboflow Dataset Page](https://universe.roboflow.com/logo-bplam/test-dxiix-ckery/dataset/1).
2. Click on the "Download" button.
3. Download the dataset as a code snippet or directly download the file.

## Training the Model
To train the YOLO model using the dataset, run the following command:

```bash
!yolo train model=yolov8n.pt data='/content/test-1/data.yaml' epochs=50 imgsz=640 batch=10 momentum=0.9 weight_decay=0.0005 warmup_epochs=1 lr0=0.001
```
  
### Explanation of Parameters
- `model=yolov8n.pt`: Specifies the pre-trained YOLOv8n model.
- `data='/content/test-1/data.yaml'`: Path to the dataset configuration file.
- `epochs=50`: Number of training epochs (full passes through the dataset).
- `imgsz=640`: Image size for training (640x640 pixels).
- `batch=10`: Batch size for training.
- `momentum=0.9`: Momentum value for optimization.
- `weight_decay=0.0005`: Weight decay value for regularization.
- `warmup_epochs=1`: Number of warmup epochs for gradual learning rate increase.
- `lr0=0.001`: Initial learning rate.

If the mAP (mean Average Precision) score is more than 0.7, the model is considered to have good performance.

## Model Visualization
You can visualize the model's performance using curve graphs (R, P, PR, or F1). Here's an example code snippet to display these graphs:

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

f1_curve = mpimg.imread('/content/runs/detect/train2/F1_curve.png')
pr_curve = mpimg.imread('/content/runs/detect/train2/PR_curve.png')
p_curve = mpimg.imread('/content/runs/detect/train2/P_curve.png')
r_curve = mpimg.imread('/content/runs/detect/train2/R_curve.png')

plt.figure(figsize=(15, 10))

# Display F1 Curve
plt.subplot(2, 2, 1)
plt.imshow(f1_curve)
plt.title('F1 Curve')
plt.axis('off')

# Display PR Curve
plt.subplot(2, 2, 2)
plt.imshow(pr_curve)
plt.title('PR Curve')
plt.axis('off')

# Display P Curve
plt.subplot(2, 2, 3)
plt.imshow(p_curve)
plt.title('P Curve')
plt.axis('off')

# Display R Curve
plt.subplot(2, 2, 4)
plt.imshow(r_curve)
plt.title('R Curve')
plt.axis('off')

plt.show()
```

## Fine-Tuning the Model
After training the initial model, you can further fine-tune it using the following commands:

### Fine-Tuning with Pre-trained Weights
```bash
python train.py --weights runs/detect/train4/weights/best.pt --data new_data.yaml --epochs 20 --img 640
```

### Fine-Tuning with Partial Layers
```bash
!python train.py --weights runs/detect/train4/weights/best.pt --data /new_data.yaml --epochs 20 --img 640 --freeze 10
```

### Layer Freezing in YOLO
- **Early Layers**: Handle general feature extraction (e.g., edges, textures).
- **Middle Layers**: Extract more complex features (e.g., shapes, patterns).
- **Later Layers**: Focus on task-specific features (e.g., detecting logos).

### Displaying Model Architecture
```python
from ultralytics import YOLO

model = YOLO('runs/detect/train4/weights/best.pt')
print(model.model)
```

## Displaying Results
You can display the validation results using the following code:

```python
validation_batch2_result = mpimg.imread('/content/runs/detect/train2/val_batch1_pred.jpg')
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.imshow(validation_batch2_result)
plt.title('Validation result batch 2')
plt.axis('off')
plt.show()
```

## Disclaimer
This project was originally referenced by a Google Developers Workshop. The code and features have since been modified and updated by the current maintainers. 
```

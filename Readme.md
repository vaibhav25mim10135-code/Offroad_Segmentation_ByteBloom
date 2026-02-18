# Duality AI – Offroad Semantic Scene Segmentation

Pixel-wise semantic segmentation for off-road environments using a **frozen DINOv2 Vision Transformer backbone** and a **lightweight ConvNeXt-style segmentation head**.

---

## ✨ Features

- Pixel-wise segmentation of off-road terrain images
- **10 terrain classes**:
  - Background
  - Trees
  - Lush Bushes
  - Dry Grass
  - Dry Bushes
  - Ground Clutter
  - Logs
  - Rocks
  - Landscape
  - Sky
- Frozen **DINOv2 ViT-S/14** backbone for feature extraction
- Lightweight **ConvNeXt-style decoder head**
- Training augmentations:
  - Random flips
  - Rotation
  - Color jitter *(images only)*
- Evaluation metrics:
  - Mean IoU (mIoU)
  - Dice Score
  - Pixel Accuracy
- Visualization:
  - Raw masks
  - Colored masks
  - Side-by-side comparisons

---

## 🧰 Prerequisites

- Python **3.12.4**
- PyTorch **2.1.0+cpu**
- Torchvision **0.15+**
- CUDA **11+** *(optional)*

### Required Packages
```bash
pip install torch torchvision numpy matplotlib opencv-python pillow tqdm
```

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd Offroad_Segmentation
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
```

**macOS / Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\\Scripts\\activate
```

### 3. Install Dependencies
```bash
pip install torch torchvision matplotlib tqdm opencv-python pillow
```

---

## 📁 Dataset Structure

```text
Offroad_Segmentation_Training_Dataset/
├─ train/
│  ├─ images/
│  └─ masks/
└─ val/
   ├─ images/
   └─ masks/
```

- Masks must contain class IDs **0–9**
- Image and mask dimensions must be divisible by **patch size = 14**

---

## 🚀 How to Run

### 1. Training
```bash
python train_mask.py
```

**Default directories**
```text
../Offroad_Segmentation_Training_Dataset/train
../Offroad_Segmentation_Training_Dataset/val
```

**Outputs**
- Model weights → `segmentation_head.pth`
- Training metrics → `train_stats/`

> Augmentation is applied **only during training**.

---

### 2. Validation / Prediction
```bash
python val_mask.py \
  --data_dir ../Offroad_Segmentation_testImages \
  --model_path segmentation_head.pth \
  --output_dir predictions
```

**Generated Outputs**
```text
predictions/
├─ masks/                # Raw predicted masks
├─ masks_color/          # Colored segmentation masks
├─ comparisons/          # Input vs GT vs Prediction
├─ evaluation_metrics.txt
└─ per_class_metrics.png
```

---

## 🧪 Usage Example

```python
import torch
from val_mask import SegmentationHeadConvNeXt, mask_to_color
from PIL import Image
import torchvision.transforms as transforms

# Load trained segmentation head
model = SegmentationHeadConvNeXt(
    in_channels=384,
    out_channels=10,
    tokenW=68,
    tokenH=38
)
model.load_state_dict(torch.load("segmentation_head.pth"))
model.eval()

# Load and preprocess image
img = Image.open("sample_image.png").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((240, 480)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
img_tensor = transform(img).unsqueeze(0)

# Predict mask
with torch.no_grad():
    output = model(img_tensor)
    pred_mask = torch.argmax(output, dim=1).squeeze(0)

# Save colored mask
pred_color = mask_to_color(pred_mask.numpy())
pred_color.save("predicted_mask.png")
```

---

## ⚙️ Technical Details

- **Backbone**: DINOv2 ViT-S/14 *(frozen)*
- **Segmentation Head**: ConvNeXt-style
  - Depthwise + pointwise convolutions
- **Input Resolution**: `480 × 240`
- **Output**: Pixel-wise predictions (10 classes)
- **Loss Function**: Cross-Entropy Loss
- **Optimizer**: SGD (momentum = 0.9)
- **Learning Rates**: `1e-4`, `3e-4`
- **Batch Sizes**: `2`, `4`
- **Epochs**: `25`, `30`

---

## 🔄 Data Augmentation (Training Only)

- Random Horizontal Flip
- Random Vertical Flip
- Random Rotation (−15° to +15°)
- Color Jitter:
  - Brightness
  - Contrast
  - Saturation
  - Hue

---

## 🧩 Key Classes & Functions

### `MaskDataset`
```python
class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=False):
        ...
    def __getitem__(self, idx):
        ...
```

- Handles RGB images and segmentation masks
- Converts raw mask values → class IDs (0–9)
- Applies augmentation only when `is_train=True`

### Evaluation Utilities
- `compute_iou`
- `compute_dice`
- `compute_pixel_accuracy`
- `mask_to_color`
- `save_prediction_comparison`
- `save_metrics_summary`

---

## ⚠️ Important Notes

- Backbone is **fully frozen**
- Only the segmentation head is trained
- Ensure image sizes are divisible by **patch size (14)**
- Validation uses **no augmentation**
- Tune batch size and learning rate based on GPU memory

---

## 🗂️ Code Structure

```text
Offroad_Segmentation/
├─ train_mask.py
├─ val_mask.py
├─ segmentation_head.pth
├─ Offroad_Segmentation_Training_Dataset/
│  ├─ train/
│  └─ val/
├─ predictions/
│  ├─ masks/
│  ├─ masks_color/
│  └─ comparisons/
└─ README.md
```

##Title 
Duality AI's Offroad Semantic Scene Segmentation

## Installation Pakages

1)Clone the repository:

``bash
git clone <your-repo-url>
cd Offroad_Segmentation

``
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

'''

3)Install required packages:

pip install torch torchvision matplotlib tqdm opencv-python pillow

##Features

-Pixel-wise segmentation of off-road terrain images.
-10 terrain classes: Background, Trees, Lush Bushes, Dry Grass, Dry Bushes, Ground Clutter, Logs, Rocks, Landscape, Sky.
-Frozen DINOv2 Vision Transformer backbone for feature extraction.
-Lightweight ConvNeXt-style segmentation head for efficient decoding.
-Training augmentation: random flips, rotation, color jitter (applied to images only).
-Evaluation using mIoU, Dice Score, Pixel Accuracy.
-Generates colored masks and side-by-side comparisons for validation.

##Prerequisites
-Python 3.12.4
-PyTorch 2.10.0+cpu
-Torchvision 0.15+
-CUDA 11+ (optional)
-Packages: numpy, matplotlib, opencv-python, Pillow, tqdm

##How to Run
1.Training
'''bash
python train_mask.py

'''

-Default training/validation directory:

   ../Offroad_Segmentation_Training_Dataset/train

  ../Offroad_Segmentation_Training_Dataset/val

-Augmentation applied only during training.

-Model weights saved as: segmentation_head.pth.

-Training metrics saved in train_stats/.

2. Validation / Prediction
python val_mask.py --data_dir ../Offroad_Segmentation_testImages --model_path segmentation_head.pth --output_dir predictions

-Generates predictions for all images.

-Outputs:
    
   Raw masks → predictions/masks/
   Colored masks → predictions/masks_color/
   Comparison images → predictions/comparisons/
   Metrics → predictions/evaluation_metrics.txt & per_class_metrics.png



## Usage Example

``python
import torch
from val_mask import SegmentationHeadConvNeXt, mask_to_color
from PIL import Image
import torchvision.transforms as transforms

## Load trained segmentation head
model = SegmentationHeadConvNeXt(in_channels=384, out_channels=10, tokenW=68, tokenH=38)
model.load_state_dict(torch.load('segmentation_head.pth'))
model.eval()

## Load and preprocess image
img = Image.open("sample_image.png").convert("RGB")
transform = transforms.Compose([
    transforms.Resize((240, 480)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])
img_tensor = transform(img).unsqueeze(0)

## Predict mask
with torch.no_grad():
    output = model(img_tensor)
    pred_mask = torch.argmax(output, dim=1).squeeze(0)

## Save colored mask
pred_color = mask_to_color(pred_mask.numpy())
pred_color.save("predicted_mask.png")

``
Technical Details

-Backbone: DINOv2 ViT-S/14 (frozen)
-Segmentation Head: ConvNeXt-style (depthwise + pointwise convolutions)
-Input resolution: 480x240 (resized to divisible by patch size 14)
-Output: Pixel-wise class predictions (10 classes)
-Loss Function: Cross-Entropy Loss
-Optimizer: SGD with momentum 0.9
-Learning Rate: 1e-4,3e-4
-Batch Size: 2,4
-Epochs: 25,30

##Data Augmentation (Training Only)

-Random Horizontal Flip
-Random Vertical Flip
-Random Rotation (-15° to 15°)
-Color Jitter (brightness, contrast, saturation, hue)

##Functions Overview

class MaskDataset(Dataset):
    def __init__(self, data_dir, transform=None, is_train=False):
        ...
    def __getitem__(self, idx)...

-Handles RGB images and masks.
-Applies augmentation only if is_train=True.
-Converts raw mask values to class IDs (0–9).

##Evaluation & Visualization
-compute_iou, compute_dice, compute_pixel_accuracy → quantitative metrics
-mask_to_color → convert class masks to RGB
-save_prediction_comparison → input vs ground truth vs predicted mask
-save_metrics_summary → text file and per-class IoU bar chart

##Important Notes
-Backbone is frozen; only segmentation head is fine-tuned.
-Ensure image/mask dimensions are divisible by patch size (14).
-Augmentation applies only during training; validation uses original images.
-Parameters like batch size, learning rate, and epochs can be tuned based on GPU memory and dataset size.

##Code Structure

Offroad_Segmentation/
│
├─ train_mask.py            # Training script
├─ val_mask.py              # Validation / inference script
├─ segmentation_head.pth    # Trained model weights
├─ Offroad_Segmentation_Training_Dataset/
│   ├─ train/
│   └─ val/
├─ predictions/             # Created during validation
│   ├─ masks/
│   ├─ masks_color/
│   └─ comparisons/
└─ README.md

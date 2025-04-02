# Installation

## Requirements

- Linux
- PyTorch 1.11.0
- Python 3.8 (Ubuntu 20.04)
- CUDA 11.3
- RTX3080

### Create and activate a conda virtual environment.

```bash
conda create -n cross python=3.8 -y
conda activate cross
```

### Install PyTorch and torchvision following official instructions.

```bash
conda install pytorch=1.11.0 torchvision cudatoolkit=11.3 -c pytorch -y
```

### Clone the fuse_fusion repository.

```bash
git clone https://github.com/Jane-o-O-o-O/fuse_fusion.git
cd fuse_fusion
```

### Install dependencies (environment configurations may require fine-tuning).

```bash
pip install -r requirements.txt
```

# Getting Started

The main execution files include `train`, `test`, and `detect` for training, metric evaluation, and detection, respectively.

- `train`: Trains the model.
- `test`: Evaluates the precision of the trained model.
- `detect`: Performs detection on images.

### Proposed Dual-Modality Detection Methods:
- `train_fusion`, `test_fusion`, `detect_fusion`: Training, testing, and detection for the proposed dual-modality algorithm.

### Baseline Dual-Modality Detection Methods:
- `train_fusion`, `test_fusion`, `detect_fusion`: Training, testing, and detection for the baseline dual-modality algorithm.

### YOLOv5 Single-Modality Detection Methods:
- `train_origin`, `test_origin`, `detect_origin`: Training, testing, and detection for the YOLOv5 single-modality algorithm.

## Dataset Preparation

You can get the dataset from here：https://drive.google.com/file/d/11PrU-Lq9jme0e8Lmh1SzNy5Uh2vqTk16/view?usp=sharing

Navigate to the `fuse_fusion/fuse_dataset` directory:
- Place visible light images in `images/rgb` and their corresponding `.txt` annotation files in `labels/rgb`.
- Place infrared light images in `images/ir` and their corresponding `.txt` annotation files in `labels/ir`.

Ensure your dataset follows the structure below:

```
fuse_dataset/
├── images/
│   ├── ir/
│   │   ├── test/
│   │   │   └── 1000.png
│   │   ├── train/
│   │   │   └── 1.png
│   │   └── val/
│   │       └── 1002.png
│   └── rgb/
│       ├── test/
│       │   └── 1000.png
│       ├── train/
│       │   └── 1.png
│       └── val/
│           └── 1002.png
└── labels/
    ├── ir/
    │   ├── test/
    │   │   └── 1000.txt
    │   ├── train/
    │   │   └── 1.txt
    │   └── val/
    │       └── 1002.txt
    └── rgb/
        ├── test/
        │   └── 1000.txt
        ├── train/
        │   └── 1.txt
        └── val/
            └── 1002.txt
```

Dataset configuration files are located in `fuse_fusion/data/multispectral`:
- `FLIR_aligned.yaml`: Configuration for dual-modality datasets.
- `rgb.yaml`: Configuration for single-modality datasets. Modify the dataset paths and category names according to your data structure.

## Training the Model (Adjust parameters as needed)

### Update Dataset Paths and Categories
Modify the YAML files in `fuse_fusion/data/multispectral` to reflect your dataset paths and categories.

### Execute Training

```bash
python train_fusion.py --data ./data/multispectral/FLIR_aligned.yaml \
--cfg ./models/yolov5l_fusion_light.yaml --weights yolov5l.pt \
--epochs 100 --batch-size 4
```

Training outputs will be saved in `fuse_fusion/runs/fusion`.

## Testing (Adjust parameters as needed)

Here are the models available for testing：https://drive.google.com/drive/folders/1-nl5_3-8VaBfWElj6Ge_MbWLlqsEh48B?usp=sharing
```bash
python detect_fusion.py --weights ./runs/fusion/exp4/weights/best.pt \
--source1 square_padding/images/rgb/test --source2 square_padding/images/ir/test
```

Detection results will be saved in `fuse_fusion/runs/two_detect`.
```

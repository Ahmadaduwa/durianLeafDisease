# Durian Leaf Disease Classification Project

A comprehensive machine learning project for classifying durian leaf diseases using deep learning models with transfer learning and data augmentation techniques. The project includes model training, evaluation, visualization using GradCAM, and a web application for real-time predictions.

## 🌿 Project Overview

This project focuses on classifying durian leaf diseases into 5 categories:
- **ALGAL_LEAF_SPOT** - Algal leaf spot disease
- **ALLOCARIDARA_ATTACK** - Allocaridara pest attack
- **HEALTHY_LEAF** - Healthy durian leaves
- **LEAF_BLIGHT** - Leaf blight disease
- **PHOMOPSIS_LEAF_SPOT** - Phomopsis leaf spot disease

## 📁 Project Structure

```
durianLeaf/
├── data/
│   ├── dataSets/           # Original dataset
│   ├── train/              # Training data (split from original)
│   ├── test/               # Test data (split from original)
│   └── train_augmented/    # Augmented training data
├── transferLearning/       # Models trained without augmentation
│   ├── denseNet-201/
│   ├── efficientNetV2/
│   ├── mobileNetV3/
│   ├── res2net50/
│   └── resnet50/
├── transferLearning_aug/   # Models trained with augmentation
│   ├── denseNet-201/
│   ├── efficientNetV2/
│   ├── mobileNetV3/
│   ├── res2net50/
│   └── resnet50/
├── web/                    # Web application
│   ├── app.py             # Flask web server
│   ├── index.html         # Web interface
│   └── durian_app.log     # Application logs
├── Augmentation.ipynb      # Data augmentation pipeline
├── SplitData.ipynb        # Dataset splitting utility
├── gradCam.ipynb          # GradCAM visualization
└── output.png             # Sample output image
```

## 🚀 Features

### 1. **Multiple Deep Learning Models**
- **ResNet50** - Residual neural network
- **MobileNetV3** - Lightweight mobile-optimized model
- **DenseNet-201** - Densely connected convolutional network
- **EfficientNetV2** - Efficient scaling of neural networks
- **Res2Net50** - Multi-scale residual network

### 2. **Data Augmentation**
- Random resized crop
- Horizontal flip
- Shift, scale, and rotation
- Noise addition (Gaussian and ISO)
- Brightness and contrast adjustment
- CLAHE (Contrast Limited Adaptive Histogram Equalization)

### 3. **Model Training**
- 5-fold cross-validation
- Transfer learning from pre-trained models
- Both original and augmented data training
- Checkpoint saving for each fold

### 4. **Visualization**
- **GradCAM** (Gradient-weighted Class Activation Mapping)
- Layer-wise activation visualization
- Heatmap generation for model interpretability

### 5. **Web Application**
- Real-time image classification
- Drag-and-drop image upload
- Top-5 predictions with confidence scores
- GradCAM visualization in web interface
- System monitoring (CPU, memory usage)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Required Packages
```bash
pip install torch torchvision
pip install pytorch-grad-cam
pip install albumentations
pip install opencv-python
pip install pillow
pip install matplotlib
pip install numpy
pip install flask
pip install psutil
pip install tqdm
```

## 📊 Usage

### 1. Data Preparation

#### Split Dataset
```python
# Run SplitData.ipynb to split your dataset into train/test sets
# Default test ratio: 13.8%
```

#### Data Augmentation
```python
# Run Augmentation.ipynb to generate augmented training data
# Configurable parameters:
# - TARGET_COUNT_PER_CLASS: Target number of images per class
# - MULTIPLIER: Augmentation multiplier
# - AUG_PER_IMAGE: Number of augmented images per original image
```

### 2. Model Training

Each model directory contains a Jupyter notebook for training:
- `resnet50.ipynb`
- `mobilenetv3_large.ipynb`
- `denseNet201.ipynb`
- `efficientnetv2.ipynb`
- `res2net50.ipynb`

Training features:
- 5-fold cross-validation
- Transfer learning from ImageNet pre-trained weights
- Automatic checkpoint saving
- Training progress monitoring

### 3. Model Evaluation and Visualization

#### GradCAM Visualization
```python
# Run gradCam.ipynb for model interpretability analysis
# Features:
# - Automatic last convolution layer detection
# - Specific layer targeting
# - Batch processing for multiple images
# - Heatmap and overlay generation
```

### 4. Web Application

#### Start the Web Server
```bash
cd web
python app.py
```

#### Access the Application
- Open your browser and navigate to `http://localhost:5000`
- Upload durian leaf images for real-time classification
- View predictions with confidence scores
- Generate GradCAM visualizations

## 🔧 Configuration

### Model Configuration
```python
# In gradCam.ipynb
MODEL_PATH = "./transferLearning_aug/resnet50/resnet50_checkpoint_fold4.pt"
IMAGE_SIZE = (224, 224)
USE_GPU = torch.cuda.is_available()
```

### Web Application Configuration
```python
# In web/app.py
MODEL_PATH = "./../transferLearning_aug/resnet50/resnet50_checkpoint_fold4.pt"
CLASS_NAMES = [
    "ALGAL_LEAF_SPOT",
    "ALLOCARIDARA_ATTACK", 
    "HEALTHY_LEAF",
    "LEAF_BLIGHT",
    "PHOMOPSIS_LEAF_SPOT"
]
IMAGE_SIZE = 224
TOPK = 5
```

## 📈 Model Performance

The project includes multiple model architectures trained with and without data augmentation:

- **Without Augmentation**: Models trained on original dataset
- **With Augmentation**: Models trained on augmented dataset for improved generalization

Each model is evaluated using 5-fold cross-validation with saved checkpoints for each fold.

## 🎯 Key Features

### Data Augmentation Pipeline
- **RandomResizedCrop**: Scale range (0.8, 1.0)
- **HorizontalFlip**: 50% probability
- **ShiftScaleRotate**: Shift (8%), Scale (12%), Rotate (±25°)
- **Noise Addition**: Gaussian and ISO noise
- **Brightness/Contrast**: ±25% adjustment
- **CLAHE**: Adaptive histogram equalization

### GradCAM Implementation
- Automatic target layer detection
- Support for all convolution layers
- Batch processing capabilities
- Multiple visualization modes (heatmap, overlay)

### Web Application Features
- Real-time image classification
- Drag-and-drop interface
- Top-K predictions with confidence scores
- GradCAM visualization integration
- System resource monitoring
- Comprehensive logging

## 📝 Notes

- The project uses ImageNet pre-trained weights for transfer learning
- All models are trained with 224x224 input resolution
- GPU acceleration is automatically detected and used when available
- Comprehensive logging is implemented for debugging and monitoring
- The web application includes error handling and user feedback

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please ensure you have appropriate permissions for any datasets used.

## 🔍 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use CPU mode
2. **Model Loading Errors**: Ensure model paths are correct and files exist
3. **Web App Issues**: Check Flask dependencies and port availability
4. **GradCAM Errors**: Verify pytorch-grad-cam installation

### Performance Tips

- Use GPU acceleration when available
- Adjust NUM_WORKERS based on your system
- Monitor memory usage during training
- Use appropriate image sizes for your hardware

---

For more detailed information, please refer to the individual Jupyter notebooks and the web application documentation.

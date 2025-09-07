# SCML4U - Simple Classification Machine Learning for You

A PyTorch-based machine learning library that provides both Multi-Layer Perceptron (MLP) and Graph Neural Network (GNN) implementations for image classification tasks. The library specializes in converting images to graph representations for GNN-based classification.

## Features

- **Dual Architecture Support**: Train and inference with both MLPs and GNNs
- **Image-to-Graph Conversion**: Automatically convert images to graph representations for GNN processing
- **Flexible Training**: Configurable training with early stopping, model checkpointing, and progress tracking
- **Multiple Graph Construction Methods**: Support for pixel-based, superpixel, and patch-based graph generation
- **Inference Tools**: Easy-to-use inference functions for both model types
- **MNIST Support**: Built-in support for MNIST dataset with optimized preprocessing
- **Caching**: Optimized data loading with caching for faster training iterations

## Installation

### Option 1: Using pip (Local Installation)

```bash
# Clone the repository
git clone <repository-url>
cd scml4u

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric dependencies (for GNN support)
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
pip install torch-geometric
```

### Option 2: Using Docker

```bash
# Build the Docker image
docker build -t scml4u .

# Run the container
docker run -it scml4u
```

### Requirements

- Python 3.10+
- PyTorch 2.1.0+
- PyTorch Geometric
- NumPy < 2.0
- PIL/Pillow
- scikit-image
- tqdm
- matplotlib
- pandas

## Quick Start

### Basic Usage

```python
import scml4u

# Train an MLP model
scml4u.train_classifier.run_train_MLP(
    epochs=100, 
    resize_value=28, 
    hidden_layers=5,
    dataset_path='mnist', 
    output_path='models/mlp_results'
)

# Train a GNN model
scml4u.train_classifier.run_train_GNN(
    epochs=100,
    resize_value=28,
    n_blocks=10,
    dataset_path='mnist', 
    output_path='models/gnn_results'
)
```

### Using the Main Script

```python
from main import train_MLP, train_GNN

# Train MLP
train_MLP(
    epochs=30,
    channels=3,
    resize_value=128,
    batch_size=8,
    hidden_layers=2,
    output_path='weights/MLP'
)

# Train GNN
train_GNN(
    epochs=30,
    channels=3,
    resize_value=64,
    batch_size=8,
    hidden_layers=2,
    method='pixel',  # 'pixel', 'superpixel', or 'patch'
    output_path='weights/GNN',
    dataset_path='dataset',
    grayscale=False
)
```

## Architecture Overview

### Multi-Layer Perceptron (MLP)
- Flexible architecture with configurable hidden layers
- Support for different activation functions and normalization
- Automatic image flattening for standard neural network processing

### Graph Neural Network (GNN)
- **GraphNet**: Core GNN architecture using MetaLayer for message passing
- **CombinedModel**: Wrapper that combines GraphNet with a linear classifier
- **Image-to-Graph Conversion**: Three methods available:
  - **Pixel-based**: Each pixel becomes a node
  - **Superpixel**: Groups of similar pixels become nodes
  - **Patch-based**: Image patches become nodes

### Key Components

#### 1. Data Loading (`dataloader.py`)
```python
# Graph-based data loading for GNN
GraphDatasetLoader(
    dataset_path='dataset',
    resize_value=28,
    grayscale=True
)

# Standard image loading for MLP
DatasetLoader(
    dataset_path='dataset',
    resize_value=28,
    grayscale=False
)
```

#### 2. Training (`train_classifier.py`)
- Automatic model checkpointing
- Early stopping with configurable patience
- Training progress logging
- Loss curve visualization

#### 3. Inference (`inference_classifier.py`)
```python
# Load and use trained models
model = load_gnn_model('weights/model.pth', resize_value=28)
prediction = gnn_inference(model, 'test_image.png')
```

## Dataset Structure

Organize your datasets in the following structure:
```
dataset/
├── class1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── class2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── ...
```

For MNIST:
```
dataset/mnist/
├── 0/
│   ├── 0000.png
│   └── ...
├── 1/
│   ├── 0001.png
│   └── ...
└── ...
```

## Advanced Usage

### Custom Graph Construction

```python
from scml4u.convert import image_to_graph_pixel_optimized

# Convert image to graph representation
x, pos, edge_index = image_to_graph_pixel_optimized(
    image,
    resize_value=28,
    npoints_of_scheme=4,
    grayscale=True
)
```

### Model Configuration

```python
# Configure GNN architecture
graph_net = scml4u.ai_gnn.GraphNet(
    num_local_features=1,  # 1 for grayscale, 3 for RGB
    space_dim=2,
    out_channels=1,
    n_blocks=3
)

model = scml4u.ai_gnn.CombinedModel(
    graph_net=graph_net,
    num_nodes=28*28,
    classes=10
)
```

### Training with Custom Parameters

```python
# Advanced GNN training
train_GNN(
    epochs=100,
    resize_value=28,
    batch_size=16,
    method='pixel',
    max_samples=5000,  # Limit dataset size for testing
    patience=10,
    grayscale=True,
    use_cache=True,
    output_path='weights/custom_model'
)
```

## Example Results

The library includes example training results in `examples/results/`:
- Model checkpoints for each epoch
- Training and validation loss matrices
- Loss curve visualizations
- Performance metrics

## API Reference

### Core Modules

- **`ai_mlp.MLP`**: Multi-layer perceptron implementation
- **`ai_gnn.GraphNet`**: Graph neural network core
- **`ai_gnn.CombinedModel`**: Complete GNN classifier
- **`train_classifier.train`**: Universal training function
- **`dataloader.GraphDatasetLoader`**: Graph data loading
- **`inference_classifier`**: Model inference utilities
- **`convert`**: Image-to-graph conversion functions

### Utility Functions

- **`lossfunction.get_loss_function`**: Flexible loss function selection
- **`utils`**: Various utility functions for data processing
- **`genmnistdownload`**: MNIST dataset utilities

## Performance Tips

1. **Use Caching**: Enable `use_cache=True` for faster data loading
2. **Limit Dataset Size**: Use `max_samples` parameter for quick testing
3. **Choose Graph Method**: 
   - `pixel` for detailed analysis (slower)
   - `superpixel` for balanced performance
   - `patch` for faster training
4. **Batch Size**: Adjust based on GPU memory (8-16 typically works well)
5. **Early Stopping**: Use patience parameter to avoid overfitting

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this library in your research, please cite:

```
@article{vannson2024scml4u,
  title={Adapter les GNNs à la classification d'images},
  author={Vannson, Alexis},
  year={2024}
}
```

## Support

For issues and questions:
1. Check the examples in the `examples/` directory
2. Review the documentation in `tex/GNN.pdf`
3. Open an issue on the repository

---

**Note**: This library is designed for educational and research purposes, providing an accessible way to experiment with both traditional MLPs and modern GNN approaches for image classification. 
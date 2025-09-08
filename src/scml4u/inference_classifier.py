#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Python packages
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import argparse
from typing import Tuple, Union, Optional

# MRG packages
from . import ai_mlp
from . import ai_gnn
from . import convert


def load_mlp_model(weights_path: str, resize_value: int = 28, num_classes: int = 10, channels: int = 3, hidden_layers: int = 2):
    """Load a trained MLP model."""
    input_dim = channels * resize_value * resize_value  # RGB or grayscale
    model = ai_mlp.MLP(in_dim=input_dim, out_dim=num_classes, hidden_layers=hidden_layers)
    model.load_state_dict(torch.load(weights_path))
    model.eval()
    return model


def load_gnn_model(weights_path: str, resize_value: int = 28, num_classes: int = 10, 
                   n_blocks: int = 3, grayscale: bool = True):
    """Load a trained GNN model."""
    num_nodes = resize_value * resize_value
    num_local_features = 1 if grayscale else 3
    
    graph_net = ai_gnn.GraphNet(
        num_local_features=num_local_features,
        space_dim=2,
        out_channels=1,
        n_blocks=n_blocks
    )
    model = ai_gnn.CombinedModel(graph_net=graph_net, num_nodes=num_nodes, classes=num_classes)
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.eval()
    return model


def preprocess_image_mlp(image_path: str, resize_value: int = 28, channels: int = 3) -> torch.Tensor:
    """Preprocess image for MLP inference (RGB or grayscale, flattened)."""
    if channels == 1:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
    else:
        img = Image.open(image_path).convert('RGB')  # Convert to RGB
    
    img = img.resize((resize_value, resize_value))
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    if channels == 1:
        input_tensor = torch.tensor(img_array.flatten(), dtype=torch.float32).unsqueeze(0)
    else:
        # For RGB, flatten the entire array (height * width * channels)
        input_tensor = torch.tensor(img_array.flatten(), dtype=torch.float32).unsqueeze(0)
    
    return input_tensor


def preprocess_image_gnn(image_path: str, resize_value: int = 28, 
                        diagonals: bool = False, grayscale: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Preprocess image for GNN inference (convert to graph)."""
    x, pos, edge_index = convert.image_to_graph_pixel_optimized(
        image_path,
        resize_value=resize_value,
        diagonals=diagonals,
        use_cache=True,
        grayscale=grayscale,
        connectivity="4" if not diagonals else "8"
    )
    
    # Convert to PyTorch tensors
    x = torch.tensor(x, dtype=torch.float32)
    pos = torch.tensor(pos, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    
    return x, pos, edge_index


def mlp_inference(image_path: str, weights_path: str, resize_value: int = 28, 
                  num_classes: int = 10, channels: int = 3, hidden_layers: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run MLP inference on a single image."""
    model = load_mlp_model(weights_path, resize_value, num_classes, channels, hidden_layers)
    input_tensor = preprocess_image_mlp(image_path, resize_value, channels)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = F.softmax(logits, dim=1)
    
    return logits, probabilities


def gnn_inference(image_path: str, weights_path: str, resize_value: int = 28, 
                  num_classes: int = 10, n_blocks: int = 3, 
                  diagonals: bool = False, grayscale: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run GNN inference on a single image."""
    model = load_gnn_model(weights_path, resize_value, num_classes, n_blocks, grayscale)
    x, pos, edge_index = preprocess_image_gnn(image_path, resize_value, diagonals, grayscale)
    
    with torch.no_grad():
        logits = model((x, pos, edge_index))
        # Ensure logits has the right shape for softmax
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        probabilities = F.softmax(logits, dim=1)
    
    return logits, probabilities


def predict_class(logits: torch.Tensor, probabilities: torch.Tensor, 
                  class_names: Optional[list] = None) -> dict:
    """Get prediction results."""
    predicted_class = torch.argmax(logits, dim=1).item()
    confidence = probabilities[0, predicted_class].item()
    
    result = {
        'predicted_class': predicted_class,
        'confidence': confidence,
        'probabilities': probabilities[0].tolist()
    }
    
    if class_names:
        result['predicted_class_name'] = class_names[predicted_class]
        result['all_classes'] = class_names
    
    return result


def run_inference_classifier_online():

    parser = argparse.ArgumentParser(description='Run inference with MLP or GNN models')
    parser.add_argument('--image_path', type=str, required=True, help='Path to input image')
    parser.add_argument('--model_type', type=str, choices=['mlp', 'gnn'], required=True, help='Model type')
    parser.add_argument('--weights_path', type=str, required=True, help='Path to model weights')
    parser.add_argument('--resize_value', type=int, default=28, help='Image resize value')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--n_blocks', type=int, default=3, help='Number of GNN blocks (for GNN only)')
    parser.add_argument('--diagonals', action='store_true', help='Use diagonal edges (for GNN only)')
    parser.add_argument('--grayscale', action='store_true', default=True, help='Use grayscale (for GNN only)')
    
    args = parser.parse_args()
    
    # MNIST class names
    class_names = [str(i) for i in range(10)]
    
    try:
        if args.model_type == 'mlp':
            logits, probabilities = mlp_inference(
                args.image_path, args.weights_path, args.resize_value, args.num_classes, 3, 2
            )
        elif args.model_type == 'gnn':
            logits, probabilities = gnn_inference(
                args.image_path, args.weights_path, args.resize_value, args.num_classes,
                args.n_blocks, args.diagonals, args.grayscale
            )
        
        result = predict_class(logits, probabilities, class_names)
        
        print(f"\n=== {args.model_type.upper()} Inference Results ===")
        print(f"Input image: {args.image_path}")
        print(f"Model weights: {args.weights_path}")
        print(f"Predicted class: {result['predicted_class_name']} (class {result['predicted_class']})")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"\nAll probabilities:")
        for i, (class_name, prob) in enumerate(zip(result['all_classes'], result['probabilities'])):
            print(f"  Class {class_name}: {prob:.4f}")
        
        return result
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return None



def run_inference_classifier(
    image_path: str,
    model_type: str,
    weights_path: str,
    resize_value: int = 28,
    num_classes: int = 10,
    channels: int = 3,
    hidden_layers: int = 2,
    n_blocks: int = 3,
    diagonals: bool = False,
    grayscale: bool = True,
    display_results: bool = True
):
    """
    Run inference with MLP or GNN models using provided parameters.

    Args:
        image_path (str): Path to input image.
        model_type (str): 'mlp' or 'gnn'.
        weights_path (str): Path to model weights.
        resize_value (int): Image resize value.
        num_classes (int): Number of classes.
        channels (int): Number of channels (for MLP only). 1 for grayscale, 3 for RGB.
        hidden_layers (int): Number of hidden layers (for MLP only).
        n_blocks (int): Number of GNN blocks (for GNN only).
        diagonals (bool): Use diagonal edges (for GNN only).
        grayscale (bool): Use grayscale (for GNN only).

    Returns:
        dict: Inference result dictionary or None if error.
    """
    # MNIST class names
    class_names = [str(i) for i in range(num_classes)]

    try:
        if model_type == 'mlp':
            logits, probabilities = mlp_inference(
                image_path, weights_path, resize_value, num_classes, channels, hidden_layers
            )
        elif model_type == 'gnn':
            logits, probabilities = gnn_inference(
                image_path, weights_path, resize_value, num_classes,
                n_blocks, diagonals, grayscale
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        result = predict_class(logits, probabilities, class_names)
        if display_results:
            print(f"\n=== {model_type.upper()} Inference Results ===")
            print(f"Input image: {image_path}")
            print(f"Model weights: {weights_path}")
            print(f"Predicted class: {result['predicted_class_name']} (class {result['predicted_class']})")
            print(f"Confidence: {result['confidence']:.4f}")
            print(f"\nAll probabilities:")
            for i, (class_name, prob) in enumerate(zip(result['all_classes'], result['probabilities'])):
                print(f"  Class {class_name}: {prob:.4f}")

        return result

    except Exception as e:
        print(f"Error during inference: {e}")
        return None








# if __name__ == "__main__":

#     _ = run_inference_classifier()

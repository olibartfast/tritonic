# ------------------------------------------------------------------------
# RF-DETR Segmentation Export
# Based on RF-DETR export module
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------

"""
Export ONNX model for RF-DETR segmentation deployment

This script uses the RF-DETR library's built-in export functionality.
The RFDETRSegPreview class handles the export internally and outputs:
- Bounding boxes (dets)
- Class labels (labels)  
- Segmentation masks (masks)
"""

import argparse
from rfdetr import RFDETRSegPreview


def main():
    parser = argparse.ArgumentParser('RF-DETR Segmentation Export Script',
                                     description='Export RF-DETR segmentation model to ONNX format')
    
    # Export options that will be passed to the model's export() method
    parser.add_argument('--output_dir', default=None, type=str,
                        help='Path to save exported model (default: current directory)')
    parser.add_argument('--opset_version', default=17, type=int,
                        help='ONNX opset version (default: 17)')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify ONNX model using onnxsim')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size for export (default: 1)')
    parser.add_argument('--input_size', default=640, type=int,
                        help='Input image size (default: 640)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("RF-DETR Segmentation Model Export")
    print("="*60)
    
    # Initialize the segmentation model
    print("\n[1/2] Loading RF-DETR Segmentation model...")
    model = RFDETRSegPreview()
    
    # Build export kwargs from arguments
    export_kwargs = {
        'opset_version': args.opset_version,
        'simplify': args.simplify,
        'batch_size': args.batch_size,
    }
    
    # Add output_dir if specified
    if args.output_dir:
        export_kwargs['output_dir'] = args.output_dir
    
    # Export using the model's built-in export method
    print("\n[2/2] Exporting to ONNX format...")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Input size: {args.input_size}x{args.input_size}")
    print(f"  - ONNX opset: {args.opset_version}")
    print(f"  - Simplify: {args.simplify}")
    
    model.export(**export_kwargs)
    
    print("\n" + "="*60)
    print("✓ Export complete!")
    print("="*60)
    print("\nModel outputs:")
    print("  - dets: Bounding boxes [batch, num_queries, 4]")
    print("  - labels: Class logits [batch, num_queries, num_classes]")
    print("  - masks: Segmentation masks [batch, num_queries, H, W]")
    print("\nNote: This is a segmentation model with mask prediction capability.")
    print("="*60)


if __name__ == '__main__':
    main()
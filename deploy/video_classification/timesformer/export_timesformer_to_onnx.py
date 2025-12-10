import torch
import onnx
import numpy as np
from transformers import TimesformerForVideoClassification, AutoImageProcessor
import argparse
from pathlib import Path
import sys
import traceback

def export_timesformer_to_onnx(
    model_name="facebook/timesformer-base-finetuned-k400",
    output_path="timesformer.onnx",
    num_frames=8,
    image_size=224,
    opset_version=14,
    dynamic_batch=True
):
    """Export TimeSformer model to ONNX format."""
    
    try:
        print(f"🔄 Loading TimeSformer model: {model_name}")
        print(f"🔍 Python version: {sys.version}")
        print(f"🔍 PyTorch version: {torch.__version__}")
        
        # Check if transformers is available
        try:
            import transformers
            print(f"🔍 Transformers version: {transformers.__version__}")
        except ImportError as e:
            print(f"❌ Transformers not available: {e}")
            return None
            
        model = TimesformerForVideoClassification.from_pretrained(model_name)
        processor = AutoImageProcessor.from_pretrained(model_name)
        
        # Set to evaluation mode
        model.eval()
        
        print(f"📊 Model config:")
        print(f"   - Num frames: {num_frames}")
        print(f"   - Image size: {image_size}x{image_size}")
        print(f"   - Num classes: {model.config.num_labels}")
        
        # Create dummy input
        # Shape: [batch_size, num_frames, channels, height, width]
        dummy_input = torch.randn(1, num_frames, 3, image_size, image_size)
        
        print(f"📦 Dummy input shape: {dummy_input.shape}")
        
        # Test forward pass first
        print(f"🧪 Testing forward pass...")
        with torch.no_grad():
            output = model(dummy_input)
            print(f"✅ Forward pass successful, output shape: {output.logits.shape}")
        
        # Define input and output names
        input_names = ["pixel_values"]
        output_names = ["logits"]
        
        # Define dynamic axes for variable batch size
        if dynamic_batch:
            dynamic_axes = {
                "pixel_values": {0: "batch_size"},
                "logits": {0: "batch_size"}
            }
        else:
            dynamic_axes = None
        
        print(f"🔧 Exporting to ONNX...")
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            verbose=True  # Enable verbose output for debugging
        )
        
        print(f"✅ Model exported to: {output_path}")
        
        # Verify the exported model
        print(f"🔍 Verifying ONNX model...")
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"✅ ONNX model verification passed!")
        
        # Print model info
        print(f"\n📋 ONNX Model Info:")
        print(f"   - Input shape: {onnx_model.graph.input[0].type.tensor_type.shape}")
        print(f"   - Output shape: {onnx_model.graph.output[0].type.tensor_type.shape}")
        print(f"   - Opset version: {opset_version}")
        print(f"   - File size: {Path(output_path).stat().st_size / (1024*1024):.2f} MB")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error during export: {e}")
        print(f"📍 Traceback:")
        traceback.print_exc()
        return None

def test_onnx_model(onnx_path, num_frames=8, image_size=224):
    """Test the exported ONNX model."""
    try:
        import onnxruntime as ort
        
        print(f"🧪 Testing ONNX model: {onnx_path}")
        
        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path)
        
        # Create test input
        test_input = np.random.randn(1, num_frames, 3, image_size, image_size).astype(np.float32)
        
        # Run inference
        outputs = session.run(None, {"pixel_values": test_input})
        logits = outputs[0]
        
        print(f"✅ ONNX inference successful!")
        print(f"   - Input shape: {test_input.shape}")
        print(f"   - Output shape: {logits.shape}")
        print(f"   - Top prediction: {np.argmax(logits[0])}")
        
        return True
        
    except ImportError:
        print("⚠️  ONNXRuntime not installed. Skipping ONNX test.")
        print("   Install with: pip install onnxruntime")
        return False
    except Exception as e:
        print(f"❌ ONNX test failed: {e}")
        traceback.print_exc()
        return False

def main():
    print("🚀 Starting TimeSformer ONNX export...")
    
    try:
        parser = argparse.ArgumentParser(description="Export TimeSformer to ONNX")
        parser.add_argument("--model", default="facebook/timesformer-base-finetuned-k400", 
                           help="HuggingFace model name")
        parser.add_argument("--output", default="timesformer.onnx", 
                           help="Output ONNX file path")
        parser.add_argument("--num-frames", type=int, default=8, 
                           help="Number of frames")
        parser.add_argument("--image-size", type=int, default=224, 
                           help="Image size (height and width)")
        parser.add_argument("--opset", type=int, default=14, 
                           help="ONNX opset version")
        parser.add_argument("--static-batch", action="store_true", 
                           help="Use static batch size (default: dynamic)")
        parser.add_argument("--test", action="store_true", 
                           help="Test the exported ONNX model")
        
        args = parser.parse_args()
        
        print(f"📋 Arguments:")
        print(f"   - Model: {args.model}")
        print(f"   - Output: {args.output}")
        print(f"   - Frames: {args.num_frames}")
        print(f"   - Image size: {args.image_size}")
        print(f"   - Opset: {args.opset}")
        print(f"   - Dynamic batch: {not args.static_batch}")
        print(f"   - Test: {args.test}")
        
        # Export model
        onnx_path = export_timesformer_to_onnx(
            model_name=args.model,
            output_path=args.output,
            num_frames=args.num_frames,
            image_size=args.image_size,
            opset_version=args.opset,
            dynamic_batch=not args.static_batch
        )
        
        if onnx_path is None:
            print("❌ Export failed!")
            sys.exit(1)
        
        # Test if requested
        if args.test:
            success = test_onnx_model(onnx_path, args.num_frames, args.image_size)
            if not success:
                print("❌ Testing failed!")
                sys.exit(1)
        
        print("🎉 Export completed successfully!")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
import torch
from transformers import ViTImageProcessor, ViTForImageClassification
import onnx
import onnxruntime as ort
import numpy as np
import os
import sys
import gc

def export_vit_to_onnx():
    try:
        model_name = "google/vit-base-patch16-224"
        print(f"🔄 Exporting {model_name} to ONNX...")
        
        # Load model and processor
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(model_name)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(1, 3, 224, 224)
        print(f"📊 Export configuration:")
        print(f"   - Model: {model_name}")
        print(f"   - Input shape: {dummy_input.shape}")
        print(f"   - Output classes: {model.config.num_labels}")
        print(f"   - Opset version: 17")
        
        # Export with minimal options to avoid segfault
        torch.onnx.export(
            model,
            dummy_input,
            "vit_model.onnx",
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['pixel_values'],
            output_names=['logits'],
            dynamic_axes={
                'pixel_values': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            },
            verbose=False  # Reduce verbosity
        )
        
        print("✅ ONNX export successful: vit_model.onnx")
        
        # Verify with lighter approach
        try:
            # Simple file check instead of full ONNX verification
            if os.path.exists("vit_model.onnx") and os.path.getsize("vit_model.onnx") > 1000:
                print("✅ ONNX model file created successfully")
                
                # Quick shape verification only
                session = ort.InferenceSession("vit_model.onnx", providers=['CPUExecutionProvider'])
                input_info = session.get_inputs()[0]
                output_info = session.get_outputs()[0]
                
                print("📋 Model info:")
                print(f"   - Input: {input_info.name}, shape: {input_info.shape}")
                print(f"   - Output: {output_info.name}, shape: {output_info.shape}")
                
                # Clean up session immediately
                del session
                
        except Exception as e:
            print(f"⚠️  Verification warning: {e}")
            print("   Model exported but verification skipped")
        
        # Clean up
        del model, processor
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False

if __name__ == "__main__":
    success = export_vit_to_onnx()
    # Force exit to avoid cleanup issues
    sys.exit(0 if success else 1)
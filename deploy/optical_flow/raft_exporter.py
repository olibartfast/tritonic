import torch
from torchvision.models.optical_flow import raft_small, raft_large
import torch.nn as nn
import argparse
import onnx
from onnx_graphsurgeon import Graph, import_onnx
import os

class TritonRAFTWrapper(nn.Module):
    """Wrapper for RAFT model to make it Triton-compatible by combining outputs."""
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
    
    def forward(self, x1, x2):
        # RAFT returns a list of flow predictions at different scales
        # We'll take the final prediction which is most refined
        flow_predictions = self.base_model(x1, x2)
        # Return only the final flow prediction
        return flow_predictions[-1]


def load_model(model_type='small'):
    """Load and prepare RAFT model for export."""
    # Select model based on type
    model_fn = raft_small if model_type == 'small' else raft_large
    model = model_fn(pretrained=True)
    
    # Wrap the model to make it Triton-compatible
    model = TritonRAFTWrapper(model)
    
    # Move model to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    return model, device


def prepare_sample_input(batch_size, height, width, device):
    """Create sample input tensors for model export."""
    input1 = torch.randn(batch_size, 3, height, width).to(device)
    input2 = torch.randn(batch_size, 3, height, width).to(device)
    return input1, input2


def check_and_fix_onnx_graph(model_path):
    """
    Check and fix the input/output order of an ONNX graph.
    
    Args:
        model_path: Path to the ONNX model file
    Returns:
        bool: True if fixes were applied, False otherwise
    """
    # Load the model
    onnx_model = onnx.load(model_path)
    graph = import_onnx(onnx_model)
    
    # Expected order of inputs and outputs
    expected_inputs = ['input1', 'input2']
    expected_outputs = ['flow_prediction']
    
    # Get current inputs and outputs
    current_inputs = [inp.name for inp in graph.inputs]
    current_outputs = [out.name for out in graph.outputs]
    
    print(f"Current input order: {current_inputs}")
    print(f"Current output order: {current_outputs}")
    
    needs_fixing = False
    
    # Check if inputs are in correct order
    if current_inputs != expected_inputs:
        print("Input order needs to be fixed")
        needs_fixing = True
        
        # Reorder inputs
        graph.inputs = sorted(graph.inputs, 
                            key=lambda x: expected_inputs.index(x.name) 
                            if x.name in expected_inputs else len(expected_inputs))
    
    # Check if outputs are in correct order
    if current_outputs != expected_outputs:
        print("Output order needs to be fixed")
        needs_fixing = True
        
        # Reorder outputs
        graph.outputs = sorted(graph.outputs, 
                             key=lambda x: expected_outputs.index(x.name) 
                             if x.name in expected_outputs else len(expected_outputs))
    
    if needs_fixing:
        # Export the fixed model
        fixed_model_path = model_path.replace('.onnx', '_fixed.onnx')
        onnx.save(graph.export(), fixed_model_path)
        print(f"Fixed model saved as: {fixed_model_path}")
        
        # Verify the fix
        fixed_graph = import_onnx(onnx.load(fixed_model_path))
        print("\nVerifying fixed model:")
        print(f"Fixed input order: {[inp.name for inp in fixed_graph.inputs]}")
        print(f"Fixed output order: {[out.name for out in fixed_graph.outputs]}")
        return True
    else:
        print("\nNo fixes needed - inputs and outputs are in correct order")
        return False


def export_torchscript_trace(model, example_inputs, model_type, output_dir, dynamic=True):
    """Export model using TorchScript tracing."""
    print(f"Exporting {model_type} model to TorchScript (Tracing)...")
    try:
        # Create wrapper for dynamic support
        class DynamicRAFT(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            
            def forward(self, x1, x2):
                return self.model(x1, x2)

        if dynamic:
            wrapped_model = DynamicRAFT(model)
            traced_model = torch.jit.trace(
                wrapped_model,
                example_inputs,
                check_trace=True,
                check_tolerance=1e-4
            )
            filename = f"{output_dir}/raft_{model_type}_traced_dynamic.pt"
            traced_model.save(filename, _extra_files={
                "dynamic_axes.json": str({
                    "input1": {0: "batch"},
                    "input2": {0: "batch"},
                    "output": {0: "batch"}
                })
            })
            print(f"Dynamic traced TorchScript model saved as '{filename}'")
        else:
            # Original static tracing
            traced_model = torch.jit.trace(model, example_inputs)
            filename = f"{output_dir}/raft_{model_type}_traced_torchscript.pt"
            traced_model.save(filename)
            print(f"Static traced TorchScript model saved as '{filename}'")
    except Exception as e:
        print(f"Tracing failed: {e}")


def export_torchscript_script(model, model_type, output_dir):
    """Export model using TorchScript scripting."""
    print(f"Exporting {model_type} model to TorchScript (Scripting)...")
    try:
        scripted_model = torch.jit.script(model)
        filename = f"{output_dir}/raft_{model_type}_scripted_torchscript.pt"
        scripted_model.save(filename)
        print(f"Scripted TorchScript model saved as '{filename}'")
    except Exception as e:
        print(f"Scripting failed: {e}")


def export_onnx(model, example_inputs, model_type, output_dir):
    """Export model to ONNX format."""
    print(f"Exporting {model_type} model to ONNX...")
    try:
        filename = f"{output_dir}/raft_{model_type}.onnx"
        input1, input2 = example_inputs
        
        torch.onnx.export(
            model,
            (input1, input2),
            filename,
            export_params=True,
            opset_version=16,
            do_constant_folding=True,
            input_names=['input1', 'input2'],
            output_names=['flow_prediction'],
            dynamic_axes={
                'input1': {0: 'batch_size', 2: 'height', 3: 'width'},
                'input2': {0: 'batch_size', 2: 'height', 3: 'width'},
                'flow_prediction': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )
        print(f"ONNX model saved as '{filename}'")
        
        # Check and fix ONNX graph if needed
        print("\nChecking ONNX graph input/output order...")
        check_and_fix_onnx_graph(filename)
        
    except Exception as e:
        print(f"ONNX export failed: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Export RAFT optical flow models to various formats.')
    
    parser.add_argument('--model-type', type=str, choices=['small', 'large', 'both'],
                      default='both', help='Type of RAFT model to export (default: both)')
    
    parser.add_argument('--batch-size', type=int, default=1,
                      help='Batch size for sample inputs (default: 1)')
    
    parser.add_argument('--height', type=int, default=520,
                      help='Height of sample inputs (default: 520)')
    
    parser.add_argument('--width', type=int, default=960,
                      help='Width of sample inputs (default: 960)')
    
    parser.add_argument('--output-dir', type=str, default='.',
                      help='Output directory for exported models (default: current directory)')
    
    parser.add_argument('--format', type=str, 
                      choices=['all', 'traced', 'scripted', 'onnx'],
                      default='all',
                      help='Export format(s) to use (default: all)')
    
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                      default=None,
                      help='Device to use (default: use cuda if available)')

    parser.add_argument('--cpu-only', action='store_true',
                      help='Force CPU-only execution (overrides --device)')

    parser.add_argument('--dynamic', action='store_true',
                      help='Enable dynamic batching for traced export')
    
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Handle device selection
    if args.cpu_only:
        device = torch.device('cpu')
        print(f"Using device: {device} (forced CPU-only)")
    elif args.device:
        device = torch.device(args.device)
        print(f"Using device: {device} (specified)")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device} (auto-detected)")
    
    # Determine which models to export
    model_types = ['small', 'large'] if args.model_type == 'both' else [args.model_type]
    
    # Process each model type
    for model_type in model_types:
        print(f"\nProcessing RAFT {model_type} model...")
        
        # Load model
        model, _ = load_model(model_type)
        model = model.to(device)
        
        # Prepare sample inputs
        example_inputs = prepare_sample_input(args.batch_size, args.height, args.width, device)
        
        # Export in specified format(s)
        if args.format in ['all', 'traced']:
            export_torchscript_trace(model, example_inputs, model_type, args.output_dir, dynamic=args.dynamic)
        
        if args.format in ['all', 'scripted']:
            export_torchscript_script(model, model_type, args.output_dir)
        
        if args.format in ['all', 'onnx']:
            export_onnx(model, example_inputs, model_type, args.output_dir)


if __name__ == "__main__":
    main()
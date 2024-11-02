import torch
import torchvision.models as torch_models
import timm
import argparse

def export_to_torchscript(model, output_script, num_classes, input_size, input_name, output_name):
    # Optional: Modify the model (e.g., change the output layer for your specific classification task)
    if hasattr(model, "fc"):
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Modify the last fully connected layer
    elif hasattr(model, "classifier"):
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

    # Export the model to TorchScript
    scripted_model = torch.jit.script(model)
    scripted_model.save(output_script)
    print(f"TorchScript model saved to {output_script}")

def export_to_onnx(model, output_onnx, num_classes, input_size, input_name, output_name):
    # Optional: Modify the model (e.g., change the output layer for your specific classification task)
    if hasattr(model, "fc"):
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  # Modify the last fully connected layer
    elif hasattr(model, "classifier"):
        model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

    # Export the model to ONNX format
    dummy_input = torch.randn(input_size)
    torch.onnx.export(model, dummy_input, output_onnx, input_names=[input_name], output_names=[output_name], verbose=True)
    print(f"ONNX model saved to {output_onnx}")

def load_torchvision_model(model_name, pretrained=True):
    return torch_models.__dict__[model_name](pretrained=pretrained)

def load_timm_model(model_name, pretrained=True):
    return timm.create_model(model_name, pretrained=pretrained)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export models from torchvision and timm to TorchScript or ONNX format.")
    parser.add_argument("--library", choices=["torchvision", "timm"], required=True, help="Library (torchvision or timm)")
    parser.add_argument("--model", required=True, help="Name of the model (e.g., resnet50, efficientnet_b3, etc.)")
    parser.add_argument("--export_format", choices=["torchscript", "onnx"], required=True, help="Export format (torchscript or onnx)")
    parser.add_argument("--output_script", help="Path to save the TorchScript model")
    parser.add_argument("--output_onnx", help="Path to save the ONNX model")
    parser.add_argument("--num_classes", type=int, default=1000, help="Number of output classes (default: 1000)")
    parser.add_argument("--input_size", nargs=4, type=int, default=[1, 3, 224, 224], help="Input size (default: 1, 3, 224, 224)")
    parser.add_argument("--input_name", default="input", help="Name of the input")
    parser.add_argument("--output_name", default="output", help="Name of the output")

    args = parser.parse_args()

    input_size = tuple(args.input_size)

    if args.library == "torchvision":
        model = load_torchvision_model(args.model)
    elif args.library == "timm":
        model = load_timm_model(args.model)

    if args.export_format == "torchscript":
        if args.output_script is None:
            print("Please specify the output path for TorchScript.")
        else:
            export_to_torchscript(model, args.output_script, args.num_classes, input_size, args.input_name, args.output_name)
    elif args.export_format == "onnx":
        if args.output_onnx is None:
            print("Please specify the output path for ONNX.")
        else:
            export_to_onnx(model, args.output_onnx, args.num_classes, input_size, args.input_name, args.output_name)

# Running Inference with the Docker Scripts

Wrapper scripts for running TritonIC in a container, with the full list of
placeholder substitutions and model-type tags.

Back to the [documentation index](README.md) | [project README](../README.md).

## Quick start

Use the provided Docker scripts for quick testing:

```bash
# Run object detection
./docker/scripts/run_client.sh

# Run with debug mode
./docker/scripts/run_debug.sh

# Run optical flow
./docker/scripts/run_optical_flow.sh

# Run unit tests
./docker/scripts/run_tests.sh
```

### Debugging Tips
Check [`.vscode/launch.json`](../.vscode/launch.json) for additional configuration examples

### Placeholder Descriptions
- **`/path/to/source.format`**: Path to the input video or image file, for optical flow you must pass two images as comma separated list
- **`<model_type>`**: Model type (e.g., `yolov5`, `yolov8`, `yolo11`, `yoloseg`, `torchvision-classifier`, `tensorflow-classifier`, `vit-classifier`, check below [Model Type Parameters](#model-type-tag-parameters))
- **`<model_name_folder_on_triton>`**: Name of the model folder on the Triton server
- **`/path/to/labels/coco.names`**: Path to the label file (e.g., COCO labels)
- **`<http or grpc>`**: Communication protocol (`http` or `grpc`)
- **`<triton-ip>`**: IP address of your Triton server
- **`<8000 for http, 8001 for grpc>`**: Port number
- **`<batch or b >`**: Batch size. For compatible independent-image models (classification, detection, segmentation, pose, depth, open-vocab) with `max_batch_size > 1` in the model config, automatic batching groups up to this many images into a single inference call, capped by the model `max_batch_size`.
- **`<inference_timeout or it>`**: Inference timeout in milliseconds. `0` keeps the backend default; positive values apply to chat HTTP requests, Triton infer requests, and model-load readiness waits.
- **`<input_sizes or -is>`**: Input sizes input for dynamic axes. Semi-colon separated list format: CHW;CHW;... (e.g., '3,224,224' for single input or '3,224,224;3,224,224' for two inputs, '3,640,640;2' for rtdetr/dfine models)


To view all available parameters, run:
```bash
./tritonic --help
```

### Model Type Tag Parameters
| Model                  | Model Type Parameter   | Notes |
|------------------------|------------------------|-------|
| YOLOv5 / v6 / v7 / v8 / v9 / v11 / v12 | `yolo` | Any `yolo*` variant works. Standard format |
| YOLOv7 End-to-End      | `yolov7e2e`            | Only for YOLOv7 exported with `--grid --end2end` flags (requires TensorRT backend) |
| YOLOv10                | `yolov10`              | Specific output format |
| YOLO26                | `yolo26`              | Specific output format (i.e. is the same of yolov10) |
| YOLO-NAS               | `yolonas`              | Specific output format |
| RT-DETR / RT-DETRv2 / RT-DETRv4 / D-FINE / DEIM / DEIMv2 | `rtdetr` | All RT-DETR style models share the same postprocessor |
| RT-DETR Ultralytics    | `rtdetrul`             |       |
| RF-DETR Detection | `rfdetr`  |       |
| YOLOv5/v8/v11/v12 Segmentation | `yoloseg`       |       |
| YOLO26 Segmentation | `yolo26seg`       |       |
| YOLOv10 Segmentation | `yolov10seg`       |    
| RF-DETR Segmentation | `rfdetrseg`  |       |
| Torchvision Classifier | `torchvision-classifier` |     |
| Tensorflow Classifier  | `tensorflow-classifier` |      |
| ViT Classifier         | `vit-classifier`       |       |
| RAFT Optical Flow      | `raft`                 |       |
| VideoMAE               | `videomae`             | 16-frame sliding window video |
| ViViT                  | `vivit`                | Video Transformer |
| TimeSformer            | `timesformer`          | Video Transformer |
| ViTPose                | `vitpose`              | Pose estimation (COCO 17 keypoints) |
| Depth Anything V2      | `depth_anything_v2`    | Monocular depth estimation |
| OWLv2                  | `owlv2`                | Open-vocabulary detection |
| OWL-ViT                | `owlvit`               | Open-vocabulary detection |
| Grounding DINO         | `grounding_dino`       | Open-vocabulary detection |
| RF-DETR Keypoints      | `rfdetr_keypoints`     | Single-stage person keypoints (17 COCO) |
| YOLOv5 Pose            | `yolov5pose`           | Pose estimation |
| YOLOv8 Pose            | `yolov8pose`           | Pose estimation |
| YOLO11 Pose            | `yolo11pose`           | Pose estimation |
| YOLO26 Pose            | `yolo26pose`           | Pose estimation |

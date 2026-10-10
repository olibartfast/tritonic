# Supported Models

This page documents the computer vision and multimodal models tested with TritonIC. See the [project README](../README.md) for quick-start instructions.

## Tested Models

### Object Detection

- [YOLOv5](https://github.com/ultralytics/yolov5)
- [YOLOv6](https://github.com/meituan/YOLOv6)
- [YOLOv7](https://github.com/WongKinYiu/yolov7)
- [YOLOv8/YOLO11/YOLO26](https://github.com/ultralytics/ultralytics)
- [YOLOv9](https://github.com/WongKinYiu/yolov9)
- [YOLOv10](https://github.com/THU-MIG/yolov10)
- [YOLOv12](https://github.com/sunsmarterjie/yolov12)
- [YOLO-NAS](https://github.com/Deci-AI/super-gradients)
- [RT-DETR](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetr_pytorch)
- [RT-DETRv2](https://github.com/lyuwenyu/RT-DETR/tree/main/rtdetrv2_pytorch)
- [RT-DETRv4](https://github.com/RT-DETRs/RT-DETRv4)
- [D-FINE](https://github.com/Peterande/D-FINE)
- [DEIM](https://github.com/ShihuaHuang95/DEIM)
- [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)
- [RF-DETR](https://github.com/roboflow/rf-detr)

### Instance Segmentation

- [YOLOv5](https://github.com/ultralytics/yolov5)
- [YOLOv8/YOLO11/YOLO26](https://github.com/ultralytics/ultralytics)
- [YOLOv10](https://github.com/THU-MIG/yolov10)
- [YOLOv12](https://github.com/sunsmarterjie/yolov12)
- [RF-DETR-Seg](https://github.com/roboflow/rf-detr)

### Classification

- [Torchvision Models](https://pytorch.org/vision/stable/models.html)
- [TensorFlow-Keras Models](https://www.tensorflow.org/api_docs/python/tf/keras/applications)
- [Hugging Face Vision Transformers (ViT)](https://huggingface.co/docs/transformers/model_doc/vit)

### Optical Flow

- [RAFT](https://pytorch.org/vision/stable/models/raft.html)

### Open Vocabulary Detection

- [OWLv2](https://huggingface.co/google/owlv2-base-patch16-ensemble)
- [OWL-ViT](https://huggingface.co/google/owlvit-base-patch32)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)

### Pose Estimation

- [YOLOv5 Pose](https://github.com/ultralytics/yolov5)
- [YOLOv8/YOLO11/YOLO26 Pose](https://github.com/ultralytics/ultralytics)
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
- [RF-DETR Keypoints](https://github.com/roboflow/rf-detr) (single-stage person keypoints, 17 COCO keypoints)

### Video Classification

- [VideoMAE](https://github.com/MCG-NJU/VideoMAE)
- [ViViT](https://github.com/google-research/scenic/tree/main/scenic/projects/vivit)
- [TimeSformer](https://github.com/facebookresearch/TimeSformer)

### Depth Estimation

- [Depth Anything V2](https://github.com/ibaiGorordo/Depth-Anything-V2)

### Image Understanding (VLM)

- [Gemma 4](https://ai.google.dev/gemma/docs) and compatible vision-language models via llama.cpp (image captioning, visual Q&A)
- LLaVA, LLaMA3-V, and other multimodal models via OpenAI-compatible endpoints

## Model Type Tags

| Task | `--model_type` | Notes |
|------|----------------|-------|
| **Object Detection** | `yolo` | YOLOv5–v9, v11, v12 |
| | `yolov7e2e` | YOLOv7 `--grid --end2end` (TRT only) |
| | `yolov10`, `yolo26` | Same output format |
| | `yolonas` | YOLO-NAS |
| | `yolov4` | YOLOv4 |
| | `rtdetr` | RT-DETR v1/v2/v4, D-FINE, DEIM v1/v2 |
| | `rtdetrul` | RT-DETR (Ultralytics) |
| | `rfdetr` | RF-DETR |
| **Instance Segmentation** | `yoloseg` | YOLOv5/v8/v11/v12 seg |
| | `yolov10seg` | YOLOv10 seg |
| | `yolo26seg` | YOLO26 seg |
| | `rfdetrseg` | RF-DETR seg |
| **Classification** | `torchvision-classifier` | ResNet, EfficientNet, etc. |
| | `tensorflow-classifier` | TF/Keras models |
| | `vit-classifier` | Vision Transformers |
| **Video Classification** | `videomae` | VideoMAE (16-frame default) |
| | `vivit` | ViViT |
| | `timesformer` | TimeSformer |
| **Optical Flow** | `raft` | Pass two images as `img1.jpg,img2.jpg` |
| **Pose Estimation** | `vitpose` | ViTPose (COCO 17 keypoints) |
| **Depth Estimation** | `depth_anything_v2` | Depth Anything V2 |
| **Open Vocabulary Detection** | `owlv2` | OWLv2 open-vocabulary detection |
| | `owlvit` | OWL-ViT open-vocabulary detection |

`TaskFactory::normalizeModelType()` strips spaces, hyphens, underscores and lowercases before matching.

## Tasks

### Export Instructions
- [Object Detection](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/detection/ObjectDetection.md)
- [Classification](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/classification/Classification.md)
- [Instance Segmentation](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/segmentation/InstanceSegmentation.md)
- [Optical Flow](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/optical_flow/OpticalFlow.md)
- [Open Vocabulary Detection](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/open_vocab_detection/OpenVocabDetection.md)
- [Pose Estimation](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/pose_estimation/PoseEstimation.md)
- [Video Classification](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/video_classification/VideoClassification.md)
- [Depth Estimation](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/depth_estimation/DepthEstimation.md)

*Other tasks are in TODO list.*

## Notes

Ensure the model export versions match those supported by your Triton release. Check Triton releases [here](https://github.com/triton-inference-server/server/releases).

# YOLO26 Object Detection — Triton Deployment

YOLO26 object detection with server-side DALI GPU preprocessing and CUDA-accelerated
bbox decoding.  For segmentation, see
[YOLO26 instance seg](../../../instance_segmentation/yolo26/README.md).

For model export, see [neuriplo-tasks export documentation](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/README.md).

## Ensemble

The [DALI/TensorRT ensemble](ensemble/README.md) runs the GPU pipeline:
DALI JPEG decode + resize → TensorRT inference → CUDA bbox decode.
Two variants: raw TensorRT (client postprocesses) and GPU postprocess (decoded
bboxes returned directly).

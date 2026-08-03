# YOLO26 Instance Segmentation — Triton Deployment

YOLO26 instance segmentation with server-side DALI GPU preprocessing and
CUDA-accelerated postprocessing (mask and polygon output).

For model export, see [neuriplo-tasks export documentation](https://github.com/olibartfast/neuriplo-tasks/blob/master/export/README.md).

## Ensemble

The [DALI/TensorRT ensemble](ensemble/README.md) runs the full GPU pipeline:
DALI JPEG decode + resize → TensorRT inference → CUDA mask or polygon
postprocess.  Four ensemble variants available (raw TensorRT, CPU post, GPU
mask post, GPU polygon post).

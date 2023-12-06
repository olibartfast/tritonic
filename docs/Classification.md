## Instructions

### Export the model for the inference
## Models from Torchvision Pytorch API

* Select a model from https://pytorch.org/vision/stable/models.html#classification, for example resnet50, then write a python script like below:

#### OnnxRuntime
 ```python
import torch
import torchvision.models as models
import argparse
import os
resnet50 = models.resnet50(pretrained=True)
dummy_input = torch.randn(1, 3, 224, 224)
resnet50 = resnet50.eval()

torch.onnx.export(resnet50,
                    dummy_input,
                    args.save,
                    export_params=True,
                    opset_version=10,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={'input': {0: 'batch_size', 2: "height", 3: 'width'},
                                'output': {0: 'batch_size'}})
  ```

 #### TensorRT
 Once you have your exported onnx model, using trtexec:
 ```bash
   trtexec --onnx=model.onnx --saveEngine=model.plan --explicitBatch --minShapes=input:1x3x224x224 --optShapes=input:1x3x224x224 --maxShapes=input:256x3x224x224

  ```

  * Pay attention to use tensorrt version corresponding to the one used on Triton image, or you could use a container like below (supposing you are using Triton release 23.08)
   ```bash
  docker run -it --gpus=all -v $(pwd):/workspace nvcr.io/nvidia/pytorch:23.08-py3 /bin/bash -cx \
   "trtexec --onnx=model.onnx --saveEngine=model.plan --explicitBatch --minShapes=input:1x3x224x224 --optShapes=input:1x3x224x224 --maxShapes=input:256x3x224x224 --fp16 
  ```  
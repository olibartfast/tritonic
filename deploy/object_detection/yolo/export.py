#!/usr/bin/env python3
"""
YOLO Universal Model Export Script

Export YOLO models (v5, v6, v7, v8, v9, v10, v11, v12, NAS) to ONNX and TensorRT formats.
Supports all major YOLO variants from different repositories.

Usage:
    # Ultralytics models (v5, v8, v9, v10, v11, v12)
    python export.py --model yolov8n.pt --format onnx
    python export.py --model yolo11s.pt --format onnx --imgsz 640
    
    # YOLOv6 (Meituan)
    python export.py --model yolov6s.pt --version v6 --format onnx
    
    # YOLOv7 (WongKinYiu)
    python export.py --model yolov7.pt --version v7 --format onnx
    
    # YOLO-NAS (Deci)
    python export.py --model yolo_nas_s --version nas --format onnx
    
    # Export with custom input size
    python export.py --model yolov8n.pt --format onnx --imgsz 640 --batch-size 1
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import logging
import shutil

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_virtual_environment():
    """Check if running in a virtual environment and warn if not."""
    in_venv = False
    venv_info = ""
    
    if hasattr(sys, 'real_prefix'):
        in_venv = True
        venv_info = "virtualenv"
    elif hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        in_venv = True
        venv_info = "venv/virtualenv"
    elif os.environ.get('VIRTUAL_ENV'):
        in_venv = True
        venv_info = f"VIRTUAL_ENV: {os.environ['VIRTUAL_ENV']}"
    elif os.environ.get('CONDA_DEFAULT_ENV'):
        in_venv = True
        venv_info = f"conda: {os.environ['CONDA_DEFAULT_ENV']}"
    elif os.environ.get('CONDA_PREFIX'):
        in_venv = True
        venv_info = f"conda prefix: {os.environ['CONDA_PREFIX']}"
    
    if in_venv:
        logger.info(f"✓ Running in virtual environment: {venv_info}")
    else:
        logger.warning("⚠️  Not running in a virtual environment!")
        logger.warning("   It's recommended to use a virtual environment to avoid dependency conflicts.")
        logger.warning("   Consider creating one with: conda create -n yolo python=3.11")
    
    return in_venv


class YOLOExporter:
    """Universal YOLO model exporter for ONNX and TensorRT formats."""
    
    # Version to repository mapping
    REPO_INFO = {
        'v5': {
            'url': 'https://github.com/ultralytics/yolov5.git',
            'package': None,  # Requires repo clone, not pip ultralytics
            'export_method': 'yolov5'
        },
        'v6': {
            'url': 'https://github.com/meituan/YOLOv6.git',
            'package': None,
            'export_method': 'yolov6'
        },
        'v7': {
            'url': 'https://github.com/WongKinYiu/yolov7.git',
            'package': None,
            'export_method': 'yolov7'
        },
        'v8': {
            'url': None,  # pip install ultralytics
            'package': 'ultralytics',
            'export_method': 'ultralytics'
        },
        'v9': {
            'url': None,
            'package': 'ultralytics',
            'export_method': 'ultralytics'
        },
        'v10': {
            'url': None,
            'package': 'ultralytics',
            'export_method': 'ultralytics'
        },
        'v11': {
            'url': None,
            'package': 'ultralytics',
            'export_method': 'ultralytics'
        },
        'v12': {
            'url': None,
            'package': 'ultralytics',
            'export_method': 'ultralytics'
        },
        'nas': {
            'url': None,
            'package': 'super-gradients',
            'export_method': 'yolo_nas'
        }
    }
    
    # YOLOv5 weight URLs (original repo)
    YOLOV5_WEIGHTS = {
        'yolov5n': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt',
        'yolov5s': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt',
        'yolov5m': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt',
        'yolov5l': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt',
        'yolov5x': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt',
        'yolov5n6': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n6.pt',
        'yolov5s6': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s6.pt',
        'yolov5m6': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m6.pt',
        'yolov5l6': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l6.pt',
        'yolov5x6': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x6.pt',
    }
    
    # Ultralytics model weight URLs (v8+)
    ULTRALYTICS_WEIGHTS = {
        # YOLOv8
        'yolov8n': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.pt',
        'yolov8s': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt',
        'yolov8m': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt',
        'yolov8l': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8l.pt',
        'yolov8x': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x.pt',
        # YOLOv9
        'yolov9t': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9t.pt',
        'yolov9s': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9s.pt',
        'yolov9m': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9m.pt',
        'yolov9c': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9c.pt',
        'yolov9e': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov9e.pt',
        # YOLOv10
        'yolov10n': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10n.pt',
        'yolov10s': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10s.pt',
        'yolov10m': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10m.pt',
        'yolov10l': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10l.pt',
        'yolov10x': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10x.pt',
        # YOLO11
        'yolo11n': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt',
        'yolo11s': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt',
        'yolo11m': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt',
        'yolo11l': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt',
        'yolo11x': 'https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x.pt',
        # YOLOv12
        'yolov12n': 'https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12n.pt',
        'yolov12s': 'https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12s.pt',
        'yolov12m': 'https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12m.pt',
        'yolov12l': 'https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12l.pt',
        'yolov12x': 'https://github.com/sunsmarterjie/yolov12/releases/download/v1.0/yolov12x.pt',
    }
    
    # YOLOv6 weight URLs
    YOLOV6_WEIGHTS = {
        'yolov6n': 'https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6n.pt',
        'yolov6s': 'https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s.pt',
        'yolov6m': 'https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6m.pt',
        'yolov6l': 'https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6l.pt',
    }
    
    # YOLOv7 weight URLs
    YOLOV7_WEIGHTS = {
        'yolov7': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt',
        'yolov7-tiny': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-tiny.pt',
        'yolov7x': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt',
        'yolov7-w6': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt',
        'yolov7-e6': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt',
        'yolov7-d6': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt',
        'yolov7-e6e': 'https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt',
    }
    
    def __init__(self, model_path: str, version: str = 'auto', output_dir: str = "./exported_models",
                 imgsz: int = 640, batch_size: int = 1):
        self.model_path = model_path
        self.version = version
        self.output_dir = Path(output_dir)
        self.imgsz = imgsz
        self.batch_size = batch_size
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Auto-detect version if needed
        if self.version == 'auto':
            self.version = self._detect_version()
        
        logger.info(f"Detected YOLO version: {self.version}")
    
    def _detect_version(self) -> str:
        """Auto-detect YOLO version from model path."""
        model_name = Path(self.model_path).stem.lower()
        
        if 'yolo_nas' in model_name or model_name.startswith('yolo_nas'):
            return 'nas'
        elif 'yolov12' in model_name:
            return 'v12'
        elif 'yolo11' in model_name:
            return 'v11'
        elif 'yolov10' in model_name:
            return 'v10'
        elif 'yolov9' in model_name:
            return 'v9'
        elif 'yolov8' in model_name:
            return 'v8'
        elif 'yolov7' in model_name:
            return 'v7'
        elif 'yolov6' in model_name:
            return 'v6'
        elif 'yolov5' in model_name:
            return 'v5'
        else:
            # Default to ultralytics for unknown models
            logger.warning(f"Could not auto-detect version for {model_name}, defaulting to v8 (ultralytics)")
            return 'v8'
    
    def download_weights(self, weights_dir: str = "./weights") -> str:
        """Download model weights if not present."""
        weights_path = Path(weights_dir)
        weights_path.mkdir(exist_ok=True, parents=True)
        
        model_name = Path(self.model_path).stem.lower()
        
        # Determine weight URL based on version
        weight_url = None
        if self.version == 'v5':
            weight_url = self.YOLOV5_WEIGHTS.get(model_name)
        elif self.version in ['v8', 'v9', 'v10', 'v11', 'v12']:
            weight_url = self.ULTRALYTICS_WEIGHTS.get(model_name)
        elif self.version == 'v6':
            weight_url = self.YOLOV6_WEIGHTS.get(model_name)
        elif self.version == 'v7':
            weight_url = self.YOLOV7_WEIGHTS.get(model_name)
        
        if not weight_url:
            logger.warning(f"No download URL found for {model_name}")
            return self.model_path
        
        output_path = weights_path / f"{model_name}.pt"
        
        if output_path.exists():
            logger.info(f"Weights already exist: {output_path}")
            return str(output_path)
        
        logger.info(f"Downloading weights from {weight_url}")
        try:
            import urllib.request
            urllib.request.urlretrieve(weight_url, output_path)
            logger.info(f"Downloaded weights to {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"Failed to download weights: {e}")
            return self.model_path
    
    def export_onnx(self, simplify: bool = True, opset: int = 12, dynamic: bool = False, repo_dir: str = None) -> str:
        """Export model to ONNX format."""
        export_method = self.REPO_INFO[self.version]['export_method']
        
        if export_method == 'ultralytics':
            return self._export_ultralytics_onnx(simplify, opset, dynamic)
        elif export_method == 'yolov5':
            return self._export_yolov5_onnx(simplify, opset, dynamic, repo_dir)
        elif export_method == 'yolov6':
            return self._export_yolov6_onnx(simplify, opset, dynamic, repo_dir)
        elif export_method == 'yolov7':
            return self._export_yolov7_onnx(simplify, opset, dynamic, repo_dir)
        elif export_method == 'yolo_nas':
            return self._export_yolo_nas_onnx(simplify, opset)
        else:
            raise ValueError(f"Unknown export method: {export_method}")
    
    def _export_ultralytics_onnx(self, simplify: bool, opset: int, dynamic: bool) -> str:
        """Export using ultralytics library (v8, v9, v10, v11, v12)."""
        try:
            from ultralytics import YOLO
        except ImportError:
            logger.error("ultralytics not installed. Install with: pip install ultralytics")
            sys.exit(1)
        
        logger.info(f"Loading model from {self.model_path}")
        model = YOLO(self.model_path)
        
        export_args = {
            'format': 'onnx',
            'imgsz': self.imgsz,
            'batch': self.batch_size,
            'simplify': simplify,
            'opset': opset,
            'dynamic': dynamic,
        }
        
        logger.info(f"Exporting to ONNX with args: {export_args}")
        output_path = model.export(**export_args)
        
        # Move to output directory
        if output_path and Path(output_path).exists():
            dest_path = self.output_dir / Path(output_path).name
            shutil.move(output_path, dest_path)
            logger.info(f"ONNX model exported to: {dest_path}")
            return str(dest_path)
        
        return output_path
    
    def _export_yolov5_onnx(self, simplify: bool, opset: int, dynamic: bool, repo_dir: str = None) -> str:
        """Export YOLOv5 to ONNX using the original ultralytics/yolov5 repository.
        
        Note: The 'ultralytics' pip package (for v8+) is NOT compatible with original YOLOv5.
        You must clone the yolov5 repository and run export from there.
        """
        try:
            import torch
        except ImportError:
            logger.error("torch not installed")
            sys.exit(1)
        
        model_name = Path(self.model_path).stem
        
        # Check if we have a repo directory
        if repo_dir and Path(repo_dir).exists():
            # Resolve repo_dir to absolute path
            repo_dir = Path(repo_dir).resolve()
            
            # Use the repo's export.py
            export_script = repo_dir / 'export.py'
            if not export_script.exists():
                logger.error(f"export.py not found in {repo_dir}")
                logger.info("Clone the repo with: ./clone_repo.sh --version v5")
                sys.exit(1)
            
            # Build command to run from repo directory (use relative path since cwd=repo_dir)
            cmd = [
                sys.executable, 'export.py',
                '--weights', str(Path(self.model_path).absolute()),
                '--img-size', str(self.imgsz),
                '--batch-size', str(self.batch_size),
                '--include', 'onnx',
                '--opset', str(opset),
            ]
            
            if simplify:
                cmd.append('--simplify')
            if dynamic:
                cmd.append('--dynamic')
            
            logger.info(f"Running YOLOv5 export from repository: {repo_dir}")
            logger.info(f"Command: {' '.join(cmd)}")
            
            # Run from repo directory
            result = subprocess.run(cmd, cwd=str(repo_dir), check=True)
            
            # Find the exported ONNX file (YOLOv5 exports next to the weights file)
            expected_onnx = Path(self.model_path).absolute().with_suffix('.onnx')
            if expected_onnx.exists():
                dest_path = self.output_dir / expected_onnx.name
                shutil.move(str(expected_onnx), str(dest_path))
                logger.info(f"ONNX model exported to: {dest_path}")
                return str(dest_path)
            
            return str(expected_onnx)
        
        else:
            # Try to load and export using torch.hub (downloads repo automatically)
            logger.info("Loading YOLOv5 via torch.hub...")
            try:
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=self.model_path, force_reload=False)
                
                # Export via torch
                output_path = self.output_dir / f"{model_name}.onnx"
                dummy_input = torch.randn(self.batch_size, 3, self.imgsz, self.imgsz)
                
                dynamic_axes = None
                if dynamic:
                    dynamic_axes = {
                        'images': {0: 'batch'},
                        'output': {0: 'batch'}
                    }
                
                torch.onnx.export(
                    model.model,
                    dummy_input,
                    str(output_path),
                    opset_version=opset,
                    input_names=['images'],
                    output_names=['output'],
                    dynamic_axes=dynamic_axes,
                )
                
                if simplify:
                    self._simplify_onnx(str(output_path))
                
                logger.info(f"ONNX model exported to: {output_path}")
                return str(output_path)
                
            except Exception as e:
                logger.error(f"torch.hub export failed: {e}")
                logger.info("")
                logger.info("For YOLOv5 export, clone the repository first:")
                logger.info("  ./clone_repo.sh --version v5 --output-dir ./repositories")
                logger.info("")
                logger.info("Then run export with --repo-dir:")
                logger.info(f"  python export.py --model {self.model_path} --version v5 --repo-dir ./repositories/yolov5")
                sys.exit(1)
    
    def _export_yolov6_onnx(self, simplify: bool, opset: int, dynamic: bool, repo_dir: str = None) -> str:
        """Export YOLOv6 to ONNX."""
        model_name = Path(self.model_path).stem
        output_path = self.output_dir / f"{model_name}.onnx"
        
        if not repo_dir:
            logger.error("YOLOv6 requires repository clone for export")
            logger.info("Clone the repo with: ./clone_repo.sh --version v6")
            logger.info("Then run: python export.py --model weights.pt --version v6 --repo-dir ./repositories/YOLOv6")
            sys.exit(1)
        
        # YOLOv6 uses its own export script
        repo_dir = Path(repo_dir).resolve()
        export_script = repo_dir / 'deploy' / 'ONNX' / 'export_onnx.py'
        if not export_script.exists():
            logger.error(f"Export script not found: {export_script}")
            sys.exit(1)
        
        # Use path relative to repo_dir for subprocess
        relative_export = 'deploy/ONNX/export_onnx.py'
        
        cmd = [
            sys.executable, relative_export,
            '--weights', str(Path(self.model_path).absolute()),
            '--img-size', str(self.imgsz),
            '--batch-size', str(self.batch_size),
        ]
        
        if simplify:
            cmd.append('--simplify')
        
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(repo_dir), check=True)
        
        return str(output_path)
    
    def _export_yolov7_onnx(self, simplify: bool, opset: int, dynamic: bool, repo_dir: str = None) -> str:
        """Export YOLOv7 to ONNX."""
        model_name = Path(self.model_path).stem
        output_path = self.output_dir / f"{model_name}.onnx"
        
        if not repo_dir:
            logger.error("YOLOv7 requires repository clone for export")
            logger.info("Clone the repo with: ./clone_repo.sh --version v7")
            logger.info("Then run: python export.py --model weights.pt --version v7 --repo-dir ./repositories/yolov7")
            sys.exit(1)
        
        repo_dir = Path(repo_dir).resolve()
        export_script = repo_dir / 'export.py'
        if not export_script.exists():
            logger.error(f"export.py not found in {repo_dir}")
            sys.exit(1)
        
        cmd = [
            sys.executable, 'export.py',
            '--weights', str(Path(self.model_path).absolute()),
            '--img-size', str(self.imgsz), str(self.imgsz),
            '--batch-size', str(self.batch_size),
            # NOTE: Removed --grid and --end2end flags for ONNX Runtime compatibility
            # These flags add TensorRT NMS plugin which requires TensorRT backend
            # Standard export produces [1, 25200, 85] format compatible with ONNX Runtime
        ]
        
        if simplify:
            cmd.append('--simplify')
        
        logger.info(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, cwd=str(repo_dir), check=True)
        
        # Find exported file (YOLOv7 exports next to weights file)
        expected_onnx = Path(self.model_path).absolute().with_suffix('.onnx')
        if expected_onnx.exists():
            dest_path = self.output_dir / expected_onnx.name
            shutil.move(str(expected_onnx), str(dest_path))
            return str(dest_path)
        
        return str(output_path)
    
    def _export_yolo_nas_onnx(self, simplify: bool, opset: int) -> str:
        """Export YOLO-NAS to ONNX."""
        try:
            from super_gradients.training import models
            from super_gradients.common.object_names import Models
            import torch
        except ImportError:
            logger.error("super-gradients not installed. Install with: pip install super-gradients")
            sys.exit(1)
        
        model_name = self.model_path.lower()
        output_path = self.output_dir / f"{model_name}.onnx"
        
        # Map model names to SuperGradients model names
        model_map = {
            'yolo_nas_s': Models.YOLO_NAS_S,
            'yolo_nas_m': Models.YOLO_NAS_M,
            'yolo_nas_l': Models.YOLO_NAS_L,
        }
        
        sg_model_name = model_map.get(model_name)
        if not sg_model_name:
            logger.error(f"Unknown YOLO-NAS model: {model_name}")
            logger.info(f"Available models: {list(model_map.keys())}")
            sys.exit(1)
        
        logger.info(f"Loading YOLO-NAS model: {sg_model_name}")
        model = models.get(sg_model_name, pretrained_weights="coco")
        
        # Export to ONNX
        logger.info(f"Exporting to ONNX: {output_path}")
        model.export(
            str(output_path),
            input_image_shape=(self.imgsz, self.imgsz),
            batch_size=self.batch_size,
        )
        
        if simplify:
            self._simplify_onnx(str(output_path))
        
        logger.info(f"ONNX model exported to: {output_path}")
        return str(output_path)
    
    def _simplify_onnx(self, onnx_path: str):
        """Simplify ONNX model using onnxsim."""
        try:
            import onnx
            from onnxsim import simplify
            
            logger.info(f"Simplifying ONNX model: {onnx_path}")
            model = onnx.load(onnx_path)
            model_simplified, check = simplify(model)
            
            if check:
                onnx.save(model_simplified, onnx_path)
                logger.info("ONNX model simplified successfully")
            else:
                logger.warning("ONNX simplification check failed, keeping original")
        except ImportError:
            logger.warning("onnxsim not installed, skipping simplification")
        except Exception as e:
            logger.warning(f"ONNX simplification failed: {e}")
    
    def export_tensorrt(self, fp16: bool = True, workspace_size: str = "4g") -> str:
        """Export model to TensorRT format."""
        # First export to ONNX
        onnx_path = self.export_onnx(simplify=True)
        
        if not onnx_path or not Path(onnx_path).exists():
            logger.error("ONNX export failed, cannot convert to TensorRT")
            return None
        
        model_name = Path(onnx_path).stem
        trt_path = self.output_dir / f"{model_name}.engine"
        
        # Build TensorRT engine using trtexec
        cmd = [
            'trtexec',
            f'--onnx={onnx_path}',
            f'--saveEngine={trt_path}',
            f'--workspace={self._parse_size(workspace_size)}',
        ]
        
        if fp16:
            cmd.append('--fp16')
        
        logger.info(f"Building TensorRT engine: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
            logger.info(f"TensorRT engine saved to: {trt_path}")
            return str(trt_path)
        except subprocess.CalledProcessError as e:
            logger.error(f"TensorRT conversion failed: {e}")
            return None
        except FileNotFoundError:
            logger.error("trtexec not found. Please install TensorRT.")
            return None
    
    def _parse_size(self, size_str: str) -> int:
        """Parse size string (e.g., '4g', '512m') to bytes."""
        size_str = size_str.lower().strip()
        if size_str.endswith('g'):
            return int(size_str[:-1]) * 1024 * 1024 * 1024
        elif size_str.endswith('m'):
            return int(size_str[:-1]) * 1024 * 1024
        elif size_str.endswith('k'):
            return int(size_str[:-1]) * 1024
        return int(size_str)
    
    def get_model_info(self):
        """Display model information (FLOPs, parameters)."""
        if self.version in ['v8', 'v9', 'v10', 'v11', 'v12']:
            try:
                from ultralytics import YOLO
                model = YOLO(self.model_path)
                model.info()
            except Exception as e:
                logger.error(f"Failed to get model info: {e}")
        else:
            logger.info("Model info only available for Ultralytics models (v8+)")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO Universal Model Export Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Export YOLOv8 to ONNX
    python export.py --model yolov8n.pt --format onnx
    
    # Export YOLO11 with custom size
    python export.py --model yolo11s.pt --format onnx --imgsz 640
    
    # Export YOLOv5 (requires repo clone)
    python export.py --model yolov5s.pt --version v5 --repo-dir ./repositories/yolov5 --format onnx
    
    # Export YOLOv7 (requires repo clone)
    python export.py --model yolov7.pt --version v7 --repo-dir ./repositories/yolov7 --format onnx
    
    # Export YOLO-NAS
    python export.py --model yolo_nas_s --version nas --format onnx
    
    # Export with auto-download
    python export.py --model yolov8n --download-weights --format onnx
        """
    )
    
    # Required arguments
    parser.add_argument('--model', '-m', required=True,
                        help='Path to model weights or model name')
    
    # Version selection
    parser.add_argument('--version', '-v', default='auto',
                        choices=['auto', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10', 'v11', 'v12', 'nas'],
                        help='YOLO version (default: auto-detect)')
    
    # Export options
    parser.add_argument('--format', '-f', default='onnx',
                        choices=['onnx', 'tensorrt', 'both'],
                        help='Export format (default: onnx)')
    parser.add_argument('--output-dir', '-o', default='./exported_models',
                        help='Output directory (default: ./exported_models)')
    
    # Repository options (for v5, v6, v7)
    parser.add_argument('--repo-dir', default=None,
                        help='Path to cloned YOLO repository (required for v5, v6, v7)')
    
    # Model options
    parser.add_argument('--imgsz', '--img-size', type=int, default=640,
                        help='Input image size (default: 640)')
    parser.add_argument('--batch-size', '-b', type=int, default=1,
                        help='Batch size (default: 1)')
    
    # ONNX options
    parser.add_argument('--no-simplify', action='store_true',
                        help='Skip ONNX simplification')
    parser.add_argument('--opset', type=int, default=12,
                        help='ONNX opset version (default: 12)')
    parser.add_argument('--dynamic', action='store_true',
                        help='Enable dynamic batch size')
    
    # TensorRT options
    parser.add_argument('--no-fp16', action='store_true',
                        help='Disable FP16 precision for TensorRT')
    parser.add_argument('--workspace-size', default='4g',
                        help='TensorRT workspace size (default: 4g)')
    
    # Weight download
    parser.add_argument('--download-weights', action='store_true',
                        help='Download model weights if not present')
    parser.add_argument('--weights-dir', default='./weights',
                        help='Directory to download weights to')
    
    # Utility options
    parser.add_argument('--model-info', action='store_true',
                        help='Display model information')
    parser.add_argument('--skip-venv-check', action='store_true',
                        help='Skip virtual environment check')
    
    args = parser.parse_args()
    
    # Check virtual environment
    if not args.skip_venv_check:
        check_virtual_environment()
    
    # Initialize exporter
    exporter = YOLOExporter(
        model_path=args.model,
        version=args.version,
        output_dir=args.output_dir,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
    )
    
    # Download weights if requested
    if args.download_weights:
        args.model = exporter.download_weights(args.weights_dir)
        exporter.model_path = args.model
    
    # Display model info if requested
    if args.model_info:
        exporter.get_model_info()
        return
    
    # Export model
    if args.format in ['onnx', 'both']:
        onnx_path = exporter.export_onnx(
            simplify=not args.no_simplify,
            opset=args.opset,
            dynamic=args.dynamic,
            repo_dir=args.repo_dir,
        )
        if onnx_path:
            logger.info(f"✓ ONNX export complete: {onnx_path}")
    
    if args.format in ['tensorrt', 'both']:
        trt_path = exporter.export_tensorrt(
            fp16=not args.no_fp16,
            workspace_size=args.workspace_size,
        )
        if trt_path:
            logger.info(f"✓ TensorRT export complete: {trt_path}")


if __name__ == '__main__':
    main()

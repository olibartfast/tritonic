#!/usr/bin/env python3
"""
RT-DETR Universal Model Export Script

Export RT-DETR models (v1, v2, v3, v4) to ONNX and TensorRT formats for deployment.
Supports all RT-DETR model variants and versions including DEIM and D-FINE.
Can automatically download model weights for supported RT-DETRv4 configs.

Usage:
    # Basic export with existing checkpoint
    python export.py --config configs/rtv4/rtv4_hgnetv2_s_coco.yml --checkpoint model.pth --format onnx
    
    # Auto-download weights and export
    python export.py --config configs/rtv4/rtv4_hgnetv2_s_coco.yml --checkpoint model.pth --download-weights --format onnx
    
    # Complete setup with repo cloning, dependency installation, and weight download
    python export.py --config configs/rtv4/rtv4_hgnetv2_s_coco.yml --checkpoint model.pth --clone-repo --install-deps --download-weights --format onnx
    
    # Other RT-DETR versions
    python export.py --config configs/rtv2/rtv2_r50_coco.yml --checkpoint model.pth --format tensorrt
    python export.py --config configs/rtdetrv3/rtdetrv3_r18vd_6x_coco.yml --checkpoint model.pdparams --format onnx
    python export.py --config configs/dfine/dfine_hgnetv2_l_coco.yml --checkpoint model.pth --format both
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
import logging
import urllib.request
import urllib.parse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_virtual_environment():
    """Check if running in a virtual environment and warn if not."""
    in_venv = False
    venv_info = ""
    
    # Check for various virtual environment indicators
    if hasattr(sys, 'real_prefix'):
        # Old virtualenv
        in_venv = True
        venv_info = "virtualenv"
    elif hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix:
        # New virtualenv or venv
        in_venv = True
        venv_info = "venv/virtualenv"
    elif os.environ.get('VIRTUAL_ENV'):
        # Virtual environment variable set
        in_venv = True
        venv_info = f"VIRTUAL_ENV: {os.environ['VIRTUAL_ENV']}"
    elif os.environ.get('CONDA_DEFAULT_ENV'):
        # Conda environment
        in_venv = True
        venv_info = f"conda: {os.environ['CONDA_DEFAULT_ENV']}"
    elif os.environ.get('CONDA_PREFIX'):
        # Conda environment (alternative check)
        in_venv = True
        venv_info = f"conda prefix: {os.environ['CONDA_PREFIX']}"
    
    if in_venv:
        logger.info(f"✓ Running in virtual environment: {venv_info}")
    else:
        logger.warning("⚠️  Not running in a virtual environment!")
        logger.warning("   It's recommended to use a virtual environment to avoid dependency conflicts.")
        logger.warning("   Consider creating one with: conda create -n rtv4 python=3.11.9")
    
    return in_venv


class RTDETRExporter:
    """Universal RT-DETR model exporter for ONNX and TensorRT formats (v1/v2/v3/v4)."""
    
    def __init__(self, config_path: str, checkpoint_path: str, output_dir: str = "./exported_models", repo_dir: str = ".", version: str = "v4"):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.output_dir = Path(output_dir)
        self.repo_dir = Path(repo_dir)
        self.version = version
        self.output_dir.mkdir(exist_ok=True)
        
        # Validate inputs
        if not Path(config_path).exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        if not self.repo_dir.exists():
            raise FileNotFoundError(f"Repository directory not found: {repo_dir}")
    
    def _is_paddlepaddle_version(self) -> bool:
        """Check if this is a PaddlePaddle-based RT-DETR version (v3)."""
        return self.version == 'v3'
    
    def _patch_export_batch_size(self):
        """Runtime patch to fix hardcoded batch size 32 -> 1 in export_onnx.py to prevent OOM."""
        export_script_path = self.repo_dir / "tools" / "deployment" / "export_onnx.py"
        
        if not export_script_path.exists():
            logger.warning(f"Export script not found for patching: {export_script_path}")
            return
            
        try:
            # Read the original file
            with open(export_script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check if already patched
            if 'torch.rand(1, 3, 640, 640)' in content:
                logger.info("Export script already patched for batch size")
                return
            
            # Patch the hardcoded batch size 32 -> 1 in multiple places
            patches_applied = []
            
            # Main data generation line
            original_line1 = 'data = torch.rand(32, 3, 640, 640)'
            patched_line1 = 'data = torch.rand(1, 3, 640, 640)'
            
            if original_line1 in content:
                content = content.replace(original_line1, patched_line1)
                patches_applied.append("data generation")
            
            # Simplify step hardcoded shape (if it exists)
            original_line2 = "'images': [32, 3, 640, 640]"
            patched_line2 = "'images': [1, 3, 640, 640]"
            
            if original_line2 in content:
                content = content.replace(original_line2, patched_line2)
                patches_applied.append("simplify input shapes")
            
            if patches_applied:
                # Write back the patched content
                with open(export_script_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info(f"Successfully patched export_onnx.py: {', '.join(patches_applied)}")
            else:
                logger.warning("Could not find expected batch size patterns in export_onnx.py")
                
        except Exception as e:
            logger.warning(f"Failed to patch export script: {e}")
            logger.warning("Export may fail with OOM if batch size 32 is too large")
    
    def export_onnx(self, check_model: bool = False, simplify: bool = True) -> str:
        """
        Export model to ONNX format.
        
        Args:
            check_model: Whether to check the exported ONNX model
            simplify: Whether to simplify the ONNX model using onnxsim
            
        Returns:
            Path to exported ONNX model
        """
        logger.info("Exporting RT-DETR model to ONNX format...")
        
        # Extract model name from config path for output filename
        config_name = Path(self.config_path).stem
        onnx_path = self.output_dir / f"{config_name}.onnx"
        
        if self._is_paddlepaddle_version():
            return self._export_onnx_v3(onnx_path)
        else:
            return self._export_onnx_pytorch(onnx_path, check_model, simplify)
    
    def _export_onnx_pytorch(self, onnx_path: Path, check_model: bool, simplify: bool) -> str:
        """Export ONNX for PyTorch-based RT-DETR versions (v1, v2, v4)."""
        # Convert paths to absolute paths before changing directory
        abs_config_path = Path(self.config_path).resolve()
        abs_checkpoint_path = Path(self.checkpoint_path).resolve()
        
        # Runtime patch to fix hardcoded batch size 32 -> 1 to prevent OOM
        self._patch_export_batch_size()
        
        # Build export command
        cmd = [
            "python", "tools/deployment/export_onnx.py",
            "--config", str(abs_config_path),
            "--resume", str(abs_checkpoint_path)
        ]
        
        if check_model:
            cmd.append("--check")
        
        if simplify:
            cmd.append("--simplify")
        
        try:
            # Change to repository directory and run export command
            original_dir = os.getcwd()
            os.chdir(self.repo_dir)
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"ONNX export successful")
            logger.info(f"Export output: {result.stdout}")
            
            # The RT-DETR script generates model.onnx in the current directory (repo_dir)
            generated_onnx = Path("model.onnx")
            abs_output_path = Path(original_dir) / onnx_path
            
            if generated_onnx.exists():
                # Move to desired output location
                abs_output_path.parent.mkdir(parents=True, exist_ok=True)
                generated_onnx.rename(abs_output_path)
                logger.info(f"ONNX model moved to: {abs_output_path}")
                return str(abs_output_path)
            else:
                logger.warning(f"Expected ONNX file not found, check output directory")
                return str(abs_output_path)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"ONNX export failed: {e.stderr}")
            raise RuntimeError(f"ONNX export failed: {e}")
        finally:
            os.chdir(original_dir)
    
    def _export_onnx_v3(self, onnx_path: Path) -> str:
        """Export ONNX for RT-DETRv3 (PaddlePaddle-based) - Two-step process."""
        logger.info("RT-DETRv3 uses PaddlePaddle - performing two-step export process...")
        
        # Step 1: Export PaddlePaddle model
        inference_dir = self.output_dir / "paddle_inference"
        inference_dir.mkdir(exist_ok=True)
        
        cmd1 = [
            "python", "tools/export_model.py",
            "-c", self.config_path,
            "-o", f"weights={self.checkpoint_path}",
            "trt=True",
            f"--output_dir={inference_dir}"
        ]
        
        try:
            # Change to repository directory and run export command
            original_dir = os.getcwd()
            os.chdir(self.repo_dir)
            
            result1 = subprocess.run(cmd1, capture_output=True, text=True, check=True)
            logger.info("PaddlePaddle model export successful")
            
            # Step 2: Convert to ONNX using paddle2onnx
            config_name = Path(self.config_path).stem
            model_dir = inference_dir / config_name
            cmd2 = [
                "paddle2onnx",
                f"--model_dir={model_dir}",
                "--model_filename=model.pdmodel",
                "--params_filename=model.pdiparams",
                "--opset_version=16",
                f"--save_file={onnx_path}"
            ]
            
            result2 = subprocess.run(cmd2, capture_output=True, text=True, check=True)
            logger.info(f"ONNX conversion successful: {onnx_path}")
            logger.info("Note: Make sure paddle2onnx is installed (pip install paddle2onnx==1.0.5)")
            
            return str(onnx_path)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"RT-DETRv3 export failed: {e.stderr}")
            logger.error("Make sure PaddlePaddle and paddle2onnx are installed:")
            logger.error("pip install paddlepaddle-gpu")
            logger.error("pip install paddle2onnx==1.0.5")
            raise RuntimeError(f"RT-DETRv3 export failed: {e}")
        finally:
            os.chdir(original_dir)
    
    def export_tensorrt(self, onnx_path: str = None, fp16: bool = True, workspace_size: str = "4g") -> str:
        """
        Export model to TensorRT format.
        
        Args:
            onnx_path: Path to ONNX model (if None, will export ONNX first)
            fp16: Whether to use FP16 precision
            workspace_size: TensorRT workspace size
            
        Returns:
            Path to exported TensorRT engine
        """
        logger.info("Exporting RT-DETR model to TensorRT format...")
        
        # If no ONNX path provided, export ONNX first
        if onnx_path is None:
            onnx_path = self.export_onnx()
        
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        
        # Generate TensorRT engine path
        onnx_path = Path(onnx_path)
        trt_path = onnx_path.with_suffix('.engine')
        
        # Build TensorRT export command
        cmd = [
            "trtexec",
            f"--onnx={onnx_path}",
            f"--saveEngine={trt_path}",
            f"--workspace={workspace_size}"
        ]
        
        if fp16:
            cmd.append("--fp16")
        
        try:
            # Run TensorRT export
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"TensorRT export successful: {trt_path}")
            return str(trt_path)
            
        except subprocess.CalledProcessError as e:
            logger.error(f"TensorRT export failed: {e.stderr}")
            raise RuntimeError(f"TensorRT export failed. Make sure TensorRT is installed: {e}")
    
    def benchmark_model(self, model_path: str, format_type: str, coco_dir: str = None):
        """
        Benchmark the exported model.
        
        Args:
            model_path: Path to exported model
            format_type: Model format ('onnx' or 'tensorrt')
            coco_dir: Path to COCO dataset for evaluation
        """
        logger.info(f"Benchmarking {format_type.upper()} model...")
        
        if format_type.lower() == "tensorrt" and coco_dir:
            cmd = [
                "python", "tools/benchmark/trt_benchmark.py",
                "--COCO_dir", coco_dir,
                "--engine_dir", model_path
            ]
            
            try:
                # Change to repository directory for benchmark
                original_dir = os.getcwd()
                os.chdir(self.repo_dir)
                
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                logger.info(f"Benchmark results:\n{result.stdout}")
                
            except subprocess.CalledProcessError as e:
                logger.warning(f"Benchmarking failed: {e.stderr}")
            finally:
                os.chdir(original_dir)
        else:
            logger.info("Skipping benchmark - TensorRT format and COCO directory required")


def detect_rtdetr_version(config_path: str) -> str:
    """Detect RT-DETR version from config path."""
    config_path = Path(config_path)
    config_str = str(config_path).lower()
    
    # Check for version indicators in path
    if 'rtv4' in config_str or 'rt-detrv4' in config_str:
        return 'v4'
    elif 'rtv3' in config_str or 'rt-detrv3' in config_str or 'rtdetrv3' in config_str:
        return 'v3'
    elif 'rtv2' in config_str or 'rt-detrv2' in config_str:
        return 'v2'
    elif 'dfine' in config_str or 'd-fine' in config_str:
        return 'dfine'
    elif 'deim' in config_str:
        return 'deim'
    elif 'rtv1' in config_str or 'rt-detr' in config_str:
        return 'v1'
    
    # Default to v4 if can't detect
    logger.warning("Could not detect RT-DETR version from config path, defaulting to v4")
    return 'v4'


def get_repo_info(version: str) -> tuple:
    """Get repository URL and default directory name for RT-DETR version."""
    repo_configs = {
        'v1': ('https://github.com/lyuwenyu/RT-DETR.git', 'RT-DETR'),
        'v2': ('https://github.com/lyuwenyu/RT-DETR.git', 'RT-DETR'), 
        'v3': ('https://github.com/clxia12/RT-DETRv3.git', 'RT-DETRv3'),
        'v4': ('https://github.com/RT-DETRs/RT-DETRv4.git', 'RT-DETRv4'),
        'dfine': ('https://github.com/Peterande/D-FINE.git', 'D-FINE'),
        'deim': ('https://github.com/Intellindust-AI-Lab/DEIM.git', 'DEIM')
    }
    
    return repo_configs.get(version, repo_configs['v4'])


def clone_rtdetr_repo(target_dir: str = None, version: str = 'v4') -> str:
    """Clone the appropriate RT-DETR repository if it doesn't exist."""
    repo_url, default_name = get_repo_info(version)
    
    if target_dir is None:
        target_dir = f"./{default_name}"
    
    target_path = Path(target_dir)
    
    if target_path.exists() and (target_path / ".git").exists():
        logger.info(f"RT-DETR repository already exists at: {target_path}")
        return str(target_path)
    
    logger.info(f"Cloning RT-DETR {version.upper()} repository...")
    
    try:
        cmd = [
            "git", "clone",
            repo_url,
            str(target_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Successfully cloned RT-DETR {version.upper()} to: {target_path}")
        return str(target_path)
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to clone repository: {e.stderr}")
        raise RuntimeError(f"Repository cloning failed: {e}")


def setup_environment(repo_dir: str):
    """Setup the RT-DETRv4 environment and install dependencies."""
    repo_path = Path(repo_dir)
    requirements_file = repo_path / "requirements.txt"
    
    if not requirements_file.exists():
        logger.warning(f"Requirements file not found at: {requirements_file}")
        return
    
    logger.info("Installing RT-DETRv4 requirements...")
    
    try:
        cmd = ["pip", "install", "-r", str(requirements_file)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Requirements installed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to install requirements: {e.stderr}")
        logger.warning("You may need to install dependencies manually")


def get_model_info(config_path: str, repo_dir: str = "."):
    """Get model information like FLOPs, MACs, and parameters."""
    try:
        # Change to repo directory for the command
        original_dir = os.getcwd()
        os.chdir(repo_dir)
        
        cmd = [
            "python", "tools/benchmark/get_info.py",
            "-c", config_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Model info:\n{result.stdout}")
        
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to get model info: {e.stderr}")
    finally:
        os.chdir(original_dir)


def get_model_weights_info() -> dict:
    """Get predefined model weights download URLs and info."""
    return {
        'rtv4_hgnetv2_s_coco.yml': {
            'url': 'https://drive.usercontent.google.com/download?id=1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z&export=download&confirm=t',
            'filename': 'rtv4_hgnetv2_s_model.pth',
            'size': '161M'
        },
        'rtv4_hgnetv2_m_coco.yml': {
            'url': 'https://drive.usercontent.google.com/download?id=1O-YpP4X-quuOXbi96y2TKkztbjroP5mX&export=download&confirm=t',
            'filename': 'rtv4_hgnetv2_m_model.pth',
            'size': '161M'
        },
        'rtv4_hgnetv2_l_coco.yml': {
            'url': 'https://drive.usercontent.google.com/download?id=1shO9EzZvXZyKedE2urLsN4dwEv8Jqa_8&export=download&confirm=t',
            'filename': 'rtv4_hgnetv2_l_model.pth',
            'size': '161M'
        },
        'rtv4_hgnetv2_x_coco.yml': {
            'url': 'https://drive.usercontent.google.com/download?id=19gnkMTgFveJsrOvSmEPQXCTG6v9oQHN3&export=download&confirm=t',
            'filename': 'rtv4_hgnetv2_x_model.pth',
            'size': '161M'
        }
    }


def download_model_weights(config_path: str, output_dir: str = ".") -> str:
    """
    Download model weights based on config file.
    
    Args:
        config_path: Path to model config file
        output_dir: Directory to save downloaded weights
        
    Returns:
        Path to downloaded weights file
    """
    config_name = Path(config_path).name
    weights_info = get_model_weights_info()
    
    if config_name not in weights_info:
        available_configs = list(weights_info.keys())
        logger.warning(f"No predefined weights for config: {config_name}")
        logger.warning(f"Available configs: {available_configs}")
        return None
    
    weight_info = weights_info[config_name]
    output_path = Path(output_dir) / weight_info['filename']
    
    # Check if weights already exist
    if output_path.exists():
        logger.info(f"Model weights already exist: {output_path}")
        return str(output_path)
    
    logger.info(f"Downloading model weights for {config_name}...")
    logger.info(f"Size: {weight_info['size']}")
    logger.info(f"Output: {output_path}")
    
    try:
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Download with progress
        def progress_hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            if percent % 10 == 0:  # Log every 10%
                logger.info(f"Download progress: {percent}%")
        
        urllib.request.urlretrieve(weight_info['url'], output_path, progress_hook)
        
        logger.info(f"Successfully downloaded weights: {output_path}")
        return str(output_path)
        
    except Exception as e:
        logger.error(f"Failed to download weights: {e}")
        if output_path.exists():
            output_path.unlink()  # Clean up partial download
        raise RuntimeError(f"Weight download failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Export RT-DETR models (v1/v2/v3/v4) to ONNX/TensorRT")
    
    # Required arguments
    parser.add_argument("-c", "--config", required=True, help="Path to model config file")
    parser.add_argument("-r", "--checkpoint", required=True, help="Path to model checkpoint")
    
    # Export format
    parser.add_argument("--format", choices=["onnx", "tensorrt", "both"], default="onnx",
                       help="Export format (default: onnx)")
    
    # Output options
    parser.add_argument("--output-dir", default="./exported_models",
                       help="Output directory for exported models")
    
    # ONNX options
    parser.add_argument("--no-check", action="store_true",
                       help="Skip ONNX model validation")
    parser.add_argument("--no-simplify", action="store_true",
                       help="Skip ONNX model simplification")
    
    # TensorRT options
    parser.add_argument("--no-fp16", action="store_true",
                       help="Disable FP16 precision for TensorRT")
    parser.add_argument("--workspace-size", default="4g",
                       help="TensorRT workspace size (default: 4g)")
    
    # Benchmarking
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark after export")
    parser.add_argument("--coco-dir", help="Path to COCO dataset for benchmarking")
    
    # Model info
    parser.add_argument("--model-info", action="store_true",
                       help="Display model FLOPs, MACs, and parameters")
    
    # Virtual environment
    parser.add_argument("--skip-venv-check", action="store_true",
                       help="Skip virtual environment check")
    
    # Repository management
    parser.add_argument("--repo-dir", 
                       help="RT-DETR repository directory (auto-detected if not specified)")
    parser.add_argument("--clone-repo", action="store_true",
                       help="Clone appropriate RT-DETR repository if it doesn't exist")
    parser.add_argument("--install-deps", action="store_true",
                       help="Install RT-DETR dependencies")
    parser.add_argument("--version", choices=['v1', 'v2', 'v3', 'v4', 'dfine', 'deim'],
                       help="RT-DETR version (auto-detected from config if not specified)")
    
    # Model weights
    parser.add_argument("--download-weights", action="store_true",
                       help="Automatically download model weights based on config file")
    parser.add_argument("--weights-dir", default="./weights",
                       help="Directory to download weights to (default: ./weights)")
    
    args = parser.parse_args()
    
    try:
        # Check virtual environment
        if not args.skip_venv_check:
            check_virtual_environment()
        
        # Detect RT-DETR version
        version = args.version or detect_rtdetr_version(args.config)
        logger.info(f"Detected RT-DETR version: {version.upper()}")
        
        # Setup repository path
        if args.repo_dir:
            repo_path = args.repo_dir
        else:
            _, default_name = get_repo_info(version)
            repo_path = f"./{default_name}"
        
        # Clone repository if requested
        if args.clone_repo:
            repo_path = clone_rtdetr_repo(repo_path, version)
        
        if args.install_deps:
            setup_environment(repo_path)
        
        # Validate repository structure
        if not Path(repo_path).exists():
            logger.error(f"Repository directory not found: {repo_path}")
            logger.info("Use --clone-repo to automatically clone the repository")
            sys.exit(1)
        
        # Download weights if requested
        checkpoint_path = args.checkpoint
        if args.download_weights:
            downloaded_weights = download_model_weights(args.config, args.weights_dir)
            if downloaded_weights:
                checkpoint_path = downloaded_weights
                logger.info(f"Using downloaded weights: {checkpoint_path}")
            else:
                logger.warning("Could not download weights, using provided checkpoint path")
        
        # Initialize exporter
        exporter = RTDETRExporter(args.config, checkpoint_path, args.output_dir, repo_path, version)
        
        # Display model info if requested
        if args.model_info:
            get_model_info(args.config, repo_path)
        
        exported_models = []
        
        # Export based on format
        if args.format in ["onnx", "both"]:
            onnx_path = exporter.export_onnx(
                check_model=not args.no_check,
                simplify=not args.no_simplify
            )
            exported_models.append(("onnx", onnx_path))
            
            if args.benchmark:
                exporter.benchmark_model(onnx_path, "onnx", args.coco_dir)
        
        if args.format in ["tensorrt", "both"]:
            # Use existing ONNX if available, otherwise export it
            onnx_path = None
            if args.format == "both" and exported_models:
                onnx_path = exported_models[0][1]
            
            trt_path = exporter.export_tensorrt(
                onnx_path=onnx_path,
                fp16=not args.no_fp16,
                workspace_size=args.workspace_size
            )
            exported_models.append(("tensorrt", trt_path))
            
            if args.benchmark:
                exporter.benchmark_model(trt_path, "tensorrt", args.coco_dir)
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("EXPORT SUMMARY")
        logger.info("="*50)
        for format_type, path in exported_models:
            logger.info(f"{format_type.upper()}: {path}")
        
        logger.info("\nExport completed successfully!")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
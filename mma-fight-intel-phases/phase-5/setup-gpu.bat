@echo off
REM Optional: replace the CPU build of PyTorch with a CUDA build for NVIDIA GPUs.
REM Requires an NVIDIA GPU with a recent driver. Run AFTER setup.bat.
cd /d "%~dp0"
if not exist backend\.venv\Scripts\python.exe (
  echo Run setup.bat first.
  pause
  exit /b 1
)
echo Installing CUDA-enabled PyTorch (large download, ~2.5 GB) ...
echo Older NVIDIA driver? Use cu126 instead of cu132. Or pick the right command for your system at https://pytorch.org/get-started/locally/
backend\.venv\Scripts\python.exe -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu132
backend\.venv\Scripts\python.exe -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
pause

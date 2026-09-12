# Geo2DGS-SLAM

1. Create and activate environment
```bash
conda create -n Geo2DGS-SLAM python=3.9 -y
conda activate Geo2DGS-SLAM
```
2. Install PyTorch (CUDA 12.1)
```bash
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu121
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Install third-party extensions
```bash
pip install thirdparty/diff-surfel-rasterization/ --no-build-isolation
pip install thirdparty/LightGlue-main/ --no-build-isolation
pip install thirdparty/simple-knn/ --no-build-isolation
pip install thirdparty/lietorch/ --no-build-isolation
pip install thirdparty/torch_scatter-2.1.2+pt25cu121-cp39-cp39-linux_x86_64.whl
pip install . --no-build-isolation
```
demo download:https://pan.quark.cn/s/18ce416928c5



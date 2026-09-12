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
pip install thirdparty/evaluate_3d_reconstruction_lib-main/ --no-build-isolation
pip install thirdparty/torch_scatter-2.1.2+pt25cu121-cp39-cp39-linux_x86_64.whl
pip install . --no-build-isolation
```
demo download:https://pan.quark.cn/s/18ce416928c5

This project is built upon and adopts code from the following open-source works.
```bash
@inproceedings{Huang2DGS2024,
    title={2D Gaussian Splatting for Geometrically Accurate Radiance Fields},
    author={Huang, Binbin and Yu, Zehao and Chen, Anpei and Geiger, Andreas and Gao, Shenghua},
    publisher = {Association for Computing Machinery},
    booktitle = {SIGGRAPH 2024 Conference Papers},
    year      = {2024},
    doi       = {10.1145/3641519.3657428}
}
```
```bash
@misc{yugay2023gaussianslam,
      title={Gaussian-SLAM: Photo-realistic Dense SLAM with Gaussian Splatting}, 
      author={Vladimir Yugay and Yue Li and Theo Gevers and Martin R. Oswald},
      year={2023},
      eprint={2312.10070},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
```bash
@inproceedings{zhang2023goslam,
    author    = {Zhang, Youmin and Tosi, Fabio and Mattoccia, Stefano and Poggi, Matteo},
    title     = {GO-SLAM: Global Optimization for Consistent 3D Instant Reconstruction},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
}
```

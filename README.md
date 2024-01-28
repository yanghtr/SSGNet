## LiDAR-Based 3D Object Detection via Hybrid 2D Semantic Scene Generation [\[paper\]](https://arxiv.org/abs/2304.01519)

This project is built on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) and [VoxSeT](https://github.com/skyhehe123/VoxSeT).

### 1. Environment

- Python 3.8
- PyTorch 1.10.0
- CUDA 10.2
- pcdet 0.5.2

The environment setup, data preparation, training, and testing processes are the same as those in OpenPCDet and VoxSeT. To facilitate an easier review of the changed files, we have cleaned up our code and built it directly on top of the original VoxSeT repository (commit 9d2b87d5f9a38). To examine the differences from the original VoxSeT repository:

```
git diff --name-only 9d2b87d5f9a38  # check file names
git diff 9d2b87d5f9a38  # check all differences
```

### 2. Train

- Train with multiple GPUs 

```shell
cd VoxSeT/tools
bash scripts/dist_train.sh --cfg_file ./cfgs/waymo_models/ssgnet/centerpoint_voxset_1x_ssgHybrid_trainSI1_testSI1_lr0006_smpW1_8B2.yaml
```

The log is in [output/waymo_models/ssgnet/voxset_paper_results/default/log_train_20221209-023159.txt](./output/waymo_models/ssgnet/voxset_paper_results/default/log_train_20221209-023159.txt)

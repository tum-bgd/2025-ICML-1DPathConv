# One-dimensional Path Convolution

Linearly scaling 1D convolution provides parameter efficiency, but its naive integration into CNNs disrupts image locality, thereby degrading performance. We present path convolution (PathConv), a novel CNN design exclusively with 1D operations, achieving ResNet-level accuracy using only 1/3 parameters. To obtain locality-preserving image traversal paths, we propose path shifting on Hilbert/Z-order paths, a succinct method to reposition sacrificed pixels. We show that three shifted paths are sufficient to offer better locality preservation than trivial raster scanning. To mitigate potential convergence issues caused by multiple paths, we design a lightweight path-aware channel attention mechanism to capture local intra-path and global inter-path dependencies.

## Environment

We provide environment configurations for both Apptainer and Docker, respectively.

### Apptainer

```bash
srun apptainer pull docker://pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
chmod +x exp/*.sh
sbatch --gpus=1 run.sh <exp>
```

Change `<exp>` to the name of the shell script to execute.

### Docker

```bash
docker pull pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel
docker run -it --shm-size=32G --gpus device=0 --name pathconv --mount type=bind,src=<data-folder>,dst=/data --mount type=bind,src=$(pwd),dst=/workspace pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel bash
python -m pip install networkx numpy matplotlib imageio timm cupy-cuda12x PuLP scikit-learn ninja
docker exec -it pathconv bash
```

You may modify `<data-folder>` in `run.sh` or the above commands according to your case.

## Usage

### Data preparation

We use built-in datasets in the `torchvision.datasets` module, where the ImageNet 64*64 dataset is not there. Hence, we provode the following steps to prepare it for training. 

#### ImageNet 64*64 dataset preparation

1. Download the raw dataset from [here](https://image-net.org/download-images.php).
2. Unzip.
3. Run `in64.py` to obtain a PyTorch `ImageFolder` compatible dataset saving to `RAW_DIR`.

    ```bash
    python ./src/in64.py
    ```

### Experiments

For the Apptainer workaround, you can modify the command to run in `./exp/example.sh`.

#### Training & Evaluation

The following command shows an example of training `PathConvS` using the raster scanning path on the CIFAR-10 dataset with a batch size of 128.

```bash
python ./src/main.py --bs 128 --dataset cf10 --path r --model s
```

Detailed parameter settings are elaborated in `main.py`.

#### Path visualization

We provide a tool to visualize paths as shown in the paper.

```bash
python ./src/model/path/visualization.py
```

#### Search for a minimal set of paths

`search.py` provides both greedy and randomized rounding solvers to find a minimal set of shifted paths to satisfy the locality
constraint.

```bash
python ./src/model/path/search.py
```

## Citation

```
@inproceedings{Luo2025onedimensional,
  title={One-dimensional Path Convolution},
  author={Luo, Xuanshu and Werner, Martin},
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025},
  url={https://openreview.net/forum?id=9aEYGpSV6v}
}
```

## License

Licensed under Apache-2.0 license (LICENSE or https://opensource.org/licenses/Apache-2.0)

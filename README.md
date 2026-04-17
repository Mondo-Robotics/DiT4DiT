<div align="center">

  <img src="media/logo.svg" width="480" alt="DiT4DiT">

  <h2>Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control</h2>

</div>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2603.10448-FF5500.svg)](https://arxiv.org/abs/2603.10448)
[![Project Page](https://img.shields.io/badge/Project-Page-FF8C00.svg)](https://dit4dit.github.io/)
[![License](https://img.shields.io/badge/License-MIT-FF8C00.svg)](LICENSE)

</div>

<div align="center">

[Teli Ma](https://teleema.github.io/)<sup>1,2</sup> &nbsp;&nbsp;
[Jia Zheng](https://jiaazheng.github.io/)<sup>1,2</sup> &nbsp;&nbsp;
[Zifan Wang](https://scholar.google.com/citations?user=GaJXZ-UAAAAJ&hl=en)<sup>1,2</sup> &nbsp;&nbsp;
[Chunli Jiang](https://scholar.google.com/citations?user=nvzF-RMAAAAJ&hl=en)<sup>1</sup> &nbsp;&nbsp;
Andy Cui<sup>1</sup> &nbsp;&nbsp;
[Junwei Liang](https://junweiliang.me/index.html)<sup>2,3,\*</sup> &nbsp;&nbsp;
[Shuo Yang](https://shuoyangrobotics.github.io/)<sup>1,\*</sup>

<sup>1</sup>Mondo Robotics &nbsp;&nbsp; <sup>2</sup>HKUST(GZ) &nbsp;&nbsp; <sup>3</sup>HKUST &nbsp;&nbsp; <sup>\*</sup>Corresponding author

</div>

---

DiT4DiT is a <b><span style="color: #FF8C00;">Vision-Action-Model (VAM)</span></b> framework that combines video generation transformers with flow-matching-based action prediction for generalizable robotic manipulation. It supports both the tabletop and whole-body control for manipulation tasks. Notably, DiT4DiT stands as the <b>first</b> efficient VAM to achieve real-time whole-body control of humanoid robots.



## News

- **[2026-04-15]** Initial release of DiT4DiT with training, evaluation, and deployment code.
- **[2026-03-11]** We release the [arXiv paper](https://arxiv.org/abs/2603.10448).


### Whole-Body Control (all 1x speed & autonomous)

<div align="center">
<table>
<tr>
<td align="center" colspan="2"><b>Shelf Organization</b></td>
</tr>
<tr>
<td align="center" colspan="2"><img src="media/shelf_organization.webp" width="800"></td>
</tr>
<tr>
<td align="center" colspan="2"><b>Relocate Chair</b></td>
</tr>
<tr>
<td align="center" colspan="2"><img src="media/relocate_chair.webp" width="800"></td>
</tr>
</table>
</div>

### Tabletop Manipulation (all 1x speed, 1 policy for all tasks)

<div align="center">
<table>
<tr>
<td align="center"><b>Stack Cups</b></td>
<td align="center"><b>Drawer Interaction</b></td>
</tr>
<tr>
<td align="center"><img src="media/cups.gif" width="400"></td>
<td align="center"><img src="media/drawer.gif" width="400"></td>
</tr>
<tr>
<td align="center"><b>Pick and Place</b></td>
<td align="center"><b>Arrange Flower</b></td>
</tr>
<tr>
<td align="center"><img src="media/eggplant.gif" width="400"></td>
<td align="center"><img src="media/flower.gif" width="400"></td>
</tr>
<tr>
<td align="center"><b>Move Spoon</b></td>
<td align="center"><b>Insert Plate</b></td>
</tr>
<tr>
<td align="center"><img src="media/spoon.gif" width="400"></td>
<td align="center"><img src="media/plate.gif" width="400"></td>
</tr>
<tr>
<td align="center"><b>Box Packing</b></td>
<td align="center"><b>Twist Cap</b></td>
</tr>
<tr>
<td align="center"><img src="media/packing.gif" width="400"></td>
<td align="center"><img src="media/twist_cap.gif" width="400"></td>
</tr>
</table>
</div>


## Table of Contents

- [News](#news)
- [TODOs](#todos)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [Simulation](#simulation)
  - [Real Robot](#real-robot)
- [Acknowledgements](#acknowledgements)
- [License](#license)



## TODOs

- [ ] Release teleoperation, training and deployment code for Unitree G1 tabletop tasks.
- [ ] Release teleoperation, training and deployment code for Unitree G1 whole-body control tasks.

## Project Structure

```
DiT4DiT/
├── DiT4DiT/                    # Core package
│   ├── config/                 # Configurations
│   │   ├── deepseeds/          # DeepSpeed configs
│   │   ├── robocasa/           # RoboCasa experiment configs
│   │   └── real_robot/         # Real robot configs
│   ├── dataloader/             # Dataset loading (LeRobot)
│   ├── model/                  # Model architecture
│   │   ├── framework/          # DiT4DiT framework
│   │   └── modules/            # Backbone & action model
│   └── training/               # Training scripts & utilities
├── deployment/                 # WebSocket-based model server
├── docs/                       # Documentation
├── examples/
│   ├── Robocasa_tabletop/      # RoboCasa simulation example
│   │   ├── train_files/        # Training scripts
│   │   └── eval_files/         # Evaluation & simulation
│   └── Real_G1/                # Real Unitree G1 example
│       ├── train_files/        # Training scripts
│       └── eval_files/         # Evaluation
└── requirements.txt
```

## Installation

### Prerequisites

- Python >= 3.10
- CUDA 12.4+
- \>8x GPUs recommended for training

### Setup

```bash
# Clone the repository
git clone https://github.com/Mondo-Robotics/DiT4DiT.git
cd DiT4DiT

# Create conda environment
conda create -n dit4dit python=3.10 -y
conda activate dit4dit

# Install PyTorch (CUDA 12.8 recommended)
pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

### Download Pretrained Backbone

Download the Cosmos-Predict2.5-2B model from Hugging Face:

```bash
huggingface-cli download nvidia/Cosmos-Predict2.5-2B --revision diffusers/base/post-trained --local-dir /path/to/Cosmos-Predict2.5-2B
```

## Model Zoo

We release pretrained checkpoints to facilitate reproduction.

### Available Checkpoints

| Model | Description | Dataset | Success Rate | Link |
| --- | --- | --- | --- | --- |
| **DiT4DiT-RoboCasa-GR1** | DiT4DiT for RoboCasa-GR1 tabletop tasks | RoboCasa-GR1 | 56.7 | [🤗 Hugging Face](https://huggingface.co/TeliMa/dit4dit_robocasa_gr1/tree/main) |

> **Note:** More checkpoints will be released soon. Stay tuned!

## Quick Start

### Simulation

- **RoboCasa-GR1 Tabletop**: See the full training and evaluation guide [here](docs/robocasa_tabletop.md).

### Real Robot

Coming soon.

## Results

### Robocasa-GR1 Benchmark

The following results are obtained using the default training parameters described in [Configure Training](#configure-training). We report three independent evaluation runs of the same checkpoint to demonstrate reproducibility. The model consistently achieves an average success rate above 56% across all runs.

| Task | Run 1 | Run 2 | Run 3 |
|------|-------|-------|-------|
| BottleToCabinetClose | 50.0 | 72.0 | 68.0 |
| CanToDrawerClose | 80.0 | 80.0 | 82.0 |
| CupToDrawerClose | 50.0 | 34.0 | 50.0 |
| MilkToMicrowaveClose | 58.0 | 60.0 | 38.0 |
| PotatoToMicrowaveClose | 40.0 | 40.0 | 36.0 |
| WineToCabinetClose | 60.0 | 48.0 | 60.0 |
| FromCuttingboardToBasket | 54.0 | 48.0 | 46.0 |
| FromCuttingboardToCardboardbox | 50.0 | 60.0 | 48.0 |
| FromCuttingboardToPan | 80.0 | 74.0 | 78.0 |
| FromCuttingboardToPot | 52.0 | 46.0 | 66.0 |
| FromCuttingboardToTieredbasket | 44.0 | 54.0 | 50.0 |
| FromPlacematToBasket | 58.0 | 40.0 | 44.0 |
| FromPlacematToBowl | 64.0 | 66.0 | 72.0 |
| FromPlacematToPlate | 66.0 | 62.0 | 64.0 |
| FromPlacematToTieredshelf | 44.0 | 48.0 | 40.0 |
| FromPlateToBowl | 64.0 | 74.0 | 54.0 |
| FromPlateToCardboardbox | 50.0 | 54.0 | 52.0 |
| FromPlateToPan | 58.0 | 68.0 | 70.0 |
| FromPlateToPlate | 62.0 | 64.0 | 72.0 |
| FromTrayToCardboardbox | 52.0 | 50.0 | 60.0 |
| FromTrayToPlate | 64.0 | 64.0 | 58.0 |
| FromTrayToPot | 68.0 | 70.0 | 66.0 |
| FromTrayToTieredbasket | 50.0 | 46.0 | 50.0 |
| FromTrayToTieredshelf | 42.0 | 36.0 | 28.0 |
| **Average** | **56.7** | **56.6** | **56.3** |


## Citation

If you find this work useful, please consider citing:

```bibtex
@article{ma2026dit4dit,
  title={DiT4DiT: Jointly Modeling Video Dynamics and Actions for Generalizable Robot Control},
  author={Ma, Teli and Zheng, Jia and Wang, Zifan and Jiang, Chunli and Cui, Andy and Liang, Junwei and Yang, Shuo},
  journal={arXiv preprint arXiv:2603.10448},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project builds upon:
- [StarVLA](https://github.com/starVLA/starVLA)
- [Cosmos-Predict2.5](https://github.com/NVIDIA/Cosmos) by NVIDIA
- [GR00T](https://github.com/NVIDIA/Isaac-GR00T) by NVIDIA
- [Robocasa](https://github.com/robocasa/robocasa)
- [LeRobot](https://github.com/huggingface/lerobot) by Hugging Face




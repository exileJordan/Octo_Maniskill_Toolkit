# Octo-ManiSkill-Toolkit
## Overview
This repository is a fork of [octo-models/octo](https://github.com/octo-models/octo).

It extends the original Octo codebase with tools for adapting Octo to ManiSkill simulation tasks, including ManiSkill-to-RLDS data conversion, Octo fine-tuning scripts, and ManiSkill-based evaluation utilities.

The goal of this project is to provide a reproducible pipeline:

ManiSkill demonstrations → RLDS dataset → Octo fine-tuning → checkpoint evaluation
## What This Fork Adds

| Component | Path | Description |
|---|---|---|
| ManiSkill-to-RLDS converter | `tools/maniskill_rlds_tool/` | Converts ManiSkill HDF5 demonstrations into RLDS format for Octo fine-tuning. |
| Fine-tuning script with validation | `examples/02_finetune_with_validation.py` | Adapts Octo fine-tuning to ManiSkill-style observations, language instructions, proprioceptive states, and action spaces and supports offline evaluation using a validation dataset. |
| ManiSkill evaluation toolkit | `tools/maniskill_eval_tool/` | Provides ManiSkill rollout evaluation utilities, including success-rate reporting and optional video saving. |
## Project Structure
```text

├── octo/                         # Original Octo core library and model codebase.
├── examples/
│   └── 02_finetune_with_validation.py  # Fine-tuning script with validation support.
├── tools/
│   ├── maniskill_rlds_tool/           # ManiSkill-to-RLDS converter
│   └── maniskill_eval_tool/           # ManiSkill rollout evaluation tools, including success rate and video generation.
└── README.md
```
## Installation
```bash
conda create -n octo_maniskill python=3.10
conda activate octo_maniskill
pip install -e .
pip install -r requirements.txt
```
ManiSkill rendering may require a working Vulkan/SAPIEN environment for RGB-based rollout evaluation.

## Convert ManiSkill Data to RLDS
This section provides a minimal PickCube example for converting ManiSkill demonstration data into an Octo-compatible RLDS / TFDS dataset. For detailed explanations and expected outputs of each command, please refer to [Maniskill_RLDS_Converter](https://github.com/exileJordan/Maniskill_RLDS_Converter).
### 1. Prepare H5 Files
(1) Download Maniskill "PickCube-v1" Trajectory and Replay
```bash
python -m mani_skill.utils.download_demo "PickCube-v1"
```
The downloaded demo directory should look like:
```bash
# result
demos/
└── PickCube-v1/motionplanning
    ├── trajectory.h5
    └── trajectory.json
```
(2) Replay PickCube trajectory and convert to target dataset. We recommend starting with a small number of trajectories.Here, `--count 4` replays only four trajectories for a quick test.
```bash
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path ${DEMO_PATH}PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_ee_delta_pos -o rgb \
  --save-traj --num-envs 10 -b physx_cpu \
  --count 4
```
(3) Create a new directory and copy your replayed H5 files into that directory:
```bash
mkdir -p /path/to/maniskill_h5
cp /path/to/trajectory*.h5 /path/to/maniskill_h5/
```
The directory should look like:
```text
/path/to/maniskill_h5/
  trajectory_*.h5
```
### 2. Inspect H5 Structure
Before building the dataset, inspect your H5 file structure:
```bash
python tools/maniskill_rlds_tool/inspect_h5.py \
  /path/to/maniskill_h5/trajectory_0.h5
```

### 3. Build TFDS/RLDS Dataset
Run `tfds build` from the repository root:
```bash
tfds build tools/maniskill_rlds_tool/maniskill_convert_dataset \
  --manual_dir=/path/to/maniskill_h5 \
  --data_dir=/path/to/tfds_output
```
- `--manual_dir`: directory containing your input H5 files.
- `--data_dir`: directory where TFDS/RLDS output will be written.

### 4. Validate with TFDS
Use the TFDS checker to read one RLDS episode and inspect one step:

```bash
python tools/maniskill_rlds_tool/check_tfds_dataset.py \
  --name=maniskill_convert_dataset \
  --data_dir=/path/to/tfds_output \
  --split=train
```

Validate the validation split as well:

```bash
python tools/maniskill_rlds_tool/check_tfds_dataset.py \
  --name=maniskill_convert_dataset \
  --data_dir=/path/to/tfds_output \
  --split=val
```

### 5. Validate with dlimp
Use dlimp to flatten RLDS episodes into trajectory-level samples:

```bash
python tools/maniskill_rlds_tool/check_dlimp_dataset.py \
  --name=maniskill_convert_dataset \
  --data_dir=/path/to/tfds_output \
  --split=train
```

Validate the validation split:

```bash
python tools/maniskill_rlds_tool/check_dlimp_dataset.py \
  --name=maniskill_convert_dataset \
  --data_dir=/path/to/tfds_output \
  --split=val
```
If both TFDS and dlimp validation pass, the RLDS dataset was generated successfully.

## Finetune Octo on ManiSkill RLDS
After converting ManiSkill demonstrations into an Octo-compatible RLDS / TFDS dataset, use the following script to fine-tune Octo:

```bash
python examples/02_finetune_with_validation.py \
  --pretrained_path=hf://rail-berkeley/octo-small-1.5 \
  --data_dir=/path/to/tfds_output \
  --save_dir=/path/to/finetuned_checkpoints \
  --batch_size=10
```
The default example is configured for the ManiSkill PickCube RLDS dataset:
```text
dataset name: maniskill_pickcube_dataset
image key: image
proprio key: proprio
language key: language_instruction
window size: 4
action horizon: 5
action dimension: 4
```
For other ManiSkill tasks, update the dataset name, standardization function, observation keys, action dimension, control mode, and language instruction.

## Evaluate Finetuned Octo in ManiSkill
After fine-tuning, use the following script to evaluate the finetuned Octo checkpoint in ManiSkill.

Make sure that `env_id`, `obs_mode`, `control_mode`, and `window_size` are consistent with the data collection and fine-tuning settings.

```bash
python tools/maniskill_eval_tool/eval_octo_maniskill.py \
  --stage=eval \
  --checkpoint_path=/path/to/finetuned_checkpoint \
  --output_dir=eval_outputs/maniskill_octo/debug \
  --env_id=PickCube-v1 \
  --obs_mode=rgb \
  --control_mode=pd_ee_delta_pos \
  --window_size=4 \
  --num_episodes=1 \
  --save_video
```
## Experimental Results and Limitations
### Offline Evaluation Results
The offline evaluation results of the finetuned Octo model on ManiSkill PickCube and PushCube tasks are shown below. The metrics include train loss, train mse, validation loss, validation mse.  

PickCube Results

<p align="center">
  <img src="docs/assets/pickcube.png" alt="PickCube Results" width="650">
</p>

PushCube Results
<p align="center">
  <img src="docs/assets/pushcube.png" alt="PushCube Results" width="650">
</p>

### Limitations

The evaluation script supports closed-loop rollout evaluation, but this release only reports offline action prediction metrics. Large-scale rollout results and success rates are not included yet.

The main limitation is that online RGB-based rollout evaluation requires ManiSkill rendering, which depends on SAPIEN and Vulkan. In the tested cloud GPU environment, Vulkan initialization failed, preventing reliable RGB rollout evaluation.

Future updates will include:

- closed-loop rollout evaluation in ManiSkill
- task success rate
- rollout videos

## Citation
This repository is built on top of the official Octo project. If you use this codebase, please cite the original Octo work:

```bibtex
@article{octo_2023,
  title={Octo: An Open-Source Generalist Robot Policy},
  author={Octo Model Team and Ghosh, Dibya and Walke, Homer and Pertsch, Karl and Black, Kevin and Mees, Oier and Dasari, Sudeep and Hejna, Joey and Kreiman, Tairan and Xu, Charles and Luo, Jianlan and Tan, You Liang and Chen, Dorsa Sadigh and Finn, Chelsea and Levine, Sergey},
  journal={arXiv preprint arXiv:2405.12213},
  year={2024}
}
```
## License and Acknowledgements
The original Octo code is licensed under the MIT License. This fork keeps the original license notice. See [LICENSE](LICENSE) for details.

This project is built upon:

- [Octo](https://github.com/octo-models/octo): the original VLA model codebase and fine-tuning framework.
- [ManiSkill](https://github.com/haosulab/ManiSkill): the simulation benchmark and demonstration data source.
- [RLDS](https://github.com/google-research/rlds): the dataset format used for robot learning trajectories.

This fork does not redistribute official Octo checkpoints, ManiSkill assets, or large-scale generated datasets.
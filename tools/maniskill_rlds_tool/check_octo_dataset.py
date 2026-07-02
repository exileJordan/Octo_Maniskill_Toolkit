"""Check Octo make_single_dataset with converted ManiSkill TFDS/RLDS data."""

import argparse
import tensorflow as tf

from octo.data.dataset import make_single_dataset
from octo.utils.spec import ModuleSpec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="maniskill_pickcube_dataset")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--action_horizon", type=int, default=4)
    args = parser.parse_args()

    # Data loading uses TensorFlow. Keep it off GPU to avoid occupying model memory.
    tf.config.set_visible_devices([], "GPU")

    dataset = make_single_dataset(
        dataset_kwargs=dict(
            name=args.name,
            data_dir=args.data_dir,
            image_obs_keys={"primary": "image"},
            proprio_obs_key="proprio",
            language_key="language_instruction",
            standardize_fn=ModuleSpec.create(
                "tools.maniskill_rlds.maniskill_standardization_transforms:"
                "maniskill_proprio_qpos_qvel_tcp_pose"
            ),
        ),
        traj_transform_kwargs=dict(
            window_size=1,
            action_horizon=args.action_horizon,
        ),
        frame_transform_kwargs=dict(
            resize_size={"primary": (256, 256)},
        ),
        train=True,
    )

    batch = (
        dataset.repeat()
        .unbatch()
        .batch(args.batch_size)
        .iterator()
        .next()
    )

    print("Batch keys:", batch.keys())
    print("Observation keys:", batch["observation"].keys())
    print("Task keys:", batch["task"].keys())

    print("image_primary:", batch["observation"]["image_primary"].shape)
    print("proprio:", batch["observation"]["proprio"].shape)
    print("action:", batch["action"].shape)
    print("action_pad_mask:", batch["action_pad_mask"].shape)
    
    print("language_instruction:", batch["task"]["language_instruction"].shape)

    print("\nDataset statistics keys:", dataset.dataset_statistics.keys())
    print("Action stats mean shape:", dataset.dataset_statistics["action"]["mean"].shape)
    print("Proprio stats mean shape:", dataset.dataset_statistics["proprio"]["mean"].shape)
    print("action_pad_mask",batch["action_pad_mask"])
    print("action:", batch["action"])

if __name__ == "__main__":
    main()
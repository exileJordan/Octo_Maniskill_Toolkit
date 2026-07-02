import argparse

import dlimp as dl
import tensorflow_datasets as tfds


def print_value(name, value):
    if hasattr(value, "shape"):
        print(f"{name} shape:", value.shape)
        print(f"{name} dtype:", value.dtype)
        return

    if isinstance(value, dict):
        print(f"{name} is a dict with keys:", value.keys())
        for key, nested_value in value.items():
            print_value(f"{name}/{key}", nested_value)
        return

    print(f"{name} type:", type(value))

# This script use dlimp to load one trajectory from the converted dataset to check if the data is correctly converted and can be loaded with error
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="maniskill_convert_dataset")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    # find the TFDS dataset 
    builder = tfds.builder(args.name, data_dir=args.data_dir)
    # use dlimp to load the dataset based on the defined "split"(train/validation)
    dataset = dl.DLataset.from_rlds(builder, split=args.split, shuffle=False)
    # load one trajectory from the dataset.
    traj = next(iter(dataset.take(1)))

    print("Trajectory keys:", traj.keys())
    print("Observation keys:", traj["observation"].keys())
    print_value("Image", traj["observation"]["image"])
    print_value("Wrist image", traj["observation"]["wrist_image"])
    print_value("Has wrist image", traj["observation"]["has_wrist_image"])
    print_value("Depth", traj["observation"]["depth"])
    print_value("Has depth", traj["observation"]["has_depth"])
    print_value("Wrist depth", traj["observation"]["wrist_depth"])
    print_value("Has wrist depth", traj["observation"]["has_wrist_depth"])
    print_value("Qpos", traj["observation"]["qpos"])
    print_value("Qvel", traj["observation"]["qvel"])
    print_value("TCP pose", traj["observation"]["tcp_pose"])
    print_value("Is grasped", traj["observation"]["is_grasped"])
    print_value("Goal pos", traj["observation"]["goal_pos"])
    print_value("Action", traj["action"])
    print_value("Language", traj["language_instruction"])


if __name__ == "__main__":
    main()

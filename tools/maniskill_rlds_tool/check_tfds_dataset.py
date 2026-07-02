import argparse

import tensorflow_datasets as tfds

# This script is used to check the step data in the converted TFDS dataset to make sure the data is correctly converted.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="maniskill_convert_dataset")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split", default="train")
    args = parser.parse_args()

    # find the TFDS dataset
    builder = tfds.builder(args.name, data_dir=args.data_dir)
    print(builder.info)

    # use tfds to load the dataset based on the defined "split"(train/validation)
    ds = builder.as_dataset(split=args.split)
    # load one episode from the dataset.
    episode = next(iter(ds.take(1)))

    print("Episode keys:", episode.keys())
    print("Metadata:", episode["episode_metadata"])
    # load all the steps in the episode
    steps = episode["steps"]
    # load one step form steps. 
    step = next(iter(steps.take(1)))

    print("Step keys:", step.keys())
    print("Observation keys:", step["observation"].keys())
    print("Image shape:", step["observation"]["image"].shape)
    print("Image dtype:", step["observation"]["image"].dtype)
    print("Wrist image shape:", step["observation"]["wrist_image"].shape)
    print("Has wrist image:", step["observation"]["has_wrist_image"])
    print("Depth shape:", step["observation"]["depth"].shape)
    print("Has depth:", step["observation"]["has_depth"])
    print("Wrist depth shape:", step["observation"]["wrist_depth"].shape)
    print("Has wrist depth:", step["observation"]["has_wrist_depth"])
    print("Qpos shape:", step["observation"]["qpos"].shape)
    print("Qvel shape:", step["observation"]["qvel"].shape)
    print("TCP pose shape:", step["observation"]["tcp_pose"].shape)
    print("Is grasped:", step["observation"]["is_grasped"])
    print("Goal pos shape:", step["observation"]["goal_pos"].shape)
    print("Action shape:", step["action"].shape)
    print("Language:", step["language_instruction"])


if __name__ == "__main__":
    main()
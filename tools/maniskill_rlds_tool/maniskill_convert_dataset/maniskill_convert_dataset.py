
from pathlib import Path

import tensorflow_datasets as tfds
import hashlib

import h5py
import numpy as np

# This is the main dataset conversion code. It reads ManiSkill replay H5 file, extracts relevant data, and yields episodes in the format expected by RLDS.
class ManiskillDataset(tfds.core.GeneratorBasedBuilder):
    """Converts ManiSkill replay H5 files to RLDS."""

    # change these constants if your dataset has different structure or you want to include more/less data.
    # make sure it is consistent with what you see in inspect_h5.py 
    # usually you only need to change the TASK_NAME, LANGUAGE_INSTRUCTION，GOAL_POS_PATH, and the dimension of qpos, qvel, tcp_pose, and is_grasped.
    TASK_NAME = "PickCube-v1"
    LANGUAGE_INSTRUCTION = "Pick up the object and move it to a goal position."

    RGB_PATH = "obs/sensor_data/base_camera/rgb"
    WRIST_RGB_PATHS = [
        "obs/sensor_data/hand_camera/rgb",
        "obs/sensor_data/wrist_camera/rgb",
    ]
    DEPTH_PATHS = [
        "obs/sensor_data/base_camera/depth",
    ]
    WRIST_DEPTH_PATHS = [
        "obs/sensor_data/hand_camera/depth",
        "obs/sensor_data/wrist_camera/depth",
    ]
    QPOS_PATH = "obs/agent/qpos"
    QVEL_PATH = "obs/agent/qvel"
    TCP_POSE_PATH = "obs/extra/tcp_pose"
    IS_GRASPED_PATH = "obs/extra/is_grasped"  
    GOAL_POS_PATH = "obs/extra/goal_pos"
    VERSION = tfds.core.Version("1.0.0")
    MANUAL_DOWNLOAD_INSTRUCTIONS = """
        Please place replayed ManiSkill H5 files in manual_dir.
        For example:
        /path/to/manual_dir/trajectory.h5
        """
    RELEASE_NOTES = {
        "1.0.0": "Initial ManiSkill RLDS conversion.",
    }

    # Here is to define the dataset structure and types. It should match what we generate in _generate_examples.
    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "episode_metadata": {
                        "episode_id": tfds.features.Text(),
                        "file_path": tfds.features.Text(),
                        "task_name": tfds.features.Text(),
                    },
                    "steps": tfds.features.Dataset(
                        {
                            "observation": {
                                "image": tfds.features.Image(
                                    shape=(128, 128, 3),
                                    dtype=np.uint8,
                                    encoding_format="png",
                                ),
                                "wrist_image": tfds.features.Image(
                                    shape=(128, 128, 3),
                                    dtype=np.uint8,
                                    encoding_format="png",
                                ),
                                "has_wrist_image": tfds.features.Scalar(dtype=np.bool_),
                                "depth": tfds.features.Tensor(
                                    shape=(128, 128, 1),
                                    dtype=np.float32,
                                  
                                ),
                                "has_depth": tfds.features.Scalar(dtype=np.bool_),
                                "wrist_depth": tfds.features.Tensor(
                                    shape=(128, 128, 1),
                                    dtype=np.float32,
                                  
                                ),
                                "has_wrist_depth": tfds.features.Scalar(dtype=np.bool_),

                                # make sure the dimensions and types here match the result you get from inspect_h5.py. 
                                "qpos": tfds.features.Tensor(
                                    shape=(9,),
                                    dtype=np.float32,
                                  
                                ),
                                "qvel": tfds.features.Tensor(
                                    shape=(9,),
                                    dtype=np.float32,
                            
                                ),
                                "tcp_pose": tfds.features.Tensor(
                                    shape=(7,),
                                    dtype=np.float32,
                                ),

                                "is_grasped": tfds.features.Scalar(dtype=np.bool_),
                                "goal_pos": tfds.features.Tensor(
                                    shape=(3,),
                                    dtype=np.float32,
                                ),
                            },
                            "action": tfds.features.Tensor(
                                shape=(4,),
                                dtype=np.float32,
                                
                            ),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(),
                        }
                    ),
                }
            )
        )
    
    # This is where we read the H5 files, extract data, and yield episodes. We also split the dataset into train/val here.
    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        manual_dir = Path(dl_manager.manual_dir)
        h5_files = sorted(manual_dir.glob("*.h5"))

        if not h5_files:
            raise FileNotFoundError(
                f"No .h5 files found in manual_dir: {manual_dir}. "
                "Put replayed ManiSkill H5 files there."
            )

        episodes = self._collect_episodes(h5_files)
        rng = np.random.default_rng(0)
        rng.shuffle(episodes)

        num_episodes = len(episodes)
        num_train = int(num_episodes * 0.9)

        train_episodes = episodes[:num_train]
        val_episodes = episodes[num_train:]

        return {
            "train": self._generate_examples(train_episodes),
            "val": self._generate_examples(val_episodes),
        }
    
    # This function collects all trajectory keys from the given H5 file and return a list of(h5_path, traj_key) tuples.
    def _collect_episodes(self, h5_files):
        episodes = []
        for h5_path in h5_files:
            with h5py.File(h5_path, "r") as f:
                traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
                for traj_key in traj_keys:
                    episodes.append((h5_path, traj_key))
        return episodes


    # This function reads a dataset from the given group and relative path.
    def _read_dataset(self, group, rel_path):
        obj = group
        for part in rel_path.split("/"):
            obj = obj[part]
        return obj[:]

    # This function checks if the given relative path exists in the group.
    def _has_path(self, group, rel_path):
        obj = group
        for part in rel_path.split("/"):
            if part not in obj:
                return False
            obj = obj[part]
        return True

    # This function tries to read datasets from the given list of relative paths and returns the first one that exists, along with the path. 
    # it is used to check if wrist camera data and depth data exist, since user may have different versions of the dataset with different sensor configurations.
    def _read_first_existing_dataset(self, group, rel_paths):
        for rel_path in rel_paths:
            if self._has_path(group, rel_path):
                return self._read_dataset(group, rel_path), rel_path
        return None, None

    # Ensure depth has shape [..., H, W, 1] and dtype float32.]
    def _ensure_depth_channel(self, depth):
        """Return depth as float32 [..., H, W, 1]."""
        depth = depth.astype(np.float32)
        if depth.ndim == 3:
            depth = depth[..., None]
        if depth.ndim != 4 or depth.shape[-1] != 1:
            raise ValueError(f"Expected depth shape [T, H, W] or [T, H, W, 1], got {depth.shape}")
        return depth

    # if depth data doesn't exist; return a zero array with the same height and width as RGB and 1 channel.
    def _zero_depth_like_rgb(self, rgb):
        return np.zeros((*rgb.shape[:-1], 1), dtype=np.float32)

    
    def _make_episode_id(self, h5_path, traj_key):
        raw = f"{h5_path}:{traj_key}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    # This is the main function that generates examples. It reads each trajectory, extracts relevant data, checks for consistency, and yields episodes in the expected format.
    # you can modify the data extraction and episode structure here if your dataset has different structure.
    # usually you only need to change goal_pose and is_grasped.
    def _generate_examples(self, episodes):
        for h5_path, traj_key in episodes:
            with h5py.File(h5_path, "r") as f:
                traj = f[traj_key]

                actions = traj["actions"][:].astype(np.float32)
                if "rewards" in traj:
                    rewards = traj["rewards"][:]
                else:
                    rewards = np.zeros(actions.shape[0], dtype=np.float32)
                terminated = traj["terminated"][:].astype(bool)
                truncated = traj["truncated"][:].astype(bool)

                rgb = self._read_dataset(traj, self.RGB_PATH)
                wrist_rgb, wrist_path = self._read_first_existing_dataset(
                    traj, self.WRIST_RGB_PATHS
                )
                has_wrist_image = wrist_rgb is not None
                depth, depth_path = self._read_first_existing_dataset(
                    traj, self.DEPTH_PATHS
                )
                has_depth = depth is not None
                wrist_depth, wrist_depth_path = self._read_first_existing_dataset(
                    traj, self.WRIST_DEPTH_PATHS
                )
                has_wrist_depth = wrist_depth is not None
                qpos = self._read_dataset(traj, self.QPOS_PATH)
                qvel = self._read_dataset(traj, self.QVEL_PATH)

                tcp_pose = self._read_dataset(traj, self.TCP_POSE_PATH)
                is_grasped = self._read_dataset(traj, self.IS_GRASPED_PATH).astype(bool)
                goal_pos = self._read_dataset(traj, self.GOAL_POS_PATH)

                num_steps = actions.shape[0]

                rgb = rgb[:num_steps]
                if has_wrist_image:
                    wrist_rgb = wrist_rgb[:num_steps]
                else:
                    wrist_rgb = np.zeros_like(rgb)
                if has_depth:
                    depth = self._ensure_depth_channel(depth[:num_steps])
                else:
                    depth = self._zero_depth_like_rgb(rgb)
                if has_wrist_depth:
                    wrist_depth = self._ensure_depth_channel(wrist_depth[:num_steps])
                else:
                    wrist_depth = self._zero_depth_like_rgb(rgb)
                qpos = qpos[:num_steps]
                qvel = qvel[:num_steps]
                tcp_pose = tcp_pose[:num_steps]
                is_grasped = is_grasped[:num_steps]
                goal_pos = goal_pos[:num_steps]
                rewards = rewards[:num_steps]
                terminated = terminated[:num_steps]
                truncated = truncated[:num_steps]

                if not (
                    rgb.shape[0]
                    == wrist_rgb.shape[0]
                    == depth.shape[0]
                    == wrist_depth.shape[0]
                    == qpos.shape[0]
                    == qvel.shape[0]
                    == tcp_pose.shape[0]
                    == is_grasped.shape[0]
                    == goal_pos.shape[0]
                    == rewards.shape[0]
                    == terminated.shape[0]
                    == truncated.shape[0]
                    == num_steps
                ):
                    raise ValueError(
                        f"Length mismatch in {h5_path}:{traj_key}. "
                        f"actions={actions.shape}, rgb={rgb.shape}, "
                        f"wrist_rgb={wrist_rgb.shape}, wrist_path={wrist_path}, "
                        f"depth={depth.shape}, depth_path={depth_path}, "
                        f"wrist_depth={wrist_depth.shape}, wrist_depth_path={wrist_depth_path}, "
                        f"qpos={qpos.shape}, qvel={qvel.shape}, "
                        f"tcp_pose={tcp_pose.shape}, is_grasped={is_grasped.shape}, "
                        f"goal_pos={goal_pos.shape}, "
                        f"rewards={rewards.shape}"
                    )

                steps = []
                for t in range(num_steps):
                    steps.append(
                        {
                            "observation": {
                                "image": rgb[t],
                                "wrist_image": wrist_rgb[t],
                                "has_wrist_image": has_wrist_image,
                                "depth": depth[t],
                                "has_depth": has_depth,
                                "wrist_depth": wrist_depth[t],
                                "has_wrist_depth": has_wrist_depth,
                                "qpos": qpos[t].astype(np.float32),
                                "qvel": qvel[t].astype(np.float32),
                                "tcp_pose": tcp_pose[t].astype(np.float32),
                                "is_grasped": bool(is_grasped[t]),
                                "goal_pos": goal_pos[t].astype(np.float32),
                            },
                            "action": actions[t],
                            "reward": rewards[t],
                            "discount": np.float32(1.0),
                            "is_first": t == 0,
                            "is_last": t == num_steps - 1,
                            "is_terminal": bool(terminated[t] or truncated[t]),
                            "language_instruction": self.LANGUAGE_INSTRUCTION,
                        }
                    )

                episode = {
                    "episode_metadata": {
                        "episode_id": self._make_episode_id(h5_path, traj_key),
                        "file_path": str(h5_path),
                        "task_name": self.TASK_NAME,
                    },
                    "steps": steps,
                }

                yield f"{h5_path.stem}_{traj_key}", episode

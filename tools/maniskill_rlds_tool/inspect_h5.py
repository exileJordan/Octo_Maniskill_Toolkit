''' 
This script is a utility for inspecting the structure of an H5 file, it prints out all the data road, shapes and dtypes in h5 file.
 After the inspection, you can check out the h5 file to see if it is legal.
 The follow is an example, including some of the output:
    traj_98/
    traj_98/actions: shape=(81, 4), dtype=float32
    traj_98/env_states/
    traj_98/env_states/actors/
    traj_98/env_states/actors/cube: shape=(82, 13), dtype=float32
    traj_98/env_states/actors/goal_site: shape=(82, 13), dtype=float32
    traj_98/env_states/actors/table-workspace: shape=(82, 13), dtype=float32
    traj_98/env_states/articulations/
    traj_98/env_states/articulations/panda: shape=(82, 31), dtype=float32
    traj_98/obs/
    traj_98/obs/agent/
    traj_98/obs/agent/qpos: shape=(82, 9), dtype=float32
    traj_98/obs/agent/qvel: shape=(82, 9), dtype=float32
    traj_98/obs/extra/
    traj_98/obs/extra/goal_pos: shape=(82, 3), dtype=float32
    traj_98/obs/extra/is_grasped: shape=(82,), dtype=bool
    traj_98/obs/extra/tcp_pose: shape=(82, 7), dtype=float32
    traj_98/obs/sensor_data/
    traj_98/obs/sensor_data/base_camera/
    traj_98/obs/sensor_data/base_camera/rgb: shape=(82, 128, 128, 3), dtype=uint8
    traj_98/obs/sensor_param/
    traj_98/obs/sensor_param/base_camera/
    traj_98/obs/sensor_param/base_camera/cam2world_gl: shape=(82, 4, 4), dtype=float32
    traj_98/obs/sensor_param/base_camera/extrinsic_cv: shape=(82, 3, 4), dtype=float32
    traj_98/obs/sensor_param/base_camera/intrinsic_cv: shape=(82, 3, 3), dtype=float32
    traj_98/success: shape=(81,), dtype=bool
    traj_98/terminated: shape=(81,), dtype=bool
    traj_98/truncated: shape=(81,), dtype=bool
This is important because you need to change mannually in the maniskill_convert_dataset.py using this information.
'''
import argparse
from pathlib import Path

import h5py


def describe_node(name, obj):
    """Print H5 group/dataset information."""
    if isinstance(obj, h5py.Dataset):
        print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
    elif isinstance(obj, h5py.Group):
        print(f"{name}/")


def inspect_h5(path: Path):
    with h5py.File(path, "r") as f:
        print(f"File: {path}")
        print("\nTop-level keys:")
        for key in f.keys():
            print(f"  {key}")

        print("\nAll groups and datasets:")
        f.visititems(describe_node)

        traj_keys = sorted(k for k in f.keys() if k.startswith("traj_"))
        print(f"\nFound {len(traj_keys)} trajectory groups")

        if not traj_keys:
            return

        traj_key = traj_keys[0]
        print(f"\nFirst trajectory: {traj_key}")
        traj = f[traj_key]

        candidate_paths = [
            "actions",
            "rewards",
            "success",
            "terminated",
            "truncated",
            "obs",
            "env_states",
        ]
        for rel_path in candidate_paths:
            if rel_path in traj:
                obj = traj[rel_path]
                if isinstance(obj, h5py.Dataset):
                    print(f"  {traj_key}/{rel_path}: shape={obj.shape}, dtype={obj.dtype}")
                else:
                    print(f"  {traj_key}/{rel_path}/")
            else:
                print(f"  missing: {traj_key}/{rel_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", type=Path)
    args = parser.parse_args()
    inspect_h5(args.h5_path)


if __name__ == "__main__":
    main()
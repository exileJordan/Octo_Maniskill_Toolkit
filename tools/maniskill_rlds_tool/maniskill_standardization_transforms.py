"""Standardization transforms for custom ManiSkill RLDS datasets."""

from typing import Any, Dict

import tensorflow as tf


def maniskill_proprio_qpos_qvel_tcp_pose(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    """Create Octo proprio from robot-only atomic observation fields.

    Proprio order:
        qpos, qvel, tcp_pose, goal_pos

    Note:
        goal_pos is task conditioning, not robot proprio. We include it here
        in the first version to work with Octo's existing proprio path.
    """
    obs = trajectory["observation"]
    trajectory["observation"]["proprio"] = tf.concat(
        [
            tf.cast(obs["qpos"], tf.float32),
            tf.cast(obs["qvel"], tf.float32),
            tf.cast(obs["tcp_pose"], tf.float32),
            tf.cast(obs["goal_pos"], tf.float32),
        ],
        axis=-1,
    )
    return trajectory


def maniskill_proprio_qpos_tcp_pos(
    trajectory: Dict[str, Any]
) -> Dict[str, Any]:
    """Alternative conditioning using qpos, TCP xyz position, and goal_pos.

    This is useful for ablation. You can switch to this transform without
    rebuilding the TFDS dataset.
    """
    obs = trajectory["observation"]
    trajectory["observation"]["proprio"] = tf.concat(
        [
            tf.cast(obs["qpos"], tf.float32),
            tf.cast(obs["tcp_pose"][:, :3], tf.float32),
            tf.cast(obs["goal_pos"], tf.float32),
        ],
        axis=-1,
    )
    return trajectory

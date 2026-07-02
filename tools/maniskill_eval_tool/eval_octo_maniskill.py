import argparse
import csv
import json
from collections import deque
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import jax
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

import mani_skill.envs  # noqa: F401

from octo.model.octo_model import OctoModel
from octo.utils.train_callbacks import supply_rng

@dataclass(frozen=True)
class ManiSkillEvalConfig:
    """Configuration values that must match the Octo finetuning setup.

    Keep the environment, observation, proprioception, and action settings aligned
    with the ManiSkill RLDS dataset used to finetune the checkpoint.
    """

    env_id: str = "PickCube-v1"
    obs_mode: str = "rgb"
    control_mode: str = "pd_ee_delta_pos"
    render_mode: str = "rgb_array"

    image_size: tuple[int, int] = (256, 256)
    window_size: int = 4
    use_wrist_image: bool = False

    qpos_dim: int = 9
    qvel_dim: int = 9
    tcp_pose_dim: int = 7
    goal_pos_dim: int = 3
    action_dim: int = 4

    language_instruction: str = (
        "Pick up the object and move it to a goal position."
    )

    @property
    def proprio_dim(self) -> int:
        return (
            self.qpos_dim
            + self.qvel_dim
            + self.tcp_pose_dim
            + self.goal_pos_dim
        )
    


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--checkpoint_step", type=int, default=None)
    parser.add_argument("--output_dir", default="eval_outputs/maniskill_octo/debug")

    parser.add_argument("--env_id", default="PickCube-v1")
    parser.add_argument("--obs_mode", default="rgb")
    parser.add_argument("--control_mode", default="pd_ee_delta_pos")
    parser.add_argument("--render_mode", default="rgb_array")

    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_episode_steps", type=int, default=200)

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--window_size", type=int, default=4)
    parser.add_argument("--use_wrist_image", action="store_true") # default False, can be set to True if the model was trained with wrist images
    parser.add_argument("--save_video", action="store_true")# default False, can be set to True to save videos of the rollouts

    parser.add_argument(
        "--stage",
        choices=["env", "obs", "model", "policy", "rollout", "eval"],
        default="eval",
    )
    return parser.parse_args()

'''Ensure the ManiSkill action space matches the model action dimension.'''
def check_action_space(env, expected_action_dim: int) -> None:
    
    action_shape = env.action_space.shape
    if len(action_shape) != 1:
        raise ValueError(f"Expected 1D action space, got {action_shape}")
    if action_shape[0] != expected_action_dim:
        raise ValueError(
            "Action dimension mismatch: "
            f"env has {action_shape[0]}, model expects {expected_action_dim}. "
            "Check ManiSkill control_mode and finetuning action_dim."
        )


'''Create a ManiSkill environment from the evaluation config.'''
def make_maniskill_env(config: ManiSkillEvalConfig):
    env = gym.make(
        config.env_id,
        obs_mode=config.obs_mode,
        control_mode=config.control_mode,
        render_mode=config.render_mode,
    )
    return env


def print_tree(x, prefix: str = "") -> None:
    if isinstance(x, dict):
        for key, value in x.items():
            next_prefix = f"{prefix}/{key}" if prefix else key
            print_tree(value, next_prefix)
    else:
        shape = getattr(x, "shape", None)
        dtype = getattr(x, "dtype", None)
        print(f"{prefix}: shape={shape}, dtype={dtype}")

# def run_env_stage(config: ManiSkillEvalConfig) -> None:
#     env = make_maniskill_env(config)
#     check_action_space(env, config.action_dim)
#     random_env_smoke_test(env)
#     env.close()

def build_config_from_args(args) -> ManiSkillEvalConfig:
    return ManiSkillEvalConfig(
        env_id=args.env_id,
        obs_mode=args.obs_mode,
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        image_size=(args.image_size, args.image_size),
        window_size=args.window_size,
        use_wrist_image=args.use_wrist_image,
    )

# run a random rollout in the environment and print the observation tree, 
# to make sure the maniskill environment is working and the observation structure is the same as the octo.
# def random_env_smoke_test(env, max_steps: int = 20) -> None:
#     obs, info = env.reset(seed=0)
#     print("Observation tree:")
#     print_tree(obs)
#     print("action_space:", env.action_space)
#     print("reset info keys:", list(info.keys()))

#     episode_return = 0.0
#     for step in range(max_steps):
#         action = env.action_space.sample()
#         obs, reward, terminated, truncated, info = env.step(action)
#         episode_return += float(to_numpy(reward).reshape(-1)[0])
#         if terminated or truncated:
#             break

#     print("random rollout steps:", step + 1)
#     print("random rollout return:", episode_return)
#     print("last info keys:", list(info.keys()))


def get_by_path(tree: dict, path: str):
    """Return a nested value from a slash-separated dictionary path.

    Example:
        get_by_path(obs, "sensor_data/base_camera/rgb") is equivalent to
        obs["sensor_data"]["base_camera"]["rgb"].
    """
    value = tree
    for part in path.split("/"):
        value = value[part]
    return value


def to_numpy(x) -> np.ndarray:
    """Convert array-like inputs to a CPU NumPy array.

    PyTorch tensors may live on CUDA devices, so they must be copied to CPU
    before NumPy conversion.
    """
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def squeeze_env_dim(x) -> np.ndarray:
    """Remove ManiSkill's leading single-environment dimension.

    ManiSkill often returns observations with shape [1, ...] for single-env
    evaluation, while Octo expects unbatched per-frame values.
    """
    x = to_numpy(x)
    if x.ndim > 0 and x.shape[0] == 1:
        x = x[0]
    return x


def resize_uint8_image(image, image_size: tuple[int, int]) -> np.ndarray:
    image = squeeze_env_dim(image)

    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)

    height, width = image_size
    if cv2 is not None:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    else:
        image = np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR))

    if image.shape[-1] != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got {image.shape}")

    return image


class ManiSkillObservationAdapter:
    """Convert ManiSkill observations into single-frame Octo observations.

    The adapter normalizes the boundary between ManiSkill and Octo: ManiSkill
    may return torch tensors with a leading single-environment dimension, while
    Octo expects unbatched NumPy arrays. The proprio vector order must match the
    RLDS standardization used during finetuning.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        use_wrist_image: bool,
        base_rgb_path: str = "sensor_data/base_camera/rgb",
        wrist_rgb_path: str = "sensor_data/hand_camera/rgb",
        qpos_path: str = "agent/qpos",
        qvel_path: str = "agent/qvel",
        tcp_pose_path: str = "extra/tcp_pose",
        goal_pos_path: str = "extra/goal_pos",
    ):
        self.image_size = image_size
        self.use_wrist_image = use_wrist_image
        self.base_rgb_path = base_rgb_path
        self.wrist_rgb_path = wrist_rgb_path
        self.qpos_path = qpos_path
        self.qvel_path = qvel_path
        self.tcp_pose_path = tcp_pose_path
        self.goal_pos_path = goal_pos_path

    def __call__(self, obs: dict) -> dict:
        """Return a single-frame Octo observation dictionary.

        Args:
            obs: Raw ManiSkill observation dictionary from env.reset() or
                env.step().

        Returns:
            A dictionary containing image_primary, optional image_wrist, and
            proprio. History stacking and timestep_pad_mask are handled later
            by ObservationHistory.
        """
        image_primary = resize_uint8_image(
            get_by_path(obs, self.base_rgb_path),
            self.image_size,
        )

        qpos = squeeze_env_dim(get_by_path(obs, self.qpos_path)).astype(np.float32)
        qvel = squeeze_env_dim(get_by_path(obs, self.qvel_path)).astype(np.float32)
        tcp_pose = squeeze_env_dim(get_by_path(obs, self.tcp_pose_path)).astype(
            np.float32
        )
        goal_pos = squeeze_env_dim(get_by_path(obs, self.goal_pos_path)).astype(
            np.float32
        )

        proprio = np.concatenate([qpos, qvel, tcp_pose, goal_pos], axis=-1)

        octo_obs = {
            "image_primary": image_primary,
            "proprio": proprio,
        }

        if self.use_wrist_image:
            image_wrist = resize_uint8_image(
                get_by_path(obs, self.wrist_rgb_path),
                self.image_size,
            )
            octo_obs["image_wrist"] = image_wrist

        return octo_obs


class ObservationHistory:
    """Maintain a fixed-length Octo observation history for inference.

    Octo is finetuned with a fixed window size. During online rollout, this
    helper stacks single-frame observations into that window and creates the
    timestep_pad_mask expected by the model.
    """

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.history = deque(maxlen=window_size)
        self.num_obs = 0

    def clear(self) -> None:
        """Clear all cached observations at the start of a new episode."""
        self.history.clear()
        self.num_obs = 0

    def reset(self, first_octo_obs: dict) -> dict:
        """Initialize history from the first observation of an episode."""
        self.clear()
        self.num_obs = 1
        for _ in range(self.window_size):
            self.history.append(first_octo_obs)
        return self._stack_and_mask()

    def step(self, octo_obs: dict) -> dict:
        """Append a new observation and return the stacked history."""
        self.num_obs += 1
        self.history.append(octo_obs)
        return self._stack_and_mask()

    def _stack_and_mask(self) -> dict:
        """Stack cached observations and mark padded timesteps as invalid."""
        if not self.history:
            raise RuntimeError("Observation history is empty. Call reset() first.")

        stacked = {
            key: np.stack([obs[key] for obs in self.history], axis=0)
            for key in self.history[0]
        }

        pad_length = self.window_size - min(self.num_obs, self.window_size)
        timestep_pad_mask = np.ones(self.window_size, dtype=bool)
        timestep_pad_mask[:pad_length] = False
        stacked["timestep_pad_mask"] = timestep_pad_mask
        return stacked

# test the observation adapter and history with a random rollout, and print the shapes and dtypes of the resulting octo_obs
# to make sure the input of octo model is correct. This can help debug any issues with the observation processing before integrating the model and policy stages.
# def run_obs_stage(config: ManiSkillEvalConfig) -> None:
#     env = make_maniskill_env(config)
#     obs, info = env.reset(seed=0)

#     obs_adapter = ManiSkillObservationAdapter(
#         image_size=config.image_size,
#         use_wrist_image=config.use_wrist_image,
#     )
#     history = ObservationHistory(window_size=config.window_size)

#     single_octo_obs = obs_adapter(obs)
#     octo_obs = history.reset(single_octo_obs)

#     print("single image_primary:", single_octo_obs["image_primary"].shape, single_octo_obs["image_primary"].dtype)
#     print("single proprio:", single_octo_obs["proprio"].shape, single_octo_obs["proprio"].dtype)
#     print("window image_primary:", octo_obs["image_primary"].shape, octo_obs["image_primary"].dtype)
#     print("window proprio:", octo_obs["proprio"].shape, octo_obs["proprio"].dtype)
#     print("timestep_pad_mask:", octo_obs["timestep_pad_mask"])

#     if octo_obs["proprio"].shape[-1] != config.proprio_dim:
#         raise ValueError(
#             f"Proprio dim mismatch: got {octo_obs['proprio'].shape[-1]}, "
#             f"expected {config.proprio_dim}"
#         )

#     env.close()

#-----------------------------Load Octo and Build the Task-----------------------------
'''load octo model from checkpoint path, and return the model.'''
def load_octo_model(checkpoint_path: str, step: int | None=None) -> OctoModel:
    if checkpoint_path is None:
        raise ValueError("--checkpoint_path is required for Octo model stages.")
    return OctoModel.load_pretrained(checkpoint_path, step=step)

"""create a language task for the maniskill evaluation"""
def create_language_task(model: OctoModel, language_instruction: str):
    return model.create_tasks(texts=[language_instruction])
    
"""add a batch dimension to the octo obs, so that it can be passed to the octo finetuned model.
from [window_size, ...] to [batch_num, window_size, ...]
"""
def add_batch_dim(tree):
    return jax.tree_map(lambda x: x[None], tree)

# run a dry run of the octo model with the maniskill environment, and print the shape and min/max of the sampled actions.
# def octo_dry_run(model, obs_adapter, history, env, language_instruction: str):
#     obs, info = env.reset(seed=0)
#     single_octo_obs = obs_adapter(obs)
#     octo_obs = history.reset(single_octo_obs)
#     batched_obs = add_batch_dim(octo_obs)
#     task = create_language_task(model, language_instruction)

#     actions = model.sample_actions(
#         batched_obs,
#         task,
#         unnormalization_statistics=model.dataset_statistics["action"],
#     )
#     actions = np.asarray(actions)

#     print("sample_actions shape:", actions.shape)
#     print("sample_actions min/max:", actions.min(), actions.max())

#     if np.isnan(actions).any():
#         raise ValueError("Octo produced NaN actions.")

#     return actions

# test the whether the transformed octo obs can be passed to the octo model, whether the sampled actions are valid.
# def run_model_stage(config: ManiSkillEvalConfig, checkpoint_path: str) -> None:
#     env = make_maniskill_env(config)
#     model = load_octo_model(checkpoint_path)
#     obs_adapter = ManiSkillObservationAdapter(
#         image_size=config.image_size,
#         use_wrist_image=config.use_wrist_image,
#     )
#     history = ObservationHistory(window_size=config.window_size)
#     octo_dry_run(
#         model=model,
#         obs_adapter=obs_adapter,
#         history=history,
#         env=env,
#         language_instruction=config.language_instruction,
#     )
#     env.close()

class ManiSkillActionAdapter:
    """Validate and clip Octo actions before passing them to ManiSkill.

    ManiSkill exposes a Gymnasium action_space, typically a Box for continuous
    control. The Box defines the legal action shape and per-dimension bounds via
    action_space.low and action_space.high. example: action_space: Box(-1.0, 1.0, (4,), float32) 
    This adapter enforces those constraints so env.step(action) receives a valid action.
    """

    def __init__(self, action_space):
        self.action_space = action_space

    def __call__(self, action) -> np.ndarray:
        """Return a float32 action clipped to the environment action bounds.

        Args:
            action: Raw action predicted by Octo after action selection and
                unnormalization.

        Returns:
            A NumPy action with shape matching env.action_space.shape. Values
            are clipped to [action_space.low, action_space.high].

        Raises:
            ValueError: If the action contains NaN or has the wrong shape.
        """
        action = np.asarray(action, dtype=np.float32)

        if np.isnan(action).any():
            raise ValueError("Action contains NaN.")

        expected_shape = self.action_space.shape
        if action.shape != expected_shape:
            raise ValueError(
                f"Action shape mismatch: got {action.shape}, "
                f"expected {expected_shape}."
            )

        low = np.asarray(self.action_space.low, dtype=np.float32)
        high = np.asarray(self.action_space.high, dtype=np.float32)
        return np.clip(action, low, high)

def select_first_action(actions):
    """Select the single action to execute from Octo's action prediction.

    Octo may return an action chunk, and some heads may include a sample
    dimension. This helper implements simple receding-horizon control by taking
    the first sample, first batch item, and first action in the predicted
    horizon, producing one action for ManiSkill env.step().

    Args:
        actions: Action prediction returned by Octo.

    Returns:
        A single action with shape [action_dim].

    Raises:
        ValueError: If the action prediction rank is unsupported.
    """
    actions = np.asarray(actions)

    if actions.ndim == 4:
        # [n_samples, batch, horizon, action_dim]
        return actions[0, 0, 0]

    if actions.ndim == 3:
        # [batch, horizon, action_dim]
        return actions[0, 0]

    if actions.ndim == 2:
        # [batch, action_dim]
        return actions[0]

    raise ValueError(f"Unsupported action output shape: {actions.shape}")

class OctoManiSkillPolicy:
    """Wrap an Octo model as a callable ManiSkill rollout policy."""

    def __init__(
        self,
        model,
        obs_adapter,
        history,
        action_adapter,
        language_instruction: str,
    ):
        self.model = model
        self.obs_adapter = obs_adapter
        self.history = history
        self.action_adapter = action_adapter
        self.task = model.create_tasks(texts=[language_instruction])
        self._history_initialized = False

        self.policy_fn = supply_rng(
            partial(
                model.sample_actions,
                unnormalization_statistics=model.dataset_statistics["action"],
            )
        )

    def reset_episode_state(self) -> None:
        """Reset cached observation history at the start of an episode."""
        self.history.clear()
        self._history_initialized = False

    def __call__(self, maniskill_obs):
        """Predict one valid ManiSkill action from the current observation."""
        single_octo_obs = self.obs_adapter(maniskill_obs)
        if self._history_initialized:
            octo_obs = self.history.step(single_octo_obs)
        else:
            octo_obs = self.history.reset(single_octo_obs)
            self._history_initialized = True

        batched_obs = add_batch_dim(octo_obs)
        actions = self.policy_fn(batched_obs, self.task)
        action = select_first_action(actions)
        return self.action_adapter(action)
    
def to_python_bool(x) -> bool:
    """Convert scalar-like tensors or arrays to a Python bool."""
    x = np.asarray(x)
    return bool(x.reshape(-1)[0])


def read_success(info: dict, previous_success: bool = False) -> bool:
    """Read and accumulate the task success flag from ManiSkill info."""
    if "success" in info:
        return previous_success or to_python_bool(info["success"])
    if "is_success" in info:
        return previous_success or to_python_bool(info["is_success"])
    return previous_success

def get_frame_from_maniskill_obs(obs: dict, obs_adapter) -> np.ndarray:
    """Extract the primary image used by the policy for video logging."""
    single_octo_obs = obs_adapter(obs)
    return single_octo_obs["image_primary"]

def run_one_episode(
    env,
    policy,
    seed: int,
    max_episode_steps: int,
    episode_idx: int,
    save_video: bool = False,
    output_dir: str | None = None,
):
    """Run one policy rollout episode and return episode-level metrics,generating the video and recording success rate."""
    obs, info = env.reset(seed=seed)
    policy.reset_episode_state()

    episode_return = 0.0
    success = False
    frames = []
    terminated = False
    truncated = False

    
    for step in range(max_episode_steps):
        frames.append(get_frame_from_maniskill_obs(obs, policy.obs_adapter))

        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        episode_return += float(to_numpy(reward).reshape(-1)[0])
        success = read_success(info, success)

        if terminated or truncated:
            break

    video_path = None
    if save_video and output_dir is not None:
        video_path = save_episode_video(
            frames=frames,
            output_dir=output_dir,
            episode_idx=episode_idx,
            seed=seed,
        )

    return {
        "episode_id": episode_idx,
        "seed": seed,
        "success": success,
        "episode_return": episode_return,
        "episode_length": len(frames),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "video_path": video_path,
    }

def build_policy(config: ManiSkillEvalConfig, checkpoint_path: str, env):
    """Build an Octo policy wired to the current ManiSkill environment.

    This helper centralizes policy construction: it loads the finetuned Octo
    checkpoint, creates the observation/history/action adapters, and returns a
    callable policy that maps ManiSkill observations to valid environment
    actions.
    """
    model = load_octo_model(checkpoint_path)
    obs_adapter = ManiSkillObservationAdapter(
        image_size=config.image_size,
        use_wrist_image=config.use_wrist_image,
    )
    history = ObservationHistory(window_size=config.window_size)
    action_adapter = ManiSkillActionAdapter(env.action_space)
    return OctoManiSkillPolicy(
        model=model,
        obs_adapter=obs_adapter,
        history=history,
        action_adapter=action_adapter,
        language_instruction=config.language_instruction,
    )

def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def save_episode_video(
    frames: list[np.ndarray],
    output_dir: str,
    episode_idx: int,
    seed: int,
    fps: int = 10,
) -> str:
    video_dir = Path(output_dir) / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)

    video_path = video_dir / f"episode_{episode_idx:04d}_seed_{seed}.mp4"
    imageio.mimsave(video_path, frames, fps=fps)
    return str(video_path)



EPISODE_FIELDS = [
    "episode_id",
    "seed",
    "success",
    "episode_return",
    "episode_length",
    "terminated",
    "truncated",
    "video_path",
]


def append_episode_result(result: dict, output_dir: str) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "episodes.csv"
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: result.get(key) for key in EPISODE_FIELDS})

def compute_metrics(results: list[dict]) -> dict:
    successes = np.asarray([r["success"] for r in results], dtype=np.float32)
    returns = np.asarray([r["episode_return"] for r in results], dtype=np.float32)
    lengths = np.asarray([r["episode_length"] for r in results], dtype=np.float32)

    return {
        "num_episodes": int(len(results)),
        "num_success": int(successes.sum()),
        "success_rate": float(successes.mean()) if len(successes) else 0.0,
        "avg_return": float(returns.mean()) if len(returns) else 0.0,
        "std_return": float(returns.std()) if len(returns) else 0.0,
        "avg_episode_length": float(lengths.mean()) if len(lengths) else 0.0,
        "std_episode_length": float(lengths.std()) if len(lengths) else 0.0,
    }

def run_eval_stage(
    config: ManiSkillEvalConfig,
    checkpoint_path: str,
    output_dir: str,
    num_episodes: int,
    seed: int,
    max_episode_steps: int,
    save_video: bool,
) -> None:
    """Run full ManiSkill rollout evaluation for a finetuned Octo checkpoint.

    This is the main evaluation entry point. It creates the environment and
    policy, runs multiple seeded episodes, appends each episode result to CSV,
    and writes summary metrics to disk. To keep output size bounded, videos are
    saved only for the first few episodes when save_video is enabled.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_json(config.__dict__, output_dir / "config.json")

    env = make_maniskill_env(config)
    check_action_space(env, config.action_dim)
    policy = build_policy(config, checkpoint_path, env)

    results = []
    for episode_idx in range(num_episodes):
        episode_seed = seed + episode_idx
        result = run_one_episode(
            env=env,
            policy=policy,
            seed=episode_seed,
            max_episode_steps=max_episode_steps,
            episode_idx=episode_idx,
            save_video=save_video and episode_idx < 3,
            output_dir=str(output_dir),
        )
        append_episode_result(result, str(output_dir))
        results.append(result)
        print(result)

    metrics = compute_metrics(results)
    save_json(metrics, output_dir / "metrics.json")
    print("metrics:", metrics)
    env.close()


def main():
    args = parse_args()
    config = build_config_from_args(args)
    if args.stage == "eval":
        run_eval_stage(
            config=config,
            checkpoint_path=args.checkpoint_path,
            output_dir=args.output_dir,
            num_episodes=args.num_episodes,
            seed=args.seed,
            max_episode_steps=args.max_episode_steps,
            save_video=args.save_video,
        )
    else:
        raise ValueError(f"Unknown stage: {args.stage}")

if __name__ == "__main__":
    main()

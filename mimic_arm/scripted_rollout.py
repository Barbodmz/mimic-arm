"""Roll the scripted pick-and-handover in gym-aloha.

The end-effector sim follows ``PickAndTransferPolicy``. Joint positions from
that rollout are replayed in the joint-position sim, which is the recording
the original ACT script saves. Images are skipped unless the caller asks for
the top camera, because the reachability survey only needs the reward.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

from mimic_arm.scripted_policy import EPISODE_LEN, PickAndTransferPolicy
from mimic_arm.wider_spawn import STIFF_WELD_FROM, STIFF_WELD_TO

SUCCESS_REWARD = 4


def _observation_without_images(task, physics) -> OrderedDict:
    obs = OrderedDict()
    obs["qpos"] = task.get_qpos(physics)
    obs["qvel"] = task.get_qvel(physics)
    obs["env_state"] = task.get_env_state(physics)
    obs["mocap_pose_left"] = np.concatenate(
        [physics.data.mocap_pos[0], physics.data.mocap_quat[0]]
    ).copy()
    obs["mocap_pose_right"] = np.concatenate(
        [physics.data.mocap_pos[1], physics.data.mocap_quat[1]]
    ).copy()
    obs["gripper_ctrl"] = physics.data.ctrl.copy()
    return obs


def _joint_observation(task, physics, *, render_top: bool) -> OrderedDict:
    obs = OrderedDict()
    obs["qpos"] = task.get_qpos(physics)
    obs["qvel"] = task.get_qvel(physics)
    obs["env_state"] = task.get_env_state(physics)
    if render_top:
        obs["images"] = {"top": physics.render(height=480, width=640, camera_id="top")}
    return obs


class _PoseBox:
    def __init__(self, pose: np.ndarray) -> None:
        self.pose = np.asarray(pose, dtype=np.float64).copy()


def _patch_sample_box_pose(box: _PoseBox):
    import gym_aloha.tasks.sim_end_effector as ee

    original = ee.sample_box_pose

    def _fixed(seed=None):
        del seed
        return box.pose.copy()

    ee.sample_box_pose = _fixed
    return original


def _stiff_weld_xml() -> Path:
    """Write a copy of the EE scene whose mocap weld actually tracks.

    The file has to sit next to the other assets because the XML includes
    them by relative path. The caller deletes it.
    """
    import gym_aloha

    assets = Path(gym_aloha.__file__).parent / "assets"
    source = assets / "bimanual_viperx_end_effector_transfer_cube.xml"
    text = source.read_text()
    if STIFF_WELD_FROM not in text:
        raise RuntimeError("gym-aloha's end-effector weld markup changed; the stiff weld was not applied.")
    destination = assets / "_mimic_arm_ee_stiff.xml"
    destination.write_text(text.replace(STIFF_WELD_FROM, STIFF_WELD_TO))
    return destination


def _make_ee_env(pose_box: _PoseBox, *, stiff_weld: bool):
    from dm_control import mujoco
    from dm_control.rl import control
    from gym_aloha.constants import DT
    from gym_aloha.tasks.sim_end_effector import TransferCubeEndEffectorTask
    import gym_aloha

    xml_path = _stiff_weld_xml() if stiff_weld else (
        Path(gym_aloha.__file__).parent / "assets" / "bimanual_viperx_end_effector_transfer_cube.xml"
    )
    original_obs = TransferCubeEndEffectorTask.get_observation

    def fast_obs(self, physics):
        return _observation_without_images(self, physics)

    TransferCubeEndEffectorTask.get_observation = fast_obs
    try:
        physics = mujoco.Physics.from_xml_path(str(xml_path))
        task = TransferCubeEndEffectorTask(random=None)
        env = control.Environment(
            physics, task, time_limit=20, control_timestep=DT, n_sub_steps=None, flat_observation=False
        )
    except Exception:
        TransferCubeEndEffectorTask.get_observation = original_obs
        if stiff_weld and xml_path.exists():
            xml_path.unlink()
        raise
    return env, original_obs, xml_path if stiff_weld else None


def _make_joint_env(*, render_top: bool):
    from dm_control import mujoco
    from dm_control.rl import control
    from gym_aloha.constants import DT
    from gym_aloha.tasks.sim import TransferCubeTask
    import gym_aloha

    xml_path = Path(gym_aloha.__file__).parent / "assets" / "bimanual_viperx_transfer_cube.xml"
    original_obs = TransferCubeTask.get_observation

    def obs(self, physics):
        return _joint_observation(self, physics, render_top=render_top)

    TransferCubeTask.get_observation = obs
    try:
        physics = mujoco.Physics.from_xml_path(str(xml_path))
        task = TransferCubeTask(random=None)
        env = control.Environment(
            physics, task, time_limit=20, control_timestep=DT, n_sub_steps=None, flat_observation=False
        )
    except Exception:
        TransferCubeTask.get_observation = original_obs
        raise
    return env, original_obs


def _restore(ee_obs, joint_obs, stiff_path: Path | None, original_sampler) -> None:
    from gym_aloha.tasks.sim import TransferCubeTask
    from gym_aloha.tasks.sim_end_effector import TransferCubeEndEffectorTask
    import gym_aloha.tasks.sim_end_effector as ee

    TransferCubeEndEffectorTask.get_observation = ee_obs
    if joint_obs is not None:
        TransferCubeTask.get_observation = joint_obs
    ee.sample_box_pose = original_sampler
    if stiff_path is not None and stiff_path.exists():
        stiff_path.unlink()


def rollout_scripted(
    pose: np.ndarray,
    *,
    stiff_weld: bool,
    render_top: bool = False,
) -> dict:
    """Run one scripted episode and replay it in the joint-position sim.

    Returns the joint-replay reward, the frames to save, and whether the
    handover finished. ``frames`` is empty when ``render_top`` is false.
    """
    from gym_aloha.constants import normalize_puppet_gripper_position
    from gym_aloha.tasks.sim import BOX_POSE

    pose = np.asarray(pose, dtype=np.float64).copy()
    pose_box = _PoseBox(pose)
    original_sampler = _patch_sample_box_pose(pose_box)
    ee_env = None
    joint_env = None
    ee_obs = None
    joint_obs = None
    stiff_path = None
    try:
        ee_env, ee_obs, stiff_path = _make_ee_env(pose_box, stiff_weld=stiff_weld)
        timestep = ee_env.reset()
        policy = PickAndTransferPolicy()
        ee_steps = [timestep]
        ee_reward = 0
        for _ in range(EPISODE_LEN):
            timestep = ee_env.step(policy(timestep.observation))
            ee_steps.append(timestep)
            ee_reward = max(ee_reward, int(timestep.reward or 0))

        joint_commands = [step.observation["qpos"].copy() for step in ee_steps]
        gripper = [step.observation["gripper_ctrl"].copy() for step in ee_steps]
        for joints, ctrl in zip(joint_commands, gripper):
            joints[6] = normalize_puppet_gripper_position(ctrl[0])
            joints[13] = normalize_puppet_gripper_position(ctrl[2])

        joint_env, joint_obs = _make_joint_env(render_top=render_top)
        BOX_POSE[0] = pose.copy()
        timestep = joint_env.reset()
        replay = [timestep]
        joint_reward = 0
        for command in joint_commands:
            timestep = joint_env.step(command)
            replay.append(timestep)
            joint_reward = max(joint_reward, int(timestep.reward or 0))

        # Match record_sim_episodes.py: drop the last state and the last
        # command so each saved frame is the observation before that action.
        commands = joint_commands[:-1]
        states = replay[:-1]
        frames = []
        for command, state in zip(commands, states):
            frame = {
                "state": np.asarray(state.observation["qpos"], dtype=np.float32).copy(),
                "action": np.asarray(command, dtype=np.float32).copy(),
            }
            if render_top:
                frame["image"] = np.asarray(state.observation["images"]["top"]).copy()
            frames.append(frame)
        # The saved demo is the joint trajectory. It counts only when that
        # replay reaches reward 4 in the stock joint-position env.
        return {
            "ee_reward": ee_reward,
            "joint_reward": joint_reward,
            "success": joint_reward >= SUCCESS_REWARD,
            "frames": frames,
            "pose": pose.copy(),
        }
    finally:
        if ee_obs is not None:
            _restore(ee_obs, joint_obs, stiff_path, original_sampler)
        else:
            import gym_aloha.tasks.sim_end_effector as ee

            ee.sample_box_pose = original_sampler


def scripted_success(x: float, y: float, *, stiff_weld: bool = True) -> bool:
    """True when the scripted handover finishes at this cube spot."""
    from mimic_arm.wider_spawn import pose_from_xy

    result = rollout_scripted(pose_from_xy(x, y), stiff_weld=stiff_weld, render_top=False)
    return bool(result["success"])


def survey_reachable_spots() -> list[tuple[float, float]]:
    """Wider-grid spots whose joint replay succeeds in the stock eval env.

    The teacher runs with the higher-impedance mocap weld. ``last_joint_only``
    counts spots where that replay succeeded and the end-effector reward
    stayed below 4. Those spots stay in the returned list, because the
    recorded demo is what the stock env scores.
    """
    from mimic_arm.wider_spawn import candidate_spots, pose_from_xy

    spots = candidate_spots()
    reachable: list[tuple[float, float]] = []
    joint_only = 0
    for index, (x, y) in enumerate(spots, start=1):
        result = rollout_scripted(pose_from_xy(x, y), stiff_weld=True, render_top=False)
        if result["success"]:
            reachable.append((x, y))
            if result["ee_reward"] < SUCCESS_REWARD:
                joint_only += 1
        if index % 16 == 0 or index == len(spots):
            print(
                f"reachability {index}/{len(spots)}  reachable {len(reachable)}  "
                f"ee-short {joint_only}",
                flush=True,
            )
    survey_reachable_spots.last_joint_only = joint_only
    return reachable

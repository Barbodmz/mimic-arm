"""Recovery checkpoints, weights-only snapshots, and auto-resume.

These tests stay on CPU and do not import LeRobot. The forced-crash test
kills a subprocess with SIGKILL while a recovery save is in progress, then
starts the same keeper again and checks the step it resumes from.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mimic_arm.checkpoints import checkpoint_step, list_numbered_checkpoints, resolve_checkpoint
from mimic_arm.saving import (
    STAGING_NAME,
    CheckpointKeeper,
    default_num_workers,
    find_resume_config,
    read_training_step,
    reconcile_recovery,
)

import train


def write_fake_checkpoint(dest: Path, step: int, *, complete: bool = True) -> None:
    """A tiny stand-in for LeRobot's checkpoint tree."""
    pretrained = dest / "pretrained_model"
    pretrained.mkdir(parents=True, exist_ok=True)
    (pretrained / "config.json").write_text(json.dumps({"type": "act", "step": step}) + "\n")
    (pretrained / "model.safetensors").write_bytes(b"weights-for-step-%d" % step)
    (pretrained / "train_config.json").write_text(json.dumps({"steps": 60000}) + "\n")
    (pretrained / "processor.json").write_text("{}\n")
    (pretrained / "step_0.safetensors").write_bytes(b"norm-stats")
    if not complete:
        return
    state = dest / "training_state"
    state.mkdir(parents=True, exist_ok=True)
    (state / "training_step.json").write_text(json.dumps({"step": step}))
    (state / "optimizer_state.safetensors").write_bytes(b"adam-moments")
    (state / "optimizer_param_groups.json").write_text("{}\n")
    (state / "rng_state.safetensors").write_bytes(b"rng")
    (state / "scheduler_state.json").write_text("{}\n")
    (dest / ".mimic_arm_complete").write_text("ok\n")


def make_keeper(output: Path, **kwargs) -> CheckpointKeeper:
    options = dict(
        output_dir=output,
        total_steps=60_000,
        weights_every=10_000,
        min_free_bytes=1024,
        recovery_reserve_bytes=1024,
        weights_reserve_bytes=1024,
    )
    options.update(kwargs)
    return CheckpointKeeper(**options)


def save_step(keeper: CheckpointKeeper, step: int) -> bool:
    def populate(dest: Path) -> None:
        write_fake_checkpoint(dest, step)

    return keeper.save_recovery(step, populate)


class RotationAndWeightsTest(unittest.TestCase):
    def test_second_recovery_replaces_the_first_and_weights_stay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            keeper = make_keeper(output, weights_every=2_000)
            self.assertTrue(save_step(keeper, 2000))
            self.assertTrue(save_step(keeper, 4000))

            recovery = output / "checkpoints" / "recovery"
            self.assertEqual(read_training_step(recovery), 4000)
            self.assertFalse((output / "checkpoints" / STAGING_NAME).exists())
            self.assertFalse((output / "checkpoints" / "recovery_prev").exists())
            self.assertFalse((output / "checkpoints" / "002000").exists())
            self.assertFalse((output / "checkpoints" / "004000").exists())

            first = output / "checkpoints" / "weights" / "002000"
            second = output / "checkpoints" / "weights" / "004000"
            self.assertTrue((first / "pretrained_model" / "model.safetensors").is_file())
            self.assertTrue((second / "pretrained_model" / "config.json").is_file())
            self.assertFalse((first / "training_state").exists())
            self.assertFalse((second / "training_state" / "optimizer_state.safetensors").exists())

    def test_final_step_writes_weights_even_between_intervals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            keeper = make_keeper(output, total_steps=2500, weights_every=10_000)
            self.assertTrue(save_step(keeper, 2500))
            weights = output / "checkpoints" / "weights" / "002500" / "pretrained_model"
            self.assertTrue((weights / "train_config.json").is_file())
            self.assertTrue((weights / "processor.json").is_file())
            self.assertTrue((weights / "step_0.safetensors").is_file())

    def test_incomplete_staging_is_dropped_and_finished_staging_is_kept(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoints = Path(tmp) / "checkpoints"
            write_fake_checkpoint(checkpoints / "recovery", 20)
            write_fake_checkpoint(checkpoints / STAGING_NAME, 40, complete=False)
            restored = reconcile_recovery(checkpoints)
            self.assertEqual(read_training_step(restored), 20)
            self.assertFalse((checkpoints / STAGING_NAME).exists())

            # training_step.json exists, but the commit marker does not. That is
            # the window where LeRobot has not finished the optimizer file.
            write_fake_checkpoint(checkpoints / STAGING_NAME, 40)
            (checkpoints / STAGING_NAME / ".mimic_arm_complete").unlink()
            restored = reconcile_recovery(checkpoints)
            self.assertEqual(read_training_step(restored), 20)
            self.assertFalse((checkpoints / STAGING_NAME).exists())

            write_fake_checkpoint(checkpoints / STAGING_NAME, 40)
            restored = reconcile_recovery(checkpoints)
            self.assertEqual(read_training_step(restored), 40)
            self.assertFalse((checkpoints / STAGING_NAME).exists())
            self.assertFalse((checkpoints / "recovery_prev").exists())

    def test_crash_after_recovery_is_renamed_aside_keeps_the_newer_tree(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoints = Path(tmp) / "checkpoints"
            write_fake_checkpoint(checkpoints / "recovery", 20)
            write_fake_checkpoint(checkpoints / STAGING_NAME, 40)
            (checkpoints / "recovery").rename(checkpoints / "recovery_prev")
            self.assertFalse((checkpoints / "recovery").exists())

            restored = reconcile_recovery(checkpoints)
            self.assertEqual(read_training_step(restored), 40)
            self.assertFalse((checkpoints / "recovery_prev").exists())
            self.assertFalse((checkpoints / STAGING_NAME).exists())


class DiskGuardTest(unittest.TestCase):
    def test_low_space_skips_weights_but_still_updates_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            free = 900 * 1024 * 1024

            def disk_usage(_path: Path) -> SimpleNamespace:
                return SimpleNamespace(free=free)

            keeper = make_keeper(
                output,
                weights_every=10_000,
                min_free_bytes=1024**3,
                recovery_reserve_bytes=800 * 1024 * 1024,
                weights_reserve_bytes=250 * 1024 * 1024,
                disk_usage=disk_usage,
            )
            with mock.patch("sys.stdout"):
                self.assertTrue(save_step(keeper, 10_000))
            self.assertEqual(read_training_step(output / "checkpoints" / "recovery"), 10_000)
            self.assertFalse((output / "checkpoints" / "weights").exists())

    def test_too_little_space_skips_recovery_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            keeper = make_keeper(output, weights_every=2_000)
            self.assertTrue(save_step(keeper, 2000))

            def disk_usage(_path: Path) -> SimpleNamespace:
                return SimpleNamespace(free=100)

            keeper.disk_usage = disk_usage
            keeper.recovery_reserve_bytes = 800 * 1024 * 1024
            with mock.patch("sys.stdout"):
                self.assertFalse(save_step(keeper, 4000))
            self.assertEqual(read_training_step(output / "checkpoints" / "recovery"), 2000)
            self.assertTrue((output / "checkpoints" / "weights" / "002000").is_dir())
            self.assertFalse((output / "checkpoints" / "weights" / "004000").exists())


class LoadWeightsTest(unittest.TestCase):
    def test_evaluate_and_compare_see_weights_only_saves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            keeper = make_keeper(output, weights_every=2_000)
            save_step(keeper, 8_000)
            save_step(keeper, 10_000)
            # Recovery has moved on; the 8k weights must still load.
            write_fake_checkpoint(output / "checkpoints" / "recovery", 12_000)

            saved = list_numbered_checkpoints(output)
            steps = [checkpoint_step(path) for path in saved]
            self.assertEqual(steps, [8000, 10000, 12000])
            weights_8k = output / "checkpoints" / "weights" / "008000"
            self.assertIn(weights_8k, saved)
            self.assertNotIn(output / "checkpoints" / "recovery", saved[:2])

            loaded = Path(resolve_checkpoint(weights_8k))
            self.assertTrue((loaded / "config.json").is_file())
            self.assertTrue((loaded / "model.safetensors").is_file())
            self.assertTrue((loaded / "train_config.json").is_file())
            self.assertTrue((loaded / "processor.json").is_file())
            self.assertTrue((loaded / "step_0.safetensors").is_file())
            self.assertFalse((weights_8k / "training_state").exists())
            self.assertEqual(checkpoint_step(loaded), 8000)

            latest = Path(resolve_checkpoint(output))
            self.assertEqual(checkpoint_step(latest), 12000)
            self.assertEqual(
                resolve_checkpoint("lerobot/act_aloha_sim_transfer_cube_human"),
                "lerobot/act_aloha_sim_transfer_cube_human",
            )


class SymlinkAndResumeDiscoveryTest(unittest.TestCase):
    def test_symlink_oserror_is_not_fatal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            keeper = make_keeper(output)
            save_step(keeper, 20)
            last = output / "checkpoints" / "last"
            if last.is_symlink() or last.exists():
                last.unlink()

            with mock.patch.object(Path, "symlink_to", side_effect=OSError(1314, "A required privilege is not held")):
                keeper.refresh_last_symlink()
            self.assertFalse(last.exists())
            self.assertEqual(read_training_step(output / "checkpoints" / "recovery"), 20)

    def test_numbered_folder_is_the_fallback_when_recovery_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            write_fake_checkpoint(output / "checkpoints" / "000020", 20)
            config = find_resume_config(output)
            self.assertEqual(config, output / "checkpoints" / "000020" / "pretrained_model" / "train_config.json")
            self.assertEqual(read_training_step(config.parent.parent), 20)


class TrainCommandTest(unittest.TestCase):
    def test_auto_resume_and_dataset_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            keeper = make_keeper(output)
            save_step(keeper, 20)
            captured: dict[str, list[str]] = {}

            def run(command: list[str], _keeper: CheckpointKeeper) -> None:
                captured["command"] = command

            with (
                mock.patch("train._run_lerobot", side_effect=run),
                mock.patch("train._default_device", return_value="cpu"),
            ):
                with mock.patch("sys.stdout"):
                    train.main(
                        [
                            "--steps",
                            "100",
                            "--device",
                            "cpu",
                            "--output-dir",
                            str(output),
                            "--dataset.repo_id=user/my_aloha",
                        ]
                    )
            command = captured["command"]
            self.assertIn("--resume=true", command)
            self.assertIn("--num_workers=4", command)
            self.assertIn("--save_freq=2000", command)
            self.assertIn("--env_eval_freq=0", command)
            self.assertIn("--prefetch_factor=2", command)
            self.assertIn("--dataset.repo_id=user/my_aloha", command)
            self.assertNotIn(f"--dataset.repo_id={train.DATASET_REPO_ID}", command)
            config_flags = [token for token in command if token.startswith("--config_path=")]
            self.assertEqual(len(config_flags), 1)
            self.assertIn(f"{output.name}/checkpoints/recovery/pretrained_model/train_config.json", config_flags[0])

    def test_startup_print_uses_the_dataset_flag(self) -> None:
        import io
        from contextlib import redirect_stdout

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fresh"
            buffer = io.StringIO()

            def run(_command: list[str], _keeper: CheckpointKeeper) -> None:
                return None

            with (
                mock.patch("train._run_lerobot", side_effect=run),
                mock.patch("train._default_device", return_value="cpu"),
                redirect_stdout(buffer),
            ):
                train.main(
                    [
                        "--steps",
                        "10",
                        "--device",
                        "cpu",
                        "--num-workers",
                        "0",
                        "--output-dir",
                        str(output),
                        "--dataset.repo_id",
                        "lab/custom_cube",
                    ]
                )
            printed = buffer.getvalue()
            self.assertIn("Training ACT on lab/custom_cube", printed)
            self.assertNotIn(f"Training ACT on {train.DATASET_REPO_ID}", printed)

    def test_windows_default_workers_and_user_save_freq(self) -> None:
        self.assertEqual(default_num_workers("win32"), 1)
        self.assertEqual(default_num_workers("linux"), 4)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "fresh"
            captured: dict[str, list[str]] = {}

            def run(command: list[str], _keeper: CheckpointKeeper) -> None:
                captured["command"] = command

            with (
                mock.patch("train._run_lerobot", side_effect=run),
                mock.patch("train._default_device", return_value="cpu"),
                mock.patch("train.default_num_workers", return_value=1),
                mock.patch("sys.stdout"),
            ):
                train.main(
                    [
                        "--steps",
                        "30",
                        "--device",
                        "cpu",
                        "--output-dir",
                        str(output),
                        "--save_freq=500",
                    ]
                )
            command = captured["command"]
            self.assertIn("--num_workers=1", command)
            self.assertIn("--save_freq=500", command)
            self.assertFalse(any(token.startswith("--save_freq=2000") for token in command))
            self.assertNotIn("--resume=true", command)

    def test_resume_refuses_a_finish_line_before_the_saved_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            save_step(make_keeper(output), 20)
            with (
                mock.patch("train._run_lerobot") as run,
                mock.patch("train._default_device", return_value="cpu"),
                mock.patch("sys.stdout"),
            ):
                with self.assertRaises(SystemExit):
                    train.main(
                        ["--steps", "10", "--device", "cpu", "--output-dir", str(output), "--resume"]
                    )
            run.assert_not_called()


def _worker(mode: str, output: Path) -> None:
    keeper = make_keeper(output)
    if mode == "seed":
        if not save_step(keeper, 20):
            raise SystemExit("seed save failed")
        return
    if mode == "crash":
        def populate(dest: Path) -> None:
            write_fake_checkpoint(dest, 40, complete=False)
            os.kill(os.getpid(), signal.SIGKILL)

        keeper.save_recovery(40, populate)
        raise SystemExit("crash worker returned, the process was not killed")
    if mode == "resume":
        config = find_resume_config(output)
        if config is None:
            raise SystemExit("no resume config")
        step = read_training_step(config.parent.parent)
        print(f"resume_step={step}")
        if not save_step(keeper, 40):
            raise SystemExit("resume save failed")
        print(f"after_step={read_training_step(output / 'checkpoints' / 'recovery')}")
        return
    raise SystemExit(f"unknown mode {mode}")


class ForcedCrashTest(unittest.TestCase):
    def test_kill_mid_save_then_rerun_resumes_at_the_last_good_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "run"
            script = Path(__file__).resolve()
            seed = subprocess.run(
                [sys.executable, str(script), "seed", str(output)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(seed.returncode, 0, seed.stderr)

            killed = subprocess.run(
                [sys.executable, str(script), "crash", str(output)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(killed.returncode, 0)
            staging = output / "checkpoints" / STAGING_NAME
            self.assertTrue(staging.is_dir())
            self.assertFalse((staging / "training_state" / "training_step.json").is_file())
            self.assertEqual(
                read_training_step(output / "checkpoints" / "recovery"),
                20,
            )

            resumed = subprocess.run(
                [sys.executable, str(script), "resume", str(output)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stderr)
            self.assertIn("resume_step=20", resumed.stdout)
            self.assertIn("after_step=40", resumed.stdout)
            self.assertFalse(staging.exists())
            self.assertEqual(read_training_step(output / "checkpoints" / "recovery"), 40)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] in {"seed", "crash", "resume"}:
        _worker(sys.argv[1], Path(sys.argv[2]))
    else:
        unittest.main()

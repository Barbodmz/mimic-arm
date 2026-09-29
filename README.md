# mimic-arm

A small, simulation-only project for learning **imitation learning** with a robot arm.

You train an [ACT](https://huggingface.co/docs/lerobot/act) policy (Action Chunking with Transformers) on human demonstrations of the ALOHA "transfer the cube" task, using [Hugging Face LeRobot](https://github.com/huggingface/lerobot) and the [gym-aloha](https://github.com/huggingface/gym-aloha) simulator. Nothing here talks to a physical robot. That is on purpose: version 1 is how you learn the training loop before a real [SO-101](https://huggingface.co/docs/lerobot/so101) arm.

The published reference policy, trained for 80,000 steps on this same dataset, is [`lerobot/act_aloha_sim_transfer_cube_human`](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human). A full training run in this repo uses LeRobot's default ACT settings and aims at 100,000 steps.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Barbodmz/mimic-arm/blob/main/notebooks/train_colab.ipynb)

## What you are teaching the robot

**Imitation learning** means the robot learns a task by copying demonstrations, instead of discovering the task by trial and error (that second approach is reinforcement learning).

Here the demonstrations already exist: the dataset [`lerobot/aloha_sim_transfer_cube_human`](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human) has 50 episodes, 20,000 frames at 50 Hz, of a person driving a simulated two-arm ALOHA robot. The task, `AlohaTransferCube`, is: the right arm picks up a red cube and hands it to the left arm.

Each frame has:

- one top-down camera image (480×640)
- the 14 joint positions of the two arms (including both grippers)
- the 14 joint commands the human sent

Training fits a neural network that looks at the camera and the joints and predicts the commands. After training, you put that network in the simulator and score it. An episode is a **success** when the left gripper is holding the cube off the table. The simulator reports that as reward `4`. Lower rewards mean "touched the cube", "lifted it", or "the other gripper touched it", so a failed episode can still have a non-zero reward.

## What ACT is

ACT (from the paper [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)) does not predict a single next command. It predicts a **chunk** of future commands. LeRobot's default chunk is 100 steps, which is 2 seconds at 50 Hz. Predicting a whole chunk, then following it, makes the motion less jittery than predicting one step at a time.

The picture goes through a ResNet-18 (a standard vision network, started from ImageNet weights). A small transformer turns that, plus the joint positions, into the action chunk. During training, a second transformer looks at the demonstration actions and compresses them into a short code. That is the variational part: it lets one model represent the several slightly different ways a person performed the same task. At evaluation time that second transformer is not used. The policy just looks at the current camera frame and joint positions.

You do not set those architecture numbers yourself. `train.py` asks LeRobot for `--policy.type=act`, and LeRobot fills in its defaults (chunk size 100, learning rate `1e-5`, batch size 8, 100,000 steps, and so on).

## What is pinned

Verified together on Python 3.12, Ubuntu 24.04, CPU, headless OSMesa rendering:

| Piece | Version |
| --- | --- |
| LeRobot | **0.6.1** |
| gym-aloha | 0.1.4 |
| MuJoCo | 3.8.1 |
| PyTorch (CPU smoke test) | 2.11.0+cpu |
| torchvision | 0.26.0+cpu |
| torchcodec | 0.11.1 |
| numpy | 2.2.6 |

`requirements.txt` pins the rest of the stack that this install resolved. `setup.sh` installs PyTorch **before** that file, because a plain `pip install lerobot` on Linux currently pulls `torch==2.11.0+cu130` (a CUDA 13 wheel). That wheel is what you want on a machine whose driver supports CUDA 13. It was not run on a GPU in this project. The CPU wheel above is the one the smoke test actually trained and evaluated with.

LeRobot 0.6.1 needs **Python 3.12+**. The older `python lerobot/scripts/train.py` path is gone. The console scripts are `lerobot-train` and `lerobot-eval`. `train.py` and `evaluate.py` call those current entry points.

## Run it locally

From the repo root:

```bash
bash setup.sh          # or: bash setup.sh --cpu   /   bash setup.sh --cuda
source .venv/bin/activate
```

`setup.sh` installs OSMesa and EGL, creates `.venv`, and installs the pinned packages. The scripts set `MUJOCO_GL` themselves (`egl` when an NVIDIA GPU is present, `osmesa` otherwise). To force one:

```bash
export MUJOCO_GL=osmesa   # CPU software rendering
export MUJOCO_GL=egl      # GPU
```

### Train

```bash
python train.py --device cuda
```

That is the real run: 100,000 steps, batch size 8, checkpoints under `outputs/train/act_aloha_transfer_cube/checkpoints/`. LeRobot also writes a `checkpoints/last` link to the newest one. The folder that evaluation needs is `checkpoints/last/pretrained_model` (it holds `config.json` and `model.safetensors`).

Useful shorter commands:

```bash
# Laptop CPU. Checkpoints every 500 steps so a crash keeps something.
python train.py --steps 2000 --batch-size 4 --device cpu --num-workers 0 \
    --output-dir outputs/train/act_cpu --save_freq=500 --env_eval_freq=0

# Continue a run that was interrupted.
python train.py --resume --output-dir outputs/train/act_cpu --device cpu
```

Flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--steps` | 100000 | Optimizer steps |
| `--batch-size` | 8 | Samples per step |
| `--device` | `cuda` if a GPU is visible, else `cpu` | |
| `--output-dir` | `outputs/train/act_aloha_transfer_cube` | Must be a new folder, unless you pass `--resume` |
| `--num-workers` | 4 | Data-loading processes. Use `0` if you are low on RAM |
| `--resume` | off | Continue from `checkpoints/last` in `--output-dir` |

Anything else is passed straight to LeRobot. `--env_eval_freq=0` turns off rollouts during training (you will evaluate afterwards with `evaluate.py`). `--save_freq=2000` writes extra checkpoints. `--policy.push_to_hub=true --policy.repo_id=YOUR_USER/act_aloha_transfer` uploads them. WandB is off unless you pass `--wandb.enable=true`.

The first run downloads the dataset (about 500 MB) and the ResNet-18 ImageNet weights (about 45 MB).

### Evaluate

```bash
python evaluate.py \
    --checkpoint outputs/train/act_aloha_transfer_cube/checkpoints/last/pretrained_model \
    --episodes 20 \
    --device cuda
```

You can also pass the training directory, a step directory (`checkpoints/000200`), or a Hub id:

```bash
python evaluate.py \
    --checkpoint lerobot/act_aloha_sim_transfer_cube_human \
    --episodes 20 \
    --device cuda
```

It prints a success rate and an average reward, and writes videos plus `eval_info.json` under `outputs/eval/act_aloha_transfer_cube/videos/`.

| Flag | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | required | Local path or Hub model id |
| `--episodes` | 20 | Simulated episodes |
| `--videos` | 2 | How many of those episodes to save as mp4 |
| `--batch-size` | 1 | Parallel simulators. Leave this at 1 on CPU |
| `--device` | auto | `cuda` or `cpu` |
| `--output-dir` | `outputs/eval/act_aloha_transfer_cube` | Videos and `eval_info.json` |
| `--seed` | 1000 | Seed of the first episode |

An untrained or barely trained policy should score near 0%. That means the loop works and the policy has not learned the task yet. The reference Hub checkpoint is the one that actually transfers the cube (the model card reports results on 500 episodes; community re-runs on newer LeRobot builds have landed lower, often around 40–70%).

## Run it in Colab

Open `notebooks/train_colab.ipynb` (or the badge, once the repo is public). Use a GPU runtime (**T4** is enough). The notebook installs LeRobot without throwing away Colab's CUDA build of PyTorch, trains for **8,000 steps**, evaluates 20 episodes, and plays a video in the page.

Why 8,000 and not 100,000: a free T4 session often dies after a couple of hours, and 100,000 steps is an overnight run. 8,000 steps is long enough to see the loss drop and short enough to finish, with a checkpoint every 2,000 steps so a disconnect does not erase everything. The policy will not be reliable yet. Raise `--steps` if the session is still alive. The loss curve matters more than the success rate at this length.

## Expected runtimes

Measured on a 4-core CPU with no GPU, batch size 2, one camera, OSMesa rendering (`MUJOCO_GL=osmesa`). A GPU run is much faster; the T4 numbers below are a planning guess, not a measurement.

| Job | Time |
| --- | --- |
| `bash setup.sh` | a few minutes, plus the PyTorch download |
| First training start (dataset + ResNet-18 weights) | a few seconds on a fast connection, once the packages are installed |
| Training, CPU, batch size 2 | **about 1.0 step/s**. The 100-step smoke test's training loop took 1 minute 39 seconds |
| Training, CPU, 100,000 steps | on the order of a day or more at this speed. Use Colab or another GPU |
| Training, free T4, 8,000 steps, batch size 8 | plan on roughly 1–2 hours, including download. Not timed on a T4 in this repo |
| Training, 100,000 steps on a decent GPU | several hours. This is the run that can approach the published checkpoint |
| Eval, CPU, 1 episode (400 sim steps at 50 Hz) | about 78 seconds in the smoke test |
| Eval, CPU, 20 episodes | about 25–30 minutes |
| Eval, T4, 20 episodes | a few minutes |

## Gotchas

- **Python 3.12.** LeRobot 0.6.1 will not install on 3.10 or 3.11.
- **`pip install lerobot` replaces a CPU torch with a CUDA 13 wheel** if you do not install `torch==2.11.0+cpu` first. `setup.sh` does this for you. On Colab, do the opposite: keep Colab's CUDA torch. The notebook does that.
- **Do not upgrade numpy past 2.2.x.** LeRobot 0.6.1 requires `numpy<2.3`. Installing the CPU torch wheel can pull numpy 2.5; `setup.sh` pins 2.2.6 again.
- **Headless machines need `MUJOCO_GL`.** Without `libosmesa6` (CPU) or `libegl1` (GPU), rendering crashes on import. `setup.sh` installs both.
- **The output directory must be new.** LeRobot raises `FileExistsError` if `--output-dir` already exists and you are not resuming.
- **`--policy.push_to_hub` defaults to true inside LeRobot.** `train.py` turns it off unless you pass `--policy.push_to_hub=true` and a `--policy.repo_id`.
- **Video decoding uses torchcodec**, which needs a matching PyTorch. If a batch fails while reading `observation.images.top`, rerun with `--dataset.video_backend=pyav`. The dataset videos are AV1.
- **Shutting the simulator down can print a `RuntimeError` from MuJoCo's render thread** (`cannot schedule new futures after shutdown`). The episode has already been scored. You can ignore that message.
- **Low memory:** `--batch-size 2 --num-workers 0`. The default batch size of 8 plus 4 data workers is aimed at a GPU workstation.
- **Success rate on a short run will be ~0%.** That is not a broken install.

## Next steps

### Record your own simulated demonstrations

This repo trains on a dataset someone else recorded. LeRobot 0.6.1's supported way to *drive a simulator yourself* and save a dataset is the gym-hil Panda environment (keyboard or gamepad), documented in [Imitation Learning in Sim](https://huggingface.co/docs/lerobot/main/en/il_sim). You install the HIL extra, record episodes with `python -m lerobot.rl.gym_manipulator` in record mode, and use `PandaPickCubeKeyboard-v0` if you do not have a gamepad.

gym-aloha, the simulator this repo evaluates in, does not ship a keyboard teleop in LeRobot 0.6.1. The 50 transfer-cube episodes were recorded by people driving the simulated arms and stored in LeRobot's dataset format (images, joint state, actions, one task sentence). Once you have a dataset in that format, train on it by overriding the dataset id. `train.py` only fills in the transfer-cube dataset when you do not pass your own:

```bash
python train.py \
    --dataset.repo_id=YOUR_USER/your_sim_dataset \
    --env.type=aloha \
    --env.task=AlohaTransferCube-v0 \
    --device cuda
```

Change `--env.type` and `--env.task` to the simulator you actually recorded. ACT itself only needs images, a state vector, and actions. The environment flags are what `evaluate.py` and mid-training rollouts use.

### Move to a real SO-101

The SO-101 is a low-cost follower arm (Feetech motors) that you drive with a matching leader arm. Same idea as this project: record demonstrations, train ACT, run the policy. The hardware steps, from the current LeRobot docs:

1. Install the hardware extras: `pip install 'lerobot[feetech,core_scripts]'`.
2. Find the USB ports (`lerobot-find-port`), give each motor an id (`lerobot-setup-motors`), and calibrate (`lerobot-calibrate`).
3. Teleoperate to confirm the leader moves the follower:

```bash
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader
```

4. Record 50 or so careful episodes of one simple task (pick up a cube, drop it in a bowl). Fifty clean episodes beat two hundred sloppy ones. Cameras stay fixed.

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader \
    --dataset.repo_id=YOUR_USER/so101_pick_cube \
    --dataset.single_task="Pick up the cube and put it in the bowl" \
    --dataset.num_episodes=50 \
    --dataset.push_to_hub=true \
    --display_data=true
```

5. Train ACT on that dataset. Drop `--env.type` until you have a simulator for your scene; you can still train from the dataset alone. Start from the same defaults this repo uses (`--policy.type=act`, batch size 8, on the order of 100,000 steps).

```bash
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=YOUR_USER/so101_pick_cube \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --output_dir=outputs/train/act_so101_pick_cube
```

6. Run the policy on the real arm with `lerobot-rollout` (in LeRobot 0.6, `lerobot-record` only collects data). Count successes the same way you would in sim: did the cube end up in the bowl?

```bash
lerobot-rollout \
    --strategy.type=base \
    --policy.path=outputs/train/act_so101_pick_cube/checkpoints/last/pretrained_model \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --task="Pick up the cube and put it in the bowl" \
    --duration=30
```

Ports, camera indexes, and calibration differ per machine. The SO-101 guide is the source of truth: <https://huggingface.co/docs/lerobot/so101>.

## Layout

```
setup.sh                 install script
requirements.txt         pinned Python packages (torch installed separately)
pyproject.toml           project metadata
train.py                 train ACT on the transfer-cube dataset
evaluate.py              roll the policy out in AlohaTransferCube
mimic_arm/mujoco_gl.py   pick EGL or OSMesa before MuJoCo imports
notebooks/train_colab.ipynb
```

## Smoke test

These commands were run on the CPU-only machine this project was built on. A near-zero success rate is the right outcome after 100 steps.

```bash
python train.py --steps 100 --batch-size 2 --device cpu --num-workers 0 \
    --output-dir outputs/train/smoke_act --env_eval_freq=0
python evaluate.py --checkpoint outputs/train/smoke_act/checkpoints/last \
    --episodes 2 --videos 2 --device cpu --output-dir outputs/eval/smoke_act
```

Training used LeRobot's ACT defaults (ResNet-18, chunk size 100, learning rate `1e-5`). The loop logged about 1 step/s. Loss fell from 48.1 at step 10 to 7.2 at step 100, and the checkpoint landed in `outputs/train/smoke_act/checkpoints/000100/pretrained_model`.

Evaluation printed:

```
Success rate: 0.0%  (2 episodes)
Average reward: 0.000
Average max reward: 0.000
Runtime: 156.7s
Videos:
  outputs/eval/smoke_act/videos/aloha_0/eval_episode_0.mp4
  outputs/eval/smoke_act/videos/aloha_0/eval_episode_1.mp4
```

Each video is 400 frames, 640×480, 50 fps (8 seconds). The arms sit near the home pose and do not pick up the cube, which is what an almost-untrained policy should do.

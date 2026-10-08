# mimic-arm

A small, simulation-only project for learning **imitation learning** with a robot arm.

You train an [ACT](https://huggingface.co/docs/lerobot/act) policy (Action Chunking with Transformers) on human demonstrations of the ALOHA "transfer the cube" task, using [Hugging Face LeRobot](https://github.com/huggingface/lerobot) and the [gym-aloha](https://github.com/huggingface/gym-aloha) simulator. Nothing here talks to a physical robot. That is on purpose: version 1 is how you learn the training loop before a real [SO-101](https://huggingface.co/docs/lerobot/so101) arm.

The published reference policy, trained for 80,000 steps on this same dataset, is [`lerobot/act_aloha_sim_transfer_cube_human`](https://huggingface.co/lerobot/act_aloha_sim_transfer_cube_human). A full training run in this repo uses LeRobot's default ACT settings and aims at 100,000 steps.

**v1.1** adds three tools on top of that loop. `evaluate.py --save-failures` keeps a video of every failed episode and one success. `compare.py` scores several checkpoints on the same episodes and prints a 95% Wilson interval, because 20–50 episodes is a noisy measurement. The Colab notebook can store checkpoints on Google Drive, resume to more steps, and print how many parameters each part of the ACT network has.

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

`setup.sh` installs OSMesa and EGL, creates `.venv`, and installs the pinned packages. The scripts set `MUJOCO_GL` themselves (`egl` when an NVIDIA GPU is present, `osmesa` otherwise; `glfw` on Windows). To force one:

```bash
export MUJOCO_GL=osmesa   # CPU software rendering
export MUJOCO_GL=egl      # GPU
```

### Windows

Training keeps going if Windows cannot create the `checkpoints/last` symlink (WinError 1314, which happens without Developer Mode or an admin account). The real checkpoint is the directory `checkpoints/recovery`, not that shortcut. Rerunning the same `train.py` command resumes from it. Older runs that only have numbered folders still resume from the newest of those when `recovery` is missing. Evaluation defaults to `MUJOCO_GL=glfw`. A `MUJOCO_GL` value you set yourself is kept. Linux still picks `egl`, then `osmesa`.

Flushing that recovery folder on Windows opens each file read/write. A read-only handle makes `os.fsync` raise `OSError: [Errno 9] Bad file descriptor` and used to kill the run at the first save. If a flush or the rename that publishes the checkpoint still fails, training logs a warning and continues. The previous recovery stays in place until a finished save can be swapped in.

`--num-workers` defaults to 1 on Windows and 4 elsewhere. Each Windows worker is a fresh process that imports PyTorch again, and that commit charge counts against RAM plus the page file.

A 60k run on a 6 GB laptop can still die with Windows error 1455 ("The paging file is too small for this operation to complete") even when the checkpoint itself fits. `train.py` cannot raise that limit. If a run dies with 1455, set a larger page file and reboot: Settings, System, About, Advanced system settings, Performance, Settings, Advanced, Virtual memory, Change. Uncheck automatic management, choose a drive that has free space, and set a custom size (16384 MB initial and 32768 MB maximum is a reasonable laptop choice). Then rerun the same training command. It continues from `checkpoints/recovery`.

### Train

```bash
python train.py --device cuda
```

That is the 100,000-step run. Checkpoints go under `outputs/train/act_aloha_transfer_cube/checkpoints/`:

* `recovery/` is the one full checkpoint (model, optimizer, scheduler, RNG, and step). It is rewritten every 2,000 steps by default. The new tree is written beside it and swapped in with a rename, so a crash mid-save leaves the previous recovery readable.
* `weights/010000/`, `weights/020000/`, and so on are permanent weights-only copies: `config.json`, `model.safetensors`, `train_config.json`, and the processor files that hold normalization stats. Optimizer state is not included. They are never deleted. The best checkpoint is not always the latest one (an 8k snapshot has beaten a 20k snapshot on this task), which is why these stay.
* `last` is a shortcut to `recovery` when the disk can store a symlink. Evaluation also accepts `recovery/pretrained_model` or `weights/<step>/pretrained_model` directly.

The ACT policy used here has 51,613,582 parameters. In fp32 that is 206,454,328 bytes (196.9 MiB) of weights, plus a few small JSON and processor files, so a weights-only folder is about 200 MB. A full recovery checkpoint is about 591 MB because it also stores Adam state. A 60k run that keeps one recovery and a weights folder every 10k steps (six of them, including the final step) is on the order of 1.8 GB of checkpoints, instead of a new 591 MB folder every 2,000 steps.

60,000 steps on the laptop this was aimed at (Windows, RTX 4050 6 GB, about 3 GB free):

```bash
python train.py --steps 60000 --batch-size 8 --device cuda --num-workers 1 \
    --recovery-every 2000 --weights-every 10000 --min-free-gb 1 \
    --env_eval_freq=0 --output-dir outputs/train/act_aloha_60k
```

Rerun that same command to resume. `--steps` is the new finish line. Before each save, free disk is checked. Below `--min-free-gb` (default 1), the permanent weights save is skipped and a warning is printed. The recovery save still runs when there is room for another full copy (about 800 MB until a recovery folder exists to measure, then about 1.1 times that folder). If even the recovery save does not fit, it is skipped instead of crashing.

Useful shorter commands:

```bash
# Laptop CPU. The recovery checkpoint refreshes every 500 steps.
python train.py --steps 2000 --batch-size 4 --device cpu --num-workers 0 \
    --output-dir outputs/train/act_cpu --recovery-every 500 --env_eval_freq=0

# Same output directory, later finish line. --resume is optional:
# a recovery checkpoint is picked up automatically.
python train.py --resume --steps 20000 --output-dir outputs/train/act_cpu --device cpu
```

Flags:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--steps` | 100000 | Optimizer steps. On resume, this is the new finish line |
| `--batch-size` | 8 | Samples per step |
| `--device` | `cuda` if a GPU is visible, else `cpu` | |
| `--output-dir` | `outputs/train/act_aloha_transfer_cube` | Rerunning the same command resumes from `checkpoints/recovery` |
| `--num-workers` | 1 on Windows, 4 elsewhere | Data-loading processes. Use `0` if you are low on RAM |
| `--recovery-every` | 2000 | How often the single full checkpoint is replaced. `--save_freq` overrides this |
| `--weights-every` | 10000 | How often a permanent weights-only folder is added. The final step is always saved |
| `--min-free-gb` | 1 | Skip a weights save when free disk is below this. Recovery still saves if it fits |
| `--resume` | automatic when a checkpoint exists | Continue from `checkpoints/recovery`. If that folder is missing, the newest numbered checkpoint from an older run is used |
| `--episodes` | all 50 demos | Comma-separated episode indices, or a JSON file from `pick_demos.py`. LeRobot 0.6.1 keeps `meta/stats.json` from the full dataset |
| `--episode-set` |  | `alike` or `random` when `--episodes` is `demo_sets/option4_sets.json` |

Anything else is passed straight to LeRobot. `--env_eval_freq=0` is set for you unless you pass another value, so training does not also roll out the simulator (evaluate afterwards with `evaluate.py`). `--prefetch_factor=2` is set when there is at least one data worker, instead of LeRobot's 4. `--policy.push_to_hub=true --policy.repo_id=YOUR_USER/act_aloha_transfer` uploads checkpoints. WandB is off unless you pass `--wandb.enable=true`.

The first run downloads the dataset (about 500 MB) and the ResNet-18 ImageNet weights (about 45 MB).

### Alike vs random human demos (issue #12)

`pick_demos.py` builds the two 20-episode lists for the consistency-vs-variety comparison. It reads the joint and action streams (no GPU), scores speed, pauses, and gripper-close timing after stretching every demo to the same length, and keeps duration as its own term. The random list is `random.Random(12)`.

The dataset has no cube-position field. The script estimates each cube from the right fingertip at the first gripper close that reaches the table. The first top-camera frame is only a backup, and this dataset did not need it: all 50 episodes recovered from the gripper. Cube-spot fairness **passed**. The 20 nearest to the typical motion and the seed-12 random 20 cover a similar spread (x spans 0.157 m vs 0.178 m, y spans 0.183 m vs 0.151 m, all four quadrants of the spawn box in both). Seven episodes are in both lists. Recovered x sits a few centimeters toward the right arm relative to the 0.0–0.2 m spawn interval; both sets share that shift, so it does not favor one list.

Regenerate the lists and the plot (needs `matplotlib`, which is not in `requirements.txt`):

```bash
python pick_demos.py --output-dir demo_sets
```

That writes `demo_sets/option4_sets.json` and `demo_sets/option4_alike_vs_random.png`. The checked-in copies are that command on `lerobot/aloha_sim_transfer_cube_human`. If a future run prints `Fairness: FAILED`, do not train: close issue #12 and retrain scripted demos with a wider cube spawn instead.

Train both sets for the same number of steps. Normalization stays the full 50-episode stats either way. These are the laptop runs, not part of the picker:

```bash
python train.py --steps 100000 --device cuda \
    --episodes demo_sets/option4_sets.json --episode-set alike \
    --output-dir outputs/train/act_alike20

python train.py --steps 100000 --device cuda \
    --episodes demo_sets/option4_sets.json --episode-set random \
    --output-dir outputs/train/act_random20
```

Score both checkpoints for 100 episodes from the same seed:

```bash
python evaluate.py --checkpoint outputs/train/act_alike20/checkpoints/recovery/pretrained_model \
    --episodes 100 --seed 1000 --device cuda
python evaluate.py --checkpoint outputs/train/act_random20/checkpoints/recovery/pretrained_model \
    --episodes 100 --seed 1000 --device cuda
```

### Evaluate

```bash
python evaluate.py \
    --checkpoint outputs/train/act_aloha_transfer_cube/checkpoints/recovery/pretrained_model \
    --episodes 20 \
    --device cuda
```

You can also pass the training directory (the latest recovery or weights snapshot is used), a weights directory (`checkpoints/weights/008000`), a recovery directory, an older numbered directory (`checkpoints/000200`), or a Hub id:

```bash
python evaluate.py \
    --checkpoint lerobot/act_aloha_sim_transfer_cube_human \
    --episodes 20 \
    --device cuda
```

It prints a success rate and an average reward, a line per episode (`seed`, success, max reward, sum of rewards), and a count of how far the failures got (0 nothing, 1 touched, 2 lifted, 3 both grippers, 4 success). Videos and `eval_info.json` (including that per-episode table) land under `outputs/eval/act_aloha_transfer_cube/`.

To keep the interesting videos, pass `--save-failures`. That saves every failed episode and at most one success, with names like `episode_07_fail_maxreward1.mp4` and `episode_03_success.mp4`. LeRobot itself always renders the first N episodes, before it knows which ones failed, so this flag renders all of them and deletes the extra successes.

```bash
python evaluate.py \
    --checkpoint outputs/train/act_aloha_transfer_cube/checkpoints/recovery/pretrained_model \
    --episodes 50 \
    --save-failures \
    --device cuda
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--checkpoint` | required | Local path or Hub model id |
| `--episodes` | 20 | Simulated episodes |
| `--videos` | 2 | How many of those episodes to save as mp4. Ignored when `--save-failures` is set |
| `--save-failures` | off | Save every failed episode and one success |
| `--batch-size` | 1 | Parallel simulators. Leave this at 1 on CPU |
| `--device` | auto | `cuda` or `cpu` |
| `--output-dir` | `outputs/eval/act_aloha_transfer_cube` | Videos and `eval_info.json` |
| `--seed` | 1000 | Seed of the first episode |
| `--temporal-ensemble` | off | ACT ensemble coefficient, for example `0.01`. Sets `n_action_steps` to 1 |
| `--cube-range` | `default` | `outside` is the 5 cm frame. `holdout-scattered` and `holdout-far-y` walk the saved holdout lists |
| `--cube-x`, `--cube-y` | off | Explicit cube ranges, `LO HI` in meters. Pass both |

`--temporal-ensemble 0.01` is the coefficient from the ACT paper. The checkpoint is loaded with that value and `n_action_steps=1`, which is what LeRobot's ACT config requires. The network then runs on every simulator step instead of once per chunk of 100, and the ensemble state is cleared at the start of each episode. On this CPU, a normal 1-episode eval took 40.6 s. The same kind of run with `--temporal-ensemble 0.01` took 146.5 s and 143.4 s for 2 episodes (about 72 s each), 1.8 times as long per episode. The coefficient is printed and stored in `eval_info.json`. `compare.py` adds a `temporal_ensemble_coeff` column when you pass the flag.

`--cube-range outside` moves the cube off the rectangle that gym-aloha's `sample_box_pose` uses: x from 0.0 to 0.2, y from 0.4 to 0.6, z fixed at 0.05. The outside range is a 5 cm frame around that rectangle, x from -0.05 to 0.25 and y from 0.35 to 0.65, skipping any sample that still falls inside. Five centimeters is far enough to leave the demonstrations, and the cube still sits on the table in view of the top camera. In the simulator those positions keep their x,y after the cube drops, rest at about z 0.02, and show a few hundred red pixels, like a cube inside the training rectangle. The far corner of the frame is about 0.73 m from the left arm base; the far corner of the training rectangle is already about 0.68 m, and the right arm is closer than that to every point in the frame. The same seed always picks the same spot, so two checkpoints see the same cubes. Each episode line includes that cube x, y. `--cube-x LO HI --cube-y LO HI` samples a rectangle you name instead. Leave both options off and the evaluation output matches a normal run.

### Wider scripted demos

`record_scripted_demos.py` records the scripted pick-and-handover. With no extra flags it uses gym-aloha's box, `sample_box_pose(seed + episode)`, and the stock mocap weld. `--wider-spawn` records the training list in `demo_sets/wider_spawn_holdouts.json`.

The wider rectangle is the training box plus the same 5 cm band as `--cube-range outside`: x from -0.05 to 0.25, y from 0.35 to 0.65. A 2 cm grid over that rectangle has 256 spots. A spot is kept when the joint-space replay of the scripted handover reaches reward 4 in the stock joint-position env, which is the env ACT is scored in. Under MuJoCo 3 the stock mocap impedance (`solimp 0.25`) leaves the right gripper about 7 cm short of the waypoint, so that teacher misses most of the original box, including the right half. The recorder keeps `solref 0.01` and raises the impedance to `solimp 0.95 0.99` only while driving the end-effector teacher. The replay env is unchanged. With that teacher, all 100 grid spots inside the original box succeed, and 242 of 256 wider-grid spots succeed. 142 of those are in the outside band. 14 spots are rejected, all on the high-X edge of the frame. Every kept spot also reached reward 4 in the end-effector sim, so the stock-env replay rate on the reachable set is 242/242.

Two groups stay out of the training demos:

- Scattered holdout: 36 spots, `round(0.15 * 242)`, drawn with `numpy.random.RandomState(14)` from the reachable spots outside the far-Y corner. The list is `scattered_holdout` in the JSON file. Eval uses three shifts around each of those 36, 108 positions in `scattered_shifts`, drawn with seed 14 and kept only when the stock joint replay reaches reward 4. Each shift is within 9 mm of its own anchor, strictly closer to that anchor than to any training spot, and farther than 5 mm from every training spot.
- Far-Y corner: the same high-X, high-Y square as before, x in (0.2, 0.25] and y in (0.6, 0.65] (9 grid points). Three of them pass the stock replay and are the corner eval: (0.21, 0.61), (0.21, 0.63), (0.23, 0.61). The other six, (0.21, 0.65), (0.23, 0.63), (0.23, 0.65), (0.25, 0.61), (0.25, 0.63), and (0.25, 0.65), fail it, so they are in neither training nor eval. The bounds are unchanged.

Training demos are the other 203 reachable spots. `demo_sets/wider_spawn_spots.png` plots training spots, the 36 scattered anchors, the 108 shifts, the reachable far-Y corner, and the rejected grid.

`train.py` still defaults to the published dataset. Wider demos are used only when you pass `--dataset.repo_id` and `--dataset.root` for a set recorded with `--wider-spawn`.

`--cube-range holdout-scattered` walks the 108 shifted positions in file order. Seed `s` uses position `s mod 108`, so `--episodes 108` covers each shift once. Omitting `--episodes` uses that same count. `--cube-range holdout-far-y` still walks the three reachable corner spots; a 50-episode run cycles them. `--cube-range outside` is unchanged and stays the comparison run. The 70% bar belongs on the two holdout runs. The outside-band run places cubes inside the wider training area, so it is a comparison number.

Permanent weights from `train.py` land in `checkpoints/weights/<step>/pretrained_model`. The step folder is zero-padded to at least six digits (`050000`, `100000`) when the finish line is 100,000 or 50,000. `checkpoints/recovery/pretrained_model` is only the latest step, so a 50k snapshot has to be the weights folder.

Regenerate the holdout file, then the shifts (the second command does not move the 36 anchors):

```bash
python record_scripted_demos.py --write-holdouts
python record_scripted_demos.py --write-shifts
```

Laptop commands. `--steps 100000` with `--weights-every 10000` writes `weights/050000` on the way to `weights/100000`. Scattered eval is 108 episodes, one per shift. Far-Y and the outside comparison stay at 50. Seed 1000.

```bash
python record_scripted_demos.py --wider-spawn --output-dir outputs/data/scripted_wide

python train.py --steps STEPS --batch-size 8 --device cuda --num-workers 1 \
    --dataset.repo_id=local/scripted_wide \
    --dataset.root=outputs/data/scripted_wide \
    --recovery-every 2000 --weights-every 10000 --min-free-gb 1 \
    --env_eval_freq=0 \
    --output-dir outputs/train/act_scripted_wide

# 50k weights. Folder name is the six-digit step, not recovery.
python evaluate.py \
    --checkpoint outputs/train/act_scripted_wide/checkpoints/weights/050000/pretrained_model \
    --episodes 108 --seed 1000 --device cuda \
    --cube-range holdout-scattered \
    --output-dir outputs/eval/scripted_wide_50k_holdout_scattered

python evaluate.py \
    --checkpoint outputs/train/act_scripted_wide/checkpoints/weights/050000/pretrained_model \
    --episodes 50 --seed 1000 --device cuda \
    --cube-range holdout-far-y \
    --output-dir outputs/eval/scripted_wide_50k_holdout_far_y

python evaluate.py \
    --checkpoint outputs/train/act_scripted_wide/checkpoints/weights/050000/pretrained_model \
    --episodes 50 --seed 1000 --device cuda \
    --cube-range outside \
    --output-dir outputs/eval/scripted_wide_50k_outside_compare

# 100k weights, same naming. recovery/pretrained_model is only the latest step.
python evaluate.py \
    --checkpoint outputs/train/act_scripted_wide/checkpoints/weights/100000/pretrained_model \
    --episodes 108 --seed 1000 --device cuda \
    --cube-range holdout-scattered \
    --output-dir outputs/eval/scripted_wide_100k_holdout_scattered

python evaluate.py \
    --checkpoint outputs/train/act_scripted_wide/checkpoints/weights/100000/pretrained_model \
    --episodes 50 --seed 1000 --device cuda \
    --cube-range holdout-far-y \
    --output-dir outputs/eval/scripted_wide_100k_holdout_far_y

python evaluate.py \
    --checkpoint outputs/train/act_scripted_wide/checkpoints/weights/100000/pretrained_model \
    --episodes 50 --seed 1000 --device cuda \
    --cube-range outside \
    --output-dir outputs/eval/scripted_wide_100k_outside_compare
```

### Compare checkpoints

```bash
python compare.py --train-dir outputs/train/act_aloha_transfer_cube --episodes 20 --device cuda
```

That evaluates every permanent weights snapshot, plus the recovery checkpoint when its step is not already in the weights list (not the `last` shortcut a second time), with the same seeds. Older numbered folders are included too. It prints step, success rate, and average max reward, plus a 95% Wilson interval, and writes `outputs/eval/compare/compare.csv`. `compare.py` takes the same `--temporal-ensemble`, `--cube-range`, `--cube-x`, and `--cube-y` flags. When you pass them, the CSV gains a column for each one you turned on.

Pass a list instead of a training directory:

```bash
python compare.py \
    --checkpoints outputs/train/act_aloha_transfer_cube/checkpoints/002000 \
                  outputs/train/act_aloha_transfer_cube/checkpoints/008000 \
    --episodes 20
```

20–50 episodes is a small sample. Read the interval before deciding that one checkpoint is better.

An untrained or barely trained policy should score near 0%. That means the loop works and the policy has not learned the task yet. The reference Hub checkpoint is the one that actually transfers the cube (the model card reports results on 500 episodes; community re-runs on newer LeRobot builds have landed lower, often around 40–70%).

## Run it in Colab

Open `notebooks/train_colab.ipynb` (or the badge, once the repo is public). Use a GPU runtime (**T4** is enough). A fresh free T4 runtime is Python 3.13 with a CUDA build of PyTorch already installed. The notebook writes a constraints file for those exact `torch` and `torchvision` versions, installs LeRobot against it, and prints the full pip log. It trains for **8,000 steps**, evaluates 20 episodes, and plays a video in the page.

Why 8,000 and not 100,000: a free T4 session often dies after a couple of hours, and 100,000 steps is an overnight run. 8,000 steps is long enough to see the loss drop and short enough to finish. The notebook keeps one full recovery checkpoint, refreshed every 2,000 steps, and a permanent weights folder at the same interval (Colab has the disk for that; a laptop should leave `--weights-every` at 10,000). A disconnect does not erase the run: rerun the same `train.py` command, or the Train longer cell, and it resumes. The policy will not be reliable yet. The loss curve matters more than the success rate at this length. An 8k weights folder can score better than a later one, so compare them instead of keeping only the last step.

The notebook has four extra sections after that first run:

- **Google Drive** (off unless you set `SAVE_TO_DRIVE = True`). Mounts Drive and points `outputs/train` at a folder there, so checkpoints survive a disconnect. In a new session, run the install cells and this cell again before you resume. `checkpoints/recovery` is a real directory, so Drive can store it. The `checkpoints/last` symlink may still be missing.
- **Train longer**. Sets `TOTAL_STEPS = 20000` and runs `train.py` on the same output directory, then evaluates 50 episodes with `--save-failures` and plays the failure videos. `--resume` is still accepted; the same command without it also resumes when `checkpoints/recovery` exists.
- **Look inside the network**. Loads the checkpoint and prints parameter counts for the ResNet backbone, transformer encoder, decoder, VAE encoder, and the input/output projections, plus the ACT config values.
- **Compare checkpoints**. Runs `compare.py` over the permanent weights folders.

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

- **Python 3.12 or newer.** LeRobot 0.6.1 will not install on 3.10 or 3.11. Local setup is verified on 3.12. Colab's free T4 runtime is 3.13.
- **`pip install lerobot` replaces a CPU torch with a CUDA 13 wheel** if you do not install `torch==2.11.0+cpu` first. `setup.sh` installs the chosen wheel and passes a constraints file so the requirements install does not swap it. On Colab, do the opposite of a fresh install: keep Colab's CUDA torch. The notebook constrains `torch` and `torchvision` to the versions already imported.
- **Do not upgrade numpy past 2.2.x, or fsspec past 2026.2.0.** LeRobot 0.6.1 requires `numpy<2.3`. datasets 4.8.5 requires `fsspec<=2026.2.0`. Installing a torch wheel from the PyTorch index can pull numpy 2.5 and fsspec 2026.7; `setup.sh` pins `numpy==2.2.6` and `fsspec==2026.2.0` again afterwards. The Colab install does not pass those pins: the LeRobot and datasets requirements themselves downgrade newer preinstalled copies.
- **`labmaze` has no Python 3.13 wheel.** It is a `dm-control` dependency, and `gym-aloha` pulls `dm-control` in, but AlohaTransferCube never imports it. On 3.13 the source build needs Bazel and fails. `setup.sh` and the notebook install a pure-Python placeholder of `labmaze==1.0.6` in that case. Python 3.12 still gets the real wheel from `requirements.txt`.
- **Headless machines need `MUJOCO_GL`.** Without `libosmesa6` (CPU) or `libegl1` (GPU), rendering crashes on import. `setup.sh` installs both.
- **Rerunning the same output directory resumes.** `train.py` turns resume on when `checkpoints/recovery` (or an older numbered checkpoint) is already there. LeRobot still raises `FileExistsError` if the directory exists and there is nothing to resume from. Delete that directory to start over. A finish line at or below the saved step is rejected so the run does not exit immediately.
- **`--policy.push_to_hub` defaults to true inside LeRobot.** `train.py` turns it off unless you pass `--policy.push_to_hub=true` and a `--policy.repo_id`.
- **Video decoding uses torchcodec**, which needs a matching PyTorch. If a batch fails while reading `observation.images.top`, rerun with `--dataset.video_backend=pyav`. The dataset videos are AV1.
- **Shutting the simulator down can print a `RuntimeError` from MuJoCo's render thread** (`cannot schedule new futures after shutdown`). The episode has already been scored. You can ignore that message.
- **Low memory:** `--batch-size 2 --num-workers 0`. On Windows the default is already `--num-workers 1`, and mid-training rollouts are off unless you pass `--env_eval_freq`. Error 1455 means the page file is too small; see the Windows section above.
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

The SO-101 is a low-cost follower arm (Feetech motors) that you drive with a matching leader arm. Same idea as this project: record demonstrations, train ACT, run the policy. The stock parts and the servo fit-test print are in [`hardware/`](hardware/README.md). The hardware steps, from the current LeRobot docs:

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
compare.py               score several checkpoints on the same episodes
record_scripted_demos.py record scripted demos; --wider-spawn uses the holdout file
demo_sets/wider_spawn_holdouts.json  training spots and the two holdout lists
demo_sets/wider_spawn_spots.png      plot of those spots
mimic_arm/mujoco_gl.py   pick EGL or OSMesa before MuJoCo imports
mimic_arm/ensure_labmaze.py  labmaze placeholder when Python 3.13 has no wheel
mimic_arm/checkpoints.py find a pretrained_model folder or list step checkpoints
mimic_arm/saving.py      one recovery checkpoint, permanent weights, disk guard
mimic_arm/act_summary.py parameter counts for a trained ACT policy
notebooks/train_colab.ipynb
```

## Smoke test

These commands were run on the CPU-only machine this project was built on. A near-zero success rate is the right outcome after 100 steps.

```bash
python train.py --steps 100 --batch-size 2 --device cpu --num-workers 0 \
    --output-dir outputs/train/smoke_act --env_eval_freq=0
python evaluate.py --checkpoint outputs/train/smoke_act \
    --episodes 2 --videos 2 --device cpu --output-dir outputs/eval/smoke_act
```

Training used LeRobot's ACT defaults (ResNet-18, chunk size 100, learning rate `1e-5`). The loop logged about 1 step/s. Loss fell from 48.1 at step 10 to 7.2 at step 100. That smoke test predates recovery checkpoints and wrote `outputs/train/smoke_act/checkpoints/000100/pretrained_model`. A run today writes `checkpoints/recovery` and, on the final step, `checkpoints/weights/000100`. Point `evaluate.py` at either of those, or at the training directory.

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

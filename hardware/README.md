# Hardware: SO-101 arm

This folder holds the 3D models for the real arm.

- `stock/` holds the official SO-101 parts. **Don't edit these.** They're the reference you compare against or fall back to.
- `custom/` holds Barbod's versions and Fusion scripts. Each changed part goes in its own subfolder with a short `NOTES.md` that says what changed, why, and which stock part it replaces.

## Rules for custom parts (until the first real training run)
- **Safe to change:** looks (shells, colours, logo, covers, cable clips), the base (wider or heavier, clamps, board spot, cable exit), and fixed add-ons like the camera mount.
- **Keep identical:** motor pockets and horn screw patterns, joint-to-joint link lengths, and the gripper. The 50 demos get recorded with the stock gripper.
- **Check every change for:** light weight on moving parts; full range of motion with no pinched cables (rotate each joint to both ends in CAD); a camera spot that never moves after recording starts; and fit (print the fit-test block first).

## File formats
Save every part as **STEP** (so it opens in any CAD tool) **plus STL** (for printing and Forge's sim test). Fusion or Onshape project files are optional extras.

## Print order
1. Fit-test block (`custom/`)
2. The official follower arm, with only safe changes, then the leader
3. Assemble with the [SO-101 assembly guide](https://huggingface.co/docs/lerobot/so101)
4. Calibrate, record 50 demos, and train
5. Only then, design your own gripper as a separate experiment with new demos

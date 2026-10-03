# Fit-test block

Print this before any arm part. It tells you the clearance your printer needs for a Feetech STS3215 servo (the SO-101 motor) and for the M2 and M3 screws.

The block is about **50 × 57 × 14 mm**. In CadQuery it is 10.1 cm³ of plastic. On a 0.4 mm nozzle that is usually a **20 to 40 minute** print. It sits flat. It does not need supports.

This is a new gauge. It does not replace a part in `hardware/stock/`.

## What you are looking at

The big opening is the servo pocket. The servo slides in from the top.

Along the front edge are three columns. Each column is engraved with its clearance: **0.10**, **0.20**, and **0.30**.

In every column, from the pocket outward:

1. Two holes. **M2 is the left hole, M3 is the right hole.**
2. A short slot, the width of an M3 screw plus that column's clearance.
3. A short peg. The peg is 4.0 mm minus that column's clearance.
4. The engraved number.

The pocket floor is engraved `servo +0.20` (or whatever pocket clearance you set).

**Clearance** here means extra size on the whole feature, not extra on each side. A 0.20 mm hole is 0.20 mm wider than the screw, so there is 0.10 mm of air on each side.

| Feature | Nominal | At 0.10 | At 0.20 | At 0.30 |
| --- | --- | --- | --- | --- |
| Servo pocket | 45.40 × 24.80 mm case | one pocket, default extra 0.20 mm | | |
| M2 hole | 2.0 mm | 2.1 | 2.2 | 2.3 |
| M3 hole | 3.0 mm | 3.1 | 3.2 | 3.3 |
| Slot width | 3.0 mm | 3.1 | 3.2 | 3.3 |
| Peg diameter | 4.0 mm | 3.9 | 3.8 | 3.7 |

## How to read the print

Wait until the part is cool. Try the real parts. Do not sand anything yet.

1. **Servo.** Slide it into the pocket, output shaft up. It should go in without forcing and without rattling. That clearance is the one to use for motor pockets.
2. If it will not go, the pocket is tight. In the script, raise `POCKET_CLEARANCE` (try 0.30) and run it again. If it rattles, lower it (try 0.10).
3. **Screws.** Set an M2 screw in the left hole of each column, and an M3 screw in the right hole. The smallest hole the screw drops into, without you pushing, is the clearance to use for that screw.
4. **Slots.** Slide an M3 screw along each slot. If a slot needs a looser number than the round hole, trust the slot. A slot shows when the printer closes a gap.
5. **Pegs.** These are the male version: 4 mm minus the clearance. Use the peg that fits the hole you care about, again without forcing.

The smallest clearance that fits is the one to copy onto the custom arm parts. A fit that needs force will not assemble. A fit that rattles will not hold.

The pocket is 12 mm deep, not the full 28.8 mm of the case. That is enough to feel a tight spot, and it keeps the print short. The horn stays above the plastic.

## Print settings

- Material: **PLA**
- Nozzle: **0.4 mm**
- Layer height: **0.2 mm**
- Walls: 2 or 3
- Infill: 15% (the walls are only 2 mm, so they come out solid anyway)
- **Supports: off**
- Orientation: the big flat face on the bed, pocket opening up, pegs up

A squished first layer can make the bottom of the pocket tight even when the screws fit. If the servo sticks only as it enters, and is fine once it is past the first layers, trust the screw holes and check that the nozzle is not too close to the bed.

## Servo dimensions

`hardware/stock/` has the 15 official SO-101 parts and **no servo model**. These numbers are measured from the SO-ARM100 solid the arm was designed around:

[STEP/SO100/STS3215_03a.step](https://github.com/TheRobotStudio/SO-ARM100/blob/main/STEP/SO100/STS3215_03a.step)

The same shape is also meshed at `Simulation/SO101/assets/sts3215_03a_v1.stl` in that repo (units metres). The STEP is in millimetres. The origin is the centre of the case. +Z points out of the output shaft.

| Measurement | Value |
| --- | --- |
| Case length (X) | 45.40 mm (−22.70 to 22.70) |
| Case width (Y) | 24.80 mm (−12.40 to 12.40) |
| Case depth (Z), the flat part that slides into a pocket | 28.80 mm (−14.40 to 14.40) |
| Overall height, including the output spline and the bottom pins | 39.60 mm |
| Corner radius, cable end | 2.0 mm |
| Corner radius, output end | 3.0 mm |
| Output shaft axis | (12.50, 0) mm |
| Horn screw holes | Ø2.50 mm at (±4.95, ±4.95) from the shaft axis |
| Case mounting holes | Ø1.50 mm at y = ±10.25. Top: x = −16.50 and 4.20. Bottom: x = −20.30 and 4.20 |

Feetech's datasheet lists the catalog outline as **45.2 × 24.7 × 35 mm**. The pocket uses the STEP case, not that catalog line. The printed SO-101 parts were drawn around the STEP.

## Run it in Fusion

You need the folder that contains `fit_test_block.py` and `fit_test_block.manifest`. They have to keep the same name.

1. Open Fusion.
2. Open the **Utilities** tab.
3. Click **Scripts and Add-Ins**. The keyboard shortcut is **Shift+S**.
4. Stay on the **Scripts** tab of that dialog (not Add-Ins).
5. Next to **My Scripts**, click the **+** button.
6. Choose the option that adds a script already on your computer (**Script or add-in from device**, or **Add existing**).
7. Select `fit_test_block.py` inside `hardware/custom/fit_test_block/`.
8. Select **fit_test_block** in the list and click **Run**.

Fusion opens a new design and builds the block. If `EXPORT_STL` is `True` at the top of the script, it also writes `fit_test_block.stl` in this folder. A message tells you the path.

You can instead copy this whole folder into Fusion's scripts folder, then run it from the same dialog:

`%APPDATA%\Autodesk\Autodesk Fusion 360\API\Scripts`

### Change the settings

Two ways:

- **Edit the top of `fit_test_block.py`** (`POCKET_CLEARANCE`, `TEST_CLEARANCES`, `WALL_THICKNESS`, `FLOOR_THICKNESS`, `POCKET_DEPTH`, and the rest) and Run the script again. That makes a new design.
- **Inside the design:** Design workspace, **Modify > Change Parameters**. The sizes are under **User Parameters**. Thicknesses, pocket size, and hole diameters are driven by those parameters, so the model updates when you edit them. The names match the settings (`pocket_clearance`, `clearance_1`, `wall_thickness`, and so on).

`clearance_1` is 0.10 mm, `clearance_2` is 0.20 mm, `clearance_3` is 0.30 mm.

### Export an STL yourself

If you turned `EXPORT_STL` off, or you changed parameters after the script ran:

1. Right-click the body in the browser.
2. **Save As Mesh**.
3. Format **STL**, binary, refinement Medium.
4. Save it somewhere you can find it. The slicer does not care about the file name.

Or use **File > Export** and choose STL.

`STL_PATH` at the top of the script can be a full Windows path, for example `C:\\Users\\Barbod\\Desktop\\fit_test_block.stl`. Leave it as `""` to save next to the script.

## Files

| File | What it is |
| --- | --- |
| `fit_test_block.py` | The Fusion script. Settings are at the top. |
| `fit_test_block.manifest` | Tells Fusion this folder is a script. |
| `fit_layout.py` | Pocket and feature positions, plain Python. No Fusion. |
| `test_layout.py` | Checks the layout. `pytest test_layout.py` |
| `make_reference_stl.py` | Rebuilds the reference mesh with CadQuery. |
| `fit_test_block_reference.stl` | That mesh, for checking the size without Fusion. |

Fusion's `adsk` modules cannot run in CI. The pytest file never imports them. The reference STL is the headless copy of the same layout: 49.6 × 57.0 × 13.6 mm, pocket included, labels engraved.

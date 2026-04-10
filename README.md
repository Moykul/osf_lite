# osf_lite

Stripped OpenSeeFace with ARKit blendshape UDP output for UE5.
Drops everything not needed for headless webcam → UE5 tracking.

## What's here

| File | Role |
|---|---|
| `facetracker_lite.py` | Main entry point (stripped facetracker.py) |
| `arkit_remapper.py`   | Converts OSF features → 61 ARKit floats + UDP sender |
| `tracker.py`          | OSF core tracking (unchanged) |
| `model.py`            | Face model definitions (unchanged) |
| `retinaface.py`       | Face detector (unchanged) |
| `input_reader.py`     | Camera input abstraction (unchanged) |
| `remedian.py`         | Robust median filter (unchanged) |
| `similaritytransform.py` | 3D pose maths (unchanged) |
| `models/`             | Only the models you need (see below) |
| `ARKitFaceReceiver.h/.cpp` | UE5 C++ component |

### Models kept (21 MB total, down from ~63 MB)

| File | Purpose |
|---|---|
| `lm_model3_opt.onnx`        | Landmark tracker (model 3 = best quality/speed) |
| `mnv3_detection_opt.onnx`   | Fast face detector |
| `mnv3_gaze32_split_opt.onnx`| Gaze / eye direction model |
| `retinaface_640x640_opt.onnx` | RetinaFace detector (fallback) |
| `priorbox_640x640.json`     | RetinaFace priors |

## Setup

### Prerequisites
- Python 3.7 or higher
- Webcam
- **Unreal Engine 5 with compatible plugin** to receive ARKit blendshape data via UDP

### Install Python Dependencies

```bash
pip install opencv-python numpy onnxruntime
```

### Optional: Install PyInstaller (for building standalone executable)

```bash
python -m pip install pyinstaller
```

## How to Run

```bash
# Default: webcam 0, ARKit on 127.0.0.1:11574, OSF on 127.0.0.1:11573
python facetracker_lite.py -c 0

# ARKit only (skip OSF packet):
python facetracker_lite.py -c 0 --arkit-only 1

# Remote UE5 machine:
python facetracker_lite.py -c 0 --arkit-ip 192.168.1.50 --arkit-port 11574

# Faster model (less CPU):
python facetracker_lite.py -c 0 --model 1
```

## How to Build Standalone Executable

To create a standalone `.exe` that doesn't require Python to be installed:

```bash
python -m PyInstaller osf_lite.spec
```

The built executable will be located in the `dist/` folder.

## Unreal Engine 5 Integration

**Important:** This tracker sends ARKit blendshape data via UDP. You need a **UE5 plugin or custom receiver component** to read and apply these values in Unreal Engine 5.

### Using the Included C++ Component

The included `ARKitFaceReceiver.h` and `ARKitFaceReceiver.cpp` files provide a ready-to-use UE5 component that receives the UDP data.

## UDP packets

### Port 11573 — original OSF binary (unchanged)
Compatible with any existing OSF receiver.

### Port 11574 — ARKit packet (new)

```
Offset  Size     Field
0       4        Magic: "ARKF"
4       8        Timestamp (double, seconds)
12      1        Success flag (uint8)
13      16       Quaternion x y z w (4 floats)
29      12       Euler pitch yaw roll in degrees (3 floats)
41      12       Head translation x y z (3 floats)
53      244      61 ARKit blendshapes (61 floats, 0–1)
Total   297 bytes
```

### ARKit shape index

```
0  eyeBlinkLeft        7  eyeBlinkRight
1  eyeLookDownLeft     8  eyeLookDownRight
2  eyeLookInLeft       9  eyeLookInRight
3  eyeLookOutLeft      10 eyeLookOutRight
4  eyeLookUpLeft       11 eyeLookUpRight
5  eyeSquintLeft       12 eyeSquintRight
6  eyeWideLeft         13 eyeWideRight
17 jawOpen             18 mouthClose
19 mouthFunnel         20 mouthPucker
21 mouthLeft           22 mouthRight
23 mouthSmileLeft      24 mouthSmileRight
25 mouthFrownLeft      26 mouthFrownRight
41 browDownLeft        42 browDownRight
43 browInnerUp         44 browOuterUpLeft    45 browOuterUpRight
```

Shapes 14–16, 31–40, 46–60 — not tracked by OSF, always 0.

## UE5 C++ setup

1. Copy `ARKitFaceReceiver.h` and `ARKitFaceReceiver.cpp` into your
   project's `Source/YourGame/` directory.

2. Replace `YOURGAME_API` with your project's API macro
   (e.g. `MYGAME_API`).

3. Add to `YourGame.Build.cs`:
   ```csharp
   PrivateDependencyModuleNames.AddRange(new string[] {
       "Networking", "Sockets"
   });
   ```

4. Add the component to any Actor in C++ or Blueprint.

5. In Blueprint, bind the `OnFaceFrameReceived` delegate and use
   `GetShape(EARKitShape)` to read individual blendshapes.

### Coordinate system note

The receiver remaps OpenCV → UE5 conventions automatically:
- Quaternion Y and Z are flipped
- Euler pitch and roll are negated
- Translation axes are reordered: `(Tz, Tx, -Ty)` → UE `(X fwd, Y right, Z up)`

### Smoothing

Set `SmoothingAlpha` (0–1) on the component:
- `0.0` = raw, no smoothing
- `0.25` = light smoothing (default)
- `0.6`  = heavy smoothing, good for slow head turns

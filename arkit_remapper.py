"""
arkit_remapper.py
Converts OpenSeeFace tracking data into a 61-float ARKit blendshape packet
and sends it via UDP.

Packet format (UE5 receiver side):
  Header  : 4 bytes  magic "ARKF"
  Timestamp: 8 bytes double
  Success  : 1 byte  (0/1)
  Head pose: 28 bytes (quat xyzw + euler xyz = 7 floats)
  ARKit    : 244 bytes (61 floats, indices match standard ARKit order)
  Total    : ~285 bytes

ARKit shape index reference:
  0  eyeBlinkLeft        1  eyeLookDownLeft     2  eyeLookInLeft
  3  eyeLookOutLeft      4  eyeLookUpLeft       5  eyeSquintLeft
  6  eyeWideLeft         7  eyeBlinkRight       8  eyeLookDownRight
  9  eyeLookInRight      10 eyeLookOutRight     11 eyeLookUpRight
  12 eyeSquintRight      13 eyeWideRight        14 jawForward
  15 jawLeft             16 jawRight            17 jawOpen
  18 mouthClose          19 mouthFunnel         20 mouthPucker
  21 mouthLeft           22 mouthRight          23 mouthSmileLeft
  24 mouthSmileRight     25 mouthFrownLeft      26 mouthFrownRight
  27 mouthDimpleLeft     28 mouthDimpleRight    29 mouthStretchLeft
  30 mouthStretchRight   31 mouthRollLower      32 mouthRollUpper
  33 mouthShrugLower     34 mouthShrugUpper     35 mouthPressLeft
  36 mouthPressRight     37 mouthLowerDownLeft  38 mouthLowerDownRight
  39 mouthUpperUpLeft    40 mouthUpperUpRight   41 browDownLeft
  42 browDownRight       43 browInnerUp         44 browOuterUpLeft
  45 browOuterUpRight    46 cheekPuff           47 cheekSquintLeft
  48 cheekSquintRight    49 noseSneerLeft       50 noseSneerRight
  51 tongueOut
  52-60 reserved (zero)
"""

import struct
import socket
import math

ARKIT_COUNT = 61
MAGIC = b"ARKF"


def _clamp01(v):
    return max(0.0, min(1.0, v))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _remap(v, in_lo, in_hi, out_lo=0.0, out_hi=1.0):
    """Linear remap + clamp."""
    if in_hi == in_lo:
        return out_lo
    t = (v - in_lo) / (in_hi - in_lo)
    return _clamp(out_lo + t * (out_hi - out_lo), out_lo, out_hi)


class ARKitRemapper:
    """
    Holds per-session calibration state and converts one OSF face frame
    into a 61-float ARKit array.

    OSF feature names (14 total):
        eye_l, eye_r
        eyebrow_steepness_l, eyebrow_updown_l, eyebrow_quirk_l
        eyebrow_steepness_r, eyebrow_updown_r, eyebrow_quirk_r
        mouth_corner_updown_l, mouth_corner_inout_l
        mouth_corner_updown_r, mouth_corner_inout_r
        mouth_open, mouth_wide

    OSF eye_state per eye: [open, eye_y, eye_x, conf]
        eye_y  — vertical gaze in image-space pixels (up = lower value)
        eye_x  — horizontal gaze in image-space pixels

    OSF eye_blink: [right_open, left_open]  (1 = open, 0 = closed)
    """

    # Gaze pixel range — empirical, tune per-camera if needed
    GAZE_Y_RANGE = 8.0
    GAZE_X_RANGE = 8.0

    def __init__(self):
        # Calibration baselines (updated on first confident frame)
        self._gaze_y_center = [0.0, 0.0]   # [right, left]
        self._gaze_x_center = [0.0, 0.0]
        self._calibrated = False
        self._calib_samples = 0
        self._calib_target = 15             # frames before locking baseline

    def calibrate(self, eye_state):
        """Call this for the first N frames to set gaze neutral position."""
        if self._calibrated:
            return
        for i in range(2):
            self._gaze_y_center[i] += eye_state[i][1]
            self._gaze_x_center[i] += eye_state[i][2]
        self._calib_samples += 1
        if self._calib_samples >= self._calib_target:
            self._gaze_y_center[0] /= self._calib_samples
            self._gaze_y_center[1] /= self._calib_samples
            self._gaze_x_center[0] /= self._calib_samples
            self._gaze_x_center[1] /= self._calib_samples
            self._calibrated = True

    def remap(self, features, eye_blink, eye_state):
        """
        Parameters
        ----------
        features   : dict  — f.current_features from OSF
        eye_blink  : list  — [right_open, left_open]  0–1
        eye_state  : list  — [[open,y,x,conf], [open,y,x,conf]] right then left

        Returns
        -------
        list of 61 floats in ARKit index order
        """
        ar = [0.0] * ARKIT_COUNT

        # ── Convenience getters ─────────────────────────────────────────────
        def f(name):
            return float(features.get(name, 0.0))

        # ── Gaze calibration ────────────────────────────────────────────────
        if not self._calibrated:
            self.calibrate(eye_state)

        # ── Eye blink / open / squint / wide ────────────────────────────────
        # eye_blink: 1=open, 0=closed  →  ARKit blink: 1=closed, 0=open
        blink_r = _clamp01(1.0 - eye_blink[0])
        blink_l = _clamp01(1.0 - eye_blink[1])

        ar[0]  = blink_l                            # eyeBlinkLeft
        ar[7]  = blink_r                            # eyeBlinkRight

        # eyeWide: inverse of blink, only in the upper register
        ar[6]  = _clamp01(-f("eye_l") * 1.5)       # eyeWideLeft
        ar[13] = _clamp01(-f("eye_r") * 1.5)       # eyeWideRight

        # eyeSquint: partial closure without full blink
        # fire when eye feature is in mid range [0.2 .. 0.6]
        squint_l = _remap(f("eye_l"), 0.2, 0.6)
        squint_r = _remap(f("eye_r"), 0.2, 0.6)
        ar[5]  = squint_l * (1.0 - blink_l)        # eyeSquintLeft
        ar[12] = squint_r * (1.0 - blink_r)        # eyeSquintRight

        # ── Gaze direction ───────────────────────────────────────────────────
        # eye_state[i] = [open, y, x, conf]
        # right eye = index 0, left eye = index 1
        for side, (es_idx, look_down, look_up, look_in, look_out) in enumerate([
            (0, 1, 4, 2, 3),    # right eye  → ARKit 8 11 9 10
            (1, 8, 11, 9, 10),  # left eye   → ARKit 1  4  2  3
        ]):
            gy = eye_state[es_idx][1] - self._gaze_y_center[es_idx]
            gx = eye_state[es_idx][2] - self._gaze_x_center[es_idx]
            gy_n = _clamp(gy / self.GAZE_Y_RANGE, -1.0, 1.0)
            gx_n = _clamp(gx / self.GAZE_X_RANGE, -1.0, 1.0)

            # Y axis: positive gy_n = looking down
            ar[look_down] = _clamp01( gy_n)
            ar[look_up]   = _clamp01(-gy_n)

            # X axis convention differs L/R eye for "in" vs "out"
            if side == 0:   # right eye: gaze left = look_in
                ar[look_in]  = _clamp01(-gx_n)
                ar[look_out] = _clamp01( gx_n)
            else:           # left eye: gaze right = look_in
                ar[look_in]  = _clamp01( gx_n)
                ar[look_out] = _clamp01(-gx_n)

        # ── Jaw ──────────────────────────────────────────────────────────────
        # jawForward/Left/Right: not tracked → stay 0
        ar[17] = _clamp01(f("mouth_open"))          # jawOpen

        # ── Mouth ────────────────────────────────────────────────────────────
        ar[18] = _clamp01(1.0 - f("mouth_open"))    # mouthClose (inverse jaw)

        # mouth_wide negative → pucker/funnel
        wide = f("mouth_wide")
        ar[19] = _clamp01(-wide * 1.2)              # mouthFunnel
        ar[20] = _clamp01(-wide * 0.8)              # mouthPucker

        # mouth corners left/right lateral movement
        ar[21] = _clamp01( f("mouth_corner_inout_l"))  # mouthLeft
        ar[22] = _clamp01( f("mouth_corner_inout_r"))  # mouthRight

        # smile / frown from corner up/down
        cup_l = f("mouth_corner_updown_l")
        cup_r = f("mouth_corner_updown_r")
        ar[23] = _clamp01( cup_l)                   # mouthSmileLeft
        ar[24] = _clamp01( cup_r)                   # mouthSmileRight
        ar[25] = _clamp01(-cup_l)                   # mouthFrownLeft
        ar[26] = _clamp01(-cup_r)                   # mouthFrownRight

        # dimple — tight inward pull, correlates with corner_inout
        ar[27] = _clamp01(f("mouth_corner_inout_l") * 0.6)  # mouthDimpleLeft
        ar[28] = _clamp01(f("mouth_corner_inout_r") * 0.6)  # mouthDimpleRight

        # stretch — wide mouth, split left/right
        ar[29] = _clamp01(wide * 0.7)               # mouthStretchLeft
        ar[30] = _clamp01(wide * 0.7)               # mouthStretchRight

        # mouthRollLower/Upper, ShrugLower/Upper, PressL/R,
        # LowerDownL/R, UpperUpL/R → no OSF source, stay 0

        # ── Brows ────────────────────────────────────────────────────────────
        bup_l = f("eyebrow_updown_l")
        bup_r = f("eyebrow_updown_r")
        ar[41] = _clamp01(-bup_l)                   # browDownLeft
        ar[42] = _clamp01(-bup_r)                   # browDownRight
        ar[43] = _clamp01(                          # browInnerUp (quirk avg)
            (f("eyebrow_quirk_l") + f("eyebrow_quirk_r")) * 0.5
        )
        ar[44] = _clamp01( bup_l)                   # browOuterUpLeft
        ar[45] = _clamp01( bup_r)                   # browOuterUpRight

        # ── Cheeks, nose, tongue → no OSF source, stay 0 ────────────────────

        return ar


class ARKitUDPSender:
    """
    Wraps a UDP socket and serialises an ARKit packet.

    Packet layout
    -------------
    Offset  Size  Type      Field
    0       4     char[4]   magic "ARKF"
    4       8     double    timestamp (seconds, float64)
    12      1     uint8     success flag
    13      4     float     quat_x
    17      4     float     quat_y
    21      4     float     quat_z
    25      4     float     quat_w
    29      4     float     euler_pitch  (degrees)
    33      4     float     euler_yaw
    37      4     float     euler_roll
    41      4     float     head_x
    45      4     float     head_y
    49      4     float     head_z
    53      244   float[61] ARKit blendshapes
    Total   297 bytes
    """

    HEADER_FMT  = "<4sdBffffffffff"   # magic(4s) + ts(d) + success(B) + quat xyzw(4f) + euler pyr(3f) + trans xyz(3f)
    SHAPES_FMT  = "<" + "f" * ARKIT_COUNT
    TOTAL_BYTES = struct.calcsize(HEADER_FMT) + struct.calcsize(SHAPES_FMT)

    def __init__(self, ip="127.0.0.1", port=11574):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[ARKit] UDP sender → {ip}:{port}  packet size {self.TOTAL_BYTES} bytes")

    def send(self, timestamp, success, quaternion, euler, translation, arkit_shapes):
        """
        Parameters
        ----------
        timestamp    : float   — Unix time
        success      : bool
        quaternion   : [x,y,z,w]
        euler        : [pitch, yaw, roll]  degrees
        translation  : [x,y,z]
        arkit_shapes : list[61 floats]
        """
        header = struct.pack(
            self.HEADER_FMT,
            MAGIC,
            timestamp,
            1 if success else 0,
            quaternion[0], quaternion[1], quaternion[2], quaternion[3],
            euler[0], euler[1], euler[2],
            translation[0], translation[1], translation[2],
        )
        shapes = struct.pack(self.SHAPES_FMT, *arkit_shapes)
        self.sock.sendto(header + shapes, self.addr)

    def close(self):
        self.sock.close()
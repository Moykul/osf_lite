"""
calibration.py - Per-user calibration for osf_lite.
Three timed poses: Neutral, Maximum, Blink.
Saves/loads calibration.json.
"""

import json
import os
import time
import numpy as np

CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration.json")

OSF_FEATURES = [
    "eye_l", "eye_r",
    "eyebrow_steepness_l", "eyebrow_updown_l", "eyebrow_quirk_l",
    "eyebrow_steepness_r", "eyebrow_updown_r", "eyebrow_quirk_r",
    "mouth_corner_updown_l", "mouth_corner_inout_l",
    "mouth_corner_updown_r", "mouth_corner_inout_r",
    "mouth_open", "mouth_wide",
]
GAZE_FEATURES = ["gaze_y_r", "gaze_x_r", "gaze_y_l", "gaze_x_l"]
ALL_FEATURES  = OSF_FEATURES + GAZE_FEATURES

FALLBACK = {
    "eye_l":                 (-0.5,  0.5),
    "eye_r":                 (-0.5,  0.5),
    "eyebrow_steepness_l":   (-10.0, 10.0),
    "eyebrow_updown_l":      (-0.3,  0.3),
    "eyebrow_quirk_l":       (-0.1,  0.3),
    "eyebrow_steepness_r":   (-10.0, 10.0),
    "eyebrow_updown_r":      (-0.3,  0.3),
    "eyebrow_quirk_r":       (-0.1,  0.3),
    "mouth_corner_updown_l": (-0.3,  0.3),
    "mouth_corner_inout_l":  (-0.1,  0.3),
    "mouth_corner_updown_r": (-0.3,  0.3),
    "mouth_corner_inout_r":  (-0.1,  0.3),
    "mouth_open":            ( 0.0,  0.6),
    "mouth_wide":            (-0.3,  0.3),
    "gaze_y_r":              (-8.0,  8.0),
    "gaze_x_r":              (-8.0,  8.0),
    "gaze_y_l":              (-8.0,  8.0),
    "gaze_x_l":              (-8.0,  8.0),
}


class CalibrationData:
    def __init__(self):
        self.neutral    = {k: 0.0            for k in ALL_FEATURES}
        self.range_lo   = {k: FALLBACK[k][0] for k in ALL_FEATURES}
        self.range_hi   = {k: FALLBACK[k][1] for k in ALL_FEATURES}
        self.calibrated = False

    def normalise(self, name, raw):
        n  = self.neutral.get(name, 0.0)
        lo = self.range_lo.get(name, -1.0)
        hi = self.range_hi.get(name,  1.0)
        v  = raw - n
        if hi == lo:
            return 0.0
        return max(0.0, min(1.0, v / (hi - lo) if v >= 0 else v / abs(lo)))

    def normalise_signed(self, name, raw):
        n    = self.neutral.get(name, 0.0)
        lo   = self.range_lo.get(name, -1.0)
        hi   = self.range_hi.get(name,  1.0)
        v    = raw - n
        span = max(abs(hi), abs(lo))
        return 0.0 if span == 0 else max(-1.0, min(1.0, v / span))

    def save(self, path=None):
        path = path or CALIB_FILE
        with open(path, "w") as fh:
            json.dump({"neutral": self.neutral, "range_lo": self.range_lo,
                       "range_hi": self.range_hi, "calibrated": True}, fh, indent=2)
        print(f"[Calib] Saved to {path}")

    def load(self, path=None):
        path = path or CALIB_FILE
        if not os.path.exists(path):
            return False
        try:
            with open(path) as fh:
                d = json.load(fh)
            self.neutral    = d.get("neutral",    self.neutral)
            self.range_lo   = d.get("range_lo",   self.range_lo)
            self.range_hi   = d.get("range_hi",   self.range_hi)
            self.calibrated = d.get("calibrated", False)
            print(f"[Calib] Loaded from {path}")
            return True
        except Exception as e:
            print(f"[Calib] Load failed: {e}")
            return False


class CalibrationSession:
    POSES = [
        ("NEUTRAL",           "Relax your face. Look straight at the camera. Hold still."),
        ("MAXIMUM EXPRESSION","Brows UP, jaw OPEN, big SMILE. Look left/right/up/down."),
        ("BLINK",             "Close BOTH eyes fully and hold."),
    ]
    SETTLE  = 1.5
    RECORD  = 3.0

    def __init__(self):
        self._pose        = 0
        self._phase       = "settle"
        self._t           = time.time()
        self._samples     = {k: [] for k in ALL_FEATURES}
        self._pose_data   = []
        self.complete     = False

    @property
    def pose_name(self):    return self.POSES[self._pose][0]
    @property
    def instruction(self):  return self.POSES[self._pose][1]
    @property
    def pose_index(self):   return self._pose
    @property
    def pose_count(self):   return len(self.POSES)
    @property
    def phase(self):        return self._phase
    @property
    def progress(self):
        dur = self.SETTLE if self._phase == "settle" else self.RECORD
        return min(1.0, (time.time() - self._t) / dur)

    def feed(self, features, eye_state):
        if self.complete:
            return
        elapsed = time.time() - self._t
        if self._phase == "settle":
            if elapsed >= self.SETTLE:
                self._phase = "record"
                self._t     = time.time()
                self._samples = {k: [] for k in ALL_FEATURES}
            return
        # recording
        for k in OSF_FEATURES:
            self._samples[k].append(float(features.get(k, 0.0)))
        if eye_state is not None:
            self._samples["gaze_y_r"].append(eye_state[0][1])
            self._samples["gaze_x_r"].append(eye_state[0][2])
            self._samples["gaze_y_l"].append(eye_state[1][1])
            self._samples["gaze_x_l"].append(eye_state[1][2])
        if elapsed >= self.RECORD:
            self._save_and_advance()

    def _save_and_advance(self):
        s = self._samples
        self._pose_data.append({
            "mean": {k: float(np.mean(v)) if v else 0.0 for k, v in s.items()},
            "max":  {k: float(np.max(v))  if v else 0.0 for k, v in s.items()},
            "min":  {k: float(np.min(v))  if v else 0.0 for k, v in s.items()},
        })
        self._pose  += 1
        if self._pose >= len(self.POSES):
            self.complete = True
            return
        self._phase = "settle"
        self._t     = time.time()

    def build(self):
        assert self.complete
        cal     = CalibrationData()
        neutral = self._pose_data[0]["mean"]
        maxpose = self._pose_data[1]
        blink   = self._pose_data[2]
        MIN     = 0.05
        for k in ALL_FEATURES:
            n  = neutral.get(k, 0.0)
            cal.neutral[k] = n
            hi = maxpose["max"].get(k, n) - n
            lo = maxpose["min"].get(k, n) - n
            if abs(hi) < MIN: hi =  (FALLBACK[k][1] - FALLBACK[k][0]) * 0.5
            if abs(lo) < MIN: lo = -(FALLBACK[k][1] - FALLBACK[k][0]) * 0.5
            cal.range_hi[k] = hi
            cal.range_lo[k] = lo
        cal.neutral["eye_l"] = blink["mean"].get("eye_l", cal.neutral["eye_l"])
        cal.neutral["eye_r"] = blink["mean"].get("eye_r", cal.neutral["eye_r"])
        cal.calibrated = True
        return cal
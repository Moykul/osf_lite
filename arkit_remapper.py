"""
arkit_remapper.py - OSF features -> 61-float ARKit packet + UDP sender.
Uses CalibrationData when available, fallback ranges otherwise.
"""

import struct
import socket
from calibration import CalibrationData

ARKIT_COUNT = 61
MAGIC       = b"ARKF"

def _c(v):
    return max(0.0, min(1.0, v))


class ARKitRemapper:
    def __init__(self, calib=None):
        self.calib = calib or CalibrationData()

    def set_calibration(self, calib):
        self.calib = calib

    def remap(self, features, eye_blink, eye_state):
        ar  = [0.0] * ARKIT_COUNT
        cal = self.calib

        def f(k):   return float(features.get(k, 0.0))
        def n(k):   return cal.normalise(k, f(k))
        def ns(k):  return cal.normalise_signed(k, f(k))

        # Eyes
        bl = _c(1.0 - eye_blink[1])
        br = _c(1.0 - eye_blink[0])
        ar[0]  = bl                                             # eyeBlinkLeft
        ar[7]  = br                                             # eyeBlinkRight
        ar[6]  = _c(-ns("eye_l"))                               # eyeWideLeft
        ar[13] = _c(-ns("eye_r"))                               # eyeWideRight
        ar[5]  = _c(ns("eye_l") * 0.5 + 0.5) * (1.0 - bl)     # eyeSquintLeft
        ar[12] = _c(ns("eye_r") * 0.5 + 0.5) * (1.0 - br)     # eyeSquintRight

        # Gaze
        if eye_state is not None:
            ny_r = cal.neutral.get("gaze_y_r", 0.0)
            nx_r = cal.neutral.get("gaze_x_r", 0.0)
            ny_l = cal.neutral.get("gaze_y_l", 0.0)
            nx_l = cal.neutral.get("gaze_x_l", 0.0)
            ry   = max(abs(cal.range_hi.get("gaze_y_r", 8.0)), abs(cal.range_lo.get("gaze_y_r", 8.0)))
            rx   = max(abs(cal.range_hi.get("gaze_x_r", 8.0)), abs(cal.range_lo.get("gaze_x_r", 8.0)))
            for side, (ei, ld, lu, li, lo, ny, nx) in enumerate([
                (0,  1,  4,  2,  3, ny_r, nx_r),
                (1,  8, 11,  9, 10, ny_l, nx_l),
            ]):
                gy = max(-1.0, min(1.0, (eye_state[ei][1] - ny) / max(ry, 1e-3)))
                gx = max(-1.0, min(1.0, (eye_state[ei][2] - nx) / max(rx, 1e-3)))
                ar[ld] = _c( gy);  ar[lu] = _c(-gy)
                if side == 0:
                    ar[li] = _c(-gx);  ar[lo] = _c( gx)
                else:
                    ar[li] = _c( gx);  ar[lo] = _c(-gx)

        # Jaw / mouth
        ar[17] = n("mouth_open")
        ar[18] = _c(1.0 - n("mouth_open"))
        ws     = ns("mouth_wide")
        ar[19] = _c(-ws);        ar[20] = _c(-ws * 0.8)
        ar[29] = _c( ws);        ar[30] = _c( ws)
        ar[21] = _c( ns("mouth_corner_inout_l"))
        ar[22] = _c( ns("mouth_corner_inout_r"))
        ar[27] = _c( ns("mouth_corner_inout_l") * 0.6)
        ar[28] = _c( ns("mouth_corner_inout_r") * 0.6)
        cl     = ns("mouth_corner_updown_l")
        cr     = ns("mouth_corner_updown_r")
        ar[23] = _c( cl);  ar[24] = _c( cr)
        ar[25] = _c(-cl);  ar[26] = _c(-cr)

        # Brows
        bl_b   = ns("eyebrow_updown_l")
        br_b   = ns("eyebrow_updown_r")
        ar[41] = _c(-bl_b);  ar[42] = _c(-br_b)
        ar[43] = _c((ns("eyebrow_quirk_l") + ns("eyebrow_quirk_r")) * 0.5)
        ar[44] = _c( bl_b);  ar[45] = _c( br_b)

        return ar


class ARKitUDPSender:
    HEADER_FMT  = "<4sdBffffffffff"
    SHAPES_FMT  = "<" + "f" * ARKIT_COUNT
    TOTAL_BYTES = struct.calcsize(HEADER_FMT) + struct.calcsize(SHAPES_FMT)

    def __init__(self, ip="127.0.0.1", port=11574):
        self.addr = (ip, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[ARKit] UDP sender -> {ip}:{port}  packet size {self.TOTAL_BYTES} bytes")

    def send(self, timestamp, success, quaternion, euler, translation, shapes):
        header = struct.pack(self.HEADER_FMT,
            b"ARKF", timestamp, 1 if success else 0,
            quaternion[0], quaternion[1], quaternion[2], quaternion[3],
            euler[0], euler[1], euler[2],
            translation[0], translation[1], translation[2])
        self.sock.sendto(header + struct.pack(self.SHAPES_FMT, *shapes), self.addr)

    def close(self):
        self.sock.close()
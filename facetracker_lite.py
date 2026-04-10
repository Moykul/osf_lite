"""
facetracker_lite.py
Stripped OpenSeeFace tracker with ARKit remapper output.
Sends TWO UDP streams:
  Port 11573  — original OSF binary packet  (unchanged, for compatibility)
  Port 11574  — ARKit 61-float packet        (for UE5 LiveLink / custom receiver)

Usage:
  python facetracker_lite.py -c 0 -m 3
  python facetracker_lite.py -c 0 -m 3 --arkit-ip 127.0.0.1 --arkit-port 11574

Original OSF arguments still work.  New arguments:
  --arkit-ip    IP for ARKit UDP output  (default 127.0.0.1)
  --arkit-port  Port for ARKit UDP       (default 11574)
  --arkit-only  When 1, skip the original OSF packet           (default 0)
  --no-arkit    When 1, skip the ARKit packet                  (default 0)
"""

import copy
import os
import sys
import argparse
import traceback
import gc

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i",  "--ip",           default="127.0.0.1")
parser.add_argument("-p",  "--port",         type=int, default=11573)
parser.add_argument("-W",  "--width",        type=int, default=640)
parser.add_argument("-H",  "--height",       type=int, default=360)
parser.add_argument("-c",  "--capture",      default="0")
parser.add_argument("-M",  "--mirror-input", action="store_true")
parser.add_argument("-m",  "--max-threads",  type=int, default=1)
parser.add_argument("-t",  "--threshold",    type=float, default=None)
parser.add_argument("-d",  "--detection-threshold", type=float, default=0.6)
parser.add_argument("-v",  "--visualize",    type=int, default=0)
parser.add_argument("-s",  "--silent",       type=int, default=-1)
parser.add_argument("--faces",               type=int, default=1)
parser.add_argument("--scan-every",          type=int, default=3)
parser.add_argument("--discard-after",       type=int, default=10)
parser.add_argument("--max-feature-updates", type=int, default=900)
parser.add_argument("--no-3d-adapt",         type=int, default=1)
parser.add_argument("--model",               type=int, default=3,
                    choices=[-3, -2, -1, 0, 1, 2, 3, 4])
parser.add_argument("--model-dir",           default=None)
parser.add_argument("--gaze-tracking",       type=int, default=1)
parser.add_argument("--raw-rgb",             type=int, default=0)
# ARKit output args
parser.add_argument("--arkit-ip",            default="127.0.0.1")
parser.add_argument("--arkit-port",          type=int, default=11574)
parser.add_argument("--arkit-only",          type=int, default=0)
parser.add_argument("--no-arkit",            type=int, default=0)
parser.add_argument("--vis",                 type=int, default=1,
                    help="Show visualiser window (1=on, 0=off)")
if os.name == 'nt':
    parser.add_argument("-l", "--list-cameras",     type=int, default=0)
    parser.add_argument("-a", "--list-dcaps",       type=int, default=None)
    parser.add_argument("-F", "--fps",              type=int, default=24)
    parser.add_argument("-D", "--dcap",             type=int, default=None)
    parser.add_argument("-B", "--blackmagic",       type=int, default=0)
    parser.add_argument("--use-dshowcapture",       type=int, default=1)
    parser.add_argument("--blackmagic-options",     type=str, default=None)
    parser.add_argument("--priority",               type=int, default=None,
                        choices=[0, 1, 2, 3, 4, 5])
args = parser.parse_args()

os.environ["OMP_NUM_THREADS"] = str(args.max_threads)

# ── Windows camera listing (unchanged) ──────────────────────────────────────
if os.name == 'nt' and args.list_cameras > 0:
    import dshowcapture
    if args.blackmagic == 1:
        dshowcapture.set_bm_enabled(True)
    caps = dshowcapture.get_capture_list()
    for i, cam in enumerate(caps):
        if args.list_cameras == 1:
            print(f"{i}: {cam['name']}")
        else:
            print(f"{cam['name']}")
    sys.exit(0)

import numpy as np
import time
import cv2
import socket
import struct
from input_reader import InputReader, VideoReader, try_int
from tracker import Tracker, get_model_base_path
from arkit_remapper import ARKitRemapper, ARKitUDPSender
from camera_picker import pick_camera
from visualiser import Visualiser

if os.name == 'nt':
    from input_reader import DShowCaptureReader

# ── OSF feature list (unchanged order — do not reorder) ─────────────────────
OSF_FEATURES = [
    "eye_l", "eye_r",
    "eyebrow_steepness_l", "eyebrow_updown_l", "eyebrow_quirk_l",
    "eyebrow_steepness_r", "eyebrow_updown_r", "eyebrow_quirk_r",
    "mouth_corner_updown_l", "mouth_corner_inout_l",
    "mouth_corner_updown_r", "mouth_corner_inout_r",
    "mouth_open", "mouth_wide",
]

# ── Sockets and remapper ─────────────────────────────────────────────────────
osf_sock   = None
arkit_sender = None
arkit_remap  = ARKitRemapper()
vis          = Visualiser() if args.vis else None

# auto-silent when visualiser is running
if args.silent == -1:
    args.silent = 1 if args.vis else 0

if args.no_arkit == 0:
    arkit_sender = ARKitUDPSender(args.arkit_ip, args.arkit_port)

# ── State ────────────────────────────────────────────────────────────────────
fps = 0
dcap = None
use_dshowcapture_flag = False

# ── Camera picker ────────────────────────────────────────────────────────────
# If capture is still the default "0" and vis is on, show the picker.
# Pass -c N explicitly to skip it.
_capture_was_default = (args.capture == "0")
if _capture_was_default and args.vis:
    _picked = pick_camera(default_index=0)
    args.capture = str(_picked)
    print(f"[OSF Lite] Camera selected: {args.capture}")

if os.name == 'nt':
    fps = args.fps
    dcap = args.dcap
    use_dshowcapture_flag = (args.use_dshowcapture == 1)
    input_reader = InputReader(
        args.capture, args.raw_rgb, args.width, args.height, fps,
        use_dshowcapture=use_dshowcapture_flag, dcap=dcap,
    )
    if args.dcap == -1 and type(input_reader) == DShowCaptureReader:
        fps = min(fps, input_reader.device.get_fps())
else:
    input_reader = InputReader(
        args.capture, args.raw_rgb, args.width, args.height, fps,
        use_dshowcapture=False,
    )

if type(input_reader.reader) == VideoReader:
    fps = 0

first        = True
height       = 0
width        = 0
tracker      = None
frame_count  = 0

target_ip    = args.ip
target_port  = args.port


def build_osf_packet(f, width, height, now, features):
    """Reproduce the original OSF binary packet verbatim."""
    packet = bytearray()
    if f.eye_blink is None:
        f.eye_blink = [1, 1]
    packet.extend(struct.pack("d", now))
    packet.extend(struct.pack("i", f.id))
    packet.extend(struct.pack("f", width))
    packet.extend(struct.pack("f", height))
    packet.extend(struct.pack("f", f.eye_blink[0]))
    packet.extend(struct.pack("f", f.eye_blink[1]))
    packet.extend(struct.pack("B", 1 if f.success else 0))
    packet.extend(struct.pack("f", f.pnp_error))
    for v in f.quaternion:
        packet.extend(struct.pack("f", v))
    for v in f.euler:
        packet.extend(struct.pack("f", v))
    for v in f.translation:
        packet.extend(struct.pack("f", v))
    for _, (y, x, c) in enumerate(f.lms):
        packet.extend(struct.pack("f", y))
        packet.extend(struct.pack("f", x))
    for (x, y, z) in f.pts_3d:
        packet.extend(struct.pack("f",  x))
        packet.extend(struct.pack("f", -y))
        packet.extend(struct.pack("f", -z))
    for feat in features:
        v = f.current_features.get(feat, 0.0) if f.current_features else 0.0
        packet.extend(struct.pack("f", v))
    return packet


# ── Main loop ────────────────────────────────────────────────────────────────
try:
    attempt         = 0
    frame_time      = time.perf_counter()
    target_duration = 0
    if fps > 0:
        target_duration = 1.0 / float(fps)
    repeat      = (type(input_reader.reader) == VideoReader)
    need_reinit = 0
    failures    = 0
    source_name = input_reader.name

    while repeat or input_reader.is_open():
        if not input_reader.is_open() or need_reinit == 1:
            input_reader = InputReader(
                args.capture, args.raw_rgb, args.width, args.height, fps,
                use_dshowcapture=use_dshowcapture_flag, dcap=dcap,
            )
            if input_reader.name != source_name:
                print(f"[ERR] Camera reinit gave {input_reader.name}, expected {source_name}")
                sys.exit(1)
            need_reinit = 2
            time.sleep(0.02)
            continue

        if not input_reader.is_ready():
            time.sleep(0.02)
            continue

        ret, frame = input_reader.read()
        if ret and args.mirror_input:
            frame = cv2.flip(frame, 1)

        if not ret:
            if repeat:
                if need_reinit == 0:
                    need_reinit = 1
                continue
            elif str(args.capture) == str(try_int(args.capture)):
                attempt += 1
                if attempt > 30:
                    break
                time.sleep(0.02)
                if attempt == 3:
                    need_reinit = 1
                continue
            else:
                break

        attempt = 0
        need_reinit = 0
        frame_count += 1
        now = time.time()

        if first:
            first = False
            height, width, _ = frame.shape
            if args.arkit_only == 0:
                osf_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            tracker = Tracker(
                width, height,
                threshold=args.threshold,
                max_threads=args.max_threads,
                max_faces=args.faces,
                discard_after=args.discard_after,
                scan_every=args.scan_every,
                silent=(args.silent == 1),
                model_type=args.model,
                model_dir=args.model_dir,
                no_gaze=(args.gaze_tracking == 0 or args.model == -1),
                detection_threshold=args.detection_threshold,
                use_retinaface=0,
                max_feature_updates=args.max_feature_updates,
                static_model=(args.no_3d_adapt == 1),
            )

        try:
            faces = tracker.predict(frame)
        except Exception:
            traceback.print_exc()
            continue

        detected = False
        last_face   = None
        last_shapes = None

        for face_num, f in enumerate(faces):
            if f.eye_blink is None:
                f.eye_blink = [1, 1]
            if f.current_features is None:
                f.current_features = {}
            for feat in OSF_FEATURES:
                if feat not in f.current_features:
                    f.current_features[feat] = 0.0

            detected = True
            last_face = f

            # ── Send original OSF packet ─────────────────────────────────
            if args.arkit_only == 0 and osf_sock is not None:
                osf_pkt = build_osf_packet(f, width, height, now, OSF_FEATURES)
                osf_sock.sendto(osf_pkt, (target_ip, target_port))

            # ── Send ARKit packet ────────────────────────────────────────
            if arkit_sender is not None and f.success:
                eye_state = f.eye_state if f.eye_state is not None else [
                    [1.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                ]
                arkit_shapes = arkit_remap.remap(
                    f.current_features,
                    f.eye_blink,
                    eye_state,
                )
                arkit_sender.send(
                    now,
                    f.success,
                    f.quaternion,
                    f.euler,
                    f.translation,
                    arkit_shapes,
                )
                last_shapes = arkit_shapes

            if args.silent != 1:
                r = "O" if f.eye_blink[0] > 0.30 else "-"
                l = "O" if f.eye_blink[1] > 0.30 else "-"
                print(
                    f"[{frame_count:5d}] face={f.id} "
                    f"eyes={r}/{l} "
                    f"jaw={f.current_features.get('mouth_open', 0):.2f} "
                    f"conf={f.conf:.2f}",
                    end="\r",
                )

        if vis is not None:
            elapsed = time.perf_counter() - frame_time
            fps_display = 1.0 / elapsed if elapsed > 0 else 0
            if vis.update(frame, last_face, last_shapes, fps_display):
                break
        elif args.visualize > 0 and detected:
            cv2.imshow("OSF Lite", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if target_duration > 0:
            remaining = target_duration - (time.perf_counter() - frame_time)
            if remaining > 0:
                time.sleep(remaining)
        frame_time = time.perf_counter()

except KeyboardInterrupt:
    print("\n[OSF Lite] Interrupted.")
except Exception:
    traceback.print_exc()
finally:
    if osf_sock:
        osf_sock.close()
    if arkit_sender:
        arkit_sender.close()
    gc.collect()
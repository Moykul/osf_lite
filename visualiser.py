"""
visualiser.py - Single window: webcam+landmarks left, ARKit bars right.
Calibration overlay shown during calibration phase.
"""

import cv2
import numpy as np

SHAPES = [
    (0,  "EyeBlinkL"),  (7,  "EyeBlinkR"),
    (6,  "EyeWideL"),   (13, "EyeWideR"),
    (5,  "EyeSquintL"), (12, "EyeSquintR"),
    (1,  "EyeLookDnL"), (4,  "EyeLookUpL"),
    (2,  "EyeLookInL"), (3,  "EyeLookOutL"),
    (8,  "EyeLookDnR"), (11, "EyeLookUpR"),
    (9,  "EyeLookInR"), (10, "EyeLookOutR"),
    (41, "BrowDnL"),    (42, "BrowDnR"),
    (43, "BrowInnerUp"),(44, "BrowOuterUpL"),
    (45, "BrowOuterUpR"),(17, "JawOpen"),
    (18, "MouthClose"), (19, "MouthFunnel"),
    (20, "MouthPucker"),(21, "MouthLeft"),
    (22, "MouthRight"), (23, "SmileLeft"),
    (24, "SmileRight"), (25, "FrownLeft"),
    (26, "FrownRight"), (27, "DimpleLeft"),
    (28, "DimpleRight"),(29, "StretchLeft"),
]

BG      = (18,  16,  12)
PANEL   = (28,  26,  22)
ACCENT  = (44, 124, 245)
EYE_C   = (60, 180, 100)
BROW_C  = (60, 160, 220)
MOUTH_C = (80, 100, 240)
TEXT    = (200, 200, 200)
DIM     = (100, 100, 100)
LM_C    = (44, 124, 245)
LM_EYE  = (60, 220, 160)
GRID    = (40,  38,  34)
FPS_C   = (80, 200, 120)
NOFACE  = (60,  60, 180)
GREEN   = (60, 200, 80)

F  = cv2.FONT_HERSHEY_SIMPLEX
FM = cv2.FONT_HERSHEY_PLAIN

def _bc(idx):
    if idx <= 13: return EYE_C
    if idx <= 18: return BROW_C
    return MOUTH_C


class Visualiser:
    WIN  = "OSF Lite"
    CW   = 640
    CH   = 360
    BW   = 370
    BH   = 7
    BP   = 2
    BL   = 14
    BM   = 180
    HDR  = 36
    FTR  = 28

    def __init__(self):
        self._tw = self.CW + self.BW
        self._th = self.CH + self.HDR + self.FTR
        self._cv = np.zeros((self._th, self._tw, 3), dtype=np.uint8)
        cv2.namedWindow(self.WIN, cv2.WINDOW_AUTOSIZE)

    def update(self, frame, face, shapes, fps):
        c = self._cv
        c[:] = BG
        self._header(c, fps, face is not None)
        self._cam(c, frame, face)
        self._bars(c, shapes)
        self._footer(c)
        cv2.imshow(self.WIN, c)
        k = cv2.waitKey(1) & 0xFF
        return k == ord('q') or k == 27 or cv2.getWindowProperty(self.WIN, cv2.WND_PROP_VISIBLE) < 1

    def show_calibration(self, frame, session):
        c = self._cv
        c[:] = BG

        # show webcam feed dimmed so user can see their face
        if frame is not None:
            h, w = frame.shape[:2]
            sc = min(self.CW / w, self.CH / h)
            nw, nh = int(w * sc), int(h * sc)
            ox = (self.CW - nw) // 2
            oy = self.HDR
            c[oy:oy+nh, ox:ox+nw] = cv2.resize(frame, (nw, nh))
            dark = c[oy:oy+nh, ox:ox+nw].copy()
            cv2.rectangle(dark, (0, 0), (nw, nh), (0, 0, 0), -1)
            cv2.addWeighted(dark, 0.45, c[oy:oy+nh, ox:ox+nw], 0.55, 0, c[oy:oy+nh, ox:ox+nw])

        # header bar
        cv2.rectangle(c, (0, 0), (self._tw, self.HDR), PANEL, -1)
        cv2.putText(c, "OSF Lite  -  CALIBRATION", (12, 24), F, 0.6, ACCENT, 1, cv2.LINE_AA)

        # pose label
        py = self.HDR + 55
        cv2.putText(c, f"Pose {session.pose_index + 1} of {session.pose_count}",
                    (30, py), F, 0.5, DIM, 1, cv2.LINE_AA)
        cv2.putText(c, session.pose_name, (30, py + 38), F, 1.0, ACCENT, 2, cv2.LINE_AA)

        # instruction
        cv2.putText(c, session.instruction, (30, py + 80), F, 0.5, TEXT, 1, cv2.LINE_AA)

        # phase status
        sy = py + 120
        if session.phase == "settle":
            cv2.putText(c, "Get ready...", (30, sy), F, 0.65, DIM, 1, cv2.LINE_AA)
        else:
            cv2.putText(c, "RECORDING", (30, sy), F, 0.7, GREEN, 2, cv2.LINE_AA)

        # progress bar
        bx1, bx2 = 30, self.CW - 30
        by = sy + 18
        cv2.rectangle(c, (bx1, by), (bx2, by + 16), GRID, -1)
        fill = int((bx2 - bx1) * session.progress)
        if fill > 0:
            col = GREEN if session.phase == "record" else (80, 80, 80)
            cv2.rectangle(c, (bx1, by), (bx1 + fill, by + 16), col, -1)

        # right panel hint
        rx = self.CW + 20
        cv2.putText(c, "Calibration running...", (rx, self.HDR + 50), F, 0.45, DIM, 1, cv2.LINE_AA)
        cv2.putText(c, "Face the camera and",    (rx, self.HDR + 75), F, 0.45, DIM, 1, cv2.LINE_AA)
        cv2.putText(c, "follow instructions.",   (rx, self.HDR + 95), F, 0.45, DIM, 1, cv2.LINE_AA)

        # footer
        fy = self._th - self.FTR
        cv2.rectangle(c, (0, fy), (self._tw, self._th), PANEL, -1)
        cv2.putText(c, "Calibrating...  Q quit", (12, fy + 18), F, 0.38, DIM, 1, cv2.LINE_AA)

        cv2.imshow(self.WIN, c)

    def close(self):
        cv2.destroyWindow(self.WIN)

    def _header(self, c, fps, tracking):
        cv2.rectangle(c, (0, 0), (self._tw, self.HDR), PANEL, -1)
        cv2.putText(c, "OSF Lite", (12, 24), F, 0.6, ACCENT, 1, cv2.LINE_AA)
        cv2.putText(c, f"{fps:.0f} fps", (110, 24), F, 0.5, FPS_C, 1, cv2.LINE_AA)
        cv2.circle(c, (200, 18), 5, GREEN if tracking else (60, 60, 200), -1, cv2.LINE_AA)
        cv2.putText(c, "tracking" if tracking else "searching",
                    (212, 24), F, 0.45, DIM, 1, cv2.LINE_AA)
        cv2.putText(c, "ARKit Shapes", (self.CW + self.BL, 24), F, 0.5, DIM, 1, cv2.LINE_AA)

    def _cam(self, c, frame, face):
        oy = self.HDR
        if frame is None:
            return
        h, w = frame.shape[:2]
        sc = min(self.CW / w, self.CH / h)
        nw, nh = int(w * sc), int(h * sc)
        ox = (self.CW - nw) // 2
        c[oy:oy+nh, ox:ox+nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        if face is None or face.lms is None:
            ov = c[oy:oy+nh, ox:ox+nw].copy()
            cv2.rectangle(ov, (0, 0), (nw, nh), (0, 0, 0), -1)
            cv2.addWeighted(ov, 0.35, c[oy:oy+nh, ox:ox+nw], 0.65, 0, c[oy:oy+nh, ox:ox+nw])
            cv2.putText(c, "no face", (ox + nw//2 - 30, oy + nh//2), F, 0.6, NOFACE, 1, cv2.LINE_AA)
            return
        for i, (ly, lx, lc) in enumerate(face.lms[:66]):
            if lc < 0.1: continue
            col = LM_EYE if 36 <= i <= 47 else LM_C
            r   = 2      if 36 <= i <= 47 else 1
            cv2.circle(c, (int(lx*sc)+ox, int(ly*sc)+oy), r, col, -1, cv2.LINE_AA)
        if face.bbox is not None:
            cv2.rectangle(c,
                (int(face.bbox[0]*sc)+ox, int(face.bbox[1]*sc)+oy),
                (int(face.bbox[2]*sc)+ox, int(face.bbox[3]*sc)+oy),
                ACCENT, 1, cv2.LINE_AA)
        cv2.putText(c, f"conf {face.conf:.2f}", (ox+4, oy+nh-6), F, 0.4, DIM, 1, cv2.LINE_AA)

    def _bars(self, c, shapes):
        px  = self.CW
        oy  = self.HDR + 8
        lx  = px + self.BL
        rh  = self.BH + self.BP
        cv2.line(c, (px, self.HDR), (px, self._th - self.FTR), GRID, 1)
        for row, (idx, name) in enumerate(SHAPES):
            y  = oy + row * rh
            bx = lx + 88
            cv2.putText(c, name, (lx, y + self.BH - 1), FM, 0.75, DIM, 1, cv2.LINE_AA)
            cv2.rectangle(c, (bx, y), (bx + self.BM, y + self.BH), GRID, -1)
            if shapes is not None and idx < len(shapes):
                v = max(0.0, min(1.0, shapes[idx]))
                f = int(v * self.BM)
                if f > 0:
                    cv2.rectangle(c, (bx, y), (bx + f, y + self.BH), _bc(idx), -1)
                cv2.putText(c, f"{v:.2f}", (bx + self.BM + 4, y + self.BH - 1),
                            FM, 0.75, TEXT, 1, cv2.LINE_AA)
            else:
                cv2.putText(c, "----", (bx + self.BM + 4, y + self.BH - 1),
                            FM, 0.75, DIM, 1, cv2.LINE_AA)

    def _footer(self, c):
        fy = self._th - self.FTR
        cv2.rectangle(c, (0, fy), (self._tw, self._th), PANEL, -1)
        cv2.putText(c, "Q  quit    |  ROFL Production", (12, fy + 18), F, 0.38, DIM, 1, cv2.LINE_AA)
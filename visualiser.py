"""
visualiser.py
Single-window visualiser for osf_lite.
Left panel  : webcam feed with facial landmark overlay
Right panel : live ARKit blendshape bars (32 tracked shapes)

Usage: facetracker_lite.py -c 0 --visualize 1
"""

import cv2
import numpy as np

SHAPES = [
    (0,  "EyeBlinkL"),
    (7,  "EyeBlinkR"),
    (6,  "EyeWideL"),
    (13, "EyeWideR"),
    (5,  "EyeSquintL"),
    (12, "EyeSquintR"),
    (1,  "EyeLookDnL"),
    (4,  "EyeLookUpL"),
    (2,  "EyeLookInL"),
    (3,  "EyeLookOutL"),
    (8,  "EyeLookDnR"),
    (11, "EyeLookUpR"),
    (9,  "EyeLookInR"),
    (10, "EyeLookOutR"),
    (41, "BrowDnL"),
    (42, "BrowDnR"),
    (43, "BrowInnerUp"),
    (44, "BrowOuterUpL"),
    (45, "BrowOuterUpR"),
    (17, "JawOpen"),
    (18, "MouthClose"),
    (19, "MouthFunnel"),
    (20, "MouthPucker"),
    (21, "MouthLeft"),
    (22, "MouthRight"),
    (23, "SmileLeft"),
    (24, "SmileRight"),
    (25, "FrownLeft"),
    (26, "FrownRight"),
    (27, "DimpleLeft"),
    (28, "DimpleRight"),
    (29, "StretchLeft"),
]

COL_BG        = (18,  16,  12)
COL_PANEL     = (28,  26,  22)
COL_ACCENT    = (44, 124, 245)
COL_BAR_EYE   = (60, 180, 100)
COL_BAR_BROW  = (60, 160, 220)
COL_BAR_MOUTH = (80, 100, 240)
COL_TEXT      = (200, 200, 200)
COL_TEXT_DIM  = (100, 100, 100)
COL_LM        = (44, 124, 245)
COL_LM_EYE   = (60, 220, 160)
COL_GRID      = (40,  38,  34)
COL_FPS       = (80, 200, 120)
COL_NO_FACE   = (60,  60, 180)

FONT      = cv2.FONT_HERSHEY_SIMPLEX
FONT_MONO = cv2.FONT_HERSHEY_PLAIN

def _bar_colour(idx):
    if idx <= 13: return COL_BAR_EYE
    if idx <= 18: return COL_BAR_BROW
    return COL_BAR_MOUTH


class Visualiser:
    WIN_NAME = "OSF Lite"
    CAM_W    = 640
    CAM_H    = 360
    BAR_W    = 360
    BAR_H    = 8
    BAR_PAD  = 2
    BAR_LEFT = 12
    BAR_MAX  = 200
    HEADER_H = 28
    FOOTER_H = 20

    def __init__(self):
        self._total_w = self.CAM_W + self.BAR_W
        self._total_h = self.CAM_H + self.HEADER_H + self.FOOTER_H
        self._canvas  = np.zeros((self._total_h, self._total_w, 3), dtype=np.uint8)
        cv2.namedWindow(self.WIN_NAME, cv2.WINDOW_AUTOSIZE)

    def update(self, frame, face, shapes, fps):
        c = self._canvas
        c[:] = COL_BG
        self._draw_header(c, fps, face is not None)
        self._draw_cam(c, frame, face)
        self._draw_bars(c, shapes)
        self._draw_footer(c)
        cv2.imshow(self.WIN_NAME, c)
        key = cv2.waitKey(1) & 0xFF
        return key == ord('q') or key == 27 or cv2.getWindowProperty(
            self.WIN_NAME, cv2.WND_PROP_VISIBLE) < 1

    def close(self):
        cv2.destroyWindow(self.WIN_NAME)

    def _draw_header(self, c, fps, tracking):
        cv2.rectangle(c, (0, 0), (self._total_w, self.HEADER_H), COL_PANEL, -1)
        cv2.putText(c, "OSF Lite", (12, 24), FONT, 0.6, COL_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(c, f"{fps:.0f} fps", (110, 24), FONT, 0.5, COL_FPS, 1, cv2.LINE_AA)
        dot_col = (60, 200, 80) if tracking else (60, 60, 200)
        cv2.circle(c, (200, 18), 5, dot_col, -1, cv2.LINE_AA)
        cv2.putText(c, "tracking" if tracking else "searching",
                    (212, 24), FONT, 0.45, COL_TEXT_DIM, 1, cv2.LINE_AA)
        cv2.putText(c, "ARKit Shapes", (self.CAM_W + self.BAR_LEFT, 24),
                    FONT, 0.5, COL_TEXT_DIM, 1, cv2.LINE_AA)

    def _draw_cam(self, c, frame, face):
        oy = self.HEADER_H
        if frame is None:
            return
        h, w = frame.shape[:2]
        scale = min(self.CAM_W / w, self.CAM_H / h)
        nw, nh = int(w * scale), int(h * scale)
        ox = (self.CAM_W - nw) // 2
        resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        c[oy:oy+nh, ox:ox+nw] = resized

        if face is None or face.lms is None:
            overlay = c[oy:oy+nh, ox:ox+nw].copy()
            cv2.rectangle(overlay, (0,0), (nw, nh), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.35, c[oy:oy+nh, ox:ox+nw], 0.65, 0,
                            c[oy:oy+nh, ox:ox+nw])
            cv2.putText(c, "no face", (ox + nw//2 - 30, oy + nh//2),
                        FONT, 0.6, COL_NO_FACE, 1, cv2.LINE_AA)
            return

        for i, (ly, lx, lc) in enumerate(face.lms[:66]):
            if lc < 0.1:
                continue
            px = int(lx * scale) + ox
            py = int(ly * scale) + oy
            col = COL_LM_EYE if 36 <= i <= 47 else COL_LM
            r   = 2 if 36 <= i <= 47 else 1
            cv2.circle(c, (px, py), r, col, -1, cv2.LINE_AA)

        if face.bbox is not None:
            bx1 = int(face.bbox[0] * scale) + ox
            by1 = int(face.bbox[1] * scale) + oy
            bx2 = int(face.bbox[2] * scale) + ox
            by2 = int(face.bbox[3] * scale) + oy
            cv2.rectangle(c, (bx1, by1), (bx2, by2), COL_ACCENT, 1, cv2.LINE_AA)

        cv2.putText(c, f"conf {face.conf:.2f}", (ox + 4, oy + nh - 6),
                    FONT, 0.4, COL_TEXT_DIM, 1, cv2.LINE_AA)

    def _draw_bars(self, c, shapes):
        panel_x = self.CAM_W
        oy      = self.HEADER_H + 8
        lx      = panel_x + self.BAR_LEFT
        row_h   = self.BAR_H + self.BAR_PAD
        cv2.line(c, (panel_x, self.HEADER_H),
                 (panel_x, self._total_h - self.FOOTER_H), COL_GRID, 1)
        for row, (idx, name) in enumerate(SHAPES):
            y  = oy + row * row_h
            bx = lx + 88
            cv2.putText(c, name, (lx, y + self.BAR_H - 1),
                        FONT_MONO, 0.75, COL_TEXT_DIM, 1, cv2.LINE_AA)
            cv2.rectangle(c, (bx, y), (bx + self.BAR_MAX, y + self.BAR_H),
                          COL_GRID, -1)
            if shapes is not None and idx < len(shapes):
                v    = max(0.0, min(1.0, shapes[idx]))
                fill = int(v * self.BAR_MAX)
                if fill > 0:
                    cv2.rectangle(c, (bx, y), (bx + fill, y + self.BAR_H),
                                  _bar_colour(idx), -1)
                cv2.putText(c, f"{v:.2f}",
                            (bx + self.BAR_MAX + 4, y + self.BAR_H - 1),
                            FONT_MONO, 0.75, COL_TEXT, 1, cv2.LINE_AA)
            else:
                cv2.putText(c, "----",
                            (bx + self.BAR_MAX + 4, y + self.BAR_H - 1),
                            FONT_MONO, 0.75, COL_TEXT_DIM, 1, cv2.LINE_AA)

    def _draw_footer(self, c):
        fy = self._total_h - self.FOOTER_H
        cv2.rectangle(c, (0, fy), (self._total_w, self._total_h), COL_PANEL, -1)
        cv2.putText(c, "Q  quit    |  ROFL Production",
                    (12, fy + 18), FONT, 0.38, COL_TEXT_DIM, 1, cv2.LINE_AA)
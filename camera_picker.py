"""
camera_picker.py
Startup camera selection window.
Probes available cameras, shows a click-to-select list,
returns the chosen index.  Pure OpenCV — no tkinter/Qt dependency.
"""

import cv2
import numpy as np

FONT       = cv2.FONT_HERSHEY_SIMPLEX
COL_BG     = (18,  16,  12)
COL_PANEL  = (28,  26,  22)
COL_ACCENT = (44, 124, 245)
COL_TEXT   = (200, 200, 200)
COL_DIM    = (110, 110, 110)
COL_HOVER  = (55,  55,  50)
COL_SEL    = (30,  70, 160)
COL_BTN    = (44, 124, 245)
COL_BTN_TXT= (240, 240, 240)

WIN   = "OSF Lite — Select Camera"
W, H  = 480, 360
ROW_H = 48
PAD   = 20
BTN_H = 40


def _probe_cameras(max_index=8):
    """Try opening camera indices 0..max_index, return list of (index, label)."""
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if hasattr(cv2, 'CAP_DSHOW') else 0)
        if cap is not None and cap.isOpened():
            # try to get a backend name
            name = f"Camera {i}"
            found.append((i, name))
            cap.release()
    return found


def _draw(canvas, cameras, hovered, selected, status):
    canvas[:] = COL_BG

    # header
    cv2.rectangle(canvas, (0, 0), (W, 52), COL_PANEL, -1)
    cv2.putText(canvas, "OSF Lite", (PAD, 30), FONT, 0.65, COL_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Select a camera to begin", (PAD, 46),
                FONT, 0.38, COL_DIM, 1, cv2.LINE_AA)

    if not cameras:
        cv2.putText(canvas, "No cameras found.", (PAD, 130),
                    FONT, 0.55, (80, 80, 200), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Check connections and restart.",
                    (PAD, 160), FONT, 0.45, COL_DIM, 1, cv2.LINE_AA)
        return

    list_y = 62
    for i, (idx, name) in enumerate(cameras):
        y0 = list_y + i * ROW_H
        y1 = y0 + ROW_H - 2

        # row background
        if selected == i:
            bg = COL_SEL
        elif hovered == i:
            bg = COL_HOVER
        else:
            bg = COL_PANEL
        cv2.rectangle(canvas, (PAD, y0), (W - PAD, y1), bg, -1)

        # camera icon (simple rect)
        ic_x, ic_y = PAD + 10, y0 + 10
        cv2.rectangle(canvas, (ic_x, ic_y), (ic_x + 22, ic_y + 16),
                      COL_ACCENT if selected == i else COL_DIM, 1)
        cv2.rectangle(canvas, (ic_x + 22, ic_y + 5), (ic_x + 28, ic_y + 11),
                      COL_ACCENT if selected == i else COL_DIM, -1)

        # index badge
        badge_col = COL_ACCENT if selected == i else (60, 60, 60)
        cv2.rectangle(canvas, (ic_x + 36, y0 + 10),
                      (ic_x + 56, y0 + 28), badge_col, -1)
        cv2.putText(canvas, str(idx), (ic_x + 41, y0 + 26),
                    FONT, 0.5, COL_BTN_TXT, 1, cv2.LINE_AA)

        # name
        cv2.putText(canvas, name, (ic_x + 64, y0 + 26),
                    FONT, 0.52, COL_TEXT if selected == i else COL_DIM,
                    1, cv2.LINE_AA)

        # selected tick
        if selected == i:
            cv2.putText(canvas, "selected", (W - PAD - 70, y0 + 26),
                        FONT, 0.38, COL_ACCENT, 1, cv2.LINE_AA)

        # divider
        cv2.line(canvas, (PAD, y1 + 1), (W - PAD, y1 + 1), (35, 33, 30), 1)

    # Start button
    btn_y = H - BTN_H - PAD
    btn_col = COL_BTN if selected is not None else (50, 50, 50)
    cv2.rectangle(canvas, (PAD, btn_y), (W - PAD, btn_y + BTN_H), btn_col, -1)
    label = "Start Tracking" if selected is not None else "Select a camera"
    tw, _ = cv2.getTextSize(label, FONT, 0.6, 1)[0], None
    cv2.putText(canvas, label,
                ((W - tw[0]) // 2, btn_y + 26),
                FONT, 0.6, COL_BTN_TXT, 1, cv2.LINE_AA)

    # status line
    if status:
        cv2.putText(canvas, status, (PAD, H - 6),
                    FONT, 0.36, COL_DIM, 1, cv2.LINE_AA)


def pick_camera(default_index=None):
    """
    Show the camera picker window.
    Returns the selected camera index (int), or default_index if the user
    closes without selecting, or 0 if default_index is None.
    """
    status_msg = "Scanning for cameras..."
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)

    # show scanning message first
    canvas[:] = COL_BG
    cv2.rectangle(canvas, (0, 0), (W, 52), COL_PANEL, -1)
    cv2.putText(canvas, "OSF Lite", (PAD, 30), FONT, 0.65, COL_ACCENT, 1, cv2.LINE_AA)
    cv2.putText(canvas, "Scanning for cameras...", (PAD, 130),
                FONT, 0.55, COL_DIM, 1, cv2.LINE_AA)
    cv2.imshow(WIN, canvas)
    cv2.waitKey(1)

    cameras = _probe_cameras()
    if not cameras:
        status_msg = "No cameras found."

    # pre-select default if provided
    selected = None
    if default_index is not None:
        for i, (idx, _) in enumerate(cameras):
            if idx == default_index:
                selected = i
                break

    hovered  = None
    result   = default_index if default_index is not None else (cameras[0][0] if cameras else 0)

    def _on_mouse(event, x, y, flags, param):
        nonlocal hovered, selected, result
        list_y = 62

        # check rows
        for i, (idx, _) in enumerate(cameras):
            y0 = list_y + i * ROW_H
            y1 = y0 + ROW_H - 2
            if PAD <= x <= W - PAD and y0 <= y <= y1:
                if event == cv2.EVENT_MOUSEMOVE:
                    hovered = i
                elif event == cv2.EVENT_LBUTTONDOWN:
                    selected = i
                    result   = cameras[i][0]
                return
        hovered = None

        # check start button
        btn_y = H - BTN_H - PAD
        if PAD <= x <= W - PAD and btn_y <= y <= btn_y + BTN_H:
            if event == cv2.EVENT_LBUTTONDOWN and selected is not None:
                param['done'] = True

    state = {'done': False}
    cv2.setMouseCallback(WIN, _on_mouse, state)

    status_msg = f"Found {len(cameras)} camera(s)." if cameras else "No cameras found."

    while True:
        _draw(canvas, cameras, hovered, selected, status_msg)
        cv2.imshow(WIN, canvas)
        key = cv2.waitKey(16) & 0xFF

        if state['done']:
            break
        if key == 13 and selected is not None:   # Enter
            break
        if key == 27:                            # Escape — use default
            break
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
            break

        # keyboard up/down
        if cameras:
            if key == 82 or key == ord('k'):    # up
                selected = max(0, (selected or 0) - 1)
                result   = cameras[selected][0]
            elif key == 84 or key == ord('j'):  # down
                selected = min(len(cameras)-1, (selected or 0) + 1)
                result   = cameras[selected][0]

    cv2.destroyWindow(WIN)
    return result
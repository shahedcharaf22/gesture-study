import cv2
import numpy as np

def draw_stroke(
    canvas,
    highlighter_canvas,
    previous_point,
    current_point,
    tool,
):
    if tool == "pen":
        cv2.line(
            canvas,
            previous_point,
            current_point,
            (0, 0, 255),
            5,
            cv2.LINE_AA,
        )

    elif tool == "highlighter":
        cv2.line(
            highlighter_canvas,
            previous_point,
            current_point,
            (0, 255, 255),
            26,
            cv2.LINE_AA,
        )

    elif tool == "eraser":
        cv2.line(
            canvas,
            previous_point,
            current_point,
            (0, 0, 0),
            40,
            cv2.LINE_AA,
        )

        cv2.line(
            highlighter_canvas,
            previous_point,
            current_point,
            (0, 0, 0),
            40,
            cv2.LINE_AA,
        )
        
def create_canvases(frame):
    canvas = np.zeros_like(
        frame
    )

    highlighter_canvas = np.zeros_like(
        frame
    )

    return (
        canvas,
        highlighter_canvas,
    )

def clear_canvases(frame):
    return create_canvases(frame)

def apply_highlighter(
    display_frame,
    highlighter_canvas,
):
    highlight_mask = np.any(
        highlighter_canvas != 0,
        axis=2,
    )

    if np.any(
        highlight_mask
    ):
        highlight_overlay = (
            display_frame.copy()
        )

        highlight_overlay[
            highlight_mask
        ] = highlighter_canvas[
            highlight_mask
        ]

        display_frame = cv2.addWeighted(
            highlight_overlay,
            0.35,
            display_frame,
            0.65,
            0,
        )

    return display_frame

def draw_cursor(
    frame,
    cursor_point,
    tool,
):
    if cursor_point is None:
        return

    if tool == "pen":
        cursor_color = (
            0,
            0,
            255,
        )

    elif tool == "highlighter":
        cursor_color = (
            0,
            255,
            255,
        )

    else:
        cursor_color = (
            255,
            255,
            255,
        )

    cv2.circle(
        frame,
        cursor_point,
        10,
        cursor_color,
        2,
        cv2.LINE_AA,
    )

def calculate_safe_zone(
    frame,
    margin_ratio,
):
    frame_height, frame_width = (
        frame.shape[:2]
    )

    safe_left = int(
        frame_width * margin_ratio
    )

    safe_right = int(
        frame_width
        * (1 - margin_ratio)
    )

    safe_top = int(
        frame_height * margin_ratio
    )

    safe_bottom = int(
        frame_height
        * (1 - margin_ratio)
    )

    return (
        safe_left,
        safe_right,
        safe_top,
        safe_bottom,
    )

def is_inside_safe_zone(
    point,
    safe_zone,
):
    if point is None:
        return False

    (
        safe_left,
        safe_right,
        safe_top,
        safe_bottom,
    ) = safe_zone

    x, y = point

    return (
        safe_left
        <= x
        <= safe_right
        and
        safe_top
        <= y
        <= safe_bottom
    )
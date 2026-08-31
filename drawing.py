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
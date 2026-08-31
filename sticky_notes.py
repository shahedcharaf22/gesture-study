import cv2
import textwrap


NOTE_WIDTH = 320
NOTE_HEIGHT = 170


NOTE_COLORS = {
    "yellow": (100, 230, 255),
    "pink": (220, 190, 255),
    "blue": (255, 220, 170),
    "green": (190, 255, 190),
}

# =========================================================
# DRAW A STICKY NOTE
# =========================================================

def draw_sticky_note(
    image,
    note,
    selected=False,
):
    x = note["x"]
    y = note["y"]

    text = note["text"]
    color = note["color"]

    pinned = note.get("pinned", False)

    height, width = image.shape[:2]

    # Keep note inside the screen
    x = max(
        0,
        min(
            x,
            width - NOTE_WIDTH,
        ),
    )

    y = max(
        0,
        min(
            y,
            height - NOTE_HEIGHT,
        ),
    )

    note["x"] = x
    note["y"] = y

    # Sticky note background
    cv2.rectangle(
        image,
        (x, y),
        (
            x + NOTE_WIDTH,
            y + NOTE_HEIGHT,
        ),
        color,
        -1,
    )

    # Selected notes get a thicker border
    if selected:
        border_color = (0, 140, 255)
        border_thickness = 4

    else:
        border_color = (60, 60, 60)
        border_thickness = 2

    cv2.rectangle(
        image,
        (x, y),
        (
            x + NOTE_WIDTH,
            y + NOTE_HEIGHT,
        ),
        border_color,
        border_thickness,
    )

    # Show whether note is pinned
    if pinned:
        status = "[PINNED]"

    else:
        status = "[UNPINNED]"

    cv2.putText(
        image,
        status,
        (x + 10, y + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (50, 50, 50),
        1,
        cv2.LINE_AA,
    )

    # Wrap long text
    lines = textwrap.wrap(
        text,
        width=34,
    )

    for line_number, line in enumerate(lines[:5]):
        cv2.putText(
            image,
            line,
            (
                x + 12,
                y + 55 + line_number * 24,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

def find_note_at_position(
    sticky_notes,
    mouse_x,
    mouse_y,
):
    # Go backwards so the top-most note is selected first
    for index in range(
        len(sticky_notes) - 1,
        -1,
        -1,
    ):
        note = sticky_notes[index]

        x = note["x"]
        y = note["y"]

        inside_x = (
            x
            <= mouse_x
            <= x + NOTE_WIDTH
        )

        inside_y = (
            y
            <= mouse_y
            <= y + NOTE_HEIGHT
        )

        if inside_x and inside_y:
            return index

    return None

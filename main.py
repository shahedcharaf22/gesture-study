import cv2
import mediapipe as mp
import time
import math
import textwrap
import numpy as np


# =========================================================
# SETTINGS
# =========================================================

WINDOW_NAME = "Gesture Study Workspace"

NOTE_WIDTH = 320
NOTE_HEIGHT = 170

SMOOTHING_ALPHA = 0.45

DETECTION_WIDTH = 640
DETECTION_HEIGHT = 360


# =========================================================
# STICKY NOTE COLORS
# OpenCV uses BGR instead of RGB
# =========================================================

NOTE_COLORS = {
    "yellow": (100, 230, 255),
    "pink": (220, 190, 255),
    "blue": (255, 220, 170),
    "green": (190, 255, 190),
}


# =========================================================
# PROGRAM STATE
# =========================================================

pinching = False
drawing_mode = False

canvas = None
highlighter_canvas = None

previous_point = None

tool = "pen"

show_skeleton = True


# Smooth cursor position
smoothed_x = None
smoothed_y = None

cursor_point = None


# =========================================================
# STICKY NOTE STATE
# =========================================================

sticky_notes = []

note_typing = False
note_text = ""
note_position = None

# None means we are creating a new note.
# A number means we are editing an existing note.
editing_note_index = None

selected_note_index = None

note_color_name = "yellow"


# =========================================================
# DRAGGING STATE
# =========================================================

dragging_note = False

drag_offset_x = 0
drag_offset_y = 0


# Current window dimensions
frame_width = 0
frame_height = 0


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


# =========================================================
# FIND WHICH NOTE WAS CLICKED
# =========================================================

def find_note_at_position(mouse_x, mouse_y):
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


# =========================================================
# MOUSE / TRACKPAD CALLBACK
# =========================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param,
):
    global selected_note_index
    global dragging_note
    global drag_offset_x
    global drag_offset_y

    global note_typing
    global note_text
    global note_position
    global editing_note_index

    global drawing_mode
    global previous_point

    global frame_width
    global frame_height


    # -----------------------------------------------------
    # DOUBLE CLICK = EDIT NOTE
    # -----------------------------------------------------

    if event == cv2.EVENT_LBUTTONDBLCLK:
        note_index = find_note_at_position(
            x,
            y,
        )

        if note_index is not None:
            selected_note_index = note_index

            note = sticky_notes[
                note_index
            ]

            # Stop drawing while editing
            drawing_mode = False
            previous_point = None

            note_typing = True

            note_text = note[
                "text"
            ]

            note_position = (
                note["x"],
                note["y"],
            )

            editing_note_index = (
                note_index
            )

            dragging_note = False

            print(
                "Editing sticky note"
            )


    # -----------------------------------------------------
    # LEFT CLICK = SELECT NOTE
    # -----------------------------------------------------

    elif event == cv2.EVENT_LBUTTONDOWN:
        note_index = find_note_at_position(
            x,
            y,
        )

        # Clicked empty space
        if note_index is None:
            selected_note_index = None

            dragging_note = False

            return

        selected_note_index = note_index

        note = sticky_notes[
            note_index
        ]

        # Stop air drawing while manipulating a note
        drawing_mode = False
        previous_point = None

        # Only unpinned notes can move
        if not note.get(
            "pinned",
            False,
        ):
            dragging_note = True

            drag_offset_x = (
                x - note["x"]
            )

            drag_offset_y = (
                y - note["y"]
            )

        else:
            dragging_note = False


    # -----------------------------------------------------
    # MOVE MOUSE WHILE HOLDING = DRAG NOTE
    # -----------------------------------------------------

    elif event == cv2.EVENT_MOUSEMOVE:
        if (
            dragging_note
            and selected_note_index
            is not None
        ):
            note = sticky_notes[
                selected_note_index
            ]

            if not note.get(
                "pinned",
                False,
            ):
                new_x = (
                    x - drag_offset_x
                )

                new_y = (
                    y - drag_offset_y
                )

                # Keep note inside window
                if frame_width > 0:
                    new_x = max(
                        0,
                        min(
                            new_x,
                            frame_width
                            - NOTE_WIDTH,
                        ),
                    )

                if frame_height > 0:
                    new_y = max(
                        0,
                        min(
                            new_y,
                            frame_height
                            - NOTE_HEIGHT,
                        ),
                    )

                note["x"] = new_x
                note["y"] = new_y


    # -----------------------------------------------------
    # RELEASE MOUSE = DROP NOTE
    # -----------------------------------------------------

    elif event == cv2.EVENT_LBUTTONUP:
        dragging_note = False


# =========================================================
# MEDIAPIPE SETUP
# =========================================================

base_options = mp.tasks.BaseOptions(
    model_asset_path="hand_landmarker.task",
    delegate=mp.tasks.BaseOptions.Delegate.CPU,
)

options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp.tasks.vision.RunningMode.VIDEO,
    num_hands=1,
)

landmarker = (
    mp.tasks.vision.HandLandmarker.create_from_options(
        options
    )
)


# =========================================================
# CAMERA SETUP
# =========================================================

camera = cv2.VideoCapture(0)

print(
    "Camera opened:",
    camera.isOpened(),
)

time.sleep(2)


# =========================================================
# CREATE WINDOW + MOUSE CALLBACK
# =========================================================

cv2.namedWindow(
    WINDOW_NAME
)

cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback,
)


# =========================================================
# MAIN LOOP
# =========================================================

while True:
    success, frame = camera.read()

    if not success:
        print(
            "Failed to capture frame from webcam. Exiting..."
        )

        break


    # Mirror webcam
    frame = cv2.flip(
        frame,
        1,
    )

    frame_height, frame_width = (
        frame.shape[:2]
    )


    # =====================================================
    # CREATE DRAWING CANVASES
    # =====================================================

    if canvas is None:
        canvas = np.zeros_like(
            frame
        )

    if highlighter_canvas is None:
        highlighter_canvas = (
            np.zeros_like(frame)
        )


    # =====================================================
    # SMALLER FRAME FOR MEDIAPIPE
    # =====================================================

    detection_frame = cv2.resize(
        frame,
        (
            DETECTION_WIDTH,
            DETECTION_HEIGHT,
        ),
    )

    rgb_frame = cv2.cvtColor(
        detection_frame,
        cv2.COLOR_BGR2RGB,
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame,
    )


    timestamp_ms = int(
        time.monotonic() * 1000
    )


    result = (
        landmarker.detect_for_video(
            mp_image,
            timestamp_ms,
        )
    )


    cursor_point = None


    # =====================================================
    # HAND DETECTED
    # =====================================================

    if result.hand_landmarks:
        hand = (
            result.hand_landmarks[0]
        )

        connections = (
            mp.tasks.vision
            .HandLandmarksConnections
            .HAND_CONNECTIONS
        )


        # Fingertips
        thumb_tip = hand[4]
        index_tip = hand[8]


        # =================================================
        # CONVERT NORMALIZED COORDINATES TO PIXELS
        # =================================================

        thumb_x = int(
            thumb_tip.x
            * frame_width
        )

        thumb_y = int(
            thumb_tip.y
            * frame_height
        )

        index_x = int(
            index_tip.x
            * frame_width
        )

        index_y = int(
            index_tip.y
            * frame_height
        )


        # =================================================
        # SMOOTH CURSOR POSITION
        # =================================================

        if (
            smoothed_x is None
            or smoothed_y is None
        ):
            smoothed_x = index_x
            smoothed_y = index_y

        else:
            smoothed_x = int(
                SMOOTHING_ALPHA
                * index_x
                + (
                    1
                    - SMOOTHING_ALPHA
                )
                * smoothed_x
            )

            smoothed_y = int(
                SMOOTHING_ALPHA
                * index_y
                + (
                    1
                    - SMOOTHING_ALPHA
                )
                * smoothed_y
            )


        cursor_point = (
            smoothed_x,
            smoothed_y,
        )


        # =================================================
        # PINCH DISTANCE
        # =================================================

        distance = math.hypot(
            index_x - thumb_x,
            index_y - thumb_y,
        )


        # New pinch
        if (
            distance < 40
            and not pinching
        ):
            pinching = True

            drawing_mode = (
                not drawing_mode
            )

            previous_point = None

            if drawing_mode:
                print(
                    "Drawing mode ON"
                )

            else:
                print(
                    "Drawing mode OFF"
                )


        # Pinch released
        elif (
            distance > 65
            and pinching
        ):
            pinching = False


        # =================================================
        # DRAWING
        # =================================================

        if (
            drawing_mode
            and not pinching
            and not note_typing
        ):
            current_point = (
                cursor_point
            )

            if (
                previous_point
                is not None
            ):

                # PEN
                if tool == "pen":
                    cv2.line(
                        canvas,
                        previous_point,
                        current_point,
                        (0, 0, 255),
                        5,
                        cv2.LINE_AA,
                    )


                # HIGHLIGHTER
                elif (
                    tool
                    == "highlighter"
                ):
                    cv2.line(
                        highlighter_canvas,
                        previous_point,
                        current_point,
                        (0, 255, 255),
                        26,
                        cv2.LINE_AA,
                    )


                # ERASER
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


            previous_point = (
                current_point
            )

        else:
            previous_point = None


        # =================================================
        # HAND SKELETON
        # =================================================

        if show_skeleton:

            for index, landmark in enumerate(
                hand
            ):
                x = int(
                    landmark.x
                    * frame_width
                )

                y = int(
                    landmark.y
                    * frame_height
                )

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1,
                )

                cv2.putText(
                    frame,
                    str(index),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )


            for connection in connections:
                start_landmark = hand[
                    connection.start
                ]

                end_landmark = hand[
                    connection.end
                ]

                x1 = int(
                    start_landmark.x
                    * frame_width
                )

                y1 = int(
                    start_landmark.y
                    * frame_height
                )

                x2 = int(
                    end_landmark.x
                    * frame_width
                )

                y2 = int(
                    end_landmark.y
                    * frame_height
                )

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )


        # =================================================
        # CURSOR
        # =================================================

        if cursor_point is not None:

            if tool == "pen":
                cursor_color = (
                    0,
                    0,
                    255,
                )

            elif (
                tool
                == "highlighter"
            ):
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


    # =====================================================
    # NO HAND
    # =====================================================

    else:
        pinching = False

        previous_point = None

        smoothed_x = None
        smoothed_y = None


    # =====================================================
    # STATUS TEXT
    # =====================================================

    if drawing_mode:
        status_text = (
            f"DRAW ON | "
            f"{tool.upper()}"
        )

    else:
        status_text = (
            f"DRAW OFF | "
            f"{tool.upper()}"
        )


    cv2.putText(
        frame,
        status_text,
        (40, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


    # =====================================================
    # COMBINE CAMERA + PEN
    # =====================================================

    display_frame = cv2.add(
        frame,
        canvas,
    )


    # =====================================================
    # TRANSPARENT HIGHLIGHTER
    # =====================================================

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

        display_frame = (
            cv2.addWeighted(
                highlight_overlay,
                0.35,
                display_frame,
                0.65,
                0,
            )
        )


    # =====================================================
    # DRAW STICKY NOTES
    # =====================================================

    for index, note in enumerate(
        sticky_notes
    ):
        selected = (
            index
            == selected_note_index
        )

        draw_sticky_note(
            display_frame,
            note,
            selected,
        )


    # =====================================================
    # NOTE PREVIEW WHILE TYPING
    # =====================================================

    if (
        note_typing
        and note_position
        is not None
    ):

        preview_text = note_text

        if preview_text == "":
            preview_text = (
                "Type your note..."
            )

        preview_note = {
            "x": note_position[0],
            "y": note_position[1],
            "text": preview_text,
            "color": NOTE_COLORS[
                note_color_name
            ],
            "pinned": False,
        }

        # If editing existing note,
        # use that note's color
        if (
            editing_note_index
            is not None
        ):
            preview_note[
                "color"
            ] = sticky_notes[
                editing_note_index
            ]["color"]

            preview_note[
                "pinned"
            ] = sticky_notes[
                editing_note_index
            ].get(
                "pinned",
                False,
            )

        draw_sticky_note(
            display_frame,
            preview_note,
            True,
        )

        cv2.putText(
            display_frame,
            (
                "NOTE MODE: "
                "ENTER = save | "
                "ESC = cancel"
            ),
            (40, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


    # =====================================================
    # SHOW WINDOW
    # =====================================================

    cv2.imshow(
        WINDOW_NAME,
        display_frame,
    )


    key = (
        cv2.waitKey(1)
        & 0xFF
    )


    # =====================================================
    # NOTE TYPING MODE
    # =====================================================

    if note_typing:

        # ENTER = save
        if key in (10, 13):

            cleaned_text = (
                note_text.strip()
            )

            if cleaned_text:

                # Editing existing note
                if (
                    editing_note_index
                    is not None
                ):
                    sticky_notes[
                        editing_note_index
                    ][
                        "text"
                    ] = cleaned_text

                    selected_note_index = (
                        editing_note_index
                    )

                    print(
                        "Sticky note updated"
                    )


                # Creating new note
                else:
                    sticky_notes.append(
                        {
                            "x": note_position[
                                0
                            ],
                            "y": note_position[
                                1
                            ],
                            "text": cleaned_text,
                            "color": NOTE_COLORS[
                                note_color_name
                            ],
                            "pinned": False,
                        }
                    )

                    selected_note_index = (
                        len(
                            sticky_notes
                        )
                        - 1
                    )

                    print(
                        "Sticky note saved"
                    )


            note_typing = False
            note_text = ""
            note_position = None

            editing_note_index = None


        # ESC = cancel
        elif key == 27:

            note_typing = False
            note_text = ""
            note_position = None

            editing_note_index = None

            print(
                "Sticky note cancelled"
            )


        # BACKSPACE
        elif key in (8, 127):

            note_text = (
                note_text[:-1]
            )


        # Normal characters
        elif 32 <= key <= 126:

            note_text += chr(
                key
            )


        continue


    # =====================================================
    # NORMAL KEYBOARD CONTROLS
    # =====================================================

    # Quit
    if key == ord("q"):
        break


    # Pen
    elif key == ord("p"):
        tool = "pen"

        previous_point = None

        print(
            "Pen selected"
        )


    # Eraser
    elif key == ord("e"):
        tool = "eraser"

        previous_point = None

        print(
            "Eraser selected"
        )


    # Highlighter
    elif key == ord("h"):
        tool = "highlighter"

        previous_point = None

        print(
            "Highlighter selected"
        )


    # Clear drawings
    elif key == ord("c"):

        canvas = np.zeros_like(
            frame
        )

        highlighter_canvas = (
            np.zeros_like(frame)
        )

        previous_point = None

        print(
            "Drawings cleared"
        )


    # Show/hide skeleton
    elif key == ord("s"):

        show_skeleton = (
            not show_skeleton
        )

        print(
            "Skeleton:",
            "ON"
            if show_skeleton
            else "OFF",
        )


    # =====================================================
    # CREATE NEW STICKY NOTE
    # =====================================================

    elif key == ord("n"):

        drawing_mode = False
        previous_point = None

        editing_note_index = None

        if cursor_point is not None:

            note_position = (
                cursor_point[0] + 15,
                cursor_point[1] + 15,
            )

        else:

            note_position = (
                frame_width // 2
                - NOTE_WIDTH // 2,
                frame_height // 2
                - NOTE_HEIGHT // 2,
            )


        note_text = ""

        note_typing = True

        selected_note_index = None

        print(
            "New sticky note"
        )


    # =====================================================
    # PIN / UNPIN SELECTED NOTE
    # =====================================================

    elif key == ord("l"):

        if (
            selected_note_index
            is not None
        ):
            note = sticky_notes[
                selected_note_index
            ]

            note[
                "pinned"
            ] = not note.get(
                "pinned",
                False,
            )

            dragging_note = False

            if note[
                "pinned"
            ]:
                print(
                    "Sticky note pinned"
                )

            else:
                print(
                    "Sticky note unpinned"
                )


    # =====================================================
    # DELETE SELECTED NOTE
    # =====================================================

    elif (
        key == ord("d")
        or key == 8
        or key == 127
    ):

        if (
            selected_note_index
            is not None
        ):
            sticky_notes.pop(
                selected_note_index
            )

            selected_note_index = None

            dragging_note = False

            print(
                "Sticky note deleted"
            )


    # =====================================================
    # STICKY NOTE COLORS
    # =====================================================

    elif key == ord("1"):

        note_color_name = "yellow"

        if (
            selected_note_index
            is not None
        ):
            sticky_notes[
                selected_note_index
            ]["color"] = (
                NOTE_COLORS["yellow"]
            )

        print(
            "Sticky note color: yellow"
        )


    elif key == ord("2"):

        note_color_name = "pink"

        if (
            selected_note_index
            is not None
        ):
            sticky_notes[
                selected_note_index
            ]["color"] = (
                NOTE_COLORS["pink"]
            )

        print(
            "Sticky note color: pink"
        )


    elif key == ord("3"):

        note_color_name = "blue"

        if (
            selected_note_index
            is not None
        ):
            sticky_notes[
                selected_note_index
            ]["color"] = (
                NOTE_COLORS["blue"]
            )

        print(
            "Sticky note color: blue"
        )


    elif key == ord("4"):

        note_color_name = "green"

        if (
            selected_note_index
            is not None
        ):
            sticky_notes[
                selected_note_index
            ]["color"] = (
                NOTE_COLORS["green"]
            )

        print(
            "Sticky note color: green"
        )


# =========================================================
# CLEANUP
# =========================================================

camera.release()

landmarker.close()

cv2.destroyAllWindows()
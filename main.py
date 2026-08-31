import cv2
import time
import math
import numpy as np

from sticky_notes import (
        NOTE_WIDTH,
        NOTE_HEIGHT,
        NOTE_COLORS,
        draw_sticky_note,
        find_note_at_position,
)

from hand_tracker import (
     create_hand_landmarker,
     detect_hand,
)

from drawing import (
    create_canvases,
    clear_canvases,
    draw_stroke,
    apply_highlighter,
)

# =========================================================
# SETTINGS
# =========================================================

WINDOW_NAME = "Gesture Study Workspace"

SMOOTHING_ALPHA = 0.45
SAFE_MARGIN_RATIO = 0.08

DETECTION_WIDTH = 640
DETECTION_HEIGHT = 360

# Number of missing-hand frames before resetting pinch state
PINCH_RESET_FRAMES = 8

# =========================================================
# PROGRAM STATE
# =========================================================

pinching = False
drawing_mode = False
pinch_armed = True

canvas = None
highlighter_canvas = None

previous_point = None

tool = "pen"

show_skeleton = True
show_safe_frame = True

# Smooth cursor position
smoothed_x = None
smoothed_y = None

cursor_point = None

# Count how long MediaPipe has not seen the hand
hand_missing_frames = 0

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
            sticky_notes,
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
            sticky_notes,
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

landmarker = create_hand_landmarker()

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
    
    safe_left = int(
        frame_width * SAFE_MARGIN_RATIO
    )

    safe_right = int(
        frame_width * (1 - SAFE_MARGIN_RATIO)
    )

    safe_top = int(
        frame_height * SAFE_MARGIN_RATIO
    )

    safe_bottom = int(
        frame_height * (1 - SAFE_MARGIN_RATIO)
    )


    # =====================================================
    # CREATE DRAWING CANVASES
    # =====================================================
    if (
        canvas is None
        or highlighter_canvas is None
    ):
        canvas, highlighter_canvas = (
            create_canvases(frame)
        )
        
    # =====================================================
    # DETECT HAND
    # =====================================================

    hand_data = detect_hand(
        landmarker,
        frame,
        DETECTION_WIDTH,
        DETECTION_HEIGHT,
    )

    cursor_point = None
    
    # =====================================================
    # HAND DETECTED
    # =====================================================

    if hand_data is not None:
        
        hand_missing_frames = 0

        hand, thumb_point, index_point, connections = (
            hand_data
        )

        thumb_x, thumb_y = thumb_point
        index_x, index_y = index_point
          
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
        
        inside_safe_zone = (
            safe_left
            <= cursor_point[0]
            <= safe_right
            and
            safe_top
            <= cursor_point[1]
            <= safe_bottom
        )

        # =================================================
        # PINCH DISTANCE
        # =================================================

        distance = math.hypot(
            index_x - thumb_x,
            index_y - thumb_y,
        )

        # Fingers are clearly separated
        if distance > 65:
           pinching = False
           pinch_armed = True

        # New pinch
        elif distance < 40 and pinch_armed:
            pinching = True
            pinch_armed = False

            drawing_mode = not drawing_mode
            previous_point = None

            if drawing_mode:
               print("Drawing mode ON")
            else:
               print("Drawing mode OFF")

        # =================================================
        # DRAWING
        # =================================================

        if (
            drawing_mode
            and not pinching
            and not note_typing
            and inside_safe_zone
        ):
            current_point = (
                cursor_point
            )

            if (
                previous_point is not None
            ):
            
                draw_stroke(
                    canvas,
                    highlighter_canvas,
                    previous_point,
                    current_point,
                    tool,
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

    # ========================s=============================
    # NO HAND
    # =====================================================

    else : 

        # Stop the current stroke so we do not draw
        # a giant line when the hand returns
        previous_point = None

        # Count how many frames the hand has been missing
        hand_missing_frames += 1

        # If the hand has been gone for long enough,
        # forget the previous pinch state
        if hand_missing_frames >= PINCH_RESET_FRAMES:
            pinching = False
            pinch_armed = False

        # Reset smoothing so the returning cursor
        # starts directly at the newly detected finger
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
    
    if show_safe_frame:
        cv2.rectangle(
            frame,
            (
                safe_left,
                safe_top,
            ),
            (
                safe_right,
                safe_bottom,
            ),
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

    # =====================================================
    # COMBINE CAMERA + PEN
    # =====================================================

    display_frame = cv2.add(
        frame,
        canvas,
    )
    
    display_frame = apply_highlighter(
        display_frame,
        highlighter_canvas,
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


    elif key == ord("c"):

        canvas, highlighter_canvas = (
            clear_canvases(frame)
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
        
    elif key == ord("b"):
        show_safe_frame = (
            not show_safe_frame
        )

        print(
            "Safe frame:",
            "ON"
            if show_safe_frame
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
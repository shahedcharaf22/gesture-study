import cv2
import time

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
    draw_hand_skeleton,
    smooth_cursor,
    get_palm_center,
)

from drawing import (
    create_canvases,
    clear_canvases,
    draw_stroke,
    apply_highlighter,
    draw_cursor,
    calculate_safe_zone,
    is_inside_safe_zone,
)

from gestures import (
    detect_pinch,
    update_palm_history,
    detect_swipe,
    is_open_palm,
)

# =========================================================
# SETTINGS
# =========================================================

WINDOW_NAME = "Gesture Study Workspace"

SMOOTHING_ALPHA = 0.45
SAFE_MARGIN_RATIO = 0.08

DETECTION_WIDTH = 640
DETECTION_HEIGHT = 360
PALM_HISTORY_LENGTH = 8
SWIPE_COOLDOWN_SECONDS = 0.7

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
smoothed_point = None

cursor_point = None
palm_history = []
last_swipe_time = 0.0

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

    safe_zone = calculate_safe_zone(
        frame,
        SAFE_MARGIN_RATIO,
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
        
        palm_point = get_palm_center(
            hand,
            frame_width,
            frame_height,
        )
        
        open_palm = is_open_palm(
            hand
        )
        
        if (
            open_palm
            and not drawing_mode
            and not note_typing
        ):
            update_palm_history(
                palm_history,
                palm_point,
                PALM_HISTORY_LENGTH,
            )

            current_time = time.monotonic()

            swipe_direction = detect_swipe(
                palm_history
            )

            if swipe_direction is not None:

                if (
                    current_time - last_swipe_time
                    >= SWIPE_COOLDOWN_SECONDS
                ):
                    print(
                        "SWIPE",
                        swipe_direction.upper(),
                    )

                    last_swipe_time = current_time

                palm_history.clear()

        else:
            palm_history.clear()
        
        current_time = time.monotonic()

        swipe_direction = detect_swipe(
            palm_history
        )

        if swipe_direction is not None:

            if (
                current_time - last_swipe_time
                >= SWIPE_COOLDOWN_SECONDS
            ):
                print(
                    "SWIPE",
                    swipe_direction.upper(),
                )

                last_swipe_time = current_time

            palm_history.clear()
                
        # =================================================
        # SMOOTH CURSOR POSITION
        # =================================================

        smoothed_point = smooth_cursor(
            index_point,
            smoothed_point,
            SMOOTHING_ALPHA,
        )

        cursor_point = smoothed_point


        # =================================================
        # SAFE ZONE CHECK
        # =================================================

        inside_safe_zone = (
            is_inside_safe_zone(
                cursor_point,
                safe_zone,
            )
        )

        # =================================================
        # PINCH DETECTION
        # =================================================

        pinching, pinch_armed, new_pinch = (
            detect_pinch(
                thumb_point,
                index_point,
                pinching,
                pinch_armed,
            )
        )

        if new_pinch:
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
            draw_hand_skeleton(
                frame,
                hand,
                connections,
         )

        # =================================================
        # CURSOR
        # =================================================

        draw_cursor(
            frame,
            cursor_point,
            tool,
        )

    # =====================================================
    # NO HAND
    # =====================================================

    else:
        palm_history.clear()
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
        smoothed_point = None

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
    
    (
        safe_left,
        safe_right,
        safe_top,
        safe_bottom,
    ) = safe_zone

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

    # Clear drawings
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

    # Show/hide safe frame
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

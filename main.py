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
    is_closed_fist,
    is_valid_page_swipe,
)

from page_view import (
    draw_page,
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
PAGE_TRANSITION_DURATION = 0.35

# Number of missing-hand frames before resetting gesture state
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

# Page-swipe state machine. A page swipe is accepted only after
# the user deliberately performs: closed fist -> open palm -> swipe.
swipe_state = "WAITING_FOR_FIST"
active_swipe_hand = None


pages = [
    "PAGE 1",
    "PAGE 2",
    "PAGE 3",
]

current_page_index = 0

transition_active = False
transition_direction = None
transition_start_time = None

transition_from_index = 0
transition_to_index = 0

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
    detected_hand_label = None

    # =====================================================
    # HAND DETECTED
    # =====================================================

    if hand_data is not None:
        hand_missing_frames = 0
        
        (
            hand,
            thumb_point,
            index_point,
            connections,
            hand_label,
        ) = hand_data

        detected_hand_label = hand_label
        
        palm_point = get_palm_center(
            hand,
            frame_width,
            frame_height,
        )
        
        open_palm = is_open_palm(
            hand
        )

        closed_fist = is_closed_fist(
            hand
        )
        
        # =================================================
        # SWIPE DETECTION
        # =================================================

        # Page swipes are only active outside drawing/note mode.
        if (
            not drawing_mode
            and not note_typing
        ):
            # -------------------------------------------------
            # STEP 1: WAIT FOR A DELIBERATE CLOSED FIST
            # -------------------------------------------------
            if swipe_state == "WAITING_FOR_FIST":
                palm_history.clear()

                if closed_fist:
                    active_swipe_hand = hand_label
                    swipe_state = "WAITING_FOR_OPEN"

                    print(
                        hand_label.upper(),
                        "HAND - FIST DETECTED | OPEN PALM",
                    )

            # -------------------------------------------------
            # STEP 2: SAME HAND MUST OPEN BEFORE WE ARM
            # -------------------------------------------------
            elif swipe_state == "WAITING_FOR_OPEN":
                palm_history.clear()

                # Switching hands cancels the sequence. The new
                # hand must start again with its own closed fist.
                if hand_label != active_swipe_hand:
                    swipe_state = "WAITING_FOR_FIST"
                    active_swipe_hand = None

                elif open_palm:
                    # Important: history is empty at the moment
                    # the palm opens. Movement used to bring the
                    # hand into view cannot become a fake swipe.
                    swipe_state = "ARMED"
                    palm_history.clear()

                    print(
                        hand_label.upper(),
                        "HAND - SWIPE READY",
                    )

            # -------------------------------------------------
            # STEP 3: ACCEPT EXACTLY ONE VALID SWIPE
            # -------------------------------------------------
            elif swipe_state == "ARMED":
                # Switching hands while armed cancels the gesture.
                if hand_label != active_swipe_hand:
                    palm_history.clear()
                    swipe_state = "WAITING_FOR_FIST"
                    active_swipe_hand = None

                # Closing again before swiping restarts the
                # close -> open preparation using the same hand.
                elif closed_fist:
                    palm_history.clear()
                    swipe_state = "WAITING_FOR_OPEN"

                elif open_palm:
                    update_palm_history(
                        palm_history,
                        palm_point,
                        PALM_HISTORY_LENGTH,
                    )

                    swipe_direction = detect_swipe(
                        palm_history
                    )

                    if swipe_direction is not None:
                        valid_page_swipe = (
                            is_valid_page_swipe(
                                hand_label,
                                swipe_direction,
                            )
                        )

                        if valid_page_swipe:
                            print(
                                hand_label.upper(),
                                "HAND - SWIPE",
                                swipe_direction.upper(),
                            )

                            # Start by assuming we stay on
                            # the current page.
                            target_page_index = (
                                current_page_index
                            )

                            # Right hand + swipe left
                            # = next page.
                            if (
                                hand_label == "Right"
                                and swipe_direction == "left"
                            ):
                                target_page_index = min(
                                    current_page_index + 1,
                                    len(pages) - 1,
                                )

                            # Left hand + swipe right
                            # = previous page.
                            elif (
                                hand_label == "Left"
                                and swipe_direction == "right"
                            ):
                                target_page_index = max(
                                    current_page_index - 1,
                                    0,
                                )

                            # Only start a transition if
                            # there is another page to show.
                            if (
                                target_page_index
                                != current_page_index
                                and not transition_active
                            ):
                                transition_active = True
                                transition_direction = (
                                    swipe_direction
                                )
                                transition_start_time = (
                                    time.monotonic()
                                )
                                transition_from_index = (
                                    current_page_index
                                )
                                transition_to_index = (
                                    target_page_index
                                )

                                print(
                                    "PAGE TRANSITION:",
                                    transition_from_index + 1,
                                    "->",
                                    transition_to_index + 1,
                                )

                            # One accepted swipe finishes the cycle.
                            # Another page action now requires a new
                            # closed fist -> open palm sequence.
                            swipe_state = "WAITING_FOR_FIST"
                            active_swipe_hand = None

                        # Discard the completed movement whether it
                        # was valid or the wrong direction.
                        palm_history.clear()

                else:
                    # A partial/unclear pose should not contribute
                    # stale coordinates to a later swipe.
                    palm_history.clear()

        else:
            # Drawing or note mode owns the hand interaction.
            palm_history.clear()
            swipe_state = "WAITING_FOR_FIST"
            active_swipe_hand = None

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

            # If the hand has really left the camera, cancel any
            # half-finished page-swipe sequence. The next gesture
            # must begin again with a closed fist.
            swipe_state = "WAITING_FOR_FIST"
            active_swipe_hand = None

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
    
    # =====================================================
    # HAND / SWIPE STATUS
    # =====================================================

    if detected_hand_label is not None:
        if swipe_state == "WAITING_FOR_FIST":
            swipe_state_text = "CLOSE HAND"

        elif swipe_state == "WAITING_FOR_OPEN":
            swipe_state_text = "OPEN PALM"

        else:
            if active_swipe_hand == "Right":
                swipe_state_text = "READY | SWIPE LEFT"
            elif active_swipe_hand == "Left":
                swipe_state_text = "READY | SWIPE RIGHT"
            else:
                swipe_state_text = "READY"

        hand_status_text = (
            f"{detected_hand_label.upper()} HAND | "
            f"{swipe_state_text}"
        )

        cv2.putText(
            frame,
            hand_status_text,
            (40, frame_height - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
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
    # CURRENT PAGE / PAGE TRANSITION
    # =====================================================

    if transition_active:
        # How long has the animation been running?
        elapsed_time = (
            time.monotonic()
            - transition_start_time
        )

        # Convert elapsed time to progress from
        # 0.0 (start) to 1.0 (finished).
        progress = min(
            elapsed_time
            / PAGE_TRANSITION_DURATION,
            1.0,
        )
        
        eased_progress = (
            1
            - (1 - progress) ** 3
        )

        # ---------------------------------------------
        # SWIPING LEFT
        # ---------------------------------------------
        if transition_direction == "left":
            # Old page moves off-screen to the left.
            old_page_offset = int(
                -frame_width
                * eased_progress
            )

            # New page starts on the right and
            # moves toward the center.
            new_page_offset = int(
                frame_width
                * (1.0 - eased_progress)
            )

        # ---------------------------------------------
        # SWIPING RIGHT
        # ---------------------------------------------
        else:
            # Old page moves off-screen to the right.
            old_page_offset = int(
                frame_width
                * eased_progress
            )

            # New page starts on the left and
            # moves toward the center.
            new_page_offset = int(
                -frame_width
                * (1.0 - eased_progress)
            )

        # Draw the page that is leaving.
        draw_page(
            display_frame,
            pages[
                transition_from_index
            ],
            old_page_offset,
        )

        # Draw the page that is entering.
        draw_page(
            display_frame,
            pages[
                transition_to_index
            ],
            new_page_offset,
        )

        # Finish the transition once progress reaches 1.0.
        if progress >= 1.0:
            current_page_index = (
                transition_to_index
            )

            transition_active = False
            transition_direction = None
            transition_start_time = None

            print(
                "CURRENT PAGE:",
                current_page_index + 1,
            )

    # No animation is active, so draw only
    # the current page in the center.
    else:
        draw_page(
            display_frame,
            pages[
                current_page_index
            ],
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

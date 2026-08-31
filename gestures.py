import math

def detect_pinch(
    thumb_point,
    index_point,
    pinching,
    pinch_armed,
    pinch_start_distance=40,
    pinch_release_distance=65,
):
    thumb_x, thumb_y = thumb_point
    index_x, index_y = index_point

    distance = math.hypot(
        index_x - thumb_x,
        index_y - thumb_y,
    )

    new_pinch = False

    # Fingers are clearly separated
    if distance > pinch_release_distance:
        pinching = False
        pinch_armed = True

    # A new pinch begins
    elif (
        distance < pinch_start_distance
        and pinch_armed
    ):
        pinching = True
        pinch_armed = False
        new_pinch = True

    return (
        pinching,
        pinch_armed,
        new_pinch,
    )

def update_palm_history(
    palm_history,
    palm_point,
    max_length,
):
    palm_history.append(
        palm_point
    )

    if len(palm_history) > max_length:
        palm_history.pop(0)

def detect_swipe(
    palm_history,
    minimum_horizontal_distance=180,
    maximum_vertical_distance=100,
):
    if len(palm_history) < 2:
        return None

    start_x, start_y = palm_history[0]
    end_x, end_y = palm_history[-1]

    horizontal_distance = (
        end_x - start_x
    )

    vertical_distance = abs(
        end_y - start_y
    )

    if (
        abs(horizontal_distance)
        < minimum_horizontal_distance
    ):
        return None

    if (
        vertical_distance
        > maximum_vertical_distance
    ):
        return None

    if horizontal_distance > 0:
        return "right"

    return "left"

def is_open_palm(hand):
    wrist = hand[0]

    finger_pairs = (
        (8, 6),    # index: tip, PIP
        (12, 10),  # middle
        (16, 14),  # ring
        (20, 18),  # pinky
    )

    extended_fingers = 0

    for tip_id, pip_id in finger_pairs:
        fingertip = hand[tip_id]
        pip_joint = hand[pip_id]

        tip_distance = math.sqrt(
            (fingertip.x - wrist.x) ** 2
            + (fingertip.y - wrist.y) ** 2
            + (fingertip.z - wrist.z) ** 2
        )

        pip_distance = math.sqrt(
            (pip_joint.x - wrist.x) ** 2
            + (pip_joint.y - wrist.y) ** 2
            + (pip_joint.z - wrist.z) ** 2
        )

        if tip_distance > pip_distance * 1.15:
            extended_fingers += 1

    return extended_fingers == 4
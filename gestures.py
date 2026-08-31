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
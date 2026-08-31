import mediapipe as mp
import cv2
import time

def create_hand_landmarker(
    model_path="hand_landmarker.task",
):
    base_options = mp.tasks.BaseOptions(
        model_asset_path=model_path,
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

    return landmarker

def detect_hand(
    landmarker,
    frame,
    detection_width,
    detection_height,
):
    frame_height, frame_width = frame.shape[:2]

    # Use a smaller image for faster hand detection
    detection_frame = cv2.resize(
        frame,
        (
            detection_width,
            detection_height,
        ),
    )

    # MediaPipe expects RGB instead of OpenCV's BGR
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

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms,
    )

    # No hand detected
    if not result.hand_landmarks:
        return None

    hand = result.hand_landmarks[0]

    connections = (
        mp.tasks.vision
        .HandLandmarksConnections
        .HAND_CONNECTIONS
    )

    thumb_tip = hand[4]
    index_tip = hand[8]

    thumb_point = (
        int(thumb_tip.x * frame_width),
        int(thumb_tip.y * frame_height),
    )

    index_point = (
        int(index_tip.x * frame_width),
        int(index_tip.y * frame_height),
    )

    return (
        hand,
        thumb_point,
        index_point,
        connections,
    )

def draw_hand_skeleton(
    frame,
    hand,
    connections,
):
    frame_height, frame_width = (
        frame.shape[:2]
    )

    # Draw every landmark
    for index, landmark in enumerate(hand):
        x = int(
            landmark.x * frame_width
        )

        y = int(
            landmark.y * frame_height
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

    # Draw lines between landmarks
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

def smooth_cursor(
    index_point,
    previous_smoothed_point,
    smoothing_alpha,
):
    index_x, index_y = index_point

    if previous_smoothed_point is None:
        return index_point

    previous_x, previous_y = (
        previous_smoothed_point
    )

    smoothed_x = int(
        smoothing_alpha
        * index_x
        + (
            1
            - smoothing_alpha
        )
        * previous_x
    )

    smoothed_y = int(
        smoothing_alpha
        * index_y
        + (
            1
            - smoothing_alpha
        )
        * previous_y
    )

    return (
        smoothed_x,
        smoothed_y,
    )

def get_palm_center(
    hand,
    frame_width,
    frame_height,
):
    palm_landmark_ids = (
        0,
        5,
        9,
        13,
        17,
    )

    x_total = 0
    y_total = 0

    for landmark_id in palm_landmark_ids:
        landmark = hand[
            landmark_id
        ]

        x_total += landmark.x
        y_total += landmark.y

    palm_x = int(
        (
            x_total
            / len(palm_landmark_ids)
        )
        * frame_width
    )

    palm_y = int(
        (
            y_total
            / len(palm_landmark_ids)
        )
        * frame_height
    )

    return (
        palm_x,
        palm_y,
    )
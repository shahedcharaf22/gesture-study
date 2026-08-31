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
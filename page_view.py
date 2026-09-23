import cv2


def draw_page(
    frame,
    page_text,
    x_offset=0,
):
    frame_height, frame_width = (
        frame.shape[:2]
    )

    page_width = int(
        frame_width * 0.55
    )

    page_height = int(
        frame_height * 0.70
    )

    page_x = (
        frame_width - page_width
    ) // 2

    page_y = (
        frame_height - page_height
    ) // 2

    page_x += x_offset

    top_left = (
        page_x,
        page_y,
    )

    bottom_right = (
        page_x + page_width,
        page_y + page_height,
    )

    cv2.rectangle(
        frame,
        top_left,
        bottom_right,
        (245, 245, 245),
        -1,
    )

    cv2.rectangle(
        frame,
        top_left,
        bottom_right,
        (60, 60, 60),
        2,
    )

    cv2.putText(
        frame,
        page_text,
        (
            page_x + 40,
            page_y + 60,
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

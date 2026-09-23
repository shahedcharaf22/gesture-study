import cv2


TOOL_NAMES = (
    "pointer",
    "pen",
    "highlighter",
    "eraser",
)

TOOL_LABELS = {
    "pointer": "POINTER",
    "pen": "PEN",
    "highlighter": "HIGHLIGHT",
    "eraser": "ERASER",
}

BUTTON_WIDTH = 118
BUTTON_HEIGHT = 46
BUTTON_GAP = 10
TOOLBAR_TOP = 18


def create_toolbar_buttons(frame_width):
    number_of_buttons = len(TOOL_NAMES)

    total_width = (
        number_of_buttons * BUTTON_WIDTH
        + (number_of_buttons - 1) * BUTTON_GAP
    )

    start_x = max(
        10,
        (frame_width - total_width) // 2,
    )

    buttons = {}

    for index, tool_name in enumerate(
        TOOL_NAMES
    ):
        x1 = start_x + index * (
            BUTTON_WIDTH + BUTTON_GAP
        )

        y1 = TOOLBAR_TOP

        x2 = x1 + BUTTON_WIDTH
        y2 = y1 + BUTTON_HEIGHT

        buttons[tool_name] = (
            x1,
            y1,
            x2,
            y2,
        )

    return buttons


def get_hovered_tool(
    cursor_point,
    buttons,
):
    if cursor_point is None:
        return None

    cursor_x, cursor_y = cursor_point

    for tool_name, rectangle in buttons.items():
        x1, y1, x2, y2 = rectangle

        if (
            x1 <= cursor_x <= x2
            and y1 <= cursor_y <= y2
        ):
            return tool_name

    return None


def draw_toolbar(
    frame,
    buttons,
    selected_tool,
    hovered_tool=None,
):
    for tool_name, rectangle in buttons.items():
        x1, y1, x2, y2 = rectangle

        if tool_name == selected_tool:
            background = (70, 70, 70)
            border_thickness = 3

        elif tool_name == hovered_tool:
            background = (50, 50, 50)
            border_thickness = 2

        else:
            background = (25, 25, 25)
            border_thickness = 1

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            background,
            -1,
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (220, 220, 220),
            border_thickness,
            cv2.LINE_AA,
        )

        label = TOOL_LABELS[tool_name]

        text_size, _ = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            1,
        )

        text_width, text_height = text_size

        text_x = (
            x1
            + (BUTTON_WIDTH - text_width) // 2
        )

        text_y = (
            y1
            + (BUTTON_HEIGHT + text_height) // 2
        )

        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

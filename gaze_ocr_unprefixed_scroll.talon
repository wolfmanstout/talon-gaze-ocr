mode: command
mode: user.dictation_command
tag: user.gaze_ocr_unprefixed_scroll
-
scroll up:
    user.move_cursor_to_gaze_point(0, 40)
    user.enhanced_scroll_up()
scroll up half:
    user.move_cursor_to_gaze_point(0, 40)
    user.enhanced_scroll_up(0.5)
scroll down:
    user.move_cursor_to_gaze_point(0, -40)
    user.enhanced_scroll_down()
scroll down half:
    user.move_cursor_to_gaze_point(0, -40)
    user.enhanced_scroll_down(0.5)
scroll left:
    user.move_cursor_to_gaze_point(40, 0)
    user.mouse_scroll_left()
scroll left half:
    user.move_cursor_to_gaze_point(40, 0)
    user.mouse_scroll_left(0.5)
scroll right:
    user.move_cursor_to_gaze_point(-40, 0)
    user.mouse_scroll_right()
scroll right half:
    user.move_cursor_to_gaze_point(-40, 0)
    user.mouse_scroll_right(0.5)

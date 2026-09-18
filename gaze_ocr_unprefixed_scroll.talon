mode: command
mode: user.dictation_command
tag: user.gaze_ocr_unprefixed_scroll
-
<user.gaze_scroll_point> up:
    user.move_cursor_to_gaze_point(gaze_scroll_point, 0, 40)
    user.enhanced_scroll_up()
<user.gaze_scroll_point> up half:
    user.move_cursor_to_gaze_point(gaze_scroll_point, 0, 40)
    user.enhanced_scroll_up(0.5)
<user.gaze_scroll_point> down:
    user.move_cursor_to_gaze_point(gaze_scroll_point, 0, -40)
    user.enhanced_scroll_down()
<user.gaze_scroll_point> down half:
    user.move_cursor_to_gaze_point(gaze_scroll_point, 0, -40)
    user.enhanced_scroll_down(0.5)
<user.gaze_scroll_point> left:
    user.move_cursor_to_gaze_point(gaze_scroll_point, 40, 0)
    user.mouse_scroll_left()
<user.gaze_scroll_point> left half:
    user.move_cursor_to_gaze_point(gaze_scroll_point, 40, 0)
    user.mouse_scroll_left(0.5)
<user.gaze_scroll_point> right:
    user.move_cursor_to_gaze_point(gaze_scroll_point, -40, 0)
    user.mouse_scroll_right()
<user.gaze_scroll_point> right half:
    user.move_cursor_to_gaze_point(gaze_scroll_point, -40, 0)
    user.mouse_scroll_right(0.5)

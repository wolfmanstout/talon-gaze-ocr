mode: command
mode: dictation
-
(eye | i) (hover | [cursor] move): user.move_cursor_to_gaze_point()
(eye | i) [left] (touch | click):
    user.move_cursor_to_gaze_point()
    mouse_click(0)
(eye | i) [left] double (touch | click):
    user.move_cursor_to_gaze_point()
    mouse_click(0)
    mouse_click(0)
(eye | i) right (touch | click):
    user.move_cursor_to_gaze_point()
    mouse_click(1)
(eye | i) middle (touch | click):
    user.move_cursor_to_gaze_point()
    mouse_click(2)
(eye | i) control (touch | click):
    user.move_cursor_to_gaze_point()
    key(ctrl:down)
    mouse_click(0)
    key(ctrl:up)

scroll up:
    user.move_cursor_to_gaze_point(0, 40)
    user.mouse_scroll_up()
    user.mouse_scroll_up()
    user.mouse_scroll_up()
    user.mouse_scroll_up()
    user.mouse_scroll_up()
    user.mouse_scroll_up()
    user.mouse_scroll_up()
scroll up half:
    user.move_cursor_to_gaze_point(0, 40)
    user.mouse_scroll_up()
    user.mouse_scroll_up()
    user.mouse_scroll_up()
    user.mouse_scroll_up()
scroll down:
    user.move_cursor_to_gaze_point(0, -40)
    user.mouse_scroll_down()
    user.mouse_scroll_down()
    user.mouse_scroll_down()
    user.mouse_scroll_down()
    user.mouse_scroll_down()
    user.mouse_scroll_down()
    user.mouse_scroll_down()
scroll down half:
    user.move_cursor_to_gaze_point(0, -40)
    user.mouse_scroll_down()
    user.mouse_scroll_down()
    user.mouse_scroll_down()
    user.mouse_scroll_down()
# parrot(shush):
#     user.move_cursor_to_gaze_point(0, -40)
#     user.power_momentum_scroll_down()
#     user.power_momentum_start(ts, 2.0)
# parrot(shush:repeat):
#     user.power_momentum_add(ts, power)
# parrot(shush:stop):
#     user.power_momentum_decaying()

# parrot(fff):
#     user.move_cursor_to_gaze_point(0, 40)
#     user.power_momentum_scroll_up()
#     user.power_momentum_start(ts, 2.0)
# parrot(fff:repeat):
#     user.power_momentum_add(ts, power)
# parrot(fff:stop):
#     user.power_momentum_decaying()

# scroll left: '"scroll left": Function(lambda: tracker.move_to_gaze_point((40, 0))) + Mouse("wheelleft:7"),'()+<wheelleft:7>
# scroll right: '"scroll right": Function(lambda: tracker.move_to_gaze_point((-40, 0))) + Mouse("wheelright:7"),'()+<wheelright:7>
# scroll start: '"scroll start": Function(lambda: scroller.start()),'()
# [scroll] stop: '"[scroll] stop": Function(lambda: scroller.stop()),'()
# scroll reset: '"scroll reset": Function(lambda: reset_scroller()),'()

(hover (seen | scene) | cursor move) <user.timestamped_prose>$: user.move_cursor_to_word(timestamped_prose)
[left] (touch | click) <user.timestamped_prose>$:
    user.move_cursor_to_word(timestamped_prose)
    mouse_click(0)
[left] double (touch | click) <user.timestamped_prose>$:
    user.move_cursor_to_word(timestamped_prose)
    mouse_click(0)
    mouse_click(0)
right (touch | click) <user.timestamped_prose>$:
    user.move_cursor_to_word(timestamped_prose)
    mouse_click(1)
middle (touch | click) <user.timestamped_prose>$:
    user.move_cursor_to_word(timestamped_prose)
    mouse_click(2)
control (touch | click) <user.timestamped_prose>$:
    user.move_cursor_to_word(timestamped_prose)
    key(ctrl:down)
    mouse_click(0)
    key(ctrl:up)
(go before | pre (seen | scene)) <user.timestamped_prose>$: user.move_text_cursor_to_word(timestamped_prose, "before")
(go after | post (seen | scene)) <user.timestamped_prose>$: user.move_text_cursor_to_word(timestamped_prose, "after")
select <user.prose_range>$:
    user.perform_ocr_action("select", "", prose_range)
{user.ocr_actions} [{user.ocr_modifiers}] (seen | scene) <user.prose_range>$:
    user.perform_ocr_action(ocr_actions, ocr_modifiers or "", prose_range)
replace [{user.ocr_modifiers}] [seen | scene] <user.prose_range> with <user.prose>$:
    user.replace_text(ocr_modifiers or "", prose_range, prose)
phones (seen | scene) <user.timestamped_homophone>$:
    user.select_text(timestamped_homophone)
    user.homophones_show_selection()

mode: command
# Optional: to use these commands in dictation mode, either use "mixed mode" (enable both dictation
# and command mode simultaneously) or define a new dictation_command mode and enable it alongside
# dictation mode. The following line will have no effect if dictation_command is not defined.
mode: user.dictation_command
-
# Commands that operate wherever you are looking.
# Example: "eye hover" to hover the cursor over where you're looking.
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
# Example: "eye control click" to control-click where you're looking.
(eye | i) <user.modifiers> (touch | click):
    user.move_cursor_to_gaze_point()
    key("{modifiers}:down")
    mouse_click(0)
    key("{modifiers}:up")

(eye | i) scroll up:
    user.move_cursor_to_gaze_point(0, 40)
    user.mouse_scroll_up()
(eye | i) scroll up half:
    user.move_cursor_to_gaze_point(0, 40)
    user.mouse_scroll_up(0.5)
(eye | i) scroll down:
    user.move_cursor_to_gaze_point(0, -40)
    user.mouse_scroll_down()
(eye | i) scroll down half:
    user.move_cursor_to_gaze_point(0, -40)
    user.mouse_scroll_down(0.5)
(eye | i) scroll left:
    user.move_cursor_to_gaze_point(40, 0)
    user.mouse_scroll_left()
(eye | i) scroll left half:
    user.move_cursor_to_gaze_point(40, 0)
    user.mouse_scroll_left(0.5)
(eye | i) scroll right:
    user.move_cursor_to_gaze_point(-40, 0)
    user.mouse_scroll_right()
(eye | i) scroll right half:
    user.move_cursor_to_gaze_point(-40, 0)
    user.mouse_scroll_right(0.5)

# Debugging commands.
ocr show [text]: user.show_ocr_overlay("text", 1)
ocr show boxes: user.show_ocr_overlay("boxes", 1)

# Commands that operate on text nearby where you're looking.
# Example: "hover seen apple" to hover the cursor over the word "apple".
(hover (seen | scene) | cursor move) <user.timestamped_prose>$: user.move_cursor_to_word(timestamped_prose)
# Example: "touch apple" to click the word "apple".
[left] (touch | click) <user.timestamped_prose>$:
    user.click_text(timestamped_prose)
[left] double (touch | click) <user.timestamped_prose>$:
    user.double_click_text(timestamped_prose)
right (touch | click) <user.timestamped_prose>$:
    user.right_click_text(timestamped_prose)
middle (touch | click) <user.timestamped_prose>$:
    user.middle_click_text(timestamped_prose)
<user.modifiers> (touch | click) <user.timestamped_prose>$:
    user.modifier_click_text(modifiers, timestamped_prose)
# Example: "go before apple" to move the text cursor before the word "apple".
(go before | pre (seen | scene)) <user.timestamped_prose>$: user.move_text_cursor_to_word(timestamped_prose, "before")
(go after | post (seen | scene)) <user.timestamped_prose>$: user.move_text_cursor_to_word(timestamped_prose, "after")
# Examples: 
# "select apple" to select the word "apple".
# "select apple through banana" to select the phrase "apple pear banana".
# "select through before apple" to select from the text cursor position to before the word "apple".
select <user.prose_range>$:
    user.perform_ocr_action("select", "", prose_range)
# Examples: 
# "take seen apple" to select the word "apple".
# "copy seen apple through banana" to copy the phrase "apple pear banana".
# "copy all seen apple" to copy all text from the field containing the word "apple".
{user.ocr_actions} [{user.ocr_modifiers}] (seen | scene) <user.prose_range>$:
    user.perform_ocr_action(ocr_actions, ocr_modifiers or "", prose_range)
# Example: "replace apple with banana" to replace the word "apple" with the word "banana".
replace [{user.ocr_modifiers}] [seen | scene] <user.prose_range> with <user.prose>$:
    user.replace_text(ocr_modifiers or "", prose_range, prose)
[go] before <user.timestamped_prose> say <user.prose>$:
    user.insert_adjacent_to_text(timestamped_prose, "before", prose)
[go] after <user.timestamped_prose> say <user.prose>$:
    user.insert_adjacent_to_text(timestamped_prose, "after", prose)
phones (seen | scene) <user.timestamped_prose>$:
    user.change_text_homophone(timestamped_prose)
resume (with | from) <user.timestamped_prose>$:
    user.resume_text(timestamped_prose)
revise with <user.timestamped_prose>$:
    user.revise_text(timestamped_prose)
revise from <user.timestamped_prose>$:
    user.revise_text_until_caret(timestamped_prose)

ocr tracker on: user.connect_ocr_eye_tracker()
ocr tracker off: user.disconnect_ocr_eye_tracker()

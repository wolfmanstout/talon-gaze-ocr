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
ocr show [text]: user.show_ocr_overlay("text")
ocr show [text] near <user.timestamped_prose>: user.show_ocr_overlay("text", timestamped_prose)
ocr show boxes: user.show_ocr_overlay("boxes")

# Commands that operate on text nearby where you're looking.
# Example: "hover seen apple" to hover the cursor over the word "apple".
(hover (seen | scene) | cursor move) <user.timestamped_prose>$: user.move_cursor_to_word(timestamped_prose)
# Example: "touch apple" to click the word "apple".
[left] (touch | click) <user.timestamped_prose>$:
    user.click_text(timestamped_prose)
# Variant which chooses best match if multiple targets are found.
lucky [left] (touch | click) <user.timestamped_prose>$:
    user.click_text_without_disambiguation(timestamped_prose)
# The following command is mostly for testing/debugging the onscreen_text capture.
screen [left] (touch | click) <user.onscreen_text>$:
    user.click_text(onscreen_text)
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
phones [word] (seen | scene) <user.timestamped_prose>$:
    user.change_text_homophone(timestamped_prose)

# Beta-only commands that offer intuitive text editing. See
# https://handsfreecoding.org/2024/03/15/making-writing-and-editing-with-your-voice-feel-natural/
# for detailed documentation.
# Example: "append with apple pear" to append "pear" after the word "apple".
append with <user.timestamped_prose_only>$:
    user.append_text(timestamped_prose_only)
# Example: "prepend with apple pear" to prepend "apple" before the word "pear".
prepend with <user.timestamped_prose_only>$:
    user.prepend_text(timestamped_prose_only)
# Example: "insert with apple pear" to either append "pear" after the word "apple" or prepend
# "apple" before the word "pear", depending on whether "apple" or "pear" is already onscreen.
insert with <user.timestamped_prose_only>$:
    user.insert_text_difference(timestamped_prose_only)
# Example: "revise with apple pear banana" to change "apple orange banana" to "apple pear banana".
revise with <user.timestamped_prose_only>$:
    user.revise_text(timestamped_prose_only)
# Example: "revise from apple pear banana" to replace all the text from "apple" to the text cursor
# with "apple pear banana".
(revise from <user.timestamped_prose_only> | revise with <user.timestamped_prose_only> cursor)$:
    user.revise_text_starting_with(timestamped_prose_only)
# Example: "revise through apple pear banana" to replace all the text from the text cursor to
# "banana" with "apple pear banana".
revise through <user.timestamped_prose_only>$:
    user.revise_text_ending_with(timestamped_prose_only)

ocr tracker on: user.connect_ocr_eye_tracker()
ocr tracker off: user.disconnect_ocr_eye_tracker()

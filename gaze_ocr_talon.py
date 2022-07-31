from dataclasses import dataclass
from math import floor
from statistics import mean
from typing import Dict, Iterable, Optional, Sequence, Union

import gaze_ocr
import gaze_ocr.talon
import numpy as np
import screen_ocr  # dependency of gaze-ocr
from talon import Context, Module, actions, app, cron, screen, settings
from talon.canvas import Canvas
from talon.grammar import Phrase
from talon.types import rect

mod = Module()
ctx = Context()

setting_ocr_logging_dir = mod.setting(
    "ocr_logging_dir",
    type=str,
    default=None,
    desc="If specified, log OCR'ed images to this directory.",
)
setting_ocr_click_offset_right = mod.setting(
    "ocr_click_offset_right",
    type=int,
    default=1,  # Windows biases towards the left of whatever is clicked.
    desc="Adjust the X-coordinate when clicking around OCR text.",
)
setting_ocr_select_pause_seconds = mod.setting(
    "ocr_select_pause_seconds",
    type=float,
    default=0.01,
    desc="Adjust the pause between clicks when performing a selection.",
)
setting_ocr_use_talon_backend = mod.setting(
    "ocr_use_talon_backend",
    type=bool,
    default=False,
    desc="If true, use Talon backend, otherwise use fast backend.",
)

mod.mode("gaze_ocr_disambiguation")


def add_homophones(
    homophones: Dict[str, Sequence[str]], to_add: Iterable[Iterable[str]]
):
    for words in to_add:
        merged_words = set(words)
        for word in words:
            old_words = homophones.get(word.lower(), [])
            merged_words.update(old_words)
        merged_words = sorted(merged_words)
        for word in merged_words:
            homophones[word.lower()] = merged_words


digits = "zero one two three four five six seven eight nine".split()
digits_map = {n: i for i, n in enumerate(digits)}


def on_ready():
    # Initialize eye tracking and OCR. See installation instructions:
    # https://github.com/wolfmanstout/gaze-ocr
    global tracker, ocr_reader, gaze_ocr_controller
    tracker = gaze_ocr.talon.TalonEyeTracker()
    homophones = actions.user.homophones_get_all()
    add_homophones(
        homophones, [(str(num), spoken) for spoken, num in digits_map.items()]
    )
    add_homophones(
        homophones,
        [
            (punctuation, spoken)
            for spoken, punctuation in actions.user.get_punctuation_words().items()
            if " " not in spoken
        ],
    )
    add_homophones(
        homophones,
        [
            # 0k is not actually a homophone but is frequently produced by OCR.
            ("ok", "okay", "0k"),
        ],
    )
    if setting_ocr_use_talon_backend.get():
        ocr_reader = screen_ocr.Reader.create_reader(
            backend="talon", radius=200, homophones=homophones
        )
    else:
        ocr_reader = screen_ocr.Reader.create_fast_reader(
            radius=200, homophones=homophones
        )
    gaze_ocr_controller = gaze_ocr.Controller(
        ocr_reader,
        tracker,
        save_data_directory=setting_ocr_logging_dir.get(),
        mouse=gaze_ocr.talon.Mouse(),
        keyboard=gaze_ocr.talon.Keyboard(),
    )


app.register("ready", on_ready)


@dataclass
class TimestampedText:
    text: str
    start: float
    end: float


@dataclass
class TextRange:
    start: Optional[TimestampedText]
    after_start: bool
    end: Optional[TimestampedText]
    before_end: bool


@dataclass
class TextPosition:
    text: TimestampedText
    position: str


# "edit" is frequently misrecognized as "at it", and is common in UIs.
@mod.capture(
    rule="(<phrase> | {user.vocabulary} | {user.punctuation} | {user.prose_snippets})+ | edit"
)
def timestamped_prose(m) -> TimestampedText:
    """Dictated text appearing onscreen."""
    words = []
    start = None
    end = None
    for item in m:
        if isinstance(item, Phrase):
            words.extend(
                actions.dictate.replace_words(actions.dictate.parse_words(item))
            )
            if not start:
                start = item.words[0].start
            end = item.words[-1].end
        else:
            words.append(str(item))
            if not start:
                start = item.start
            end = item.end
    assert start
    assert end
    return TimestampedText(text=" ".join(words), start=start, end=end)


@mod.capture(rule="[before | after] <self.timestamped_prose>")
def prose_position(m) -> TextPosition:
    """Position relative to prose."""
    return TextPosition(
        text=m.timestamped_prose,
        position=m[0] if m[0] in ("before", "after") else "",
    )


@mod.capture(rule="<self.prose_position> [through <self.prose_position>]")
def prose_range(m) -> TextRange:
    """A range of onscreen text."""
    has_through = len(m.prose_position_list) == 2
    if m.prose_position_1.position and not has_through:
        return TextRange(
            start=None,
            after_start=False,
            end=m.prose_position_1.text,
            before_end=m.prose_position_1.position == "before",
        )
    else:
        return TextRange(
            start=m.prose_position_1.text,
            after_start=m.prose_position_1.position == "after",
            end=m.prose_position_2.text if has_through else None,
            before_end=(
                (m.prose_position_2.position == "before") if has_through else None
            ),
        )


mod.list("ocr_actions", desc="Actions to perform on selected text.")
mod.list("ocr_modifiers", desc="Modifiers to perform on selected text.")
ctx.lists["self.ocr_actions"] = {
    "take": "select",
    "copy": "copy",
    "carve": "cut",
    "paste to": "paste",
    "clear": "delete",
    "delete": "delete",
    "chuck": "delete",
    "cap": "capitalize",
    "lower": "lowercase",
}
ctx.lists["self.ocr_modifiers"] = {
    "all": "selectAll",
}


def has_light_background(screenshot):
    array = np.array(screenshot)
    # From https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
    grayscale = 0.299 * array[:, :, 0] + 0.587 * array[:, :, 1] + 0.114 * array[:, :, 2]
    return np.mean(grayscale) > 128


disambiguation_canvas = None
debug_canvas = None
ambiguous_matches = None
disambiguation_generator = None


def reset_disambiguation():
    global ambiguous_matches, disambiguation_generator, disambiguation_canvas, debug_canvas
    ambiguous_matches = None
    disambiguation_generator = None
    if disambiguation_canvas:
        disambiguation_canvas.close()
    disambiguation_canvas = None
    if debug_canvas:
        debug_canvas.close()
    debug_canvas = None


def show_disambiguation():
    global ambiguous_matches, disambiguation_canvas

    def on_draw(c):
        contents = gaze_ocr_controller.latest_screen_contents()
        debug_color = (
            "000000" if has_light_background(contents.screenshot) else "ffffff"
        )
        nearest = contents.find_nearest_words_within_matches(ambiguous_matches)
        used_locations = set()
        for i, match in enumerate(ambiguous_matches):
            c.paint.typeface = "arial"
            c.paint.textsize = round(match[0].height * 2)
            c.paint.font.embolden = nearest == match
            c.paint.style = c.paint.Style.FILL
            c.paint.color = debug_color
            location = (match[0].left, match[0].top)
            while location in used_locations:
                # Shift right.
                location = (location[0] + match[0].height, location[1])
            used_locations.add(location)
            c.draw_text(str(i + 1), *location)

    actions.mode.enable("user.gaze_ocr_disambiguation")
    if disambiguation_canvas:
        disambiguation_canvas.close()
    disambiguation_canvas = Canvas.from_rect(screen.main().rect)
    disambiguation_canvas.register("draw", on_draw)
    disambiguation_canvas.freeze()


def begin_generator(generator):
    global ambiguous_matches, disambiguation_generator, disambiguation_canvas
    reset_disambiguation()
    try:
        ambiguous_matches = next(generator)
        disambiguation_generator = generator
        show_disambiguation()
    except StopIteration:
        # Execution completed without need for disambiguation.
        pass


def move_cursor_to_word_generator(text: TimestampedText):
    result = yield from gaze_ocr_controller.move_cursor_to_words_generator(
        text.text,
        disambiguate=True,
        timestamp=text.start,
        click_offset_right=setting_ocr_click_offset_right.get(),
    )
    if not result:
        actions.user.show_ocr_overlay("text", False, f"{text.text}")
        raise RuntimeError('Unable to find: "{}"'.format(text))


def move_text_cursor_to_word_generator(
    text: TimestampedText,
    position: str,
    include_whitespace: bool = False,
    hold_shift: bool = False,
):
    result = yield from gaze_ocr_controller.move_text_cursor_to_words_generator(
        text.text,
        disambiguate=True,
        cursor_position=position,
        timestamp=text.start,
        click_offset_right=setting_ocr_click_offset_right.get(),
        include_whitespace=include_whitespace,
        hold_shift=hold_shift,
    )
    if not result:
        actions.user.show_ocr_overlay("text", False, f"{text.text}")
        raise RuntimeError('Unable to find: "{}"'.format(text))


def select_text_generator(
    start: TimestampedText,
    end: Union[TimestampedText, str] = "",
    for_deletion: bool = False,
    after_start: bool = False,
    before_end: bool = False,
):
    start_text = start.text
    end_text = end.text if end else None
    result = yield from gaze_ocr_controller.select_text_generator(
        start_text,
        disambiguate=True,
        end_words=end_text,
        for_deletion=for_deletion,
        start_timestamp=start.start,
        end_timestamp=end.start if end else start.end,
        click_offset_right=setting_ocr_click_offset_right.get(),
        after_start=after_start,
        before_end=before_end,
        select_pause_seconds=setting_ocr_select_pause_seconds.get(),
    )
    if not result:
        actions.user.show_ocr_overlay(
            "text", False, f"{start.text}...{end.text if end else None}"
        )
        raise RuntimeError('Unable to select "{}" to "{}"'.format(start, end))


def perform_ocr_action_generator(
    ocr_action: str,
    ocr_modifier: str,
    text_range: TextRange,
    for_deletion: Optional[bool] = None,
):
    if text_range.end and not text_range.start:
        yield from move_text_cursor_to_word_generator(
            text_range.end,
            position="before" if text_range.before_end else "after",
            # Handle common case of cleanly removing the previous words.
            include_whitespace=text_range.before_end,
            hold_shift=True,
        )
    else:
        for_deletion = (
            for_deletion
            if for_deletion is not None
            else ocr_action in ("cut", "delete")
        )
        yield from select_text_generator(
            text_range.start,
            text_range.end,
            for_deletion,
            after_start=text_range.after_start,
            before_end=text_range.before_end,
        )
    if ocr_modifier == "":
        pass
    elif ocr_modifier == "selectAll":
        actions.edit.select_all()
    else:
        raise RuntimeError(f"Modifier not supported: {ocr_modifier}")

    if ocr_action == "select":
        pass
    elif ocr_action == "copy":
        actions.edit.copy()
    elif ocr_action == "cut":
        actions.edit.cut()
    elif ocr_action == "paste":
        actions.edit.paste()
    elif ocr_action == "delete":
        actions.key("backspace")
    elif ocr_action == "capitalize":
        text = actions.edit.selected_text()
        actions.insert(text[0].capitalize() + text[1:] if text else "")
    elif ocr_action == "lowercase":
        text = actions.edit.selected_text()
        actions.insert(text.lower())
    else:
        raise RuntimeError(f"Action not supported: {ocr_action}")


@mod.action_class
class GazeOcrActions:
    def move_cursor_to_word(text: TimestampedText):
        """Moves cursor to onscreen word."""
        begin_generator(move_cursor_to_word_generator(text))

    def move_text_cursor_to_word(
        text: TimestampedText,
        position: str,
        include_whitespace: bool = False,
        hold_shift: bool = False,
    ):
        """Moves text cursor near onscreen word."""
        begin_generator(
            move_text_cursor_to_word_generator(
                text, position, include_whitespace, hold_shift
            )
        )

    def select_text(
        start: TimestampedText,
        end: Union[TimestampedText, str] = "",
        for_deletion: bool = False,
        after_start: bool = False,
        before_end: bool = False,
    ):
        """Selects text near onscreen word at phrase timestamps."""
        begin_generator(
            select_text_generator(start, end, for_deletion, after_start, before_end)
        )

    def move_cursor_to_gaze_point(offset_right: int = 0, offset_down: int = 0):
        """Moves mouse cursor to gaze location."""
        tracker.move_to_gaze_point((offset_right, offset_down))

    def perform_ocr_action(
        ocr_action: str,
        ocr_modifier: str,
        text_range: TextRange,
        for_deletion: Optional[bool] = None,
    ):
        """Selects text and performs an action."""
        begin_generator(
            perform_ocr_action_generator(
                ocr_action, ocr_modifier, text_range, for_deletion
            )
        )

    def replace_text(ocr_modifier: str, text_range: TextRange, replacement: str):
        """Replaces onscreen text."""

        def run():
            yield from perform_ocr_action_generator(
                "select",
                ocr_modifier,
                text_range,
                for_deletion=settings.get("user.context_sensitive_dictation"),
            )
            if settings.get("user.context_sensitive_dictation"):
                actions.user.dictation_insert(replacement)
            else:
                actions.insert(replacement)

        begin_generator(run())

    def insert_adjacent_to_text(
        find_text: TimestampedText, position: str, insertion_text: str
    ):
        """Insert text adjacent to onscreen text."""

        def run():
            yield from move_text_cursor_to_word_generator(
                find_text,
                position,
            )
            if settings.get("user.context_sensitive_dictation"):
                actions.user.dictation_insert(insertion_text)
            else:
                actions.insert(insertion_text)

        begin_generator(run())

    def show_ocr_overlay(type: str, refresh: bool, query: str = ""):
        """Display overlay over primary screen."""
        global debug_canvas
        if debug_canvas:
            debug_canvas.close()
        if refresh:
            gaze_ocr_controller.read_nearby()
        contents = gaze_ocr_controller.latest_screen_contents()

        def on_draw(c):
            debug_color = (
                "000000" if has_light_background(contents.screenshot) else "ffffff"
            )
            c.paint.style = c.paint.Style.STROKE
            c.paint.color = debug_color
            c.draw_circle(
                contents.screen_coordinates[0],
                contents.screen_coordinates[1],
                contents.search_radius,
            )
            if query:
                c.paint.typeface = "arial"
                c.paint.textsize = 30
                c.paint.style = c.paint.Style.FILL
                c.paint.color = "FFFFFF"
                c.draw_text(query, x=screen.main().x + screen.main().width / 2, y=20)
                c.paint.style = c.paint.Style.STROKE
                c.paint.color = "000000"
                c.draw_text(query, x=screen.main().x + screen.main().width / 2, y=20)
            for line in contents.result.lines:
                for word in line.words:
                    if type == "text":
                        c.paint.typeface = "arial"
                        c.paint.textsize = floor(word.height)
                        c.paint.style = c.paint.Style.FILL
                        c.paint.color = debug_color
                        c.draw_text(word.text, word.left, word.top)
                    elif type == "boxes":
                        c.paint.style = c.paint.Style.STROKE
                        c.paint.color = debug_color
                        c.draw_rect(
                            rect.Rect(
                                x=word.left,
                                y=word.top,
                                width=word.width,
                                height=word.height,
                            )
                        )
                    else:
                        raise RuntimeError(f"Type not recognized: {type}")
            cron.after("2s", debug_canvas.close)

        debug_canvas = Canvas.from_rect(screen.main().rect)
        debug_canvas.register("draw", on_draw)
        debug_canvas.freeze()

    def choose_gaze_ocr_option(index: int):
        """Disambiguate with the provided index."""
        global ambiguous_matches, disambiguation_generator, disambiguation_canvas
        if (
            not ambiguous_matches
            or not disambiguation_generator
            or not disambiguation_canvas
        ):
            assert not ambiguous_matches
            assert not disambiguation_generator
            assert not disambiguation_canvas
            raise RuntimeError("Disambiguation not active")
        actions.mode.disable("user.gaze_ocr_disambiguation")
        disambiguation_canvas.close()
        disambiguation_canvas = None
        # Give the canvas a moment to disappear so it doesn't interfere with subsequent screenshots.
        actions.sleep("10ms")
        match = ambiguous_matches[index - 1]
        try:
            ambiguous_matches = disambiguation_generator.send(match)
            show_disambiguation()
        except StopIteration:
            # Execution completed successfully.
            reset_disambiguation()

    def hide_gaze_ocr_options():
        """Hide the disambiguation UI."""
        global ambiguous_matches, disambiguation_generator, disambiguation_canvas
        actions.mode.disable("user.gaze_ocr_disambiguation")
        reset_disambiguation()

    def click_text(text: TimestampedText):
        """Click on the provided on-screen text."""

        def run():
            yield from move_cursor_to_word_generator(text)
            actions.mouse_click(0)

        begin_generator(run())

    def double_click_text(text: TimestampedText):
        """Double-lick on the provided on-screen text."""

        def run():
            yield from move_cursor_to_word_generator(text)
            actions.mouse_click(0)
            actions.mouse_click(0)

        begin_generator(run())

    def right_click_text(text: TimestampedText):
        """Righ-click on the provided on-screen text."""

        def run():
            yield from move_cursor_to_word_generator(text)
            actions.mouse_click(1)

        begin_generator(run())

    def middle_click_text(text: TimestampedText):
        """Middle-click on the provided on-screen text."""

        def run():
            yield from move_cursor_to_word_generator(text)
            actions.mouse_click(2)

        begin_generator(run())

    def control_click_text(text: TimestampedText):
        """Control-click on the provided on-screen text."""

        def run():
            yield from move_cursor_to_word_generator(text)
            actions.key("ctrl:down")
            actions.mouse_click(0)
            actions.key("ctrl:up")

        begin_generator(run())

    def command_click_text(text: TimestampedText):
        """Command-click on the provided on-screen text."""

        def run():
            yield from move_cursor_to_word_generator(text)
            actions.key("cmd:down")
            actions.mouse_click(0)
            actions.key("cmd:up")

        begin_generator(run())

    def change_text_homophone(text: TimestampedText):
        """Switch the text to a different homophone."""

        def run():
            yield from select_text_generator(text)
            actions.user.homophones_show_selection()

        begin_generator(run())

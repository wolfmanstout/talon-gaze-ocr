from dataclasses import dataclass
from typing import Optional

from talon import Module, actions
from talon.grammar import Phrase

mod = Module()


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


@mod.capture(rule="<user.timestamped_phrase>+")
def timestamped_prose(m) -> TimestampedText:
    """Dictated text appearing onscreen."""
    return TimestampedText(
        text=" ".join([item.text for item in m]), start=m[0].start, end=m[-1].end
    )


# Forward to enable easy extension via a Context.
@mod.capture(rule="<user.timestamped_phrase_default>")
def timestamped_phrase(m) -> TimestampedText:
    """Dictated phrase appearing onscreen."""
    return m[0]


# "edit" is frequently misrecognized as "at it", and is common in UIs.
@mod.capture(
    rule="<phrase> | {user.vocabulary} | {user.punctuation} | {user.prose_snippets} | edit"
)
def timestamped_phrase_default(m) -> TimestampedText:
    """Dictated phrase appearing onscreen (default capture)."""
    item = m[0]
    if isinstance(item, Phrase):
        text = " ".join(
            actions.dictate.replace_words(actions.dictate.parse_words(item))
        )
        start = item.words[0].start
        end = item.words[-1].end
    else:
        text = str(item)
        start = item.start
        end = item.end
    return TimestampedText(text=text, start=start, end=end)


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
                (m.prose_position_2.position == "before") if has_through else False
            ),
        )

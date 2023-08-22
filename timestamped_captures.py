from dataclasses import dataclass
import re
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
    formatter = DictationFormat()
    return TimestampedText(
        text="".join([formatter.format(item.text) for item in m]),
        start=m[0].start,
        end=m[-1].end,
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


@mod.capture(
    rule="<self.one_ended_prose_range> | <self.prose_position> through <self.prose_position>"
)
def prose_range(m) -> TextRange:
    """A range of onscreen text."""
    if hasattr(m, "one_ended_prose_range"):
        return m.one_ended_prose_range
    return TextRange(
        start=m.prose_position_1.text,
        after_start=m.prose_position_1.position == "after",
        end=m.prose_position_2.text,
        before_end=m.prose_position_2.position == "before",
    )


@mod.capture(rule="[through] <self.prose_position>")
def one_ended_prose_range(m) -> TextRange:
    """A range of onscreen text with only start or end specified."""
    has_through = m[0] == "through"
    # As a convenience, allow dropping "through" if position is provided.
    if has_through or m.prose_position.position:
        if not m.prose_position.position:
            raise ValueError(
                'Text range "through <phrase>" not supported because cursor position is unknown.'
            )
        return TextRange(
            start=None,
            after_start=False,
            end=m.prose_position.text,
            before_end=m.prose_position.position == "before",
        )
    else:
        return TextRange(
            start=m.prose_position.text,
            after_start=m.prose_position.position == "after",
            end=None,
            before_end=False,
        )


# Remainder copied from
# https://github.com/talonhub/community/blob/98ba0bef2db67a39fcfe42640fe5b22090defdc7/core/text/text_and_dictation.py

# There must be a simpler way to do this, but I don't see it right now.
no_space_after = re.compile(
    r"""
  (?:
    [\s\-_/#@([{‘“]     # characters that never need space after them
  | (?<!\w)[$£€¥₩₽₹]    # currency symbols not preceded by a word character
  # quotes preceded by beginning of string, space, opening braces, dash, or other quotes
  | (?: ^ | [\s([{\-'"] ) ['"]
  )$""",
    re.VERBOSE,
)
no_space_before = re.compile(
    r"""
  ^(?:
    [\s\-_.,!?/%)\]}’”]   # characters that never need space before them
  | [$£€¥₩₽₹](?!\w)        # currency symbols not followed by a word character
  | [;:](?!-\)|-\()        # colon or semicolon except for smiley faces
  # quotes followed by end of string, space, closing braces, dash, other quotes, or some punctuation.
  | ['"] (?: $ | [\s)\]}\-'".,!?;:/] )
  # apostrophe s
  | 's(?!\w)
  )""",
    re.VERBOSE,
)


def omit_space_before(text: str) -> bool:
    return not text or no_space_before.search(text)


def omit_space_after(text: str) -> bool:
    return not text or no_space_after.search(text)


def needs_space_between(before: str, after: str) -> bool:
    return not (omit_space_after(before) or omit_space_before(after))


no_cap_after = re.compile(
    r"""(
    e\.g\.
    | i\.e\.
    )$""",
    re.VERBOSE,
)


def auto_capitalize(text, state=None):
    """
    Auto-capitalizes text. Text must contain complete words, abbreviations, and
    formatted expressions. `state` argument means:

    - None: Don't capitalize initial word.
    - "sentence start": Capitalize initial word.
    - "after newline": Don't capitalize initial word, but we're after a newline.
      Used for double-newline detection.

    Returns (capitalized text, updated state).
    """
    output = ""
    # Imagine a metaphorical "capitalization charge" travelling through the
    # string left-to-right.
    charge = state == "sentence start"
    newline = state == "after newline"
    sentence_end = False
    for c in text:
        # Sentence endings followed by space & double newlines create a charge.
        if (sentence_end and c in " \n\t") or (newline and c == "\n"):
            charge = True
        # Alphanumeric characters and commas/colons absorb charge & try to
        # capitalize (for numbers & punctuation this does nothing, which is what
        # we want).
        elif charge and (c.isalnum() or c in ",:"):
            charge = False
            c = c.capitalize()
        # Otherwise the charge just passes through.
        output += c
        newline = c == "\n"
        sentence_end = c in ".!?" and not no_cap_after.search(output)
    return output, (
        "sentence start"
        if charge or sentence_end
        else "after newline"
        if newline
        else None
    )


# ---------- DICTATION AUTO FORMATTING ---------- #
class DictationFormat:
    def __init__(self):
        self.reset()

    def reset(self):
        self.reset_context()
        self.force_no_space = False
        self.force_capitalization = None  # Can also be "cap" or "no cap".

    def reset_context(self):
        self.before = ""
        self.state = "sentence start"

    def update_context(self, before):
        if before is None:
            return
        self.reset_context()
        self.pass_through(before)

    def pass_through(self, text):
        _, self.state = auto_capitalize(text, self.state)
        self.before = text or self.before

    def format(self, text, auto_cap=True):
        if not self.force_no_space and needs_space_between(self.before, text):
            text = " " + text
        self.force_no_space = False
        if auto_cap:
            text, self.state = auto_capitalize(text, self.state)
        if self.force_capitalization == "cap":
            text = format_first_letter(text, lambda s: s.capitalize())
            self.force_capitalization = None
        if self.force_capitalization == "no cap":
            text = format_first_letter(text, lambda s: s.lower())
            self.force_capitalization = None
        self.before = text or self.before
        return text

    # These are used as callbacks by prose modifiers / dictation_mode commands.
    def cap(self):
        self.force_capitalization = "cap"

    def no_cap(self):
        self.force_capitalization = "no cap"

    def no_space(self):
        # This is typically used after repositioning the cursor, so it is helpful to
        # reset capitalization as well.
        #
        # FIXME: this sets state to "sentence start", capitalizing the next
        # word. probably undesirable, since most places are not the start of
        # sentences?
        self.reset_context()
        self.force_no_space = True


def format_first_letter(text, formatter):
    i = -1
    for i, c in enumerate(text):
        if c.isalpha():
            break
    if i >= 0 and i < len(text):
        text = text[:i] + formatter(text[i]) + text[i + 1 :]
    return text

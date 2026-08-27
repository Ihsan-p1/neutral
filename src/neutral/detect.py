import py3langid as langid

# Below this length, language detection is not reliable enough to act on.
_MIN_CHARS_FOR_DETECTION = 40


class TextTooLargeError(ValueError):
    """Text exceeds the configured size limit."""


def is_english(text: str) -> bool:
    """Return True if the text looks like English.

    Very short text is treated as English: wrongly flagging valid input as
    out of scope costs more than letting a short non-English snippet through.
    """
    stripped = text.strip()
    if len(stripped) < _MIN_CHARS_FOR_DETECTION:
        return True
    lang, _ = langid.classify(stripped)
    return lang == "en"

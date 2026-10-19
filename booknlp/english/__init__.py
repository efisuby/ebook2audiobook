"""English language support for BookNLP."""

from .english_booknlp import EnglishBookNLP


def __getattr__(name: str) -> object:
    """Allow 'import booknlp.english' and access to EnglishBookNLP."""
    from .english_booknlp import EnglishBookNLP

    if name == "EnglishBookNLP":
        return EnglishBookNLP
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

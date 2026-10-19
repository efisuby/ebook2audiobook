"""Russian language support for BookNLP."""

from .russian_booknlp import RussianBookNLP


def __getattr__(name: str) -> object:
    """Allow 'import booknlp.russian' and access to RussianBookNLP."""
    from .russian_booknlp import RussianBookNLP

    if name == "RussianBookNLP":
        return RussianBookNLP
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

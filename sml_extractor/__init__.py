"""SML Book Dialog Extractor - Convert books to SML format for ebook2audiobook."""

import functools
import time

__version__ = "1.0.0"


def log(message:str)->None:
    """Print a status line of the SML pipeline.

    Every step announces what it started and what it finished (and how long it
    took), so a run can be followed in the console the application prints to.
    ``flush`` keeps the lines in order when stdout is a pipe (the launcher
    captures it).
    """
    print(f"[sml] {message}", flush=True)


def traced(name:str, describe=None):
    """Decorate a function so it announces its start, its end and its failures.

    The call prints ``[sml] <name> — начали`` before it runs and, afterwards,
    ``[sml] <name> — закончили[: <outcome>] за <секунды> с``; an exception is
    reported as ``ошибка`` and passes through unchanged, so the console still
    tells which step failed.  ``describe`` turns the return value into the
    outcome (``lambda result: f'{len(result)} персонажей'``).
    """
    def decorate(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            started = time.monotonic()
            log(f"{name} — начали")
            try:
                result = function(*args, **kwargs)
            except Exception as error:
                log(f"{name} — ошибка через {time.monotonic() - started:.1f} с: {error}")
                raise
            detail = ""
            if describe is not None:
                try:
                    outcome = describe(result)
                except Exception:
                    outcome = None
                if outcome:
                    detail = f": {outcome}"
            log(f"{name} — закончили{detail} за {time.monotonic() - started:.1f} с")
            return result
        return wrapper
    return decorate

"""Core module for running BookNLP and extracting dialog/character data."""

import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable


def configure_booknlp_cache(data_root:str|Path|None=None)->Path:
    '''Configure the Hugging Face cache before its first import.'''
    if data_root:
        repo_dir = Path(data_root).expanduser().resolve()
    else:
        from .voice_library import configured_library_root
        repo_dir = configured_library_root()
    model_dir = repo_dir / 'models' / 'booknlp_models'
    os.environ['HF_HOME'] = str(model_dir / 'huggingface')
    return model_dir


def check_booknlp_installation(language:str='eng')->tuple[bool,str]:
    """Check if BookNLP and its dependencies are properly installed.

    Args:
        language: Language code ('eng' or 'rus') to check.

    Returns:
        Tuple of (is_ok, message). If not ok, message describes what's missing.
    """
    errors = []
    lang_name = "Russian" if language == "rus" else "English"

    # Check booknlp package for the specified language
    try:
        if language == "eng":
            import booknlp.english as booknlp_lang
            from booknlp.english import EnglishBookNLP  # noqa: F401
        else:  # rus
            import booknlp.russian as booknlp_lang
            from booknlp.russian import RussianBookNLP  # noqa: F401
    except ImportError:
        errors.append(
            f"BookNLP {lang_name.lower()} module is not found. Ensure the 'booknlp' directory "
            f"is in the same folder as this script."
        )
        return False, "\n".join(errors)

    # The vendored English BookNLP code reads these files during model
    # initialization. The Russian pipeline uses spaCy only, so the resource
    # check applies to English alone.
    if language == "eng":
        resource_dir = Path(booknlp_lang.__file__).parent / "data"
        required_resources = (
            "aliases.txt",
            "entity_cat.tagset",
            "gutenberg_prop_gender_terms.txt",
            "supersense.tagset",
            "wordnet.first.sense",
        )
        missing_resources = [
            name for name in required_resources if not (resource_dir / name).is_file()
        ]
        if missing_resources:
            return False, (
                f"BookNLP {language.upper()} resource files are missing: "
                + ", ".join(missing_resources)
                + f". Expected them in {resource_dir}. Reinstall E2A-SML from a complete checkout."
            )

    # Check key dependencies that commonly fail
    dep_checks = [
        ("torch", "torch", "uv pip install torch"),
        ("transformers", "transformers", "uv pip install 'transformers>=4.30.0,<5'"),
        ("spacy", "spacy", "uv pip install 'spacy>=3.5.0'"),
        ("sentence_transformers", "sentence-transformers", "uv pip install sentence-transformers"),
        ("numpy", "numpy", "uv pip install 'numpy>=1.24.0'"),
        ("pkg_resources", "setuptools<81", "uv pip install 'setuptools<81'"),
    ]

    for module_name, pkg_name, install_cmd in dep_checks:
        try:
            __import__(module_name)
        except ImportError:
            errors.append(f"  - {pkg_name} is missing. Fix: {install_cmd}")

    if errors:
        return False, (
            "BookNLP dependencies are missing:\n"
            + "\n".join(errors)
            + "\n\nOr install all at once:\n"
            "  uv pip install -r requirements.txt\n"
            '  uv pip install "$(python -m spacy info en_core_web_sm --url)"'
        )

    # Check spacy model for the specified language
    spacy_model = "en_core_web_sm" if language == "eng" else "ru_core_news_sm"
    try:
        import spacy
        spacy.load(spacy_model)
    except OSError:
        lang_name = "English" if language == "eng" else "Russian"
        install_cmd = f'  uv pip install "$(python -m spacy info {spacy_model} --url)"'
        errors.append(
            f"{lang_name} spaCy model ({spacy_model}) not found. Install it with:\n" + install_cmd
        )

    if errors:
        return False, "\n".join(errors)

    # Try the actual BookNLP import that tends to fail
    try:
        if language == "eng":
            from booknlp.booknlp import BookNLP  # noqa: F401
        else:  # rus
            from booknlp.russian import RussianBookNLP  # noqa: F401
    except (ImportError, ModuleNotFoundError, AttributeError, OSError) as e:
        return False, (
            f"BookNLP failed to initialize: {e}\n\n"
            "This usually means a dependency version conflict.\n"
            "Try reinstalling dependencies in a clean environment:\n"
            "  uv pip install -r requirements.txt\n"
            '  uv pip install "$(python -m spacy info en_core_web_sm --url)"'
        )

    return True, f"BookNLP {language.upper()} is ready."


def convert_ebook_to_txt(input_file: str, output_dir: str) -> str:
    """Convert an ebook file to plain text using calibre's ebook-convert if needed.

    Supports: .txt, .epub, .mobi, .pdf, .html, .fb2, .azw, .azw3
    Returns the path to the plain text file.
    """
    ext = Path(input_file).suffix.lower()
    if ext == ".txt":
        return input_file

    txt_path = os.path.join(output_dir, Path(input_file).stem + ".txt")
    try:
        import subprocess

        result = subprocess.run(
            ["ebook-convert", input_file, txt_path],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ebook-convert failed: {result.stderr}"
            )
        return txt_path
    except FileNotFoundError:
        raise RuntimeError(
            "ebook-convert not found. Install Calibre to convert non-txt ebook formats. "
            "Download from: https://calibre-ebook.com/download"
        )


def run_booknlp(
    input_file:str,
    output_dir:str,
    model:str='small',
    language:str='eng',
    progress_callback:Callable[[str,int],None]|None=None,
    data_root:str|Path|None=None,
)->dict[str,str]:
    """Run BookNLP pipeline on a text file and return extracted data.

    Args:
        input_file: Path to the input text file.
        output_dir: Directory for BookNLP output files.
        model: BookNLP model size ('small' or 'big').
        language: Language code for text analysis ('eng' or 'rus').
        progress_callback: Optional callable(message, pct) for progress updates.
        data_root: This tool's data directory for the model cache.

    Returns:
        Dict with keys: 'book_id', 'output_dir', 'characters', 'tokens_file',
                        'quotes_file', 'entities_file', 'book_file'

    Raises:
        RuntimeError: If BookNLP or its dependencies are not properly installed.
    """
    model_dir = configure_booknlp_cache(data_root)
    # Pre-check installation before attempting import
    ok, msg = check_booknlp_installation(language)
    if not ok:
        raise RuntimeError(f"BookNLP installation check failed:\n{msg}")

    os.makedirs(output_dir, exist_ok=True)
    book_id = Path(input_file).stem

    if progress_callback:
        progress_callback("Initializing BookNLP...", 5)

    model_params = {
        'pipeline': 'entity,quote,supersense,event,coref',
        'model': model,
        'model_path': str(model_dir),
    }

    # Use the Russian pipeline for 'rus'; the vendored English BookNLP shim
    # keys its language check on "en" while this project uses "eng"/"rus".
    if language == "rus":
        from booknlp.russian import RussianBookNLP

        booknlp_cls, booknlp_language = RussianBookNLP, "rus"
    else:
        from booknlp.booknlp import BookNLP

        booknlp_cls, booknlp_language = BookNLP, "en"

    booknlp = booknlp_cls(booknlp_language, model_params)

    if progress_callback:
        progress_callback(f"Processing book with BookNLP ({model} model)...", 10)

    booknlp.process(input_file, output_dir, book_id)

    if progress_callback:
        progress_callback("BookNLP processing complete.", 60)

    result = {
        "book_id": book_id,
        "output_dir": output_dir,
        "tokens_file": os.path.join(output_dir, f"{book_id}.tokens"),
        "quotes_file": os.path.join(output_dir, f"{book_id}.quotes"),
        "entities_file": os.path.join(output_dir, f"{book_id}.entities"),
        "book_file": os.path.join(output_dir, f"{book_id}.book"),
    }

    # Also check for generated character/book text files
    char_json = os.path.join(output_dir, f"{book_id}.characters_simple.json")
    book_txt = os.path.join(output_dir, f"{book_id}.book.txt")
    if os.path.exists(char_json):
        result["characters_simple_file"] = char_json
    if os.path.exists(book_txt):
        result["book_txt_file"] = book_txt

    return result


def load_booknlp_output(output_dir: str, book_id: str) -> dict:
    """Load BookNLP output files and return structured data.

    Returns dict with: tokens, quotes, entities, characters, book_data
    """
    data = {"book_id": book_id, "output_dir": output_dir}

    # Load tokens
    tokens_file = os.path.join(output_dir, f"{book_id}.tokens")
    if os.path.exists(tokens_file):
        data["tokens"] = _parse_tokens_file(tokens_file)

    # Load quotes
    quotes_file = os.path.join(output_dir, f"{book_id}.quotes")
    if os.path.exists(quotes_file):
        data["quotes"] = _parse_quotes_file(quotes_file)

    # Load entities
    entities_file = os.path.join(output_dir, f"{book_id}.entities")
    if os.path.exists(entities_file):
        data["entities"] = _parse_entities_file(entities_file)

    # Load book JSON
    book_file = os.path.join(output_dir, f"{book_id}.book")
    if os.path.exists(book_file):
        with open(book_file, "r", encoding="utf-8") as f:
            data["book_data"] = json.load(f)

    # Load characters_simple.json if exists
    char_file = os.path.join(output_dir, f"{book_id}.characters_simple.json")
    if os.path.exists(char_file):
        with open(char_file, "r", encoding="utf-8") as f:
            data["characters_simple"] = json.load(f)

    # Load book.txt if exists
    book_txt = os.path.join(output_dir, f"{book_id}.book.txt")
    if os.path.exists(book_txt):
        with open(book_txt, "r", encoding="utf-8") as f:
            data["book_txt"] = f.read()

    return data


def _parse_tokens_file(filepath: str) -> list:
    """Parse BookNLP tokens file into list of token dicts."""
    tokens = []
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= len(header):
                token = dict(zip(header, parts))
                tokens.append(token)
    return tokens


def _parse_quotes_file(filepath: str) -> list:
    """Parse BookNLP quotes file into list of quote dicts."""
    quotes = []
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= len(header):
                quote = dict(zip(header, parts))
                quotes.append(quote)
    return quotes


def _parse_entities_file(filepath: str) -> list:
    """Parse BookNLP entities file into list of entity dicts."""
    entities = []
    with open(filepath, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= len(header):
                entity = dict(zip(header, parts))
                entities.append(entity)
    return entities


def extract_characters(booknlp_data: dict, language:str='eng') -> list:
    """Extract character information from BookNLP output.

    Args:
        booknlp_data: BookNLP output data dictionary.
        language: Language code for TTS engine settings ('eng' or 'rus').

    Returns list of character dicts with:
        normalized_name, inferred_gender, inferred_age_category, voice, language
    """
    # If characters_simple already exists from BookNLP, use that
    if "characters_simple" in booknlp_data:
        return booknlp_data["characters_simple"].get("characters", [])

    # Otherwise, extract from book_data
    characters = []

    # Always add narrator first
    chars_list = [{"name": "Narrator"}]  # type: ignore

    for char in chars_list:
        characters.append(
            {
                "normalized_name": f"Narrator-{char['name']}",
                "inferred_gender": "unknown",
                "inferred_age_category": "unknown",
                "tts_engine": "XTTSv2" if language == "eng" else "XTTSv2-Russian",
                "language": language,
                "voice": None,
            }
        )

    if "book_data" not in booknlp_data:
        return characters

    book_data = booknlp_data["book_data"]
    char_data = book_data.get("characters", [])

    for char in char_data:
        gender = _infer_gender(char, language)
        age = _infer_age(char, language)
        raw_name = _first_proper_name(char) or f"Character{char.get('id', 0)}"
        name = _normalize_name(raw_name)

        characters.append(
            {
                "normalized_name": name,
                "inferred_gender": gender,
                "inferred_age_category": age,
                "tts_engine": "XTTSv2" if language == "eng" else "XTTSv2-Russian",
                "language": language,
                "voice": None,
            }
        )

    return characters


def _first_proper_name(char: dict) -> str:
    """Return the most frequent proper name of a BookNLP character, if any.

    BookNLP's ``.book`` JSON stores names under ``mentions.proper`` as
    ``{"c": count, "n": name}`` entries; some outputs use ``names.proper``
    with plain strings instead.
    """
    for key in ("mentions", "names"):
        data = char.get(key)
        if isinstance(data, dict):
            proper = data.get("proper") or []
            if proper:
                first = proper[0]
                if isinstance(first, dict):
                    return str(first.get("n", ""))
                return str(first)
    return str(char.get("name", ""))


# Russian suffix and name tables shared by gender inference.
_RU_FEMALE_SUFFIXES = ("а", "я", "ия", "ая", "ья", "на", "ина")
_RU_MALE_SUFFIXES = ("ов", "ев", "ёв", "ин", "ын", "ий", "ый", "ой", "ь", "й")
_RU_MALE_NAMES = {
    "иван", "пётр", "петр", "александр", "дмитрий", "сергей", "николай",
    "андрей", "павел", "игорь", "владимир", "алексей", "роман", "алексей",
    "максим", "артём", "артем", "юрий", "борис", "виктор", "виталий",
    "георгий", "григорий", "денис", "евгений", "емельян", "захар", "илья",
    "кирилл", "леонид", "михаил", "олег", "ростислав", "станислав", "степан",
    "тимур", "фёдор", "федор", "филипп", "харитон", "ярослав",
}
_RU_FEMALE_NAMES = {
    "анна", "мария", "елена", "наталья", "ольга", "татьяна", "ксения",
    "дарья", "марина", "светлана", "вера", "екатерина", "анастасия",
    "алина", "полина", "юлия", "ирина", "лариса", "милана", "софия",
    "алиса", "варвара", "валерия", "кристина", "надежда", "людмила",
    "галя", "лидия", "зоя", "тамара", "раиса", "нина", "яна", "рина",
}


def _infer_gender_from_russian_name(name: str) -> str | None:
    """Guess a Russian character's gender from their name, or None."""
    name = name.lower().strip()
    if not name:
        return None

    if name in _RU_FEMALE_NAMES:
        return "female"
    if name in _RU_MALE_NAMES:
        return "male"

    # Female endings are checked first: "Пушкина" also matches the male "ин".
    for suffix in _RU_FEMALE_SUFFIXES:
        if name.endswith(suffix):
            return "female"
    for suffix in _RU_MALE_SUFFIXES:
        if name.endswith(suffix):
            return "male"
    return None


def _infer_gender(char: dict, language: str = 'eng') -> str:
    """Infer gender from character data.

    Args:
        char: Character dictionary from BookNLP.
        language: Language code for language-specific inference logic.

    Returns:
        Gender as 'male', 'female', or 'unknown'.
    """
    gender_data = char.get("g", None)

    if isinstance(gender_data, dict):
        # English BookNLP counts pronoun mentions under English keys; a
        # Russian pipeline may count Russian pronouns instead.
        he_count = (
            gender_data.get("he/him/his", 0)
            + gender_data.get("он", 0)
            + gender_data.get("он/его/ему/им", 0)
        )
        she_count = (
            gender_data.get("she/her", 0)
            + gender_data.get("она", 0)
            + gender_data.get("она/её/ей/ею", 0)
        )
        they_count = (
            gender_data.get("they/them/their", 0)
            + gender_data.get("они", 0)
            + gender_data.get("они/их/ими", 0)
        )

        if he_count > she_count and he_count > they_count:
            return "male"
        if she_count > he_count and she_count > they_count:
            return "female"
    elif isinstance(gender_data, str):
        if gender_data in ("he/him/his", "male"):
            return "male"
        if gender_data in ("she/her", "female"):
            return "female"

    # Language-specific fallback: infer gender from the character's name.
    # Note: this must run before giving up, otherwise Russian characters whose
    # `g` field is missing always end up as "unknown".
    name = _first_proper_name(char)
    if name:
        if language == "rus":
            gender = _infer_gender_from_russian_name(name)
            if gender:
                return gender

    return "unknown"


def _infer_age(char: dict, language:str='eng') -> str:
    """Infer age category from character data.

    Args:
        char: Character dictionary from BookNLP.
        language: Language code for language-specific inference logic.

    Returns:
        Age category as 'child', 'teen', 'adult', or 'elder'.
    """
    # Check if age was already inferred
    if "inferred_age_category" in char:
        return char["inferred_age_category"]

    # Look at modifiers and actions for age clues
    modifiers = char.get("modifiers", [])
    mod_text = " ".join(str(m) for m in modifiers).lower() if modifiers else ""

    child_words = {"child", "boy", "girl", "baby", "infant", "toddler", "kid", "little", "young"}
    teen_words = {"teen", "teenager", "adolescent", "youth", "teenage"}
    elder_words = {"old", "elderly", "aged", "ancient", "grandfather", "grandmother", "grandpa", "grandma"}

    for word in child_words:
        if word in mod_text:
            return "child"
    for word in teen_words:
        if word in mod_text:
            return "teen"
    for word in elder_words:
        if word in mod_text:
            return "elder"

    # Russian language fallback (search descriptive modifiers for age terms)
    if language == "rus":
        hints = f"{mod_text} {_first_proper_name(char)}".lower()
        child_terms = {"ребёнок", "ребенок", "мальчик", "девочка", "дитя", "младенец", "малыш"}
        teen_terms = {"подросток", "подростковый", "отрок"}
        elder_terms = {"старик", "старуха", "бабушка", "дедушка", "пожилой", "дряхлый"}

        for term in child_terms:
            if term in hints:
                return "child"
        for term in teen_terms:
            if term in hints:
                return "teen"
        for term in elder_terms:
            if term in hints:
                return "elder"

    return "adult"


def _normalize_name(name: str) -> str:
    """Normalize a character name to CamelCase.

    Keeps any alphabet's letters (Latin and Cyrillic alike) so Russian names
    survive normalization instead of collapsing to "Unknown".
    """
    # Remove special characters, keep letters/digits/whitespace/hyphens
    name = re.sub(r"[^\w\s-]", "", name, flags=re.UNICODE).replace("_", "")
    # Convert to CamelCase
    parts = name.strip().split()
    return "".join(word.capitalize() for word in parts) if parts else "Unknown"

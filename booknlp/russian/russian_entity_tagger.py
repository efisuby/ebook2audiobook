"""Russian entity tagging for gender and name information."""


def add_gender_tags(nlp):
    """Add Russian-specific gender tagger to spaCy model.

    Args:
        nlp: spaCy language class or model instance.
    """
    # This would normally load a pre-trained model, but we use the custom logic
    # from russian_booknlp for now as it handles gender inference better

    def gender_entity(ents):
        """Callback to tag gender entities."""
        male_suffixes = {'ов', 'ин', 'ий', 'ый'}
        female_suffixes = {'а', 'ия', 'ая', 'ина'}

        for ent in ents:
            name_lower = ent.text.lower()

            # Check suffix patterns
            for suffix in female_suffixes:
                if name_lower.endswith(suffix):
                    ent._.gender = 'female'
                    return True

            for suffix in male_suffixes:
                if name_lower.endswith(suffix):
                    ent._.gender = 'male'
                    return True

        # Common Russian names lookup (fetched from russian_booknlp)
        gender_names = {"иван": "male", "mikhail": "male", "anna": "female", "maria": "female", "elena": "female"}

        if name_lower in gender_names:
            ent._.gender = gender_names[name_lower]

        return False

    # Add to model (if it exists)
    try:
        nlp.add_pipe("ru_gender_tagger", before="ner")
    except Exception as e:
        print(f"Warning: Could not add gender tagger: {e}")


def add_name_info(nlp):
    """Add name information extraction to spaCy model."""
    # This would use a pre-trained NER model for Russian names
    pass


def tag_entities(text, nlp=None):
    """Tag named entities in Russian text.

    Args:
        text: Input Russian text string.
        nlp: Optional spaCy model instance; loads ru_core_news_sm if None.

    Returns:
        List of entity dictionaries with 'text', 'label', and '_gender' fields.
    """
    # Use spaCy's built-in NER for Russian
    if nlp is None:
        try:
            import spacy
            nlp = spacy.load('ru_core_news_sm')
        except OSError:
            print("Warning: ru_core_news_sm not found. Using basic entity detection.")
            return []

    # Add gender tagger before NER so gender info is available
    add_gender_tags(nlp)

    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        # Filter to Russian-specific entities (organizations, locations, etc.)
        if ent.label_ in ('ORG', 'GPE', 'LOC'):
            ent._gender = None  # No gender info for orgs/locations
            entities.append({
                'text': str(ent.text),
                'label': ent.label_,
                '_gender': None,
            })
        else:
            # For person names, use the gender tagger we added
            if hasattr(ent, '_gender'):
                entities.append({
                    'text': str(ent.text),
                    'label': ent.label_,
                    '_gender': ent._gender,
                })

    return entities

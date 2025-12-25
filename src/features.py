def extract_behavioral_features(text):
    words = text.split()
    unique_words = set(words)
    text_length = len(text)
    word_count = len(words)

    return {
        "text_length": text_length,
        "word_count": word_count,
        "unique_word_ratio": len(unique_words) / (word_count + 1),
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "capital_ratio": sum(1 for c in text if c.isupper()) / (text_length + 1),
        "avg_word_length": sum(len(w) for w in words) / (word_count + 1)
    }

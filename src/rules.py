def rule_based_score(review_text):
    score = 0
    words = review_text.split()
    word_count = len(words)

    # Short review
    if word_count < 60:
        score += 1

    # Excessive exclamations
    if review_text.count("!") >= 3:
        score += 1

    # Uppercase emphasis
    upper_ratio = sum(1 for c in review_text if c.isupper()) / max(len(review_text), 1)
    if upper_ratio > 0.1:
        score += 1

    # Marketing phrases
    marketing_phrases = [
        "best ever", "highly recommended", "must buy",
        "life changing", "worth every penny"
    ]
    text_lower = review_text.lower()
    if any(p in text_lower for p in marketing_phrases):
        score += 1

    # Low vocabulary diversity
    if len(set(words)) / max(len(words), 1) < 0.5:
        score += 1

    return score  # 0–5

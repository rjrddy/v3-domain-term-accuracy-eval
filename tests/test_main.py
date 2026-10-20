from main import scores


def test_terms_use_occurrence_alignment_not_set_membership():
    result = scores("country country", "country", ["country"])
    assert result["term_occurrences"] == 2 and result["term_correct"] == 1
    assert result["term_accuracy"] == 0.5


def test_wer_can_hide_key_term_error():
    result = scores(
        "the patient has atrial fibrillation today",
        "the patient has atrial confusion today",
        ["atrial fibrillation"],
    )
    assert result["wer"] < 0.5 and result["term_accuracy"] == 0

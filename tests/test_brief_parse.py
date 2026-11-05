import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from brief_parse import ALLERGENS, parse_brief, warning_text


def test_common_names_map_to_the_14():
    brief, unk_a, unk_d = parse_brief("veggie", "Shellfish, dairy, tree nuts, egg")
    assert brief.allergens == ["crustaceans", "molluscs", "milk", "nuts", "eggs"]
    assert brief.diets == ["vegetarian"]
    assert unk_a == [] and unk_d == []


def test_unknown_items_are_reported_not_dropped_or_raised():
    brief, unk_a, unk_d = parse_brief("keto", "peanuts, kiwi")
    assert brief.allergens == ["peanuts"]
    assert unk_a == ["kiwi"] and unk_d == ["keto"]
    msg = warning_text(unk_a, unk_d)
    assert "Not checked: kiwi" in msg and "keto" in msg


def test_canonical_names_pass_through_and_empty_input_is_fine():
    brief, unk_a, _ = parse_brief("", ", ".join(ALLERGENS))
    assert brief.allergens == list(ALLERGENS) and unk_a == []
    assert parse_brief(None, None)[0].allergens == []

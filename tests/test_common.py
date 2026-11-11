import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import common  # noqa: E402

LABELS = ["card_arrival", "lost_or_stolen_card", "refund_not_showing_up"]


def test_to_label_is_strict():
    assert common.to_label("card_arrival", LABELS) == "card_arrival"
    assert common.to_label("  `Card_Arrival`.\nextra", LABELS) == "card_arrival"
    assert common.to_label("my card has not arrived", LABELS) is None


def test_label_with_capital_letter_matches_canonical_form():
    # Banking77 has a label spelled 'Refund_not_showing_up' (capital R)
    labels = ["card_arrival", "Refund_not_showing_up"]
    assert common.to_label("Refund_not_showing_up", labels) == "Refund_not_showing_up"
    assert common.to_label("refund_not_showing_up", labels) == "Refund_not_showing_up"


def test_prompted_message_lists_labels_and_shots():
    m = common.user_msg_prompted("where is my card", LABELS, [("hi", "card_arrival")])
    assert "lost_or_stolen_card" in m and "Message: hi\nIntent: card_arrival" in m
    assert m.rstrip().endswith("Intent:")


def test_finetuned_message_has_no_label_list():
    assert "lost_or_stolen_card" not in common.user_msg_finetuned("where is my card")


class FakeSplit(dict):
    def select(self, idx):
        return {k: [v[i] for i in idx] for k, v in self.items()}


def test_stratified_sample_balanced_and_deterministic():
    split = FakeSplit(text=[f"t{i}" for i in range(30)], label=[i % 3 for i in range(30)])
    a = common.stratified_sample(split, 4, seed=1)
    b = common.stratified_sample(split, 4, seed=1)
    assert a == b
    assert sorted(a["label"]).count(0) == 4 and len(a["label"]) == 12

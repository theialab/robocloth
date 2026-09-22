"""sacct parsing for the B1 orchestrator: the whole run's progress tracking depends on it."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "leonardo_dataset" / "scripts" / "orchestrate_b1.py"
_spec = importlib.util.spec_from_file_location("orchestrate_b1", SCRIPT)
O = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(O)


def test_parse_running_and_finished_tasks():
    out = ("58355300_0|COMPLETED|04:12:33\n"
           "58355300_1|RUNNING|02:01:10\n"
           "58355300_2|FAILED|00:00:31\n"
           "58355300_3|TIMEOUT|08:00:05\n")
    assert O.parse_sacct(out) == {"0": "COMPLETED", "1": "RUNNING", "2": "FAILED", "3": "TIMEOUT"}


def test_parse_expands_pending_ranges_and_throttle():
    st = O.parse_sacct("58355300_[4-7,9]%16|PENDING|00:00:00\n")
    assert st == {k: "PENDING" for k in ("4", "5", "6", "7", "9")}


def test_parse_ignores_non_array_and_junk():
    assert O.parse_sacct("58355270|COMPLETED|00:12:00\n\ngarbage\n") == {}


def test_states_by_cancelled_carry_only_the_first_word():
    # sacct prints "CANCELLED by 12345"; the orchestrator compares against bare state names
    assert O.parse_sacct("1_0|CANCELLED by 30123|00:03:00\n") == {"0": "CANCELLED"}
    assert "CANCELLED" not in O.ACTIVE          # a cancelled task must count as inactive

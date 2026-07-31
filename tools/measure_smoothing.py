"""Measure Rule G2's eligible set on the REAL merged turns, after the timestamp fix.

G2: a turn is absorbed only if it is sandwiched (same speaker on both sides, a
different speaker itself), <=2 words, EXACTLY zero duration, and not a lexical
backchannel token.

Every candidate is dumped with its neighbours so each one can be judged by hand,
because the failure mode here is destroying real speech and a count cannot show
that. The hallucination span is reported SEPARATELY -- a prior revision of this
work built a stability claim on a contaminated replicate and had to retract it.
"""
import pickle
import re
from pathlib import Path

from audio_to_text.fusion import match_speakers, merge_turns, _shift_and_remap
from audio_to_text.transcribe import relabel_speakers, detect_repetition_loops

CAP = Path(__file__).parent / "capture" / "capture.pkl"

BACKCHANNEL = {
    "yeah", "yep", "yes", "yup", "mm-hmm", "-hmm", "mhm", "uh-huh", "ok", "okay",
    "no", "nope", "right", "sure", "correct", "exactly", "great", "cool", "thanks",
}


def key(text: str) -> str:
    return re.sub(r"[^a-z-]", "", text.lower())


def eligible(turns, i):
    if not (0 < i < len(turns) - 1):
        return False
    t = turns[i]
    if turns[i - 1]["speaker"] != turns[i + 1]["speaker"]:
        return False
    if turns[i - 1]["speaker"] == t["speaker"]:
        return False
    if len(t["text"].split()) > 2:
        return False
    if t["end"] - t["start"] != 0.0:
        return False
    return key(t["text"]) not in BACKCHANNEL


def main() -> None:
    cap = pickle.load(open(CAP, "rb"))
    a_to_b = match_speakers(cap["a"]["embeddings"], cap["b"]["embeddings"])
    turns_b = _shift_and_remap(cap["b"]["grouped"], cap["offset"], {v: k for k, v in a_to_b.items()})
    turns = relabel_speakers(merge_turns(cap["a"]["grouped"], turns_b))

    loops = detect_repetition_loops(turns)
    loop_spans = [(l["start"], l["end"]) for l in loops]
    def in_loop(t):
        return any(s <= t["start"] <= e for s, e in loop_spans)

    print(f"blocks={len(turns)}   hallucination spans={loop_spans}\n")

    cands = [i for i in range(len(turns)) if eligible(turns, i)]
    clean = [i for i in cands if not in_loop(turns[i])]
    dirty = [i for i in cands if in_loop(turns[i])]

    print(f"G2 eligible: {len(cands)} blocks ({len(cands)/len(turns):.1%} of {len(turns)})")
    print(f"   of which inside a hallucination span: {len(dirty)}")
    print(f"   CLEAN candidates to judge by hand:    {len(clean)}\n")

    # Breakdown of why the other short blocks were excluded -- shows how much of
    # G2's safety comes from the duration clause versus the (author-written) word list.
    short = [i for i in range(1, len(turns) - 1)
             if len(turns[i]["text"].split()) <= 2
             and turns[i - 1]["speaker"] == turns[i + 1]["speaker"]
             and turns[i - 1]["speaker"] != turns[i]["speaker"]]
    by_duration = [i for i in short if turns[i]["end"] - turns[i]["start"] != 0.0]
    by_wordlist = [i for i in short if turns[i]["end"] - turns[i]["start"] == 0.0
                   and key(turns[i]["text"]) in BACKCHANNEL]
    print(f"sandwiched <=2-word blocks: {len(short)}")
    print(f"   excluded by DURATION clause: {len(by_duration)}")
    print(f"   excluded by WORD LIST only:  {len(by_wordlist)}  "
          f"({[turns[i]['text'] for i in by_wordlist]})\n")

    print("=" * 78)
    print("EVERY CLEAN CANDIDATE, WITH CONTEXT -- judge each by hand")
    print("=" * 78)
    for i in clean:
        p, t, n = turns[i - 1], turns[i], turns[i + 1]
        print(f"\n[{i}] absorb {t['text']!r}  ({t['speaker']}, dur=0, "
              f"t={t['start']:.1f}s)  into {p['speaker']}")
        print(f"    prev {p['speaker']}: ...{p['text'][-70:]!r}")
        print(f"    THIS {t['speaker']}: {t['text']!r}")
        print(f"    next {n['speaker']}: {n['text'][:70]!r}...")


if __name__ == "__main__":
    main()

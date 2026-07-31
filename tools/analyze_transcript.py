"""Measure quality defects in a rendered transcript.

Reproduces the numbers cited in docs/validate/2026-07-31-bugs-md-triage.md so they
can be checked rather than taken on trust.

    uv run python tools/analyze_transcript.py "output/<name>.md"

Reports:
  * redundancy   -- near-duplicate text between blocks, scanned at block distances
                    1..5 and split by same-speaker vs cross-speaker. Distance > 1
                    matters: _group_consecutive flushes on every speaker change, so
                    two same-speaker turns are ALWAYS separated by at least one
                    intervening turn. An adjacent-only scan cannot see the
                    same-speaker case at all.
  * fragmentation -- share of blocks carrying only a word or two, and how many are
                    "sandwiched" (bracketed by the same speaker on both sides), which
                    is the subset a merge could absorb without crossing a real
                    speaker boundary.
  * backchannel   -- of the shortest blocks, how many are lexical agreement tokens
                    ("yeah", "mm-hmm"). These are genuine turns, not jitter, so they
                    bound how aggressive any smoothing rule may safely be.
  * degeneration  -- Whisper repetition loops, counted BOTH within a block and
                    DOC-WIDE across block boundaries. The doc-wide pass matters: with
                    ~31% of blocks holding a single word, a repetition loop is shredded
                    across dozens of blocks and is invisible to a within-block scan.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

BLOCK_RE = re.compile(r"^## (Person \d+) — ([\d:]+)\n\n(.*?)(?=\n## |\Z)", re.S | re.M)

# Agreement/acknowledgement tokens that are real one-word speaker turns in a meeting,
# as opposed to function words stolen from a neighbour's sentence by boundary jitter.
BACKCHANNEL = {
    "yeah", "yep", "yes", "yup", "mm-hmm", "-hmm", "mhm", "uh-huh", "ok", "okay",
    "no", "nope", "right", "sure", "correct", "exactly", "great", "cool", "thanks",
}

MIN_SHARED_CHARS = 40
MIN_SHARED_FRACTION = 0.5
MAX_DISTANCE = 5


def parse_blocks(path: Path) -> list[dict]:
    return [
        {"speaker": m.group(1), "time": m.group(2), "text": m.group(3).strip()}
        for m in BLOCK_RE.finditer(path.read_text(encoding="utf-8"))
    ]


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower())


def _seconds(stamp: str) -> int:
    parts = [int(x) for x in stamp.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def longest_shared(a: str, b: str) -> int:
    if not a or not b:
        return 0
    match = SequenceMatcher(None, a, b, autojunk=False).find_longest_match(0, len(a), 0, len(b))
    return match.size


def scan_redundancy(blocks: list[dict], max_distance: int = MAX_DISTANCE) -> dict[int, list[dict]]:
    """Near-duplicate block pairs, keyed by block distance."""
    found: dict[int, list[dict]] = {}
    for distance in range(1, max_distance + 1):
        hits = []
        for i in range(len(blocks) - distance):
            a, b = blocks[i], blocks[i + distance]
            text_a, text_b = _normalize(a["text"]), _normalize(b["text"])
            shared = longest_shared(text_a, text_b)
            if not text_a or not text_b:
                continue
            if shared > MIN_SHARED_CHARS and shared > MIN_SHARED_FRACTION * min(len(text_a), len(text_b)):
                hits.append({
                    "index": i,
                    "speaker_a": a["speaker"],
                    "speaker_b": b["speaker"],
                    "time_a": a["time"],
                    "time_b": b["time"],
                    "delta_seconds": _seconds(b["time"]) - _seconds(a["time"]),
                    "shared_chars": shared,
                    "same_speaker": a["speaker"] == b["speaker"],
                })
        found[distance] = hits
    return found


def report_redundancy(blocks: list[dict]) -> None:
    print("== REDUNDANCY (near-duplicate text between blocks) ==")
    print(f"   criterion: shared substring > {MIN_SHARED_CHARS} chars AND "
          f"> {MIN_SHARED_FRACTION:.0%} of the shorter block\n")
    results = scan_redundancy(blocks)
    total_same = total_cross = 0
    for distance, hits in results.items():
        same = sum(1 for h in hits if h["same_speaker"])
        cross = len(hits) - same
        total_same += same
        total_cross += cross
        print(f"   distance={distance}: SAME-speaker={same:>3}  cross-speaker={cross:>3}")
        for h in hits:
            tag = "SAME " if h["same_speaker"] else "cross"
            print(f"        {tag} idx={h['index']:>4} {h['speaker_a']}->{h['speaker_b']} "
                  f"{h['time_a']}->{h['time_b']} Δ={h['delta_seconds']:>4}s "
                  f"shared={h['shared_chars']:>4}")
    print(f"\n   TOTAL across distances 1..{MAX_DISTANCE}: "
          f"SAME-speaker={total_same}  cross-speaker={total_cross}")
    print("   (same-speaker pairs are invisible to a distance-1-only scan -- see module docstring)\n")


def report_fragmentation(blocks: list[dict]) -> None:
    print("== FRAGMENTATION ==")
    speakers = [b["speaker"] for b in blocks]
    counts = [len(b["text"].split()) for b in blocks]
    total_words = sum(counts)
    print(f"   blocks={len(blocks)}  words={total_words}")
    for k in (1, 2, 3, 5):
        idx = [i for i, c in enumerate(counts) if c <= k]
        sandwiched = [
            i for i in idx
            if 0 < i < len(blocks) - 1 and speakers[i - 1] == speakers[i + 1]
        ]
        carried = sum(counts[i] for i in idx)
        print(f"   <= {k} word(s): {len(idx):>3} blocks ({len(idx)/len(blocks):>4.0%}) "
              f"carrying {carried:>4} words ({carried/total_words:>4.1%} of text) | "
              f"sandwiched by same speaker: {len(sandwiched):>3} "
              f"({len(sandwiched)/max(1,len(idx)):>3.0%})")

    adjacent_same = sum(1 for i in range(len(blocks) - 1) if speakers[i] == speakers[i + 1])
    print(f"\n   adjacent SAME-speaker block pairs: {adjacent_same}")
    print("   (_group_consecutive can never emit these -- they indicate turns appended")
    print("    post-grouping, i.e. by fusion's merge_turns gap-fill, never re-grouped)\n")


def report_backchannel(blocks: list[dict]) -> None:
    print("== BACKCHANNEL vs JITTER (among <=2-word blocks) ==")
    short = [b for b in blocks if len(b["text"].split()) <= 2]
    def key(b):
        return re.sub(r"[^a-z-]", "", b["text"].lower())
    real = [b for b in short if key(b) in BACKCHANNEL]
    other = [b for b in short if key(b) not in BACKCHANNEL]
    print(f"   short blocks: {len(short)}")
    print(f"   lexical backchannel (GENUINE turns -- destroying these misattributes speech): "
          f"{len(real)} ({len(real)/max(1,len(short)):.0%})")
    print(f"   other/likely jitter: {len(other)} ({len(other)/max(1,len(short)):.0%})")
    print(f"   most common backchannel: {Counter(key(b) for b in real).most_common(8)}")
    print(f"   most common other:       {Counter(key(b) for b in other).most_common(8)}")
    print("   NOTE: a length threshold alone cannot separate these two populations.\n")


def report_degeneration(blocks: list[dict]) -> None:
    """Whisper repetition loops, within-block AND doc-wide.

    A within-block scan alone is blind to the fragmented case: when a loop is split
    across many one-word blocks (see report_fragmentation), no single block contains
    a repeat. The doc-wide pass flattens the transcript to a token stream first.
    """
    print("== WHISPER DEGENERATION (repetition loops) ==")

    within = 0
    for b in blocks:
        w = [x.lower().strip(".,?!") for x in b["text"].split()]
        if any(len(set(w[k:k + 4])) == 1 and w[k] for k in range(len(w) - 3)):
            within += 1
    print(f"   blocks containing 4+ identical consecutive words: {within}")

    tokens = [x.lower().strip(".,?!\"'") for b in blocks for x in b["text"].split()]
    runs: list[tuple[str, int]] = []
    i = 0
    while i < len(tokens):
        j = i
        while j + 1 < len(tokens) and tokens[j + 1] == tokens[i]:
            j += 1
        if j - i + 1 >= 4 and tokens[i]:
            runs.append((tokens[i], j - i + 1))
        i = j + 1
    print(f"   DOC-WIDE runs of 4+ identical consecutive tokens: {len(runs)}")
    for tok, n in sorted(runs, key=lambda r: -r[1])[:6]:
        print(f"        {tok!r} x{n}")

    total = Counter(tokens)
    if total:
        tok, n = total.most_common(1)[0]
        print(f"   most frequent token overall: {tok!r} x{n} ({n/len(tokens):.1%} of all words)")
    # A single token dominating the transcript is the signature of a loop that
    # fragmentation has scattered rather than kept contiguous.
    suspicious = [(t, n) for t, n in total.most_common(40)
                  if n > 50 and t not in {
                      "the", "and", "a", "to", "of", "i", "it", "that", "you", "in", "is",
                      "so", "we", "yeah", "on", "for", "this", "but", "have", "with",
                      "be", "not", "they", "just", "can", "do", "at", "if", "or", "as",
                      "what", "was", "are", "there", "then", "all", "like", "up", "out",
                      "one", "get", "go", "know", "think", "well", "right", "okay", "no",
                      "he", "she", "will", "would", "from", "your", "my", "me", "us",
                      "them", "how", "when", "which", "who", "about", "some", "more",
                      "our", "their", "an", "by", "has", "had", "were", "been", "its",
                  }]
    if suspicious:
        print(f"   high-frequency non-stopword tokens (possible loop): {suspicious[:6]}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("transcript", type=Path, help="A rendered .md transcript.")
    args = parser.parse_args()

    blocks = parse_blocks(args.transcript)
    if not blocks:
        print(f"error: no '## Person N — mm:ss' blocks found in {args.transcript}")
        return 1

    print(f"\nTranscript: {args.transcript.name}")
    print(f"Speakers:   {Counter(b['speaker'] for b in blocks).most_common()}\n")
    report_redundancy(blocks)
    report_fragmentation(blocks)
    report_backchannel(blocks)
    report_degeneration(blocks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

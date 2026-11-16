"""Fighting-pattern engine (Phase 4). Pure functions over action observations — no database, no video.

Input: a list of ``Obs`` (one per recognised or annotated action). The engine
  1. cleans them (ordering, overlap de-duplication, duplicate-footage removal),
  2. splits each fighter's actions per video into *sequences* (bursts separated by pauses),
  3. computes combinations (contiguous n-grams), sequential patterns (PrefixSpan),
     conditional probabilities under three explicitly separate definitions, Markov
     transition matrices (orders 1-3, additive smoothing), tendencies, opponent-dependent
     behaviour and fighter comparisons — every number with its sample size, a Wilson
     interval where it is a proportion, and a reliability label.

Definitions (never mixed):
  * ``next``            : the next action of the same fighter *in the same sequence*; END if the sequence ends.
  * ``next_offensive``  : the next *offensive* action in the same sequence; END if none follows.
  * ``window``          : the next action of the same fighter starting within ``window`` seconds after the
                          previous action ends (same video); NONE if nothing starts in that window.
"""
from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from app.actions.taxonomy import group as action_group
from app.actions.taxonomy import label
from app.patterns.stats import binom_greater, chi_square, entropy, js_distance, reliability, wilson

END, NONE, START = "END", "NONE", "START"


@dataclass
class Obs:
    fighter_key: str
    video_id: str
    fight_key: str
    category: str
    start: float
    end: float
    confidence: float = 1.0
    source: str = "detection"
    action_id: str = ""
    track_id: int | None = None
    fighter_id: str | None = None
    opponent_key: str | None = None
    opponent_id: str | None = None
    distance: float | None = None  # torso lengths, if known
    stance: str | None = None
    fight_date: str | None = None
    # filled by prepare()
    seq_id: str = ""
    seq_pos: int = 0
    interval_prev: float | None = None
    opponent_action: str | None = None
    opponent_action_lag: float | None = None

    @property
    def group(self) -> str:
        return action_group(self.category)

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class PatternConfig:
    sequence_gap: float = 2.0  # new sequence after this pause (s)
    window: float = 1.5  # "within window" definition (s)
    overlap_iou: float = 0.5  # same-fighter detections overlapping more than this are duplicates
    min_occurrences: int = 2  # combinations / patterns shown only from this count
    min_fights: int = 1  # ... and in at least this many fights (2 = only cross-fight patterns)
    min_observations: int = 5  # probabilities based on fewer observations are flagged insufficient
    max_n: int = 5  # longest combination length
    smoothing_alpha: float = 0.5  # additive (Lidstone) smoothing for Markov matrices
    max_order: int = 3
    min_context_count: int = 5  # higher-order contexts need this many observations
    significance: float = 0.05
    opponent_reaction_window: float = 1.0  # fighter's response to an opponent attack must start within this (s)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Prepared:
    obs: list[Obs]
    sequences: dict[str, list[Obs]]  # seq_id -> ordered actions
    by_fighter: dict[str, list[Obs]]
    stats: dict = field(default_factory=dict)


# ============================================================================ preparation
def _iou(a: Obs, b: Obs) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def prepare(obs: list[Obs], cfg: PatternConfig) -> Prepared:
    stats = Counter()
    stats["input_actions"] = len(obs)
    # 1. order & remove overlapping duplicate detections of the same fighter
    streams: dict[tuple[str, str], list[Obs]] = defaultdict(list)
    for o in obs:
        if o.end <= o.start:
            stats["invalid_timestamps"] += 1
            continue
        streams[(o.fighter_key, o.video_id)].append(o)
    cleaned: dict[tuple[str, str], list[Obs]] = {}
    for key, items in streams.items():
        items.sort(key=lambda o: (o.start, o.end))
        kept: list[Obs] = []
        for o in items:
            dup = next((k for k in kept[-3:] if _iou(o, k) > cfg.overlap_iou), None)
            if dup is None:
                kept.append(o)
            else:
                stats["overlapping_duplicates_removed"] += 1
                if o.confidence > dup.confidence:  # keep the more confident detection
                    kept[kept.index(dup)] = o
        cleaned[key] = sorted(kept, key=lambda o: o.start)

    # 2. split into sequences (bursts)
    sequences: dict[str, list[Obs]] = {}
    for (fk, vid), items in cleaned.items():
        n = 0
        cur: list[Obs] = []
        prev_end = None
        for o in items:
            if cur and o.start - prev_end > cfg.sequence_gap:
                sequences[f"{fk}|{vid}|{n}"] = cur
                n += 1
                cur = []
            cur.append(o)
            prev_end = max(prev_end, o.end) if prev_end is not None and cur[:-1] else o.end
        if cur:
            sequences[f"{fk}|{vid}|{n}"] = cur

    # 3. duplicate footage: the same fight uploaded as overlapping videos produces identical sequences
    seen: dict[tuple, str] = {}
    for sid in sorted(sequences, key=lambda s: (sequences[s][0].fight_key, sequences[s][0].video_id, sequences[s][0].start)):
        seq = sequences[sid]
        if len(seq) < 2:
            continue
        sig = (seq[0].fighter_key, seq[0].fight_key, tuple(o.category for o in seq),
               tuple(round(seq[i + 1].start - seq[i].start, 1) for i in range(len(seq) - 1)))
        other = seen.get(sig)
        if other and sequences[other][0].video_id != seq[0].video_id:
            del sequences[sid]
            stats["duplicate_footage_sequences_removed"] += 1
        else:
            seen[sig] = sid

    # 4. annotate positions, intervals and opponent context
    by_video: dict[str, list[Obs]] = defaultdict(list)
    for sid, seq in sequences.items():
        for i, o in enumerate(seq):
            o.seq_id, o.seq_pos = sid, i
            o.interval_prev = round(o.start - seq[i - 1].end, 4) if i else None
            by_video[o.video_id].append(o)
    for items in by_video.values():
        items.sort(key=lambda o: o.start)
        for o in items:
            if not o.opponent_key:
                continue
            # opponent's most recent action that started before this one (within 1.5 s), or overlaps it
            best = None
            for p in items:
                if p.fighter_key != o.opponent_key:
                    continue
                if p.start <= o.start + 0.05 and o.start - p.start <= 1.5:
                    if best is None or p.start > best.start:
                        best = p
            if best is not None:
                o.opponent_action = best.category
                o.opponent_action_lag = round(o.start - best.start, 3)

    flat = [o for seq in sequences.values() for o in seq]
    by_fighter: dict[str, list[Obs]] = defaultdict(list)
    for o in flat:
        by_fighter[o.fighter_key].append(o)
    for v in by_fighter.values():
        v.sort(key=lambda o: (o.fight_key, o.video_id, o.start))
    stats["actions_used"] = len(flat)
    stats["sequences"] = len(sequences)
    return Prepared(flat, sequences, dict(by_fighter), dict(stats))


def fighter_sequences(p: Prepared, fighter_key: str) -> list[list[Obs]]:
    return [s for s in p.sequences.values() if s and s[0].fighter_key == fighter_key]


# ============================================================================ combinations (contiguous n-grams)
def _baseline_next(seqs: list[list[Obs]]) -> Counter:
    """Distribution of actions in non-initial positions (what can follow anything)."""
    c = Counter()
    for s in seqs:
        for o in s[1:]:
            c[o.category] += 1
    return c


def combinations(seqs: list[list[Obs]], cfg: PatternConfig, fights_observed: int | None = None) -> list[dict]:
    fights_observed = fights_observed or len({s[0].fight_key for s in seqs}) or 1
    base = _baseline_next(seqs)
    base_tot = sum(base.values())
    occ: dict[tuple, list[tuple[list[Obs], int]]] = defaultdict(list)
    for s in seqs:
        cats = [o.category for o in s]
        for n in range(2, cfg.max_n + 1):
            for i in range(len(cats) - n + 1):
                occ[tuple(cats[i:i + n])].append((s, i))
    # all occurrences of each prefix: confidence = P(last | prefix) = Count(prefix -> last) / Count(prefix)
    prefix_followed: Counter = Counter()
    for s in seqs:
        cats = [o.category for o in s]
        for n in range(1, cfg.max_n):
            for i in range(len(cats) - n + 1):
                prefix_followed[tuple(cats[i:i + n])] += 1

    out = []
    for key, hits in occ.items():
        count = len(hits)
        fights = {s[0].fight_key for s, _ in hits}
        if count < cfg.min_occurrences or len(fights) < cfg.min_fights:
            continue
        n = len(key)
        intervals, durations, confs, followed = [], [], [], 0
        for s, i in hits:
            seg = s[i:i + n]
            intervals += [max(0.0, seg[j + 1].start - seg[j].end) for j in range(n - 1)]
            durations.append(seg[-1].end - seg[0].start)
            confs += [o.confidence for o in seg]
            followed += 1 if i + n < len(s) else 0
        pref_n = prefix_followed[key[:-1]]
        conf_val = count / pref_n if pref_n else None
        p0 = base[key[-1]] / base_tot if base_tot else None
        pval = binom_greater(count, pref_n, p0) if p0 and pref_n else None
        lift = (conf_val / p0) if conf_val is not None and p0 else None
        single = len(fights) == 1
        if single:
            cls = "single_fight"
        elif pval is not None and pval < cfg.significance and lift and lift > 1:
            cls = "repeated_pattern"
        else:
            cls = "not_distinguishable_from_chance"
        out.append({
            "sequence": list(key), "labels": [label(k) for k in key], "length": n,
            "occurrences": count, "distinct_fights": len(fights), "distinct_videos": len({s[0].video_id for s, _ in hits}),
            "frequency_per_fight": round(count / fights_observed, 3),
            "avg_interval": round(statistics.fmean(intervals), 3) if intervals else None,
            "avg_duration": round(statistics.fmean(durations), 3),
            "avg_confidence": round(statistics.fmean(confs), 4),
            "followed_by_action_pct": round(followed / count, 4),
            "confidence": round(conf_val, 4) if conf_val is not None else None,
            "confidence_n": pref_n,
            "confidence_ci95": wilson(count, pref_n) if pref_n else None,
            "lift": round(lift, 3) if lift else None, "p_value": round(pval, 5) if pval is not None else None,
            "classification": cls,
        })
    out.sort(key=lambda d: (-d["distinct_fights"], -d["occurrences"], -d["length"]))
    return out


# ============================================================================ PrefixSpan (gapped sequential patterns)
def prefixspan(seqs: list[list[Obs]], cfg: PatternConfig, min_support_count: int | None = None) -> list[dict]:
    """Sequential patterns (elements in order, gaps allowed) occurring in >= min support sequences.

    Support counts each sequence once; leftmost embeddings are used for timing statistics.
    """
    db = [[o.category for o in s] for s in seqs]
    minsup = min_support_count or cfg.min_occurrences
    results: list[tuple[tuple, list[tuple[int, list[int]]]]] = []

    def grow(prefix: tuple, projected: list[tuple[int, list[int]]]):
        # projected: (sequence index, matched positions of prefix)
        ext: dict[str, list[tuple[int, list[int]]]] = defaultdict(list)
        for si, pos in projected:
            start = pos[-1] + 1 if pos else 0
            seen = set()
            for j in range(start, len(db[si])):
                c = db[si][j]
                if c not in seen:
                    seen.add(c)
                    ext[c].append((si, pos + [j]))
        for c, proj in ext.items():
            if len(proj) >= minsup:
                pat = prefix + (c,)
                if len(pat) >= 2:
                    results.append((pat, proj))
                if len(pat) < cfg.max_n:
                    grow(pat, proj)

    grow((), [(i, []) for i in range(len(db))])
    total = len(db) or 1
    support_of: dict[tuple, int] = {pat: len(proj) for pat, proj in results}
    first = Counter()
    for s in db:
        for c in set(s):
            first[c] += 1
    out = []
    for pat, proj in results:
        fights = {seqs[si][0].fight_key for si, _ in proj}
        if len(fights) < cfg.min_fights:
            continue
        gaps = []
        for si, pos in proj:
            s = seqs[si]
            gaps += [max(0.0, s[pos[k + 1]].start - s[pos[k]].end) for k in range(len(pos) - 1)]
        prefix_sup = support_of.get(pat[:-1]) if len(pat) > 2 else first[pat[0]]
        out.append({
            "pattern": list(pat), "labels": [label(k) for k in pat], "length": len(pat),
            "support": round(len(proj) / total, 4), "sequences_containing": len(proj), "distinct_fights": len(fights),
            "confidence": round(len(proj) / prefix_sup, 4) if prefix_sup else None,
            "avg_time_between": round(statistics.fmean(gaps), 3) if gaps else None,
            "contiguous_only": all(all(pos[k + 1] == pos[k] + 1 for k in range(len(pos) - 1)) for _, pos in proj),
            "single_fight": len(fights) == 1,
        })
    out.sort(key=lambda d: (-d["distinct_fights"], -d["sequences_containing"], -d["length"]))
    return out


# ============================================================================ conditional probabilities
def _pairs(seqs: list[list[Obs]], definition: str, order: int, cfg: PatternConfig) -> list[tuple[tuple, str, str]]:
    """(context, outcome, fight_key) observations for one definition and Markov order."""
    out = []
    for s in seqs:
        cats = [o.category for o in s]
        for i in range(order - 1, len(s)):
            ctx = tuple(cats[i - order + 1:i + 1])
            if definition == "next":
                nxt = cats[i + 1] if i + 1 < len(s) else END
            elif definition == "next_offensive":
                nxt = next((o.category for o in s[i + 1:] if o.group == "offensive"), END)
            else:
                raise ValueError(definition)
            out.append((ctx, nxt, s[0].fight_key))
    return out


def _window_pairs(fighter_obs: list[Obs], order: int, cfg: PatternConfig) -> list[tuple[tuple, str, str]]:
    """Window definition: next action starting within cfg.window s after the previous one ends (same video)."""
    by_vid: dict[str, list[Obs]] = defaultdict(list)
    for o in fighter_obs:
        by_vid[o.video_id].append(o)
    out = []
    for items in by_vid.values():
        items.sort(key=lambda o: o.start)
        cats = [o.category for o in items]
        for i in range(order - 1, len(items)):
            ctx = tuple(cats[i - order + 1:i + 1])
            # context actions must themselves be chained within the window
            if any(items[j + 1].start - items[j].end > cfg.window for j in range(i - order + 1, i)):
                continue
            nxt = NONE
            if i + 1 < len(items) and items[i + 1].start - items[i].end <= cfg.window:
                nxt = cats[i + 1]
            out.append((ctx, nxt, items[i].fight_key))
    return out


def conditional_probabilities(fighter_obs: list[Obs], seqs: list[list[Obs]], definition: str, order: int,
                              cfg: PatternConfig) -> list[dict]:
    pairs = _window_pairs(fighter_obs, order, cfg) if definition == "window" else _pairs(seqs, definition, order, cfg)
    ctx_n: Counter = Counter()
    ctx_fights: dict[tuple, set] = defaultdict(set)
    joint: Counter = Counter()
    joint_fights: dict[tuple, set] = defaultdict(set)
    for ctx, nxt, fk in pairs:
        ctx_n[ctx] += 1
        ctx_fights[ctx].add(fk)
        joint[(ctx, nxt)] += 1
        joint_fights[(ctx, nxt)].add(fk)
    total = len(pairs) or 1
    out = []
    for (ctx, nxt), k in joint.items():
        n = ctx_n[ctx]
        ci = wilson(k, n)
        out.append({
            "definition": definition, "order": order, "context": list(ctx), "context_labels": [label(c) for c in ctx],
            "next": nxt, "next_label": label(nxt) if nxt not in (END, NONE) else ("End of sequence" if nxt == END else f"No action within {cfg.window}s"),
            "count": k, "context_count": n, "probability": round(k / n, 4), "ci95": ci,
            "fights": len(joint_fights[(ctx, nxt)]), "context_fights": len(ctx_fights[ctx]),
            "coverage": round(n / total, 4),
            "reliability": reliability(n, len(ctx_fights[ctx]), ci, cfg.min_observations if order == 1 else cfg.min_context_count),
            "window_seconds": cfg.window if definition == "window" else None,
        })
    out.sort(key=lambda d: (d["context"], -d["count"]))
    return out


# ============================================================================ Markov chains
def markov(seqs: list[list[Obs]], cfg: PatternConfig, order: int = 1, states: list[str] | None = None) -> dict:
    """Transition matrix P(next | previous `order` actions) with additive smoothing; END is an absorbing outcome."""
    pairs = _pairs(seqs, "next", order, cfg)
    states = states or sorted({c for s in seqs for c in (o.category for o in s)})
    cols = states + [END]
    V = len(cols)
    a = cfg.smoothing_alpha
    ctx_n: Counter = Counter(c for c, _, _ in pairs)
    joint: Counter = Counter((c, n) for c, n, _ in pairs)
    fights: dict[tuple, set] = defaultdict(set)
    for c, _, fk in pairs:
        fights[c].add(fk)
    rows = []
    for ctx in sorted(ctx_n, key=lambda c: (-ctx_n[c], c)):
        n = ctx_n[ctx]
        if order > 1 and n < cfg.min_context_count:
            continue
        cells = []
        for col in cols:
            k = joint[(ctx, col)]
            cells.append({"next": col, "count": k, "raw": round(k / n, 4) if n else 0.0,
                          "smoothed": round((k + a) / (n + a * V), 4), "ci95": wilson(k, n)})
        best = max(cells, key=lambda c: c["count"])
        rows.append({"context": list(ctx), "context_labels": [label(c) for c in ctx], "n": n, "fights": len(fights[ctx]),
                     "most_likely_next": best["next"] if best["count"] else None, "cells": cells,
                     "reliability": reliability(n, len(fights[ctx]), wilson(best["count"], n), cfg.min_observations)})
    trans = sorted(((c, nx, k) for (c, nx), k in joint.items()), key=lambda t: -t[2])
    starts = Counter(s[0].category for s in seqs if s)
    return {
        "order": order, "states": states, "columns": cols, "column_labels": [label(c) if c != END else "End" for c in cols],
        "smoothing": {"method": "additive (Lidstone)", "alpha": a}, "rows": rows,
        "start_distribution": [{"action": k, "label": label(k), "count": v, "probability": round(v / max(1, len(seqs)), 4)} for k, v in starts.most_common()],
        "most_common": [{"from": list(c), "to": nx, "count": k, "probability": round(k / ctx_n[c], 4)} for c, nx, k in trans[:10]],
        "rare": [{"from": list(c), "to": nx, "count": k, "probability": round(k / ctx_n[c], 4)} for c, nx, k in trans if k == 1][:15],
        "observations": len(pairs),
        "entropy_rate_bits": round(sum(ctx_n[c] / max(1, len(pairs)) * entropy(Counter({nx: joint[(c, nx)] for nx in cols})) for c in ctx_n), 4),
        "insufficient_contexts": sum(1 for c in ctx_n if order > 1 and ctx_n[c] < cfg.min_context_count),
    }


# ============================================================================ tendencies
def _minutes(fighter_obs: list[Obs], footage: dict[str, float]) -> float:
    vids = {o.video_id for o in fighter_obs}
    return sum(footage.get(v, 0.0) for v in vids) / 60.0


def tendencies(fighter_obs: list[Obs], seqs: list[list[Obs]], cfg: PatternConfig, footage: dict[str, float],
               all_obs: list[Obs]) -> dict:
    n = len(fighter_obs)
    fights = sorted({o.fight_key for o in fighter_obs})
    minutes = _minutes(fighter_obs, footage)
    groups = Counter(o.group for o in fighter_obs)
    off = [o for o in fighter_obs if o.group == "offensive"]
    dfn = [o for o in fighter_obs if o.group == "defensive"]

    def dist(items: list[Obs]) -> list[dict]:
        c = Counter(o.category for o in items)
        tot = sum(c.values())
        return [{"action": k, "label": label(k), "count": v, "share": round(v / tot, 4), "ci95": wilson(v, tot),
                 "per_minute": round(v / minutes, 3) if minutes else None,
                 "fights": len({o.fight_key for o in items if o.category == k})} for k, v in c.most_common()]

    combos = combinations(seqs, cfg, len(fights))
    off_seq_lengths = [sum(1 for o in s if o.group == "offensive") for s in seqs if any(o.group == "offensive" for o in s)]
    intervals = [o.interval_prev for o in fighter_obs if o.interval_prev is not None]

    # defensive reactions after specific opponent attacks
    opp_keys = {o.opponent_key for o in fighter_obs if o.opponent_key}
    opp_attacks = [p for p in all_obs if p.fighter_key in opp_keys and p.group == "offensive"
                   and p.video_id in {o.video_id for o in fighter_obs}]
    reactions = reaction_table(fighter_obs, opp_attacks, cfg)

    evasions = sum(1 for o in dfn if o.category in ("slip", "duck", "backward_evasion", "parry"))
    counters = sum(1 for o in fighter_obs if o.category == "counterattack")
    counter_after_attack = sum(r["offensive_responses"] for r in reactions["by_attack"])
    total_attacks = reactions["opponent_attacks"]

    # behaviour across fights
    per_fight = {}
    for fk in fights:
        c = Counter(o.category for o in fighter_obs if o.fight_key == fk)
        per_fight[fk] = c
    overall = Counter(o.category for o in fighter_obs)
    changes = [{"fight_key": fk, "actions": sum(c.values()), "js_distance_from_overall": js_distance(c, overall),
                "top": [label(k) for k, _ in c.most_common(3)]} for fk, c in per_fight.items()]
    cats = sorted(overall)
    homog = chi_square([[per_fight[fk].get(c, 0) for c in cats] for fk in fights]) if len(fights) >= 2 else \
        {"valid": False, "reason": "only one fight analysed"}

    return {
        "actions": n, "fights": len(fights), "minutes_observed": round(minutes, 2),
        "single_fight_only": len(fights) <= 1,
        "group_counts": dict(groups),
        "offensive": {
            "most_used": dist(off), "attack_frequency_per_minute": round(len(off) / minutes, 3) if minutes else None,
            "share_of_actions": round(len(off) / n, 4) if n else None,
            "average_combination_length": round(statistics.fmean(off_seq_lengths), 3) if off_seq_lengths else None,
            "top_combinations": [c for c in combos if all(action_group(a) == "offensive" for a in c["sequence"])][:10],
            "preferred_attacking_sequences": [c for c in combos if action_group(c["sequence"][0]) == "offensive"][:10],
        },
        "defensive": {
            "most_used": dist(dfn), "share_of_actions": round(len(dfn) / n, 4) if n else None,
            "evasions": evasions, "evasion_rate_per_opponent_attack": round(evasions / total_attacks, 4) if total_attacks else None,
            "counterattacks_labelled": counters,
            "counterattack_rate": round(counter_after_attack / total_attacks, 4) if total_attacks else None,
            "counterattack_ci95": wilson(counter_after_attack, total_attacks) if total_attacks else None,
            "reactions_to_opponent_attacks": reactions,
        },
        "behavioral": {
            "repeated_sequences": [c for c in combos if c["classification"] == "repeated_pattern"][:10],
            "typical_interval": {"median": round(statistics.median(intervals), 3), "p25": round(_q(intervals, 0.25), 3),
                                 "p75": round(_q(intervals, 0.75), 3), "n": len(intervals)} if intervals else None,
            "changes_across_fights": changes, "homogeneity_test": homog,
            "sequence_diversity": diversity(seqs),
        },
    }


def _q(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return math.nan
    k = (len(s) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def diversity(seqs: list[list[Obs]]) -> dict:
    uni = Counter(o.category for s in seqs for o in s)
    bi = Counter((s[i].category, s[i + 1].category) for s in seqs for i in range(len(s) - 1))
    return {"distinct_actions": len(uni), "action_entropy_bits": round(entropy(uni), 4),
            "distinct_transitions": len(bi), "transition_entropy_bits": round(entropy(bi), 4),
            "distinct_transition_ratio": round(len(bi) / sum(bi.values()), 4) if bi else None}


def reaction_table(fighter_obs: list[Obs], opp_attacks: list[Obs], cfg: PatternConfig) -> dict:
    """For each opponent attack: the fighter's first action starting within the reaction window."""
    by_vid: dict[str, list[Obs]] = defaultdict(list)
    for o in fighter_obs:
        by_vid[o.video_id].append(o)
    for v in by_vid.values():
        v.sort(key=lambda o: o.start)
    per_attack: dict[str, Counter] = defaultdict(Counter)
    fights: dict[str, set] = defaultdict(set)
    for a in opp_attacks:
        resp = next((o for o in by_vid.get(a.video_id, []) if a.start - 0.1 <= o.start <= a.end + cfg.opponent_reaction_window), None)
        per_attack[a.category][resp.category if resp else NONE] += 1
        fights[a.category].add(a.fight_key)
    rows = []
    for atk, c in sorted(per_attack.items(), key=lambda kv: -sum(kv[1].values())):
        n = sum(c.values())
        defensive = sum(v for k, v in c.items() if k != NONE and action_group(k) == "defensive")
        offensive = sum(v for k, v in c.items() if k != NONE and action_group(k) == "offensive")
        rows.append({
            "opponent_attack": atk, "label": label(atk), "n": n, "fights": len(fights[atk]),
            "responses": [{"action": k, "label": label(k) if k != NONE else "No recognised response", "count": v,
                           "probability": round(v / n, 4), "ci95": wilson(v, n)} for k, v in c.most_common()],
            "defensive_responses": defensive, "offensive_responses": offensive,
            "reliability": reliability(n, len(fights[atk]), wilson(max(c.values()), n), cfg.min_observations),
        })
    return {"opponent_attacks": len(opp_attacks), "window_seconds": cfg.opponent_reaction_window, "by_attack": rows}


# ============================================================================ explorer, opponent analysis, comparison
def explorer(seqs: list[list[Obs]], action: str, cfg: PatternConfig) -> dict:
    before, after = Counter(), Counter()
    occ = 0
    fights = set()
    for s in seqs:
        for i, o in enumerate(s):
            if o.category != action:
                continue
            occ += 1
            fights.add(s[0].fight_key)
            before[s[i - 1].category if i else START] += 1
            after[s[i + 1].category if i + 1 < len(s) else END] += 1

    def dist(c: Counter):
        return [{"action": k, "label": label(k) if k not in (START, END) else ("Sequence start" if k == START else "Sequence end"),
                 "count": v, "probability": round(v / occ, 4), "ci95": wilson(v, occ)} for k, v in c.most_common()] if occ else []

    def grams(n: int):
        c, f = Counter(), defaultdict(set)
        for s in seqs:
            cats = [o.category for o in s]
            for i in range(len(cats) - n + 1):
                g = tuple(cats[i:i + n])
                if action in g:
                    c[g] += 1
                    f[g].add(s[0].fight_key)
        return [{"sequence": list(g), "labels": [label(x) for x in g], "count": v, "fights": len(f[g])}
                for g, v in c.most_common(10)]

    return {"action": action, "label": label(action), "occurrences": occ, "fights": len(fights),
            "reliability": reliability(occ, len(fights), wilson(max(after.values()) if after else 0, occ), cfg.min_observations),
            "before": dist(before), "after": dist(after), "three_action": grams(3), "five_action": grams(5)}


def style_vector(fighter_obs: list[Obs], seqs: list[list[Obs]], footage: dict[str, float]) -> dict:
    n = len(fighter_obs) or 1
    minutes = _minutes(fighter_obs, footage) or None
    off = sum(1 for o in fighter_obs if o.group == "offensive")
    lens = [len(s) for s in seqs]
    return {"offensive_share": off / n, "defensive_share": sum(1 for o in fighter_obs if o.group == "defensive") / n,
            "kick_share": sum(1 for o in fighter_obs if o.category.endswith("kick")) / n,
            "attacks_per_minute": off / minutes if minutes else 0.0,
            "mean_sequence_length": statistics.fmean(lens) if lens else 0.0}


def style_label(v: dict, n: int, min_actions: int) -> str:
    """Interpretable style from observed action shares (needs enough actions)."""
    if n < min_actions:
        return "insufficient_data"
    if v["offensive_share"] >= 0.6:
        return "aggressive"
    if v["defensive_share"] >= 0.4:
        return "defensive"
    return "balanced"


def opponent_analysis(p: Prepared, fighter_key: str, cfg: PatternConfig, footage: dict[str, float],
                      names: dict[str, str], min_actions_per_opponent: int = 10) -> dict:
    mine = p.by_fighter.get(fighter_key, [])
    opps = sorted({o.opponent_key for o in mine if o.opponent_key})
    per_opp = []
    table_rows, cats = [], sorted({o.category for o in mine})
    for ok in opps:
        vs = [o for o in mine if o.opponent_key == ok]
        opp_obs = [o for o in p.by_fighter.get(ok, []) if o.video_id in {x.video_id for x in vs}]
        opp_seqs = [s for s in fighter_sequences(p, ok) if s[0].video_id in {x.video_id for x in vs}]
        sv = style_vector(opp_obs, opp_seqs, footage)
        my_seqs = [s for s in fighter_sequences(p, fighter_key) if s[0].opponent_key == ok]
        reactions = reaction_table(vs, [o for o in opp_obs if o.group == "offensive"], cfg)
        c = Counter(o.category for o in vs)
        table_rows.append([c.get(k, 0) for k in cats])
        att = reactions["opponent_attacks"]
        off_resp = sum(r["offensive_responses"] for r in reactions["by_attack"])
        def_resp = sum(r["defensive_responses"] for r in reactions["by_attack"])
        per_opp.append({
            "opponent_key": ok, "opponent": names.get(ok, ok), "fights": len({o.fight_key for o in vs}), "actions": len(vs),
            "opponent_style": style_label(sv, len(opp_obs), min_actions_per_opponent), "opponent_style_vector": {k: round(v, 3) for k, v in sv.items()},
            "action_distribution": [{"action": k, "label": label(k), "count": v, "share": round(v / len(vs), 4)} for k, v in c.most_common()],
            "offensive_share": round(sum(1 for o in vs if o.group == "offensive") / len(vs), 4) if vs else None,
            "combinations": combinations(my_seqs, cfg)[:5],
            "after_opponent_attacks": {"opponent_attacks": att, "counter_rate": round(off_resp / att, 4) if att else None,
                                       "counter_ci95": wilson(off_resp, att) if att else None,
                                       "defensive_rate": round(def_resp / att, 4) if att else None,
                                       "defensive_ci95": wilson(def_resp, att) if att else None, "detail": reactions["by_attack"]},
            "sufficient": len(vs) >= min_actions_per_opponent,
        })
    eligible = [i for i, o in enumerate(per_opp) if o["sufficient"]]
    test = chi_square([table_rows[i] for i in eligible]) if len(eligible) >= 2 else {
        "valid": False, "reason": f"needs at least two opponents with >= {min_actions_per_opponent} of this fighter's actions each"}
    by_style: dict[str, list[dict]] = defaultdict(list)
    for o in per_opp:
        if o["sufficient"] and o["opponent_style"] != "insufficient_data":
            by_style[o["opponent_style"]].append(o)
    findings = []
    if test.get("valid") and test["p_value"] < cfg.significance:
        findings.append(f"Action mix differs between opponents (chi-square p = {test['p_value']}).")
    return {"opponents": per_opp, "difference_test": test,
            "by_opponent_style": {k: [o["opponent"] for o in v] for k, v in by_style.items()},
            "findings": findings,
            "note": "Findings are listed only when a statistical test with adequate expected counts supports them."}


def compare(p: Prepared, a: str, b: str, cfg: PatternConfig, footage: dict[str, float]) -> dict:
    out = {}
    for key in (a, b):
        obs = p.by_fighter.get(key, [])
        seqs = fighter_sequences(p, key)
        t = tendencies(obs, seqs, cfg, footage, p.obs)
        out[key] = {"actions": len(obs), "fights": t["fights"], "minutes_observed": t["minutes_observed"],
                    "attack_frequency_per_minute": t["offensive"]["attack_frequency_per_minute"],
                    "offensive_share": t["offensive"]["share_of_actions"], "defensive_share": t["defensive"]["share_of_actions"],
                    "most_used": t["offensive"]["most_used"][:8], "defensive": t["defensive"]["most_used"][:8],
                    "counterattack_rate": t["defensive"]["counterattack_rate"],
                    "top_combinations": combinations(seqs, cfg, t["fights"])[:8],
                    "diversity": t["behavioral"]["sequence_diversity"]}
    states = sorted({o.category for k in (a, b) for o in p.by_fighter.get(k, [])})
    ma = markov(fighter_sequences(p, a), cfg, 1, states)
    mb = markov(fighter_sequences(p, b), cfg, 1, states)
    ca = Counter(o.category for o in p.by_fighter.get(a, []))
    cb = Counter(o.category for o in p.by_fighter.get(b, []))
    test = chi_square([[ca.get(s, 0) for s in states], [cb.get(s, 0) for s in states]])
    return {"fighters": out, "states": states, "markov": {a: ma, b: mb},
            "action_distribution_js_distance": js_distance(ca, cb), "difference_test": test}

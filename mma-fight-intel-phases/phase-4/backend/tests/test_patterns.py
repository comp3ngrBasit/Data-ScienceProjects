"""Phase 4 tests: every expected value below was counted by hand from the listed action sequences."""
from __future__ import annotations

import time

import pytest

from app.patterns import engine as E
from app.patterns.stats import chi_square, entropy, js_distance, reliability, wilson


def O(fighter, video, cat, start, dur=0.3, conf=0.9, fight=None, opp=None):
    return E.Obs(fighter_key=fighter, video_id=video, fight_key=fight or video, category=cat, start=start,
                 end=start + dur, confidence=conf, opponent_key=opp, action_id=f"{video}-{start}-{cat}")


def verified_set():
    """Fighter A, two fights.

    Fight F1 / video v1:  [left_jab 0.0, right_cross 0.5, left_hook 1.0]   (sequence 1)
                          [left_jab 5.0, right_cross 5.5]                  (sequence 2, 4.2 s pause > 2 s gap)
                          [left_jab 10.0] + a duplicate detection at 10.05 (lower confidence, IoU 0.83 -> removed)
    Fight F2 / video v2:  [left_jab 0.0, right_cross 0.4]                  (sequence 4)
                          [slip 3.0]                                       (sequence 5, 2.3 s pause)
    """
    return [
        O("A", "v1", "left_jab", 0.0), O("A", "v1", "right_cross", 0.5), O("A", "v1", "left_hook", 1.0),
        O("A", "v1", "left_jab", 5.0), O("A", "v1", "right_cross", 5.5),
        O("A", "v1", "left_jab", 10.0), O("A", "v1", "left_jab", 10.05, dur=0.25, conf=0.5),
        O("A", "v2", "left_jab", 0.0), O("A", "v2", "right_cross", 0.4), O("A", "v2", "slip", 3.0),
    ]


CFG = E.PatternConfig(min_occurrences=2, min_observations=5)


def cats(seqs):
    return sorted([o.category for o in s] for s in seqs)


# ============================================================================ statistics helpers
def test_wilson_hand_computed():
    assert wilson(3, 4) == pytest.approx((0.3006, 0.9544), abs=1e-4)
    assert wilson(0, 10)[0] == 0.0 and wilson(10, 10)[1] == 1.0
    assert wilson(0, 0) is None


def test_reliability_labels():
    assert reliability(3, 2, (0.1, 0.9), 5) == "insufficient"
    assert reliability(40, 1, (0.4, 0.6), 5) == "single_fight"
    assert reliability(40, 3, (0.4, 0.6), 5) == "high"
    assert reliability(12, 2, (0.3, 0.7), 5) == "medium"
    assert reliability(6, 2, (0.1, 0.9), 5) == "low"


def test_entropy_js_chi_square():
    assert entropy({"a": 1, "b": 1}) == pytest.approx(1.0)
    assert entropy({"a": 5}) == 0.0
    assert js_distance({"a": 1}, {"a": 3}) == pytest.approx(0.0, abs=1e-9)
    assert js_distance({"a": 1}, {"b": 1}) == pytest.approx(1.0)
    small = chi_square([[1, 2], [2, 1]])
    assert small["valid"] is False  # expected counts < 5: test refused rather than reported
    big = chi_square([[30, 10], [10, 30]])
    assert big["valid"] and big["p_value"] < 0.001


# ============================================================================ preparation
def test_prepare_orders_dedupes_and_splits():
    obs = verified_set()[::-1]  # deliberately unordered input
    p = E.prepare(obs, CFG)
    assert p.stats["overlapping_duplicates_removed"] == 1
    assert p.stats["sequences"] == 5
    seqs = E.fighter_sequences(p, "A")
    assert cats(seqs) == sorted([["left_jab", "right_cross", "left_hook"], ["left_jab", "right_cross"], ["left_jab"],
                                 ["left_jab", "right_cross"], ["slip"]])
    kept = [o for o in p.obs if o.category == "left_jab" and o.video_id == "v1" and o.start >= 10]
    assert len(kept) == 1 and kept[0].confidence == 0.9  # the more confident duplicate survives
    first = next(s for s in seqs if len(s) == 3)
    assert [o.interval_prev for o in first] == [None, 0.2, 0.2]


def test_duplicate_footage_of_same_fight_is_not_double_counted():
    a = [O("A", "cam1", "left_jab", 0, fight="F"), O("A", "cam1", "right_cross", 0.5, fight="F")]
    b = [O("A", "cam2", "left_jab", 30, fight="F"), O("A", "cam2", "right_cross", 30.5, fight="F")]  # same exchange, 2nd upload
    p = E.prepare(a + b, CFG)
    assert p.stats["duplicate_footage_sequences_removed"] == 1
    assert len(E.fighter_sequences(p, "A")) == 1


def test_whole_video_is_not_one_sequence():
    obs = [O("A", "v", "left_jab", t) for t in (0, 10, 20, 30)]
    p = E.prepare(obs, CFG)
    assert len(E.fighter_sequences(p, "A")) == 4


# ============================================================================ combinations & conditional probability
def test_combinations_counts():
    p = E.prepare(verified_set(), CFG)
    seqs = E.fighter_sequences(p, "A")
    combos = {tuple(c["sequence"]): c for c in E.combinations(seqs, CFG)}
    jc = combos[("left_jab", "right_cross")]
    assert jc["occurrences"] == 3 and jc["distinct_fights"] == 2 and jc["distinct_videos"] == 2
    assert jc["frequency_per_fight"] == 1.5
    # P(right_cross | left_jab) = 3 / 4 left jabs
    assert jc["confidence"] == 0.75 and jc["confidence_n"] == 4
    # intervals: 0.2, 0.2, 0.1 -> mean 0.1667
    assert jc["avg_interval"] == pytest.approx(0.167, abs=1e-3)
    # followed by another action: only in sequence 1 (by left_hook)
    assert jc["followed_by_action_pct"] == pytest.approx(1 / 3, abs=1e-4)
    assert ("left_jab", "right_cross", "left_hook") not in combos  # 1 occurrence < min_occurrences
    single = E.combinations([s for s in seqs if s[0].fight_key == "v1"], CFG)
    assert single[0]["classification"] == "single_fight"


def test_conditional_probability_definitions_are_separate():
    p = E.prepare(verified_set(), CFG)
    seqs = E.fighter_sequences(p, "A")
    nxt = {(tuple(r["context"]), r["next"]): r for r in E.conditional_probabilities(p.by_fighter["A"], seqs, "next", 1, CFG)}
    assert nxt[(("left_jab",), "right_cross")]["count"] == 3
    assert nxt[(("left_jab",), "right_cross")]["context_count"] == 4
    assert nxt[(("left_jab",), "right_cross")]["probability"] == 0.75
    assert nxt[(("left_jab",), "right_cross")]["ci95"] == pytest.approx((0.3006, 0.9544), abs=1e-4)
    assert nxt[(("left_jab",), "right_cross")]["reliability"] == "insufficient"  # 4 < 5 observations
    assert nxt[(("left_jab",), E.END)]["count"] == 1
    assert nxt[(("right_cross",), E.END)]["probability"] == pytest.approx(0.6667, abs=1e-4)
    # window definition (1.5 s): cross@0.8 -> hook@1.0 within; cross@5.8 -> next jab 4.2 s later; v2 cross -> slip 2.3 s later
    win = {(tuple(r["context"]), r["next"]): r for r in E.conditional_probabilities(p.by_fighter["A"], seqs, "window", 1, CFG)}
    assert win[(("right_cross",), "left_hook")]["count"] == 1
    assert win[(("right_cross",), E.NONE)]["count"] == 2
    assert all(r["window_seconds"] == 1.5 for r in win.values())


def test_next_offensive_skips_defence():
    obs = [O("B", "v", "left_jab", 0), O("B", "v", "slip", 0.5), O("B", "v", "right_cross", 1.0)]
    p = E.prepare(obs, CFG)
    seqs = E.fighter_sequences(p, "B")
    nxt = E.conditional_probabilities(p.by_fighter["B"], seqs, "next", 1, CFG)
    off = E.conditional_probabilities(p.by_fighter["B"], seqs, "next_offensive", 1, CFG)
    assert [r["next"] for r in nxt if r["context"] == ["left_jab"]] == ["slip"]
    assert [r["next"] for r in off if r["context"] == ["left_jab"]] == ["right_cross"]


def test_second_order_context():
    obs = []
    for v in range(6):  # 6 fights of jab-cross-hook
        obs += [O("C", f"v{v}", "left_jab", 0), O("C", f"v{v}", "right_cross", 0.5), O("C", f"v{v}", "left_hook", 1.0)]
    p = E.prepare(obs, CFG)
    seqs = E.fighter_sequences(p, "C")
    rows = E.conditional_probabilities(p.by_fighter["C"], seqs, "next", 2, CFG)
    r = next(r for r in rows if r["context"] == ["left_jab", "right_cross"])
    assert (r["next"], r["count"], r["context_count"], r["probability"], r["fights"]) == ("left_hook", 6, 6, 1.0, 6)
    assert r["reliability"] in ("low", "medium")  # 6 obs: above min, but CI too wide for "high"


# ============================================================================ Markov, PrefixSpan, explorer
def test_markov_smoothing_hand_computed():
    p = E.prepare(verified_set(), CFG)
    seqs = E.fighter_sequences(p, "A")
    m = E.markov(seqs, CFG, 1)
    assert m["states"] == ["left_hook", "left_jab", "right_cross", "slip"] and m["columns"][-1] == E.END
    row = next(r for r in m["rows"] if r["context"] == ["left_jab"])
    cell = {c["next"]: c for c in row["cells"]}
    assert cell["right_cross"]["count"] == 3 and cell["right_cross"]["raw"] == 0.75
    assert cell["right_cross"]["smoothed"] == pytest.approx((3 + 0.5) / (4 + 0.5 * 5), abs=1e-4)
    assert cell["slip"]["smoothed"] == pytest.approx(0.5 / 6.5, abs=1e-4)
    assert row["most_likely_next"] == "right_cross"
    assert sum(c["smoothed"] for c in row["cells"]) == pytest.approx(1.0, abs=1e-3)
    starts = {s["action"]: s["count"] for s in m["start_distribution"]}
    assert starts == {"left_jab": 4, "slip": 1}


def test_prefixspan_finds_gapped_patterns():
    obs = [O("D", "v1", "left_jab", 0), O("D", "v1", "slip", 0.5), O("D", "v1", "right_cross", 1.0),
           O("D", "v2", "left_jab", 0), O("D", "v2", "right_cross", 0.5),
           O("D", "v3", "left_jab", 0), O("D", "v3", "duck", 0.5), O("D", "v3", "right_cross", 1.0)]
    p = E.prepare(obs, CFG)
    seqs = E.fighter_sequences(p, "D")
    pats = {tuple(x["pattern"]): x for x in E.prefixspan(seqs, CFG)}
    jc = pats[("left_jab", "right_cross")]
    assert jc["sequences_containing"] == 3 and jc["support"] == 1.0 and not jc["contiguous_only"]
    contiguous = {tuple(c["sequence"]): c for c in E.combinations(seqs, E.PatternConfig(min_occurrences=1))}
    assert contiguous[("left_jab", "right_cross")]["occurrences"] == 1  # n-grams only see the adjacent case


def test_explorer_before_after():
    p = E.prepare(verified_set(), CFG)
    ex = E.explorer(E.fighter_sequences(p, "A"), "right_cross", CFG)
    assert ex["occurrences"] == 3
    assert {b["action"]: b["count"] for b in ex["before"]} == {"left_jab": 3}
    assert {a["action"]: a["count"] for a in ex["after"]} == {E.END: 2, "left_hook": 1}


# ============================================================================ tendencies & opponents
def test_tendencies_and_reactions():
    a = [O("A", "v", "left_jab", 0, opp="B"), O("A", "v", "right_cross", 0.5, opp="B"),
         O("A", "v", "slip", 5.2, opp="B"), O("A", "v", "right_cross", 10.4, opp="B")]
    b = [O("B", "v", "left_jab", 5.0, opp="A"), O("B", "v", "low_kick", 10.0, opp="A"), O("B", "v", "left_jab", 20.0, opp="A")]
    p = E.prepare(a + b, CFG)
    t = E.tendencies(p.by_fighter["A"], E.fighter_sequences(p, "A"), CFG, {"v": 120.0}, p.obs)
    assert t["actions"] == 4 and t["minutes_observed"] == 2.0
    assert t["offensive"]["attack_frequency_per_minute"] == 1.5  # 3 attacks / 2 min
    assert t["defensive"]["evasions"] == 1
    r = t["defensive"]["reactions_to_opponent_attacks"]
    assert r["opponent_attacks"] == 3
    jab = next(x for x in r["by_attack"] if x["opponent_attack"] == "left_jab")
    assert {x["action"]: x["count"] for x in jab["responses"]} == {"slip": 1, E.NONE: 1}
    assert t["defensive"]["counterattack_rate"] == pytest.approx(1 / 3, abs=1e-4)  # cross after the low kick
    assert t["single_fight_only"] is True
    assert t["behavioral"]["homogeneity_test"]["valid"] is False
    # opponent context attached to the fighter's actions
    slip = next(o for o in p.by_fighter["A"] if o.category == "slip")
    assert slip.opponent_action == "left_jab" and slip.opponent_action_lag == pytest.approx(0.2)


def test_opponent_analysis_requires_sufficient_data():
    a = [O("A", "v1", "left_jab", t, opp="B") for t in (0, 5)] + [O("A", "v2", "low_kick", t, opp="C") for t in (0, 5)]
    p = E.prepare(a, CFG)
    res = E.opponent_analysis(p, "A", CFG, {}, {"B": "Bee", "C": "Cee"}, min_actions_per_opponent=10)
    assert {o["opponent"] for o in res["opponents"]} == {"Bee", "Cee"}
    assert all(not o["sufficient"] for o in res["opponents"])
    assert res["difference_test"]["valid"] is False and res["findings"] == []


def test_compare_two_fighters():
    a = [O("A", "v", "left_jab", t) for t in (0, 5, 10)]
    b = [O("B", "v", "low_kick", t) for t in (0, 5)]
    p = E.prepare(a + b, CFG)
    c = E.compare(p, "A", "B", CFG, {"v": 60.0})
    assert c["fighters"]["A"]["actions"] == 3 and c["fighters"]["B"]["actions"] == 2
    assert c["action_distribution_js_distance"] == pytest.approx(1.0)
    assert c["fighters"]["A"]["diversity"]["distinct_actions"] == 1


# ============================================================================ API (database + background job)
def _seed(client):
    """Two fighter profiles, two fully-annotated videos (two fights) with verified annotations."""
    from app.db.action_models import ActionAnnotation, VideoAnnotationMeta
    from app.db.database import session_scope
    from app.db.models import Video, VideoStatus

    fa = client.post("/api/fighters", json={"name": "Fighter Alpha", "stance": "Orthodox"}).json()["id"]
    fb = client.post("/api/fighters", json={"name": "Fighter Beta"}).json()["id"]
    plan = {
        "vid1": [(fa, "left_jab", 0.0), (fa, "right_cross", 0.5), (fa, "left_hook", 1.0), (fb, "slip", 0.6),
                 (fa, "left_jab", 5.0), (fa, "right_cross", 5.5), (fb, "low_kick", 8.0), (fa, "backward_evasion", 8.2)],
        "vid2": [(fa, "left_jab", 0.0), (fa, "right_cross", 0.4), (fb, "left_jab", 3.0), (fa, "slip", 3.2)],
    }
    ids = {}
    with session_scope() as db:
        for i, (name, rows) in enumerate(plan.items()):
            v = Video(title=name, original_filename=f"{name}.mp4", stored_path=f"videos/{name}.mp4", file_ext=".mp4",
                      file_size=1, sha256=f"{i:064d}", status=VideoStatus.completed, duration_seconds=60.0)
            db.add(v)
            db.flush()
            ids[name] = v.id
            db.add(VideoAnnotationMeta(video_id=v.id, fully_annotated=True))
            for fid, cat, t in rows:
                db.add(ActionAnnotation(video_id=v.id, fighter_id=fid, category=cat, start_time=t, end_time=t + 0.3, status="verified"))
    return fa, fb, ids


def test_patterns_api_end_to_end(client):
    fa, fb, vids = _seed(client)
    ov = client.get("/api/patterns").json()
    assert ov["totals"]["actions"] == 12 and ov["totals"]["fights"] == 2
    alpha = next(f for f in ov["fighters"] if f["fighter_key"] == fa)
    assert alpha["actions"] == 9 and alpha["analysis"] == "not_analysed"

    s = client.get(f"/api/patterns/{fa}").json()
    assert s["meta"]["mode"] == "live" and s["summary"]["fights"] == 2 and s["summary"]["sequences"] == 5

    combos = client.get(f"/api/patterns/{fa}/combinations").json()["combinations"]
    jc = next(c for c in combos if c["sequence"] == ["left_jab", "right_cross"])
    assert (jc["occurrences"], jc["distinct_fights"], jc["confidence"]) == (3, 2, 1.0)

    pr = client.get(f"/api/patterns/{fa}/probabilities", params={"action": "right_cross"}).json()
    got = {r["next"]: (r["count"], r["context_count"]) for r in pr["probabilities"]}
    assert got == {"left_hook": (1, 3), "END": (2, 3)}
    assert client.get(f"/api/patterns/{fa}/probabilities", params={"definition": "bogus"}).status_code == 422

    tr = client.get(f"/api/patterns/{fa}/transitions").json()["markov"]
    assert tr["order"] == 1 and "END" in tr["columns"]

    tend = client.get(f"/api/patterns/{fa}/tendencies").json()["tendencies"]
    reactions = tend["defensive"]["reactions_to_opponent_attacks"]
    assert reactions["opponent_attacks"] == 2  # Beta's low kick and jab
    assert tend["defensive"]["evasions"] == 2

    seq = client.get(f"/api/patterns/{fa}/sequences", params={"action": "right_cross"}).json()
    assert seq["explorer"]["occurrences"] == 3 and seq["total_sequences"] == 5
    first = seq["sequences"][0]["actions"][0]
    assert first["stance"] == "Orthodox" and "interval_prev" in first

    opp = client.get(f"/api/patterns/{fa}/opponent-analysis").json()
    assert [o["opponent"] for o in opp["opponents"]] == ["Fighter Beta"] and opp["opponents"][0]["sufficient"] is False

    # fight grouping: declare both videos as one fight -> cross-fight count drops to 1
    for v in vids.values():
        assert client.put(f"/api/patterns/fights/{v}", json={"fight_key": "same-fight", "fight_date": "2025-03-01"}).status_code == 200
    s1 = client.get(f"/api/patterns/{fa}").json()
    assert s1["summary"]["fights"] == 1 and s1["summary"]["period"] == {"from": "2025-03-01", "to": "2025-03-01"}
    assert client.get(f"/api/patterns/{fa}", params={"date_from": "2026-01-01"}).json()["summary"]["actions"] == 0

    # background analysis -> stored snapshot + derived tables; second run skips unchanged data
    job = client.post(f"/api/patterns/{fa}/analyze").json()
    for _ in range(100):
        st = client.get(f"/api/patterns/{fa}/analysis-status").json()
        if st["state"] == "up_to_date":
            break
        time.sleep(0.1)
    assert st["state"] == "up_to_date" and st["job"]["id"] == job["id"]
    assert client.get(f"/api/patterns/{fa}").json()["meta"]["mode"] == "snapshot"
    job2 = client.post(f"/api/patterns/{fa}/analyze").json()
    for _ in range(100):
        j = next(x for x in client.get("/api/patterns/jobs").json() if x["id"] == job2["id"])
        if j["status"] == "completed":
            break
        time.sleep(0.1)
    assert j["stats"]["unchanged_skipped"] == [fa]
    from app.db.database import session_scope
    from app.db.pattern_models import ActionTransition, FightingCombination, FighterTendency
    with session_scope() as db:
        assert db.query(ActionTransition).filter_by(fighter_key=fa, definition="next", order=1).count() > 0
        assert db.query(FightingCombination).filter_by(fighter_key=fa, sequence_key="left_jab>right_cross").one().occurrences == 3
        assert db.query(FighterTendency).filter_by(fighter_key=fa, category="offensive").count() == 3

    # new data makes the snapshot stale (incremental: only the changed video is re-extracted)
    from app.db.action_models import ActionAnnotation
    with session_scope() as db:
        db.add(ActionAnnotation(video_id=vids["vid2"], fighter_id=fa, category="uppercut", start_time=0.8, end_time=1.0, status="verified"))
    ex = client.get("/api/patterns").json()["extraction"]
    assert ex["cache_hits"] == 1 and ex["cache_misses"] == 1  # only the changed video was re-extracted
    assert client.get(f"/api/patterns/{fa}/analysis-status").json()["state"] == "stale"

    cmp = client.get("/api/patterns/compare", params={"fighter_a": fa, "fighter_b": fb}).json()
    assert cmp["fighters"][fa]["actions"] == 10 and cmp["fighters"][fb]["actions"] == 3
    assert client.get("/api/patterns/compare", params={"fighter_a": fa, "fighter_b": fa}).status_code == 422
    assert client.get("/api/patterns/doesnotexist").status_code == 404

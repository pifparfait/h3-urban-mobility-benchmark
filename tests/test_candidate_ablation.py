from dataclasses import replace
from types import SimpleNamespace

import h3
import numpy as np

from h3uab.benchmarking import make_query_set
from h3uab.config import Config
from h3uab.continuous_indexed_backend import ContinuousIndexedBackend
from h3uab.geometry import haversine_m
from h3uab.h3_backend import MultiLayerH3Backend
from h3uab.models import AccessibilityState, World
from h3uab.static_city import get_h10_candidate_poi_ids


def world(n=40):
    lat = 40.4 + np.arange(n) * .0001
    return World(agent_lat=np.array([40.4, 40.401]), agent_lng=np.array([-3.7, -3.7]),
                 agent_profile_type=np.array([0, 0]),
                 profile_schedule_purpose_ids=np.array([[1]]),
                 profile_schedule_layer_ids=np.array([[1]]),
                 amenity_lat=lat, amenity_lng=np.full(n, -3.7),
                 amenity_layer_code=np.full(n, "FOOD"),
                 amenity_attractiveness=np.arange(n, dtype=float),
                 amenity_osm_id=np.arange(n).astype(str),
                 amenity_name=np.full(n, "poi"), active_layer_codes=("FOOD",))


def backend(w, strategy="radius_exact", cap=None, radius=2000):
    cfg = Config(search_radius_m=radius,
                 baseline=replace(Config().baseline, candidate_strategy=strategy, candidate_cap=cap))
    b = ContinuousIndexedBackend()
    b.prepare(w, cfg)
    b.trace = {}
    return b


def test_exact_matches_original_full_radius_score():
    w = world()
    b = backend(w)
    d = b.decide(0, 40.4, -3.7, "FOOD")
    distances = haversine_m(40.4, -3.7, w.amenity_lat, w.amenity_lng)
    eligible = np.flatnonzero(distances <= b.cfg.search_radius_m)
    scores = -b.cfg.beta_distance * distances[eligible] + b.cfg.gamma_attractiveness * w.amenity_attractiveness[eligible]
    assert d.destination_id == int(eligible[np.argmax(scores)])
    assert b.trace["scored"] == len(eligible)
    assert Config().baseline.candidate_strategy == "radius_exact"
    assert Config().destination_refinement.max_candidates == 24
    assert Config().destination_refinement.fallback_max_candidates == 48


def test_radius_cap_does_not_change_small_candidate_set():
    w = world(5)
    exact = backend(w)
    capped = backend(w, "radius_cap", 24)
    assert exact.decide(0, 40.4, -3.7, "FOOD") == capped.decide(0, 40.4, -3.7, "FOOD")
    assert exact.trace["eligible"] == capped.trace["retained"] == 5


def test_caps_and_deterministic_nearest_selection():
    w = world()
    exact = backend(w)
    capped = backend(w, "radius_cap", 24)
    knn = backend(w, "knn", 24)
    exact_decision = exact.decide(0, 40.4, -3.7, "FOOD")
    first = capped.decide(0, 40.4, -3.7, "FOOD")
    assert exact.trace["returned"] == capped.trace["returned"]
    assert capped.trace["retained"] == capped.trace["scored"] == 24
    assert first.destination_id == 23  # nearest 24, then unchanged score
    assert first.weighted_accessibility == exact_decision.weighted_accessibility
    assert first == capped.decide(0, 40.4, -3.7, "FOOD")
    knn.decide(0, 40.4, -3.7, "FOOD")
    assert knn.trace["requested"] == knn.trace["returned"] == 24
    assert knn.trace["scored"] <= 24


def test_knn_respects_radius():
    w = world()
    b = backend(w, "knn", 24, radius=20)
    d = b.decide(0, 40.4, -3.7, "FOOD")
    assert b.trace["returned"] == 24
    assert b.trace["eligible"] < 24
    assert d.straight_distance_m <= 20


def test_h3_cap_only_changes_selection_not_scoring():
    w = world()
    h12 = h3.latlng_to_cell(40.4, -3.7, 12)
    h10 = h3.cell_to_parent(h12, 10)
    city = SimpleNamespace(poi_index_h10={(h10, "FOOD"): tuple(range(40))})
    cfg24 = Config()
    cfg48 = replace(cfg24, destination_refinement=replace(cfg24.destination_refinement, max_candidates=48))
    ids24 = get_h10_candidate_poi_ids(city, h12, "FOOD", cfg24)
    ids48 = get_h10_candidate_poi_ids(city, h12, "FOOD", cfg48)
    diagnostics = {}
    assert get_h10_candidate_poi_ids(city, h12, "FOOD", cfg24, diagnostics) == ids24
    assert diagnostics["retained"] == 24
    assert diagnostics["raw_pois"] == 40
    assert ids24 == ids48[:24] and len(ids24) == 24 and len(ids48) == 40
    backend = MultiLayerH3Backend()
    backend.world = w
    backend.cfg = cfg24
    scored24 = backend._evaluate_candidates(ids24, 40.4, -3.7)
    backend.cfg = cfg48
    assert scored24 == backend._evaluate_candidates(ids24, 40.4, -3.7)
    assert cfg24.beta_distance == cfg48.beta_distance
    assert cfg24.gamma_attractiveness == cfg48.gamma_attractiveness
    decision_city = SimpleNamespace(
        poi_index_h10=city.poi_index_h10,
        get_macro=lambda cell: None,
        get_amenity_aggregate=lambda cell, layer, res: None,
        get_accessibility=lambda cell, layer: None,
    )
    backend.city = decision_city
    backend.dynamic = SimpleNamespace(position=lambda aid: h12)
    backend.cfg = cfg24
    assert backend.decide(0, 40.4, -3.7, "FOOD") == backend.decide(0, 40.4, -3.7, "FOOD")


def test_query_workload_replay_is_seeded_and_shared():
    w = world()
    cfg = Config()
    q1 = make_query_set(w, cfg, 100)
    q2 = make_query_set(w, cfg, 100)
    np.testing.assert_array_equal(q1.agent_ids, q2.agent_ids)
    np.testing.assert_array_equal(q1.layer_codes, q2.layer_codes)
    b1 = backend(w)
    b2 = backend(w)
    outputs1 = [b1.decide(int(a), float(w.agent_lat[a]), float(w.agent_lng[a]), "FOOD").destination_id for a in q1.agent_ids]
    outputs2 = [b2.decide(int(a), float(w.agent_lat[a]), float(w.agent_lng[a]), "FOOD").destination_id for a in q2.agent_ids]
    assert outputs1 == outputs2


def test_h3_cached_best_and_explicit_fallback_remain_available():
    w = world()
    h12 = h3.latlng_to_cell(40.4, -3.7, 12)
    h9 = h3.cell_to_parent(h12, 9)
    cfg = replace(Config(), destination_refinement=replace(Config().destination_refinement,
                                                             enabled=False))
    b = MultiLayerH3Backend()
    b.world = w
    b.cfg = cfg
    b.dynamic = SimpleNamespace(position=lambda aid: h12)
    state = AccessibilityState(candidate_count=1, weighted_accessibility=0.5,
                               best_poi_id=7, best_distance_m=0.0)
    b.city = SimpleNamespace(get_macro=lambda cell: None,
                             get_amenity_aggregate=lambda cell, layer, res: None,
                             get_accessibility=lambda cell, layer: state,
                             accessibility_h9={(h9, "FOOD"): state})
    assert b.decide(0, 40.4, -3.7, "FOOD").destination_id == 7
    b.city.get_accessibility = lambda cell, layer: None
    assert b.decide(0, 40.4, -3.7, "FOOD").destination_id == 7

from grounded_memory.perception import rank_matches


def test_context_prior_disambiguates_without_erasing_appearance():
    region = {"features": [1.0, 0.0], "encoder": "fixture"}
    a = {
        "id": "towels",
        "label": "paper towels",
        "examples": [{"features": [1.0, 0.0], "encoder": "fixture", "place_id": "kitchen"}],
    }
    b = {
        "id": "tissue",
        "label": "toilet paper",
        "examples": [{"features": [1.0, 0.0], "encoder": "fixture", "place_id": "bathroom"}],
    }
    assert rank_matches(region, [a, b], "kitchen")[0]["id"] == "towels"
    assert rank_matches(region, [a, b], "bathroom")[0]["id"] == "tissue"
    a["examples"][0]["features"] = [0.6, 0.8]
    assert rank_matches(region, [a, b], "kitchen")[0]["id"] == "tissue"


def test_incompatible_feature_versions_do_not_match():
    assert not rank_matches(
        {"features": [1.0], "encoder": "new"},
        [{"id": "x", "label": "x", "examples": [{"features": [1.0], "encoder": "old"}]}],
    )


def test_proposals_prefer_the_central_object_over_edge_slivers_and_blank_frames():
    import cv2
    import numpy as np
    from grounded_memory.perception import propose

    image = np.full((480, 360, 3), 60, np.uint8)
    cv2.rectangle(image, (130, 180), (230, 320), (240, 240, 240), -1)  # bottle-sized object, center
    cv2.rectangle(image, (345, 100), (359, 400), (255, 255, 255), -1)  # bright sliver on the edge
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flat = np.full(gray.shape, 0.3)
    x, y, w, h = propose(gray, flat, np.zeros_like(flat))[0][1]
    assert 110 <= x <= 135 and 160 <= y <= 185 and w < 140
    assert propose(np.zeros_like(gray), flat, np.zeros_like(flat)) == []


def test_different_lens_projections_never_match_and_matches_state_sensor_relation():
    region = {"features": [1.0, 0.0], "encoder": "e", "sensor_id": "phone", "projection": "rectilinear"}
    fisheye = {
        "id": "a",
        "label": "mug",
        "examples": [{"features": [1.0, 0.0], "encoder": "e", "projection": "fisheye"}],
    }
    assert rank_matches(region, [fisheye]) == []
    same = {
        "id": "b",
        "label": "mug",
        "examples": [{"features": [1.0, 0.0], "encoder": "e", "sensor_id": "phone"}],
    }
    imported = {"id": "c", "label": "mug", "examples": [{"features": [1.0, 0.0], "encoder": "e"}]}
    assert rank_matches(region, [same])[0]["sensors"] == "same"
    assert rank_matches(region, [imported])[0]["sensors"] == "unverified"


def test_clean_keeps_the_pointed_piece_fills_specks_and_resize_needs_a_real_change():
    import numpy as np
    from grounded_memory.perception import clean, pick_resize

    mask = np.zeros((200, 200), bool)
    mask[20:120, 20:120] = True  # the object
    mask[60:64, 60:64] = False  # a speck hole
    mask[150:190, 150:190] = True  # a separate piece
    mask[5, 190] = True  # a stray pixel
    cleaned = clean(mask, [[0.3, 0.3, True]])
    assert cleaned[62, 62] and not cleaned[170, 170] and not cleaned[5, 190]
    assert clean(mask, [[0.85, 0.85, True]])[170, 170] and not clean(mask, [[0.85, 0.85, True]])[50, 50]
    ring = np.zeros((200, 200), bool)
    ring[20:180, 20:180] = True
    ring[50:150, 50:150] = False  # a large real opening stays open
    assert not clean(ring)[100, 100]

    current = np.zeros((100, 100), bool)
    current[40:60, 40:60] = True
    same = current.copy()
    bigger = np.zeros((100, 100), bool)
    bigger[30:70, 30:70] = True
    elsewhere = np.zeros((100, 100), bool)
    elsewhere[0:40, 0:40] = True
    assert pick_resize(current, [same, elsewhere], "larger") is None
    assert pick_resize(current, [same, bigger], "larger") is bigger
    assert pick_resize(bigger, [same, bigger], "smaller") is same


def test_camera_motion_is_cancelled_and_a_moving_thing_remains():
    # A synthetic regression check; real-recording behaviour is measured by docs/NEXT_STEPS.md.
    import cv2
    import numpy as np
    from grounded_memory.perception import world_motion

    rng = np.random.default_rng(1)
    scene = cv2.GaussianBlur(rng.integers(0, 255, (520, 680), dtype=np.uint8), (5, 5), 0)
    before = scene[20:500, 20:660].copy()
    panned = scene[12:492, 32:672].copy()  # the camera turned; nothing in the world moved
    still = world_motion(before, panned)
    assert still["world_motion"] < 0.02 and still["camera_motion"] > 0.01
    moved = panned.copy()
    moved[200:290, 300:400] = 255  # something entered the view while the camera turned
    found = world_motion(before, moved)
    assert 0.02 <= found["world_motion"] <= 0.25
    x, y, w, h = found["world_motion_box"]
    assert abs(x - 300 / 640) < 0.03 and abs(y - 200 / 480) < 0.03
    assert world_motion(np.zeros((480, 640), np.uint8), moved) is None  # nothing to track: no guess

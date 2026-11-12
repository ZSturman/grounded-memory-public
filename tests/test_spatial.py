import numpy as np
from grounded_memory.spatial import relate, relations, world_extent


def box(lo, hi):
    return {"min": lo, "max": hi}


def test_relations_from_world_extents():
    desk = box([0, 0.70, 0], [1.2, 0.75, 0.6])
    bottle = box([0.4, 0.75, 0.2], [0.46, 0.87, 0.26])
    lamp = box([1.3, 0.70, 0.1], [1.45, 1.10, 0.25])
    shelf = box([0, 1.40, 0], [1.2, 1.45, 0.3])
    drawer = box([0, 0, 0], [0.5, 0.4, 0.5])
    pen = box([0.1, 0.05, 0.1], [0.25, 0.08, 0.12])
    assert relate(bottle, desk) == "on"
    assert relate(shelf, desk) == "above"
    assert relate(pen, drawer) == "inside"
    assert relate(lamp, desk) == "next_to"
    assert relate(desk, bottle) is None
    located = {"d": {"extent": desk}, "b": {"extent": bottle}, "l": {"extent": lamp}}
    assert sorted(relations(located)) == [("b", "on", "d"), ("d", "next_to", "l")]


def test_world_extent_maps_upright_mask_through_sensor_depth():
    # Identity pose, camera looking down -Z; a flat wall 2 m away fills the depth map.
    camera = dict(
        transform=np.eye(4),
        intrinsics=np.array([[1000.0, 0, 960], [0, 1000.0, 720], [0, 0, 1]]),
        resolution=np.array([1920.0, 1440.0]),
    )
    depth = np.full((192, 256), 2.0)
    mask = np.zeros((640, 480), bool)
    mask[0:40, 200:280] = True  # top-centre of the upright image
    extent = world_extent(mask, depth, camera)
    assert abs(extent["center"][2] + 2.0) < 1e-6
    # Phone upright: the image top is the sensor's left edge (camera -X), centred vertically.
    assert extent["center"][0] < -0.8 and abs(extent["center"][1]) < 0.2
    assert world_extent(mask, np.zeros((192, 256)), camera) is None


def test_project_inverts_world_extent_and_hides_occluded_points():
    from grounded_memory.spatial import project

    camera = dict(
        transform=np.eye(4),
        intrinsics=np.array([[1000.0, 0, 960], [0, 1000.0, 720], [0, 0, 1]]),
        resolution=np.array([1920.0, 1440.0]),
    )
    depth = np.full((192, 256), 2.0)
    mask = np.zeros((640, 480), bool)
    mask[300:340, 100:140] = True
    centre = world_extent(mask, depth, camera)["center"]
    u, v = project(centre, camera)
    assert abs(u - 120 / 480) < 0.02 and abs(v - 320 / 640) < 0.02
    assert project(centre, camera, depth) is not None
    assert project(centre, camera, np.full((192, 256), 1.0)) is None  # something nearer covers it
    assert project([0, 0, 1.0], camera) is None  # behind the camera

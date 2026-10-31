import unittest
import lanelet2  # if we fail here, there is something wrong with lanelet2 registration
import numpy as np
from lanelet2.core import getId, Point3d, BasicPoint2d, BasicPoint3d, LineString3d, Lanelet, LaneletMap
from lanelet2.core import TrafficLight
from lanelet2.ml_converter import MapDataInterface, getRotatedRect, ParametrizationType, LineStringType, TEType


def get_sample_lanelet_map():
    mymap = LaneletMap()
    x_left = 4
    x_middle = 2
    x_right = 0
    ls_left = LineString3d(1234, [Point3d(getId(), x_left, i, 0)
                           for i in range(0, 3)], {"type": "road_border"})
    ls_middle = LineString3d(getId(), [Point3d(getId(), x_middle, i, 0) for i in range(
        0, 3)], {"type": "line", "subtype": "dashed"})
    ls_right = LineString3d(getId(), [Point3d(
        getId(), x_right, i, 0) for i in range(0, 3)], {"type": "road_border"})
    llet = Lanelet(getId(), ls_left, ls_middle)
    llet_2 = Lanelet(getId(), ls_middle, ls_right)
    mymap.add(llet)
    mymap.add(llet_2)

    # Add traffic elements
    # Stop line across the first lanelet at y=1.5
    stop_line = LineString3d(getId(), [
        Point3d(getId(), x_left, 1.5, 0),
        Point3d(getId(), x_middle, 1.5, 0)
    ], {"type": "stop_line"})
    mymap.add(stop_line)

    # Traffic light positioned above the stop line
    traffic_light = LineString3d(getId(), [
        Point3d(getId(), x_left + 0.5, 1.5, 3),
        Point3d(getId(), x_left + 0.5, 1.5, 4)
    ], {"type": "traffic_light", "subtype": "red_yellow_green"})
    mymap.add(traffic_light)

    # Straight arrow on the first lanelet
    arrow = LineString3d(getId(), [
        Point3d(getId(), x_left - 0.5, 0.8, 0),
        Point3d(getId(), x_left - 0.5, 1.2, 0)
    ], {"type": "arrow", "subtype": "straight"})
    mymap.add(arrow)

    # Add regulatory element connecting traffic light to stop line
    reg_elem = TrafficLight(getId(), {"type": "regulatory_element", "subtype": "traffic_light"},
                            [traffic_light], stop_line)
    mymap.add(reg_elem)

    return mymap


def get_bike_lane_skip_over_map():
    """Two road lanes with a bicycle lane between them, all bounds of the bike lane are dashed bike markings.

    Driving +x, y increases to the left: v1 (y 0..2) | bike (y 2..4) | v2 (y 4..6).
    """
    def line(y, attrs):
        return LineString3d(getId(), [Point3d(getId(), x, y, 0) for x in (0, 10)], attrs)
    dashed_bike = {"type": "bike_marking", "subtype": "dashed"}
    v1_right = line(0, {"type": "road_border"})
    v1_bike = line(2, dashed_bike)
    bike_v2 = line(4, dashed_bike)
    v2_left = line(6, {"type": "road_border"})
    v1 = Lanelet(getId(), v1_bike, v1_right, {"subtype": "road"})
    bike = Lanelet(getId(), bike_v2, v1_bike, {"subtype": "bicycle_lane"})
    v2 = Lanelet(getId(), v2_left, bike_v2, {"subtype": "road"})
    mymap = LaneletMap()
    for ll in (v1, bike, v2):
        mymap.add(ll)
    return mymap, v1, bike, v2


class MapDataInterfaceTestCase(unittest.TestCase):
    def test_map_data_interface(self):
        mymap = get_sample_lanelet_map()
        pos = BasicPoint2d(1, 1)
        mDataIf = MapDataInterface(mymap)
        mDataIf.setCurrPosAndExtractSubmap2d(pos, 0)
        mData = mDataIf.mapData(True)
        tfData = mData.getTensorInstanceData(True, False)
        self.assertEqual(len(tfData.compoundLineStringsOfType(
            lanelet2.ml_converter.LineStringType.Centerline)), 2)
        # Check that we can filter lane dividers (Dashed type as example)
        dividers = tfData.compoundLineStringsOfType(
            lanelet2.ml_converter.LineStringType.Dashed)
        # Get the type int from first divider if it exists
        if len(dividers) > 0:
            self.assertEqual(lanelet2.ml_converter.LineStringType.Dashed, 1)

    def test_traffic_elements(self):
        """Test that traffic elements are properly collected and accessible"""
        mymap = get_sample_lanelet_map()
        pos = BasicPoint2d(3, 1.5)  # Position near stop line
        mDataIf = MapDataInterface(mymap)
        mDataIf.setCurrPosAndExtractSubmap2d(pos, 0)
        mData = mDataIf.mapData(True)
        tfData = mData.getTensorInstanceData(True, False)

        # Test stop lines
        stop_lines = tfData.teInstancesOfType(
            lanelet2.ml_converter.TEType.StopLine)
        self.assertGreater(len(stop_lines), 0,
                           "Should have at least one stop line")

        # Test traffic lights
        traffic_lights = tfData.teInstancesOfType(
            lanelet2.ml_converter.TEType.TLCar)
        self.assertGreater(len(traffic_lights), 0,
                           "Should have at least one traffic light")

        # Test arrows
        arrows = tfData.teInstancesOfType(
            lanelet2.ml_converter.TEType.ArrowGoStraight)
        self.assertGreater(
            len(arrows), 0, "Should have at least one straight arrow")

    def test_traffic_element_edges(self):
        """Test that traffic element edges are properly created"""
        mymap = get_sample_lanelet_map()
        pos = BasicPoint2d(3, 1.5)
        mDataIf = MapDataInterface(mymap)
        mDataIf.setCurrPosAndExtractSubmap2d(pos, 0)
        mData = mDataIf.mapData(True)
        tfData = mData.getTensorInstanceData(True, False)

        # Test TE to centerline edges (arrows and stop lines to lanelets)
        te_to_centerline_edges = tfData.teToCenterlineIndexEdges
        self.assertGreater(len(te_to_centerline_edges), 0,
                           "Should have TE to centerline edges")

        # Check edge structure: each edge should be a tuple (TEType, te_idx, centerline_idx)
        for edge in te_to_centerline_edges:
            self.assertEqual(len(edge), 3, "Edge should be a 3-tuple")
            # First element should be a TEType (enum value)
            self.assertIsInstance(edge[0], lanelet2.ml_converter.TEType)
            # Second and third elements should be indices (integers)
            self.assertIsInstance(edge[1], int)
            self.assertIsInstance(edge[2], int)

        # Test TE to TE edges (traffic lights to stop lines)
        te_to_te_edges = tfData.teToTEIndexEdges
        self.assertGreater(len(te_to_te_edges), 0,
                           "Should have TE to TE edges (traffic light to stop line)")

        # Check edge structure: each edge should be a tuple (TEType, te_idx, TEType, te_idx)
        for edge in te_to_te_edges:
            self.assertEqual(len(edge), 4, "Edge should be a 4-tuple")
            # First and third elements should be TEType
            self.assertIsInstance(edge[0], lanelet2.ml_converter.TEType)
            self.assertIsInstance(edge[2], lanelet2.ml_converter.TEType)
            # Second and fourth elements should be indices
            self.assertIsInstance(edge[1], int)
            self.assertIsInstance(edge[3], int)

    def test_map_data_edges(self):
        """Test that MapData edge properties are accessible"""
        mymap = get_sample_lanelet_map()
        pos = BasicPoint2d(3, 1.5)
        mDataIf = MapDataInterface(mymap)
        mDataIf.setCurrPosAndExtractSubmap2d(pos, 0)
        mData = mDataIf.mapData(True)

        # Test llEdges (lanelet edges)
        ll_edges = mData.llEdges
        self.assertIsInstance(ll_edges, dict, "llEdges should be a dictionary")

        # Test teEdges (traffic element edges)
        te_edges = mData.teEdges
        self.assertIsInstance(te_edges, dict, "teEdges should be a dictionary")

        # Test teToCenterlineEdges (list of pairs)
        te_to_centerline = mData.teToCenterlineEdges
        self.assertIsInstance(te_to_centerline, list,
                              "teToCenterlineEdges should be a list")

        # Test teToTEEdges (list of pairs)
        te_to_te = mData.teToTEEdges
        self.assertIsInstance(te_to_te, list, "teToTEEdges should be a list")


class BicycleLaneSkipOverTestCase(unittest.TestCase):
    def lane_change_edges(self, allow):
        mymap, v1, bike, v2 = get_bike_lane_skip_over_map()
        config = MapDataInterface.Configuration()
        self.assertFalse(config.allowLaneChangeAcrossBicycleLane)
        config.allowLaneChangeAcrossBicycleLane = allow
        mDataIf = MapDataInterface(mymap, config)
        mDataIf.setCurrPosAndExtractSubmap2d(BasicPoint2d(5, 3), 0)
        mData = mDataIf.mapData(True)
        edges = {(e.el1, e.el2) for es in mData.llEdges.values() for e in es if e.isLaneChange}
        return edges, v1.id, bike.id, v2.id

    def test_disabled_by_default(self):
        edges, v1, _, v2 = self.lane_change_edges(False)
        self.assertNotIn((v1, v2), edges)
        self.assertNotIn((v2, v1), edges)

    def test_enabled(self):
        edges, v1, bike, v2 = self.lane_change_edges(True)
        self.assertIn((v1, v2), edges)
        self.assertIn((v2, v1), edges)
        # the bike lane is not part of the vehicle graph
        self.assertFalse(any(bike in e for e in edges))


class MapDataReprocessingTestCase(unittest.TestCase):
    """processAll() can be called again with another box, the result equals a fresh MapData for that box"""

    def summary(self, mData):
        """Valid instance counts and tensor data (taken from the buffer) of all types"""
        tf = mData.getTensorInstanceData(True, False)
        res = {}
        for t in LineStringType.values.values():
            res[("ls", t)] = (len(mData.validLineStringsOfType(t)), tf.lineStringsOfType(t))
            res[("cpd", t)] = (len(mData.validCompoundLineStringsOfType(t)), tf.compoundLineStringsOfType(t))
        for t in TEType.values.values():
            res[("te", t)] = (len(mData.validTEInstancesOfType(t)), tf.teInstancesOfType(t))
        res["edges"] = (list(tf.teToCenterlineIndexEdges), list(tf.teToTEIndexEdges))
        return res

    def assertSameSummary(self, actual, expected):
        self.assertEqual(actual.keys(), expected.keys())
        self.assertEqual(actual.pop("edges"), expected.pop("edges"))
        for key, (count, mats) in expected.items():
            self.assertEqual(actual[key][0], count, key)
            self.assertEqual(len(actual[key][1]), len(mats), key)
            for a, e in zip(actual[key][1], mats):
                np.testing.assert_allclose(a, e, atol=1e-9)

    def test_process_all_twice(self):
        mDataIf = MapDataInterface(get_sample_lanelet_map())
        mDataIf.setCurrPosAndExtractSubmap2d(BasicPoint2d(3, 1.5), 0)
        near = getRotatedRect(BasicPoint3d(3, 1.5, 0), 30, 15, 0, True)
        far = getRotatedRect(BasicPoint3d(703, 1.5, 0), 30, 15, 0, True)

        def process(mData, box):
            return mData.processAll(box, ParametrizationType.LineString, True, 20, True, 20)

        fresh = mDataIf.mapData(False)
        process(fresh, near)
        expected = self.summary(fresh)
        self.assertGreater(sum(c for k, (c, _) in expected.items() if k != "edges"), 0)

        reused = mDataIf.mapData(False)
        process(reused, far)
        self.assertEqual(sum(len(reused.validLineStringsOfType(t)) for t in LineStringType.values.values()), 0)
        self.summary(reused)  # fills the tensor data buffer for the far box
        process(reused, near)
        self.assertSameSummary(self.summary(reused), expected)



class MergeIdenticalCompoundsTestCase(unittest.TestCase):
    """Compound centerlines of paths that part outside the box are one line inside it (kitscenes3)"""

    @staticmethod
    def fork_map():
        """Northbound 0..20, 20..40, then a fork at y 40: straight on (to y 60) and to the right (x + 5)"""
        def pt(x, y):
            return Point3d(getId(), x, y, 0)

        def ls(*pts):
            return LineString3d(getId(), list(pts), {"type": "line_thin", "subtype": "solid"})
        l0, r0, l20, r20, l40, r40 = pt(0, 0), pt(3.5, 0), pt(0, 20), pt(3.5, 20), pt(0, 40), pt(3.5, 40)
        mymap = LaneletMap()
        for left, right in ((ls(l0, l20), ls(r0, r20)), (ls(l20, l40), ls(r20, r40)),
                            (ls(l40, pt(0, 60)), ls(r40, pt(3.5, 60))), (ls(l40, pt(5, 60)), ls(r40, pt(8.5, 60)))):
            mymap.add(Lanelet(getId(), left, right, {"subtype": "road"}))
        return mymap

    def map_data(self, through_interface):
        config = MapDataInterface.Configuration()
        config.submapExtentLongitudinal, config.submapExtentLateral = 15, 10
        mDataIf = MapDataInterface(self.fork_map(), config)
        # the box ends at y 39, the fork (y 40) is in the extracted submap
        mDataIf.setCurrPosAndExtractSubmap2d(BasicPoint2d(1.75, 24), np.pi / 2)
        mData = mDataIf.mapData(through_interface)
        if not through_interface:
            box = getRotatedRect(BasicPoint3d(1.75, 24, 0), 15, 10, np.pi / 2, True)
            mData.processAll(box, ParametrizationType.LineString, True, 20, True, 20)
        return mData

    def test_merge(self):
        for through_interface in (True, False):
            mData = self.map_data(through_interface)
            paths = mData.validCompoundLineStringsOfType(LineStringType.Centerline)
            self.assertEqual(len(paths), 2)  # the instances keep both paths
            lines = mData.getTensorInstanceData(True, False).compoundLineStringsOfType(LineStringType.Centerline)
            self.assertEqual(len(lines), 1)

if __name__ == '__main__':
    unittest.main()

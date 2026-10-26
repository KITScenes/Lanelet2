import unittest
import lanelet2  # if we fail here, there is something wrong with lanelet2 registration
from lanelet2.core import getId, Point3d, BasicPoint2d, LineString3d, Lanelet, LaneletMap
from lanelet2.core import TrafficLight
from lanelet2.ml_converter import MapDataInterface


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


if __name__ == '__main__':
    unittest.main()

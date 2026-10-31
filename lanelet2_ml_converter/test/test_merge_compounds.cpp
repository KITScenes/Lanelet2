#include <gtest/gtest.h>
#include <lanelet2_core/LaneletMap.h>
#include <lanelet2_core/primitives/BasicRegulatoryElements.h>
#include <lanelet2_traffic_rules/TrafficRulesFactory.h>

#include <cmath>

#include "lanelet2_ml_converter/MapData.h"
#include "lanelet2_ml_converter/MapInstances.h"
#include "lanelet2_ml_converter/Utils.h"
#include "lanelet2_routing/RoutingGraph.h"

using namespace lanelet;
using namespace lanelet::ml_converter;

// Compound centerlines are built per path through the submap and only then cut to the bounding box. Two paths
// that part outside of the box are the same line inside it; the tensor data has it once.

namespace {

class MergeIdenticalCompoundsTest : public ::testing::Test {
 protected:
  // Northbound trunk 10 (y 0..20) and 11 (y 20..40), then a fork: 12 straight on (to y 60), 13 to the right (x + 5).
  // A stop line at y 10 belongs to 10 and 11, so the converter links it to every compound centerline through 10 and
  // through 11: twice to each path, which the tensor data has once.
  void SetUp() override {
    auto pt = [this](double x, double y) { return Point3d(nextId++, x, y, 0); };
    auto ls = [this](const Points3d& pts) {
      LineString3d line(nextId++, pts);
      line.setAttribute(AttributeName::Type, "line_thin");
      line.setAttribute(AttributeName::Subtype, "solid");
      return line;
    };
    Point3d l0 = pt(0, 0), r0 = pt(3.5, 0), l20 = pt(0, 20), r20 = pt(3.5, 20), l40 = pt(0, 40), r40 = pt(3.5, 40);
    Lanelet trunk1(10, ls({l0, l20}), ls({r0, r20}));
    Lanelet trunk2(11, ls({l20, l40}), ls({r20, r40}));
    Lanelet straight(12, ls({l40, pt(0, 60)}), ls({r40, pt(3.5, 60)}));
    Lanelet right(13, ls({l40, pt(5, 60)}), ls({r40, pt(8.5, 60)}));
    LineString3d stopLine(nextId++, {pt(0, 10), pt(3.5, 10)});
    stopLine.setAttribute(AttributeName::Type, "stop_line");
    auto rightOfWay = RightOfWay::make(nextId++, AttributeMap(), {trunk1}, {}, stopLine);
    trunk1.addRegulatoryElement(rightOfWay);
    trunk2.addRegulatoryElement(rightOfWay);

    Lanelets lanelets{trunk1, trunk2, straight, right};
    for (auto& ll : lanelets) {
      ll.setAttribute(AttributeName::Subtype, AttributeValueString::Road);
    }
    trafficRules = traffic_rules::TrafficRulesFactory::create(Locations::Germany, Participants::Vehicle);
    auto submap = utils::createSubmap(lanelets, {});
    submap->add(stopLine);
    graph = routing::RoutingGraph::build(*submap, *trafficRules);
    laneletSubmap = std::shared_ptr<const LaneletSubmap>(std::move(submap));
  }

  MapDataPtr process(const OrientedRect& box) {
    MapDataPtr mapData = MapData::build(laneletSubmap, graph, trafficRules);
    mapData->processAll(box, ParametrizationType::LineString, true, 20, true, 20, 0.0, 0.0);
    return mapData;
  }

  Id nextId{100};
  traffic_rules::TrafficRulesPtr trafficRules;
  routing::RoutingGraphConstPtr graph;
  LaneletSubmapConstPtr laneletSubmap;
  // y 5..35: ends before the fork
  const OrientedRect beforeFork{getRotatedRect(BasicPoint3d{1.75, 20, 0}, 15, 10, M_PI / 2.0, true)};
  // y 0..60: contains the fork
  const OrientedRect withFork{getRotatedRect(BasicPoint3d{1.75, 30, 0}, 30, 10, M_PI / 2.0, true)};
};

}  // namespace

TEST_F(MergeIdenticalCompoundsTest, PathsThatPartOutsideTheBoxAreOneCenterline) {  // NOLINT
  MapDataPtr mapData = process(beforeFork);
  auto data = mapData->getTensorInstanceData(false, false);
  // both paths are instances, with the same processed geometry
  CompoundLaneLineStringInstanceList paths = mapData->validCompoundLineStringsOfType(LineStringType::Centerline);
  ASSERT_EQ(paths.size(), 2U);
  ASSERT_EQ(paths[0]->pointMatrices(false).size(), 1U);
  EXPECT_TRUE(paths[0]->pointMatrices(false)[0] == paths[1]->pointMatrices(false)[0]);
  // the tensor data has the line once, and the stop line's two edges (one per path) as one
  std::vector<MatrixXd> lines = data.compoundLineStringsOfType(LineStringType::Centerline);
  ASSERT_EQ(lines.size(), 1U);
  EXPECT_TRUE(lines[0] == paths[0]->pointMatrices(false)[0]);
  ASSERT_EQ(data.teToCenterlineIndexEdges().size(), 1U);
  EXPECT_EQ(std::get<2>(data.teToCenterlineIndexEdges()[0]), 0U);
}

TEST_F(MergeIdenticalCompoundsTest, PathsThatPartInsideTheBoxStayTwo) {  // NOLINT
  auto data = process(withFork)->getTensorInstanceData(false, false);
  EXPECT_EQ(data.compoundLineStringsOfType(LineStringType::Centerline).size(), 2U);
  EXPECT_EQ(data.teToCenterlineIndexEdges().size(), 2U);
}

TEST_F(MergeIdenticalCompoundsTest, ReprocessingMergesForTheCurrentBox) {  // NOLINT
  MapDataPtr mapData = process(withFork);
  mapData->getTensorInstanceData(false, false);  // fill the buffer
  mapData->processAll(beforeFork, ParametrizationType::LineString, true, 20, true, 20, 0.0, 0.0);
  EXPECT_EQ(mapData->getTensorInstanceData(false, false).compoundLineStringsOfType(LineStringType::Centerline).size(),
            1U);
}

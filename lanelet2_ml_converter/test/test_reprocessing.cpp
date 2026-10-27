#include <gtest/gtest.h>
#include <lanelet2_traffic_rules/TrafficRulesFactory.h>

#include "lanelet2_ml_converter/MapData.h"
#include "lanelet2_ml_converter/MapInstances.h"
#include "lanelet2_routing/RoutingGraph.h"
#include "test_map.h"

using namespace lanelet;
using namespace lanelet::ml_converter;
using namespace lanelet::ml_converter::tests;

// Processing a MapData must be a pure function of (raw instances, bounding box, parameters): calling processAll()
// for one box and then for another one has to give exactly what a freshly built MapData gives for the second box
// alone. This is needed to build a MapData once per scene and clip it for every frame.

namespace {

constexpr int32_t NPointsLanes = 20;
constexpr int32_t NPointsTE = 10;

void expectSameMatrices(const std::vector<MatrixXd>& actual, const std::vector<MatrixXd>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(actual[i].rows(), expected[i].rows());
    ASSERT_EQ(actual[i].cols(), expected[i].cols());
    EXPECT_TRUE(actual[i].isApprox(expected[i], 1e-12)) << "matrix " << i << " differs";
  }
}

void expectSameLineStrings(const BasicLineStrings3d& actual, const BasicLineStrings3d& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    ASSERT_EQ(actual[i].size(), expected[i].size());
    for (size_t j = 0; j < expected[i].size(); ++j) {
      EXPECT_TRUE(actual[i][j].isApprox(expected[i][j], 1e-12));
    }
  }
}

void expectSameProcessedState(const LineStringInstance& actual, const LineStringInstance& expected) {
  EXPECT_EQ(actual.mapID(), expected.mapID());
  EXPECT_EQ(actual.valid(), expected.valid()) << "instance " << expected.mapID();
  EXPECT_EQ(actual.wasCut(), expected.wasCut()) << "instance " << expected.mapID();
  expectSameLineStrings(actual.cutInstance(), expected.cutInstance());
  expectSameLineStrings(actual.cutAndTransformedInstance(), expected.cutAndTransformedInstance());
  expectSameLineStrings(actual.cutTransformedAndResampledInstance(), expected.cutTransformedAndResampledInstance());
}

void expectSameMapData(MapData& actual, MapData& expected) {
  for (int t = static_cast<int>(LineStringType::RoadBorder); t <= static_cast<int>(LineStringType::ZebraCrossing);
       ++t) {
    auto type = static_cast<LineStringType>(t);
    SCOPED_TRACE("LineStringType " + std::to_string(t));

    // all instances (valid or not) must carry the same processed state
    LaneLineStringInstances allActual = actual.lineStringsOfType(type);
    LaneLineStringInstances allExpected = expected.lineStringsOfType(type);
    ASSERT_EQ(allActual.size(), allExpected.size());
    for (const auto& el : allExpected) {
      auto it = allActual.find(el.first);
      ASSERT_NE(it, allActual.end());
      expectSameProcessedState(*it->second, *el.second);
    }
    EXPECT_EQ(actual.validLineStringsOfType(type).size(), expected.validLineStringsOfType(type).size());

    CompoundLaneLineStringInstanceList cpdActual = actual.compoundLineStringsOfType(type);
    CompoundLaneLineStringInstanceList cpdExpected = expected.compoundLineStringsOfType(type);
    ASSERT_EQ(cpdActual.size(), cpdExpected.size());
    for (size_t i = 0; i < cpdExpected.size(); ++i) {
      expectSameProcessedState(*cpdActual[i], *cpdExpected[i]);
      EXPECT_EQ(cpdActual[i]->processedInstancesValid(), cpdExpected[i]->processedInstancesValid());
      ASSERT_EQ(cpdActual[i]->pathLengthsProcessed().size(), cpdExpected[i]->pathLengthsProcessed().size());
      for (size_t j = 0; j < cpdExpected[i]->pathLengthsProcessed().size(); ++j) {
        EXPECT_NEAR(cpdActual[i]->pathLengthsProcessed()[j], cpdExpected[i]->pathLengthsProcessed()[j], 1e-9);
      }
      ASSERT_EQ(cpdActual[i]->features().size(), cpdExpected[i]->features().size());
      for (size_t j = 0; j < cpdExpected[i]->features().size(); ++j) {
        expectSameProcessedState(*cpdActual[i]->features()[j], *cpdExpected[i]->features()[j]);
      }
    }
    EXPECT_EQ(actual.validCompoundLineStringsOfType(type).size(), expected.validCompoundLineStringsOfType(type).size());
  }

  for (int t = static_cast<int>(TEType::TLCar); t <= static_cast<int>(TEType::Unknown); ++t) {
    auto type = static_cast<TEType>(t);
    SCOPED_TRACE("TEType " + std::to_string(t));
    TEInstances allActual = actual.teInstancesOfType(type);
    TEInstances allExpected = expected.teInstancesOfType(type);
    ASSERT_EQ(allActual.size(), allExpected.size());
    for (const auto& el : allExpected) {
      auto it = allActual.find(el.first);
      ASSERT_NE(it, allActual.end());
      expectSameProcessedState(*it->second, *el.second);
    }
    EXPECT_EQ(actual.validTEInstancesOfType(type).size(), expected.validTEInstancesOfType(type).size());
  }

  ASSERT_EQ(actual.laneletInstances().size(), expected.laneletInstances().size());
  for (const auto& el : expected.laneletInstances()) {
    auto it = actual.laneletInstances().find(el.first);
    ASSERT_NE(it, actual.laneletInstances().end());
    EXPECT_EQ(it->second->valid(), el.second->valid()) << "lanelet " << el.first;
    EXPECT_EQ(it->second->wasCut(), el.second->wasCut()) << "lanelet " << el.first;
  }

  // The tensor data is buffered - taking it from the buffer must not return the data of the previous box
  MapData::TensorInstanceData tdActual = actual.getTensorInstanceData(true, false);
  MapData::TensorInstanceData tdExpected = expected.getTensorInstanceData(true, false);
  for (int t = static_cast<int>(LineStringType::RoadBorder); t <= static_cast<int>(LineStringType::ZebraCrossing);
       ++t) {
    auto type = static_cast<LineStringType>(t);
    SCOPED_TRACE("tensor data, LineStringType " + std::to_string(t));
    expectSameMatrices(tdActual.lineStringsOfType(type), tdExpected.lineStringsOfType(type));
    expectSameMatrices(tdActual.compoundLineStringsOfType(type), tdExpected.compoundLineStringsOfType(type));
  }
  for (int t = static_cast<int>(TEType::TLCar); t <= static_cast<int>(TEType::Unknown); ++t) {
    auto type = static_cast<TEType>(t);
    SCOPED_TRACE("tensor data, TEType " + std::to_string(t));
    expectSameMatrices(tdActual.teInstancesOfType(type), tdExpected.teInstancesOfType(type));
  }
  EXPECT_EQ(tdActual.teToCenterlineIndexEdges(), tdExpected.teToCenterlineIndexEdges());
  EXPECT_EQ(tdActual.teToTEIndexEdges(), tdExpected.teToTEIndexEdges());
}

size_t countValidLineStrings(const MapData& mapData) {
  size_t count = 0;
  for (int t = static_cast<int>(LineStringType::RoadBorder); t <= static_cast<int>(LineStringType::ZebraCrossing);
       ++t) {
    count += mapData.validLineStringsOfType(static_cast<LineStringType>(t)).size();
  }
  return count;
}

size_t countValidTEs(const MapData& mapData) {
  size_t count = 0;
  for (int t = static_cast<int>(TEType::TLCar); t <= static_cast<int>(TEType::Unknown); ++t) {
    count += mapData.validTEInstancesOfType(static_cast<TEType>(t)).size();
  }
  return count;
}

}  // namespace

class MapDataReprocessingTest : public MLConverterTest {
 protected:
  void SetUp() override {
    trafficRules = traffic_rules::TrafficRulesFactory::create(Locations::Germany, Participants::Vehicle);
    traffic_rules::TrafficRulesPtr bikeTrafficRules{
        traffic_rules::TrafficRulesFactory::create(Locations::Germany, Participants::Bicycle)};
    laneletMapGraph = routing::RoutingGraph::build(*laneletMap, *trafficRules);
    bikeMapGraph = routing::RoutingGraph::build(*laneletMap, *bikeTrafficRules);

    auto nonConstMap = std::const_pointer_cast<LaneletMap>(laneletMap);
    Lanelets allLanelets(nonConstMap->laneletLayer.begin(), nonConstMap->laneletLayer.end());
    auto submapUPtr = utils::createSubmap(allLanelets, {});
    LineStrings3d allLineStrings(nonConstMap->lineStringLayer.begin(), nonConstMap->lineStringLayer.end());
    for (const auto& lineString : allLineStrings) {
      submapUPtr->add(lineString);
    }
    laneletSubmap = std::shared_ptr<const LaneletSubmap>(std::move(submapUPtr));
  }

  MapDataPtr build() { return MapData::build(laneletSubmap, laneletMapGraph, trafficRules, bikeMapGraph); }

  static bool process(MapData& mapData, const OrientedRect& box) {
    return mapData.processAll(box, ParametrizationType::LineString, true, NPointsLanes, true, NPointsTE, 0.0, 0.0);
  }

  traffic_rules::TrafficRulesPtr trafficRules;
  routing::RoutingGraphConstPtr laneletMapGraph;
  routing::RoutingGraphConstPtr bikeMapGraph;
  LaneletSubmapConstPtr laneletSubmap;

  // Contains nothing of the test map, which lies within x in [-5, 15], y in [-10, 20]
  const OrientedRect farBbox{getRotatedRect(BasicPoint3d{705, 5, 0}, 15, 10, M_PI / 2.0, true)};
  // Contains only parts of the test map, so instances are cut differently than with bbox
  const OrientedRect partialBbox{getRotatedRect(BasicPoint3d{2, 0, 0}, 4, 3, 0.3, true)};
};

TEST_F(MapDataReprocessingTest, FarBoxThenFullBox) {  // NOLINT
  MapDataPtr fresh = build();
  bool freshValid = process(*fresh, bbox);
  ASSERT_GT(countValidLineStrings(*fresh), 0U);
  ASSERT_GT(countValidTEs(*fresh), 0U);

  MapDataPtr reused = build();
  EXPECT_FALSE(process(*reused, farBbox));
  EXPECT_EQ(countValidLineStrings(*reused), 0U);
  EXPECT_EQ(countValidTEs(*reused), 0U);
  // the tensor data of the far box is buffered here, processAll() has to drop it
  EXPECT_TRUE(reused->getTensorInstanceData(true, false).lineStringsOfType(LineStringType::RoadBorder).empty());

  EXPECT_EQ(process(*reused, bbox), freshValid);
  EXPECT_EQ(countValidLineStrings(*reused), countValidLineStrings(*fresh));
  EXPECT_EQ(countValidTEs(*reused), countValidTEs(*fresh));
  expectSameMapData(*reused, *fresh);
}

TEST_F(MapDataReprocessingTest, PartialBoxThenFullBox) {  // NOLINT
  MapDataPtr fresh = build();
  bool freshValid = process(*fresh, bbox);

  MapDataPtr reused = build();
  process(*reused, partialBbox);
  EXPECT_LT(countValidLineStrings(*reused), countValidLineStrings(*fresh));
  EXPECT_EQ(process(*reused, bbox), freshValid);
  expectSameMapData(*reused, *fresh);
}

TEST_F(MapDataReprocessingTest, FullBoxThenOtherBoxes) {  // NOLINT
  // going from more to fewer valid instances must not leave stale geometry behind
  for (const OrientedRect& second : {partialBbox, farBbox}) {
    MapDataPtr fresh = build();
    bool freshValid = process(*fresh, second);

    MapDataPtr reused = build();
    process(*reused, bbox);
    reused->getTensorInstanceData(true, false);  // fill the buffer
    EXPECT_EQ(process(*reused, second), freshValid);
    expectSameMapData(*reused, *fresh);
  }
}

TEST_F(MapDataReprocessingTest, RepeatedSameBoxIsStable) {  // NOLINT
  MapDataPtr fresh = build();
  bool freshValid = process(*fresh, bbox);

  MapDataPtr reused = build();
  process(*reused, bbox);
  process(*reused, farBbox);
  process(*reused, partialBbox);
  EXPECT_EQ(process(*reused, bbox), freshValid);
  expectSameMapData(*reused, *fresh);
}

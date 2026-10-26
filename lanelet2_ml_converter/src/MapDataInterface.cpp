#include "lanelet2_ml_converter/MapDataInterface.h"

#include <lanelet2_core/Exceptions.h>
#include <lanelet2_core/Forward.h>
#include <lanelet2_core/geometry/LineString.h>
#include <lanelet2_core/primitives/BasicRegulatoryElements.h>
#include <lanelet2_traffic_rules/TrafficRulesFactory.h>

#include <boost/geometry.hpp>
#include <type_traits>

#include "lanelet2_ml_converter/MapData.h"
#include "lanelet2_ml_converter/Utils.h"

namespace lanelet {
namespace ml_converter {

void MapDataInterface::setCurrPosAndExtractSubmap2d(const BasicPoint2d& pt, double yaw) {
  setCurrPosAndExtractSubmap(BasicPoint3d{pt.x(), pt.y(), 0}, yaw, 0, 0);
  currBbox_ = getRotatedRect(*currPos_, config_.submapExtentLongitudinal, config_.submapExtentLateral, *currYaw_, true);
}

void MapDataInterface::setCurrPosAndExtractSubmap(const BasicPoint3d& pt, double yaw) {
  setCurrPosAndExtractSubmap(pt, yaw, 0, 0);
  currBbox_ =
      getRotatedRect(*currPos_, config_.submapExtentLongitudinal, config_.submapExtentLateral, *currYaw_, false);
}

void MapDataInterface::setCurrPosAndExtractSubmap(const BasicPoint3d& pt, double yaw, double pitch, double roll) {
  currPos_ = pt;
  currYaw_ = yaw;
  currPitch_ = pitch;
  currRoll_ = roll;
  localSubmap_ =
      extractSubmap(laneletMap_, currPos_->head(2), config_.submapExtentLongitudinal, config_.submapExtentLateral);
  localSubmapGraph_ = lanelet::routing::RoutingGraph::build(*localSubmap_, *trafficRules_,
                                                            routing::defaultRoutingCosts(), routingGraphConfig());
  bikeSubmapGraph_ = lanelet::routing::RoutingGraph::build(*localSubmap_, *bikeTrafficRules_,
                                                           routing::defaultRoutingCosts(), routingGraphConfig());
  currBbox_ =
      getRotatedRect(*currPos_, config_.submapExtentLongitudinal, config_.submapExtentLateral, *currYaw_, false);
}

MapDataInterface::MapDataInterface(LaneletMapConstPtr laneletMap)
    : laneletMap_{laneletMap},
      config_{},
      trafficRules_{traffic_rules::TrafficRulesFactory::create(Locations::Germany, Participants::Vehicle)},
      bikeTrafficRules_{traffic_rules::TrafficRulesFactory::create(Locations::Germany, Participants::Bicycle)} {}

MapDataInterface::MapDataInterface(LaneletMapConstPtr laneletMap, Configuration config)
    : laneletMap_{laneletMap},
      config_{config},
      trafficRules_{traffic_rules::TrafficRulesFactory::create(Locations::Germany, Participants::Vehicle)},
      bikeTrafficRules_{traffic_rules::TrafficRulesFactory::create(Locations::Germany, Participants::Bicycle)} {}

routing::RoutingGraph::Configuration MapDataInterface::routingGraphConfig() const {
  routing::RoutingGraph::Configuration graphConfig;
  if (config_.allowLaneChangeAcrossBicycleLane) {
    graphConfig.emplace(routing::RoutingGraph::AllowLaneChangeAcrossBicycleLane, true);
  }
  return graphConfig;
}

MapDataPtr MapDataInterface::getMapData(LaneletSubmapConstPtr localSubmap, const OrientedRect& bbox,
                                        lanelet::routing::RoutingGraphConstPtr localSubmapGraph,
                                        lanelet::routing::RoutingGraphConstPtr bikeSubmapGraph, double pitch,
                                        double roll, bool processAll) {
  MapDataPtr mapData =
      MapData::build(localSubmap, localSubmapGraph, trafficRules_, bikeSubmapGraph, config_.ignoreMapElevation,
                     config_.lineStringTypeGrouping, config_.teTypeGrouping);
  if (processAll) {
    mapData->processAll(bbox, config_.paramType, config_.resampleLanes, config_.nPointsLanes, config_.resampleTE,
                        config_.nPointsTE, pitch, roll);
  }
  return mapData;
}

std::vector<MapDataPtr> MapDataInterface::mapDataBatch2d(std::vector<BasicPoint2d> pts, std::vector<double> yaws) {
  if (pts.size() != yaws.size()) {
    throw std::runtime_error("Unequal sizes of pts and rotation angles!");
  }
  std::vector<MapDataPtr> mDataVec;
  for (size_t i = 0; i < pts.size(); i++) {
    LaneletSubmapConstPtr localSubmap =
        extractSubmap(laneletMap_, pts[i], config_.submapExtentLongitudinal, config_.submapExtentLateral);
    routing::RoutingGraphConstPtr localSubmapGraph = lanelet::routing::RoutingGraph::build(
        *localSubmap, *trafficRules_, routing::defaultRoutingCosts(), routingGraphConfig());
    routing::RoutingGraphConstPtr bikeSubmapGraph = lanelet::routing::RoutingGraph::build(
        *localSubmap, *bikeTrafficRules_, routing::defaultRoutingCosts(), routingGraphConfig());
    OrientedRect bbox = getRotatedRect(BasicPoint3d{pts[i].x(), pts[i].y(), 0}, config_.submapExtentLongitudinal,
                                       config_.submapExtentLateral, yaws[i], true);
    mDataVec.push_back(getMapData(localSubmap, bbox, localSubmapGraph, bikeSubmapGraph, 0, 0, true));
  }
  return mDataVec;
}

std::vector<MapDataPtr> MapDataInterface::mapDataBatch(std::vector<BasicPoint3d> pts, std::vector<double> yaws) {
  return mapDataBatch(pts, yaws, std::vector<double>(yaws.size(), 0), std::vector<double>(yaws.size(), 0));
}

std::vector<MapDataPtr> MapDataInterface::mapDataBatch(std::vector<BasicPoint3d> pts, std::vector<double> yaws,
                                                       std::vector<double> pitches, std::vector<double> rolls) {
  if ((pts.size() != yaws.size()) || (pts.size() != pitches.size()) || (pts.size() != rolls.size())) {
    throw std::runtime_error("Unequal sizes of pts and rotation angles!");
  }
  std::vector<MapDataPtr> mDataVec;
  for (size_t i = 0; i < pts.size(); i++) {
    LaneletSubmapConstPtr localSubmap =
        extractSubmap(laneletMap_, pts[i].head(2), config_.submapExtentLongitudinal, config_.submapExtentLateral);
    routing::RoutingGraphConstPtr localSubmapGraph = lanelet::routing::RoutingGraph::build(
        *localSubmap, *trafficRules_, routing::defaultRoutingCosts(), routingGraphConfig());
    routing::RoutingGraphConstPtr bikeSubmapGraph = lanelet::routing::RoutingGraph::build(
        *localSubmap, *bikeTrafficRules_, routing::defaultRoutingCosts(), routingGraphConfig());
    OrientedRect bbox =
        getRotatedRect(pts[i], config_.submapExtentLongitudinal, config_.submapExtentLateral, yaws[i], false);
    mDataVec.push_back(getMapData(localSubmap, bbox, localSubmapGraph, bikeSubmapGraph, pitches[i], rolls[i], true));
  }
  return mDataVec;
}

MapDataPtr MapDataInterface::mapData(bool processAll) {
  if (!currPos_) {
    throw InvalidObjectStateError(
        "Your current position is not set! Call setCurrPosAndExtractSubmap() before trying to get the data!");
  }
  if (!currYaw_) {
    throw InvalidObjectStateError(
        "Your current yaw angle is not set! Call setCurrPosAndExtractSubmap() before trying to get the data!");
  }
  if (!currPitch_) {
    throw InvalidObjectStateError(
        "Your current pitch angle is not set! Call setCurrPosAndExtractSubmap() before trying to get the data!");
  }
  if (!currRoll_) {
    throw InvalidObjectStateError(
        "Your current roll angle is not set! Call setCurrPosAndExtractSubmap() before trying to get the data!");
  }
  return getMapData(localSubmap_, *currBbox_, localSubmapGraph_, bikeSubmapGraph_, *currPitch_, *currRoll_, processAll);
}

}  // namespace ml_converter
}  // namespace lanelet
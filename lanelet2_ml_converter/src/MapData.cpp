#include "lanelet2_ml_converter/MapData.h"

#include <cmath>
#include <iostream>

#include "lanelet2_core/Attribute.h"
#include "lanelet2_core/geometry/LineString.h"
#include "lanelet2_core/geometry/Polygon.h"
#include "lanelet2_ml_converter/Utils.h"

namespace lanelet {
namespace ml_converter {

using namespace internal;

MapDataPtr MapData::build(LaneletSubmapConstPtr& localSubmap, lanelet::routing::RoutingGraphConstPtr localSubmapGraph,
                          traffic_rules::TrafficRulesPtr trafficRules,
                          lanelet::routing::RoutingGraphConstPtr bikeSubmapGraph, bool ignoreMapElevation,
                          const LineStringTypeGrouping& lineStringTypeGrouping, const TETypeGrouping& teTypeGrouping) {
  checkLineStringTypeGroupingCoverage(lineStringTypeGrouping);

  MapDataPtr data = std::make_shared<MapData>();
  data->lineStringTypeGrouping_ = lineStringTypeGrouping;
  data->teTypeGrouping_ = teTypeGrouping;
  data->initLeftBoundaries(localSubmap, localSubmapGraph, trafficRules, ignoreMapElevation);
  data->initRightBoundaries(localSubmap, localSubmapGraph, trafficRules, ignoreMapElevation);
  data->initLaneletInstances(localSubmap, localSubmapGraph, trafficRules, ignoreMapElevation);
  data->initCompoundInstances(localSubmap, localSubmapGraph, trafficRules, bikeSubmapGraph, ignoreMapElevation);

  // Collect non-lane traffic elements
  data->collectStopLines(localSubmap, ignoreMapElevation);
  data->collectArrows(localSubmap, ignoreMapElevation);
  data->collectTrafficLights(localSubmap, ignoreMapElevation);
  data->collectTrafficSigns(localSubmap, ignoreMapElevation);
  data->collectSymbols(localSubmap, ignoreMapElevation);
  data->collectPedestrianCrossings(localSubmap, ignoreMapElevation);

  // Update association indices after all compound instances are collected
  data->updateAssociatedCpdInstanceIndices();

  // Convert TE edges to instance associations
  data->convertTEEdges();

  return data;
}

void MapData::initLeftBoundaries(LaneletSubmapConstPtr& localSubmap,
                                 lanelet::routing::RoutingGraphConstPtr localSubmapGraph,
                                 traffic_rules::TrafficRulesPtr trafficRules, bool ignoreMapElevation) {
  for (const auto& ll : localSubmap->laneletLayer) {
    Id boundID = ll.leftBound3d().id();
    BasicLineString3d bound =
        ll.leftBound3d().inverted() ? ll.leftBound3d().invert().basicLineString() : ll.leftBound3d().basicLineString();
    if (ignoreMapElevation) {
      for (auto& pt : bound) {
        pt[2] = 0;
      }
    }
    LaneLineStringInstances::iterator itLineString = laneLineStrings_.find(boundID);
    if (itLineString != laneLineStrings_.end()) {
      itLineString->second->addLaneletID(ll.id());
    } else {
      laneLineStrings_.insert({boundID, std::make_shared<LaneLineStringInstance>(
                                            bound, boundID, bdTypeToEnum(ll.leftBound3d()), Ids{ll.id()}, false)});
    }

    Optional<ConstLanelet> leftLL = localSubmapGraph->left(ll);
    Optional<ConstLanelet> adjLeftLL = localSubmapGraph->adjacentLeft(ll);

    if (leftLL) {
      llEdges_[ll.id()].push_back(Edge(ll.id(), leftLL->id(), true));
    }
  }
}

void MapData::initRightBoundaries(LaneletSubmapConstPtr& localSubmap,
                                  lanelet::routing::RoutingGraphConstPtr localSubmapGraph,
                                  traffic_rules::TrafficRulesPtr trafficRules, bool ignoreMapElevation) {
  for (const auto& ll : localSubmap->laneletLayer) {
    Id boundID = ll.rightBound3d().id();
    BasicLineString3d bound = ll.rightBound3d().inverted() ? ll.rightBound3d().invert().basicLineString()
                                                           : ll.rightBound3d().basicLineString();
    if (ignoreMapElevation) {
      for (auto& pt : bound) {
        pt[2] = 0;
      }
    }
    LaneLineStringInstances::iterator itLineString = laneLineStrings_.find(boundID);
    if (itLineString != laneLineStrings_.end()) {
      itLineString->second->addLaneletID(ll.id());
    } else {
      laneLineStrings_.insert({boundID, std::make_shared<LaneLineStringInstance>(
                                            bound, boundID, bdTypeToEnum(ll.rightBound3d()), Ids{ll.id()}, false)});
    }
    Optional<ConstLanelet> rightLL = localSubmapGraph->right(ll);
    Optional<ConstLanelet> adjRightLL = localSubmapGraph->adjacentRight(ll);
    if (rightLL) {
      llEdges_[ll.id()].push_back(Edge(ll.id(), rightLL->id(), true));
    }
  }
}

void MapData::initLaneletInstances(LaneletSubmapConstPtr& localSubmap,
                                   lanelet::routing::RoutingGraphConstPtr localSubmapGraph,
                                   traffic_rules::TrafficRulesPtr trafficRules, bool ignoreMapElevation) {
  for (const auto& ll : localSubmap->laneletLayer) {
    Attribute subtype = ll.attributeOr(AttributeName::Subtype, "");
    // Non-driving lanelets do not become lanelet instances
    if (subtype == AttributeValueString::Crosswalk || subtype == AttributeValueString::Walkway ||
        subtype == AttributeValueString::SharedWalkway || subtype == AttributeValueString::Stairs) {
      continue;
    }

    LaneLineStringInstancePtr leftBoundary = getLineStringFeatFromId(ll.leftBound().id(), ll.leftBound().inverted());
    LaneLineStringInstancePtr rightBoundary = getLineStringFeatFromId(ll.rightBound().id(), ll.rightBound().inverted());
    BasicLineString3d centerlineLString = ll.centerline3d().basicLineString();
    if (ignoreMapElevation) {
      for (auto& pt : centerlineLString) {
        pt[2] = 0;
      }
    }

    // Determine if this is a bike lane based on subtype attribute
    LineStringType centerlineType =
        (subtype == AttributeValueString::BicycleLane) ? LineStringType::BikeCenterline : LineStringType::Centerline;

    LaneLineStringInstancePtr centerline = std::make_shared<LaneLineStringInstance>(
        centerlineLString, ll.centerline3d().id(), centerlineType, Ids{ll.id()}, ll.centerline3d().inverted());
    laneletInstances_.insert(
        {ll.id(), std::make_shared<LaneletInstance>(leftBoundary, rightBoundary, centerline, ll.id())});
  }
}

bool isLaneletInPath(const ConstLanelets& path, const ConstLanelet& ll) {
  for (const auto& el : path) {
    if (ll.id() == el.id()) {
      return true;
    }
  }
  return false;
}

// Idea: algorithm for paths that starts with LLs with no previous, splits on junctions and terminates on LLs with
// no successors
void MapData::getPaths(lanelet::routing::RoutingGraphConstPtr localSubmapGraph, std::vector<ConstLanelets>& paths,
                       ConstLanelet start, ConstLanelets initPath) {
  initPath.push_back(start);
  ConstLanelet current = start;

  // Get the subtype of the starting lanelet to maintain consistency along the path
  Attribute startSubtype = start.attributeOr(AttributeName::Subtype, "");

  ConstLanelets successorLLs = localSubmapGraph->following(current, false);

  // Filter successors to only include those with matching subtype
  ConstLanelets filteredSuccessors;
  for (const auto& successor : successorLLs) {
    Attribute successorSubtype = successor.attributeOr(AttributeName::Subtype, "");
    if (successorSubtype == startSubtype) {
      filteredSuccessors.push_back(successor);
    }
  }
  successorLLs = filteredSuccessors;

  while (!successorLLs.empty()) {
    for (size_t i = 1; i != successorLLs.size(); i++) {
      if (isLaneletInPath(initPath, successorLLs[i])) {
        continue;
      }
      llEdges_[current.id()].push_back(Edge(current.id(), successorLLs[i].id(), false));
      getPaths(localSubmapGraph, paths, successorLLs[i], initPath);
    }
    if (isLaneletInPath(initPath, successorLLs.front())) {
      break;
    }
    initPath.push_back(successorLLs.front());
    llEdges_[current.id()].push_back(Edge(current.id(), successorLLs.front().id(), false));
    current = successorLLs.front();
    successorLLs = localSubmapGraph->following(current, false);

    // Filter successors again for the new current lanelet
    filteredSuccessors.clear();
    for (const auto& successor : successorLLs) {
      Attribute successorSubtype = successor.attributeOr(AttributeName::Subtype, "");
      if (successorSubtype == startSubtype) {
        filteredSuccessors.push_back(successor);
      }
    }
    successorLLs = filteredSuccessors;
  }
  paths.push_back(initPath);
}

LineStringType MapData::getLineStringTypeFromId(Id id) {
  LaneLineStringInstances::iterator it = laneLineStrings_.find(id);
  if (it != laneLineStrings_.end()) {
    return it->second->type();
  } else {
    throw std::runtime_error("Lanelet boundary " + std::to_string(id) + " is not in the instance list!");
  }
}

LaneLineStringInstancePtr makeInverted(const LaneLineStringInstancePtr& feat) {
  return std::make_shared<LaneLineStringInstance>(
      BasicLineString3d(feat->rawInstance().rbegin(), feat->rawInstance().rend()), feat->mapID(), feat->type(),
      feat->laneletIDs(), !feat->inverted());
}

bool pointsMatchIn2d(const BasicPoint3d& p1, const BasicPoint3d& p2) {
  constexpr double kPointMatchTolerance = 1e-2;
  return std::abs(p1.x() - p2.x()) <= kPointMatchTolerance && std::abs(p1.y() - p2.y()) <= kPointMatchTolerance;
}

double pointDistanceSquared2d(const BasicPoint3d& p1, const BasicPoint3d& p2) {
  const double dx = p1.x() - p2.x();
  const double dy = p1.y() - p2.y();
  return dx * dx + dy * dy;
}

BasicLineString3d orientBorderForClosedPerimeter(const BasicLineString3d& firstBorder,
                                                 const BasicLineString3d& secondBorder) {
  if (firstBorder.empty() || secondBorder.empty()) {
    return secondBorder;
  }

  // Pick the orientation whose endpoint pairings produce the shorter two closing edges.
  const double asIsConnectionCost = pointDistanceSquared2d(firstBorder.back(), secondBorder.front()) +
                                    pointDistanceSquared2d(secondBorder.back(), firstBorder.front());
  const double reversedConnectionCost = pointDistanceSquared2d(firstBorder.back(), secondBorder.back()) +
                                        pointDistanceSquared2d(secondBorder.front(), firstBorder.front());

  if (asIsConnectionCost <= reversedConnectionCost) {
    return secondBorder;
  }
  return BasicLineString3d(secondBorder.rbegin(), secondBorder.rend());
}

LaneLineStringInstancePtr MapData::getLineStringFeatFromId(Id id, bool inverted) {
  LaneLineStringInstances::iterator it = laneLineStrings_.find(id);
  if (it != laneLineStrings_.end()) {
    return (inverted == it->second->inverted()) ? it->second : makeInverted(it->second);
  } else {
    throw std::runtime_error("Lanelet boundary " + std::to_string(id) + " is not in the instance list!");
  }
}

std::vector<CompoundElsList> MapData::computeCompoundLeftBorders(const ConstLanelets& path) {
  std::vector<CompoundElsList> compoundBorders;
  ConstLanelet start = path.front();
  LineStringType currType = getLineStringTypeFromId(start.leftBound3d().id());
  LineStringType currRepType = getLineStringTypeRepresentative(currType, lineStringTypeGrouping_);
  int currGroupIdx = getLineStringTypeGroupIndex(currType, lineStringTypeGrouping_);

  compoundBorders.push_back(CompoundElsList{start.leftBound3d().id(), start.leftBound3d().inverted(), currRepType});

  for (size_t i = 1; i != path.size(); i++) {
    LineStringType newType = getLineStringTypeFromId(path[i].leftBound3d().id());
    LineStringType newRepType = getLineStringTypeRepresentative(newType, lineStringTypeGrouping_);
    int newGroupIdx = getLineStringTypeGroupIndex(newType, lineStringTypeGrouping_);
    if (currGroupIdx == newGroupIdx) {
      compoundBorders.back().ids.push_back(path[i].leftBound3d().id());
      compoundBorders.back().inverted.push_back(path[i].leftBound3d().inverted());
    } else {
      compoundBorders.push_back(
          CompoundElsList{path[i].leftBound3d().id(), path[i].leftBound3d().inverted(), newRepType});
      currType = newType;
      currRepType = newRepType;
      currGroupIdx = newGroupIdx;
    }
  }
  return compoundBorders;
}

std::vector<CompoundElsList> MapData::computeCompoundRightBorders(const ConstLanelets& path) {
  std::vector<CompoundElsList> compoundBorders;
  ConstLanelet start = path.front();
  LineStringType currType = getLineStringTypeFromId(start.rightBound3d().id());
  LineStringType currRepType = getLineStringTypeRepresentative(currType, lineStringTypeGrouping_);
  int currGroupIdx = getLineStringTypeGroupIndex(currType, lineStringTypeGrouping_);

  compoundBorders.push_back(CompoundElsList{start.rightBound3d().id(), start.rightBound3d().inverted(), currRepType});

  for (size_t i = 1; i != path.size(); i++) {
    LineStringType newType = getLineStringTypeFromId(path[i].rightBound3d().id());
    LineStringType newRepType = getLineStringTypeRepresentative(newType, lineStringTypeGrouping_);
    int newGroupIdx = getLineStringTypeGroupIndex(newType, lineStringTypeGrouping_);
    if (currGroupIdx == newGroupIdx) {
      compoundBorders.back().ids.push_back(path[i].rightBound3d().id());
      compoundBorders.back().inverted.push_back(path[i].rightBound3d().inverted());
    } else {
      compoundBorders.push_back(
          CompoundElsList{path[i].rightBound3d().id(), path[i].rightBound3d().inverted(), newRepType});
      currType = newType;
      currRepType = newRepType;
      currGroupIdx = newGroupIdx;
    }
  }
  return compoundBorders;
}

CompoundLaneLineStringInstancePtr MapData::computeCompoundCenterline(const ConstLanelets& path,
                                                                     bool ignoreMapElevation) {
  LaneLineStringInstanceList compoundCenterlines;

  // Determine centerline type from first lanelet in path
  Attribute subtype = path.front().attributeOr(AttributeName::Subtype, "");
  LineStringType centerlineType =
      (subtype == AttributeValueString::BicycleLane) ? LineStringType::BikeCenterline : LineStringType::Centerline;

  for (const auto& ll : path) {
    BasicLineString3d centerlineLString = ll.centerline3d().basicLineString();
    if (ignoreMapElevation) {
      for (auto& pt : centerlineLString) {
        pt[2] = 0;
      }
    }
    compoundCenterlines.push_back(std::make_shared<LaneLineStringInstance>(centerlineLString, ll.id(), centerlineType,
                                                                           Ids{ll.id()}, ll.centerline3d().inverted()));
  }
  return std::make_shared<CompoundLaneLineStringInstance>(compoundCenterlines, centerlineType);
}

void MapData::computeDrivableAreaBorders(LaneletSubmapConstPtr& localSubmap, bool ignoreMapElevation) {
  // Collect all LineStrings from the map with drivable_space_border attribute
  std::vector<Id> drivableAreaIds;
  std::map<Id, std::pair<Id, Id>> endpointIdMap;  // Maps linestring ID to (first_point_id, last_point_id)
  std::map<Id, std::pair<BasicPoint3d, BasicPoint3d>> endpointCoordMap;

  for (const auto& lineString : localSubmap->lineStringLayer) {
    Attribute drivableAreaBorder = lineString.attributeOr("drivable_space_border", "");
    if (drivableAreaBorder == "true") {
      Id lsId = lineString.id();
      BasicLineString3d lsBasic = lineString.basicLineString();
      if (ignoreMapElevation) {
        for (auto& pt : lsBasic) {
          pt[2] = 0;
        }
      }

      // Add to laneLineStrings_ if not already present
      LaneLineStringInstances::iterator it = laneLineStrings_.find(lsId);
      if (it == laneLineStrings_.end()) {
        laneLineStrings_.insert({lsId, std::make_shared<LaneLineStringInstance>(
                                           lsBasic, lsId, LineStringType::DrivableArea, Ids{}, false)});
      }

      // Store the point IDs of the endpoints for connectivity checking
      if (lineString.size() >= 2) {
        drivableAreaIds.push_back(lsId);
        endpointIdMap[lsId] = {lineString.front().id(), lineString.back().id()};
        endpointCoordMap[lsId] = {BasicPoint3d(lineString.front().x(), lineString.front().y(), lineString.front().z()),
                                  BasicPoint3d(lineString.back().x(), lineString.back().y(), lineString.back().z())};
      }
    }
  }

  auto getDrivableAreaFeatureInEndpointDirection = [&](Id id, bool followEndpointOrder) {
    LaneLineStringInstancePtr feature = getLineStringFeatFromId(id, false);
    const auto& rawInstance = feature->rawInstance();
    if (rawInstance.empty()) {
      throw std::runtime_error("Drivable area feature " + std::to_string(id) + " has empty raw geometry!");
    }

    const auto endpointCoordIt = endpointCoordMap.find(id);
    if (endpointCoordIt == endpointCoordMap.end()) {
      throw std::runtime_error("Drivable area feature " + std::to_string(id) + " has no endpoint coordinates!");
    }

    const auto& expectedEndpoints = endpointCoordIt->second;
    const bool sameDirection = pointsMatchIn2d(rawInstance.front(), expectedEndpoints.first) &&
                               pointsMatchIn2d(rawInstance.back(), expectedEndpoints.second);
    const bool reversedDirection = pointsMatchIn2d(rawInstance.front(), expectedEndpoints.second) &&
                                   pointsMatchIn2d(rawInstance.back(), expectedEndpoints.first);

    if (!sameDirection && !reversedDirection) {
      throw std::runtime_error("Drivable area feature " + std::to_string(id) +
                               " geometry does not match the source endpoints!");
    }

    LaneLineStringInstancePtr endpointAlignedFeature = sameDirection ? feature : makeInverted(feature);
    return followEndpointOrder ? endpointAlignedFeature : makeInverted(endpointAlignedFeature);
  };

  // Build adjacency list for connected components based on point IDs
  std::map<Id, std::vector<Id>> adjacency;

  for (size_t i = 0; i < drivableAreaIds.size(); ++i) {
    for (size_t j = i + 1; j < drivableAreaIds.size(); ++j) {
      Id id1 = drivableAreaIds[i];
      Id id2 = drivableAreaIds[j];

      const auto& endpoints1 = endpointIdMap[id1];
      const auto& endpoints2 = endpointIdMap[id2];

      // Check if they share an endpoint (based on point IDs)
      bool connected = endpoints1.first == endpoints2.first || endpoints1.first == endpoints2.second ||
                       endpoints1.second == endpoints2.first || endpoints1.second == endpoints2.second;

      if (connected) {
        adjacency[id1].push_back(id2);
        adjacency[id2].push_back(id1);
      }
    }
  }

  // Find connected components using DFS
  std::set<Id> visited;
  for (const auto& id : drivableAreaIds) {
    if (visited.find(id) != visited.end()) {
      continue;
    }

    // DFS to find component
    std::vector<Id> component;
    std::vector<Id> stack;
    stack.push_back(id);

    while (!stack.empty()) {
      Id current = stack.back();
      stack.pop_back();

      if (visited.find(current) != visited.end()) {
        continue;
      }

      visited.insert(current);
      component.push_back(current);

      for (const auto& neighbor : adjacency[current]) {
        if (visited.find(neighbor) == visited.end()) {
          stack.push_back(neighbor);
        }
      }
    }

    // Handle linestrings in component - build chains starting from nodes with dangling endpoints
    std::set<size_t> usedIndices;

    while (usedIndices.size() < component.size()) {
      // Find the first unused linestring, preferring nodes with dangling endpoints
      size_t startIdx = component.size();
      bool startFirstEndpointDangling = false;
      bool startSecondEndpointDangling = false;

      // First, try to find an unused node where either endpoint is not connected to neighbors
      for (size_t i = 0; i < component.size(); ++i) {
        if (usedIndices.count(i)) continue;

        Id nodeId = component[i];
        const auto& nodeEndpoints = endpointIdMap[nodeId];

        // Check if either endpoint is dangling (not matching any neighbor's endpoints)
        bool firstEndpointDangling = true;
        bool secondEndpointDangling = true;

        for (const auto& neighborId : adjacency[nodeId]) {
          const auto& neighborEndpoints = endpointIdMap[neighborId];

          if (nodeEndpoints.first == neighborEndpoints.first || nodeEndpoints.first == neighborEndpoints.second) {
            firstEndpointDangling = false;
          }
          if (nodeEndpoints.second == neighborEndpoints.first || nodeEndpoints.second == neighborEndpoints.second) {
            secondEndpointDangling = false;
          }
        }

        if (firstEndpointDangling || secondEndpointDangling) {
          startIdx = i;
          startFirstEndpointDangling = firstEndpointDangling;
          startSecondEndpointDangling = secondEndpointDangling;
          break;
        }
      }

      // If no dangling endpoint found, use the first unused node
      if (startIdx >= component.size()) {
        for (size_t i = 0; i < component.size(); ++i) {
          if (!usedIndices.count(i)) {
            startIdx = i;
            break;
          }
        }
      }

      if (startIdx >= component.size()) {
        break;  // All linestrings have been used
      }

      // Start a new chain with this linestring
      LaneLineStringInstanceList compoundFeatures;
      usedIndices.insert(startIdx);

      // Determine which endpoint to start from based on dangling status
      Id startNodeId = component[startIdx];
      const auto& startEndpoints = endpointIdMap[startNodeId];
      Id currentEndpoint = startEndpoints.second;  // Default to second endpoint

      // If first endpoint is dangling, start from the second one; otherwise start from the first
      if (startFirstEndpointDangling && !startSecondEndpointDangling) {
        currentEndpoint = startEndpoints.second;
        compoundFeatures.push_back(getDrivableAreaFeatureInEndpointDirection(component[startIdx], true));
      } else {
        currentEndpoint = startEndpoints.first;
        compoundFeatures.push_back(getDrivableAreaFeatureInEndpointDirection(component[startIdx], false));
      }

      // Build the continuous chain
      bool foundConnection = true;
      while (foundConnection && usedIndices.size() < component.size()) {
        foundConnection = false;

        for (size_t i = 0; i < component.size(); ++i) {
          if (usedIndices.count(i)) continue;

          const auto& endpoints = endpointIdMap[component[i]];

          // Check if this linestring connects to current endpoint
          if (endpoints.first == currentEndpoint) {
            // Connects normally, no inversion needed
            compoundFeatures.push_back(getDrivableAreaFeatureInEndpointDirection(component[i], true));
            currentEndpoint = endpoints.second;
            usedIndices.insert(i);
            foundConnection = true;
            break;
          } else if (endpoints.second == currentEndpoint) {
            // Connects with inversion - manually invert the linestring
            compoundFeatures.push_back(getDrivableAreaFeatureInEndpointDirection(component[i], false));
            currentEndpoint = endpoints.first;
            usedIndices.insert(i);
            foundConnection = true;
            break;
          }
        }
      }

      // Create a compound instance for this chain
      if (!compoundFeatures.empty()) {
        compoundLaneLineStrings_.push_back(
            std::make_shared<CompoundLaneLineStringInstance>(compoundFeatures, LineStringType::DrivableArea));
      }
    }
  }
}

// Template helper to find nearest intersecting lanelet for both linestrings and polygons
template <typename T>
Optional<Id> findNearestIntersectingLanelet(const T& element, LaneletSubmapConstPtr& localSubmap,
                                            bool requireIntersection = true, double maxDistance = 2.0) {
  // Convert to BasicLineString3d - works for both ConstLineString3d and ConstPolygon3d
  BasicLineString3d basicElement = element.basicLineString();

  // Convert to 2D linestring directly
  BasicLineString2d ls2d;
  for (const auto& pt : basicElement) {
    ls2d.push_back(BasicPoint2d(pt.x(), pt.y()));
  }

  // Calculate centroid for nearest lanelet query
  BasicPoint2d centroid(0.0, 0.0);
  for (const auto& pt : basicElement) {
    centroid.x() += pt.x();
    centroid.y() += pt.y();
  }
  centroid.x() /= basicElement.size();
  centroid.y() /= basicElement.size();

  // Get only the nearest lanelet to the centroid
  ConstLanelets nearestLanelets = localSubmap->laneletLayer.nearest(centroid, 5);

  if (!nearestLanelets.empty()) {
    for (const auto& ll : nearestLanelets) {
      // Get lanelet's 2D polygon
      BasicPolygon2d laneletPolygon;
      for (const auto& pt : ll.polygon3d()) {
        laneletPolygon.push_back(BasicPoint2d(pt.x(), pt.y()));
      }

      if (requireIntersection) {
        // Check if linestring intersects with lanelet polygon
        if (boost::geometry::intersects(ls2d, laneletPolygon)) {
          return ll.id();
        }
      } else {
        // Check if distance to lanelet centerline is below threshold
        BasicLineString2d centerline2d;
        for (const auto& pt : ll.centerline()) {
          centerline2d.push_back(BasicPoint2d(pt.x(), pt.y()));
        }
        double distance = boost::geometry::distance(ls2d, centerline2d);
        if (distance <= maxDistance) {
          return ll.id();
        }
      }
    }
  }
  return {};
}

void MapData::collectStopLines(LaneletSubmapConstPtr& localSubmap, bool ignoreMapElevation) {
  auto processElement = [&](const auto& element) {
    Attribute type = element.attributeOr(AttributeName::Type, "");
    if (type == "stop_line") {
      Attribute artificial = element.attributeOr("artificial", "");
      if (artificial == 1) {
        return;
      }

      Id lsId = element.id();
      BasicLineString3d lsBasic = element.basicLineString();

      if (ignoreMapElevation) {
        for (auto& pt : lsBasic) {
          pt[2] = 0;
        }
      }

      TEType teType = teTypeToEnum(element);
      TEType representativeType = getTETypeRepresentative(teType, teTypeGrouping_);
      teInstances_.insert({lsId, std::make_shared<TEInstance>(lsBasic, lsId, representativeType)});

      // Find associated lanelets via regulatory elements
      auto regElemsOwningLs = localSubmap->regulatoryElementLayer.findUsages(element);
      for (const auto& regElem : regElemsOwningLs) {
        // Find lanelets that reference this regulatory element
        auto laneletsOwningRegelem = localSubmap->laneletLayer.findUsages(regElem);
        for (const auto& lanelet : laneletsOwningRegelem) {
          teEdges_[lsId].push_back(Edge(lsId, lanelet.id(), false));
        }
      }
    }
  };

  for (const auto& lineString : localSubmap->lineStringLayer) {
    processElement(lineString);
  }
  for (const auto& polygon : localSubmap->polygonLayer) {
    processElement(polygon);
  }
}

void MapData::collectArrows(LaneletSubmapConstPtr& localSubmap, bool ignoreMapElevation) {
  auto processElement = [&](const auto& element) {
    Attribute type = element.attributeOr(AttributeName::Type, "");
    if (type == "arrow") {
      Id lsId = element.id();
      BasicLineString3d lsBasic = element.basicLineString();

      if (ignoreMapElevation) {
        for (auto& pt : lsBasic) {
          pt[2] = 0;
        }
      }

      TEType teType = teTypeToEnum(element);
      TEType representativeType = getTETypeRepresentative(teType, teTypeGrouping_);
      teInstances_.insert({lsId, std::make_shared<TEInstance>(lsBasic, lsId, representativeType)});

      // Find the nearest intersecting lanelet
      Optional<Id> laneletId = findNearestIntersectingLanelet(element, localSubmap);
      if (laneletId) {
        teEdges_[lsId].push_back(Edge(lsId, *laneletId, false));
      }
    }
  };

  for (const auto& lineString : localSubmap->lineStringLayer) {
    processElement(lineString);
  }
  for (const auto& polygon : localSubmap->polygonLayer) {
    processElement(polygon);
  }
}

void MapData::collectTrafficLights(LaneletSubmapConstPtr& localSubmap, bool ignoreMapElevation) {
  auto processElement = [&](const auto& element) {
    Attribute type = element.attributeOr(AttributeName::Type, "");
    if (type == AttributeValueString::TrafficLight || type == "traffic_light_pedestrians" ||
        type == "traffic_light_bikes") {
      Id lsId = element.id();
      BasicLineString3d lsBasic = element.basicLineString();

      // Skip elements with elevation lower than -1000m (e.g., unlifted elements marked with FLOAT_MIN)
      bool hasLowElevation = false;
      for (const auto& pt : lsBasic) {
        if (pt[2] < -1000.0) {
          hasLowElevation = true;
          break;
        }
      }
      if (hasLowElevation) {
        return;
      }

      if (ignoreMapElevation) {
        for (auto& pt : lsBasic) {
          pt[2] = 0;
        }
      }

      TEType teType = teTypeToEnum(element);
      TEType representativeType = getTETypeRepresentative(teType, teTypeGrouping_);
      teInstances_.insert({lsId, std::make_shared<TEInstance>(lsBasic, lsId, representativeType)});

      // Find associated stop line via regulatory elements
      auto regElemsOwningLs = localSubmap->regulatoryElementLayer.findUsages(element);
      for (const auto& regElem : regElemsOwningLs) {
        // Check if this is a TrafficLight regulatory element
        if (regElem->attribute(AttributeName::Subtype).value() == "traffic_light") {
          // Get the stop line from the regulatory element using template syntax
          auto stopLines = regElem->template getParameters<ConstLineString3d>(RoleName::RefLine);
          if (!stopLines.empty()) {
            Id stopLineId = stopLines.front().id();
            // Check if this stop line is in our teInstances_ (already collected)
            if (teInstances_.find(stopLineId) != teInstances_.end()) {
              teEdges_[lsId].push_back(Edge(lsId, stopLineId, false));
            }
          }
        }
      }
    }
  };

  for (const auto& lineString : localSubmap->lineStringLayer) {
    processElement(lineString);
  }
  for (const auto& polygon : localSubmap->polygonLayer) {
    processElement(polygon);
  }
}

void MapData::collectTrafficSigns(LaneletSubmapConstPtr& localSubmap, bool ignoreMapElevation) {
  auto processElement = [&](const auto& element) {
    Attribute type = element.attributeOr(AttributeName::Type, "");
    if (type == AttributeValueString::TrafficSign) {
      Id lsId = element.id();
      BasicLineString3d lsBasic = element.basicLineString();

      // Skip elements with elevation lower than -1000m (e.g., unlifted elements marked with FLOAT_MIN)
      bool hasLowElevation = false;
      for (const auto& pt : lsBasic) {
        if (pt[2] < -1000.0) {
          hasLowElevation = true;
          break;
        }
      }
      if (hasLowElevation) {
        return;
      }

      if (ignoreMapElevation) {
        for (auto& pt : lsBasic) {
          pt[2] = 0;
        }
      }

      TEType teType = teTypeToEnum(element);
      TEType representativeType = getTETypeRepresentative(teType, teTypeGrouping_);
      teInstances_.insert({lsId, std::make_shared<TEInstance>(lsBasic, lsId, representativeType)});
    }
  };

  for (const auto& lineString : localSubmap->lineStringLayer) {
    processElement(lineString);
  }
  for (const auto& polygon : localSubmap->polygonLayer) {
    processElement(polygon);
  }
}

void MapData::collectSymbols(LaneletSubmapConstPtr& localSubmap, bool ignoreMapElevation) {
  auto processElement = [&](const auto& element) {
    Attribute type = element.attributeOr(AttributeName::Type, "");
    if (type == "symbol") {
      Id lsId = element.id();
      BasicLineString3d lsBasic = element.basicLineString();

      if (ignoreMapElevation) {
        for (auto& pt : lsBasic) {
          pt[2] = 0;
        }
      }

      TEType teType = teTypeToEnum(element);
      TEType representativeType = getTETypeRepresentative(teType, teTypeGrouping_);
      teInstances_.insert({lsId, std::make_shared<TEInstance>(lsBasic, lsId, representativeType)});

      // Find the nearest intersecting lanelet
      Optional<Id> laneletId = findNearestIntersectingLanelet(element, localSubmap);
      if (laneletId) {
        teEdges_[lsId].push_back(Edge(lsId, *laneletId, false));
      }
    }
  };

  for (const auto& lineString : localSubmap->lineStringLayer) {
    processElement(lineString);
  }
  for (const auto& polygon : localSubmap->polygonLayer) {
    processElement(polygon);
  }
}

void MapData::collectPedestrianCrossings(LaneletSubmapConstPtr& localSubmap, bool ignoreMapElevation) {
  for (const auto& ll : localSubmap->laneletLayer) {
    Attribute subtype = ll.attributeOr(AttributeName::Subtype, "");
    if (subtype == AttributeValueString::Crosswalk) {
      // Determine the crossing type from borders (Zebra has precedence over PedestrianMarking)
      LineStringType leftType = bdTypeToEnum(ll.leftBound3d());
      LineStringType rightType = bdTypeToEnum(ll.rightBound3d());

      LineStringType crossingType = LineStringType::PedestrianCrossing;  // default
      if (leftType == LineStringType::ZebraCrossing || rightType == LineStringType::ZebraCrossing) {
        crossingType = LineStringType::ZebraCrossing;
      } else if (leftType == LineStringType::PedestrianCrossing || rightType == LineStringType::PedestrianCrossing) {
        crossingType = LineStringType::PedestrianCrossing;
      } else {
        std::cout << "Warning: Crosswalk Lanelet " << ll.id()
                  << " has left and right borders that are not zebra_marking or pedestrian_marking, assuming "
                     "LineStringType::PedestrianCrossing"
                  << std::endl;
      }

      // Construct perimeter: left border + endpoint-aligned right border + start point to close
      BasicLineString3d leftBorder = ll.leftBound3d().inverted() ? ll.leftBound3d().invert().basicLineString()
                                                                 : ll.leftBound3d().basicLineString();
      BasicLineString3d rightBorder = ll.rightBound3d().inverted() ? ll.rightBound3d().invert().basicLineString()
                                                                   : ll.rightBound3d().basicLineString();
      BasicLineString3d orientedRightBorder = orientBorderForClosedPerimeter(leftBorder, rightBorder);

      // Build perimeter linestring
      BasicLineString3d perimeter;

      // Add left border points
      for (const auto& pt : leftBorder) {
        perimeter.push_back(ignoreMapElevation ? BasicPoint3d(pt.x(), pt.y(), 0) : pt);
      }

      // Add right border points in the direction that closes the perimeter without crossing edges
      for (const auto& pt : orientedRightBorder) {
        perimeter.push_back(ignoreMapElevation ? BasicPoint3d(pt.x(), pt.y(), 0) : pt);
      }

      // Close the perimeter by adding the first point of left border again
      if (!leftBorder.empty()) {
        perimeter.push_back(ignoreMapElevation ? BasicPoint3d(leftBorder.front().x(), leftBorder.front().y(), 0)
                                               : leftBorder.front());
      }

      // Create a LaneLineStringInstance for the perimeter
      // Use the left border's ID as mapID to avoid duplicate association (since laneletID is already tracked)
      Id perimeterMapId = ll.leftBound3d().id();
      LaneLineStringInstancePtr perimeterInstance =
          std::make_shared<LaneLineStringInstance>(perimeter, perimeterMapId, crossingType, Ids{ll.id()}, false);

      // Wrap in a list and create CompoundLaneLineStringInstance
      LaneLineStringInstanceList perimeterList;
      perimeterList.push_back(perimeterInstance);
      compoundLaneLineStrings_.push_back(std::make_shared<CompoundLaneLineStringInstance>(perimeterList, crossingType));
    }
  }
}

void MapData::convertTEEdges() {
  teToCenterlineEdges_.clear();
  teToTEEdges_.clear();

  for (const auto& edgeEntry : teEdges_) {
    Id sourceId = edgeEntry.first;

    // Find source TE instance
    auto sourceIt = teInstances_.find(sourceId);
    if (sourceIt == teInstances_.end()) {
      continue;  // Skip if source TE not found
    }
    TEInstancePtr sourceTEPtr = sourceIt->second;

    for (const auto& edge : edgeEntry.second) {
      Id targetId = edge.el2_;

      // Check if target is a TE instance (TE to TE edge)
      auto targetTEIt = teInstances_.find(targetId);
      if (targetTEIt != teInstances_.end()) {
        teToTEEdges_.push_back({sourceTEPtr, targetTEIt->second});
        continue;
      }

      // Otherwise, target should be a lanelet - find all compound centerlines containing it
      CompoundLaneLineStringInstanceList centerlines = compoundLineStringsOfType(LineStringType::Centerline);
      for (const auto& cpdLineString : centerlines) {
        // Check if this compound centerline contains the target lanelet
        bool containsTargetLanelet = false;
        for (const auto& feature : cpdLineString->features()) {
          for (const auto& laneletId : feature->laneletIDs()) {
            if (laneletId == targetId) {
              containsTargetLanelet = true;
              break;
            }
          }
          if (containsTargetLanelet) break;
        }

        if (containsTargetLanelet) {
          teToCenterlineEdges_.push_back({sourceTEPtr, cpdLineString});
        }
      }
    }
  }
}

/// @brief Removes the given elements from elsList and cuts what remains at the removal positions
/// Every returned chain is contiguous again, so removing a run from the middle yields two chains rather than one
/// with a gap in it. Returns an empty vector if every element was removed.
std::vector<CompoundElsList> splitAtRemovedElements(const CompoundElsList& elsList,
                                                    const std::vector<Id>& removedElements) {
  std::vector<CompoundElsList> remaining;
  std::vector<Id> runIds;
  std::vector<bool> runInverted;

  auto closeRun = [&]() {
    if (!runIds.empty()) {
      remaining.push_back(CompoundElsList(runIds, runInverted, elsList.type));
      runIds.clear();
      runInverted.clear();
    }
  };

  for (size_t i = 0; i < elsList.ids.size(); i++) {
    if (std::find(removedElements.begin(), removedElements.end(), elsList.ids[i]) != removedElements.end()) {
      closeRun();
    } else {
      runIds.push_back(elsList.ids[i]);
      runInverted.push_back(elsList.inverted[i]);
    }
  }
  closeRun();
  return remaining;
}

/// @brief Deterministic order for candidate chains: longer chains first, chains of equal length by their elements
bool isPreferredCandidate(const CompoundElsList& lhs, const CompoundElsList& rhs) {
  if (lhs.ids.size() != rhs.ids.size()) {
    return lhs.ids.size() > rhs.ids.size();
  }
  return std::lexicographical_compare(lhs.ids.begin(), lhs.ids.end(), rhs.ids.begin(), rhs.ids.end());
}

bool coversSameElements(const CompoundElsList& elsList1, const CompoundElsList& elsList2) {
  return std::set<Id>(elsList1.ids.begin(), elsList1.ids.end()) ==
         std::set<Id>(elsList2.ids.begin(), elsList2.ids.end());
}

/// @brief Maps every element to the chain that currently holds it
std::map<Id, size_t> indexChainElements(const std::vector<CompoundElsList>& chains) {
  std::map<Id, size_t> chainOfElement;
  for (size_t i = 0; i < chains.size(); i++) {
    for (const Id& el : chains[i].ids) {
      chainOfElement[el] = i;
    }
  }
  return chainOfElement;
}

/// @brief Resolves candidate chains that all carry the same representative type into disjoint chains
///
/// The greedy pass keeps the longest candidates intact and lets the shorter ones contribute whatever is still
/// free. The improvement pass afterwards adopts a candidate whenever emitting it as a whole removes more chains
/// than its re-cut creates, which repairs the cases where the greedy order had to cut a candidate into
/// fragments. The result depends only on the set of candidates, not on the order they come in.
std::vector<CompoundElsList> resolveCandidatesOfSameType(std::vector<CompoundElsList> candidates) {
  std::sort(candidates.begin(), candidates.end(), isPreferredCandidate);

  std::vector<CompoundElsList> chains;
  std::set<Id> claimed;
  for (const CompoundElsList& candidate : candidates) {
    std::vector<Id> alreadyClaimed;
    for (const Id& el : candidate.ids) {
      if (claimed.count(el) != 0) {
        alreadyClaimed.push_back(el);
      }
    }
    for (const CompoundElsList& chain : splitAtRemovedElements(candidate, alreadyClaimed)) {
      claimed.insert(chain.ids.begin(), chain.ids.end());
      chains.push_back(chain);
    }
  }

  std::map<Id, size_t> chainOfElement = indexChainElements(chains);
  while (true) {
    int bestGain = 0;
    size_t bestCandidate = candidates.size();
    std::set<size_t> bestOverlapping;
    std::vector<CompoundElsList> bestRemainder;

    for (size_t c = 0; c < candidates.size(); c++) {
      // the index yields the affected chains straight from the candidate's own elements, so the chains that
      // have nothing in common with it are never looked at
      std::set<size_t> overlapping;
      for (const Id& el : candidates[c].ids) {
        std::map<Id, size_t>::const_iterator it = chainOfElement.find(el);
        if (it != chainOfElement.end()) {
          overlapping.insert(it->second);
        }
      }
      if (overlapping.empty() ||
          (overlapping.size() == 1 && coversSameElements(chains[*overlapping.begin()], candidates[c]))) {
        continue;  // nothing to gain, the candidate is already there
      }

      std::vector<CompoundElsList> remainder;
      for (const size_t& i : overlapping) {
        std::vector<CompoundElsList> rest = splitAtRemovedElements(chains[i], candidates[c].ids);
        remainder.insert(remainder.end(), rest.begin(), rest.end());
      }
      // chains that the adoption removes, minus the candidate itself and the pieces its re-cut leaves behind
      int gain = static_cast<int>(overlapping.size()) - 1 - static_cast<int>(remainder.size());
      if (gain > bestGain) {
        bestGain = gain;
        bestCandidate = c;
        bestOverlapping = overlapping;
        bestRemainder = remainder;
      }
    }
    if (bestCandidate == candidates.size()) {
      return chains;
    }

    std::vector<CompoundElsList> updated;
    for (size_t i = 0; i < chains.size(); i++) {
      if (bestOverlapping.count(i) == 0) {
        updated.push_back(chains[i]);
      }
    }
    updated.push_back(candidates[bestCandidate]);
    updated.insert(updated.end(), bestRemainder.begin(), bestRemainder.end());
    chains = updated;
    chainOfElement = indexChainElements(chains);
  }
}

/// @brief Turns the overlapping candidate chains of all lanelet paths into disjoint chains covering the same
/// elements, using as few chains as possible
///
/// Every returned chain is a contiguous part of one candidate, so it stays traceable to a single lanelet path.
/// Candidates of different representative types never share an element, so they are resolved independently.
std::vector<CompoundElsList> resolveCompoundCandidates(std::vector<CompoundElsList> candidates) {
  std::map<LineStringType, std::vector<CompoundElsList>> candidatesPerType;
  for (const CompoundElsList& candidate : candidates) {
    candidatesPerType[candidate.type].push_back(candidate);
  }

  std::vector<CompoundElsList> chains;
  for (auto& typeAndCandidates : candidatesPerType) {
    std::vector<CompoundElsList> resolved = resolveCandidatesOfSameType(typeAndCandidates.second);
    chains.insert(chains.end(), resolved.begin(), resolved.end());
  }
  return chains;
}

void MapData::initCompoundInstances(LaneletSubmapConstPtr& localSubmap,
                                    lanelet::routing::RoutingGraphConstPtr localSubmapGraph,
                                    traffic_rules::TrafficRulesPtr trafficRules,
                                    lanelet::routing::RoutingGraphConstPtr bikeSubmapGraph, bool ignoreMapElevation) {
  std::vector<CompoundElsList> candidates;

  // Process vehicle routing graph
  std::vector<ConstLanelets> vehiclePaths;
  for (const auto& ll : localSubmap->laneletLayer) {
    if (!trafficRules->canPass(ll)) {
      continue;
    }

    ConstLanelets previousLLs = localSubmapGraph->previous(ll, false);
    if (previousLLs.empty()) {
      getPaths(localSubmapGraph, vehiclePaths, ll);
    }

    if (!trafficRules->isOneWay(ll)) {
      ConstLanelets previousLLs = localSubmapGraph->previous(ll.invert(), false);
      if (previousLLs.empty()) {
        getPaths(localSubmapGraph, vehiclePaths, ll.invert());
      }
    }
  }

  // Process bicycle routing graph if provided
  std::vector<ConstLanelets> bikePaths;
  if (bikeSubmapGraph) {
    for (const auto& ll : localSubmap->laneletLayer) {
      // Check if this is a bike lane
      Attribute subtype = ll.attributeOr(AttributeName::Subtype, "");
      if (subtype != AttributeValueString::BicycleLane) {
        continue;
      }

      ConstLanelets previousLLs = bikeSubmapGraph->previous(ll, false);
      if (previousLLs.empty()) {
        getPaths(bikeSubmapGraph, bikePaths, ll);
      }
    }
  }

  // Combine all paths for border processing to avoid duplicating shared dividers
  std::vector<ConstLanelets> allPaths;
  allPaths.insert(allPaths.end(), vehiclePaths.begin(), vehiclePaths.end());
  allPaths.insert(allPaths.end(), bikePaths.begin(), bikePaths.end());

  // Collect the border chains of all paths first; which of the overlapping ones survive is decided afterwards,
  // so that the result does not depend on the order in which the paths were enumerated
  for (const auto& path : allPaths) {
    std::vector<CompoundElsList> compoundedLeft = computeCompoundLeftBorders(path);
    candidates.insert(candidates.end(), compoundedLeft.begin(), compoundedLeft.end());
    std::vector<CompoundElsList> compoundedRight = computeCompoundRightBorders(path);
    candidates.insert(candidates.end(), compoundedRight.begin(), compoundedRight.end());
    compoundLaneLineStrings_.push_back(computeCompoundCenterline(path, ignoreMapElevation));
  }
  for (const CompoundElsList& compFeat : resolveCompoundCandidates(candidates)) {
    if (compFeat.ids.size() != compFeat.inverted.size()) {
      throw std::runtime_error("Unequal sizes of ids and inverted!");
    }
    if (compFeat.ids.empty()) {
      continue;
    }

    LaneLineStringInstanceList toBeCompounded;
    for (size_t i = 0; i < compFeat.ids.size(); i++) {
      LaneLineStringInstancePtr cmpdFeat = getLineStringFeatFromId(compFeat.ids[i], compFeat.inverted[i]);
      toBeCompounded.push_back(cmpdFeat);
    }
    // the compound carries the representative type of its group, not the type of its first member -
    // that is the whole point of the grouping (e.g. dashed + solid members become one Divider)
    compoundLaneLineStrings_.push_back(std::make_shared<CompoundLaneLineStringInstance>(toBeCompounded, compFeat.type));
  }

  // Process drivable area borders
  computeDrivableAreaBorders(localSubmap, ignoreMapElevation);
}

void MapData::updateAssociatedCpdInstanceIndices() {
  for (size_t i = 0; i < compoundLaneLineStrings_.size(); i++) {
    const auto& cpdFeat = compoundLaneLineStrings_[i];
    LineStringType type = cpdFeat->type();
    for (const auto& indFeat : cpdFeat->features()) {
      for (const auto& id : indFeat->laneletIDs()) {
        if (type != LineStringType::Centerline) {
          associatedCpdLineStringsIndices_[type][id].push_back(i);
        }
      }
      associatedCpdLineStringsIndices_[type][indFeat->mapID()].push_back(i);
    }
  }
}

bool MapData::processAll(const OrientedRect& bbox, const ParametrizationType& paramType, bool resampleLanes,
                         int32_t nPointsLanes, bool resampleTE, int32_t nPointsTE, double pitch, double roll) {
  // The tensor data buffer belongs to the previous bounding box
  tfData_.reset();

  // Process lane line strings: -1 signals no resampling
  int32_t laneNPoints = (resampleLanes && nPointsLanes > 1) ? nPointsLanes : -1;
  bool validLineStrings = processInstances(laneLineStrings_, bbox, paramType, laneNPoints, pitch, roll);
  bool validLaneletInstances = processInstances(laneletInstances_, bbox, paramType, laneNPoints, pitch, roll);
  bool validCompoundLineStrings = processInstances(compoundLaneLineStrings_, bbox, paramType, laneNPoints, pitch, roll);

  // Process TE instances: -1 signals no resampling
  int32_t teNPoints = (resampleTE && nPointsTE > 1) ? nPointsTE : -1;
  bool validTEInstances = processInstances(teInstances_, bbox, paramType, teNPoints, pitch, roll);

  if (validLineStrings && validLaneletInstances && validCompoundLineStrings && validTEInstances) {
    return true;
  } else {
    return false;
  }
}

MapData::TensorInstanceData MapData::getTensorInstanceData(bool pointsIn2d, bool ignoreBuffer) {
  if (!tfData_.has_value() || ignoreBuffer) {
    tfData_ = MapData::TensorInstanceData();
    tfData_->uuid_ = uuid_;

    // Collect all valid linestrings organized by type
    size_t globalLineStringIdx = 0;
    for (const auto& type_pair : laneLineStrings_) {
      if (type_pair.second->valid()) {
        std::vector<MatrixXd> matrices = type_pair.second->pointMatrices(pointsIn2d);
        LineStringType type = type_pair.second->type();
        for (const auto& mat : matrices) {
          tfData_->lineStringsByType_[type].push_back(mat);
          globalLineStringIdx++;
        }
      }
    }

    // Collect all valid compound linestrings organized by type
    // Build mapping from instance pointer to (type, index)
    std::map<CompoundLaneLineStringInstancePtr, std::pair<LineStringType, size_t>> cpdInstanceToIndex;
    for (size_t i = 0; i < compoundLaneLineStrings_.size(); ++i) {
      const auto& cpdFeat = compoundLaneLineStrings_[i];
      if (cpdFeat->valid()) {
        std::vector<MatrixXd> matrices = cpdFeat->pointMatrices(pointsIn2d);
        LineStringType type = cpdFeat->type();
        size_t typeIndex = tfData_->compoundLineStringsByType_[type].size();
        for (const auto& mat : matrices) {
          tfData_->compoundLineStringsByType_[type].push_back(mat);
          tfData_->compoundLineStringInstancesByType_[type].push_back(cpdFeat);
        }
        cpdInstanceToIndex[cpdFeat] = {type, typeIndex};
      }
    }

    // Collect all valid traffic elements organized by type
    // Build mapping from instance pointer to (type, index)
    std::map<TEInstancePtr, std::pair<TEType, size_t>> teInstanceToIndex;
    for (const auto& te_pair : teInstances_) {
      if (te_pair.second->valid()) {
        std::vector<MatrixXd> matrices = te_pair.second->pointMatrices(pointsIn2d);
        TEType type = te_pair.second->teType();
        size_t typeIndex = tfData_->teInstancesByType_[type].size();
        for (const auto& mat : matrices) {
          tfData_->teInstancesByType_[type].push_back(mat);
        }
        teInstanceToIndex[te_pair.second] = {type, typeIndex};
      }
    }

    // Convert teToCenterlineEdges_ to index-based edges
    for (const auto& edge : teToCenterlineEdges_) {
      auto teIt = teInstanceToIndex.find(edge.first);
      auto cpdIt = cpdInstanceToIndex.find(edge.second);
      if (teIt != teInstanceToIndex.end() && cpdIt != cpdInstanceToIndex.end()) {
        if (cpdIt->second.first != LineStringType::Centerline) {
          std::cerr << "Warning: TE to centerline edge contains non-centerline compound linestring (type: "
                    << static_cast<int>(cpdIt->second.first) << ")" << std::endl;
        } else {
          tfData_->teToCenterlineIndexEdges_.push_back(
              std::make_tuple(teIt->second.first, teIt->second.second, cpdIt->second.second));
        }
      }
    }

    // Convert teToTEEdges_ to index-based edges
    for (const auto& edge : teToTEEdges_) {
      auto teSourceIt = teInstanceToIndex.find(edge.first);
      auto teTargetIt = teInstanceToIndex.find(edge.second);
      if (teSourceIt != teInstanceToIndex.end() && teTargetIt != teInstanceToIndex.end()) {
        tfData_->teToTEIndexEdges_.push_back(std::make_tuple(teSourceIt->second.first, teSourceIt->second.second,
                                                             teTargetIt->second.first, teTargetIt->second.second));
      }
    }
  }
  return tfData_.value();
}

std::vector<MatrixXd> MapData::TensorInstanceData::lineStringsOfType(LineStringType type) const {
  auto it = lineStringsByType_.find(type);
  if (it != lineStringsByType_.end()) {
    return it->second;
  }
  return std::vector<MatrixXd>();
}

std::vector<MatrixXd> MapData::TensorInstanceData::compoundLineStringsOfType(LineStringType type) const {
  auto it = compoundLineStringsByType_.find(type);
  if (it != compoundLineStringsByType_.end()) {
    return it->second;
  }
  return std::vector<MatrixXd>();
}

std::vector<MatrixXd> MapData::TensorInstanceData::teInstancesOfType(TEType type) const {
  auto it = teInstancesByType_.find(type);
  if (it != teInstancesByType_.end()) {
    return it->second;
  }
  return std::vector<MatrixXd>();
}

CompoundLaneLineStringInstancePtr MapData::TensorInstanceData::pointMatrixCpdLineStrings(LineStringType type,
                                                                                         size_t index) {
  auto typeIt = compoundLineStringInstancesByType_.find(type);
  if (typeIt == compoundLineStringInstancesByType_.end()) {
    throw std::out_of_range("No compound line strings exist for the given type!");
  }
  if (index >= typeIt->second.size()) {
    throw std::out_of_range("A point matrix with type-local index " + std::to_string(index) + " does not exist!");
  }
  return typeIt->second[index];
}

CompoundLaneLineStringInstanceList MapData::associatedCpdLineStringsOfType(Id mapId, LineStringType type) const {
  CompoundLaneLineStringInstanceList assoFeats;
  try {
    const auto& typeIndices = associatedCpdLineStringsIndices_.at(type);
    for (const auto& idx : typeIndices.at(mapId)) {
      assoFeats.push_back(compoundLaneLineStrings_[idx]);
    }
  } catch (const std::out_of_range&) {
    return assoFeats;
  }
  return assoFeats;
}

LaneLineStringInstances MapData::lineStringsOfType(LineStringType type) const {
  LaneLineStringInstances result;
  for (const auto& pair : laneLineStrings_) {
    if (pair.second->type() == type) {
      result.insert(pair);
    }
  }
  return result;
}

CompoundLaneLineStringInstanceList MapData::compoundLineStringsOfType(LineStringType type) const {
  CompoundLaneLineStringInstanceList result;
  for (const auto& feat : compoundLaneLineStrings_) {
    if (feat->type() == type) {
      result.push_back(feat);
    }
  }
  return result;
}

LaneLineStringInstances MapData::validLineStringsOfType(LineStringType type) const {
  return getValidElements(lineStringsOfType(type));
}

CompoundLaneLineStringInstanceList MapData::validCompoundLineStringsOfType(LineStringType type) const {
  return getValidElements(compoundLineStringsOfType(type));
}

TEInstances MapData::teInstancesOfType(TEType type) const {
  TEInstances result;
  for (const auto& pair : teInstances_) {
    if (pair.second->teType() == type) {
      result.insert(pair);
    }
  }
  return result;
}

TEInstances MapData::validTEInstancesOfType(TEType type) const { return getValidElements(teInstancesOfType(type)); }

}  // namespace ml_converter
}  // namespace lanelet
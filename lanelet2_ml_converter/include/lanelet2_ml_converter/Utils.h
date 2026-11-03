#pragma once
#include <lanelet2_core/Exceptions.h>
#include <lanelet2_core/Forward.h>
#include <lanelet2_core/geometry/LineString.h>
#include <lanelet2_core/primitives/BasicRegulatoryElements.h>
#include <lanelet2_traffic_rules/TrafficRulesFactory.h>

#include <boost/geometry.hpp>

#include "Forward.h"
#include "Serialize.h"
#include "Types.h"
namespace lanelet {
namespace ml_converter {

/// @brief Build the local reference frame around a pose
/// @param center Origin of the local reference frame in the map frame
/// @param extentLongitudinal Half extent [m] in driving direction, i.e. the rectangle reaches this far to the
/// front and to the rear of center
/// @param extentLateral Half extent [m] in lateral direction, i.e. this far to the left and to the right
/// @param yaw Heading of the local reference frame in the map frame [rad], counter clockwise from the x axis
/// @param from2dPos Whether the pose this is built from is 2d, see OrientedRect::from2d
OrientedRect getRotatedRect(const BasicPoint3d& center, double extentLongitudinal, double extentLateral, double yaw,
                            bool from2dPos);

/// @brief Extract the part of the map around a position
/// @param center Center of the region to extract, in the map frame
/// @param extentLongitudinal Half extent [m] in driving direction, see getRotatedRect()
/// @param extentLateral Half extent [m] in lateral direction, see getRotatedRect()
/// @return A submap holding the lanelets, line strings, polygons and regulatory elements of the region
/// @note The search region is axis aligned and slightly larger than the rectangle of getRotatedRect(), so that
/// it contains the rotated rectangle for any yaw angle. The final crop happens when instances are processed
LaneletSubmapConstPtr extractSubmap(LaneletMapConstPtr laneletMap, const BasicPoint2d& center,
                                    double extentLongitudinal, double extentLateral);

/// @brief Get the name of a LineStringType
/// @throw std::runtime_error if the type is unknown
inline std::string lineStringTypeToString(LineStringType type) {
  if (type == LineStringType::RoadBorder)
    return "RoadBorder";
  else if (type == LineStringType::DrivableArea)
    return "DrivableArea";
  else if (type == LineStringType::Dashed)
    return "Dashed";
  else if (type == LineStringType::Solid)
    return "Solid";
  else if (type == LineStringType::CurbstoneHigh)
    return "CurbstoneHigh";
  else if (type == LineStringType::CurbstoneLow)
    return "CurbstoneLow";
  else if (type == LineStringType::Fence)
    return "Fence";
  else if (type == LineStringType::SolidSolid)
    return "SolidSolid";
  else if (type == LineStringType::SolidDashed)
    return "SolidDashed";
  else if (type == LineStringType::DashedSolid)
    return "DashedSolid";
  else if (type == LineStringType::Virtual)
    return "Virtual";
  else if (type == LineStringType::Centerline)
    return "Centerline";
  else if (type == LineStringType::BikeCenterline)
    return "BikeCenterline";
  else if (type == LineStringType::Unknown)
    return "Unknown";
  else if (type == LineStringType::Divider)
    return "Divider";
  else if (type == LineStringType::BikeMarkingDashed)
    return "BikeMarkingDashed";
  else if (type == LineStringType::BikeMarkingSolid)
    return "BikeMarkingSolid";
  else if (type == LineStringType::GuardRail)
    return "GuardRail";
  else if (type == LineStringType::PedestrianCrossing)
    return "PedestrianCrossing";
  else if (type == LineStringType::ZebraCrossing)
    return "ZebraCrossing";
  else if (type == LineStringType::Building)
    return "Building";
  else if (type == LineStringType::Wall)
    return "Wall";
  else {
    throw std::runtime_error("Unexpected Line String type!");
    return "Unknown";
  }
}

/// @brief Get the LineStringType of a lanelet boundary from its tagging
/// The `type` tag takes precedence over the `subtype` tag, which is only consulted for lane markings and
/// bicycle lane markings (`type=bike_marking`: dashed -> BikeMarkingDashed; solid and the mixed patterns
/// solid_solid, solid_dashed, dashed_solid -> BikeMarkingSolid, as at least one side may not be crossed;
/// any other subtype -> Unknown).
/// Works with both ConstLineString3d and ConstPolygon3d.
/// @return The type, LineStringType::Unknown if no tag matches
/// @note Road borders are only recognized if they are tagged as such - an untagged road border is
/// indistinguishable from a lane divider and will be treated as one
template <typename T>
inline LineStringType bdTypeToEnum(const T& element) {
  Attribute type = element.attributeOr(AttributeName::Type, "");
  Attribute subtype = element.attributeOr(AttributeName::Subtype, "");

  // Check for specific types first
  if (type == AttributeValueString::RoadBorder) {
    return LineStringType::RoadBorder;
  } else if (type == AttributeValueString::Curbstone) {
    // For Curbstone, check the subtype
    if (subtype == AttributeValueString::Low) {
      return LineStringType::CurbstoneLow;
    }
    // Default to CurbstoneHigh if no subtype or subtype is High
    return LineStringType::CurbstoneHigh;
  } else if (type == AttributeValueString::Fence) {
    return LineStringType::Fence;
  } else if (type == AttributeValueString::GuardRail) {
    return LineStringType::GuardRail;
  } else if (type == AttributeValueString::Virtual) {
    return LineStringType::Virtual;
  } else if (type == AttributeValueString::Zebra) {
    return LineStringType::ZebraCrossing;
  } else if (type == AttributeValueString::PedestrianMarking) {
    return LineStringType::PedestrianCrossing;
  } else if (type == AttributeValueString::Building) {
    return LineStringType::Building;
  } else if (type == AttributeValueString::Wall) {
    return LineStringType::Wall;
  } else if (type == AttributeValueString::BikeMarking) {
    if (subtype == AttributeValueString::Dashed) {
      return LineStringType::BikeMarkingDashed;
    }
    if (subtype == AttributeValueString::Solid || subtype == AttributeValueString::SolidSolid ||
        subtype == AttributeValueString::SolidDashed || subtype == AttributeValueString::DashedSolid) {
      return LineStringType::BikeMarkingSolid;
    }
    return LineStringType::Unknown;
  }

  // Check subtype for lane marking types
  if (subtype == AttributeValueString::Dashed) {
    return LineStringType::Dashed;
  } else if (subtype == AttributeValueString::Solid) {
    return LineStringType::Solid;
  } else if (subtype == AttributeValueString::SolidSolid) {
    return LineStringType::SolidSolid;
  } else if (subtype == AttributeValueString::SolidDashed) {
    return LineStringType::SolidDashed;
  } else if (subtype == AttributeValueString::DashedSolid) {
    return LineStringType::DashedSolid;
  }

  // Default to Unknown if no match found
  return LineStringType::Unknown;
}

//! Concrete overload of bdTypeToEnum() for ConstLineString3d (used by Python bindings)
inline LineStringType bdTypeToEnum(const ConstLineString3d& lString) {
  return bdTypeToEnum<ConstLineString3d>(lString);
}

//! Concrete overload of bdTypeToEnum() for ConstPolygon3d (used by Python bindings)
inline LineStringType bdTypeToEnumPolygon(const ConstPolygon3d& polygon) {
  return bdTypeToEnum<ConstPolygon3d>(polygon);
}

/// @brief The LineStringTypes a lanelet boundary can have, i.e. everything bdTypeToEnum can return
/// Keep this in sync with bdTypeToEnum.
inline std::vector<LineStringType> boundaryLineStringTypes() {
  return {LineStringType::RoadBorder,    LineStringType::CurbstoneHigh,
          LineStringType::CurbstoneLow,  LineStringType::Fence,
          LineStringType::GuardRail,     LineStringType::Virtual,
          LineStringType::ZebraCrossing, LineStringType::PedestrianCrossing,
          LineStringType::Building,      LineStringType::Wall,
          LineStringType::Dashed,        LineStringType::Solid,
          LineStringType::SolidSolid,    LineStringType::SolidDashed,
          LineStringType::DashedSolid,   LineStringType::BikeMarkingDashed,
          LineStringType::BikeMarkingSolid, LineStringType::Unknown};
}

/// @brief Throws unless the grouping assigns a group to every type a lanelet boundary can have
/// Boundaries are chained into compound instances while their group index stays the same. Types that are in no
/// group at all share the "not found" index, so leaving one out would let unrelated types be chained into a
/// single compound instance and be labelled with whichever type happens to come first. Types that can never
/// appear on a boundary (Divider, Centerline, ...) do not have to be grouped.
inline void checkLineStringTypeGroupingCoverage(const LineStringTypeGrouping& grouping) {
  std::string missingTypes;
  for (const LineStringType& type : boundaryLineStringTypes()) {
    if (getLineStringTypeGroupIndex(type, grouping) < 0) {
      missingTypes =
          missingTypes.empty() ? lineStringTypeToString(type) : missingTypes + ", " + lineStringTypeToString(type);
    }
  }
  if (!missingTypes.empty()) {
    throw std::runtime_error("LineStringTypeGrouping does not cover every type a lanelet boundary can have! " +
                             std::string("Missing: ") + missingTypes);
  }
}

/// @brief Get the TEType of a traffic element from its tagging
/// Recognizes stop lines, arrow and symbol road markings, traffic lights and traffic signs. Traffic signs are
/// mapped from their German StVO code in the `subtype` tag, unknown codes become TEType::TSMisc.
/// Works with both ConstLineString3d and ConstPolygon3d.
/// @return The type, TEType::Unknown if no tag matches
template <typename T>
inline TEType teTypeToEnum(const T& te) {
  Attribute type = te.attributeOr(AttributeName::Type, "");
  Attribute subtype = te.attributeOr(AttributeName::Subtype, "");

  // Handle stop_line type
  if (type == "stop_line") {
    return TEType::StopLine;
  }

  // Handle arrows
  if (type == "arrow") {
    if (subtype == "right") {
      return TEType::ArrowTurnRight;
    } else if (subtype == "left") {
      return TEType::ArrowTurnLeft;
    } else if (subtype == "straight") {
      return TEType::ArrowGoStraight;
    } else if (subtype == "straight_right") {
      return TEType::ArrowGoStraightOrRight;
    } else if (subtype == "straight_left") {
      return TEType::ArrowGoStraightOrLeft;
    } else if (subtype == "left_right") {
      return TEType::ArrowTurnLeftOrRight;
    }
    return TEType::Unknown;  // Unknown arrow subtype
  }

  // Handle symbols
  if (type == "symbol") {
    if (subtype == "bicycle") {
      return TEType::BikeSymbol;
    } else if (subtype == "bus") {
      return TEType::BusSymbol;
    } else if (subtype == "30") {
      return TEType::Symbol30;
    } else if (subtype == "50") {
      return TEType::Symbol50;
    } else if (subtype == "70") {
      return TEType::Symbol70;
    }
    return TEType::Unknown;  // Unknown symbol subtype
  }

  // Handle traffic lights
  if (type == AttributeValueString::TrafficLight) {
    return TEType::TLCar;
  } else if (type == "traffic_light_bikes") {
    return TEType::TLBike;
  } else if (type == "traffic_light_pedestrians") {
    return TEType::TLPedestrian;
  } else if (type == "traffic_light_misc") {
    return TEType::TLMisc;
  }

  // Handle traffic signs
  if (type == AttributeValueString::TrafficSign) {
    // German traffic sign codes based on official StVO regulation
    // Regulatory signs (200-299)
    if (subtype.value().find("de201") == 0) {  // de201-* (Andreaskreuz / crossbuck)
      return TEType::TSCrossbuck;
    } else if (subtype == "de205") {
      return TEType::TSYield;  // Yield sign
    } else if (subtype == "de206") {
      return TEType::TSStop;  // Stop sign
    } else if (subtype == "de209") {
      return TEType::TSTurnRight;  // Turn right ahead
    } else if (subtype == "de209-10") {
      return TEType::TSTurnLeft;  // Turn left ahead
    } else if (subtype == "de209-30") {
      return TEType::TSGoStraight;  // Go straight ahead
    } else if (subtype == "de211") {
      return TEType::TSTurnRight;  // Turn right (here)
    } else if (subtype == "de211-10") {
      return TEType::TSTurnLeft;  // Turn left (here)
    } else if (subtype == "de214") {
      return TEType::TSGoStraightOrRight;  // Go straight or turn right ahead
    } else if (subtype == "de214-10") {
      return TEType::TSGoStraightOrLeft;  // Go straight or turn left ahead
    } else if (subtype == "de214-30") {
      return TEType::TSTurnLeftOrRight;  // Turn left or right ahead
    } else if (subtype == "de215") {
      return TEType::TSRoundabout;  // Roundabout
    } else if (subtype == "de220-10" || subtype == "de220-20") {
      return TEType::TSOneWayStreet;  // One-way street
    } else if (subtype == "de222") {
      return TEType::TSPassRight;  // Pass on the right
    } else if (subtype == "de222-10") {
      return TEType::TSPassLeft;  // Pass on the left
    } else if (subtype == "de267") {
      return TEType::TSNoEntry;                       // No entry
    } else if (subtype.value().find("de274") == 0) {  // de274-* (speed limit signs)
      return TEType::TSSpeedLimit;
      // Directional signs (300-399)
    } else if (subtype == "de301") {
      return TEType::TSRightOfWay;  // Intersection ahead - right-of-way only for this intersection
    } else if (subtype == "de306") {
      return TEType::TSPriorityRoad;                  // Priority road - right-of-way on all following intersections
    } else if (subtype.value().find("de310") == 0) {  // de310 (town/city limit sign)
      return TEType::TSMisc;
    } else if (subtype == "de350-10" || subtype == "de350-20") {
      return TEType::TSPedestrianCrossing;  // Pedestrian crossing directional signs
    }
    return TEType::TSMisc;  // Default for unknown traffic sign subtypes
  }

  // Default to Unknown if no match found
  return TEType::Unknown;
}

//! Concrete overload of teTypeToEnum() for ConstLineString3d (used by Python bindings)
inline TEType teTypeToEnum(const ConstLineString3d& te) { return teTypeToEnum<ConstLineString3d>(te); }

//! Concrete overload of teTypeToEnum() for ConstPolygon3d (used by Python bindings)
inline TEType teTypeToEnumPolygon(const ConstPolygon3d& te) { return teTypeToEnum<ConstPolygon3d>(te); }

/// @brief Resample a polyline to a fixed number of equidistant points
/// @param nPoints Number of points of the result, first and last point of the polyline are kept
/// @return The resampled polyline, empty if the polyline is degenerate (shorter than 0.1 m)
/// @throw std::runtime_error if the polyline is empty or if nPoints is smaller than 2
BasicLineString3d resampleLineString(const BasicLineString3d& polyline, int32_t nPoints);

/// @brief Clip a polyline to the local reference frame
/// @return The parts of the polyline inside of the rectangle, empty if it lies completely outside. A polyline
/// that leaves and re-enters the rectangle results in more than one part
/// @note Clipping happens in 2d. The z coordinate of the clipped points is restored from the nearest point of
/// the original polyline, searching along it so that elements with several points above each other (e.g.
/// traffic signs) keep their elevation
std::vector<BasicLineString3d> cutLineString(const OrientedRect& bbox, const BasicLineString3d& polyline);

/// @brief Transform a polyline from the map frame into the local reference frame given by bbox
/// @param pitch Pitch angle of the local reference frame [rad]
/// @param roll Roll angle of the local reference frame [rad]
/// @note The rotation is applied as yaw, then pitch, then roll, with the yaw angle taken from the bounding box
BasicLineString3d transformLineString(const OrientedRect& bbox, const BasicLineString3d& polyline, double pitch,
                                      double roll);

/// @brief Save map data objects, all of them in one file
/// @param binary If true, a boost binary archive is written, otherwise a human readable XML archive
/// @throw std::runtime_error if the file cannot be opened
void saveMapData(const std::string& filename, const std::vector<MapDataPtr>& mDataVec, bool binary);

/// @brief Load the map data objects that saveMapData() wrote into one file
/// @throw std::runtime_error if the file does not exist or cannot be opened
std::vector<MapDataPtr> loadMapData(const std::string& filename, bool binary);

/// @brief Save map data objects, one file per MapData object
/// @param path Directory the files are written to, is prepended to the file names as it is
/// @param filenames One file name per map data object, must have the same size as mDataVec
/// @throw std::runtime_error if the sizes do not match or if a file cannot be opened
void saveMapDataMultiFile(const std::string& path, const std::vector<std::string>& filenames,
                          const std::vector<MapDataPtr>& mDataVec, bool binary);

/// @brief Load the map data objects that saveMapDataMultiFile() wrote into one file each
/// @throw std::runtime_error if one of the files does not exist or cannot be opened
std::vector<MapDataPtr> loadMapDataMultiFile(const std::string& path, const std::vector<std::string>& filenames,
                                             bool binary);

}  // namespace ml_converter
}  // namespace lanelet
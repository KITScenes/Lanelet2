#pragma once

#include <lanelet2_core/Forward.h>
#include <lanelet2_core/utility/Optional.h>

#include "lanelet2_ml_converter/Forward.h"
#include "lanelet2_ml_converter/MapData.h"
#include "lanelet2_ml_converter/Types.h"
#include "lanelet2_routing/RoutingGraph.h"

namespace lanelet {
class LaneletLayer;

namespace ml_converter {

/**
 * @brief Main class of the module: turns a Lanelet2 map into local instance labels
 *
 * Usage: set a local reference frame pose, then get the labels for it.
 * ```c++
 * MapDataInterface mDataIf(laneletMap);
 * mDataIf.setCurrPosAndExtractSubmap2d(BasicPoint2d(10, 10), 0);
 * MapDataPtr mData = mDataIf.mapData(true);
 * ```
 * For more than one pose, prefer the mapDataBatch() overloads - they are self contained and do not touch the
 * current position.
 *
 * Every call to setCurrPosAndExtractSubmap() extracts a new local submap around the given position and builds
 * the routing graphs for it, which is the expensive part of the work.
 *
 * @note mapData() throws unless a position was set before, the batch methods do not have this requirement.
 * @note The traffic rules are currently fixed to Locations::Germany for Participants::Vehicle and
 * Participants::Bicycle.
 */
class MapDataInterface {
 public:
  //! Parameters of the extraction and processing, see MapData::build() and MapData::processAll()
  struct Configuration {
    Configuration() noexcept
        : lineStringTypeGrouping{getDefaultLineStringTypeGrouping()}, teTypeGrouping{getDefaultTETypeGrouping()} {}
    Configuration(LaneletRepresentationType reprType, ParametrizationType paramType, double submapExtentLongitudinal,
                  double submapExtentLateral, int nPointsLanes, int nPointsTE) noexcept
        : reprType{reprType},
          paramType{paramType},
          submapExtentLongitudinal{submapExtentLongitudinal},
          submapExtentLateral{submapExtentLateral},
          nPointsLanes{nPointsLanes},
          nPointsTE{nPointsTE},
          lineStringTypeGrouping{getDefaultLineStringTypeGrouping()},
          teTypeGrouping{getDefaultTETypeGrouping()} {}
    //! how lanelet instances are laid out in their instance vectors
    LaneletRepresentationType reprType{LaneletRepresentationType::Boundaries};

    //! parametrization of the instance geometry, only ParametrizationType::LineString is implemented
    ParametrizationType paramType{ParametrizationType::LineString};

    //! half extent [m] of the local reference frame in driving direction, i.e. this much to the front and to the rear
    double submapExtentLongitudinal{30};

    //! half extent [m] of the local reference frame in lateral direction, i.e. this much to the left and to the right
    double submapExtentLateral{15};

    //! if true, elevation (z coordinate) in map elements is ignored and set to 0
    bool ignoreMapElevation{false};

    //! if true, the vehicle routing graph links two lanes across a bicycle lane between them when both bike lane
    //! bounds may be crossed (e.g. dashed bike markings), see RoutingGraph::AllowLaneChangeAcrossBicycleLane. The
    //! lanes then get a left/right lane change edge to each other.
    bool allowLaneChangeAcrossBicycleLane{false};

    //! if true, lane instances are resampled; if false, no fixed resampling
    bool resampleLanes{true};

    //! number of points for lane resampling when resampleLanes=true, if set < 2 then no resampling as well
    int nPointsLanes{20};

    //! if true, TE instances are resampled; if false, no fixed resampling
    bool resampleTE{true};

    //! number of points for TE resampling when resampleTE=true, if set < 2 then no resampling as well
    int nPointsTE{20};

    //! grouping of types for compound instance generation, see LineStringTypeGrouping
    LineStringTypeGrouping lineStringTypeGrouping;

    //! grouping of types for traffic element instance generation, see TETypeGrouping
    TETypeGrouping teTypeGrouping;
  };

  //! Construct with the default configuration
  MapDataInterface(LaneletMapConstPtr laneletMap);

  MapDataInterface(LaneletMapConstPtr laneletMap, Configuration config);

  const Configuration& config() { return config_; }

  /// @brief Set the local reference frame from a 2d pose and extract the submap around it
  /// @param pt Origin of the local reference frame in the map frame
  /// @param yaw Heading of the local reference frame in the map frame [rad]
  /// @note Since there is no elevation to relate to, all instance output of this frame will be 2d
  void setCurrPosAndExtractSubmap2d(const BasicPoint2d& pt, double yaw);

  //! Like setCurrPosAndExtractSubmap2d(), but from a 3d position with pitch and roll assumed to be 0
  void setCurrPosAndExtractSubmap(const BasicPoint3d& pt, double yaw);

  /// @brief Set the local reference frame from a full 3d pose and extract the submap around it
  /// @param pt Origin of the local reference frame in the map frame
  /// @param yaw Heading of the local reference frame in the map frame [rad]
  /// @param pitch Pitch angle of the local reference frame [rad]
  /// @param roll Roll angle of the local reference frame [rad]
  void setCurrPosAndExtractSubmap(const BasicPoint3d& pt, double yaw, double pitch, double roll);

  /// @brief Get the instance labels of the current local reference frame
  /// @param processAll If true, the instances are processed into the local reference frame right away. If
  /// false, you get the raw extracted data and have to call MapData::processAll() yourself
  /// @throw InvalidObjectStateError if no position was set with setCurrPosAndExtractSubmap() before
  MapDataPtr mapData(bool processAll);

  /// @brief Get the processed instance labels of several 2d poses at once
  /// @throw std::runtime_error if the number of positions and angles does not match
  std::vector<MapDataPtr> mapDataBatch2d(std::vector<BasicPoint2d> pts, std::vector<double> yaws);

  //! Like mapDataBatch2d(), but from 3d positions with pitch and roll assumed to be 0
  std::vector<MapDataPtr> mapDataBatch(std::vector<BasicPoint3d> pts, std::vector<double> yaws);

  /// @brief Get the processed instance labels of several full 3d poses at once
  /// @throw std::runtime_error if the number of positions and angles does not match
  std::vector<MapDataPtr> mapDataBatch(std::vector<BasicPoint3d> pts, std::vector<double> yaws,
                                       std::vector<double> pitches, std::vector<double> rolls);

 private:
  //! The routing graph configuration derived from config_
  routing::RoutingGraph::Configuration routingGraphConfig() const;

  //! Builds the map data of one submap and optionally processes it, see MapData::build()
  MapDataPtr getMapData(LaneletSubmapConstPtr localSubmap, const OrientedRect& bbox,
                        lanelet::routing::RoutingGraphConstPtr localSubmapGraph,
                        lanelet::routing::RoutingGraphConstPtr bikeSubmapGraph, double pitch, double roll,
                        bool processAll);

  LaneletMapConstPtr laneletMap_;
  LaneletSubmapConstPtr localSubmap_;
  std::unordered_map<Id, int> teId2Index_;
  routing::RoutingGraphConstPtr localSubmapGraph_;
  routing::RoutingGraphConstPtr bikeSubmapGraph_;
  Optional<BasicPoint3d> currPos_;  // in the map frame
  Optional<double> currRoll_;       // in the map frame, [rad]
  Optional<double> currPitch_;      // in the map frame, [rad]
  Optional<double> currYaw_;        // in the map frame, [rad]
  Optional<OrientedRect> currBbox_;
  Configuration config_;
  traffic_rules::TrafficRulesPtr trafficRules_;
  traffic_rules::TrafficRulesPtr bikeTrafficRules_;
};

}  // namespace ml_converter
}  // namespace lanelet
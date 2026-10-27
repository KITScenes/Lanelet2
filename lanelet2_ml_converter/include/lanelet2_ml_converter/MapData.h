#pragma once
#include <lanelet2_core/Exceptions.h>
#include <lanelet2_core/Forward.h>
#include <lanelet2_core/geometry/LineString.h>
#include <lanelet2_routing/Forward.h>

#include <boost/geometry.hpp>
#include <boost/uuid/uuid.hpp>
#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <type_traits>

#include "lanelet2_ml_converter/Forward.h"
#include "lanelet2_ml_converter/MapInstances.h"
#include "lanelet2_ml_converter/Types.h"
#include "lanelet2_routing/RoutingGraph.h"
#include "lanelet2_routing/internal/Graph.h"

namespace lanelet {
namespace ml_converter {

using MapDataPtr = std::shared_ptr<MapData>;

/// @brief One edge of the lane graph or the traffic element graph, given by the ids of the elements it connects
struct Edge {
  Edge() = default;
  Edge(Id el1, Id el2, bool isLaneChange) : el1_{el1}, el2_{el2}, isLaneChange_{isLaneChange} {}
  Id el1_;             //!< id of the element the edge starts at
  Id el2_;             //!< id of the element the edge leads to
  bool isLaneChange_;  //!< true for a lateral (lane change) edge, false for a longitudinal (successor) edge
};

namespace internal {
/// @brief A chain of line string ids that is a candidate for becoming one compound instance
struct CompoundElsList {
  CompoundElsList(const Id& start, bool startInverted, LineStringType type)
      : ids{start}, inverted{startInverted}, type{type} {}
  CompoundElsList(const std::vector<Id>& ids, const std::vector<bool>& inverted, LineStringType type)
      : ids{ids}, inverted{inverted}, type{type} {}
  std::vector<Id> ids;         //!< the chained line strings, in order
  std::vector<bool> inverted;  //!< per element: whether it runs against the direction of the map element
  LineStringType type;         //!< representative type of the group all elements belong to
};
}  // namespace internal

//! Edge lists by source element id
using Edges = std::map<Id, std::vector<Edge>>;  // key = id from

//! Traffic element to the compound centerline(s) of the lanelet(s) it applies to, as instance pointers
using TEToCenterlineEdges = std::vector<std::pair<TEInstancePtr, CompoundLaneLineStringInstancePtr>>;

//! Traffic element to traffic element, e.g. a traffic light to its stop line, as instance pointers
using TEToTEEdges = std::vector<std::pair<TEInstancePtr, TEInstancePtr>>;

// Index-based edge types for TensorInstanceData
// TEToCenterline: (source_te_type, source_idx, target_centerline_idx)
// TEToTE: (source_type, source_idx, target_type, target_idx)

/// @brief TEToCenterlineEdges expressed as (source type, source index, target centerline index)
/// @note The indices are local to their type, i.e. they index into
/// MapData::TensorInstanceData::teInstancesOfType() and ::compoundLineStringsOfType() respectively
using TEToCenterlineIndexEdges = std::vector<std::tuple<TEType, size_t, size_t>>;

/// @brief TEToTEEdges expressed as (source type, source index, target type, target index)
/// @note The indices are local to their type, see TEToCenterlineIndexEdges
using TEToTEIndexEdges = std::vector<std::tuple<TEType, size_t, TEType, size_t>>;

//! Filter out all instances that did not survive processing, see MapInstance::valid()
template <typename T>
std::vector<std::shared_ptr<T>> getValidElements(const std::vector<std::shared_ptr<T>>& vec) {
  std::vector<std::shared_ptr<T>> res;
  std::copy_if(vec.begin(), vec.end(), std::back_inserter(res), [](std::shared_ptr<T> el) { return el->valid(); });
  return res;
}

//! Overload of getValidElements() for instances given by id
template <typename T>
std::map<Id, std::shared_ptr<T>> getValidElements(const std::map<Id, std::shared_ptr<T>>& map) {
  std::map<Id, std::shared_ptr<T>> res;
  std::copy_if(map.begin(), map.end(), std::inserter(res, res.end()),
               [](const auto& pair) { return pair.second->valid(); });
  return res;
}

/**
 * @brief All local instance labels of one local reference frame pose
 *
 * A MapData object is created in two phases:
 * 1. build() extracts the instances from a local submap: the lane line strings, the lanelets, the compound
 *    instances chained along the lanelet paths, the traffic elements and the edges between them
 * 2. processAll() brings every instance into the local reference frame, see MapInstance
 *
 * Only after the second phase are the accessors and getTensorInstanceData() meaningful. MapDataInterface does
 * both for you.
 *
 * @note All instances are shared pointers, so a line string that several lanelets use, or that is part of a
 * compound instance, exists exactly once and can be traced back to its map element through its mapID().
 */
class MapData {
 public:
  /**
   * @brief All instance labels of a MapData object in matrix form, ready to be used as tensors
   *
   * Instances are grouped by type and only valid ones are included. Within one type, an instance is identified
   * by its position in the returned list - that is the index the edge lists refer to.
   *
   * @note From python, the matrices are converted to numpy arrays. Every access creates a copy, so modifying
   * the returned array does not modify this object.
   */
  struct TensorInstanceData {
   public:
    //! Point matrices of all valid lane line strings of the given type
    std::vector<MatrixXd> lineStringsOfType(LineStringType type) const;

    //! Point matrices of all valid compound line strings of the given type
    std::vector<MatrixXd> compoundLineStringsOfType(LineStringType type) const;

    //! Point matrices of all valid traffic elements of the given type
    std::vector<MatrixXd> teInstancesOfType(TEType type) const;

    /// @brief Get the instance a point matrix of compoundLineStringsOfType() came from
    /// @param type Type the index refers to
    /// @param index Position within compoundLineStringsOfType(type)
    /// @return The compound instance, which gives access to the map elements it was built from
    /// @throw std::out_of_range if the type or the index does not exist
    CompoundLaneLineStringInstancePtr pointMatrixCpdLineStrings(LineStringType type, size_t index);

    //! Traffic element to centerline edges, with indices local to their type
    const TEToCenterlineIndexEdges& teToCenterlineIndexEdges() const { return teToCenterlineIndexEdges_; }

    //! Traffic element to traffic element edges, with indices local to their type
    const TEToTEIndexEdges& teToTEIndexEdges() const { return teToTEIndexEdges_; }

    //! Id of the sample, identical to the uuid of the MapData object this was created from
    const std::string& uuid() { return uuid_; }
    friend class MapData;

   private:
    // All linestrings organized by type for efficient O(1) lookup
    std::map<LineStringType, std::vector<MatrixXd>> lineStringsByType_;
    // All compound linestrings organized by type for efficient O(1) lookup
    std::map<LineStringType, std::vector<MatrixXd>> compoundLineStringsByType_;
    // Mapping from type to vector of compound feature instances (index is implicit in vector position)
    std::map<LineStringType, std::vector<CompoundLaneLineStringInstancePtr>> compoundLineStringInstancesByType_;
    // All traffic elements organized by type for efficient O(1) lookup
    std::map<TEType, std::vector<MatrixXd>> teInstancesByType_;

    // Edge connections using type-local indices
    TEToCenterlineIndexEdges teToCenterlineIndexEdges_;
    TEToTEIndexEdges teToTEIndexEdges_;

    std::string uuid_;
  };

  MapData() noexcept : uuid_{boost::lexical_cast<std::string>(boost::uuids::random_generator()())} {}

  /**
   * @brief Extract all instances of a local submap
   * @param localSubmap The local submap to extract from, see extractSubmap()
   * @param localSubmapGraph Routing graph of the submap for vehicles, used to find the lanelet paths
   * @param trafficRules Traffic rules for vehicles, used to decide which lanelets are passable
   * @param bikeSubmapGraph Routing graph of the submap for bicycles. If given, bicycle lanes get their own
   * compound centerlines
   * @param ignoreMapElevation If true, the z coordinate of all map elements is set to 0 on extraction
   * @param lineStringTypeGrouping Grouping that decides chaining and labelling of compound instances
   * @param teTypeGrouping Grouping that decides the labelling of traffic elements
   * @return The extracted, not yet processed data. Call processAll() before using it
   * @throw std::runtime_error if lineStringTypeGrouping does not cover every boundary type,
   * see checkLineStringTypeGroupingCoverage()
   */
  static MapDataPtr build(LaneletSubmapConstPtr& localSubmap, lanelet::routing::RoutingGraphConstPtr localSubmapGraph,
                          traffic_rules::TrafficRulesPtr trafficRules,
                          lanelet::routing::RoutingGraphConstPtr bikeSubmapGraph = nullptr,
                          bool ignoreMapElevation = false,
                          const LineStringTypeGrouping& lineStringTypeGrouping = getDefaultLineStringTypeGrouping(),
                          const TETypeGrouping& teTypeGrouping = getDefaultTETypeGrouping());

  /**
   * @brief Process all instances into the local reference frame given by bbox, see MapInstance::process()
   * @param bbox The local reference frame, everything outside of it is cut away
   * @param paramType Parametrization of the result. Only ParametrizationType::LineString is implemented
   * @param resampleLanes If false, lane instances are not resampled
   * @param nPointsLanes Number of points to resample lane instances to. Values < 2 disable resampling as well
   * @param resampleTE If false, traffic element instances are not resampled
   * @param nPointsTE Number of points to resample traffic elements to. Values < 2 disable resampling as well
   * @param pitch Pitch angle of the local reference frame [rad]
   * @param roll Roll angle of the local reference frame [rad]
   * @return True if every instance is still valid afterwards. False is not an error, it just means that some
   * instances are outside of the bounding box - use the validXXX() accessors to get the remaining ones
   * @note Can be called repeatedly with different bounding boxes, e.g. to build a MapData once per scene and clip
   * it per frame. Every call overwrites the result of the previous one (including the buffer of
   * getTensorInstanceData()), so the result equals that of a freshly built MapData processed only with bbox.
   */
  bool processAll(const OrientedRect& bbox, const ParametrizationType& paramType, bool resampleLanes = true,
                  int32_t nPointsLanes = 0, bool resampleTE = true, int32_t nPointsTE = 0, double pitch = 0,
                  double roll = 0);

  //! All lane line strings of the given type, by map id
  LaneLineStringInstances lineStringsOfType(LineStringType type) const;

  //! Like lineStringsOfType(), but without the instances that did not survive processing
  LaneLineStringInstances validLineStringsOfType(LineStringType type) const;

  //! All compound line strings of the given type
  CompoundLaneLineStringInstanceList compoundLineStringsOfType(LineStringType type) const;

  //! Like compoundLineStringsOfType(), but without the instances that did not survive processing
  CompoundLaneLineStringInstanceList validCompoundLineStringsOfType(LineStringType type) const;

  /// @brief All compound instances of the given type that a map element is part of
  /// @param mapId Id of a line string that is a member of the compound instance, or of a lanelet that uses it.
  /// For compound centerlines this is the lanelet id, since their members are the centerlines of lanelets
  /// @return The associated compound instances, empty if there are none
  CompoundLaneLineStringInstanceList associatedCpdLineStringsOfType(Id mapId, LineStringType type) const;

  //! All traffic elements of the given type, by map id
  TEInstances teInstancesOfType(TEType type) const;

  //! Like teInstancesOfType(), but without the instances that did not survive processing
  TEInstances validTEInstancesOfType(TEType type) const;

  //! All lanelet instances, by map id. Non-driving lanelets are not part of this
  const LaneletInstances& laneletInstances() { return laneletInstances_; }

  //! Lane graph edges by source lanelet id: successors and lane changes between lanelets
  const Edges& llEdges() { return llEdges_; }

  //! Traffic element edges by source element id, targets are either lanelets or other traffic elements
  const Edges& teEdges() { return teEdges_; }

  //! teEdges() resolved to instance pointers, for the edges that lead to a lanelet
  const TEToCenterlineEdges& teToCenterlineEdges() const { return teToCenterlineEdges_; }

  //! teEdges() resolved to instance pointers, for the edges that lead to another traffic element
  const TEToTEEdges& teToTEEdges() const { return teToTEEdges_; }

  //! Randomly generated id of this sample, kept across serialization
  const std::string& uuid() { return uuid_; }

  /// @brief Get all instance labels in matrix form
  /// @param pointsIn2d If true, points are 2d (x, y). Forced to true if the bounding box came from a 2d pose
  /// @param ignoreBuffer If true, the data is rebuilt instead of being taken from the buffer
  /// @return The tensor data, only useful if the instances were processed beforehand (e.g. with processAll())
  /// @note The computed data is buffered, so if the underlying instances change you need to set ignoreBuffer
  /// appropriately. In particular, changing pointsIn2d alone does not invalidate the buffer
  TensorInstanceData getTensorInstanceData(bool pointsIn2d, bool ignoreBuffer);

  template <class Archive>
  friend void boost::serialization::serialize(Archive& ar, lanelet::ml_converter::MapData& feat,
                                              const unsigned int /*version*/);

 private:
  //! Creates the instances of all left lanelet boundaries and the lane change edges to the left
  void initLeftBoundaries(LaneletSubmapConstPtr& localSubmap, lanelet::routing::RoutingGraphConstPtr localSubmapGraph,
                          traffic_rules::TrafficRulesPtr trafficRules, bool ignoreMapElevation = false);

  //! Creates the instances of all right lanelet boundaries and the lane change edges to the right
  void initRightBoundaries(LaneletSubmapConstPtr& localSubmap, lanelet::routing::RoutingGraphConstPtr localSubmapGraph,
                           traffic_rules::TrafficRulesPtr trafficRules, bool ignoreMapElevation = false);

  //! Creates a lanelet instance for every lanelet that can be driven on, skipping crosswalks, walkways and stairs
  void initLaneletInstances(LaneletSubmapConstPtr& localSubmap, lanelet::routing::RoutingGraphConstPtr localSubmapGraph,
                            traffic_rules::TrafficRulesPtr trafficRules, bool ignoreMapElevation = false);

  /// @brief Creates all compound instances of the submap
  /// Enumerates the lanelet paths of the submap (see getPaths()), chains the boundaries of each path into
  /// candidate chains and resolves the overlapping candidates into disjoint chains afterwards, so that the
  /// result does not depend on the order in which the paths were enumerated. Also creates the compound
  /// centerlines and the drivable area borders.
  void initCompoundInstances(LaneletSubmapConstPtr& localSubmap,
                             lanelet::routing::RoutingGraphConstPtr localSubmapGraph,
                             traffic_rules::TrafficRulesPtr trafficRules,
                             lanelet::routing::RoutingGraphConstPtr bikeSubmapGraph, bool ignoreMapElevation = false);

  //! Builds the lookup behind associatedCpdLineStringsOfType(), must run after all compound instances exist
  void updateAssociatedCpdInstanceIndices();

  /// @brief Collects the lanelet paths that start at the given lanelet, splitting on junctions
  /// Paths start at lanelets without predecessor, split into several paths on junctions and end at lanelets
  /// without successor. Only successors of the same subtype are followed, so a path never mixes e.g. a bicycle
  /// lane with a road. Also fills llEdges_ with the successor edges along the way.
  void getPaths(lanelet::routing::RoutingGraphConstPtr localSubmapGraph, std::vector<ConstLanelets>& paths,
                ConstLanelet start, ConstLanelets initPath = ConstLanelets());

  //! @throw std::runtime_error if the id is not among the lane line strings
  LineStringType getLineStringTypeFromId(Id id);

  //! Gets the instance with that id, inverting it if it does not already run in the requested direction
  //! @throw std::runtime_error if the id is not among the lane line strings
  LaneLineStringInstancePtr getLineStringFeatFromId(Id id, bool inverted);

  //! Chains the left boundaries along the path, starting a new chain whenever the type group changes
  std::vector<internal::CompoundElsList> computeCompoundLeftBorders(const ConstLanelets& path);

  //! Chains the right boundaries along the path, starting a new chain whenever the type group changes
  std::vector<internal::CompoundElsList> computeCompoundRightBorders(const ConstLanelets& path);

  //! Chains the centerlines of the path into one compound instance, of bike type if the path is a bicycle lane
  CompoundLaneLineStringInstancePtr computeCompoundCenterline(const ConstLanelets& path,
                                                              bool ignoreMapElevation = false);

  /// @brief Collects the line strings tagged with `drivable_space_border` and chains them into compound instances
  /// These line strings state the extent of the physically drivable space directly, independently of how the
  /// lane boundaries are tagged. They are not lanelet boundaries, so their instances have no lanelet ids.
  /// Two of them are chained if they share an endpoint *point*, i.e. the same Point3d id - coordinates are
  /// not compared. Each connected group is walked into continuous chains, starting at an open end where there
  /// is one; a junction of more than two borders ends a chain, so a branching drivable space yields several
  /// compound instances rather than one with a jump in it.
  void computeDrivableAreaBorders(LaneletSubmapConstPtr& localSubmap, bool ignoreMapElevation = false);

  // Collect non-lane traffic elements

  //! Collects stop lines and the edges to the lanelets they apply to. Artificial stop lines are skipped
  void collectStopLines(LaneletSubmapConstPtr& localSubmap, bool ignoreMapElevation = false);

  //! Collects arrow markings and the edge to the lanelet each one lies on
  void collectArrows(LaneletSubmapConstPtr& localSubmap, bool ignoreMapElevation = false);

  //! Collects traffic lights and the edges to their stop lines. Unlifted elements (z < -1000) are skipped
  void collectTrafficLights(LaneletSubmapConstPtr& localSubmap, bool ignoreMapElevation = false);

  //! Collects traffic signs. Unlifted elements (z < -1000) are skipped
  void collectTrafficSigns(LaneletSubmapConstPtr& localSubmap, bool ignoreMapElevation = false);

  //! Collects symbol markings (bike, bus, speed limits) and the edge to the lanelet each one lies on
  void collectSymbols(LaneletSubmapConstPtr& localSubmap, bool ignoreMapElevation = false);

  //! Turns crosswalk lanelets into compound instances that trace their closed perimeter
  void collectPedestrianCrossings(LaneletSubmapConstPtr& localSubmap, bool ignoreMapElevation = false);

  //! Resolves teEdges_ into teToCenterlineEdges_ and teToTEEdges_, must run after all instances exist
  void convertTEEdges();

  //! all lane line strings (road borders, lane dividers, ...)
  LaneLineStringInstances laneLineStrings_;

  //! all compound line strings (road borders, lane dividers, centerlines, ...)
  CompoundLaneLineStringInstanceList compoundLaneLineStrings_;

  //! all traffic element instances (stop lines, road markings, traffic lights, traffic signs, ...)
  TEInstances teInstances_;

  //! all lanelets that can be driven on
  LaneletInstances laneletInstances_;

  //! Maps LineStringType to a map of (mapId -> indices into compoundLaneLineStrings_)
  std::map<LineStringType, std::map<Id, std::vector<size_t>>> associatedCpdLineStringsIndices_;

  Edges llEdges_;  //!< edge list for lanelet/centerline connectivity
  Edges teEdges_;  //!< edge list for traffic element connectivity (e.g., traffic light to stop line)

  // Converted TE edge associations
  TEToCenterlineEdges teToCenterlineEdges_;
  TEToTEEdges teToTEEdges_;

  std::string uuid_;  //!< sample id

  Optional<TensorInstanceData> tfData_;  //!< buffer of getTensorInstanceData()
  LineStringTypeGrouping lineStringTypeGrouping_{getDefaultLineStringTypeGrouping()};
  TETypeGrouping teTypeGrouping_{getDefaultTETypeGrouping()};
};

}  // namespace ml_converter
}  // namespace lanelet
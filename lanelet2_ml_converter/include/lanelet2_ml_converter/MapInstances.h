#pragma once
#include <lanelet2_core/Exceptions.h>
#include <lanelet2_core/Forward.h>
#include <lanelet2_core/geometry/LineString.h>
#include <lanelet2_core/utility/Optional.h>

#include <boost/geometry.hpp>
#include <type_traits>

#include "lanelet2_ml_converter/Forward.h"
#include "lanelet2_ml_converter/Types.h"

namespace lanelet {
namespace ml_converter {

using VectorXd = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using MatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

using BasicLineStrings3d = std::vector<BasicLineString3d>;

struct LStringProcessResult;  // internal result of processing one line string, see MapInstances.cpp

/**
 * @brief Base class of all local instance labels
 *
 * An instance is one label of a MapData object. It is created from one map element (its mapID()) and is then
 * brought into the local reference frame by process(), which runs the following pipeline:
 * 1. **cut**: the raw geometry is clipped to the local bounding box (see cutLineString())
 * 2. **transform**: the clipped geometry is moved into the bounding box frame (see transformLineString())
 * 3. **resample**: the transformed geometry is resampled to a fixed number of points (see resampleLineString()),
 *    unless resampling was disabled by passing nPoints < 2
 *
 * The results of all three stages are kept, see LineStringInstance. Because clipping can split one line into
 * several disjoint pieces, every stage stores a *list* of line strings, not a single one.
 *
 * @note An instance is only meaningful after process() has been called on it - MapData::processAll() does this
 * for every instance it holds.
 * @note process() can be called repeatedly, e.g. with the bounding boxes of consecutive frames. Every call
 * overwrites all processed state (valid(), wasCut() and the processed geometry), so the result only depends on
 * the raw instance and the arguments of the latest call.
 */
class MapInstance {
 public:
  //! Id of the map element this instance was created from, InvalId if the instance was default constructed
  const Id mapID() const { return mapID_.get_value_or(InvalId); }

  //! Whether this instance is linked to a map element, i.e. whether it was constructed with a map id
  bool initialized() const { return initialized_; }

  /// @brief Whether this instance still carries usable geometry after processing
  /// False if the local bounding box does not contain (enough of) this instance, see process(). Invalid
  /// instances are filtered out by the validXXX() accessors of MapData and are not part of the tensor data.
  bool valid() const { return valid_; }

  //! Whether processing removed geometry from this instance, i.e. whether it reaches beyond the bounding box
  bool wasCut() const { return wasCut_; }

  /// @brief Get the instance as flat vectors of the form [x, y, (z)] * n (+ type as last element)
  /// @param onlyPoints if true, the type is not appended
  /// @param pointsIn2d if true, points are 2d (x, y). Forced to true if this instance was processed from a 2d pose
  /// @return One vector per line string piece the instance consists of after processing
  virtual std::vector<VectorXd> computeInstanceVectors(bool onlyPoints, bool pointsIn2d) const = 0;

  /// @brief Cut, transform and resample this instance into the local reference frame of bbox
  /// @param bbox Local bounding box, everything outside of it is cut away
  /// @param paramType Parametrization of the result. Only ParametrizationType::LineString is implemented
  /// @param nPoints Number of points to resample to. Values < 2 disable resampling
  /// @param pitch Pitch angle of the local reference frame [rad]
  /// @param roll Roll angle of the local reference frame [rad]
  /// @return valid(), i.e. false if the instance does not survive the cut
  /// @note Overwrites the result of any previous call, see the class documentation
  virtual bool process(const OrientedRect& bbox, const ParametrizationType& paramType, int32_t nPoints,
                       double pitch = 0, double roll = 0) = 0;

  template <class Archive>
  friend void boost::serialization::serialize(Archive& ar, lanelet::ml_converter::MapInstance& feat,
                                              const unsigned int /*version*/);

  virtual ~MapInstance() noexcept = default;

 protected:
  bool initialized_{false};
  bool valid_{true};
  bool wasCut_{false};
  bool processedFrom2d_{false};  //!< set by process(), true if the bounding box was built from a 2d pose
  Optional<Id> mapID_;

  MapInstance() = default;
  MapInstance(Id mapID) : mapID_{mapID}, initialized_{true} {}
};

/**
 * @brief Abstract instance whose geometry is a line string
 *
 * Keeps the result of every stage of the processing pipeline described in MapInstance. The lists are index
 * aligned per surviving piece: element i of cutInstance(), cutAndTransformedInstance() and
 * cutTransformedAndResampledInstance() belong to the same piece of the original line.
 *
 * @note If resampling is enabled and a piece is too short to be resampled, that piece is dropped from all three
 * lists instead of being emitted as an empty geometry. The lists therefore stay aligned, but they can be
 * shorter than the number of pieces the cut produced.
 */
class LineStringInstance : public MapInstance {
 public:
  //! The unprocessed geometry in the map frame, as it was taken from the map element
  const BasicLineString3d& rawInstance() const { return rawInstance_; }

  virtual std::vector<VectorXd> computeInstanceVectors(bool onlyPoints, bool pointsIn2d) const = 0;

  /// @brief Get the points of the processed instance as matrices of shape (nPoints, 2 or 3)
  /// @param pointsIn2d if true, points are 2d (x, y). Forced to true if this instance was processed from a 2d pose
  /// @return One matrix per line string piece the instance consists of after processing
  virtual std::vector<MatrixXd> pointMatrices(bool pointsIn2d) const = 0;

  virtual bool process(const OrientedRect& bbox, const ParametrizationType& paramType, int32_t nPoints,
                       double pitch = 0, double roll = 0) = 0;

  //! Geometry clipped to the bounding box, still in the map frame
  const BasicLineStrings3d& cutInstance() const { return cutInstances_; }

  //! Geometry clipped to the bounding box and transformed into its local frame
  const BasicLineStrings3d& cutAndTransformedInstance() const { return cutAndTransformedInstances_; }

  //! Geometry clipped, transformed and resampled. Empty if resampling was disabled
  const BasicLineStrings3d& cutTransformedAndResampledInstance() const { return cutTransformedAndResampledInstances_; }

  template <class Archive>
  friend void boost::serialization::serialize(Archive& ar, lanelet::ml_converter::LineStringInstance& feat,
                                              const unsigned int /*version*/);

  virtual ~LineStringInstance() noexcept = default;

 protected:
  BasicLineString3d rawInstance_;
  BasicLineStrings3d cutInstances_;
  BasicLineStrings3d cutAndTransformedInstances_;
  BasicLineStrings3d cutTransformedAndResampledInstances_;
  LineStringInstance() {}
  LineStringInstance(const BasicLineString3d& feature, Id mapID) : MapInstance(mapID), rawInstance_{feature} {}

  //! Replaces all processed state (geometry lists, valid_, wasCut_, processedFrom2d_) with the given result
  void applyProcessResult(LStringProcessResult&& result, bool from2d);
};

/**
 * @brief A line string that is part of a lane, e.g. a lane divider, a road border or a centerline
 *
 * The geometry of an instance always runs in the driving direction of the lanelets that use it, which is not
 * necessarily the direction of the map element it comes from - see inverted().
 */
class LaneLineStringInstance : public LineStringInstance {
 public:
  LaneLineStringInstance() {}

  /// @brief Construct from raw geometry
  /// @param feature The raw geometry in the map frame
  /// @param mapID Id of the map element this instance comes from
  /// @param type Type of this line string
  /// @param laneletID Ids of the lanelets this line string belongs to
  /// @param inverted Whether the geometry runs against the direction of the map element, see inverted()
  LaneLineStringInstance(const BasicLineString3d& feature, Id mapID, LineStringType type,
                         const std::vector<Id>& laneletID, bool inverted)
      : LineStringInstance(feature, mapID), type_{type}, laneletIDs_{laneletID}, inverted_{inverted} {}
  virtual ~LaneLineStringInstance() noexcept = default;

  virtual bool process(const OrientedRect& bbox, const ParametrizationType& paramType, int32_t nPoints,
                       double pitch = 0, double roll = 0) override;

  //! Uses the resampled geometry if available, the transformed one otherwise. Empty before process() was called
  virtual std::vector<VectorXd> computeInstanceVectors(bool onlyPoints, bool pointsIn2d) const override;

  //! Uses the resampled geometry if available, the transformed one otherwise. Empty before process() was called
  virtual std::vector<MatrixXd> pointMatrices(bool pointsIn2d) const override;

  //! Type of this line string. For compound instances this is the representative type of the group, see
  //! CompoundLaneLineStringInstance
  LineStringType type() const { return type_; }

  //! Whether the geometry runs against the direction of the element with mapID() in the lineStringLayer of the map
  bool inverted() const { return inverted_; }

  //! type() as int, this is what is appended to the instance vectors
  int typeInt() const { return static_cast<int>(type_); }

  //! Ids of all lanelets that use this line string, e.g. both neighbours of a shared lane divider
  const Ids& laneletIDs() const { return laneletIDs_; }

  void addLaneletID(Id id) { laneletIDs_.push_back(id); }

  template <class Archive>
  friend void boost::serialization::serialize(Archive& ar, lanelet::ml_converter::LaneLineStringInstance& feat,
                                              const unsigned int /*version*/);

 protected:
  LineStringType type_;
  bool inverted_{false};  // = inverted compared to element with that Id in lineStringLayer
  Ids laneletIDs_;
};

using LaneLineStringInstancePtr = std::shared_ptr<LaneLineStringInstance>;
using LaneLineStringInstances = std::map<Id, LaneLineStringInstancePtr>;
using LaneLineStringInstanceList = std::vector<LaneLineStringInstancePtr>;

/**
 * @brief A traffic element instance, e.g. a stop line, a road marking, a traffic light or a traffic sign
 *
 * Traffic elements are not part of a lane. They are collected from line strings and polygons of the map that
 * carry the respective tags, see teTypeToEnum(). Traffic elements that are physically above the road (traffic
 * lights and traffic signs) keep their elevation unless it was ignored on map data creation.
 */
class TEInstance : public LineStringInstance {
 public:
  TEInstance() {}

  /// @brief Construct from raw geometry
  /// @param feature The raw geometry in the map frame
  /// @param mapID Id of the map element this instance comes from
  /// @param type Type of this traffic element
  TEInstance(const BasicLineString3d& feature, Id mapID, TEType type)
      : LineStringInstance(feature, mapID), teType_{type} {}
  virtual ~TEInstance() noexcept = default;

  bool process(const OrientedRect& bbox, const ParametrizationType& paramType, int32_t nPoints, double pitch = 0,
               double roll = 0) override;

  //! Uses the resampled geometry if available, the transformed one otherwise. Empty before process() was called
  std::vector<VectorXd> computeInstanceVectors(bool onlyPoints, bool pointsIn2d) const override;

  //! Uses the resampled geometry if available, the transformed one otherwise. Empty before process() was called
  virtual std::vector<MatrixXd> pointMatrices(bool pointsIn2d) const override;

  //! Type of this traffic element. This is the representative type of its group, see TETypeGrouping
  TEType teType() { return teType_; }

  template <class Archive>
  friend void boost::serialization::serialize(Archive& ar, lanelet::ml_converter::TEInstance& feat,
                                              const unsigned int /*version*/);

 private:
  TEType teType_;
};

/**
 * @brief One lanelet, made up of the instances of its left boundary, right boundary and centerline
 *
 * The layout of computeInstanceVectors() depends on the representation type set with setReprType():
 * - LaneletRepresentationType::Centerline: centerline points, then the type of the left and the right boundary
 * - LaneletRepresentationType::Boundaries: left boundary points, right boundary points, then both types
 *
 * @note Only lanelets that can be driven on become lanelet instances. Crosswalks, walkways, shared walkways and
 * stairs are skipped by MapData - crosswalks are collected as compound instances instead.
 */
class LaneletInstance : public MapInstance {
 public:
  LaneletInstance() {}

  //! Construct from already existing boundary and centerline instances
  LaneletInstance(LaneLineStringInstancePtr leftBoundary, LaneLineStringInstancePtr rightBoundary,
                  LaneLineStringInstancePtr centerline, Id mapID);

  //! Construct standalone from a lanelet of the map, creating its boundary and centerline instances
  LaneletInstance(const ConstLanelet& ll);
  virtual ~LaneletInstance() noexcept = default;

  //! Processes the boundaries and the centerline. Invalid if any of the three is invalid
  bool process(const OrientedRect& bbox, const ParametrizationType& paramType, int32_t nPoints, double pitch = 0,
               double roll = 0) override;

  //! Layout depends on the representation type, see the class documentation
  std::vector<VectorXd> computeInstanceVectors(bool onlyPoints, bool pointsIn2d) const override;

  //! Sets how computeInstanceVectors() represents this lanelet, defaults to LaneletRepresentationType::Centerline
  void setReprType(LaneletRepresentationType reprType) { reprType_ = reprType; }

  LaneLineStringInstancePtr leftBoundary() const { return leftBoundary_; }
  LaneLineStringInstancePtr rightBoundary() const { return rightBoundary_; }
  LaneLineStringInstancePtr centerline() const { return centerline_; }

  template <class Archive>
  friend void boost::serialization::serialize(Archive& ar, lanelet::ml_converter::LaneletInstance& feat,
                                              const unsigned int /*version*/);

 private:
  LaneLineStringInstancePtr leftBoundary_;
  LaneLineStringInstancePtr rightBoundary_;
  LaneLineStringInstancePtr centerline_;
  LaneletRepresentationType reprType_{LaneletRepresentationType::Centerline};
};

/**
 * @brief Several line string instances chained into one, e.g. all lane dividers along one lane
 *
 * Compound instances make the labels independent of how the map happens to be split into individual elements.
 * They are built by MapData along the lanelet paths of the local submap: consecutive boundaries are chained as
 * long as their types stay within the same group of the LineStringTypeGrouping in use.
 *
 * @note The type of a compound instance is the *representative* type of that group, not the type of its first
 * member - with the MapTR default grouping, for instance, a chain of dashed and solid boundaries becomes one
 * instance of type LineStringType::Divider.
 * @note The individual instances stay accessible through features(), so every compound instance can be traced
 * back to the map elements it was built from.
 */
class CompoundLaneLineStringInstance : public LaneLineStringInstance {
 public:
  CompoundLaneLineStringInstance() {}

  /// @brief Chain the given instances into one compound instance
  /// @param features The instances to chain, required to be given in already sorted order
  /// @param compoundType Type of the resulting instance, usually the representative type of the group
  CompoundLaneLineStringInstance(const LaneLineStringInstanceList& features, LineStringType compoundType);

  virtual ~CompoundLaneLineStringInstance() noexcept = default;

  //! Processes the chained geometry as well as every individual instance. Valid if at least one member survives
  bool process(const OrientedRect& bbox, const ParametrizationType& paramType, int32_t nPoints, double pitch = 0,
               double roll = 0) override;

  //! The individual instances this compound instance was built from, in the order they are chained
  LaneLineStringInstanceList features() const { return individualInstances_; }

  //! Cumulative length [m] of the raw geometry up to and including each member of features()
  const std::vector<double>& pathLengthsRaw() const { return pathLengthsRaw_; }

  //! Cumulative length [m] of the cut geometry up to and including each member of features(), valid after process()
  const std::vector<double>& pathLengthsProcessed() const { return pathLengthsProcessed_; }

  //! Per member of features(): whether enough of it (more than validLengthThresh_) survived the cut
  const std::vector<bool>& processedInstancesValid() const { return processedInstancesValid_; }

  template <class Archive>
  friend void boost::serialization::serialize(Archive& ar, lanelet::ml_converter::CompoundLaneLineStringInstance& feat,
                                              const unsigned int /*version*/);

 private:
  LaneLineStringInstanceList individualInstances_;
  std::vector<double> pathLengthsRaw_;
  std::vector<double> pathLengthsProcessed_;
  std::vector<bool> processedInstancesValid_;
  double validLengthThresh_{0.3};  //!< minimum length [m] a member must keep to count as valid after the cut
};

using CompoundLaneLineStringInstancePtr = std::shared_ptr<CompoundLaneLineStringInstance>;
using TEInstancePtr = std::shared_ptr<TEInstance>;
using LaneletInstancePtr = std::shared_ptr<LaneletInstance>;
using CompoundLaneLineStringInstanceList = std::vector<CompoundLaneLineStringInstancePtr>;
using TEInstances = std::map<Id, TEInstancePtr>;
using LaneletInstances = std::map<Id, LaneletInstancePtr>;

/// @brief Stack the instance vectors of all given instances into one matrix, one instance vector per row
/// @throw std::runtime_error if one of the instances is invalid, if there is nothing to stack or if the
/// instance vectors do not all have the same length (e.g. because the instances were not resampled)
template <class T>
MatrixXd getInstanceVectorMatrix(const std::map<Id, std::shared_ptr<T>>& mapInstances, bool onlyPoints,
                                 bool pointsIn2d) {
  std::vector<std::shared_ptr<T>> featList;
  for (const auto& pair : mapInstances) {
    featList.push_back(pair.second);
  }
  return getInstanceVectorMatrix(featList, onlyPoints, pointsIn2d);
}

//! Overload of getInstanceVectorMatrix() for instances given as a list
template <class T>
MatrixXd getInstanceVectorMatrix(const std::vector<std::shared_ptr<T>>& mapInstances, bool onlyPoints,
                                 bool pointsIn2d) {
  std::vector<VectorXd> featureVectors;
  for (const auto& feat : mapInstances) {
    if (!feat->valid()) {
      throw std::runtime_error("Invalid feature in list! This function requires all given features to be valid!");
    }
    std::vector<VectorXd> individualVecs = feat->computeInstanceVectors(onlyPoints, pointsIn2d);
    featureVectors.insert(featureVectors.end(), individualVecs.begin(), individualVecs.end());
  }
  if (featureVectors.empty()) {
    throw std::runtime_error("No feature vectors to build a matrix from!");
  }
  if (std::adjacent_find(featureVectors.begin(), featureVectors.end(), [](const VectorXd& v1, const VectorXd& v2) {
        return v1.size() != v2.size();
      }) != featureVectors.end()) {
    throw std::runtime_error(
        "Unequal length of feature vectors! To create a matrix all feature vectors must have the same length!");
  }
  MatrixXd featureMat(featureVectors.size(), featureVectors[0].size());
  for (size_t i = 0; i < featureVectors.size(); i++) {
    featureMat.row(i) = featureVectors[i];
  }
  return featureMat;
}

/// @brief Collect the point matrices of all given instances
/// @throw std::runtime_error if one of the instances is invalid
template <class T>
std::vector<MatrixXd> getPointMatrices(const std::map<Id, std::shared_ptr<T>>& mapInstances, bool pointsIn2d) {
  std::vector<std::shared_ptr<T>> featList;
  for (const auto& pair : mapInstances) {
    featList.push_back(pair.second);
  }
  return getPointMatrices(featList, pointsIn2d);
}

//! Overload of getPointMatrices() for instances given as a list
template <class T>
std::vector<MatrixXd> getPointMatrices(const std::vector<std::shared_ptr<T>>& mapInstances, bool pointsIn2d) {
  std::vector<MatrixXd> pointMatrices;
  for (const auto& feat : mapInstances) {
    if (!feat->valid()) {
      throw std::runtime_error("Invalid feature in list! This function requires all given features to be valid!");
    }
    std::vector<MatrixXd> individualMats = feat->pointMatrices(pointsIn2d);
    pointMatrices.insert(pointMatrices.end(), individualMats.begin(), individualMats.end());
  }
  return pointMatrices;
}

/// @brief Call process() on all given instances
/// @return True if every instance is still valid afterwards
template <typename T>
bool processInstances(std::map<Id, std::shared_ptr<T>>& featMap, const OrientedRect& bbox,
                      const ParametrizationType& paramType, int32_t nPoints, double pitch = 0, double roll = 0) {
  bool allValid = true;
  for (auto& feat : featMap) {
    if (!feat.second->process(bbox, paramType, nPoints, pitch, roll)) {
      allValid = false;
    }
  }
  return allValid;
}

//! Overload of processInstances() for instances given as a list
template <typename T>
bool processInstances(std::vector<std::shared_ptr<T>>& featVec, const OrientedRect& bbox,
                      const ParametrizationType& paramType, int32_t nPoints, double pitch = 0, double roll = 0) {
  bool allValid = true;
  for (auto& feat : featVec) {
    if (!feat->process(bbox, paramType, nPoints, pitch, roll)) {
      allValid = false;
    }
  }
  return allValid;
}

/// @brief Convert a line string to a matrix of shape (nPoints, 2 or 3)
MatrixXd toPointMatrix(const BasicLineString3d& lString, bool pointsIn2d);

/// @brief Convert a line string to a flat vector of the form [x, y, (z)] * n (+ typeInt as last element)
/// @param typeInt The type to append, ignored if onlyPoints is true
VectorXd toInstanceVector(const BasicLineString3d& line, int typeInt, bool onlyPoints, bool pointsIn2d);
}  // namespace ml_converter
}  // namespace lanelet
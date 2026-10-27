#include "lanelet2_ml_converter/MapInstances.h"

#include <boost/geometry.hpp>
#include <type_traits>

#include "lanelet2_ml_converter/Utils.h"

namespace lanelet {
namespace ml_converter {

struct LStringProcessResult {
  BasicLineStrings3d cutInstances;
  BasicLineStrings3d cutAndTransformedInstances;
  BasicLineStrings3d cutTransformedAndResampledInstances;
  bool wasCut_{false};
  bool valid_{true};
};

namespace {
LStringProcessResult processLineStringImpl(const BasicLineString3d& lstring, const OrientedRect& bbox,
                                           const ParametrizationType& paramType, int32_t nPoints, double pitch,
                                           double roll) {
  LStringProcessResult result;
  if (paramType != ParametrizationType::LineString) {
    throw std::runtime_error("Only polyline parametrization is implemented so far!");
  }
  std::vector<BasicLineString3d> cutLines = cutLineString(bbox, lstring);
  if (cutLines.empty()) {
    result.wasCut_ = true;
    result.valid_ = false;
    return result;
  }

  double lengthOriginal = boost::geometry::length(lstring, boost::geometry::strategy::distance::pythagoras<double>());
  double lengthProcessed = 0;
  for (const auto& line : cutLines) {
    lengthProcessed =
        lengthProcessed + boost::geometry::length(line, boost::geometry::strategy::distance::pythagoras<double>());
  }
  if (lengthProcessed < 1e-1) {
    result.wasCut_ = true;
    result.valid_ = false;
    return result;
  }
  if (lengthOriginal - lengthProcessed > 1e-2) {
    result.wasCut_ = true;
  }

  // If nPoints <= 0, skip resampling (only cut and transform)
  bool skipResampling = (nPoints <= 0);

  for (const auto& line : cutLines) {
    // Always transform after cutting
    BasicLineString3d lineTransformed = transformLineString(bbox, line, pitch, roll);

    if (!skipResampling) {
      // Resample the already-transformed line
      BasicLineString3d lineResampled = resampleLineString(lineTransformed, nPoints);
      if (lineResampled.empty()) {
        // Piece too short to resample: drop it entirely instead of emitting an empty point matrix on an
        // otherwise valid instance. The three lists stay index-aligned per surviving piece.
        continue;
      }
      result.cutTransformedAndResampledInstances.push_back(lineResampled);
    }
    result.cutInstances.push_back(line);
    result.cutAndTransformedInstances.push_back(lineTransformed);
  }
  if (result.cutInstances.empty()) {
    result.wasCut_ = true;
    result.valid_ = false;
  }
  return result;
}

}  // namespace

void LineStringInstance::applyProcessResult(LStringProcessResult&& result, bool from2d) {
  // Overwrite every per-box field, so that the outcome depends only on the raw instance and the arguments of the
  // current process() call and never on a previous call with another box. An invalid result has empty lists.
  cutInstances_ = std::move(result.cutInstances);
  cutAndTransformedInstances_ = std::move(result.cutAndTransformedInstances);
  cutTransformedAndResampledInstances_ = std::move(result.cutTransformedAndResampledInstances);
  valid_ = result.valid_;
  wasCut_ = result.wasCut_;
  processedFrom2d_ = from2d;
}

bool LaneLineStringInstance::process(const OrientedRect& bbox, const ParametrizationType& paramType, int32_t nPoints,
                                     double pitch, double roll) {
  applyProcessResult(processLineStringImpl(rawInstance_, bbox, paramType, nPoints, pitch, roll), bbox.from2d);
  return valid_;
}

std::vector<VectorXd> LaneLineStringInstance::computeInstanceVectors(bool onlyPoints, bool pointsIn2d) const {
  std::vector<VectorXd> featVecs;
  // Use resampled if available, otherwise use transformed only
  const BasicLineStrings3d& source = !cutTransformedAndResampledInstances_.empty()
                                         ? cutTransformedAndResampledInstances_
                                         : cutAndTransformedInstances_;
  for (const auto& split : source) {
    featVecs.push_back(toInstanceVector(split, typeInt(), onlyPoints, (pointsIn2d || processedFrom2d_)));
  }
  return featVecs;
}

std::vector<VectorXd> TEInstance::computeInstanceVectors(bool onlyPoints, bool pointsIn2d) const {
  std::vector<VectorXd> featVecs;
  // Use resampled if available, otherwise use transformed only
  const BasicLineStrings3d& source = !cutTransformedAndResampledInstances_.empty()
                                         ? cutTransformedAndResampledInstances_
                                         : cutAndTransformedInstances_;
  for (const auto& split : source) {
    featVecs.push_back(
        toInstanceVector(split, static_cast<int>(teType_), onlyPoints, (pointsIn2d || processedFrom2d_)));
  }
  return featVecs;
}

std::vector<MatrixXd> LaneLineStringInstance::pointMatrices(bool pointsIn2d) const {
  std::vector<MatrixXd> pointMatrices;
  // Use resampled if available, otherwise use transformed only
  const BasicLineStrings3d& source = !cutTransformedAndResampledInstances_.empty()
                                         ? cutTransformedAndResampledInstances_
                                         : cutAndTransformedInstances_;
  for (const auto& split : source) {
    pointMatrices.push_back(toPointMatrix(split, (pointsIn2d || processedFrom2d_)));
  }
  return pointMatrices;
}

std::vector<MatrixXd> TEInstance::pointMatrices(bool pointsIn2d) const {
  std::vector<MatrixXd> pointMatrices;

  // Use resampled if available, otherwise use transformed only
  const BasicLineStrings3d& source = !cutTransformedAndResampledInstances_.empty()
                                         ? cutTransformedAndResampledInstances_
                                         : cutAndTransformedInstances_;
  for (const auto& split : source) {
    pointMatrices.push_back(toPointMatrix(split, (pointsIn2d || processedFrom2d_)));
  }

  return pointMatrices;
}

bool TEInstance::process(const OrientedRect& bbox, const ParametrizationType& paramType, int32_t nPoints, double pitch,
                         double roll) {
  applyProcessResult(processLineStringImpl(rawInstance_, bbox, paramType, nPoints, pitch, roll), bbox.from2d);
  return valid_;
}

LaneletInstance::LaneletInstance(LaneLineStringInstancePtr leftBoundary, LaneLineStringInstancePtr rightBoundary,
                                 LaneLineStringInstancePtr centerline, Id mapID)
    : MapInstance(mapID), leftBoundary_{leftBoundary}, rightBoundary_{rightBoundary}, centerline_{centerline} {}

LaneletInstance::LaneletInstance(const ConstLanelet& ll)
    : leftBoundary_{std::make_shared<LaneLineStringInstance>(ll.leftBound3d().basicLineString(), ll.leftBound3d().id(),
                                                             bdTypeToEnum(ll.leftBound3d()), Ids{ll.id()},
                                                             ll.leftBound3d().inverted())},
      rightBoundary_{std::make_shared<LaneLineStringInstance>(ll.rightBound3d().basicLineString(),
                                                              ll.rightBound3d().id(), bdTypeToEnum(ll.rightBound3d()),
                                                              Ids{ll.id()}, ll.rightBound3d().inverted())},
      centerline_{std::make_shared<LaneLineStringInstance>(ll.centerline3d().basicLineString(), ll.centerline3d().id(),
                                                           LineStringType::Centerline, Ids{ll.id()}, false)} {}

bool LaneletInstance::process(const OrientedRect& bbox, const ParametrizationType& paramType, int32_t nPoints,
                              double pitch, double roll) {
  leftBoundary_->process(bbox, paramType, nPoints, pitch, roll);
  rightBoundary_->process(bbox, paramType, nPoints, pitch, roll);
  centerline_->process(bbox, paramType, nPoints, pitch, roll);

  // Derived from the current state of the members only, never from a previous call
  wasCut_ = leftBoundary_->wasCut() || rightBoundary_->wasCut() || centerline_->wasCut();
  valid_ = leftBoundary_->valid() && rightBoundary_->valid() && centerline_->valid();
  processedFrom2d_ = bbox.from2d;
  return valid_;
}

VectorXd stackVector(const std::vector<VectorXd>& vec) {
  size_t flattenedLength = 0;
  for (const auto& el : vec) {
    flattenedLength = flattenedLength + el.size();
  }
  VectorXd stacked(flattenedLength);
  size_t currIndex = 0;
  for (const auto& el : vec) {
    stacked.segment(currIndex, el.size()) = el;
    currIndex = currIndex + el.size();
  }
  return stacked;
}

std::vector<VectorXd> LaneletInstance::computeInstanceVectors(bool onlyPoints, bool pointsIn2d) const {
  if (reprType_ == LaneletRepresentationType::Centerline) {
    std::vector<VectorXd> vecCenterlinePts = centerline_->computeInstanceVectors(true, pointsIn2d);
    if (onlyPoints) {
      return vecCenterlinePts;
    }
    std::vector<VectorXd> featureVecs(vecCenterlinePts.size());
    for (size_t i = 0; i < vecCenterlinePts.size(); i++) {
      VectorXd vec(vecCenterlinePts[i].size() + 2);  // pts vec + left and right type
      vec.segment(0, vecCenterlinePts[i].size()) = vecCenterlinePts[i];
      vec[vec.size() - 2] = leftBoundary_->typeInt();
      vec[vec.size() - 1] = rightBoundary_->typeInt();
      featureVecs[i] = vec;
    }
    return featureVecs;
  } else if (reprType_ == LaneletRepresentationType::Boundaries) {
    std::vector<VectorXd> vecLeftBdPts = leftBoundary_->computeInstanceVectors(true, pointsIn2d);
    VectorXd leftBdPts = stackVector(vecLeftBdPts);
    std::vector<VectorXd> vecRightBdPts = rightBoundary_->computeInstanceVectors(true, pointsIn2d);
    VectorXd rightBdPts = stackVector(vecRightBdPts);

    VectorXd vec(leftBdPts.size() + rightBdPts.size() + 2);  // pts vec + left and right type
    vec.segment(0, leftBdPts.size()) = leftBdPts;
    vec.segment(leftBdPts.size(), rightBdPts.size()) = rightBdPts;
    vec[vec.size() - 2] = leftBoundary_->typeInt();
    vec[vec.size() - 1] = rightBoundary_->typeInt();

    if (onlyPoints) {
      return std::vector<VectorXd>{vec.segment(0, vec.size() - 2)};
    } else {
      return std::vector<VectorXd>{vec};
    }
  } else {
    throw std::runtime_error("Unknown LaneletRepresentationType!");
    return std::vector<VectorXd>();
  }
}

CompoundLaneLineStringInstance::CompoundLaneLineStringInstance(const LaneLineStringInstanceList& features,
                                                               LineStringType compoundType)
    : individualInstances_{features},
      pathLengthsRaw_{std::vector<double>(features.size())},
      pathLengthsProcessed_{std::vector<double>(features.size())},
      processedInstancesValid_{std::vector<bool>(features.size())} {
  type_ = compoundType;

  for (size_t i = 0; i < features.size(); i++) {
    if (features[i]->rawInstance().empty()) {
      throw std::runtime_error("Instance with empty rawInstance() supplied!");
    }
    // checking if the linestring is correctly
    if (i == features.size() - 1) {
      rawInstance_.insert(rawInstance_.end(), features[i]->rawInstance().begin(), features[i]->rawInstance().end());
    } else {
      rawInstance_.insert(rawInstance_.end(), features[i]->rawInstance().begin(), features[i]->rawInstance().end() - 1);
    }

    double rawLength =
        boost::geometry::length(features[i]->rawInstance(), boost::geometry::strategy::distance::pythagoras<double>());
    if (i > 0) {
      pathLengthsRaw_[i] = pathLengthsRaw_[i - 1] + rawLength;
    } else {
      pathLengthsRaw_[i] = rawLength;
    }
  }
}

bool CompoundLaneLineStringInstance::process(const OrientedRect& bbox, const ParametrizationType& paramType,
                                             int32_t nPoints, double pitch, double roll) {
  bool valid = false;
  LaneLineStringInstance::process(bbox, paramType, nPoints, pitch, roll);
  // Reset the per-member results of a previous call, they are only ever set to true below
  processedInstancesValid_.assign(individualInstances_.size(), false);
  pathLengthsProcessed_.assign(individualInstances_.size(), 0.);
  for (size_t i = 0; i < individualInstances_.size(); i++) {
    individualInstances_[i]->process(bbox, paramType, nPoints, pitch, roll);

    double processedLength = 0;
    for (const auto& line : individualInstances_[i]->cutInstance()) {
      processedLength =
          processedLength + boost::geometry::length(line, boost::geometry::strategy::distance::pythagoras<double>());
    }

    if (processedLength > validLengthThresh_) {
      processedInstancesValid_[i] = true;
      valid = true;
    }

    if (i > 0) {
      pathLengthsProcessed_[i] = pathLengthsProcessed_[i - 1] + processedLength;
    } else {
      pathLengthsProcessed_[i] = processedLength;
    }
  }
  processedFrom2d_ = bbox.from2d;
  return valid;
}

MatrixXd toPointMatrix(const BasicLineString3d& lString, bool pointsIn2d) {
  MatrixXd mat =
      pointsIn2d ? MatrixXd(lString.size(), 2) : MatrixXd(lString.size(), 3);  // n points with 2/3 dims + type
  if (pointsIn2d == true) {
    for (size_t i = 0; i < lString.size(); i++) {
      mat.row(i) = lString[i].segment(0, 2);
    }
  } else {
    for (size_t i = 0; i < lString.size(); i++) {
      mat.row(i) = lString[i].segment(0, 3);
    }
  }
  return mat;
}

VectorXd toInstanceVector(const BasicLineString3d& line, int typeInt, bool onlyPoints, bool pointsIn2d) {
  VectorXd vec =
      pointsIn2d ? VectorXd(2 * line.size() + 1) : VectorXd(3 * line.size() + 1);  // n points with 2/3 dims + type
  if (pointsIn2d == true) {
    for (size_t i = 0; i < line.size(); i++) {
      vec.segment(2 * i, 2) = line[i].segment(0, 2);
    }
  } else {
    for (size_t i = 0; i < line.size(); i++) {
      vec.segment(3 * i, 3) = line[i].segment(0, 3);
    }
  }
  vec[vec.size() - 1] = typeInt;
  if (onlyPoints) {
    return vec.segment(0, vec.size() - 1);
  } else {
    return vec;
  }
}

}  // namespace ml_converter
}  // namespace lanelet
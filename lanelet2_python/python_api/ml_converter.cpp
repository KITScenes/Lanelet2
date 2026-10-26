#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <map>
#include <vector>

#include "lanelet2_ml_converter/MapData.h"
#include "lanelet2_ml_converter/MapDataInterface.h"
#include "lanelet2_ml_converter/MapInstances.h"
#include "lanelet2_ml_converter/Utils.h"
#include "lanelet2_python/internal/converter.h"
#include "lanelet2_python/internal/eigen_converter.h"

using namespace boost::python;
using namespace lanelet;
using namespace lanelet::ml_converter;

class MapInstanceWrap : public MapInstance, public wrapper<MapInstance> {
 public:
  std::vector<VectorXd> computeInstanceVectors(bool onlyPoints, bool pointsIn2d) const {
    return this->get_override("computeInstanceVectors")(onlyPoints, pointsIn2d);
  }
  bool process(const OrientedRect &bbox, const ParametrizationType &paramType, int32_t nPoints, double pitch,
               double roll) {
    return this->get_override("process")(bbox, paramType, nPoints, pitch, roll);
  }
};

class LineStringInstanceWrap : public LineStringInstance, public wrapper<LineStringInstance> {
 public:
  std::vector<VectorXd> computeInstanceVectors(bool onlyPoints, bool pointsIn2d) const {
    return this->get_override("computeInstanceVectors")(onlyPoints, pointsIn2d);
  }
  bool process(const OrientedRect &bbox, const ParametrizationType &paramType, int32_t nPoints, double pitch,
               double roll) {
    return this->get_override("process")(bbox, paramType, nPoints, pitch, roll);
  }
  std::vector<MatrixXd> pointMatrices(bool pointsIn2d) const { return this->get_override("pointMatrices")(pointsIn2d); }
};

class LaneLineStringInstanceWrap : public LaneLineStringInstance, public wrapper<LaneLineStringInstance> {
 public:
  LaneLineStringInstanceWrap() {}

  LaneLineStringInstanceWrap(const BasicLineString3d &feature, Id mapID, LineStringType type,
                             const std::vector<Id> &laneletID, bool inverted)
      : LaneLineStringInstance(feature, mapID, type, laneletID, inverted) {}

  std::vector<VectorXd> computeInstanceVectors(bool onlyPoints, bool pointsIn2d) const {
    if (override f = this->get_override("computeInstanceVectors")) return f(onlyPoints, pointsIn2d);
    return LaneLineStringInstance::computeInstanceVectors(onlyPoints, pointsIn2d);
  }
  std::vector<VectorXd> default_computeInstanceVectors(bool onlyPoints, bool pointsIn2d) const {
    return this->LaneLineStringInstance::computeInstanceVectors(onlyPoints, pointsIn2d);
  }
  bool process(const OrientedRect &bbox, const ParametrizationType &paramType, int32_t nPoints, double pitch,
               double roll) {
    if (override f = this->get_override("process")) return f(bbox, paramType, nPoints, pitch, roll);
    return LaneLineStringInstance::process(bbox, paramType, nPoints, pitch, roll);
  }
  bool default_process(const OrientedRect &bbox, const ParametrizationType &paramType, int32_t nPoints, double pitch,
                       double roll) {
    return this->LaneLineStringInstance::process(bbox, paramType, nPoints, pitch, roll);
  }
  std::vector<MatrixXd> pointMatrices(bool pointsIn2d) const {
    if (override f = this->get_override("pointMatrices")) return f(pointsIn2d);
    return LaneLineStringInstance::pointMatrices(pointsIn2d);
  }
  std::vector<MatrixXd> default_pointMatrices(bool pointsIn2d) const {
    return this->LaneLineStringInstance::pointMatrices(pointsIn2d);
  }
};

class TEInstanceWrap : public TEInstance, public wrapper<TEInstance> {
 public:
  TEInstanceWrap() {}

  TEInstanceWrap(const BasicLineString3d &feature, Id mapID, TEType type) : TEInstance(feature, mapID, type) {}

  std::vector<VectorXd> computeInstanceVectors(bool onlyPoints, bool pointsIn2d) const {
    if (override f = this->get_override("computeInstanceVectors")) return f(onlyPoints, pointsIn2d);
    return TEInstance::computeInstanceVectors(onlyPoints, pointsIn2d);
  }
  std::vector<VectorXd> default_computeInstanceVectors(bool onlyPoints, bool pointsIn2d) const {
    return this->TEInstance::computeInstanceVectors(onlyPoints, pointsIn2d);
  }
  bool process(const OrientedRect &bbox, const ParametrizationType &paramType, int32_t nPoints, double pitch,
               double roll) {
    if (override f = this->get_override("process")) return f(bbox, paramType, nPoints, pitch, roll);
    return TEInstance::process(bbox, paramType, nPoints, pitch, roll);
  }
  bool default_process(const OrientedRect &bbox, const ParametrizationType &paramType, int32_t nPoints, double pitch,
                       double roll) {
    return this->TEInstance::process(bbox, paramType, nPoints, pitch, roll);
  }
  std::vector<MatrixXd> pointMatrices(bool pointsIn2d) const {
    if (override f = this->get_override("pointMatrices")) return f(pointsIn2d);
    return TEInstance::pointMatrices(pointsIn2d);
  }
  std::vector<MatrixXd> default_pointMatrices(bool pointsIn2d) const {
    return this->TEInstance::pointMatrices(pointsIn2d);
  }
};

class CompoundLaneLineStringInstanceWrap : public CompoundLaneLineStringInstance,
                                           public wrapper<CompoundLaneLineStringInstance> {
 public:
  CompoundLaneLineStringInstanceWrap() {}

  CompoundLaneLineStringInstanceWrap(const LaneLineStringInstanceList &features, LineStringType compoundType)
      : CompoundLaneLineStringInstance(features, compoundType) {}

  std::vector<VectorXd> computeInstanceVectors(bool onlyPoints, bool pointsIn2d) const {
    if (override f = this->get_override("computeInstanceVectors")) return f(onlyPoints, pointsIn2d);
    return CompoundLaneLineStringInstance::computeInstanceVectors(onlyPoints, pointsIn2d);
  }
  std::vector<VectorXd> default_computeInstanceVectors(bool onlyPoints, bool pointsIn2d) const {
    return this->CompoundLaneLineStringInstance::computeInstanceVectors(onlyPoints, pointsIn2d);
  }
  bool process(const OrientedRect &bbox, const ParametrizationType &paramType, int32_t nPoints, double pitch,
               double roll) {
    if (override f = this->get_override("process")) return f(bbox, paramType, nPoints, pitch, roll);
    return CompoundLaneLineStringInstance::process(bbox, paramType, nPoints, pitch, roll);
  }
  bool default_process(const OrientedRect &bbox, const ParametrizationType &paramType, int32_t nPoints, double pitch,
                       double roll) {
    return this->CompoundLaneLineStringInstance::process(bbox, paramType, nPoints, pitch, roll);
  }
  std::vector<MatrixXd> pointMatrices(bool pointsIn2d) const {
    if (override f = this->get_override("pointMatrices")) return f(pointsIn2d);
    return CompoundLaneLineStringInstance::pointMatrices(pointsIn2d);
  }
  std::vector<MatrixXd> default_pointMatrices(bool pointsIn2d) const {
    return this->CompoundLaneLineStringInstance::pointMatrices(pointsIn2d);
  }
};

class LaneletInstanceWrap : public LaneletInstance, public wrapper<LaneletInstance> {
 public:
  LaneletInstanceWrap() {}

  LaneletInstanceWrap(LaneLineStringInstancePtr leftBoundary, LaneLineStringInstancePtr rightBoundary,
                      LaneLineStringInstancePtr centerline, Id mapID)
      : LaneletInstance(leftBoundary, rightBoundary, centerline, mapID) {}
  LaneletInstanceWrap(const ConstLanelet &ll) : LaneletInstance(ll) {}

  std::vector<VectorXd> computeInstanceVectors(bool onlyPoints, bool pointsIn2d) const {
    if (override f = this->get_override("computeInstanceVectors")) return f(onlyPoints, pointsIn2d);
    return LaneletInstance::computeInstanceVectors(onlyPoints, pointsIn2d);
  }
  std::vector<VectorXd> default_computeInstanceVectors(bool onlyPoints, bool pointsIn2d) const {
    return this->LaneletInstance::computeInstanceVectors(onlyPoints, pointsIn2d);
  }
  bool process(const OrientedRect &bbox, const ParametrizationType &paramType, int32_t nPoints, double pitch = 0,
               double roll = 0) {
    if (override f = this->get_override("process")) return f(bbox, paramType, nPoints, pitch, roll);
    return LaneletInstance::process(bbox, paramType, nPoints, pitch, roll);
  }
  bool default_process(const OrientedRect &bbox, const ParametrizationType &paramType, int32_t nPoints,
                       double pitch = 0, double roll = 0) {
    return this->LaneletInstance::process(bbox, paramType, nPoints, pitch, roll);
  }
};

template <typename T>
struct DictToMapConverter {
  DictToMapConverter() { converter::registry::push_back(&convertible, &construct, type_id<T>()); }
  static void *convertible(PyObject *obj) {
    if (!PyDict_CheckExact(obj)) {  // NOLINT
      return nullptr;
    }
    return obj;
  }
  static void construct(PyObject *obj, converter::rvalue_from_python_stage1_data *data) {
    dict d(borrowed(obj));
    list keys = d.keys();
    list values = d.values();
    T map;
    for (auto i = 0u; i < len(keys); ++i) {
      typename T::key_type key = extract<typename T::key_type>(keys[i]);
      typename T::mapped_type value = extract<typename T::mapped_type>(values[i]);
      map.insert(std::make_pair(key, value));
    }
    using StorageType = converter::rvalue_from_python_storage<T>;
    void *storage = reinterpret_cast<StorageType *>(data)->storage.bytes;  // NOLINT
    new (storage) T(map);
    data->convertible = storage;
  }
};

BOOST_PYTHON_MODULE(PYTHON_API_MODULE_NAME) {  // NOLINT

  Py_Initialize();
  np::initialize();

  enum_<LaneletRepresentationType>("LaneletRepresentationType")
      .value("Centerline", LaneletRepresentationType::Centerline)
      .value("Boundaries", LaneletRepresentationType::Boundaries);

  enum_<ParametrizationType>("ParametrizationType")
      .value("Bezier", ParametrizationType::Bezier)
      .value("BezierEndpointFixed", ParametrizationType::BezierEndpointFixed)
      .value("LineString", ParametrizationType::LineString);

  enum_<LineStringType>("LineStringType")
      .value("RoadBorder", LineStringType::RoadBorder)
      .value("Dashed", LineStringType::Dashed)
      .value("Solid", LineStringType::Solid)
      .value("CurbstoneHigh", LineStringType::CurbstoneHigh)
      .value("CurbstoneLow", LineStringType::CurbstoneLow)
      .value("Fence", LineStringType::Fence)
      .value("Building", LineStringType::Building)
      .value("Wall", LineStringType::Wall)
      .value("SolidSolid", LineStringType::SolidSolid)
      .value("SolidDashed", LineStringType::SolidDashed)
      .value("DashedSolid", LineStringType::DashedSolid)
      .value("Virtual", LineStringType::Virtual)
      .value("Centerline", LineStringType::Centerline)
      .value("BikeCenterline", LineStringType::BikeCenterline)
      .value("Unknown", LineStringType::Unknown)
      .value("DrivableArea", LineStringType::DrivableArea)
      .value("Divider", LineStringType::Divider)
      .value("BikeMarkingDashed", LineStringType::BikeMarkingDashed)
      .value("BikeMarkingSolid", LineStringType::BikeMarkingSolid)
      .value("GuardRail", LineStringType::GuardRail)
      .value("PedestrianCrossing", LineStringType::PedestrianCrossing)
      .value("ZebraCrossing", LineStringType::ZebraCrossing);

  enum_<TEType>("TEType")
      .value("TLCar", TEType::TLCar)
      .value("TLBike", TEType::TLBike)
      .value("TLPedestrian", TEType::TLPedestrian)
      .value("TLMisc", TEType::TLMisc)
      .value("TSMisc", TEType::TSMisc)
      .value("TSNoEntry", TEType::TSNoEntry)
      .value("TSTurnRight", TEType::TSTurnRight)
      .value("TSTurnLeft", TEType::TSTurnLeft)
      .value("TSTurnLeftOrRight", TEType::TSTurnLeftOrRight)
      .value("TSGoStraight", TEType::TSGoStraight)
      .value("TSGoStraightOrRight", TEType::TSGoStraightOrRight)
      .value("TSGoStraightOrLeft", TEType::TSGoStraightOrLeft)
      .value("TSPassRight", TEType::TSPassRight)
      .value("TSPassLeft", TEType::TSPassLeft)
      .value("TSOneWayStreet", TEType::TSOneWayStreet)
      .value("TSYield", TEType::TSYield)
      .value("TSRightOfWay", TEType::TSRightOfWay)
      .value("TSPriorityRoad", TEType::TSPriorityRoad)
      .value("TSStop", TEType::TSStop)
      .value("TSCrossbuck", TEType::TSCrossbuck)
      .value("TSRoundabout", TEType::TSRoundabout)
      .value("TSSpeedLimit", TEType::TSSpeedLimit)
      .value("TSPedestrianCrossing", TEType::TSPedestrianCrossing)
      .value("ArrowTurnRight", TEType::ArrowTurnRight)
      .value("ArrowTurnLeft", TEType::ArrowTurnLeft)
      .value("ArrowTurnLeftOrRight", TEType::ArrowTurnLeftOrRight)
      .value("ArrowGoStraight", TEType::ArrowGoStraight)
      .value("ArrowGoStraightOrRight", TEType::ArrowGoStraightOrRight)
      .value("ArrowGoStraightOrLeft", TEType::ArrowGoStraightOrLeft)
      .value("BikeSymbol", TEType::BikeSymbol)
      .value("BusSymbol", TEType::BusSymbol)
      .value("Symbol30", TEType::Symbol30)
      .value("Symbol50", TEType::Symbol50)
      .value("Symbol70", TEType::Symbol70)
      .value("StopLine", TEType::StopLine)
      .value("Unknown", TEType::Unknown);

  // LineStringTypeGrouping is a vector of pairs mapping LineStringType lists to representative LineStringType
  typedef std::pair<std::vector<LineStringType>, LineStringType> LineStringTypeGroupingPair;
  class_<LineStringTypeGroupingPair>("LineStringTypeGroupingPair",
                                     "Pair of LineStringType list and representative LineStringType")
      .def_readwrite("types", &LineStringTypeGroupingPair::first)
      .def_readwrite("representative", &LineStringTypeGroupingPair::second);

  typedef std::vector<std::pair<std::vector<LineStringType>, LineStringType>> LineStringTypeGrouping;
  class_<LineStringTypeGrouping>(
      "LineStringTypeGrouping",
      "Type grouping for compound instances: maps LineStringType lists to representative types. A grouping "
      "decides both which boundaries are chained into one compound instance - consecutive boundaries are chained "
      "as long as they stay within the same group - and the label of the result. It must assign a group to every "
      "type a lanelet boundary can have")
      .def(vector_indexing_suite<LineStringTypeGrouping>());

  def("getDefaultLineStringTypeGrouping", &getDefaultLineStringTypeGrouping,
      "Get the default LineStringTypeGrouping where each type has its own group");
  def("getRoadBorderMergedGrouping", &getRoadBorderMergedGrouping,
      "Get a RoadBorderMerged grouping: merges road border with fence, curbstone high and curbstone low");
  def("getMapTRDefaultSimpleGrouping", &getMapTRDefaultSimpleGrouping,
      "Get MapTR default simple grouping: RoadBorderMerged grouping with all lane dividers merged as well (including "
      "dashed and solid, excluding Virtual)");
  def("getM3TRDefaultGrouping", &getM3TRDefaultGrouping,
      "Get M3TR default grouping: RoadBorderMerged grouping, merges Solid, SolidSolid, SolidDashed, and DashedSolid "
      "LineStringTypes");

  // TETypeGrouping is a vector of pairs mapping TEType lists to representative TEType
  typedef std::pair<std::vector<TEType>, TEType> TETypeGroupingPair;
  class_<TETypeGroupingPair>("TETypeGroupingPair", "Pair of TEType list and representative TEType")
      .def_readwrite("types", &TETypeGroupingPair::first)
      .def_readwrite("representative", &TETypeGroupingPair::second);

  typedef std::vector<std::pair<std::vector<TEType>, TEType>> TETypeGrouping;
  class_<TETypeGrouping>("TETypeGrouping",
                         "Type grouping for traffic elements: maps TEType lists to representative types")
      .def(vector_indexing_suite<TETypeGrouping>());

  def("getDefaultTETypeGrouping", &getDefaultTETypeGrouping,
      "Get the default TETypeGrouping where each type has its own group");
  def("getTETypeRepresentative", &getTETypeRepresentative, (arg("type"), arg("grouping")),
      "Get the representative type for a TEType from the grouping");
  def("getTETypeGroupIndex", &getTETypeGroupIndex, (arg("type"), arg("grouping")), "Get the group index for a TEType");
  def("areTETypesSameGroup", &areTETypesSameGroup, (arg("type1"), arg("type2"), arg("grouping")),
      "Check if two TE types are in the same group");
  def("getLineStringTypeRepresentative", &getLineStringTypeRepresentative, (arg("type"), arg("grouping")),
      "Get the representative type for a LineStringType from the grouping");
  def("getLineStringTypeGroupIndex", &getLineStringTypeGroupIndex, (arg("type"), arg("grouping")),
      "Get the group index for a LineStringType");
  def("areLineStringTypesSameGroup", &areLineStringTypesSameGroup, (arg("type1"), arg("type2"), arg("grouping")),
      "Check if two LineString types are in the same group");

  class_<OrientedRect>("OrientedRect",
                       "The local reference frame: an axis aligned rectangle rotated by a yaw angle. Everything "
                       "outside of it is cut away when instances are processed. Use getRotatedRect to create one",
                       no_init)
      .add_property("bounds", make_function(&OrientedRect::bounds_const, return_value_policy<copy_const_reference>()),
                    "The corner points of the rectangle in the map frame");

  def("getRotatedRect", &getRotatedRect,
      (arg("center"), arg("extentLongitudinal"), arg("extentLateral"), arg("yaw"), arg("from2dPos")),
      "Build the local reference frame around a pose. The extents are half extents [m], i.e. the rectangle "
      "reaches that far to the front/rear and to the left/right of the center");
  def("extractSubmap", &extractSubmap,
      (arg("laneletMap"), arg("center"), arg("extentLongitudinal"), arg("extentLateral")),
      "Extract the part of the map around a position, including line strings, polygons and regulatory elements");
  def("bdTypeToEnum", static_cast<LineStringType (*)(const ConstLineString3d &)>(&bdTypeToEnum), (arg("lstring")),
      "Get the LineStringType of a lanelet boundary from its tagging, Unknown if no tag matches");
  def("bdTypeToEnumPolygon", &bdTypeToEnumPolygon, (arg("polygon")),
      "Get the LineStringType of a polygon from its tagging, Unknown if no tag matches");
  def("teTypeToEnum", static_cast<TEType (*)(const ConstLineString3d &)>(&teTypeToEnum), (arg("te")),
      "Get the TEType of a traffic element from its tagging, Unknown if no tag matches");
  def("teTypeToEnumPolygon", &teTypeToEnumPolygon, (arg("te")),
      "Get the TEType of a polygon traffic element from its tagging, Unknown if no tag matches");
  def("resampleLineString", &resampleLineString, (arg("polyline"), arg("nPoints")),
      "Resample a polyline to nPoints equidistant points, keeping its first and last point. Returns an empty "
      "polyline if it is shorter than 0.1 m");
  def("cutLineString", &cutLineString, (arg("bbox"), arg("polyline")),
      "Clip a polyline to the local reference frame. Returns one polyline per part that is inside of it");
  def("transformLineString", &transformLineString, (arg("bbox"), arg("polyline"), arg("pitch"), arg("roll")),
      "Transform a polyline from the map frame into the local reference frame given by bbox");
  def("saveMapData", &saveMapData, (arg("filename"), arg("mDataVec"), arg("binary")),
      "Save all given MapData objects into one file, as a binary or a human readable XML archive");
  def("loadMapData", &loadMapData, (arg("filename"), arg("binary")),
      "Load the MapData objects that saveMapData wrote into one file");
  def("saveMapDataMultiFile", &saveMapDataMultiFile, (arg("path"), arg("filenames"), arg("mDataVec"), arg("binary")),
      "Save the given MapData objects into one file each, path is prepended to the file names as it is");
  def("loadMapDataMultiFile", &loadMapDataMultiFile, (arg("path"), arg("filenames"), arg("binary")),
      "Load the MapData objects that saveMapDataMultiFile wrote into one file each");

  class_<MapInstanceWrap, boost::noncopyable>(
      "MapInstance",
      "Abstract base class of all local instance labels. An instance is created from one map element and is "
      "brought into the local reference frame by process(), which cuts, transforms and resamples its geometry",
      no_init)
      .add_property("wasCut", &MapInstance::wasCut,
                    "Whether processing removed geometry, i.e. whether this instance reaches beyond the bounding box")
      .add_property("mapID", &MapInstance::mapID, "Id of the map element this instance was created from")
      .add_property("initialized", &MapInstance::initialized, "Whether this instance is linked to a map element")
      .add_property("valid", &MapInstance::valid,
                    "Whether this instance still carries usable geometry after processing. False means the local "
                    "bounding box does not contain (enough of) it")
      .def("computeInstanceVectors", pure_virtual(&MapInstance::computeInstanceVectors),
           (arg("onlyPoints"), arg("pointsIn2d")),
           "Get the instance as flat vectors of the form [x, y, (z)] * n (+ type as last element unless onlyPoints). "
           "One vector per line string piece the instance consists of after processing")
      .def("process", pure_virtual(&MapInstance::process),
           (arg("bbox"), arg("paramType"), arg("nPoints"), arg("pitch") = 0, arg("roll") = 0),
           "Cut, transform and resample this instance into the local reference frame of bbox. Values of nPoints "
           "below 2 disable resampling. Returns whether the instance survives the cut");

  class_<LineStringInstanceWrap, bases<MapInstance>, boost::noncopyable>(
      "LineStringInstance",
      "Abstract instance whose geometry is a line string. Keeps the result of every processing stage as a list, "
      "since clipping can split one line into several pieces. The lists are index aligned per surviving piece",
      no_init)
      .add_property("rawInstance",
                    make_function(&LineStringInstance::rawInstance, return_value_policy<copy_const_reference>()),
                    "The unprocessed geometry in the map frame, as it was taken from the map element")
      .add_property("cutInstance",
                    make_function(&LineStringInstance::cutInstance, return_value_policy<copy_const_reference>()),
                    "Geometry clipped to the bounding box, still in the map frame")
      .add_property("cutAndTransformedInstance",
                    make_function(&LineStringInstance::cutAndTransformedInstance,
                                  return_value_policy<copy_const_reference>()),
                    "Geometry clipped to the bounding box and transformed into its local frame")
      .add_property("cutTransformedAndResampledInstance",
                    make_function(&LineStringInstance::cutTransformedAndResampledInstance,
                                  return_value_policy<copy_const_reference>()),
                    "Geometry clipped, transformed and resampled. Empty if resampling was disabled")
      .def("computeInstanceVectors", pure_virtual(&LineStringInstance::computeInstanceVectors),
           (arg("onlyPoints"), arg("pointsIn2d")), "See MapInstance.computeInstanceVectors")
      .def("process", pure_virtual(&LineStringInstance::process),
           (arg("bbox"), arg("paramType"), arg("nPoints"), arg("pitch") = 0, arg("roll") = 0), "See MapInstance.process")
      .def("pointMatrices", pure_virtual(&LineStringInstance::pointMatrices), (arg("pointsIn2d")),
           "Get the points of the processed instance as numpy arrays of shape (nPoints, 2 or 3), one per piece");

  class_<LaneLineStringInstanceWrap, bases<LineStringInstance>, LaneLineStringInstancePtr, boost::noncopyable>(
      "LaneLineStringInstance",
      "A line string that is part of a lane, e.g. a lane divider, a road border or a centerline. Its geometry "
      "always runs in the driving direction of the lanelets using it, see the inverted property",
      init<BasicLineString3d, Id, LineStringType, Ids, bool>())
      .def(init<>())
      .add_property("type", &LaneLineStringInstance::type, "Type of this line string")
      .add_property("inverted", &LaneLineStringInstance::inverted,
                    "Whether the geometry runs against the direction of the map element with mapID")
      .add_property("typeInt", &LaneLineStringInstance::typeInt,
                    "The type as int, this is what is appended to the instance vectors")
      .add_property("laneletIDs",
                    make_function(&LaneLineStringInstance::laneletIDs, return_value_policy<copy_const_reference>()),
                    "Ids of all lanelets that use this line string, e.g. both neighbours of a shared lane divider")
      .def("addLaneletID", &LaneLineStringInstance::addLaneletID, (arg("id")),
           "Register another lanelet as a user of this line string")
      .def("computeInstanceVectors", &LaneLineStringInstance::computeInstanceVectors,
           &LaneLineStringInstanceWrap::default_computeInstanceVectors, (arg("onlyPoints"), arg("pointsIn2d")),
           "See MapInstance.computeInstanceVectors. Empty before process was called")
      .def("process", &LaneLineStringInstance::process, &LaneLineStringInstanceWrap::default_process,
           (arg("bbox"), arg("paramType"), arg("nPoints"), arg("pitch") = 0, arg("roll") = 0), "See MapInstance.process")
      .def("pointMatrices", &LaneLineStringInstance::pointMatrices, &LaneLineStringInstanceWrap::default_pointMatrices,
           (arg("pointsIn2d")), "See LineStringInstance.pointMatrices. Empty before process was called");

  class_<TEInstanceWrap, bases<LineStringInstance>, TEInstancePtr, boost::noncopyable>(
      "TEInstance",
      "A traffic element instance, e.g. a stop line, a road marking, a traffic light or a traffic sign. Traffic "
      "elements are not part of a lane, they are collected from tagged line strings and polygons of the map",
      init<BasicLineString3d, Id, TEType>())
      .def(init<>())
      .add_property("teType", &TEInstance::teType, "Type of this traffic element")
      .def("computeInstanceVectors", &TEInstance::computeInstanceVectors,
           &TEInstanceWrap::default_computeInstanceVectors, (arg("onlyPoints"), arg("pointsIn2d")),
           "See MapInstance.computeInstanceVectors. Empty before process was called")
      .def("process", &TEInstance::process, &TEInstanceWrap::default_process,
           (arg("bbox"), arg("paramType"), arg("nPoints"), arg("pitch") = 0, arg("roll") = 0), "See MapInstance.process")
      .def("pointMatrices", &TEInstance::pointMatrices, &TEInstanceWrap::default_pointMatrices, (arg("pointsIn2d")),
           "See LineStringInstance.pointMatrices. Empty before process was called");

  class_<LaneletInstanceWrap, bases<MapInstance>, LaneletInstancePtr, boost::noncopyable>(
      "LaneletInstance",
      "One lanelet, made up of the instances of its left boundary, right boundary and centerline. Only lanelets "
      "that can be driven on become lanelet instances",
      init<LaneLineStringInstancePtr, LaneLineStringInstancePtr, LaneLineStringInstancePtr, Id>())
      .def(init<ConstLanelet>())
      .def(init<>())
      .add_property("leftBoundary", make_function(&LaneletInstance::leftBoundary), "Instance of the left boundary")
      .add_property("rightBoundary", make_function(&LaneletInstance::rightBoundary), "Instance of the right boundary")
      .add_property("centerline", make_function(&LaneletInstance::centerline), "Instance of the centerline")
      .def("setReprType", &LaneletInstance::setReprType, (arg("reprType")),
           "Set how computeInstanceVectors represents this lanelet, defaults to LaneletRepresentationType.Centerline")
      .def("computeInstanceVectors", &LaneletInstance::computeInstanceVectors,
           &LaneletInstanceWrap::default_computeInstanceVectors, (arg("onlyPoints"), arg("pointsIn2d")),
           "Get the instance vectors. With representation type Centerline: centerline points, then the type of the "
           "left and the right boundary. With Boundaries: left points, right points, then both types")
      .def("process", &LaneletInstance::process, &LaneletInstanceWrap::default_process,
           (arg("bbox"), arg("paramType"), arg("nPoints"), arg("pitch") = 0, arg("roll") = 0),
           "Processes the boundaries and the centerline. Invalid if any of the three is invalid");

  class_<CompoundLaneLineStringInstanceWrap, bases<LaneLineStringInstance>, CompoundLaneLineStringInstancePtr,
         boost::noncopyable>(
      "CompoundLaneLineStringInstance",
      "Several line string instances chained into one, e.g. all lane dividers along one lane. Its type is the "
      "representative type of its group, not the type of its first member. The individual instances stay "
      "accessible through the features property, so it can be traced back to the map elements it was built from",
      init<LaneLineStringInstanceList, LineStringType>())
      .def(init<>())
      .add_property("features", make_function(&CompoundLaneLineStringInstance::features),
                    "The individual instances this was built from, in the order they are chained")
      .add_property("pathLengthsRaw",
                    make_function(&CompoundLaneLineStringInstance::pathLengthsRaw,
                                  return_value_policy<copy_const_reference>()),
                    "Cumulative length [m] of the raw geometry up to and including each member of features")
      .add_property("pathLengthsProcessed",
                    make_function(&CompoundLaneLineStringInstance::pathLengthsProcessed,
                                  return_value_policy<copy_const_reference>()),
                    "Cumulative length [m] of the cut geometry up to and including each member of features")
      .add_property("processedInstancesValid",
                    make_function(&CompoundLaneLineStringInstance::processedInstancesValid,
                                  return_value_policy<copy_const_reference>()),
                    "Per member of features: whether enough of it survived the cut")
      .def("computeInstanceVectors", &CompoundLaneLineStringInstance::computeInstanceVectors,
           &CompoundLaneLineStringInstanceWrap::default_computeInstanceVectors, (arg("onlyPoints"), arg("pointsIn2d")),
           "See MapInstance.computeInstanceVectors. Empty before process was called")
      .def("process", &CompoundLaneLineStringInstance::process, &CompoundLaneLineStringInstanceWrap::default_process,
           (arg("bbox"), arg("paramType"), arg("nPoints"), arg("pitch") = 0, arg("roll") = 0),
           "Processes the chained geometry as well as every individual instance. Valid if at least one member survives")
      .def("pointMatrices", &CompoundLaneLineStringInstance::pointMatrices,
           &CompoundLaneLineStringInstanceWrap::default_pointMatrices, (arg("pointsIn2d")),
           "See LineStringInstance.pointMatrices. Empty before process was called");

  class_<Edge>("Edge", "One edge of the lane graph or the traffic element graph, given by the ids it connects",
               init<Id, Id, bool>())
      .def(init<>())
      .def_readwrite("el1", &Edge::el1_, "Id of the element the edge starts at")
      .def_readwrite("el2", &Edge::el2_, "Id of the element the edge leads to")
      .def_readwrite("isLaneChange", &Edge::isLaneChange_,
                     "True for a lateral (lane change) edge, False for a longitudinal (successor) edge");

  {
    scope inMapData =
        class_<MapData, MapDataPtr>(
            "MapData",
            "All local instance labels of one local reference frame pose. Created in two phases: build extracts "
            "the instances from a local submap, processAll brings them into the local reference frame. Only after "
            "the second phase are the accessors and getTensorInstanceData meaningful")
            .def(init<>())
            .def("build", &MapData::build,
                 (arg("localSubmap"), arg("localSubmapGraph"), arg("trafficRules"),
                  arg("bikeSubmapGraph") = routing::RoutingGraphConstPtr(), arg("ignoreMapElevation") = false,
                  arg("lineStringTypeGrouping") = getDefaultLineStringTypeGrouping(),
                  arg("teTypeGrouping") = getDefaultTETypeGrouping()),
                 "Extract all instances of a local submap. Throws if lineStringTypeGrouping does not cover every "
                 "type a lanelet boundary can have")
            .staticmethod("build")
            .def("processAll", &MapData::processAll,
                 (arg("bbox"), arg("paramType"), arg("resampleLanes") = true, arg("nPointsLanes") = 0,
                  arg("resampleTE") = true, arg("nPointsTE") = 0, arg("pitch") = 0, arg("roll") = 0),
                 "Process all instances into the local reference frame given by bbox. Values of nPointsLanes or "
                 "nPointsTE below 2 disable resampling as well. Returns whether every instance is still valid "
                 "afterwards - False just means that some are outside of the bounding box")
            .def("lineStringsOfType", &MapData::lineStringsOfType, (arg("type")),
                 "All lane line strings of the given type, by map id")
            .def("validLineStringsOfType", &MapData::validLineStringsOfType, (arg("type")),
                 "Like lineStringsOfType, but without the instances that did not survive processing")
            .def("compoundLineStringsOfType", &MapData::compoundLineStringsOfType, (arg("type")),
                 "All compound line strings of the given type")
            .def("validCompoundLineStringsOfType", &MapData::validCompoundLineStringsOfType, (arg("type")),
                 "Like compoundLineStringsOfType, but without the instances that did not survive processing")
            .def("associatedCpdLineStringsOfType", &MapData::associatedCpdLineStringsOfType,
                 (arg("mapId"), arg("type")),
                 "All compound instances of the given type that a map element is part of. mapId is the id of a "
                 "member line string or of a lanelet using it, for compound centerlines it is the lanelet id")
            .def("teInstancesOfType", &MapData::teInstancesOfType, (arg("type")),
                 "All traffic elements of the given type, by map id")
            .def("validTEInstancesOfType", &MapData::validTEInstancesOfType, (arg("type")),
                 "Like teInstancesOfType, but without the instances that did not survive processing")
            .add_property("laneletInstances",
                          make_function(&MapData::laneletInstances, return_value_policy<copy_const_reference>()),
                          "All lanelet instances by map id. Non-driving lanelets are not part of this")
            .add_property("llEdges", make_function(&MapData::llEdges, return_value_policy<copy_const_reference>()),
                          "Lane graph edges by source lanelet id: successors and lane changes between lanelets")
            .add_property("teEdges", make_function(&MapData::teEdges, return_value_policy<copy_const_reference>()),
                          "Traffic element edges by source element id, targets are lanelets or traffic elements")
            .add_property("teToCenterlineEdges",
                          make_function(&MapData::teToCenterlineEdges, return_value_policy<copy_const_reference>()),
                          "teEdges resolved to instance pairs, for the edges that lead to a lanelet")
            .add_property("teToTEEdges",
                          make_function(&MapData::teToTEEdges, return_value_policy<copy_const_reference>()),
                          "teEdges resolved to instance pairs, for the edges that lead to another traffic element")
            .add_property("uuid", make_function(&MapData::uuid, return_value_policy<copy_const_reference>()),
                          "Randomly generated id of this sample, kept across serialization")
            .def("getTensorInstanceData", &MapData::getTensorInstanceData, (arg("pointsIn2d"), arg("ignoreBuffer")),
                 "Get all instance labels as numpy arrays. The result is buffered, so if the underlying instances "
                 "change you need to set ignoreBuffer. Changing pointsIn2d alone does not invalidate the buffer");

    class_<MapData::TensorInstanceData>(
        "TensorInstanceData",
        "All instance labels of a MapData object as numpy arrays. Instances are grouped by type and only valid "
        "ones are included. Within one type, an instance is identified by its position in the returned list - "
        "that is the index the edge lists refer to. Every access copies, so modifying a returned array does not "
        "modify this object",
        init<>())
        .def("lineStringsOfType", &MapData::TensorInstanceData::lineStringsOfType, (arg("type")),
             "Point matrices of all valid lane line strings of the given type")
        .def("compoundLineStringsOfType", &MapData::TensorInstanceData::compoundLineStringsOfType, (arg("type")),
             "Point matrices of all valid compound line strings of the given type")
        .def("teInstancesOfType", &MapData::TensorInstanceData::teInstancesOfType, (arg("type")),
             "Point matrices of all valid traffic elements of the given type")
        .def("pointMatrixCpdLineStrings", &MapData::TensorInstanceData::pointMatrixCpdLineStrings,
             (arg("type"), arg("index")),
             "Get the instance a point matrix of compoundLineStringsOfType came from, which gives access to the "
             "map elements it was built from. Throws if the type or the index does not exist")
        .add_property("teToCenterlineIndexEdges",
                      make_function(&MapData::TensorInstanceData::teToCenterlineIndexEdges,
                                    return_value_policy<copy_const_reference>()),
                      "Traffic element to centerline edges as (source type, source index, target index) tuples, "
                      "with indices local to their type")
        .add_property("teToTEIndexEdges",
                      make_function(&MapData::TensorInstanceData::teToTEIndexEdges,
                                    return_value_policy<copy_const_reference>()),
                      "Traffic element to traffic element edges as (source type, source index, target type, "
                      "target index) tuples, with indices local to their type")
        .add_property("uuid",
                      make_function(&MapData::TensorInstanceData::uuid, return_value_policy<copy_const_reference>()),
                      "Id of the sample, identical to the uuid of the MapData object this came from");
  }

  {
    void (MapDataInterface::*setCurrPosAndExtractSubmap2d)(const lanelet::BasicPoint2d &, double) =
        &MapDataInterface::setCurrPosAndExtractSubmap2d;
    void (MapDataInterface::*setCurrPosAndExtractSubmap4d)(const lanelet::BasicPoint3d &, double) =
        &MapDataInterface::setCurrPosAndExtractSubmap;
    void (MapDataInterface::*setCurrPosAndExtractSubmap6d)(const lanelet::BasicPoint3d &, double, double, double) =
        &MapDataInterface::setCurrPosAndExtractSubmap;

    std::vector<MapDataPtr> (MapDataInterface::*mapDataBatch2d)(std::vector<BasicPoint2d>, std::vector<double>) =
        &MapDataInterface::mapDataBatch2d;
    std::vector<MapDataPtr> (MapDataInterface::*mapDataBatch4d)(std::vector<BasicPoint3d>, std::vector<double>) =
        &MapDataInterface::mapDataBatch;
    std::vector<MapDataPtr> (MapDataInterface::*mapDataBatch6d)(std::vector<BasicPoint3d>, std::vector<double>,
                                                                std::vector<double>, std::vector<double>) =
        &MapDataInterface::mapDataBatch;
    scope inMapDataInterface =
        class_<MapDataInterface>(
            "MapDataInterface",
            "Main interface class of the module: turns a Lanelet2 map into local instance labels. Set a local "
            "reference frame pose with setCurrPosAndExtractSubmap, then get the labels for it with mapData. For "
            "more than one pose, prefer the mapDataBatch methods - they are self contained and do not touch the "
            "current position. The traffic rules are currently fixed to Germany for vehicles and bicycles",
            init<LaneletMapConstPtr>())
            .def(init<LaneletMapConstPtr, MapDataInterface::Configuration>())
            .add_property("config",
                          make_function(&MapDataInterface::config, return_value_policy<copy_const_reference>()),
                          "The configuration this interface was created with")
            .def("setCurrPosAndExtractSubmap2d", setCurrPosAndExtractSubmap2d, (arg("pt"), arg("yaw")),
                 "Set the local reference frame from a 2d pose [m, rad] and extract the submap around it. Since "
                 "there is no elevation to relate to, all instance output of this frame will be 2d")
            .def("setCurrPosAndExtractSubmap", setCurrPosAndExtractSubmap4d, (arg("pt"), arg("yaw")),
                 "Set the local reference frame from a 3d position and a yaw angle [m, rad], with pitch and roll "
                 "assumed to be 0, and extract the submap around it")
            .def("setCurrPosAndExtractSubmap", setCurrPosAndExtractSubmap6d,
                 (arg("pt"), arg("yaw"), arg("pitch"), arg("roll")),
                 "Set the local reference frame from a full 3d pose [m, rad] and extract the submap around it")
            .def("mapData", &MapDataInterface::mapData, (arg("processAll")),
                 "Get the instance labels of the current local reference frame. If processAll is False you get "
                 "the raw extracted data and have to call MapData.processAll yourself. Throws if no position was "
                 "set with setCurrPosAndExtractSubmap before")
            .def("mapDataBatch2d", mapDataBatch2d, (arg("pts"), arg("yaws")),
                 "Get the processed instance labels of several 2d poses at once")
            .def("mapDataBatch", mapDataBatch4d, (arg("pts"), arg("yaws")),
                 "Get the processed instance labels of several 3d positions with yaw angles at once, with pitch "
                 "and roll assumed to be 0")
            .def("mapDataBatch", mapDataBatch6d, (arg("pts"), arg("yaws"), arg("pitches"), arg("rolls")),
                 "Get the processed instance labels of several full 3d poses at once");

    class_<MapDataInterface::Configuration>(
        "Configuration", "Parameters of the extraction and processing done by MapDataInterface", init<>())
        .def(init<LaneletRepresentationType, ParametrizationType, double, double, int, int>())
        .def_readwrite("reprType", &MapDataInterface::Configuration::reprType,
                       "How lanelet instances are laid out in their instance vectors")
        .def_readwrite("paramType", &MapDataInterface::Configuration::paramType,
                       "Parametrization of the instance geometry, only ParametrizationType.LineString is implemented")
        .def_readwrite("submapExtentLongitudinal", &MapDataInterface::Configuration::submapExtentLongitudinal,
                       "Half extent [m] of the local reference frame in driving direction, i.e. this much to the "
                       "front and to the rear")
        .def_readwrite("submapExtentLateral", &MapDataInterface::Configuration::submapExtentLateral,
                       "Half extent [m] of the local reference frame in lateral direction, i.e. this much to the "
                       "left and to the right")
        .def_readwrite("ignoreMapElevation", &MapDataInterface::Configuration::ignoreMapElevation,
                       "If True, elevation (z coordinate) in map elements is ignored and set to 0")
        .def_readwrite("allowLaneChangeAcrossBicycleLane",
                       &MapDataInterface::Configuration::allowLaneChangeAcrossBicycleLane,
                       "If True, two vehicle lanes with a bicycle lane between them get a lane change edge when both "
                       "bike lane bounds may be crossed (e.g. dashed bike markings). Default False")
        .def_readwrite("resampleLanes", &MapDataInterface::Configuration::resampleLanes,
                       "If True, lane instances are resampled; if False, no fixed resampling")
        .def_readwrite("nPointsLanes", &MapDataInterface::Configuration::nPointsLanes,
                       "Number of points for lane resampling, values below 2 disable resampling as well")
        .def_readwrite("resampleTE", &MapDataInterface::Configuration::resampleTE,
                       "If True, traffic element instances are resampled; if False, no fixed resampling")
        .def_readwrite("nPointsTE", &MapDataInterface::Configuration::nPointsTE,
                       "Number of points for traffic element resampling, values below 2 disable resampling as well")
        .def_readwrite("lineStringTypeGrouping", &MapDataInterface::Configuration::lineStringTypeGrouping,
                       "Grouping of types for compound instance generation, see LineStringTypeGrouping")
        .def_readwrite("teTypeGrouping", &MapDataInterface::Configuration::teTypeGrouping,
                       "Grouping of types for traffic element instance generation, see TETypeGrouping");
  }

  // Eigen, stl etc. converters
  converters::convertMatrix<MatrixXd>(true);
  converters::convertVector<VectorXd>(true);

  converters::VectorToListConverter<std::vector<MatrixXd>>();
  converters::VectorToListConverter<std::vector<VectorXd>>();
  converters::VectorToListConverter<BasicLineStrings3d>();
  converters::VectorToListConverter<LaneLineStringInstanceList>();
  converters::VectorToListConverter<CompoundLaneLineStringInstanceList>();
  converters::VectorToListConverter<
      boost::geometry::model::ring<BasicPoint2d, true, true, std::vector, std::allocator>>();
  converters::VectorToListConverter<std::vector<double>>();
  converters::VectorToListConverter<std::vector<int>>();
  converters::VectorToListConverter<std::vector<MapDataPtr>>();
  converters::VectorToListConverter<std::vector<Edge>>();
  converters::VectorToListConverter<TEToCenterlineEdges>();
  converters::VectorToListConverter<TEToTEEdges>();
  converters::VectorToListConverter<TEToCenterlineIndexEdges>();
  converters::VectorToListConverter<TEToTEIndexEdges>();
  converters::IterableConverter()
      .fromPython<std::vector<MatrixXd>>()
      .fromPython<BasicLineString3d>()
      .fromPython<BasicLineStrings3d>()
      .fromPython<std::vector<BasicPoint2d>>()
      .fromPython<std::vector<double>>()
      .fromPython<std::vector<std::string>>()
      .fromPython<std::vector<MapDataPtr>>()
      .fromPython<LaneLineStringInstanceList>()
      .fromPython<CompoundLaneLineStringInstanceList>();
  converters::MapToDictConverter<LaneLineStringInstances>();
  converters::MapToDictConverter<LaneletInstances>();
  converters::MapToDictConverter<Edges>();
  converters::MapToDictConverter<TEInstances>();
  DictToMapConverter<LaneLineStringInstances>();
  DictToMapConverter<LaneletInstances>();
  DictToMapConverter<TEInstances>();
  converters::PairConverter<std::pair<TEInstancePtr, CompoundLaneLineStringInstancePtr>>();
  converters::PairConverter<std::pair<TEInstancePtr, TEInstancePtr>>();
  converters::PyTuple<TEType, size_t, size_t>();
  converters::PyTuple<TEType, size_t, TEType, size_t>();
  class_<std::vector<bool>>("BoolList").def(vector_indexing_suite<std::vector<bool>>());

  def("toPointMatrix", &toPointMatrix, (arg("lString"), arg("pointsIn2d")),
      "Convert a line string to a numpy array of shape (nPoints, 2 or 3)");

  implicitly_convertible<routing::RoutingGraphPtr, routing::RoutingGraphConstPtr>();
}

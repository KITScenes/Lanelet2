from typing import Dict, List, Tuple, overload
import numpy
from numpy.typing import NDArray
import lanelet2.core
import lanelet2.routing
import lanelet2.traffic_rules


class LaneletRepresentationType:
    """
    Enum for how a LaneletInstance is laid out in its instance vector.
    """
    Centerline: int
    Boundaries: int


class ParametrizationType:
    """
    Enum for how the geometry of an instance is parametrized.

    Only LineString is implemented, the others raise when passed to process().
    """
    Bezier: int
    BezierEndpointFixed: int
    LineString: int


class LineStringType:
    """
    Enum for the type of a lane line string instance.

    Most values are derived from the tagging of a lanelet boundary, see bdTypeToEnum. The following ones are
    not:

    - Centerline and BikeCenterline: computed centerlines of lanelets and compound lanelet paths
    - DrivableArea: the extent of the physically drivable space, from line strings tagged
      drivable_space_border, which are not lanelet boundaries and are stated independently of how the lane
      boundaries are tagged
    - Divider: only ever appears as the representative type of a grouping that merges lane markings
    """
    RoadBorder: int
    Dashed: int
    Solid: int
    CurbstoneHigh: int
    CurbstoneLow: int
    Fence: int
    Building: int
    Wall: int
    SolidSolid: int
    SolidDashed: int
    DashedSolid: int
    Virtual: int
    Centerline: int
    BikeCenterline: int
    Unknown: int
    DrivableArea: int
    Divider: int
    BikeMarkingDashed: int
    BikeMarkingSolid: int
    GuardRail: int
    PedestrianCrossing: int
    ZebraCrossing: int


class TEType:
    """
    Enum for the type of a traffic element instance.

    Prefixes: TL = traffic light, TS = traffic sign. The remaining values are road surface markings (arrows,
    symbols) and stop lines. Traffic signs are mapped from their German StVO codes, see teTypeToEnum.
    """
    TLCar: int
    TLBike: int
    TLPedestrian: int
    TLMisc: int
    TSMisc: int
    TSNoEntry: int
    TSTurnRight: int
    TSTurnLeft: int
    TSTurnLeftOrRight: int
    TSGoStraight: int
    TSGoStraightOrRight: int
    TSGoStraightOrLeft: int
    TSPassRight: int
    TSPassLeft: int
    TSOneWayStreet: int
    TSYield: int
    TSRightOfWay: int
    TSPriorityRoad: int
    TSStop: int
    TSCrossbuck: int
    TSRoundabout: int
    TSSpeedLimit: int
    TSPedestrianCrossing: int
    ArrowTurnRight: int
    ArrowTurnLeft: int
    ArrowTurnLeftOrRight: int
    ArrowGoStraight: int
    ArrowGoStraightOrRight: int
    ArrowGoStraightOrLeft: int
    BikeSymbol: int
    BusSymbol: int
    Symbol30: int
    Symbol50: int
    Symbol70: int
    StopLine: int
    Unknown: int


class LineStringTypeGroupingPair:
    """
    Pair of LineStringType list and representative LineStringType
    """
    types: List[LineStringType]
    representative: LineStringType


class LineStringTypeGrouping(List[LineStringTypeGroupingPair]):
    """
    Type grouping for compound instances: maps LineStringType lists to representative types.

    A grouping decides both which boundaries are chained into one compound instance - consecutive boundaries are
    chained as long as they stay within the same group - and the label of the result. It must assign a group to
    every type a lanelet boundary can have, otherwise unrelated types would end up chained together.
    """
    ...


class TETypeGroupingPair:
    """
    Pair of TEType list and representative TEType
    """
    types: List[TEType]
    representative: TEType


class TETypeGrouping(List[TETypeGroupingPair]):
    """
    Type grouping for traffic elements: maps TEType lists to representative types.

    Unlike a LineStringTypeGrouping this only relabels: traffic elements are never chained. Types that are in no
    group keep their own type.
    """
    ...


class BoolList(List[bool]):
    """
    List of booleans.
    """
    ...


def getDefaultLineStringTypeGrouping() -> LineStringTypeGrouping:
    """
    Get the default LineStringTypeGrouping where each type has its own group
    """
    ...


def getRoadBorderMergedGrouping() -> LineStringTypeGrouping:
    """
    Get a RoadBorderMerged grouping: merges road border with fence, curbstone high and curbstone low
    """
    ...


def getMapTRDefaultSimpleGrouping() -> LineStringTypeGrouping:
    """
    Get MapTR default simple grouping: RoadBorderMerged grouping with all lane dividers merged as well
    (including dashed and solid, excluding Virtual)
    """
    ...


def getM3TRDefaultGrouping() -> LineStringTypeGrouping:
    """
    Get M3TR default grouping: RoadBorderMerged grouping, merges Solid, SolidSolid, SolidDashed, and
    DashedSolid LineStringTypes
    """
    ...


def getDefaultTETypeGrouping() -> TETypeGrouping:
    """
    Get the default TETypeGrouping where each type has its own group
    """
    ...


def getLineStringTypeRepresentative(type: LineStringType, grouping: LineStringTypeGrouping) -> LineStringType:
    """
    Get the representative type for a LineStringType from the grouping, the input type if it is in no group
    """
    ...


def getLineStringTypeGroupIndex(type: LineStringType, grouping: LineStringTypeGrouping) -> int:
    """
    Get the group index for a LineStringType, -1 if it is in no group
    """
    ...


def areLineStringTypesSameGroup(type1: LineStringType, type2: LineStringType,
                                grouping: LineStringTypeGrouping) -> bool:
    """
    Check if two LineString types are in the same group
    """
    ...


def getTETypeRepresentative(type: TEType, grouping: TETypeGrouping) -> TEType:
    """
    Get the representative type for a TEType from the grouping, the input type if it is in no group
    """
    ...


def getTETypeGroupIndex(type: TEType, grouping: TETypeGrouping) -> int:
    """
    Get the group index for a TEType, -1 if it is in no group
    """
    ...


def areTETypesSameGroup(type1: TEType, type2: TEType, grouping: TETypeGrouping) -> bool:
    """
    Check if two TE types are in the same group
    """
    ...


class OrientedRect:
    """
    The local reference frame: an axis aligned rectangle rotated by a yaw angle.

    Everything outside of it is cut away when instances are processed. Use getRotatedRect to create one.
    """

    @property
    def bounds(self) -> List[lanelet2.core.BasicPoint2d]:
        """
        The corner points of the rectangle in the map frame
        """
        ...


def getRotatedRect(center: lanelet2.core.BasicPoint3d, extentLongitudinal: float, extentLateral: float,
                   yaw: float, from2dPos: bool) -> OrientedRect:
    """
    Build the local reference frame around a pose.

    The extents are half extents [m], i.e. the rectangle reaches that far to the front/rear and to the
    left/right of the center. from2dPos marks the pose as 2d, which forces all instance output of this frame
    to be 2d.
    """
    ...


def extractSubmap(laneletMap: lanelet2.core.LaneletMap, center: lanelet2.core.BasicPoint2d,
                  extentLongitudinal: float, extentLateral: float) -> lanelet2.core.LaneletSubmap:
    """
    Extract the part of the map around a position, including line strings, polygons and regulatory elements.

    The search region is axis aligned and slightly larger than the rectangle of getRotatedRect, so that it
    contains the rotated rectangle for any yaw angle. The final crop happens when instances are processed.
    """
    ...


def bdTypeToEnum(lstring: lanelet2.core.ConstLineString3d) -> LineStringType:
    """
    Get the LineStringType of a lanelet boundary from its tagging, Unknown if no tag matches.

    Road borders are only recognized if they are tagged as such - an untagged road border is
    indistinguishable from a lane divider and will be treated as one.
    """
    ...


def bdTypeToEnumPolygon(polygon: lanelet2.core.ConstPolygon3d) -> LineStringType:
    """
    Get the LineStringType of a polygon from its tagging, Unknown if no tag matches
    """
    ...


def teTypeToEnum(te: lanelet2.core.ConstLineString3d) -> TEType:
    """
    Get the TEType of a traffic element from its tagging, Unknown if no tag matches
    """
    ...


def teTypeToEnumPolygon(te: lanelet2.core.ConstPolygon3d) -> TEType:
    """
    Get the TEType of a polygon traffic element from its tagging, Unknown if no tag matches
    """
    ...


def resampleLineString(polyline: List[lanelet2.core.BasicPoint3d],
                       nPoints: int) -> List[lanelet2.core.BasicPoint3d]:
    """
    Resample a polyline to nPoints equidistant points, keeping its first and last point.

    Returns an empty polyline if it is shorter than 0.1 m. Raises if the polyline is empty or if nPoints is
    smaller than 2.
    """
    ...


def cutLineString(bbox: OrientedRect,
                  polyline: List[lanelet2.core.BasicPoint3d]) -> List[List[lanelet2.core.BasicPoint3d]]:
    """
    Clip a polyline to the local reference frame.

    Returns one polyline per part that is inside of it, empty if it lies completely outside. Clipping happens
    in 2d, the z coordinate of the clipped points is restored from the original polyline.
    """
    ...


def transformLineString(bbox: OrientedRect, polyline: List[lanelet2.core.BasicPoint3d], pitch: float,
                        roll: float) -> List[lanelet2.core.BasicPoint3d]:
    """
    Transform a polyline from the map frame into the local reference frame given by bbox
    """
    ...


def toPointMatrix(lString: List[lanelet2.core.BasicPoint3d], pointsIn2d: bool) -> NDArray[numpy.float64]:
    """
    Convert a line string to a numpy array of shape (nPoints, 2 or 3)
    """
    ...


class MapInstance:
    """
    Abstract base class of all local instance labels.

    An instance is created from one map element and is brought into the local reference frame by process(),
    which runs the following pipeline:

    1. cut: the raw geometry is clipped to the local bounding box
    2. transform: the clipped geometry is moved into the bounding box frame
    3. resample: the transformed geometry is resampled to a fixed number of points, unless resampling was
       disabled by passing nPoints < 2

    Because clipping can split one line into several disjoint pieces, every stage results in a list of line
    strings, not a single one.
    """

    @property
    def mapID(self) -> int:
        """
        Id of the map element this instance was created from
        """
        ...

    @property
    def initialized(self) -> bool:
        """
        Whether this instance is linked to a map element
        """
        ...

    @property
    def valid(self) -> bool:
        """
        Whether this instance still carries usable geometry after processing.

        False means the local bounding box does not contain (enough of) it.
        """
        ...

    @property
    def wasCut(self) -> bool:
        """
        Whether processing removed geometry, i.e. whether this instance reaches beyond the bounding box
        """
        ...

    def computeInstanceVectors(self, onlyPoints: bool, pointsIn2d: bool) -> List[NDArray[numpy.float64]]:
        """
        Get the instance as flat vectors of the form [x, y, (z)] * n (+ type as last element unless onlyPoints).

        One vector per line string piece the instance consists of after processing.
        """
        ...

    def process(self, bbox: OrientedRect, paramType: ParametrizationType, nPoints: int, pitch: float = 0,
                roll: float = 0) -> bool:
        """
        Cut, transform and resample this instance into the local reference frame of bbox.

        Values of nPoints below 2 disable resampling. Returns whether the instance survives the cut.
        """
        ...


class LineStringInstance(MapInstance):
    """
    Abstract instance whose geometry is a line string.

    Keeps the result of every processing stage as a list, since clipping can split one line into several
    pieces. The lists are index aligned per surviving piece. If resampling is enabled and a piece is too short
    to be resampled, that piece is dropped from all three lists instead of being emitted as an empty geometry.
    """

    @property
    def rawInstance(self) -> List[lanelet2.core.BasicPoint3d]:
        """
        The unprocessed geometry in the map frame, as it was taken from the map element
        """
        ...

    @property
    def cutInstance(self) -> List[List[lanelet2.core.BasicPoint3d]]:
        """
        Geometry clipped to the bounding box, still in the map frame
        """
        ...

    @property
    def cutAndTransformedInstance(self) -> List[List[lanelet2.core.BasicPoint3d]]:
        """
        Geometry clipped to the bounding box and transformed into its local frame
        """
        ...

    @property
    def cutTransformedAndResampledInstance(self) -> List[List[lanelet2.core.BasicPoint3d]]:
        """
        Geometry clipped, transformed and resampled. Empty if resampling was disabled
        """
        ...

    def pointMatrices(self, pointsIn2d: bool) -> List[NDArray[numpy.float64]]:
        """
        Get the points of the processed instance as numpy arrays of shape (nPoints, 2 or 3), one per piece
        """
        ...


class LaneLineStringInstance(LineStringInstance):
    """
    A line string that is part of a lane, e.g. a lane divider, a road border or a centerline.

    Its geometry always runs in the driving direction of the lanelets using it, see the inverted property.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, feature: List[lanelet2.core.BasicPoint3d], mapID: int, type: LineStringType,
                 laneletID: List[int], inverted: bool) -> None:
        """
        Initialize from raw geometry in the map frame, the id of the map element it comes from, its type, the
        ids of the lanelets it belongs to and whether it runs against the direction of the map element.
        """
        ...

    @property
    def type(self) -> LineStringType:
        """
        Type of this line string
        """
        ...

    @property
    def typeInt(self) -> int:
        """
        The type as int, this is what is appended to the instance vectors
        """
        ...

    @property
    def inverted(self) -> bool:
        """
        Whether the geometry runs against the direction of the map element with mapID
        """
        ...

    @property
    def laneletIDs(self) -> List[int]:
        """
        Ids of all lanelets that use this line string, e.g. both neighbours of a shared lane divider
        """
        ...

    def addLaneletID(self, id: int) -> None:
        """
        Register another lanelet as a user of this line string
        """
        ...


class TEInstance(LineStringInstance):
    """
    A traffic element instance, e.g. a stop line, a road marking, a traffic light or a traffic sign.

    Traffic elements are not part of a lane, they are collected from tagged line strings and polygons of the
    map.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, feature: List[lanelet2.core.BasicPoint3d], mapID: int, type: TEType) -> None:
        """
        Initialize from raw geometry in the map frame, the id of the map element it comes from and its type.
        """
        ...

    @property
    def teType(self) -> TEType:
        """
        Type of this traffic element
        """
        ...


class CompoundLaneLineStringInstance(LaneLineStringInstance):
    """
    Several line string instances chained into one, e.g. all lane dividers along one lane.

    Compound instances make the labels independent of how the map happens to be split into individual
    elements. Their type is the representative type of their group, not the type of their first member - with
    the MapTR default grouping, for instance, a chain of dashed and solid boundaries becomes one instance of
    type LineStringType.Divider. The individual instances stay accessible through the features property, so
    every compound instance can be traced back to the map elements it was built from.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, features: List[LaneLineStringInstance], compoundType: LineStringType) -> None:
        """
        Chain the given instances into one compound instance. They are required to be given in already sorted
        order.
        """
        ...

    @property
    def features(self) -> List[LaneLineStringInstance]:
        """
        The individual instances this was built from, in the order they are chained
        """
        ...

    @property
    def pathLengthsRaw(self) -> List[float]:
        """
        Cumulative length [m] of the raw geometry up to and including each member of features
        """
        ...

    @property
    def pathLengthsProcessed(self) -> List[float]:
        """
        Cumulative length [m] of the cut geometry up to and including each member of features
        """
        ...

    @property
    def processedInstancesValid(self) -> BoolList:
        """
        Per member of features: whether enough of it survived the cut
        """
        ...


class LaneletInstance(MapInstance):
    """
    One lanelet, made up of the instances of its left boundary, right boundary and centerline.

    Only lanelets that can be driven on become lanelet instances. Crosswalks, walkways, shared walkways and
    stairs are skipped by MapData - crosswalks are collected as compound instances instead.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, leftBoundary: LaneLineStringInstance, rightBoundary: LaneLineStringInstance,
                 centerline: LaneLineStringInstance, mapID: int) -> None:
        """
        Initialize from already existing boundary and centerline instances.
        """
        ...

    @overload
    def __init__(self, lanelet: lanelet2.core.ConstLanelet) -> None:
        """
        Initialize standalone from a lanelet of the map, creating its boundary and centerline instances.
        """
        ...

    @property
    def leftBoundary(self) -> LaneLineStringInstance:
        """
        Instance of the left boundary
        """
        ...

    @property
    def rightBoundary(self) -> LaneLineStringInstance:
        """
        Instance of the right boundary
        """
        ...

    @property
    def centerline(self) -> LaneLineStringInstance:
        """
        Instance of the centerline
        """
        ...

    def setReprType(self, reprType: LaneletRepresentationType) -> None:
        """
        Set how computeInstanceVectors represents this lanelet.

        Defaults to LaneletRepresentationType.Centerline: centerline points, then the type of the left and the
        right boundary. With Boundaries: left points, right points, then both types.
        """
        ...


class Edge:
    """
    One edge of the lane graph or the traffic element graph, given by the ids it connects.
    """

    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, el1: int, el2: int, isLaneChange: bool) -> None: ...

    el1: int
    el2: int
    isLaneChange: bool


class MapData:
    """
    All local instance labels of one local reference frame pose.

    Created in two phases: build extracts the instances from a local submap, processAll brings them into the
    local reference frame. Only after the second phase are the accessors and getTensorInstanceData meaningful.
    MapDataInterface does both for you.

    All instances are shared, so a line string that several lanelets use, or that is part of a compound
    instance, exists exactly once and can be traced back to its map element through its mapID.
    """

    class TensorInstanceData:
        """
        All instance labels of a MapData object as numpy arrays.

        Instances are grouped by type and only valid ones are included. Within one type, an instance is
        identified by its position in the returned list - that is the index the edge lists refer to.

        Every access copies, so modifying a returned array does not modify this object.
        """

        def __init__(self) -> None: ...

        def lineStringsOfType(self, type: LineStringType) -> List[NDArray[numpy.float64]]:
            """
            Point matrices of all valid lane line strings of the given type
            """
            ...

        def compoundLineStringsOfType(self, type: LineStringType) -> List[NDArray[numpy.float64]]:
            """
            Point matrices of all valid compound line strings of the given type
            """
            ...

        def teInstancesOfType(self, type: TEType) -> List[NDArray[numpy.float64]]:
            """
            Point matrices of all valid traffic elements of the given type
            """
            ...

        def pointMatrixCpdLineStrings(self, type: LineStringType,
                                      index: int) -> CompoundLaneLineStringInstance:
            """
            Get the instance a point matrix of compoundLineStringsOfType came from.

            This gives access to the map elements it was built from. Raises if the type or the index does not
            exist.
            """
            ...

        @property
        def teToCenterlineIndexEdges(self) -> List[Tuple[TEType, int, int]]:
            """
            Traffic element to centerline edges as (source type, source index, target index) tuples, with
            indices local to their type
            """
            ...

        @property
        def teToTEIndexEdges(self) -> List[Tuple[TEType, int, TEType, int]]:
            """
            Traffic element to traffic element edges as (source type, source index, target type, target index)
            tuples, with indices local to their type
            """
            ...

        @property
        def uuid(self) -> str:
            """
            Id of the sample, identical to the uuid of the MapData object this came from
            """
            ...

    def __init__(self) -> None: ...

    @staticmethod
    def build(localSubmap: lanelet2.core.LaneletSubmap, localSubmapGraph: lanelet2.routing.RoutingGraph,
              trafficRules: lanelet2.traffic_rules.TrafficRules,
              bikeSubmapGraph: lanelet2.routing.RoutingGraph = None, ignoreMapElevation: bool = False,
              lineStringTypeGrouping: LineStringTypeGrouping = ...,
              teTypeGrouping: TETypeGrouping = ...) -> 'MapData':
        """
        Extract all instances of a local submap.

        If bikeSubmapGraph is given, bicycle lanes get their own compound centerlines. Raises if
        lineStringTypeGrouping does not cover every type a lanelet boundary can have.
        """
        ...

    def processAll(self, bbox: OrientedRect, paramType: ParametrizationType, resampleLanes: bool = True,
                   nPointsLanes: int = 0, resampleTE: bool = True, nPointsTE: int = 0, pitch: float = 0,
                   roll: float = 0) -> bool:
        """
        Process all instances into the local reference frame given by bbox.

        Values of nPointsLanes or nPointsTE below 2 disable resampling as well. Returns whether every instance
        is still valid afterwards - False just means that some are outside of the bounding box, use the
        validXXX accessors to get the remaining ones.
        """
        ...

    def lineStringsOfType(self, type: LineStringType) -> Dict[int, LaneLineStringInstance]:
        """
        All lane line strings of the given type, by map id
        """
        ...

    def validLineStringsOfType(self, type: LineStringType) -> Dict[int, LaneLineStringInstance]:
        """
        Like lineStringsOfType, but without the instances that did not survive processing
        """
        ...

    def compoundLineStringsOfType(self, type: LineStringType) -> List[CompoundLaneLineStringInstance]:
        """
        All compound line strings of the given type
        """
        ...

    def validCompoundLineStringsOfType(self, type: LineStringType) -> List[CompoundLaneLineStringInstance]:
        """
        Like compoundLineStringsOfType, but without the instances that did not survive processing
        """
        ...

    def associatedCpdLineStringsOfType(self, mapId: int,
                                       type: LineStringType) -> List[CompoundLaneLineStringInstance]:
        """
        All compound instances of the given type that a map element is part of.

        mapId is the id of a member line string or of a lanelet using it, for compound centerlines it is the
        lanelet id. Empty if there are none.
        """
        ...

    def teInstancesOfType(self, type: TEType) -> Dict[int, TEInstance]:
        """
        All traffic elements of the given type, by map id
        """
        ...

    def validTEInstancesOfType(self, type: TEType) -> Dict[int, TEInstance]:
        """
        Like teInstancesOfType, but without the instances that did not survive processing
        """
        ...

    @property
    def laneletInstances(self) -> Dict[int, LaneletInstance]:
        """
        All lanelet instances by map id. Non-driving lanelets are not part of this
        """
        ...

    @property
    def llEdges(self) -> Dict[int, List[Edge]]:
        """
        Lane graph edges by source lanelet id: successors and lane changes between lanelets
        """
        ...

    @property
    def teEdges(self) -> Dict[int, List[Edge]]:
        """
        Traffic element edges by source element id, targets are lanelets or traffic elements
        """
        ...

    @property
    def teToCenterlineEdges(self) -> List[Tuple[TEInstance, CompoundLaneLineStringInstance]]:
        """
        teEdges resolved to instance pairs, for the edges that lead to a lanelet
        """
        ...

    @property
    def teToTEEdges(self) -> List[Tuple[TEInstance, TEInstance]]:
        """
        teEdges resolved to instance pairs, for the edges that lead to another traffic element
        """
        ...

    @property
    def uuid(self) -> str:
        """
        Randomly generated id of this sample, kept across serialization
        """
        ...

    def getTensorInstanceData(self, pointsIn2d: bool, ignoreBuffer: bool) -> 'MapData.TensorInstanceData':
        """
        Get all instance labels as numpy arrays.

        The result is buffered, so if the underlying instances change you need to set ignoreBuffer. Changing
        pointsIn2d alone does not invalidate the buffer.
        """
        ...


class MapDataInterface:
    """
    Main interface class of the module: turns a Lanelet2 map into local instance labels.

    Set a local reference frame pose with setCurrPosAndExtractSubmap, then get the labels for it with mapData.
    For more than one pose, prefer the mapDataBatch methods - they are self contained and do not touch the
    current position.

    Every call to setCurrPosAndExtractSubmap extracts a new local submap around the given position and builds
    the routing graphs for it, which is the expensive part of the work. The traffic rules are currently fixed
    to Germany for vehicles and bicycles.
    """

    class Configuration:
        """
        Parameters of the extraction and processing done by MapDataInterface.
        """

        @overload
        def __init__(self) -> None: ...

        @overload
        def __init__(self, reprType: LaneletRepresentationType, paramType: ParametrizationType,
                     submapExtentLongitudinal: float, submapExtentLateral: float, nPointsLanes: int,
                     nPointsTE: int) -> None: ...

        reprType: LaneletRepresentationType
        paramType: ParametrizationType
        submapExtentLongitudinal: float
        submapExtentLateral: float
        ignoreMapElevation: bool
        allowLaneChangeAcrossBicycleLane: bool
        resampleLanes: bool
        nPointsLanes: int
        resampleTE: bool
        nPointsTE: int
        lineStringTypeGrouping: LineStringTypeGrouping
        teTypeGrouping: TETypeGrouping

    @overload
    def __init__(self, laneletMap: lanelet2.core.LaneletMap) -> None:
        """
        Initialize with the default configuration.
        """
        ...

    @overload
    def __init__(self, laneletMap: lanelet2.core.LaneletMap,
                 config: 'MapDataInterface.Configuration') -> None: ...

    @property
    def config(self) -> 'MapDataInterface.Configuration':
        """
        The configuration this interface was created with
        """
        ...

    def setCurrPosAndExtractSubmap2d(self, pt: lanelet2.core.BasicPoint2d, yaw: float) -> None:
        """
        Set the local reference frame from a 2d pose [m, rad] and extract the submap around it.

        Since there is no elevation to relate to, all instance output of this frame will be 2d.
        """
        ...

    @overload
    def setCurrPosAndExtractSubmap(self, pt: lanelet2.core.BasicPoint3d, yaw: float) -> None:
        """
        Set the local reference frame from a 3d position and a yaw angle [m, rad], with pitch and roll assumed
        to be 0, and extract the submap around it.
        """
        ...

    @overload
    def setCurrPosAndExtractSubmap(self, pt: lanelet2.core.BasicPoint3d, yaw: float, pitch: float,
                                   roll: float) -> None:
        """
        Set the local reference frame from a full 3d pose [m, rad] and extract the submap around it.
        """
        ...

    def mapData(self, processAll: bool) -> MapData:
        """
        Get the instance labels of the current local reference frame.

        If processAll is False you get the raw extracted data and have to call MapData.processAll yourself.
        Raises if no position was set with setCurrPosAndExtractSubmap before.
        """
        ...

    def mapDataBatch2d(self, pts: List[lanelet2.core.BasicPoint2d],
                       yaws: List[float]) -> List[MapData]:
        """
        Get the processed instance labels of several 2d poses at once
        """
        ...

    @overload
    def mapDataBatch(self, pts: List[lanelet2.core.BasicPoint3d], yaws: List[float]) -> List[MapData]:
        """
        Get the processed instance labels of several 3d positions with yaw angles at once, with pitch and roll
        assumed to be 0
        """
        ...

    @overload
    def mapDataBatch(self, pts: List[lanelet2.core.BasicPoint3d], yaws: List[float], pitches: List[float],
                     rolls: List[float]) -> List[MapData]:
        """
        Get the processed instance labels of several full 3d poses at once
        """
        ...


def saveMapData(filename: str, mDataVec: List[MapData], binary: bool) -> None:
    """
    Save all given MapData objects into one file, as a binary or a human readable XML archive
    """
    ...


def loadMapData(filename: str, binary: bool) -> List[MapData]:
    """
    Load the MapData objects that saveMapData wrote into one file
    """
    ...


def saveMapDataMultiFile(path: str, filenames: List[str], mDataVec: List[MapData], binary: bool) -> None:
    """
    Save the given MapData objects into one file each, path is prepended to the file names as it is
    """
    ...


def loadMapDataMultiFile(path: str, filenames: List[str], binary: bool) -> List[MapData]:
    """
    Load the MapData objects that saveMapDataMultiFile wrote into one file each
    """
    ...

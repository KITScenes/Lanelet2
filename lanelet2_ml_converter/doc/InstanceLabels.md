# Instance Labels

This page describes how `lanelet2_ml_converter` gets from a Lanelet2 map to the local instance labels of one
pose, and what the resulting data model looks like. For a quick start and an API overview, see the module
README.

![](summary_flowchart.png)

## From a map to one sample

One sample is the set of labels of one **local reference frame pose**: a position, a heading, and optionally a
pitch and roll angle. Producing it goes through the following steps.

### 1. Submap extraction

`extractSubmap()` searches an axis aligned region around the position and copies the lanelets, line strings,
polygons and regulatory elements it finds into a `LaneletSubmap`. The region is slightly larger than the
requested extents, so that it contains the rotated local reference frame no matter what the heading is — the
exact crop happens later.

Routing graphs are then built on that submap, one for vehicles and one for bicycles. This is the expensive
part of producing a sample, which is why `MapDataInterface` keeps them between calls to `mapData()`, and why
the batch methods exist.

### 2. The local reference frame

`getRotatedRect()` builds the `OrientedRect` that everything is expressed relative to: a rectangle centered on
the position, rotated by the heading. Its extents are **half extents** — `submapExtentLongitudinal` reaches
that far both forwards and backwards.

A rectangle built from a 2d pose is marked with `from2d`. Such a frame has no meaningful elevation, so all
instance output is forced to 2d regardless of what is asked for.

### 3. Instance extraction

`MapData::build()` walks the submap and creates the instances:

- **lane line strings** — every left and right lanelet boundary becomes a `LaneLineStringInstance`. A boundary
  shared by two lanelets exists exactly once and lists both of them in `laneletIDs`. Boundaries are stored in
  the direction the lanelets drive them, which is not always the direction of the map element — hence the
  `inverted` flag.
- **lanelets** — every lanelet that can be driven on becomes a `LaneletInstance` holding its two boundaries and
  its centerline. Crosswalks, walkways, shared walkways and stairs are skipped here.
- **compound instances** — see below.
- **traffic elements** — stop lines, arrow and symbol markings, traffic lights and traffic signs become
  `TEInstance`s, together with the edges that connect them to lanes and to each other.

Nothing has been cut or moved at this point; all geometry is still in the map frame.

### 4. Processing

`MapData::processAll()` calls `process()` on every instance, which clips it to the `OrientedRect`, transforms
what remains into the frame of that rectangle and optionally resamples it to a fixed number of points. Every
stage is kept, see the pipeline table in the README.

Instances that fall outside of the frame, or that keep less than 0.1 m of length, are marked invalid. They are
not removed — they stay accessible for traceability but are skipped by the `validXXX()` accessors and by the
tensor data.

### 5. Tensor data

`MapData::getTensorInstanceData()` produces the form that is actually fed to a model: point matrices grouped by
type, plus the edges expressed as indices into those groups. The result is buffered, so repeated calls are
cheap — pass `ignoreBuffer` if the underlying instances changed.

## Compound instances

A map splits a lane boundary wherever the annotation happens to need a split: at a change of marking type, at
an intersection with another line string, or simply because the mapper drew it in two pieces. A model should
not have to reproduce those artifacts, so the converter chains boundaries back together into
`CompoundLaneLineStringInstance`s.

### Candidate chains

`getPaths()` enumerates the **lanelet paths** of the submap: sequences that start at a lanelet without
predecessor, split into several paths at junctions, and end at a lanelet without successor. Only successors of
the same subtype are followed, so a path never mixes a bicycle lane with a road.

Along each path, the left boundaries are chained as long as their types stay in the same group of the
`LineStringTypeGrouping`, and likewise the right ones. Each such chain is a *candidate*. The centerlines of the
path are chained into one compound centerline unconditionally.

### Resolving the candidates

Candidates from different paths overlap wherever paths share lanelets, so they have to be resolved into
disjoint chains. Doing that greedily in enumeration order would make the result depend on the order the paths
came in, which is not acceptable for a label generator.

Instead, all candidates are collected first and resolved afterwards, per representative type (candidates of
different types never share an element):

1. a greedy pass takes the candidates longest-first, keeping each one intact where it can and letting shorter
   ones contribute whatever elements are still free
2. an improvement pass then adopts a candidate whenever emitting it whole removes more chains than re-cutting
   the chains it overlaps creates

The result depends only on the *set* of candidates, not on their order, and every resulting chain is still a
contiguous part of one candidate — so it stays traceable to a single lanelet path.

### Labelling

A compound instance carries the **representative type of its group**, not the type of its first member. With
`getMapTRDefaultSimpleGrouping()`, a run of dashed and solid boundaries becomes one instance of type `Divider`;
with `getM3TRDefaultGrouping()`, the same run becomes two instances, one `Dashed` and one `Solid`.

### Other compound instances

Two kinds of compound instances are not built from lanelet paths: the drivable space borders described in the
next section, and **pedestrian crossings** — crosswalk lanelets turned into a closed perimeter that runs along
the left border and back along the right one, labelled `ZebraCrossing` or `PedestrianCrossing` depending on how
those borders are tagged.

## Drivable space borders

The lane structure of a map says where a vehicle is *supposed* to drive. It does not say where a vehicle
*could* drive: the extent of the paved, physically drivable space around the lanes. That is expected to be annotated
separately, and it is where `LineStringType::DrivableArea` comes from.

### Annotation

A LineString is part of the drivable space border if it carries the tag `drivable_space_border=true`. Such a
LineString:

- does **not** have to be a lanelet boundary. It lives next to the lanelets, is not referenced by any of them, and its
  instance consequently has an empty `laneletIDs` list
- does not have to follow the lane structure. It may run past several lanes, cut across an intersection, or
  exist where there is no lanelet at all
- keeps whatever `type` / `subtype` tags it carries for other purposes; those are ignored here, the
  `drivable_space_border` tag alone decides

### Why this exists

Trying to infer the drivable area polygon from the lanelets is algorithmically difficult and very
ambiguous depending on map-specific tagging and definitions, therefore this explicit annotation is required.

### Chaining

The individual borders are split by the annotation just as lane boundaries are, so `computeDrivableAreaBorders()`
chains them back together. Unlike the lanelet path chaining, this does not go through the candidate resolution —
there are no paths and no overlapping candidates to resolve, only a graph of borders to walk:

1. **Collect**: every LineString of the submap with the tag and at least two points becomes an individual
   `DrivableArea` instance, stored under its own map id.
2. **Connect**: two borders are adjacent if they share an endpoint **point**, i.e. the very same `Point3d` id.
   Coordinates are not compared — two borders that end at the same position in two distinct points are not
   adjacent. This makes chaining a question about the annotation's topology rather than about geometric
   tolerances.
3. **Group**: the adjacency graph is split into connected components by depth first search.
4. **Walk**: each component is walked into continuous chains. A chain starts, where possible, at a border with a
   *dangling* endpoint — one not shared with any neighbour — so that open runs are traversed end to end rather
   than started in the middle. From there the walk follows shared endpoints, inverting each border as needed so
   that the chain runs in one consistent direction.
5. **Emit**: each chain becomes one `CompoundLaneLineStringInstance` of type `DrivableArea`.

A component is not necessarily one chain. Where more than two borders meet, the walk consumes one branch and
then stops, and the remaining borders of that component are walked into further chains. A branching drivable
space therefore yields several compound instances, each of which is a genuinely continuous run, instead of one
instance that jumps across the junction.

Closed loops — a component whose borders all connect at both ends, e.g. an enclosed area — have no dangling
endpoint to start from, so the walk starts at an arbitrary member and comes back around to it.

### Traceability

Each border ends up in the data twice: as an individual instance under its own map id, and as a member of the
compound instance it was chained into. The usual traceability accessors apply —
`associatedCpdLineStringsOfType(mapId, LineStringType::DrivableArea)` goes from a map element to the compound
instances containing it, and `features()` goes back the other way.

## Traceability

Every instance knows the map element it came from through `mapID`, including the compound ones — a
`CompoundLaneLineStringInstance` exposes the individual instances it chained through `features()`, each with
its own map id, its own `inverted` flag and its own share of the path length.

The chain survives all the way into the tensor data:
`TensorInstanceData::pointMatrixCpdLineStrings(type, index)` takes the type and the index of a point matrix and
returns the compound instance it was produced from. Going the other way,
`MapData::associatedCpdLineStringsOfType(mapId, type)` answers which compound instances a given map element
ended up in.

This is what makes it possible to relate a model's prediction back to the map it was trained on — for example
to evaluate per map element, or to feed a prediction back into map annotation.

## Coordinate conventions

- Positions are in metres, angles in radians.
- The heading is measured counter clockwise from the map frame x axis.
- The rotation into the local frame is applied as yaw, then pitch, then roll.
- The extents of the local reference frame are half extents in each direction.
- `ignoreMapElevation` zeroes the z coordinate at *extraction* time, so it affects the raw geometry too, not
  only the output.
- `allowLaneChangeAcrossBicycleLane` (default off) builds the routing graphs with
  `RoutingGraph::AllowLaneChangeAcrossBicycleLane`: two vehicle lanes with a bicycle lane between them get a
  left/right lane change edge to each other when both bike lane bounds may be crossed (dashed `bike_marking`,
  dashed `line_thin`/`line_thick`). See the lanelet2_routing README.

## Serialization

`saveMapData()` / `loadMapData()` write and read complete `MapData` objects, either as a boost binary archive
or as human readable XML. `saveMapDataMultiFile()` / `loadMapDataMultiFile()` do the same with one file per
object, which is the more convenient layout for a dataset.

The `uuid` of a sample is preserved, so a stored label set can be matched back to whatever it was generated
alongside. The archives are not versioned — data written by one version of the module can only be read back by
a version whose types still have the same members.

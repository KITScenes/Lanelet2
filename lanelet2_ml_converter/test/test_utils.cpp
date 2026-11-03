#include <gtest/gtest.h>

#include <algorithm>
#include <string>

#include "lanelet2_ml_converter/Utils.h"
#include "test_map.h"

using namespace lanelet;
using namespace lanelet::ml_converter;
using namespace lanelet::ml_converter::tests;

TEST_F(MLConverterTest, GetRotatedRect) {  // NOLINT
  EXPECT_DOUBLE_EQ(bbox.bounds_const()[0].x(), 15);
  EXPECT_DOUBLE_EQ(bbox.bounds_const()[0].y(), -10);
  EXPECT_DOUBLE_EQ(bbox.bounds_const()[1].x(), -5);
  EXPECT_DOUBLE_EQ(bbox.bounds_const()[1].y(), -10);
  EXPECT_DOUBLE_EQ(bbox.bounds_const()[2].x(), -5);
  EXPECT_DOUBLE_EQ(bbox.bounds_const()[2].y(), 20);
  EXPECT_DOUBLE_EQ(bbox.bounds_const()[3].x(), 15);
  EXPECT_DOUBLE_EQ(bbox.bounds_const()[3].y(), 20);
}

TEST_F(MLConverterTest, ExtractSubmap) {  // NOLINT
  LaneletSubmapConstPtr submap =
      extractSubmap(laneletMap, centerBbox.head(2), extentLongitudinalBbox, extentLateralBbox);
  EXPECT_TRUE(submap->laneletLayer.exists(2000));
  EXPECT_TRUE(submap->laneletLayer.exists(2001));
  EXPECT_TRUE(submap->laneletLayer.exists(2002));
  EXPECT_TRUE(submap->laneletLayer.exists(2003));
  EXPECT_TRUE(submap->laneletLayer.exists(2004));
  EXPECT_TRUE(submap->laneletLayer.exists(2005));
  EXPECT_TRUE(submap->laneletLayer.exists(2009));
  EXPECT_TRUE(submap->laneletLayer.exists(2010));
}

TEST(UtilsTest, BoundaryTypeOfBikeMarkings) {  // NOLINT
  Id id = 9000;
  auto line = [&id](const char* type, const char* subtype) {
    AttributeMap attrs;
    attrs[AttributeName::Type] = type;
    if (std::string(subtype).size() > 0) {
      attrs[AttributeName::Subtype] = subtype;
    }
    Point3d p0(id++, 0, 0, 0);
    Point3d p1(id++, 1, 0, 0);
    return ConstLineString3d(LineString3d(id++, {p0, p1}, attrs));
  };
  EXPECT_EQ(bdTypeToEnum(line("bike_marking", "dashed")), LineStringType::BikeMarkingDashed);
  EXPECT_EQ(bdTypeToEnum(line("bike_marking", "solid")), LineStringType::BikeMarkingSolid);
  EXPECT_EQ(bdTypeToEnum(line("bike_marking", "solid_solid")), LineStringType::BikeMarkingSolid);
  EXPECT_EQ(bdTypeToEnum(line("bike_marking", "solid_dashed")), LineStringType::BikeMarkingSolid);
  EXPECT_EQ(bdTypeToEnum(line("bike_marking", "dashed_solid")), LineStringType::BikeMarkingSolid);
  EXPECT_EQ(bdTypeToEnum(line("bike_marking", "")), LineStringType::Unknown);
  // lane markings of the road are not affected
  EXPECT_EQ(bdTypeToEnum(line("line_thin", "dashed")), LineStringType::Dashed);
  EXPECT_EQ(bdTypeToEnum(line("line_thick", "solid")), LineStringType::Solid);
  // every type bdTypeToEnum returns is listed as a boundary type (the groupings are checked against that list)
  auto types = boundaryLineStringTypes();
  for (auto t : {LineStringType::BikeMarkingDashed, LineStringType::BikeMarkingSolid}) {
    EXPECT_NE(std::find(types.begin(), types.end(), t), types.end());
  }
}

TEST(UtilsTest, ResampleLineString) {  // NOLINT
  BasicLineString3d polyline{BasicPoint3d{0, 0, 0}, BasicPoint3d{3, 0, 0}, BasicPoint3d{6, 0, 0},
                             BasicPoint3d{15, 0, 0}};
  BasicLineString3d polylineResampled = resampleLineString(polyline, 11);

  EXPECT_EQ(polylineResampled.size(), 11);
  EXPECT_NEAR(polylineResampled[3].x(), 4.5, 10e-5);
  EXPECT_NEAR(polylineResampled[6].x(), 9, 10e-5);
  EXPECT_NEAR(polylineResampled[10].x(), 15, 10e-5);
}

TEST_F(MLConverterTest, CutLineString) {  // NOLINT
  BasicLineString3d polyline{BasicPoint3d{0, 0, 0}, BasicPoint3d{30, 0, 0}, BasicPoint3d{40, 0, 0},
                             BasicPoint3d{50, 0, 0}};
  std::vector<BasicLineString3d> polylineCut = cutLineString(bbox, polyline);

  EXPECT_EQ(polylineCut.size(), 1);
  EXPECT_EQ(polylineCut[0].size(), 2);
  EXPECT_NEAR(polylineCut[0][1].x(), 15, 10e-5);
}

// Verify z-restoration for a vertical traffic element rectangle whose
// corners share the same (x,y) but differ in z  (BL→TL→TR→BR).
TEST_F(MLConverterTest, CutLineStringVerticalTE) {  // NOLINT
  // A small rectangle entirely inside the bbox (x∈[-5,15], y∈[-10,20]).
  // BL=(3,2,0), TL=(3,2,3), TR=(5,2,3), BR=(5,2,0)
  BasicLineString3d rect{BasicPoint3d{3, 2, 0}, BasicPoint3d{3, 2, 3}, BasicPoint3d{5, 2, 3}, BasicPoint3d{5, 2, 0}};
  std::vector<BasicLineString3d> result = cutLineString(bbox, rect);

  ASSERT_EQ(result.size(), 1);
  ASSERT_EQ(result[0].size(), 4);

  // Each clipped point must have the correct z restored via segment
  // interpolation — in particular BL≠TL and BR≠TR despite sharing (x,y).
  EXPECT_NEAR(result[0][0].z(), 0.0, 1e-6);  // BL
  EXPECT_NEAR(result[0][1].z(), 3.0, 1e-6);  // TL
  EXPECT_NEAR(result[0][2].z(), 3.0, 1e-6);  // TR
  EXPECT_NEAR(result[0][3].z(), 0.0, 1e-6);  // BR
}
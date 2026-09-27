"""Generate QGIS Project (data/qgis_picking.qgz) and layers for Held-Out Point Picking.

Layers generated:
- data/intersection_outline.geojson (42% overlap region in red outline)
- data/spacing_grid.geojson (30m spacing grid clipped to overlap region)
- data/heldout_points_template.geojson (Scratch vector point layer template)
- data/qgis_picking.qgz (Zipped QGIS 3.x project file with pre-loaded layers)
"""

import json
import os
import zipfile
from pathlib import Path

def create_layers():
    Path("data").mkdir(parents=True, exist_ok=True)
    
    # 1. Intersection outline GeoJSON (42% overlap polygon in pixel coords + stereo coords)
    # Bounding box of 42% overlap region in pixel space: cols 553..2531, rows 37269..49148
    # In GIS/QGIS pixel space (y inverted): y_min = 52224 - 49148 = 3076, y_max = 52224 - 37269 = 14955
    x_min, x_max = 553.0, 2531.0
    y_min, y_max = 37269.0, 49148.0
    
    intersection_geojson = {
        "type": "FeatureCollection",
        "name": "intersection_outline",
        "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "42% Overlap Region",
                    "area_pct_ohrc": 42.04,
                    "area_pct_nac": 9.87,
                    "description": "Ground-truth picking zone between OHRC strip and NAC reference"
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [x_min, y_min],
                        [x_max, y_min],
                        [x_max, y_max],
                        [x_min, y_max],
                        [x_min, y_min]
                    ]]
                }
            }
        ]
    }
    
    with open("data/intersection_outline.geojson", "w") as f:
        json.dump(intersection_geojson, f, indent=2)

    # 2. Spacing grid GeoJSON (30m / 60px cell spacing within overlap polygon)
    grid_features = []
    step_x = 60.0  # ~30m at 0.5m GSD or 60 px
    step_y = 60.0
    
    # Generate grid cell lines/polygons across the intersection bbox
    cur_x = x_min
    cell_id = 1
    while cur_x < x_max:
        cur_y = y_min
        while cur_y < y_max:
            cell_x1 = cur_x
            cell_x2 = min(cur_x + step_x, x_max)
            cell_y1 = cur_y
            cell_y2 = min(cur_y + step_y, y_max)
            
            grid_features.append({
                "type": "Feature",
                "properties": { "cell_id": cell_id },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [cell_x1, cell_y1],
                        [cell_x2, cell_y1],
                        [cell_x2, cell_y2],
                        [cell_x1, cell_y2],
                        [cell_x1, cell_y1]
                    ]]
                }
            })
            cell_id += 1
            cur_y += step_y * 10  # grid cell spacing for visualization (~300m cells for readable grid overlay)
        cur_x += step_x * 10

    grid_geojson = {
        "type": "FeatureCollection",
        "name": "spacing_grid",
        "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
        "features": grid_features
    }

    with open("data/spacing_grid.geojson", "w") as f:
        json.dump(grid_geojson, f, indent=2)

    # 3. Scratch heldout points template GeoJSON
    points_geojson = {
        "type": "FeatureCollection",
        "name": "heldout_points",
        "crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
        "features": []
    }
    with open("data/heldout_points_template.geojson", "w") as f:
        json.dump(points_geojson, f, indent=2)

    print("Created GeoJSON layers in data/")


def create_qgis_project_file():
    # Construct QGIS project XML (.qgs)
    abs_root = os.path.abspath(".").replace("\\", "/")
    
    qgs_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<qgis version="3.28.0" projectname="QGIS Held-Out Point Picking Workspace">
  <title>QGIS Held-Out Point Picking Workspace</title>
  <homePath path="{abs_root}"/>
  <layer-tree-group>
    <customproperties/>
    <layer-tree-layer name="heldout_points (Scratch)" checked="Qt::Checked" id="heldout_points_scratch"/>
    <layer-tree-layer name="30m Spacing Grid" checked="Qt::Checked" id="spacing_grid_layer"/>
    <layer-tree-layer name="42% Overlap Region (Red Outline)" checked="Qt::Checked" id="intersection_outline_layer"/>
    <layer-tree-layer name="Registered OHRC Product (registered.tif)" checked="Qt::Checked" id="registered_ohrc_layer"/>
    <layer-tree-layer name="NAC Reference (M113679075LC)" checked="Qt::Checked" id="nac_reference_layer"/>
  </layer-tree-group>
  <projectlayers>
    <maplayer type="vector" name="heldout_points (Scratch)" id="heldout_points_scratch" geometry="Point">
      <id>heldout_points_scratch</id>
      <datasource>./heldout_points_template.geojson</datasource>
      <layername>heldout_points (Scratch)</layername>
      <srs>
        <spatialrefsys>
          <authid>EPSG:4326</authid>
          <description>WGS 84</description>
        </spatialrefsys>
      </srs>
      <providername>ogr</providername>
      <vectorjoins/>
      <layerDependencies/>
      <fieldConfiguration>
        <field name="feature_type"><editWidget type="TextEdit"/></field>
        <field name="confidence"><editWidget type="Range"/></field>
        <field name="notes"><editWidget type="TextEdit"/></field>
      </fieldConfiguration>
      <renderer-v2 type="singleSymbol">
        <symbol alpha="1" type="marker" name="0">
          <layer class="SimpleMarker" locked="0" pass="0">
            <prop k="color" v="255,255,0,255"/>
            <prop k="outline_color" v="0,0,0,255"/>
            <prop k="size" v="4"/>
          </layer>
        </symbol>
      </renderer-v2>
    </maplayer>

    <maplayer type="vector" name="30m Spacing Grid" id="spacing_grid_layer" geometry="Polygon">
      <id>spacing_grid_layer</id>
      <datasource>./spacing_grid.geojson</datasource>
      <layername>30m Spacing Grid</layername>
      <providername>ogr</providername>
      <renderer-v2 type="singleSymbol">
        <symbol alpha="0.5" type="fill" name="0">
          <layer class="SimpleLine" locked="0" pass="0">
            <prop k="line_color" v="0,255,255,255"/>
            <prop k="line_width" v="0.5"/>
          </layer>
        </symbol>
      </renderer-v2>
    </maplayer>

    <maplayer type="vector" name="42% Overlap Region (Red Outline)" id="intersection_outline_layer" geometry="Polygon">
      <id>intersection_outline_layer</id>
      <datasource>./intersection_outline.geojson</datasource>
      <layername>42% Overlap Region (Red Outline)</layername>
      <providername>ogr</providername>
      <renderer-v2 type="singleSymbol">
        <symbol alpha="1" type="fill" name="0">
          <layer class="SimpleLine" locked="0" pass="0">
            <prop k="line_color" v="255,0,0,255"/>
            <prop k="line_width" v="2.0"/>
          </layer>
        </symbol>
      </renderer-v2>
    </maplayer>

    <maplayer type="raster" name="Registered OHRC Product (registered.tif)" id="registered_ohrc_layer">
      <id>registered_ohrc_layer</id>
      <datasource>../outputs/ohrc_test_001/registered.tif</datasource>
      <layername>Registered OHRC Product (registered.tif)</layername>
      <providername>gdal</providername>
      <pipe>
        <provider>
          <resampling maxOversampling="2" enabled="true" type="1"/>
        </provider>
        <brightnesscontrast brightness="0" contrast="20"/>
        <rasterrenderer type="singlebandgray" grayBand="1" alphaBand="-1">
          <contrastEnhancement>
            <minValue>0</minValue>
            <maxValue>255</maxValue>
            <algorithm>StretchToMinimumMaximum</algorithm>
          </contrastEnhancement>
        </rasterrenderer>
      </pipe>
    </maplayer>

    <maplayer type="raster" name="NAC Reference (M113679075LC)" id="nac_reference_layer">
      <id>nac_reference_layer</id>
      <datasource>./lroc_nac/M113679075LC.IMG</datasource>
      <layername>NAC Reference (M113679075LC)</layername>
      <providername>gdal</providername>
      <pipe>
        <provider>
          <resampling maxOversampling="2" enabled="true" type="1"/>
        </provider>
        <brightnesscontrast brightness="0" contrast="15"/>
        <rasterrenderer type="singlebandgray" grayBand="1" alphaBand="-1">
          <contrastEnhancement>
            <minValue>0</minValue>
            <maxValue>3000</maxValue>
            <algorithm>StretchToMinimumMaximum</algorithm>
          </contrastEnhancement>
        </rasterrenderer>
      </pipe>
    </maplayer>
  </projectlayers>
</qgis>
"""

    qgs_path = "data/qgis_picking.qgs"
    qgz_path = "data/qgis_picking.qgz"

    with open(qgs_path, "w", encoding="utf-8") as f:
        f.write(qgs_xml)

    # Zip .qgs file into .qgz archive
    with zipfile.ZipFile(qgz_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(qgs_path, arcname="qgis_picking.qgs")

    print(f"Generated QGIS Project file: {qgz_path} and XML: {qgs_path}")

if __name__ == "__main__":
    create_layers()
    create_qgis_project_file()

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Union
from chandra_align.pipeline.router import SensorMeta


def _strip_namespace(tag: str) -> str:
    """Removes XML namespace prefix if present (e.g., '{http://pds.nasa.gov/pds4/pds/v1}Target' -> 'Target')."""
    return tag.split('}')[-1] if '}' in tag else tag


def parse_pds4_metadata(xml_path: Union[str, Path]) -> SensorMeta:
    """
    Parses a PDS4 XML label file and extracts sensor metadata into SensorMeta.
    Handles XML namespace variations and provides safe defaults for optional fields.
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(f"PDS4 XML file not found: {xml_path}")

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Failed to parse PDS4 XML label '{xml_path}': {e}")

    # Create a mapping of lowercased tag names (without namespaces) to elements
    tag_map: Dict[str, ET.Element] = {}
    for elem in root.iter():
        clean_tag = _strip_namespace(elem.tag).lower()
        if clean_tag not in tag_map:
            tag_map[clean_tag] = elem

    # 1. Instrument / Sensor Name
    sensor_name = "UNKNOWN_SENSOR"
    for tag_key in ["instrument_name", "instrument_id", "telescope_name", "logical_identifier"]:
        if tag_key in tag_map and tag_map[tag_key].text:
            text = tag_map[tag_key].text.upper()
            if "OHRC" in text:
                sensor_name = "OHRC"
            elif "IIRS" in text:
                sensor_name = "IIRS"
            elif "TMC" in text:
                sensor_name = "TMC2"
            elif "NAC" in text or "LROC" in text:
                sensor_name = "LROC_NAC"
            elif "WAC" in text:
                sensor_name = "LROC_WAC"
            else:
                sensor_name = text.split(":")[-1]
            break

    # 2. Sensor Type
    sensor_type = "SWIR" if ("IIRS" in sensor_name or "SWIR" in sensor_name) else "PANCHROMATIC"

    # 3. Spatial Resolution / GSD (meters)
    gsd_meters = 0.5  # Default fallback
    for gsd_key in ["pixel_resolution", "map_scale", "spatial_resolution", "pixel_scale"]:
        if gsd_key in tag_map and tag_map[gsd_key].text:
            try:
                gsd_meters = float(tag_map[gsd_key].text)
                break
            except ValueError:
                continue

    # 4. Solar Azimuth Angle (degrees)
    solar_azimuth_deg = 0.0
    for az_key in ["solar_azimuth_angle", "illumination_azimuth_angle", "sub_solar_azimuth"]:
        if az_key in tag_map and tag_map[az_key].text:
            try:
                solar_azimuth_deg = float(tag_map[az_key].text) % 360.0
                break
            except ValueError:
                continue

    # 5. Solar Elevation / Incidence Angle (degrees)
    solar_elevation_deg = 45.0
    for el_key in ["solar_elevation_angle", "incidence_angle", "solar_zenith_angle"]:
        if el_key in tag_map and tag_map[el_key].text:
            try:
                val = float(tag_map[el_key].text)
                # Convert zenith/incidence to elevation if needed
                solar_elevation_deg = (90.0 - val) if ("zenith" in el_key or "incidence" in el_key) else val
                break
            except ValueError:
                continue

    # Map to SensorMeta fields: sun_azimuth_deg, sun_incidence_deg
    return SensorMeta(
        sensor_name=sensor_name,
        sensor_type=sensor_type,
        gsd_meters=gsd_meters,
        sun_azimuth_deg=solar_azimuth_deg,
        sun_incidence_deg=solar_elevation_deg,
    )
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
import pytest
from chandra_align.ingestion.pds4 import parse_pds4_metadata, _strip_namespace
from chandra_align.pipeline.router import SensorMeta


class TestPDS4Ingestion:
    """Unit tests for PDS4 XML ingestion."""

    def test_strip_namespace(self):
        """Test namespace stripping from XML tags."""
        assert _strip_namespace("{http://pds.nasa.gov/pds4/pds/v1}Target") == "Target"
        assert _strip_namespace("Target") == "Target"
        assert _strip_namespace("{ns}Instrument_Name") == "Instrument_Name"

    def test_parse_basic_ohrc_xml(self):
        """Test parsing OHRC PDS4 XML with standard tags."""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<Product xmlns="http://pds.nasa.gov/pds4/pds/v1">
    <Identification_Area>
        <logical_identifier>CH2_OHR_NCP_20210405</logical_identifier>
        <instrument_name>OHRC</instrument_name>
        <pixel_resolution>0.25</pixel_resolution>
        <solar_azimuth_angle>45.0</solar_azimuth_angle>
        <incidence_angle>30.0</incidence_angle>
    </Identification_Area>
</Product>'''
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml_content)
            xml_path = f.name
        
        try:
            meta = parse_pds4_metadata(xml_path)
            assert meta.sensor_name == "OHRC"
            assert meta.sensor_type == "PANCHROMATIC"
            assert meta.gsd_meters == 0.25
            assert meta.solar_azimuth_deg == 45.0
            assert meta.solar_elevation_deg == 60.0  # 90 - 30
        finally:
            Path(xml_path).unlink()

    def test_parse_iirs_xml(self):
        """Test parsing IIRS PDS4 XML with SWIR sensor type."""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<Product xmlns="http://pds.nasa.gov/pds4/pds/v1">
    <Identification_Area>
        <instrument_id>IIRS</instrument_id>
        <spatial_resolution>80.0</spatial_resolution>
        <illumination_azimuth_angle>140.0</illumination_azimuth_angle>
        <solar_elevation_angle>15.0</solar_elevation_angle>
    </Identification_Area>
</Product>'''
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml_content)
            xml_path = f.name
        
        try:
            meta = parse_pds4_metadata(xml_path)
            assert meta.sensor_name == "IIRS"
            assert meta.sensor_type == "SWIR"
            assert meta.gsd_meters == 80.0
            assert meta.solar_azimuth_deg == 140.0
            assert meta.solar_elevation_deg == 15.0
        finally:
            Path(xml_path).unlink()

    def test_parse_lroc_nac_xml(self):
        """Test parsing LROC NAC PDS4 XML."""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<Product xmlns="http://pds.nasa.gov/pds4/pds/v1">
    <Identification_Area>
        <telescope_name>LROC NAC</telescope_name>
        <map_scale>0.5</map_scale>
        <sub_solar_azimuth>52.0</sub_solar_azimuth>
        <solar_zenith_angle>62.0</solar_zenith_angle>
    </Identification_Area>
</Product>'''
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml_content)
            xml_path = f.name
        
        try:
            meta = parse_pds4_metadata(xml_path)
            assert meta.sensor_name == "LROC_NAC"
            assert meta.sensor_type == "PANCHROMATIC"
            assert meta.gsd_meters == 0.5
            assert meta.solar_azimuth_deg == 52.0
            assert meta.solar_elevation_deg == 28.0  # 90 - 62
        finally:
            Path(xml_path).unlink()

    def test_parse_with_namespace_variations(self):
        """Test parsing with different namespace prefixes."""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<pds:Product xmlns:pds="http://pds.nasa.gov/pds4/pds/v1">
    <pds:Identification_Area>
        <pds:instrument_name>OHRC</pds:instrument_name>
        <pds:pixel_scale>0.25</pds:pixel_scale>
        <pds:solar_azimuth_angle>30.0</pds:solar_azimuth_angle>
        <pds:incidence_angle>45.0</pds:incidence_angle>
    </pds:Identification_Area>
</pds:Product>'''
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml_content)
            xml_path = f.name
        
        try:
            meta = parse_pds4_metadata(xml_path)
            assert meta.sensor_name == "OHRC"
            assert meta.gsd_meters == 0.25
            assert meta.solar_azimuth_deg == 30.0
            assert meta.solar_elevation_deg == 45.0
        finally:
            Path(xml_path).unlink()

    def test_missing_tags_fallback_to_defaults(self):
        """Test fallback to defaults when tags are missing."""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<Product>
    <Identification_Area>
        <instrument_name>UNKNOWN</instrument_name>
    </Identification_Area>
</Product>'''
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml_content)
            xml_path = f.name
        
        try:
            meta = parse_pds4_metadata(xml_path)
            assert meta.sensor_name == "UNKNOWN"
            assert meta.sensor_type == "PANCHROMATIC"
            assert meta.gsd_meters == 0.5  # default
            assert meta.solar_azimuth_deg == 0.0  # default
            assert meta.solar_elevation_deg == 45.0  # default
        finally:
            Path(xml_path).unlink()

    def test_malformed_xml_raises_error(self):
            """Test that malformed XML raises appropriate error."""
            xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
        <Product>
            <Identification_Area>
                <instrument_name>OHRC
            </Identification_Area>
        </Product>'''
            with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
                f.write(xml_content)
                xml_path = f.name
        
            try:
                with pytest.raises(ValueError, match="Failed to parse PDS4 XML label"):
                    parse_pds4_metadata(xml_path)
            finally:
                Path(xml_path).unlink()

    def test_file_not_found_raises_error(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            parse_pds4_metadata("/nonexistent/path.xml")

    def test_azimuth_wraparound(self):
        """Test azimuth angle normalization to [0, 360)."""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<Product>
    <Identification_Area>
        <instrument_name>OHRC</instrument_name>
        <solar_azimuth_angle>400.0</solar_azimuth_angle>
    </Identification_Area>
</Product>'''
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml_content)
            xml_path = f.name
        
        try:
            meta = parse_pds4_metadata(xml_path)
            assert meta.solar_azimuth_deg == 40.0  # 400 % 360
        finally:
            Path(xml_path).unlink()

    def test_negative_azimuth(self):
        """Test negative azimuth angle handling."""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<Product>
    <Identification_Area>
        <instrument_name>OHRC</instrument_name>
        <solar_azimuth_angle>-30.0</solar_azimuth_angle>
    </Identification_Area>
</Product>'''
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml_content)
            xml_path = f.name
        
        try:
            meta = parse_pds4_metadata(xml_path)
            assert meta.solar_azimuth_deg == 330.0  # -30 % 360
        finally:
            Path(xml_path).unlink()

    def test_zenith_to_elevation_conversion(self):
        """Test solar zenith angle converted to elevation."""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<Product>
    <Identification_Area>
        <instrument_name>OHRC</instrument_name>
        <solar_zenith_angle>60.0</solar_zenith_angle>
    </Identification_Area>
</Product>'''
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write(xml_content)
            xml_path = f.name
        
        try:
            meta = parse_pds4_metadata(xml_path)
            assert meta.solar_elevation_deg == 30.0  # 90 - 60
        finally:
            Path(xml_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
import tempfile
import json
import csv
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
from chandra_align.cli.batch import (
    load_manifest,
    process_single_pair,
    run_batch_pipeline,
    BatchPairResult,
)
from chandra_align.pipeline.router import SensorMeta, MatcherStrategy


class TestBatchCLI:
    """Unit tests for Batch CLI."""

    def test_batch_pair_result_dataclass(self):
        """Test BatchPairResult dataclass creation."""
        result = BatchPairResult(
            pair_id="test_001",
            source_pds4="src.xml",
            ref_pds4="ref.xml",
            primary_strategy="LIGHTGLUE_ALIKED",
            delta_azimuth_deg=7.0,
            scale_ratio=0.5,
            status="SUCCESS",
        )
        assert result.pair_id == "test_001"
        assert result.status == "SUCCESS"
        assert result.error_msg == ""

    def test_batch_pair_result_failed(self):
        """Test BatchPairResult for failed case."""
        result = BatchPairResult(
            pair_id="test_002",
            source_pds4="src.xml",
            ref_pds4="ref.xml",
            primary_strategy="FAILED",
            delta_azimuth_deg=0.0,
            scale_ratio=1.0,
            status="FAILED",
            error_msg="File not found",
        )
        assert result.status == "FAILED"
        assert result.error_msg == "File not found"

    @patch('chandra_align.cli.batch.parse_pds4_metadata')
    @patch('chandra_align.cli.batch.evaluate_route')
    def test_process_single_pair_success(self, mock_evaluate, mock_parse):
        """Test successful single pair processing."""
        # Setup mocks
        mock_parse.side_effect = [
            SensorMeta("OHRC", "PANCHROMATIC", 0.25, 45.0, 30.0),
            SensorMeta("LROC_NAC", "PANCHROMATIC", 0.5, 52.0, 28.0),
        ]
        mock_evaluate.return_value = MagicMock(
            primary_strategy=MatcherStrategy.LIGHTGLUE_ALIKED,
            delta_azimuth_deg=7.0,
            scale_ratio=0.5,
        )
        
        result = process_single_pair(("src.xml", "ref.xml"))
        
        assert result.status == "SUCCESS"
        assert result.primary_strategy == "LIGHTGLUE_ALIKED"
        assert result.delta_azimuth_deg == 7.0
        assert result.scale_ratio == 0.5
        assert result.pair_id == "src_VS_ref"

    @patch('chandra_align.cli.batch.parse_pds4_metadata')
    def test_process_single_pair_file_not_found(self, mock_parse):
        """Test single pair processing with missing file."""
        mock_parse.side_effect = FileNotFoundError("PDS4 XML file not found: missing.xml")
        
        result = process_single_pair(("missing.xml", "ref.xml"))
        
        assert result.status == "FAILED"
        assert "File not found" in result.error_msg or "not found" in result.error_msg.lower()

    @patch('chandra_align.cli.batch.parse_pds4_metadata')
    def test_process_single_pair_invalid_xml(self, mock_parse):
        """Test single pair processing with invalid XML."""
        from xml.etree import ElementTree as ET
        mock_parse.side_effect = ValueError("Failed to parse PDS4 XML label 'corrupt.xml': invalid token")
        
        result = process_single_pair(("corrupt.xml", "ref.xml"))
        
        assert result.status == "FAILED"
        assert "parse" in result.error_msg.lower() or "invalid" in result.error_msg.lower()

    @patch('chandra_align.cli.batch.parse_pds4_metadata')
    def test_process_single_pair_unrecognized_sensor(self, mock_parse):
        """Test single pair processing with unrecognized sensor."""
        mock_parse.return_value = SensorMeta(
            sensor_name="UNKNOWN",
            sensor_type="PANCHROMATIC",
            gsd_meters=1.0,
            sun_azimuth_deg=0.0,
            sun_incidence_deg=45.0,
        )
        mock_evaluate = MagicMock()
        from chandra_align.pipeline.router import RoutingDecision, MatcherStrategy
        mock_evaluate.return_value = RoutingDecision(
            primary_strategy=MatcherStrategy.LIGHTGLUE_ALIKED,
            fallback_strategy=MatcherStrategy.RIFT2_PHASE_CONGRUENCY,
            delta_azimuth_deg=0.0,
            delta_incidence_deg=0.0,
            scale_ratio=1.0,
            needs_pre_resampling=False,
            target_resample_gsd=None,
            routing_reason="Standard optical matching regime.",
        )
        from chandra_align.pipeline.router import evaluate_route
        
        with patch('chandra_align.cli.batch.evaluate_route', mock_evaluate):
            result = process_single_pair(("src.xml", "ref.xml"))
        
        assert result.status == "SUCCESS"
        assert result.primary_strategy == "LIGHTGLUE_ALIKED"

    def test_load_manifest_json_valid(self):
        """Test loading valid JSON manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            manifest = base_dir / "manifest.json"
            manifest.write_text(json.dumps([
                {"source": "img1.xml", "reference": "img2.xml"},
                {"source": "img3.xml", "reference": "img4.xml"},
            ]))
            
            pairs = load_manifest(manifest, base_dir)
            assert len(pairs) == 2
            assert pairs[0] == (base_dir / "img1.xml", base_dir / "img2.xml")
            assert pairs[1] == (base_dir / "img3.xml", base_dir / "img4.xml")

    def test_load_manifest_json_missing_keys(self):
        """Test loading JSON manifest with missing keys."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            manifest = base_dir / "manifest.json"
            manifest.write_text(json.dumps([
                {"source": "img1.xml"},  # missing 'reference'
            ]))
            
            with pytest.raises(ValueError, match="must contain 'source' and 'reference' keys"):
                load_manifest(manifest, base_dir)

    def test_load_manifest_json_invalid(self):
        """Test loading invalid JSON manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            manifest = base_dir / "manifest.json"
            manifest.write_text("{ invalid json")
            
            with pytest.raises(ValueError, match="Invalid JSON"):
                load_manifest(manifest, base_dir)

    def test_load_manifest_csv_valid(self):
        """Test loading valid CSV manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            manifest = base_dir / "manifest.csv"
            manifest.write_text("source,reference\nimg1.xml,img2.xml\nimg3.xml,img4.xml\n")
            
            pairs = load_manifest(manifest, base_dir)
            assert len(pairs) == 2

    def test_load_manifest_csv_with_comments(self):
        """Test loading CSV manifest with comments and header."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            manifest = base_dir / "manifest.csv"
            manifest.write_text("# Comment line\nsource,reference\nimg1.xml,img2.xml\n# Another comment\nimg3.xml,img4.xml\n")
            
            pairs = load_manifest(manifest, base_dir)
            assert len(pairs) == 2

    def test_load_manifest_csv_insufficient_columns(self):
        """Test loading CSV manifest with insufficient columns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            manifest = base_dir / "manifest.csv"
            manifest.write_text("source,reference\nimg1.xml\n")
            
            with pytest.raises(ValueError, match="fewer than 2 columns"):
                load_manifest(manifest, base_dir)

    def test_load_manifest_unsupported_format(self):
        """Test loading unsupported manifest format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            base_dir = Path(tmpdir)
            manifest = base_dir / "manifest.txt"
            manifest.write_text("img1.xml img2.xml")
            
            with pytest.raises(ValueError, match="Unsupported manifest format"):
                load_manifest(manifest, base_dir)

    def test_load_manifest_file_not_found(self):
        """Test loading non-existent manifest."""
        with pytest.raises(FileNotFoundError):
            load_manifest("nonexistent.json", Path("/tmp"))

    @patch('chandra_align.cli.batch.process_single_pair')
    def test_run_batch_pipeline_success(self, mock_process):
        """Test successful batch pipeline run using threads for test isolation."""
        mock_process.side_effect = [
            BatchPairResult(
                pair_id="img1_VS_img2",
                source_pds4="img1.xml",
                ref_pds4="img2.xml",
                primary_strategy="LIGHTGLUE_ALIKED",
                delta_azimuth_deg=7.0,
                scale_ratio=0.5,
                status="SUCCESS",
            ),
            BatchPairResult(
                pair_id="img3_VS_img4",
                source_pds4="img3.xml",
                ref_pds4="img4.xml",
                primary_strategy="RIFT2_PHASE_CONGRUENCY",
                delta_azimuth_deg=110.0,
                scale_ratio=0.5,
                status="SUCCESS",
            ),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            manifest = input_dir / "manifest.json"
            input_dir.mkdir()
            
            manifest.write_text(json.dumps([
                {"source": "img1.xml", "reference": "img2.xml"},
                {"source": "img3.xml", "reference": "img4.xml"},
            ]))
            
            # Use threads for test to avoid pickling issues
            results = run_batch_pipeline(manifest, input_dir, output_dir, max_workers=2, use_threads=True)
            
            assert len(results) == 2
            assert results[0].status == "SUCCESS"
            assert results[1].status == "SUCCESS"
            
            # Check reports written
            json_report = output_dir / "batch_summary.json"
            md_report = output_dir / "batch_summary.md"
            assert json_report.exists()
            assert md_report.exists()

    @patch('chandra_align.cli.batch.process_single_pair')
    def test_run_batch_pipeline_partial_failure(self, mock_process):
        """Test batch pipeline with partial failures using threads."""
        mock_process.side_effect = [
            BatchPairResult(
                pair_id="valid_VS_ref",
                source_pds4="valid.xml",
                ref_pds4="ref.xml",
                primary_strategy="LIGHTGLUE_ALIKED",
                delta_azimuth_deg=7.0,
                scale_ratio=0.5,
                status="SUCCESS",
            ),
            BatchPairResult(
                pair_id="missing_VS_ref",
                source_pds4="missing.xml",
                ref_pds4="ref.xml",
                primary_strategy="FAILED",
                delta_azimuth_deg=0.0,
                scale_ratio=1.0,
                status="FAILED",
                error_msg="PDS4 XML file not found: missing.xml",
            ),
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            manifest = input_dir / "manifest.json"
            input_dir.mkdir()
            
            manifest.write_text(json.dumps([
                {"source": "valid.xml", "reference": "ref.xml"},
                {"source": "missing.xml", "reference": "ref.xml"},
            ]))
            
            # Use threads for test to avoid pickling issues
            results = run_batch_pipeline(manifest, input_dir, output_dir, max_workers=2, use_threads=True)
            
            assert len(results) == 2
            success_count = sum(1 for r in results if r.status == "SUCCESS")
            failed_count = sum(1 for r in results if r.status == "FAILED")
            assert success_count == 1
            assert failed_count == 1

    def test_run_batch_pipeline_empty_manifest(self):
        """Test batch pipeline with empty manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_dir = Path(tmpdir) / "input"
            output_dir = Path(tmpdir) / "output"
            manifest = input_dir / "manifest.json"
            input_dir.mkdir()
            manifest.write_text(json.dumps([]))
            
            results = run_batch_pipeline(manifest, input_dir, output_dir, max_workers=2)
            assert results == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
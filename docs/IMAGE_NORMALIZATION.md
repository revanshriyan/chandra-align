# Image normalization

The loaders preserve the existing radiometry for `uint8` inputs. Higher-bit-depth and floating-point inputs are stretched before feature matching, with percentiles selected by the input path:

| Input path | Stretch |
| --- | --- |
| Direct image input in `app._read_grayscale_image` | 1st–99th percentile |
| Higher-bit-depth input in `app.load_lunar_raster` | 2nd–98th percentile |
| IIRS pair in `_align_core` | 1st–99th percentile (passed explicitly to `preprocess_iirs_raster`) |
| Direct `preprocess_iirs_raster` call without an override | 2nd–98th percentile |

This documents the runtime behavior rather than applying one stretch to every loader. The issue #1 benchmark inputs loaded through `load_lunar_raster` retain its 2nd–98th percentile behavior; 8-bit synthetic PNG inputs are unchanged.

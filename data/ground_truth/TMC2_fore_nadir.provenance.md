# TMC-2 fore/nadir ground-truth points

Status: **pending human landmark picking**. This file is an empty schema template, not a measured point set.

Benchmark inputs: `ch2_tmc_ncf_20231101T0125121344_d_img_d18` fore and `ch2_tmc_ncn_20231101T0125121377_d_img_d18` nadir; the issue #1 pair uses the audited 2048×2048 windows. Local LROC NAC tutorial stereo inputs are not documented as covering these exact fore/nadir windows, no SLDEM raster or independently projected control points were found, and the existing browse-derived window placement is not independent ground truth.

Pick at least 20 unambiguous corresponding terrain landmarks, at least 3 in each quadrant and spanning at least 60% of each crop's width and height. Use coordinates relative to the exact 2048×2048 crops, record landmark descriptions, picker/date and crop identity in a companion provenance record, and keep all picks out of the matcher/fitting inputs. CSV columns: `id,x_ref,y_ref,x_src,y_src`.

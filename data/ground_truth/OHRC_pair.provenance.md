# OHRC pair ground-truth points

Status: **pending human landmark picking**. This file is an empty schema template, not a measured point set.

Benchmark inputs: `ch2_ohr_nrp_20240425T1209509264_d_img_d18` (reference) and `ch2_ohr_nrp_20240425T1406019344_d_img_d18` (source), loaded by the issue #1 runner at a 4096-pixel cap. No local LROC NAC or SLDEM control product covering the documented approximately 69°S, 32°E benchmark area, and no independent tie-point records, were found.

Pick at least 20 unambiguous corresponding terrain landmarks, at least 3 in each quadrant and spanning at least 60% of each crop's width and height. Use original crop pixel coordinates, record landmark descriptions, picker/date and image crop identity in a companion provenance record, and keep all picks out of the matcher/fitting inputs. CSV columns: `id,x_ref,y_ref,x_src,y_src`.

# Quick 10-Line QGIS Ground-Truth Point Picking Guide

1. **Open Project**: Launch QGIS and double-click `data/qgis_picking.qgz` to load rasters and overlap grid.
2. **Target Area**: Pick points only inside the red outline (42% overlap region) across center and corners.
3. **Target Count & Spacing**: Aim for **25 points total**, with at least 1 point per $30\text{ m}$ grid cell.
4. **What to Click**: Choose sharp crater rim crests, bright boulder clusters, small crater centers, or crisp ridge crests.
5. **What to Avoid**: Never click shadow interiors, feature-less mare, features $< 5\text{ px}$, or anything near frame edges.
6. **Two-Pass Picking**: Pick every physical feature **TWICE** — once on Reference layer, once on Registered Product layer.
7. **Export CSV 1**: Select Reference layer $\rightarrow$ Export Features As $\rightarrow$ CSV as `data/heldout_ref.csv`.
8. **Export CSV 2**: Select Registered Product layer $\rightarrow$ Export Features As $\rightarrow$ CSV as `data/heldout_prod.csv`.
9. **Row Pairing Rule**: Ensure Row $N$ in `heldout_ref.csv` corresponds to the exact same feature as Row $N$ in `heldout_prod.csv`.
10. **Attribute Fields**: Assign `confidence` (1=Certain, 2=Pretty Sure, 3=Best Guess) and `feature_type` (crater_rim/boulder/crater_center/ridge).

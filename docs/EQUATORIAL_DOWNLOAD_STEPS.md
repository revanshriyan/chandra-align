# Human Equatorial OHRC Download Checklist

Follow these steps to acquire the fallback equatorial OHRC dataset from ISDC PRADAN:

1. **Access Portal**: Open [https://chmapbrowse.issdc.gov.in/](https://chmapbrowse.issdc.gov.in/) in your browser.
2. **Set Search Parameters**:
   - **Min Lat**: `5` | **Max Lat**: `10`
   - **Min Lon**: `15` | **Max Lon**: `20` *(Mare Tranquillitatis)*
   - **Product Level**: Check **Calibrated** (Level 2)
   - **Instrument**: Select **OHRC**
3. **Inspect Search Results**:
   - Click each matching product ID starting with `ch2_ohr_`.
   - Click **View PDS4 Label** (`.xml`).
4. **Verify Sun Elevation Criteria**:
   - Locate the `<Sun_Elevation>` tag in the XML label.
   - Confirm sun elevation is between **30.0° and 60.0°** (moderate sun angle, ideal for correspondence).
5. **Shortlist Candidates**: Record 2–3 candidate Product IDs with their logged sun elevations.
6. **Add to Cart & Download**: Add the shortlisted candidates to your ISSDC cart and download the `.img` data and `.xml` label files.
7. **Local File Drop**: Move/extract the downloaded files directly into:
   ```
   c:\Users\Revan\OneDrive\Desktop\SIH chandra\data\ch2_equatorial\
   ```

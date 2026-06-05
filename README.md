# Google Location History Visualizer

Supplementary code for the thesis:
"Inferring Sensitive Personal Information from Google Location History Using Data Visualization Techniques"
— Matthijs Hagedoorn

## Scripts
- **GH_location_heatmap_pins_with_information.py** — Interactive heatmap with top-15 most-visited pins (HTML output)
- **GH_temporal_patterns.py** — Temporal activity patterns: hour of day, weekday heatmap, monthly visits (PNG output)

## How to use
Each script contains step-by-step instructions in its header. In short:
1. Export your Google Location History as a JSON file (see instructions inside each script)
2. Place it in the same folder as the script and rename it to `Timeline.json`
3. Run the script with `python <scriptname>.py`

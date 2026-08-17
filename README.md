# Emboss Detection

A Python image-processing tool that detects and measures **white embossing shade** around text/logos on product labels (originally built for Jordan t-shirt labels with gray backgrounds and black text).

## What it does

1. **Detects text/symbols** on a label (dark elements on a gray background).
2. **Detects the white embossing** (highlight/halo) around that text.
3. **Measures the embossing thickness and coverage** in pixels/percentage.
4. **Compares test images against a reference image** to judge quality:
   - Less/thinner embossing than reference → ✅ PASS
   - More/thicker embossing than reference → ❌ FAIL / ⚠️ WARNING
5. **Outputs visualizations and reports**: per-image analysis charts, a comparison bar chart, and a JSON report with detailed stats.

## Project structure

- `type1/`, `type2/`, `type3/` — each folder holds a set of sample label images (`ref.jpg` + test images) and a `comparison.py` script that runs the detection/comparison pipeline on that set. Results are saved into a matching subfolder (e.g. `type1/type1/`) as `*_embossing.png`, `comparison_chart.png`, and `embossing_report.json`.
- `grid/` — `color3.py`, a separate tool for grid detection on label images (edge detection, manual ROI selection, grid-line completion, cell numbering) plus text-halo detection.

## Requirements

- Python 3
- `numpy`, `scipy`, `matplotlib`, `opencv-python`

Install with:
```bash
pip install numpy scipy matplotlib opencv-python
```

## Usage

**Embossing comparison** (run from inside a `typeN/` folder):
```bash
python comparison.py
```
This reads the reference and test images listed at the bottom of the script, analyzes each, and writes results to the output subfolder.

**Grid/halo tool**:
```bash
python grid/color3.py <input_image> [output] [method] [threshold1] [threshold2]
```
This opens an interactive window — click 4 points to define a region to mask, then it completes the grid lines, numbers the cells, and runs halo detection.

## Output

Each run produces:
- Individual visualization images showing the detected embossing (highlighted in red/cyan) with a thickness heatmap
- A comparison chart ranking all test images against the reference
- A JSON report (`embossing_report.json`) with full statistics per image



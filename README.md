# estimate-dpi

Estimate the **printing DPI** of a thermal printer from a **high-resolution scan** using FFT analysis.  
Even when printing solid black, many thermal printers produce alternating vertical or horizontal stripes.  
By scanning a small area and analyzing its frequency spectrum, this tool detects the dominant stripe spacing and infers the printer’s DPI.

---

## Features

- Detects the most likely **printing DPI** from a scan.
- Works on small cropped samples (e.g. **10×10 mm**).
- Uses a **1-D FFT** to find the strongest repeating stripe frequency.
- Supports automatic or manual axis detection.
- Snaps result to common thermal DPI values (e.g. 203, 300, 600).

---

## Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for dependency and script management.

Clone the repository and install the CLI tool:

```bash
uv tool install .
````

After installation, the command `estimate-dpi` becomes available globally.

Alternatively, you can run it without installing:

```bash
uvx --from . estimate-dpi --help
```

---

## Usage

```bash
estimate-dpi path/to/scan.png [options]
```

### Example

```bash
estimate-dpi scans/sample_crop.png --scan-dpi 4800 -v
```

### Common options

| Option                                | Description                                      | Default    |
| ------------------------              | ------------------------------------------------ | ---------- |
| `--scan-dpi <value>`                  | Scanner DPI if not embedded in metadata          | 4800       |
| `--axis <auto\|horizontal\|vertical>` | Force or auto-detect analysis axis               | `auto`     |
| `--thickness <px>`                    | Number of rows/cols averaged for noise reduction | 24         |
| `--dpi-min`, `--dpi-max`              | Expected DPI bounds                              | 100–1200   |
| `--no-snap`                           | Disable snapping to common DPI values            | —          |
| `-v`, `--verbose`                     | Print diagnostic info                            | —          |

---

## Tips for accurate results

* Use a **flatbed scanner** (not a camera).
* Scan a **10×10 mm** dark region at **4800 DPI**.
* Ensure stripes are either **vertical or horizontal**.
* Crop the image tightly around the printed area.
* If the stripes appear horizontal, use `--axis vertical`.

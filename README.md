# CALPIR

Code for *CALPIR: CALibrated Prediction Intervals and a Rejection rule for
machine-learning Cd²⁺ voltammetry*.

`kode/calpir.py` holds the method itself: **method A**, conformal prediction
intervals whose width is scaled by the disagreement among ensemble members, and
**method B**, a two-stage rejection rule made of a shape gate followed by a
novelty score. `kode/percobaan.py` holds **method C**, the multi-seed evaluation
protocol with nested selection, which runs A and B through every experiment in
the paper. `kode/pra_proses.py` turns the raw voltammograms into the arrays the
method reads, and `kode/gambar.py` draws every figure. `results/` holds the
computed results behind every number and figure in the paper.

## Run

Python 3.11. Download `01_raw_voltammograms.zip` from Zenodo,
[doi:10.5281/zenodo.22236055](https://doi.org/10.5281/zenodo.22236055), then:

```bash
pip install -r requirements.txt
cd kode
python3 pra_proses.py /path/to/01_raw_voltammograms.zip
python3 percobaan.py --cepat
python3 percobaan.py
python3 gambar.py
```

`pra_proses.py` writes to `data/`. `--cepat` runs every code path at toy size
first. Without it, experiments whose result file already exists in `results/`
are skipped, so `gambar.py` reproduces the figures straight away; add `--ulang`
to recompute them. Figures 1 to 3 are written as draw.io files. Code comments
are in Indonesian.

## Revision scripts

Each reads the same preprocessed arrays and runs on its own.

| Script | Result file | Reports |
|---|---|---|
| `uji_rentang.py` | `hasil_rentang_linier.json` | Linear working range |
| `uji_lod.py` | `hasil_lod_blanko.json` | Detection and quantification limits |
| `uji_lod_stabil.py` | `hasil_lod_stabil.json` | The limits after unequilibrated blanks are set aside |
| `uji_lod_metode.py` | `hasil_lod_metode.json` | A limit in the method's own terms |
| `diagnosa_blanko.py` | `diagnosa_blanko.json` | Every blank replicate down to its 10 scans |
| `uji_ambang.py` | `hasil_uji_ambang.json` | Joint sweep of the 2 rejection thresholds |
| `uji_cakupan.py` | `hasil_cakupan_level.json` | Level-wise coverage and exceedance errors |
| `uji_gangguan.py` | `hasil_uji_gangguan.json` | Baseline shift and added noise |
| `uji_gangguan_level.py` | `hasil_gangguan_level.json` | The same perturbations per level |
| `uji_skala_gangguan.py` | `hasil_skala_gangguan.json` | Whether disagreement scaling slows degradation |

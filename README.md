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
[doi:10.5281/zenodo.22236054](https://doi.org/10.5281/zenodo.22236054), then:

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

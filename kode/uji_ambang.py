"""Optimasi ambang aturan penolakan dua tahap.

Reviewer 1 butir 4 meminta penjelasan optimasi ambang supaya penolakan palsu
berkurang. Reviewer 2 butir 4 menyatakan sasaran penolakan 5 persen tetapi
sekitar 15 persen pengukuran dalam jangkauan ikut tertolak, sebab kedua tahap
bekerja berurutan, lalu meminta biayanya dikuantifikasi dan ambangnya
dioptimalkan dengan kriteria keputusan yang jelas.

Skrip ini menyapu pasangan ambang (buang, beta) pada protokol satu tingkat
disembunyikan yang sama dengan percobaan penjaga, lalu mencatat tiga hal untuk
tiap pasangan, yaitu penolakan dalam jangkauan, penolakan di luar kalibrasi, dan
cakupan sisa yang diterima.

Sebab kedua tahap bekerja berurutan, penolakan dalam jangkauan yang diharapkan
adalah 1 - (1 - buang)(1 - beta), bukan beta saja. Pasangan baku (0,10 dan 0,05)
karena itu menghasilkan sekitar 0,145, bukan 0,05.

Kriteria keputusan yang dipakai, yaitu tahan penolakan dalam jangkauan pada
anggaran yang dinyatakan lalu pilih pasangan yang paling banyak menolak
pengukuran di luar kalibrasi.

Keluaran ditulis ke results/hasil_uji_ambang.json.

Cara pakai: python3 uji_ambang.py
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calpir import ALAT, ALPHA, HASIL, ambang, belah, jarak_knn, muat, snv

BUANG = (0.00, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20)
BETA = (0.01, 0.02, 0.05, 0.08, 0.10, 0.15)
ANGGARAN = 0.05          # penolakan dalam jangkauan yang dinyatakan naskah
BAKU = (0.10, 0.05)      # pasangan yang dipakai naskah saat ini


def satu_level(X, y, lv_out, rng):
    """Jarak dan sisa satu tingkat disembunyikan, siap diambangi berkali-kali."""
    level = np.unique(y)
    sisa = level[level != lv_out]
    i_fit, i_kal, i_in = belah(y, sisa, rng, 25, 15, 10)
    i_out = np.flatnonzero(y == lv_out)

    Zf = snv(X[i_fit])
    d_bentuk = {"fit": jarak_knn(Zf, Zf, sama=True)}
    d_mentah = {}
    for nm, idx in (("kal", i_kal), ("in", i_in), ("out", i_out)):
        d_bentuk[nm] = jarak_knn(snv(X[idx]), Zf)
        d_mentah[nm] = jarak_knn(X[idx], X[i_fit])

    reg = RandomForestRegressor(n_estimators=300, random_state=0,
                                n_jobs=-1).fit(X[i_fit], y[i_fit].astype(float))
    sisa_abs = {nm: np.abs(y[idx] - reg.predict(X[idx]))
                for nm, idx in (("kal", i_kal), ("in", i_in))}
    return d_bentuk, d_mentah, sisa_abs


def nilai_pasangan(d_bentuk, d_mentah, sisa_abs, buang, beta):
    """Penolakan dalam jangkauan, penolakan di luar, dan cakupan yang diterima."""
    if buang > 0:
        batas = float(np.percentile(d_bentuk["fit"], 100 * (1 - buang)))
        lolos = {nm: d_bentuk[nm] <= batas for nm in ("kal", "in", "out")}
    else:
        lolos = {nm: np.ones(d_bentuk[nm].size, bool) for nm in ("kal", "in", "out")}

    if not lolos["kal"].any():
        return None
    amb = float(np.percentile(d_mentah["kal"][lolos["kal"]], 100 * (1 - beta)))

    tolak = {}
    for nm in ("in", "out"):
        n = d_mentah[nm].size
        tahap2 = int((d_mentah[nm][lolos[nm]] > amb).sum())
        tolak[nm] = (n - int(lolos[nm].sum()) + tahap2) / n

    terima_kal = lolos["kal"] & (d_mentah["kal"] <= amb)
    terima_in = lolos["in"] & (d_mentah["in"] <= amb)
    if not terima_kal.any() or not terima_in.any():
        return None
    q = ambang(sisa_abs["kal"][terima_kal], ALPHA)
    cakupan = float(np.mean(sisa_abs["in"][terima_in] <= q))
    return tolak["in"], tolak["out"], cakupan


def satu_alat(X, y, level):
    """Rata-ratakan tiap pasangan ambang atas 15 tingkat yang disembunyikan."""
    rng = np.random.default_rng(42)
    bahan = []
    for lv in level:
        bahan.append(satu_level(X, y, lv, rng))
        print(f"  tingkat {lv:>4} siap", flush=True)

    kisi = {}
    for bu in BUANG:
        for be in BETA:
            nilai = [nilai_pasangan(*b, bu, be) for b in bahan]
            nilai = [v for v in nilai if v is not None]
            if not nilai:
                continue
            a = np.array(nilai)
            kisi[f"{bu}|{be}"] = {
                "buang": bu, "beta": be,
                "tolak_in": float(a[:, 0].mean()),
                "tolak_out": float(a[:, 1].mean()),
                "cakupan_diterima": float(a[:, 2].mean()),
                "tolak_in_diharapkan": 1 - (1 - bu) * (1 - be),
                "n_tingkat": int(a.shape[0]),
            }
    return kisi


def pilih(kisi, anggaran=ANGGARAN):
    """Kriteria keputusan, yaitu tahan penolakan dalam jangkauan pada anggaran
    lalu ambil penolakan di luar kalibrasi yang tertinggi."""
    layak = [v for v in kisi.values() if v["tolak_in"] <= anggaran]
    if not layak:
        return None
    return max(layak, key=lambda v: v["tolak_out"])


if __name__ == "__main__":
    t0 = time.time()
    data = muat()
    keluar = {}
    for alat in ALAT:
        print(alat, flush=True)
        X, y, t, level = data[alat]
        kisi = satu_alat(X, y, level)
        baku = kisi[f"{BAKU[0]}|{BAKU[1]}"]
        keluar[alat] = {"kisi": kisi, "baku": baku, "terpilih": pilih(kisi)}
    keluar["catatan"] = {"alpha": ALPHA, "anggaran": ANGGARAN,
                         "baku": list(BAKU), "buang": list(BUANG),
                         "beta": list(BETA),
                         "kriteria": "tahan tolak_in <= anggaran, maksimalkan tolak_out",
                         "menit": round((time.time() - t0) / 60, 2)}
    p = os.path.join(HASIL, "hasil_uji_ambang.json")
    json.dump(keluar, open(p, "w"), indent=1)
    print("ditulis", p)

    for alat in ALAT:
        h = keluar[alat]
        print(f"\n{alat}")
        for nm in ("baku", "terpilih"):
            v = h[nm]
            if v is None:
                print(f"  {nm}: tidak ada pasangan yang memenuhi anggaran")
                continue
            print(f"  {nm:9s} buang={v['buang']:.2f} beta={v['beta']:.2f} -> "
                  f"tolak_in {v['tolak_in']:.3f} (diharapkan {v['tolak_in_diharapkan']:.3f}), "
                  f"tolak_out {v['tolak_out']:.3f}, cakupan {v['cakupan_diterima']:.3f}")

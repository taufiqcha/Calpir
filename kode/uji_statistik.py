"""Uji statistik untuk klaim utama CALPIR.

Protokol A pada percobaan.py hanya menyimpan rata-rata, simpangan, minimum, dan
maksimum. Nilai per benih dibuang, sehingga klaim utama tidak punya uji apa pun.
Skrip ini mengulang protokol A dengan benih, partisi, dan model yang persis sama,
menyimpan nilai per benih, lalu menjalankan dua uji berpasangan.

  H0 lebar   median selisih lebar antara konstruksi berskala dan konstruksi tanpa
             skala sama dengan nol
  H1 lebar   konstruksi berskala lebih sempit

  H0 cakupan median cakupan empiris sama dengan tingkat nominal 0,90
  H1 cakupan median cakupan empiris berbeda dari 0,90

Keluaran ditulis ke results/hasil_uji_klaim.json.

Cara pakai: python3 uji_statistik.py
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calpir
from calpir import ALAT, ALPHA, HASIL, belah, konformal, muat, skala_sebar

N_BENIH = 20
NOMINAL = 1 - ALPHA


def protokol_a(data):
    """Ulangi protokol A dan simpan nilai tiap benih, bukan hanya ringkasannya."""
    keluar = {}
    for alat in ALAT:
        X, y, t, level = data[alat]
        per_benih = {"global": {"cakupan": [], "lebar": []},
                     "ternormalkan": {"cakupan": [], "lebar": []}}
        for b in range(N_BENIH):
            rng = np.random.default_rng(1000 + b)
            i_fit, i_kal, i_uji = belah(y, level, rng, 20, 20, 10)
            rf = RandomForestRegressor(n_estimators=300, random_state=b,
                                       n_jobs=-1).fit(X[i_fit], t[i_fit])
            pk, pu = rf.predict(X[i_kal]), rf.predict(X[i_uji])
            sk, su = (skala_sebar(rf, X, i, lantai=1e-6) for i in (i_kal, i_uji))
            c_g, l_g = konformal(t[i_kal], pk, t[i_uji], pu)
            c_n, l_n = konformal(t[i_kal], pk, t[i_uji], pu, sk, su)
            per_benih["global"]["cakupan"].append(float(c_g))
            per_benih["global"]["lebar"].append(float(l_g))
            per_benih["ternormalkan"]["cakupan"].append(float(c_n))
            per_benih["ternormalkan"]["lebar"].append(float(l_n))
            print(f"  {alat} benih {b}", flush=True)
        keluar[alat] = per_benih
    return keluar


def uji(per_benih):
    """Dua uji berpasangan Wilcoxon untuk tiap alat."""
    hasil = {}
    for alat, d in per_benih.items():
        lg = np.array(d["global"]["lebar"])
        ln = np.array(d["ternormalkan"]["lebar"])
        cn = np.array(d["ternormalkan"]["cakupan"])
        W1, p1 = wilcoxon(ln, lg, alternative="less")
        W2, p2 = wilcoxon(cn - NOMINAL, alternative="two-sided")
        hasil[alat] = {
            "n_benih": int(len(lg)),
            "lebar_tanpa_skala": {"rata": float(lg.mean()), "sd": float(lg.std(ddof=1)),
                                  "min": float(lg.min()), "maks": float(lg.max()),
                                  "nilai": [float(v) for v in lg]},
            "lebar_berskala": {"rata": float(ln.mean()), "sd": float(ln.std(ddof=1)),
                               "min": float(ln.min()), "maks": float(ln.max()),
                               "nilai": [float(v) for v in ln]},
            "cakupan_berskala": {"rata": float(cn.mean()), "sd": float(cn.std(ddof=1)),
                                 "nilai": [float(v) for v in cn]},
            "uji_lebar": {"nama": "Wilcoxon signed-rank, satu sisi",
                          "W": float(W1), "p": float(p1),
                          "benih_berskala_lebih_sempit": int(np.sum(ln < lg)),
                          "rasio_median": float(np.median(lg) / np.median(ln))},
            "uji_cakupan": {"nama": "Wilcoxon signed-rank, dua sisi, terhadap 0,90",
                            "W": float(W2), "p": float(p2),
                            "selisih_median": float(np.median(cn - NOMINAL))},
        }
    return hasil


if __name__ == "__main__":
    t0 = time.time()
    hasil = uji(protokol_a(muat()))
    hasil["catatan"] = {
        "nominal": NOMINAL, "n_benih": N_BENIH,
        "satuan_lebar": "desimal log10 ppm",
        "menit": round((time.time() - t0) / 60, 2),
    }
    p = os.path.join(HASIL, "hasil_uji_klaim.json")
    json.dump(hasil, open(p, "w"), indent=1)
    print("ditulis", p)
    for alat in ALAT:
        h = hasil[alat]
        print(f"{alat}: lebar {h['lebar_tanpa_skala']['rata']:.4f} -> "
              f"{h['lebar_berskala']['rata']:.4f}, W={h['uji_lebar']['W']:.0f}, "
              f"p={h['uji_lebar']['p']:.2e}, "
              f"cakupan {h['cakupan_berskala']['rata']:.4f}, "
              f"p={h['uji_cakupan']['p']:.3f}")

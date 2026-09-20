"""Uji berpasangan lebar interval, CALPIR lawan deep ensemble yang ditala.

Naskah mengklaim CALPIR lebih sempit daripada deep ensemble yang ditala lewat
pencarian 40 arsitektur, tetapi klaim itu hanya berupa rata-rata atas 5 benih
tanpa uji apa pun. Skrip ini menaikkan ulangan menjadi 20 benih, menghitung
kedua metode pada partisi yang sama persis di tiap benih, lalu menguji selisih
lebarnya secara berpasangan.

  H0  median selisih lebar antara CALPIR dan deep ensemble sama dengan nol
  H1  CALPIR lebih sempit

Arsitektur terbaik tidak dicari ulang. Skrip memakai hasil pencarian yang sudah
tersimpan di results/besar_tahap2.json, sehingga yang dijalankan hanya
pelatihan ulangnya.

Keluaran ditulis ke results/hasil_uji_deep.json.

Cara pakai: python3 uji_deep.py
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import wilcoxon
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calpir
from calpir import ALAT, ALPHA, HASIL, belah, konformal, muat, skala_sebar

N_BENIH = 20
N_ANGGOTA = 5
EPOCH = 200
ALFA = (0.05, 0.10, 0.20)   # tiga tingkat nominal, sama dengan Table 5


def arsitektur_terbaik():
    """Ambil arsitektur hasil pencarian 40 calon yang sudah tersimpan."""
    p = os.path.join(HASIL, "besar_tahap2.json")
    return {a: d["terbaik"] for a, d in json.load(open(p))["hasil"].items()}


def satu_benih(X, t, i_fit, i_kal, i_uji, benih, arsitektur):
    """Lebar interval kedua metode pada satu partisi yang sama."""
    et = ExtraTreesRegressor(n_estimators=300, random_state=benih,
                             n_jobs=-1).fit(X[i_fit], t[i_fit])
    pk, pu = et.predict(X[i_kal]), et.predict(X[i_uji])
    sk, su = (skala_sebar(et, X, i, lantai=1e-6) for i in (i_kal, i_uji))
    calpir_per_alfa = {str(a): konformal(t[i_kal], pk, t[i_uji], pu, sk, su, a)
                       for a in ALFA}

    sc = StandardScaler().fit(X[i_fit])
    Xf, Xk, Xu = (sc.transform(X[i]) for i in (i_fit, i_kal, i_uji))
    mk, mu, vk, vu = [], [], [], []
    for a in range(N_ANGGOTA):
        m = calpir.latih_gauss(
            calpir.jaringan_gauss(X.shape[1],
                                  (arsitektur["unit"],) * arsitektur["lapis"],
                                  arsitektur["dropout"], arsitektur["lr"],
                                  10 * benih + a),
            Xf, t[i_fit], 10 * benih + a, EPOCH)
        ok, ou = m.predict(Xk, verbose=0), m.predict(Xu, verbose=0)
        mk.append(ok[:, 0]); vk.append(ok[:, 1] ** 2)
        mu.append(ou[:, 0]); vu.append(ou[:, 1] ** 2)
    Mk, Mu = np.mean(mk, 0), np.mean(mu, 0)
    Sk = np.sqrt(np.mean(vk, 0) + np.var(mk, 0)) + 1e-6
    Su = np.sqrt(np.mean(vu, 0) + np.var(mu, 0)) + 1e-6
    deep_per_alfa = {str(a): konformal(t[i_kal], Mk, t[i_uji], Mu, Sk, Su, a)
                     for a in ALFA}
    return calpir_per_alfa, deep_per_alfa


def jalankan():
    data = muat()
    ars = arsitektur_terbaik()
    keluar = {}
    for alat in ALAT:
        X, y, t, level = data[alat]
        nilai = {str(a): {"calpir_cakupan": [], "calpir_lebar": [],
                          "deep_cakupan": [], "deep_lebar": []} for a in ALFA}
        for b in range(N_BENIH):
            # benih 600 + b memakai partisi yang sama dengan percobaan ensemble
            rng = np.random.default_rng(600 + b)
            i_fit, i_kal, i_uji = belah(y, level, rng, 20, 20, 10)
            cal, dep = satu_benih(X, t, i_fit, i_kal, i_uji, b, ars[alat])
            for a in ALFA:
                k = str(a)
                nilai[k]["calpir_cakupan"].append(float(cal[k][0]))
                nilai[k]["calpir_lebar"].append(float(cal[k][1]))
                nilai[k]["deep_cakupan"].append(float(dep[k][0]))
                nilai[k]["deep_lebar"].append(float(dep[k][1]))
            print(f"  {alat} benih {b}: CALPIR {cal['0.1'][1]:.4f}  "
                  f"deep {dep['0.1'][1]:.4f}", flush=True)

        per_alfa = {}
        for a in ALFA:
            k = str(a)
            lc = np.array(nilai[k]["calpir_lebar"]); ld = np.array(nilai[k]["deep_lebar"])
            W, p = wilcoxon(lc, ld, alternative="less")
            per_alfa[k] = {
                "nominal": round(1 - a, 2),
                "calpir": {"lebar_rata": float(lc.mean()), "lebar_sd": float(lc.std(ddof=1)),
                           "cakupan_rata": float(np.mean(nilai[k]["calpir_cakupan"]))},
                "deep": {"lebar_rata": float(ld.mean()), "lebar_sd": float(ld.std(ddof=1)),
                         "cakupan_rata": float(np.mean(nilai[k]["deep_cakupan"]))},
                "uji": {"nama": "Wilcoxon signed-rank, satu sisi",
                        "W": float(W), "p": float(p),
                        "benih_calpir_lebih_sempit": int(np.sum(lc < ld)),
                        "rasio_median": float(np.median(ld) / np.median(lc))},
            }
        keluar[alat] = {"arsitektur": ars[alat], "n_benih": N_BENIH,
                        "nilai": nilai, "per_alfa": per_alfa}
    return keluar


if __name__ == "__main__":
    t0 = time.time()
    hasil = jalankan()
    hasil["catatan"] = {"alpha": ALPHA, "n_benih": N_BENIH,
                        "n_anggota": N_ANGGOTA, "epoch": EPOCH, "alfa": list(ALFA),
                        "satuan_lebar": "desimal log10 ppm",
                        "menit": round((time.time() - t0) / 60, 2)}
    p = os.path.join(HASIL, "hasil_uji_deep.json")
    json.dump(hasil, open(p, "w"), indent=1)
    print("ditulis", p)
    for alat in ALAT:
        for k, h in hasil[alat]["per_alfa"].items():
            print(f"{alat} nominal {h['nominal']}: CALPIR {h['calpir']['lebar_rata']:.4f} "
                  f"lawan deep {h['deep']['lebar_rata']:.4f}, rasio median "
                  f"{h['uji']['rasio_median']:.2f}, W={h['uji']['W']:.0f}, "
                  f"p={h['uji']['p']:.3e}, "
                  f"{h['uji']['benih_calpir_lebih_sempit']}/{N_BENIH} benih")

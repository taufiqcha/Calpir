"""Ketahanan terhadap gangguan, dipisah menurut tingkat kadar.

Pertanyaannya bukan sekadar tahan atau tidak tahan, melainkan sampai di mana
ketahanan itu berlaku. Pada kadar rendah pola voltamogram samar, sehingga
geseran kecil pun dapat menenggelamkannya. Pada kadar tinggi polanya jelas,
sehingga geseran yang sama belum tentu berpengaruh.

Skrip ini mengulang percobaan gangguan, tetapi mencatat cakupan dan penolakan
untuk tiap tingkat kadar secara terpisah. Hasilnya memungkinkan ketahanan
dinyatakan sebagai rentang kerja, sama seperti rentang linier yang dipakai
untuk batas deteksi, bukan sebagai satu keputusan tahan atau jebol.

Taraf gangguan diambil dari sebaran garis dasar blanko alat yang bersangkutan,
sehingga seluruh taraf berada di dalam jangkauan yang memang terjadi.

Keluaran ditulis ke results/hasil_gangguan_level.json.

Cara pakai: python3 uji_gangguan_level.py
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calpir import (ALAT, ALPHA, BETA, BUANG_GERBANG, HASIL, ambang, belah,
                    muat, penjaga_dua_tahap, skala_sebar)
from uji_gangguan import ganggu, satuan

N_BENIH = 20
TARAF = (0.0, 0.2, 0.5, 1.0, 2.0)      # dalam satuan sebaran garis dasar nyata
BENTUK = ("geseran", "derau")


def satu_benih(X, t, y, level, benih, s0):
    """Cakupan, lebar, penolakan tiap tingkat kadar pada tiap taraf gangguan."""
    rng = np.random.default_rng(1000 + benih)
    i_fit, i_kal, i_uji = belah(y, level, rng, 20, 20, 10)
    rf = RandomForestRegressor(n_estimators=300, random_state=benih,
                               n_jobs=-1).fit(X[i_fit], t[i_fit])
    p_kal = rf.predict(X[i_kal])
    s_kal = skala_sebar(rf, X, i_kal, lantai=1e-6)
    q = ambang(np.abs(t[i_kal] - p_kal) / s_kal, ALPHA)

    acak = np.random.default_rng(5000 + benih)
    keluar = {}
    for bentuk in BENTUK:
        for taraf in TARAF:
            Xg = ganggu(X[i_uji], bentuk, taraf * s0, acak)
            p = rf.predict(Xg)
            s = np.std([e.predict(Xg) for e in rf.estimators_], axis=0) + 1e-6
            bawah, atas = p - q * s, p + q * s
            tutup = (t[i_uji] >= bawah) & (t[i_uji] <= atas)

            _, lolos_g, tolak2, _ = penjaga_dua_tahap(
                X[i_fit], X[i_kal], Xg, BETA, BUANG_GERBANG)
            ditolak = np.ones(Xg.shape[0], bool)
            idx = np.flatnonzero(lolos_g)
            if idx.size:
                ditolak[idx[~tolak2]] = False

            for lv in level:
                m = y[i_uji] == lv
                if not m.any():
                    continue
                keluar.setdefault(f"{bentuk}|{taraf}|{int(lv)}", []).append(
                    (float(tutup[m].mean()), float(np.median(atas[m] - bawah[m])),
                     float(ditolak[m].mean())))
    return keluar


def main():
    data = muat()
    keluar = {}
    for alat in ALAT:
        X, y, t, level = data[alat]
        s0 = satuan(X, y)
        kumpul = {}
        for b in range(N_BENIH):
            for k, v in satu_benih(X, t, y, level, b, s0).items():
                kumpul.setdefault(k, []).extend(v)
            print(f"  {alat} benih {b} selesai", flush=True)
        ring = {}
        for k, daftar in kumpul.items():
            a = np.array(daftar)
            ring[k] = {"cakupan": float(a[:, 0].mean()),
                       "lebar": float(a[:, 1].mean()),
                       "tolak": float(a[:, 2].mean())}
        keluar[alat] = {"satuan_gangguan": s0, "per_level": ring}
    keluar["catatan"] = {"alpha": ALPHA, "n_benih": N_BENIH, "taraf": list(TARAF),
                         "bentuk": list(BENTUK),
                         "satuan": "kelipatan simpangan baku di dalam tingkat kadar",
                         "catatan_uji": "gangguan hanya dikenakan pada himpunan uji"}
    p = os.path.join(HASIL, "hasil_gangguan_level.json")
    json.dump(keluar, open(p, "w"), indent=1)
    print("ditulis", p)

    for alat in ALAT:
        lv = sorted({int(k.split("|")[2]) for k in keluar[alat]["per_level"]})
        for bentuk in BENTUK:
            print(f"\n{alat}  {bentuk}, cakupan tiap tingkat kadar")
            print("  kadar  " + "".join(f"{t:>8.1f}" for t in TARAF))
            for v in lv:
                baris = [keluar[alat]["per_level"].get(f"{bentuk}|{t}|{v}")
                         for t in TARAF]
                print(f"  {v:>5}  " + "".join(
                    "       -" if b is None else f"{b['cakupan']:8.3f}" for b in baris))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"selesai dalam {(time.time()-t0)/60:.1f} menit")

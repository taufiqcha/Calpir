"""Apakah penskalaan sebaran ensembel memperlambat kerusakan saat terganggu.

Naskah mengklaim penskalaan sebaran ensembel mempersempit selang pada data
bersih. Klaim itu tidak menyinggung ketahanan sama sekali. Skrip ini menguji
apakah penskalaan itu juga membawa keuntungan kedua yang belum diklaim, yaitu
cakupan yang meluruh lebih lambat ketika pengukuran terganggu.

Alasan menduga demikian bersifat mekanis. Pengukuran yang terganggu membuat
anggota ensembel saling berbeda lebih jauh, sehingga skala setempat membesar
dan selang melebar sendiri. Selang tanpa skala memakai satu lebar untuk semua
pengukuran, sehingga tidak dapat menanggapi gangguan.

Kedua konstruksi dihitung pada partisi, model, dan gangguan yang sama persis,
sehingga selisihnya hanya berasal dari penskalaannya.

  H0  median selisih cakupan antara konstruksi berskala dan tanpa skala pada
      taraf gangguan yang sama adalah nol
  H1  konstruksi berskala mempertahankan cakupan lebih tinggi

Keluaran ditulis ke results/hasil_skala_gangguan.json.

Cara pakai: python3 uji_skala_gangguan.py
"""
import json
import os
import sys
import time

import numpy as np
from scipy.stats import wilcoxon
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calpir import ALAT, ALPHA, HASIL, ambang, belah, muat, skala_sebar
from uji_gangguan import ganggu, satuan

N_BENIH = 20
TARAF = (0.0, 0.2, 0.5, 1.0, 2.0)
BENTUK = ("geseran", "derau")


def satu_benih(X, t, y, level, benih, s0):
    """Cakupan dan lebar kedua konstruksi pada tiap taraf gangguan."""
    rng = np.random.default_rng(1000 + benih)
    i_fit, i_kal, i_uji = belah(y, level, rng, 20, 20, 10)
    rf = RandomForestRegressor(n_estimators=300, random_state=benih,
                               n_jobs=-1).fit(X[i_fit], t[i_fit])
    p_kal = rf.predict(X[i_kal])
    s_kal = skala_sebar(rf, X, i_kal, lantai=1e-6)
    sisa = np.abs(t[i_kal] - p_kal)
    q_skala = ambang(sisa / s_kal, ALPHA)
    q_polos = ambang(sisa, ALPHA)

    acak = np.random.default_rng(5000 + benih)
    keluar = {}
    for bentuk in BENTUK:
        for taraf in TARAF:
            Xg = ganggu(X[i_uji], bentuk, taraf * s0, acak)
            p = rf.predict(Xg)
            s = np.std([e.predict(Xg) for e in rf.estimators_], axis=0) + 1e-6
            galat = np.abs(t[i_uji] - p)
            keluar[f"{bentuk}|{taraf}"] = {
                "skala_cakupan": float(np.mean(galat <= q_skala * s)),
                "skala_lebar": float(np.median(2 * q_skala * s)),
                "polos_cakupan": float(np.mean(galat <= q_polos)),
                "polos_lebar": float(2 * q_polos),
                "sebar_rata": float(s.mean()),
            }
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
                kumpul.setdefault(k, []).append(v)
            print(f"  {alat} benih {b} selesai", flush=True)

        ring = {}
        for k, daftar in kumpul.items():
            cs = np.array([d["skala_cakupan"] for d in daftar])
            cp = np.array([d["polos_cakupan"] for d in daftar])
            selisih = cs - cp
            if np.allclose(selisih, 0):
                W, pv = 0.0, 1.0
            else:
                W, pv = wilcoxon(cs, cp, alternative="greater")
            ring[k] = {
                "skala_cakupan": float(cs.mean()),
                "polos_cakupan": float(cp.mean()),
                "selisih_cakupan": float(selisih.mean()),
                "skala_lebar": float(np.mean([d["skala_lebar"] for d in daftar])),
                "polos_lebar": float(np.mean([d["polos_lebar"] for d in daftar])),
                "sebar_rata": float(np.mean([d["sebar_rata"] for d in daftar])),
                "W": float(W), "p": float(pv),
                "benih_skala_lebih_tinggi": int(np.sum(cs > cp)),
            }
        keluar[alat] = {"satuan_gangguan": s0, "taraf": ring}

    keluar["catatan"] = {"alpha": ALPHA, "n_benih": N_BENIH, "taraf": list(TARAF),
                         "bentuk": list(BENTUK),
                         "uji": "Wilcoxon signed-rank, satu sisi, berskala lebih tinggi",
                         "catatan_uji": "gangguan hanya dikenakan pada himpunan uji"}
    p = os.path.join(HASIL, "hasil_skala_gangguan.json")
    json.dump(keluar, open(p, "w"), indent=1)
    print("ditulis", p)

    for alat in ALAT:
        print(f"\n{'='*86}\n{alat}\n{'='*86}")
        print(f"  {'bentuk':9s} {'taraf':>6} {'cak skala':>10} {'cak polos':>10} "
              f"{'selisih':>8} {'lbr skala':>10} {'lbr polos':>10} {'p':>9} {'benih':>6}")
        for bentuk in BENTUK:
            for taraf in TARAF:
                v = keluar[alat]["taraf"][f"{bentuk}|{taraf}"]
                print(f"  {bentuk:9s} {taraf:6.1f} {v['skala_cakupan']:10.3f} "
                      f"{v['polos_cakupan']:10.3f} {v['selisih_cakupan']:+8.3f} "
                      f"{v['skala_lebar']:10.3f} {v['polos_lebar']:10.3f} "
                      f"{v['p']:9.2e} {v['benih_skala_lebih_tinggi']:>4}/20")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"selesai dalam {(time.time()-t0)/60:.1f} menit")

"""Ketahanan terhadap geseran garis dasar dan terhadap derau.

Reviewer 1 butir 3 menyatakan bahwa naskah memakai voltamogram mentah, lalu
meminta pengaruh geseran arus dasar dan derau terhadap kinerja diperiksa lebih
rinci.

Model dilatih dan dikalibrasi pada data bersih, sedangkan gangguan hanya
dikenakan pada himpunan uji. Urutan itu meniru pemakaian sesungguhnya, yaitu
model dikalibrasi di laboratorium lalu menerima pengukuran yang garis dasarnya
sudah bergeser.

Tiga bentuk gangguan dikenakan terpisah.

  geseran  satu tetapan ditambahkan ke seluruh peubah
  kemiringan  tanjakan lurus sepanjang sumbu potensial, berpusat di nol
  derau    bilangan acak normal ditambahkan ke tiap peubah

Besarnya dinyatakan sebagai kelipatan simpangan baku di dalam tingkat kadar,
sehingga satu kelipatan berarti gangguan sebesar ragam alami data itu sendiri.

Untuk tiap taraf dicatat empat besaran, yaitu cakupan, lebar selang, laju
penolakan oleh aturan dua tahap, lalu cakupan di antara yang diterima. Besaran
terakhir memperlihatkan apakah aturan penolakan menyelamatkan jaminan ketika
gangguan membesar.

Keluaran ditulis ke results/hasil_uji_gangguan.json.

Cara pakai: python3 uji_gangguan.py
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

N_BENIH = 20
TARAF = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
BENTUK = ("geseran", "kemiringan", "derau")


def satuan(X, y):
    """Simpangan baku khas di dalam satu tingkat kadar, dirata atas peubah."""
    sd = [X[y == lv].std(axis=0, ddof=1) for lv in np.unique(y)]
    return float(np.median(np.mean(sd, axis=0)))


def ganggu(X, bentuk, besar, rng):
    """Kenakan satu bentuk gangguan pada seluruh baris."""
    if besar == 0.0:
        return X.copy()
    if bentuk == "geseran":
        return X + besar
    if bentuk == "kemiringan":
        tanjak = np.linspace(-1.0, 1.0, X.shape[1])
        return X + besar * tanjak
    return X + rng.normal(0.0, besar, size=X.shape)


def satu_benih(X, t, y, level, benih, s0):
    """Latih sekali pada data bersih, lalu uji pada seluruh taraf gangguan."""
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

            tolak, lolos_g, tolak2, _ = penjaga_dua_tahap(
                X[i_fit], X[i_kal], Xg, BETA, BUANG_GERBANG)
            terima = np.zeros(Xg.shape[0], bool)
            idx = np.flatnonzero(lolos_g)
            if idx.size:
                terima[idx[~tolak2]] = True

            keluar[f"{bentuk}|{taraf}"] = {
                "cakupan": float(tutup.mean()),
                "lebar": float(np.median(atas - bawah)),
                "tolak": float(tolak),
                "cakupan_diterima": (float(tutup[terima].mean())
                                     if terima.any() else None),
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
            ct = [d["cakupan_diterima"] for d in daftar if d["cakupan_diterima"] is not None]
            ring[k] = {
                "cakupan": float(np.mean([d["cakupan"] for d in daftar])),
                "lebar": float(np.mean([d["lebar"] for d in daftar])),
                "tolak": float(np.mean([d["tolak"] for d in daftar])),
                "cakupan_diterima": float(np.mean(ct)) if ct else None,
                "n_benih_ada_diterima": len(ct),
            }
        keluar[alat] = {"satuan_gangguan": s0, "taraf": ring}
    keluar["catatan"] = {"alpha": ALPHA, "beta": BETA,
                         "buang_gerbang": BUANG_GERBANG, "n_benih": N_BENIH,
                         "taraf": list(TARAF), "bentuk": list(BENTUK),
                         "satuan": "kelipatan simpangan baku di dalam tingkat kadar",
                         "catatan_uji": "gangguan hanya dikenakan pada himpunan uji"}
    p = os.path.join(HASIL, "hasil_uji_gangguan.json")
    json.dump(keluar, open(p, "w"), indent=1)
    print("ditulis", p)

    for alat in ALAT:
        h = keluar[alat]
        print(f"\n{'='*74}\n{alat}   satu satuan gangguan = {h['satuan_gangguan']:.4f}\n{'='*74}")
        print(f"  {'bentuk':11s} {'taraf':>6} {'cakupan':>8} {'lebar':>7} "
              f"{'tolak':>7} {'cakupan diterima':>17}")
        for bentuk in BENTUK:
            for taraf in TARAF:
                v = h["taraf"][f"{bentuk}|{taraf}"]
                cd = "-" if v["cakupan_diterima"] is None else f"{v['cakupan_diterima']:.3f}"
                print(f"  {bentuk:11s} {taraf:6.2f} {v['cakupan']:8.3f} "
                      f"{v['lebar']:7.3f} {v['tolak']:7.3f} {cd:>17}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"selesai dalam {(time.time()-t0)/60:.1f} menit")

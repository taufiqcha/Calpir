"""Batas deteksi menurut metode, bukan menurut kurva kalibrasi univariat.

Kaidah tiga sigma menilai satu tanggapan univariat. CALPIR tidak bekerja begitu,
sebab ia melaporkan selang terkalibrasi dari seluruh sapuan. Maka batas
deteksinya perlu dinyatakan dalam bahasanya sendiri, yaitu kadar terendah yang
selangnya sudah terpisah dari selang yang dilaporkan untuk elektrolit blanko.

Dua besaran dihitung untuk tiap tingkat kadar.

  1. Pemisahan terhadap blanko, yaitu pecahan replikat tingkat itu yang batas
     bawah selangnya melampaui batas atas median selang blanko.
  2. Lebar selang dalam faktor pengali kadar, supaya terbaca sebagai mutu
     kuantifikasi di tingkat itu.

Keluaran ditulis ke results/hasil_lod_metode.json.

Cara pakai: python3 uji_lod_metode.py [berkas_arsip.zip]
"""
import json
import os
import sys
import zipfile

import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calpir import (ALAT, ALPHA, DATA, HASIL, ambang, belah, muat, skala_sebar)
from pra_proses import satu_berkas
from uji_lod import baca_blanko, topeng_kolom

N_BENIH = 20


def siapkan_blanko(zf, alat, X):
    """Blanko pada sumbu yang sama dengan data terukur."""
    Vb, Ib, okb = baca_blanko(zf, alat)
    Xb = np.vstack([satu_berkas(Vb[n], Ib[n], okb[n])
                    for n in range(Vb.shape[0])])[:, topeng_kolom()]
    med = np.nanmedian(X, axis=0)
    bar, kol = np.where(np.isnan(Xb))
    Xb[bar, kol] = med[kol]
    return Xb


def jalankan(X, y, t, level, Xb):
    """Kumpulkan pemisahan terhadap blanko dan lebar selang tiap tingkat."""
    pisah = {str(int(lv)): [] for lv in level}
    lebar = {str(int(lv)): [] for lv in level}
    atas_blanko = []
    for b in range(N_BENIH):
        rng = np.random.default_rng(1000 + b)
        i_fit, i_kal, i_uji = belah(y, level, rng, 20, 20, 10)
        rf = RandomForestRegressor(n_estimators=300, random_state=b,
                                   n_jobs=-1).fit(X[i_fit], t[i_fit])
        p_kal = rf.predict(X[i_kal])
        s_kal = skala_sebar(rf, X, i_kal, lantai=1e-6)
        q = ambang(np.abs(t[i_kal] - p_kal) / s_kal, ALPHA)

        p_b = rf.predict(Xb)
        s_b = np.std([e.predict(Xb) for e in rf.estimators_], axis=0) + 1e-6
        batas_atas_blanko = float(np.median(p_b + q * s_b))
        atas_blanko.append(10 ** batas_atas_blanko)

        p_u = rf.predict(X[i_uji])
        s_u = skala_sebar(rf, X, i_uji, lantai=1e-6)
        bawah, atas = p_u - q * s_u, p_u + q * s_u
        for lv in level:
            m = y[i_uji] == lv
            if m.any():
                pisah[str(int(lv))].append(float(np.mean(bawah[m] > batas_atas_blanko)))
                lebar[str(int(lv))].append(float(np.median(10 ** (atas[m] - bawah[m]))))
        print(f"  benih {b} selesai", flush=True)

    return {
        "batas_atas_blanko_ppm": float(np.median(atas_blanko)),
        "per_level": {k: {"pisah_dari_blanko": float(np.mean(v)),
                          "faktor_lebar": float(np.mean(lebar[k]))}
                      for k, v in pisah.items()},
    }


def main(arsip):
    data = muat()
    keluar = {}
    with zipfile.ZipFile(arsip) as zf:
        for alat in ALAT:
            print(alat, flush=True)
            X, y, t, level = data[alat]
            h = jalankan(X, y, t, level, siapkan_blanko(zf, alat, X))
            lolos = [int(k) for k, v in h["per_level"].items()
                     if v["pisah_dari_blanko"] >= 0.95]
            h["lod_metode_ppm"] = min(lolos) if lolos else None
            h["syarat"] = "kadar terendah yang 95 persen replikatnya terpisah dari blanko"
            keluar[alat] = h
    p = os.path.join(HASIL, "hasil_lod_metode.json")
    json.dump(keluar, open(p, "w"), indent=1)
    print("ditulis", p)

    for alat in ALAT:
        h = keluar[alat]
        print(f"\n{alat}  batas atas selang blanko {h['batas_atas_blanko_ppm']:.2f} ppm, "
              f"batas deteksi metode {h['lod_metode_ppm']} ppm")
        print(f"  {'kadar':>6} {'pisah':>7} {'faktor lebar':>13}")
        for k in sorted(h["per_level"], key=lambda s: int(s)):
            v = h["per_level"][k]
            print(f"  {k:>6} {v['pisah_dari_blanko']:7.3f} {v['faktor_lebar']:13.3f}")


if __name__ == "__main__":
    arsip = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CALPIR_ZIP", "")
    assert os.path.isfile(arsip), "tunjuk arsip voltamogram mentah sebagai argumen"
    main(arsip)

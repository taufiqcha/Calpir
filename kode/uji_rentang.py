"""Rentang kerja linier dan batas deteksi di dalamnya.

Melaporkan satu batas deteksi atas seluruh rentang 2 sampai 1000 ppm keliru,
sebab tanggapan voltametri jarang lurus sepanjang tiga dekade. Praktik baku
kimia analitik adalah menyatakan rentang kerja linier lebih dulu, lalu
menghitung batas deteksi di dalam rentang itu.

Skrip ini menyapu seluruh jendela kadar yang bersambung, menghitung kecocokan
garis pada tiap jendela, lalu memilih jendela terlebar yang kecocokannya
memenuhi syarat. Dua bentuk tanggapan diuji, yaitu arus puncak katodik dan
arus pada titik potensial yang paling erat berperingkat dengan kadar.

Simpangan baku blanko diambil dari hasil_lod_blanko.json, jadi arsip mentah
tidak perlu dibaca ulang.

Keluaran ditulis ke results/hasil_rentang_linier.json.

Cara pakai: python3 uji_rentang.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calpir import ALAT, DATA, HASIL, muat

MIN_TITIK = 4            # jendela sah bila memuat sekurangnya empat kadar
SYARAT_R2 = (0.99, 0.98, 0.95)


def kecocokan(kadar, respons, sigma):
    """Garis lurus pada satu jendela, beserta batas deteksi dan kuantifikasi."""
    kem, pot = np.polyfit(kadar, respons, 1)
    sisa = respons - (kem * kadar + pot)
    r2 = float(1 - sisa.var() / respons.var()) if respons.var() > 0 else 0.0
    if kem == 0:
        return None
    return {"dari_ppm": float(kadar[0]), "sampai_ppm": float(kadar[-1]),
            "n_titik": int(kadar.size), "R2": r2,
            "kemiringan": float(kem),
            "LOD_ppm": float(3 * sigma / abs(kem)),
            "LOQ_ppm": float(10 * sigma / abs(kem))}


def sapu(kadar, respons, sigma):
    """Seluruh jendela bersambung yang memuat sekurangnya MIN_TITIK kadar."""
    keluar = []
    n = kadar.size
    for i in range(n):
        for j in range(i + MIN_TITIK - 1, n):
            h = kecocokan(kadar[i:j + 1], respons[i:j + 1], sigma)
            if h is not None:
                keluar.append(h)
    return keluar


def terlebar(jendela, syarat):
    """Jendela dengan kadar terbanyak yang kecocokannya memenuhi syarat."""
    layak = [j for j in jendela if j["R2"] >= syarat]
    if not layak:
        return None
    return max(layak, key=lambda j: (j["n_titik"], -j["LOD_ppm"]))


def main():
    lod = json.load(open(os.path.join(HASIL, "hasil_lod_blanko.json")))
    z = np.load(os.path.join(DATA, "sumbu_sama.npz"))
    data = muat()
    keluar = {}

    for alat in ALAT:
        X, y, t, level = data[alat]
        sc = np.load(os.path.join(DATA, f"scans_{alat}.npz"))
        I, ok, y_sc = sc["I"], sc["ok"], sc["y"]
        puncak = np.array([float(np.median([I[n, s].min()
                                            for s in range(2, I.shape[1])
                                            if ok[n, s]]))
                           for n in range(I.shape[0])])
        j = lod[alat]["titik_terpilih"]["indeks"]
        respons = {
            "arus_puncak": (np.array([puncak[y_sc == lv].mean() for lv in level]),
                            lod[alat]["puncak_blanko"]["sd"]),
            "titik_terpilih": (np.array([X[y == lv, j].mean() for lv in level]),
                               lod[alat]["titik_terpilih"]["sd_blanko"]),
        }
        keluar[alat] = {}
        for nama, (r, sigma) in respons.items():
            jendela = sapu(level.astype(float), r, sigma)
            keluar[alat][nama] = {
                "sigma_blanko": sigma,
                "semua_jendela": sorted(jendela, key=lambda d: -d["R2"])[:8],
                "terlebar": {str(s): terlebar(jendela, s) for s in SYARAT_R2},
            }
    keluar["catatan"] = {"min_titik": MIN_TITIK, "syarat_R2": list(SYARAT_R2),
                         "kaidah": "LOD = 3 sigma blanko / kemiringan jendela"}
    p = os.path.join(HASIL, "hasil_rentang_linier.json")
    json.dump(keluar, open(p, "w"), indent=1)
    print("ditulis", p)

    for alat in ALAT:
        print(f"\n{'='*66}\n{alat}\n{'='*66}")
        for nama in ("arus_puncak", "titik_terpilih"):
            h = keluar[alat][nama]
            print(f"  {nama}, sigma blanko {h['sigma_blanko']:.3f}")
            for s in SYARAT_R2:
                v = h["terlebar"][str(s)]
                if v is None:
                    print(f"    R2 >= {s}: tidak ada jendela")
                else:
                    print(f"    R2 >= {s}: {v['dari_ppm']:>6.0f} sampai "
                          f"{v['sampai_ppm']:>6.0f} ppm, {v['n_titik']} kadar, "
                          f"R2={v['R2']:.4f}, LOD={v['LOD_ppm']:.2f} ppm, "
                          f"LOQ={v['LOQ_ppm']:.2f} ppm")


if __name__ == "__main__":
    main()

"""Batas deteksi setelah replikat blanko yang belum setimbang disisihkan.

Membuang dua sapuan pertama menghilangkan transien pada kebanyakan replikat,
tetapi tidak pada semuanya. Pada sebagian replikat arus puncak masih menanjak
sampai sapuan kesepuluh, yang berarti elektrodanya belum setimbang selama
seluruh perekaman replikat itu.

Kriteria yang dipakai di sini hanya melihat ke dalam satu replikat.

    Replikat dipakai bila arus puncaknya tidak menanjak lebih dari ambang
    tertentu dari sapuan ketiga ke sapuan kesepuluh.

Kriteria itu tidak menyentuh nilai arusnya, tidak memakai urutan berkas, dan
tidak memakai kadar, sehingga tidak dapat menarik hasil ke arah mana pun.
Replikat setimbang bergeser sedikit menurun, sedangkan replikat yang belum
setimbang menanjak belasan sampai puluhan persen.

Kemiringan garis kalibrasi diambil dari jendela linier yang sudah ditetapkan,
sehingga yang berubah hanya simpangan baku blanko.

Keluaran ditulis ke results/hasil_lod_stabil.json.

Cara pakai: python3 uji_lod_stabil.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calpir import ALAT, HASIL

AMBANG = (None, 5.0, 0.0)     # None berarti seluruh replikat dipakai
SAPUAN = list(range(2, 10))   # sapuan 3 sampai 10


def geser(r):
    """Geseran arus puncak dari sapuan ketiga ke sapuan kesepuluh, dalam persen."""
    p = [r["sapuan"][s]["puncak"] for s in SAPUAN if r["sapuan"][s] is not None]
    if len(p) < 2 or p[0] == 0:
        return None
    return 100.0 * (p[-1] - p[0]) / abs(p[0])


def puncak_replikat(r):
    """Arus puncak replikat, yaitu median atas sapuan ketiga sampai kesepuluh."""
    p = [r["sapuan"][s]["puncak"] for s in SAPUAN if r["sapuan"][s] is not None]
    return float(np.median(p)) if p else None


def main():
    diag = json.load(open(os.path.join(HASIL, "diagnosa_blanko.json")))
    rent = json.load(open(os.path.join(HASIL, "hasil_rentang_linier.json")))
    keluar = {}

    for alat in ALAT:
        rep = diag[alat]["replikat"]
        nilai = [(puncak_replikat(r), geser(r), r["berkas"]) for r in rep]
        nilai = [v for v in nilai if v[0] is not None and v[1] is not None]

        jendela = {}
        for syarat in ("0.99", "0.98", "0.95"):
            j = rent[alat]["arus_puncak"]["terlebar"][syarat]
            if j is not None:
                jendela[syarat] = j

        per_ambang = {}
        for a in AMBANG:
            pakai = nilai if a is None else [v for v in nilai if v[1] <= a]
            if len(pakai) < 5:
                continue
            p = np.array([v[0] for v in pakai])
            sigma = float(p.std(ddof=1))
            kunci = "semua" if a is None else f"geser<={a}"
            per_ambang[kunci] = {
                "n_dipakai": len(pakai), "n_dibuang": len(nilai) - len(pakai),
                "dibuang": [v[2] for v in nilai if a is not None and v[1] > a],
                "sigma": sigma, "rata": float(p.mean()),
                "lod": {s: {"dari_ppm": j["dari_ppm"], "sampai_ppm": j["sampai_ppm"],
                            "R2": j["R2"],
                            "LOD_ppm": 3 * sigma / abs(j["kemiringan"]),
                            "LOQ_ppm": 10 * sigma / abs(j["kemiringan"])}
                        for s, j in jendela.items()},
            }
        keluar[alat] = {"geser_tiap_replikat":
                        {v[2]: round(v[1], 2) for v in sorted(nilai, key=lambda x: -x[1])},
                        "per_ambang": per_ambang}

    keluar["catatan"] = {
        "sapuan_dipakai": "ketiga sampai kesepuluh",
        "kriteria": "replikat dipakai bila geseran arus puncak sapuan 3 ke 10 "
                    "tidak melampaui ambang",
        "ambang_persen": [a for a in AMBANG if a is not None],
        "kemiringan": "dari jendela linier pada hasil_rentang_linier.json",
    }
    p = os.path.join(HASIL, "hasil_lod_stabil.json")
    json.dump(keluar, open(p, "w"), indent=1)
    print("ditulis", p)

    for alat in ALAT:
        print(f"\n{'='*70}\n{alat}\n{'='*70}")
        for kunci, h in keluar[alat]["per_ambang"].items():
            print(f"  {kunci:12s} n={h['n_dipakai']:2d} dipakai, "
                  f"{h['n_dibuang']:2d} dibuang, sigma={h['sigma']:.3f}")
            for s, j in h["lod"].items():
                print(f"      jendela R2>={s}  {j['dari_ppm']:.0f} sampai "
                      f"{j['sampai_ppm']:.0f} ppm  ->  LOD {j['LOD_ppm']:6.2f} ppm, "
                      f"LOQ {j['LOQ_ppm']:7.2f} ppm")


if __name__ == "__main__":
    main()

"""Cakupan per tingkat kadar dan mutu keputusan lampau ambang.

Reviewer 2 butir 5 menyatakan cakupan marginal saja tidak memadai untuk
keputusan regulasi, lalu meminta dua hal, yaitu cakupan yang khas tiap tingkat
kadar dan laju salah lampau beserta laju lampau terlewat di sekitar ambang
keputusan.

Skrip ini mengulang protokol A dengan benih, partisi, dan model yang sama
persis, lalu menyimpan batas bawah dan batas atas tiap titik uji, bukan hanya
cakupan rata-ratanya. Dari batas itu dihitung tiga hal.

  1. Cakupan tiap tingkat kadar, pada konstruksi marginal dan konstruksi
     Mondrian. Mondrian memakai satu ambang tiap tingkat, sehingga cakupannya
     terkondisi pada kadar.
  2. Keputusan tiga arah terhadap ambang T, yaitu di atas bila batas bawah
     melampaui T, di bawah bila batas atas di bawah T, dan belum tentu bila
     selangnya memuat T.
  3. Pembanding tanpa selang, yaitu keputusan dari ramalan titik saja.

  H0 cakupan  median cakupan tiap tingkat sama dengan tingkat nominal 0,90
  H1 cakupan  berbeda dari 0,90

  H0 Mondrian median simpangan mutlak cakupan tiap tingkat terhadap 0,90 sama
              antara konstruksi marginal dan konstruksi Mondrian
  H1 Mondrian simpangan Mondrian lebih kecil

Ambang keputusan sengaja diletakkan di antara dua tingkat terukur, sehingga
tidak ada titik uji yang tepat berada di ambang.

Keluaran ditulis ke results/hasil_cakupan_level.json.

Cara pakai: python3 uji_cakupan.py
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

N_BENIH = 20
NOMINAL = 1 - ALPHA
AMBANG_PPM = (15.0, 50.0, 150.0, 500.0)   # di antara dua tingkat terukur


def batas(t_kal, p_kal, s_kal, p_uji, s_uji, y_kal=None, y_uji=None):
    """Batas bawah dan batas atas selang konformal tiap titik uji.

    Memberi y_kal dan y_uji memilih ragam Mondrian, yaitu satu ambang tiap
    tingkat kadar alih-alih satu ambang bersama.
    """
    if y_kal is None:
        q = np.full(p_uji.size, ambang(np.abs(t_kal - p_kal) / s_kal, ALPHA))
    else:
        q = np.zeros(p_uji.size)
        for lv in np.unique(y_kal):
            mk = y_kal == lv
            q[y_uji == lv] = ambang(np.abs(t_kal[mk] - p_kal[mk]) / s_kal[mk], ALPHA)
    return p_uji - q * s_uji, p_uji + q * s_uji


def keputusan(bawah, atas, p_uji, t_uji, tau):
    """Cacah keputusan tiga arah dan keputusan ramalan titik terhadap ambang."""
    benar_atas = t_uji > tau
    nyata_atas, nyata_bawah = bawah > tau, atas < tau
    ragu = ~(nyata_atas | nyata_bawah)
    titik_atas = p_uji > tau
    n_bawah, n_atas = int((~benar_atas).sum()), int(benar_atas.sum())
    return {
        "n_benar_bawah": n_bawah,
        "n_benar_atas": n_atas,
        "salah_lampau": int((nyata_atas & ~benar_atas).sum()),
        "lampau_terlewat": int((nyata_bawah & benar_atas).sum()),
        "ragu": int(ragu.sum()),
        "ragu_bawah": int((ragu & ~benar_atas).sum()),
        "ragu_atas": int((ragu & benar_atas).sum()),
        "titik_salah_lampau": int((titik_atas & ~benar_atas).sum()),
        "titik_lampau_terlewat": int((~titik_atas & benar_atas).sum()),
        "n": int(t_uji.size),
    }


def satu_alat(X, t, y, level):
    """Jalankan 20 benih dan kumpulkan hasil per tingkat dan per ambang."""
    per_level = {str(lv): {"marginal": [], "mondrian": []} for lv in level}
    per_ambang = {str(T): [] for T in AMBANG_PPM}
    lebar = {"marginal": [], "mondrian": []}

    for b in range(N_BENIH):
        rng = np.random.default_rng(1000 + b)
        i_fit, i_kal, i_uji = belah(y, level, rng, 20, 20, 10)
        rf = RandomForestRegressor(n_estimators=300, random_state=b,
                                   n_jobs=-1).fit(X[i_fit], t[i_fit])
        p_kal, p_uji = rf.predict(X[i_kal]), rf.predict(X[i_uji])
        s_kal, s_uji = (skala_sebar(rf, X, i, lantai=1e-6) for i in (i_kal, i_uji))

        hasil = {
            "marginal": batas(t[i_kal], p_kal, s_kal, p_uji, s_uji),
            "mondrian": batas(t[i_kal], p_kal, s_kal, p_uji, s_uji,
                              y[i_kal], y[i_uji]),
        }
        for nama, (bawah, atas) in hasil.items():
            tutup = (t[i_uji] >= bawah) & (t[i_uji] <= atas)
            lebar[nama].append(float(np.median(atas - bawah)))
            for lv in level:
                m = y[i_uji] == lv
                if m.any():
                    per_level[str(lv)][nama].append(float(tutup[m].mean()))

        bawah, atas = hasil["marginal"]
        for T in AMBANG_PPM:
            per_ambang[str(T)].append(
                keputusan(bawah, atas, p_uji, t[i_uji], np.log10(T)))
        print(f"  benih {b} selesai", flush=True)

    return per_level, per_ambang, lebar


def ringkas(per_level, per_ambang, lebar, level):
    """Rata-ratakan tiap tingkat, gabungkan tiap ambang, lalu uji keduanya."""
    cak = {n: np.array([np.mean(per_level[str(lv)][n]) for lv in level])
           for n in ("marginal", "mondrian")}
    W1, p1 = wilcoxon(cak["marginal"] - NOMINAL, alternative="two-sided")
    W2, p2 = wilcoxon(np.abs(cak["mondrian"] - NOMINAL),
                      np.abs(cak["marginal"] - NOMINAL), alternative="less")

    ambang_ring = {}
    for T, daftar in per_ambang.items():
        tot = {k: sum(d[k] for d in daftar) for k in daftar[0]}
        nb, na = tot["n_benar_bawah"], tot["n_benar_atas"]
        ambang_ring[T] = {
            "cacah": tot,
            "laju_salah_lampau": tot["salah_lampau"] / nb,
            "laju_lampau_terlewat": tot["lampau_terlewat"] / na,
            "pangsa_ragu": tot["ragu"] / tot["n"],
            "titik_laju_salah_lampau": tot["titik_salah_lampau"] / nb,
            "titik_laju_lampau_terlewat": tot["titik_lampau_terlewat"] / na,
        }

    return {
        "per_level": {str(lv): {n: float(np.mean(per_level[str(lv)][n]))
                                for n in ("marginal", "mondrian")}
                      for lv in level},
        "cakupan_level": {n: {"rata": float(v.mean()), "min": float(v.min()),
                              "maks": float(v.max()),
                              "simpangan_mutlak_rata": float(np.abs(v - NOMINAL).mean()),
                              "level_di_bawah_nominal": int((v < NOMINAL).sum())}
                          for n, v in cak.items()},
        "lebar_median": {n: float(np.mean(v)) for n, v in lebar.items()},
        "uji_cakupan_level": {"nama": "Wilcoxon signed-rank, dua sisi, 15 tingkat lawan 0,90",
                              "W": float(W1), "p": float(p1)},
        "uji_mondrian": {"nama": "Wilcoxon signed-rank, satu sisi, simpangan mutlak",
                         "W": float(W2), "p": float(p2)},
        "per_ambang": ambang_ring,
    }


if __name__ == "__main__":
    t0 = time.time()
    data = muat()
    keluar = {}
    for alat in ALAT:
        print(alat, flush=True)
        X, y, t, level = data[alat]
        keluar[alat] = ringkas(*satu_alat(X, t, y, level), level)
    keluar["catatan"] = {"nominal": NOMINAL, "n_benih": N_BENIH,
                         "ambang_ppm": list(AMBANG_PPM),
                         "satuan_lebar": "desimal log10 ppm",
                         "menit": round((time.time() - t0) / 60, 2)}
    p = os.path.join(HASIL, "hasil_cakupan_level.json")
    json.dump(keluar, open(p, "w"), indent=1)
    print("ditulis", p)
    for alat in ALAT:
        h = keluar[alat]
        print(f"\n{alat}")
        for n in ("marginal", "mondrian"):
            c = h["cakupan_level"][n]
            print(f"  {n:9s} cakupan tingkat {c['min']:.3f} sampai {c['maks']:.3f}, "
                  f"rata {c['rata']:.3f}, di bawah nominal "
                  f"{c['level_di_bawah_nominal']}/15, lebar {h['lebar_median'][n]:.3f}")
        print(f"  uji lawan 0,90 p={h['uji_cakupan_level']['p']:.4f}, "
              f"Mondrian lebih rapat p={h['uji_mondrian']['p']:.4f}")
        for T, a in h["per_ambang"].items():
            print(f"  ambang {T} ppm: salah lampau {a['laju_salah_lampau']:.4f} "
                  f"lawan titik {a['titik_laju_salah_lampau']:.4f}, "
                  f"lampau terlewat {a['laju_lampau_terlewat']:.4f} "
                  f"lawan titik {a['titik_laju_lampau_terlewat']:.4f}, "
                  f"ragu {a['pangsa_ragu']:.3f}")

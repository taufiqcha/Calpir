"""Seluruh percobaan naskah CALPIR, satu pintu masuk.

Tiap percobaan menyimpan hasilnya sendiri ke ../results, sehingga kegagalan di
percobaan belakang tidak menghapus yang sudah selesai, dan percobaan yang
berkasnya sudah ada akan dilewati kecuali dipaksa dengan --ulang.

  interval    kisi penuh 6 model dasar kali 6 ragam konformal kali 3 alfa kali
              20 benih. Menjawab apakah keunggulan penskalaan milik metodenya
              atau milik satu model tertentu.       -> besar_tahap1.json
  ensemble    ensembel dalam DITALA lewat pencarian acak 40 calon, lalu
              dikonformalkan.                       -> besar_tahap2.json
  bersarang   ragam, skala, dan model dipilih di dalam lipatan; lipatan luar
              tidak pernah tersentuh.                -> besar_tahap3.json
  benih       ulangan berbenih banyak, 20 benih untuk protokol acak, 10 untuk
              level disembunyikan dan untuk penjaga. -> hasil_ulangan_benih.json
  pembanding  sepuluh cara menakar ketidakpastian pada data yang sama.
                                                     -> hasil_pembanding.json
  penjaga     penjaga satu tahap lawan dua tahap, tiga calon skor kebaruan,
              anggaran penolakan disapu.             -> hasil_penjaga_v3.json
  sesi        kontrol: apakah penjaga menangkap kebaruan KONSENTRASI atau
              sekadar keadaan sesi.                  -> hasil_kontrol_sesi.json

Pemakaian:
    python3 percobaan.py                    seluruhnya, jalan penuh
    python3 percobaan.py interval penjaga   hanya yang disebut
    python3 percobaan.py --cepat            ukuran mainan, seluruh jalur kode
    python3 percobaan.py --ulang            paksa hitung ulang yang sudah ada
"""
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

import calpir
from calpir import (ALPHA, BUANG_GERBANG, HASIL, ambang, belah, belah_dalam,
                    buat_model, jarak_knn, konformal, muat, muat_sapuan,
                    penjaga_dua_tahap, periksa_partisi, ramal, skala_sebar,
                    skala_terlatih)

CEPAT = "--cepat" in sys.argv
ULANG = "--ulang" in sys.argv
AKHIRAN = "_cepat" if CEPAT else ""

ALAT = ("LC",) if CEPAT else calpir.ALAT
MODEL = ("RF", "LGBM") if CEPAT else calpir.MODEL
ALFA = (0.10,) if CEPAT else (0.05, 0.10, 0.20)
EPOCH = 5 if CEPAT else 200
N_BENIH_1 = 2 if CEPAT else 20            # kisi penuh
N_PERCOBAAN_2 = 3 if CEPAT else 40        # calon arsitektur
N_BENIH_2 = 2 if CEPAT else 5             # ensembel dalam ditala
N_BENIH_3 = 2 if CEPAT else 5             # lipatan luar bersarang
N_BENIH_A = 2 if CEPAT else 20            # protokol acak
N_BENIH_B = 2 if CEPAT else 10            # level disembunyikan
N_BENIH_C = 2 if CEPAT else 10            # penjaga
N_BENIH_D = 1 if CEPAT else 3             # ensembel dalam dikonformalkan
N_ULANG_A = 2 if CEPAT else 5             # pembanding, protokol acak
N_ANGGOTA, N_LINTASAN = 5, 50
BETA_SAPU = (0.01, 0.05, 0.10, 0.20)
Z = 1.6448536269514722                    # kuantil normal 95 persen
RAGAM = ("global", "mondrian", "ternormalkan", "mondrian_ternormalkan")
PENJAGA_CALON = ("jarak_knn", "konformal", "gabungan")
CARA = ("konformal_rf_global", "konformal_rf_mondrian", "konformal_rf_ternormalkan",
        "konformal_rf_mondrian_ternormalkan", "cqr_gbm", "ensemble_dalam",
        "ensemble_dalam_konformal", "mc_dropout", "mc_dropout_konformal", "rf_sebar")


def tersimpan(nama):
    p = os.path.join(HASIL, f"{nama}{AKHIRAN}.json")
    if os.path.exists(p) and not ULANG:
        print(f"{nama} dilewati, sudah ada {p}")
        return p, True
    return p, False


def tulis(p, isi, t0=None):
    os.makedirs(HASIL, exist_ok=True)
    if t0 is not None:
        isi = {"menit": (time.time() - t0) / 60, "hasil": isi}
    json.dump(isi, open(p, "w"), indent=1)
    print("ditulis", p)


# ============================================================ metode A, kisi
def satu_pemasangan(X, y, t, i_fit, i_kal, i_uji, nama, benih, pakai_terlatih=True):
    """Cakupan dan lebar tiap (skala, ragam, alfa) untuk satu model dan satu benih."""
    periksa_partisi(i_fit, i_kal, i_uji)
    m = buat_model(nama, benih).fit(X[i_fit], t[i_fit])
    pk, pu = ramal(m, X[i_kal]), ramal(m, X[i_uji])
    assert np.isfinite(pk).all() and np.isfinite(pu).all(), "ramalan memuat NaN"

    diag = {"pecahan_residu_nol": float(np.mean(np.abs(t[i_kal] - pk) == 0)),
            "ambang_nol": []}
    skala = {"tanpa": (None, None)}
    if nama in calpir.PUNYA_POHON:
        skala["sebar"] = (skala_sebar(m, X, i_kal), skala_sebar(m, X, i_uji))
    if pakai_terlatih:
        skala["terlatih"] = skala_terlatih(nama, X, t, i_fit, benih, X[i_kal], X[i_uji])

    out = {}
    for nama_sk, (sk, su) in skala.items():
        for mond in (False, True):
            for a in ALFA:
                c, w = konformal(t[i_kal], pk, t[i_uji], pu, sk, su, a,
                                 y[i_kal] if mond else None, y[i_uji] if mond else None)
                if c is None:
                    continue
                kunci = f"{nama_sk}|{'mondrian' if mond else 'marginal'}|{a}"
                if w == 0:
                    diag["ambang_nol"].append(kunci)
                out[kunci] = (c, w)
    return out, diag


def interval(data):
    p, ada = tersimpan("besar_tahap1")
    if ada:
        return
    t0, keluaran = time.time(), {}
    for alat in ALAT:
        X, y, t, level = data[alat]
        keluaran[alat] = {}
        for nama in MODEL:
            kum, diagn = {}, []
            for b in range(N_BENIH_1):
                rng = np.random.default_rng(1000 + b)
                i_fit, i_kal, i_uji = belah(y, level, rng, 20, 20, 10)
                h, dg = satu_pemasangan(X, y, t, i_fit, i_kal, i_uji, nama, b)
                diagn.append(dg)
                for k, v in h.items():
                    kum.setdefault(k, []).append(v)
            keluaran[alat][nama] = {
                k: {"cakupan": float(np.mean([x[0] for x in v])),
                    "cakupan_sd": float(np.std([x[0] for x in v])),
                    "lebar": float(np.mean([x[1] for x in v])),
                    "lebar_sd": float(np.std([x[1] for x in v])),
                    "lebar_min": float(np.min([x[1] for x in v])),
                    "lebar_maks": float(np.max([x[1] for x in v])), "n": len(v)}
                for k, v in kum.items()}
            keluaran[alat][nama]["_diagnostik"] = {
                "pecahan_residu_nol": float(np.mean([d["pecahan_residu_nol"] for d in diagn])),
                "benih_dengan_ambang_nol": int(sum(bool(d["ambang_nol"]) for d in diagn)),
                "contoh_ambang_nol": sorted({k for d in diagn for k in d["ambang_nol"]})[:6]}
            print(f"  interval {alat} {nama} selesai, residu nol "
                  f"{keluaran[alat][nama]['_diagnostik']['pecahan_residu_nol']:.3f}",
                  flush=True)
    tulis(p, keluaran, t0)


# =================================================== pembanding ensembel dalam
def ensemble(data):
    p, ada = tersimpan("besar_tahap2")
    if ada:
        return
    t0 = time.time()
    rngc = np.random.default_rng(7)
    calon = [{"lapis": int(rngc.choice([2, 3])), "unit": int(rngc.choice([32, 64, 128])),
              "dropout": float(rngc.choice([0.0, 0.1, 0.2])),
              "lr": float(rngc.choice([1e-3, 3e-4]))} for _ in range(N_PERCOBAAN_2)]
    buat = lambda c, n_masuk, bn: calpir.jaringan_gauss(
        n_masuk, (c["unit"],) * c["lapis"], c["dropout"], c["lr"], bn)
    keluaran = {}
    for alat in ALAT:
        X, y, t, level = data[alat]
        rng = np.random.default_rng(500)
        i_fit, i_kal, i_uji = belah(y, level, rng, 20, 20, 10)
        # pemilihan konfigurasi HANYA memakai belahan di dalam himpunan latih
        o = np.random.default_rng(0).permutation(i_fit.size)
        n_dalam = int(0.8 * i_fit.size)
        i_tr, i_val = i_fit[o[:n_dalam]], i_fit[o[n_dalam:]]
        sc = StandardScaler().fit(X[i_tr])
        nilai = []
        for c in calon:
            m = calpir.latih_gauss(buat(c, X.shape[1], 0),
                                   sc.transform(X[i_tr]), t[i_tr], 0, EPOCH)
            mae = float(np.mean(np.abs(
                t[i_val] - m(sc.transform(X[i_val]), training=False).numpy()[:, 0])))
            nilai.append(mae)
            print(f"  ensemble {alat} calon {c} mae {mae:.4f}", flush=True)
        terbaik = calon[int(np.argmin(nilai))]

        hasil = []
        for b in range(N_BENIH_2):
            rng = np.random.default_rng(600 + b)
            i_fit, i_kal, i_uji = belah(y, level, rng, 20, 20, 10)
            periksa_partisi(i_fit, i_kal, i_uji)
            sc = StandardScaler().fit(X[i_fit])
            Xf, Xk, Xu = (sc.transform(X[i]) for i in (i_fit, i_kal, i_uji))
            mk, mu, vk, vu = [], [], [], []
            for a in range(N_ANGGOTA):
                m = calpir.latih_gauss(buat(terbaik, X.shape[1], 10 * b + a),
                                       Xf, t[i_fit], 10 * b + a, EPOCH)
                ok, ou = m.predict(Xk, verbose=0), m.predict(Xu, verbose=0)
                mk.append(ok[:, 0]); vk.append(ok[:, 1] ** 2)
                mu.append(ou[:, 0]); vu.append(ou[:, 1] ** 2)
            Mk, Mu = np.mean(mk, 0), np.mean(mu, 0)
            Sk = np.sqrt(np.mean(vk, 0) + np.var(mk, 0)) + 1e-6
            Su = np.sqrt(np.mean(vu, 0) + np.var(mu, 0)) + 1e-6
            assert Su.std() > 0, "skala jaringan merata, kepala ragam tidak belajar"
            hasil.append({str(a_): konformal(t[i_kal], Mk, t[i_uji], Mu, Sk, Su, a_)
                          for a_ in ALFA})
            print(f"  ensemble {alat} benih {b} selesai", flush=True)
        keluaran[alat] = {"terbaik": terbaik, "mae_calon": nilai,
                          "per_alfa": {str(a_): {
                              "cakupan": float(np.mean([h[str(a_)][0] for h in hasil])),
                              "lebar": float(np.mean([h[str(a_)][1] for h in hasil])),
                              "lebar_min": float(np.min([h[str(a_)][1] for h in hasil]))}
                              for a_ in ALFA}}
    tulis(p, keluaran, t0)


# ============================================================ metode C, bersarang
def bersarang(data):
    p, ada = tersimpan("besar_tahap3")
    if ada:
        return
    t0, a_tetap, keluaran = time.time(), 0.10, {}
    for alat in ALAT:
        X, y, t, level = data[alat]
        dipilih, luar = [], []
        for b in range(N_BENIH_3):
            rng = np.random.default_rng(900 + b)
            i_fit, i_kal, i_uji = belah(y, level, rng, 20, 20, 10)
            periksa_partisi(i_fit, i_kal, i_uji)
            # pemilihan DI DALAM: himpunan latih dibelah lagi menjadi tiga
            i_td, i_kd, i_ud = belah_dalam(i_fit, y, np.random.default_rng(b), 8, 6, 6)
            periksa_partisi(i_td, i_kd, i_ud)
            terbaik, nilai_terbaik = None, np.inf
            for nama in MODEL:
                h, _ = satu_pemasangan(X, y, t, i_td, i_kd, i_ud, nama, b)
                for k, (c, w) in h.items():
                    if not k.endswith(str(a_tetap)):
                        continue
                    if c >= 1 - a_tetap - 0.02 and w < nilai_terbaik:
                        terbaik, nilai_terbaik = (nama, k), w
            assert terbaik is not None, "tidak ada calon yang memenuhi cakupan di dalam"
            nama, kunci = terbaik
            h, _ = satu_pemasangan(X, y, t, i_fit, i_kal, i_uji, nama, b)
            dipilih.append(f"{nama}|{kunci}")
            luar.append(h[kunci])
            print(f"  bersarang {alat} benih {b} memilih {nama} {kunci} "
                  f"cakupan luar {h[kunci][0]:.3f} lebar {h[kunci][1]:.3f}", flush=True)
        keluaran[alat] = {"dipilih": dipilih,
                          "cakupan_luar": float(np.mean([x[0] for x in luar])),
                          "cakupan_luar_sd": float(np.std([x[0] for x in luar])),
                          "lebar_luar": float(np.mean([x[1] for x in luar])),
                          "lebar_luar_sd": float(np.std([x[1] for x in luar]))}
    tulis(p, keluaran, t0)


# ============================================================ metode C, benih
def empat_ragam(X, t, i_fit, i_kal, i_uji, y, benih, pakai_mondrian=True):
    """Empat ragam konformal di atas satu random forest."""
    rf = RandomForestRegressor(n_estimators=300, random_state=benih,
                               n_jobs=-1).fit(X[i_fit], t[i_fit])
    pk, pu = rf.predict(X[i_kal]), rf.predict(X[i_uji])
    sk, su = (skala_sebar(rf, X, i, lantai=1e-6) for i in (i_kal, i_uji))
    out = {"global": konformal(t[i_kal], pk, t[i_uji], pu),
           "ternormalkan": konformal(t[i_kal], pk, t[i_uji], pu, sk, su)}
    if pakai_mondrian:
        out["mondrian"] = konformal(t[i_kal], pk, t[i_uji], pu,
                                    y_kal=y[i_kal], y_uji=y[i_uji])
        out["mondrian_ternormalkan"] = konformal(t[i_kal], pk, t[i_uji], pu, sk, su,
                                                 y_kal=y[i_kal], y_uji=y[i_uji])
    return out


def benih(data):
    """Ulangan berbenih banyak. Benih mengendalikan partisi DAN penanaman model
    sekaligus, sehingga tiap ulangan merupakan tarikan yang benar-benar bebas."""
    p, ada = tersimpan("hasil_ulangan_benih")
    if ada:
        return
    keluaran = {"alpha": ALPHA, "benih": {"A": N_BENIH_A, "B": N_BENIH_B,
                                          "C": N_BENIH_C, "D": N_BENIH_D}}
    for alat in ALAT:
        X, y, t, level = data[alat]
        hasil = {"A": {r: {"cakupan": [], "lebar": []} for r in RAGAM},
                 "B": {r: [] for r in ("global", "ternormalkan")},
                 "C": {"tolak_out": [], "tolak_in": []}, "D": []}

        for b in range(N_BENIH_A):                       # protokol acak
            rng = np.random.default_rng(1000 + b)
            i_fit, i_kal, i_uji = belah(y, level, rng, 20, 20, 10)
            h = empat_ragam(X, t, i_fit, i_kal, i_uji, y, b)
            for r in RAGAM:
                hasil["A"][r]["cakupan"].append(h[r][0])
                hasil["A"][r]["lebar"].append(h[r][1])
            print(f"  benih {alat} A {b}", flush=True)

        for b in range(N_BENIH_B):                       # satu level disembunyikan
            rng = np.random.default_rng(2000 + b)
            per_level = {r: [] for r in ("global", "ternormalkan")}
            for lv_out in level:
                sisa = level[level != lv_out]
                i_fit, i_kal, _ = belah(y, sisa, rng, 35, 15, 0)
                h = empat_ragam(X, t, i_fit, i_kal, np.flatnonzero(y == lv_out), y, b,
                                pakai_mondrian=False)
                for r in per_level:
                    per_level[r].append(h[r][0])
            for r in per_level:
                hasil["B"][r].append(float(np.mean(per_level[r])))
            print(f"  benih {alat} B {b}", flush=True)

        for b in range(N_BENIH_C):                       # penjaga dua tahap
            rng = np.random.default_rng(3000 + b)
            t_out = []
            for lv_out in level:
                sisa = level[level != lv_out]
                i_fit, i_kal, _ = belah(y, sisa, rng, 25, 15, 10)
                t_out.append(penjaga_dua_tahap(X[i_fit], X[i_kal],
                                               X[np.flatnonzero(y == lv_out)])[0])
            hasil["C"]["tolak_out"].append(float(np.mean(t_out)))
            rng = np.random.default_rng(3000 + b)
            i_fit, i_kal, i_in = belah(y, level, rng, 25, 15, 10)
            hasil["C"]["tolak_in"].append(
                penjaga_dua_tahap(X[i_fit], X[i_kal], X[i_in])[0])
            print(f"  benih {alat} C {b}", flush=True)

        for b in range(N_BENIH_D):                       # ensembel dalam, pembanding
            rng = np.random.default_rng(4000 + b)
            i_fit, i_kal, i_uji = belah(y, level, rng, 20, 20, 10)
            sc = StandardScaler().fit(X[i_fit])
            Xf, Xk, Xu = (sc.transform(X[i]) for i in (i_fit, i_kal, i_uji))
            mk, mu, vk, vu = [], [], [], []
            for a in range(N_ANGGOTA):
                m = calpir.latih_gauss(
                    calpir.jaringan_gauss(Xf.shape[1], benih=100 * b + a),
                    Xf, t[i_fit], 100 * b + a, EPOCH)
                ok, ou = m.predict(Xk, verbose=0), m.predict(Xu, verbose=0)
                mk.append(ok[:, 0]); vk.append(ok[:, 1] ** 2)
                mu.append(ou[:, 0]); vu.append(ou[:, 1] ** 2)
            Mk, Mu = np.mean(mk, 0), np.mean(mu, 0)
            Sk = np.sqrt(np.mean(vk, 0) + np.var(mk, 0)) + 1e-6
            Su = np.sqrt(np.mean(vu, 0) + np.var(mu, 0)) + 1e-6
            hasil["D"].append(konformal(t[i_kal], Mk, t[i_uji], Mu, Sk, Su))
            print(f"  benih {alat} D {b}", flush=True)

        lm = np.array(hasil["A"]["mondrian_ternormalkan"]["lebar"])
        ln = np.array(hasil["A"]["ternormalkan"]["lebar"])
        W, pw = wilcoxon(lm, ln)
        keluaran[alat] = {
            "A": {r: {k: {"rata": float(np.mean(v)), "sd": float(np.std(v)),
                          "min": float(np.min(v)), "maks": float(np.max(v))}
                      for k, v in d.items()} for r, d in hasil["A"].items()},
            "A_uji_mondrian_lawan_ternormalkan": {
                "W": float(W), "p": float(pw),
                "selisih_median": float(np.median(lm - ln)),
                "benih_mondrian_lebih_sempit": int(np.sum(lm < ln))},
            "B": {r: {"rata": float(np.mean(v)), "sd": float(np.std(v))}
                  for r, v in hasil["B"].items()},
            "C": {k: {"rata": float(np.mean(v)), "sd": float(np.std(v)),
                      "min": float(np.min(v)), "maks": float(np.max(v))}
                  for k, v in hasil["C"].items()},
            "D": {"cakupan_rata": float(np.mean([x[0] for x in hasil["D"]])),
                  "lebar_rata": float(np.mean([x[1] for x in hasil["D"]])),
                  "lebar_min": float(np.min([x[1] for x in hasil["D"]]))}}
    tulis(p, keluaran)

    print(f"\n## protokol acak, {N_BENIH_A} benih")
    print(f"  {'alat':5} {'ragam':24} {'cakupan':>17} {'lebar':>21}")
    for alat in ALAT:
        for r in RAGAM:
            c, l = (keluaran[alat]["A"][r][k] for k in ("cakupan", "lebar"))
            print(f"  {alat:5} {r:24} {c['rata']:.3f} sd {c['sd']:.3f}   "
                  f"{l['rata']:.3f} sd {l['sd']:.3f} [{l['min']:.3f} {l['maks']:.3f}]")
    print("\n## uji berpasangan Mondrian ternormalkan lawan ternormalkan")
    for alat in ALAT:
        u = keluaran[alat]["A_uji_mondrian_lawan_ternormalkan"]
        print(f"  {alat}: W {u['W']:.0f}  p {u['p']:.4f}  selisih median "
              f"{u['selisih_median']:+.4f}  Mondrian lebih sempit "
              f"{u['benih_mondrian_lebih_sempit']}/{N_BENIH_A}")
    print(f"\n## level disembunyikan, {N_BENIH_B} benih")
    for alat in ALAT:
        for r in ("global", "ternormalkan"):
            v = keluaran[alat]["B"][r]
            print(f"  {alat} {r:14} cakupan {v['rata']:.3f} sd {v['sd']:.3f}")
    print(f"\n## penjaga dua tahap, {N_BENIH_C} benih")
    for alat in ALAT:
        for k in ("tolak_out", "tolak_in"):
            v = keluaran[alat]["C"][k]
            print(f"  {alat} {k:10} {v['rata']:.3f} sd {v['sd']:.3f} "
                  f"[{v['min']:.3f} {v['maks']:.3f}]")
    print(f"\n## ensembel dalam dikonformalkan, {N_BENIH_D} benih")
    for alat in ALAT:
        v = keluaran[alat]["D"]
        print(f"  {alat}: cakupan {v['cakupan_rata']:.3f}  lebar rata "
              f"{v['lebar_rata']:.3f}  lebar terbaik {v['lebar_min']:.3f}")


# ================================================== pembanding ketidakpastian
def satu_partisi(X, y, i_fit, i_kal, i_uji):
    """Sepuluh cara menakar ketidakpastian pada satu partisi. Seluruh sasaran
    dalam log10 ppm, sehingga lebarnya sebanding antar cara."""
    t = np.log10(y).astype("float32")
    sc = StandardScaler().fit(X[i_fit])
    Xf, Xk, Xu = (sc.transform(X[i]) for i in (i_fit, i_kal, i_uji))
    hasil = {}

    rf = RandomForestRegressor(n_estimators=300, random_state=0,
                               n_jobs=-1).fit(X[i_fit], t[i_fit])
    pk, pu = rf.predict(X[i_kal]), rf.predict(X[i_uji])
    sk, su = (skala_sebar(rf, X, i, lantai=1e-6) for i in (i_kal, i_uji))
    yk, yu = y[i_kal], y[i_uji]
    hasil["konformal_rf_global"] = konformal(t[i_kal], pk, t[i_uji], pu)
    hasil["konformal_rf_mondrian"] = konformal(t[i_kal], pk, t[i_uji], pu,
                                               y_kal=yk, y_uji=yu)
    hasil["konformal_rf_ternormalkan"] = konformal(t[i_kal], pk, t[i_uji], pu, sk, su)
    hasil["konformal_rf_mondrian_ternormalkan"] = konformal(
        t[i_kal], pk, t[i_uji], pu, sk, su, y_kal=yk, y_uji=yu)

    lo, hi = (GradientBoostingRegressor(loss="quantile", alpha=a, random_state=0,
                                        n_estimators=200).fit(X[i_fit], t[i_fit])
              for a in (ALPHA / 2, 1 - ALPHA / 2))
    qc = ambang(np.maximum(lo.predict(X[i_kal]) - t[i_kal],
                           t[i_kal] - hi.predict(X[i_kal])))
    bawah, atas = lo.predict(X[i_uji]) - qc, hi.predict(X[i_uji]) + qc
    hasil["cqr_gbm"] = (float(np.mean((t[i_uji] >= bawah) & (t[i_uji] <= atas))),
                        float(np.median(atas - bawah)))

    def z_dan_konformal(mu_k, sd_k, mu_u, sd_u, nama):
        hasil[nama] = (float(np.mean(np.abs(t[i_uji] - mu_u) <= Z * sd_u)),
                       float(np.median(2 * Z * sd_u)))
        hasil[nama + "_konformal"] = konformal(t[i_kal], mu_k, t[i_uji], mu_u,
                                               sd_k, sd_u)

    mus_k, mus_u, vars_k, vars_u = [], [], [], []
    for b in range(N_ANGGOTA):
        m = calpir.latih_gauss(calpir.jaringan_gauss(Xf.shape[1], benih=b),
                               Xf, t[i_fit], b, EPOCH)
        ok, ou = m.predict(Xk, verbose=0), m.predict(Xu, verbose=0)
        mus_k.append(ok[:, 0]); vars_k.append(ok[:, 1] ** 2)
        mus_u.append(ou[:, 0]); vars_u.append(ou[:, 1] ** 2)
    z_dan_konformal(np.mean(mus_k, 0),
                    np.sqrt(np.mean(vars_k, 0) + np.var(mus_k, 0)) + 1e-6,
                    np.mean(mus_u, 0),
                    np.sqrt(np.mean(vars_u, 0) + np.var(mus_u, 0)) + 1e-6,
                    "ensemble_dalam")

    md = calpir.latih_gauss(calpir.jaringan_gauss(Xf.shape[1], dropout=0.2, benih=123),
                            Xf, t[i_fit], 123, EPOCH)
    # Dropout harus tetap menyala saat meramal, dan itu hanya terjadi bila model
    # dipanggil langsung dengan training=True, bukan lewat predict.
    lintas = lambda Xq: np.stack(
        [md(Xq, training=True).numpy()[:, 0] for _ in range(N_LINTASAN)])
    Pk, Pu = lintas(Xk), lintas(Xu)
    assert Pk.std(0).mean() > 1e-6, "dropout tidak aktif saat meramal"
    z_dan_konformal(Pk.mean(0), Pk.std(0) + 1e-6, Pu.mean(0), Pu.std(0) + 1e-6,
                    "mc_dropout")

    hasil["rf_sebar"] = (float(np.mean(np.abs(t[i_uji] - pu) <= Z * su)),
                         float(np.median(2 * Z * su)))
    return hasil


def pembanding(data):
    p, ada = tersimpan("hasil_pembanding")
    if ada:
        return
    keluaran = {"alpha": ALPHA, "n_ulang_A": N_ULANG_A, "satuan_lebar": "log10 ppm"}
    for alat in ALAT:
        X, y, t, level = data[alat]
        rng = np.random.default_rng(42)
        kump = {"A": {n: [] for n in CARA}, "B": {n: [] for n in CARA}}
        for r in range(N_ULANG_A):
            i_fit, i_kal, i_uji = belah(y, level, rng, 20, 20, 10)
            h = satu_partisi(X, y, i_fit, i_kal, i_uji)
            for n in CARA:
                if h[n][0] is not None:
                    kump["A"][n].append(h[n])
            print(f"  pembanding {alat} protokol A ulangan {r+1}", flush=True)
        for lv_out in level:
            sisa = level[level != lv_out]
            i_fit, i_kal, _ = belah(y, sisa, rng, 35, 15, 0)
            h = satu_partisi(X, y, i_fit, i_kal, np.flatnonzero(y == lv_out))
            for n in CARA:
                if h[n][0] is not None:
                    kump["B"][n].append(h[n])
            print(f"  pembanding {alat} protokol B level {lv_out}", flush=True)
        keluaran[alat] = {q: {n: {"cakupan": float(np.mean([x[0] for x in v])),
                                  "lebar": float(np.mean([x[1] for x in v])),
                                  "n": len(v)}
                              for n, v in kump[q].items() if v}
                          for q in ("A", "B")}
    tulis(p, keluaran)

    print(f"\nsasaran cakupan {1-ALPHA:.0%}, lebar dalam log10 ppm")
    print(f"{'alat':5} {'cara':26} {'A cakupan':>10} {'A lebar':>8} "
          f"{'B cakupan':>10} {'B lebar':>8}")
    for alat in ALAT:
        for n in CARA:
            a = keluaran[alat]["A"].get(n); b = keluaran[alat]["B"].get(n)
            if not a:
                continue
            print(f"{alat:5} {n:26} {a['cakupan']:10.3f} {a['lebar']:8.3f} "
                  + (f"{b['cakupan']:10.3f} {b['lebar']:8.3f}" if b
                     else f"{'-':>10} {'-':>8}"))


# ======================================================= metode B, penjaga
def peringkat_bersama(v, batas):
    o = np.argsort(np.argsort(v)) / max(len(v) - 1, 1)
    return np.split(o, batas)


def satu_level(X, y, lv_out, rng, dua_tahap):
    """Satu level disembunyikan seluruhnya, lalu tiga calon skor kebaruan diadu
    pada empat anggaran penolakan."""
    level = np.unique(y)
    sisa = level[level != lv_out]
    i_fit, i_kal, i_in = belah(y, sisa, rng, 25, 15, 10)
    i_out = np.flatnonzero(y == lv_out)

    lolos, _ = calpir.gerbang_bentuk(X[i_fit])
    if dua_tahap:
        g_kal, g_in, g_out = (lolos(X[i]) for i in (i_kal, i_in, i_out))
    else:
        g_kal, g_in, g_out = (np.ones(i.size, bool) for i in (i_kal, i_in, i_out))
    k_kal, k_in, k_out = i_kal[g_kal], i_in[g_in], i_out[g_out]

    reg = RandomForestRegressor(n_estimators=300, random_state=0,
                               n_jobs=-1).fit(X[i_fit], y[i_fit].astype(float))
    q = ambang(np.abs(y[k_kal] - reg.predict(X[k_kal])))
    tutup_in = np.abs(y[k_in] - reg.predict(X[k_in])) <= q
    tutup_out = np.abs(y[k_out] - reg.predict(X[k_out])) <= q if k_out.size else None

    cls = RandomForestClassifier(n_estimators=300, random_state=0,
                                 n_jobs=-1).fit(X[i_fit], y[i_fit])
    kol = {c: i for i, c in enumerate(cls.classes_)}
    P_kal = cls.predict_proba(X[k_kal])
    tak_cocok = {c: 1.0 - P_kal[y[k_kal] == c, kol[c]] for c in sisa}

    def skor_kon(idx):
        if idx.size == 0:
            return np.zeros(0)
        P = cls.predict_proba(X[idx])
        return 1.0 - np.array([
            max((np.sum(tak_cocok[c] >= 1.0 - P[j, kol[c]]) + 1) / (tak_cocok[c].size + 1)
                for c in sisa if tak_cocok[c].size) for j in range(idx.size)])

    s = {nm: {"jarak_knn": jarak_knn(X[idx], X[i_fit]) if idx.size else np.zeros(0),
              "konformal": skor_kon(idx)}
         for nm, idx in (("kal", k_kal), ("in", k_in), ("out", k_out))}
    gab = peringkat_bersama(
        (np.argsort(np.argsort(np.r_[s["kal"]["jarak_knn"], s["in"]["jarak_knn"],
                                     s["out"]["jarak_knn"]]))
         + np.argsort(np.argsort(np.r_[s["kal"]["konformal"], s["in"]["konformal"],
                                       s["out"]["konformal"]]))).astype(float),
        [k_kal.size, k_kal.size + k_in.size])
    for nm, v in zip(("kal", "in", "out"), gab):
        s[nm]["gabungan"] = v

    hasil = {}
    for g in PENJAGA_CALON:
        auc = (float(roc_auc_score(np.r_[np.zeros(k_in.size), np.ones(k_out.size)],
                                   np.r_[s["in"][g], s["out"][g]]))
               if k_out.size and k_in.size else None)
        per_beta = {}
        for b in BETA_SAPU:
            amb = float(np.percentile(s["kal"][g], 100 * (1 - b)))
            t_in = s["in"][g] > amb
            t_out = s["out"][g] > amb if k_out.size else np.zeros(0, bool)
            # laju tolak dihitung terhadap SELURUH sapuan, termasuk yang
            # sudah digugurkan gerbang
            per_beta[str(b)] = {
                "tolak_out": float((i_out.size - k_out.size + t_out.sum()) / i_out.size),
                "tolak_in": float((i_in.size - k_in.size + t_in.sum()) / i_in.size),
                "cakupan_in_diterima": (float(tutup_in[~t_in].mean())
                                        if (~t_in).any() else None),
                "cakupan_out_diterima": (float(tutup_out[~t_out].mean())
                                         if k_out.size and (~t_out).any() else None)}
        hasil[g] = {"auc": auc, "beta": per_beta}
    return (hasil, float(tutup_in.mean()),
            float(tutup_out.mean()) if tutup_out is not None and tutup_out.size else None,
            float(1 - g_in.mean()), float(1 - g_out.mean()))


def penjaga(data):
    p, ada = tersimpan("hasil_penjaga_v3")
    if ada:
        return
    keluaran = {"alpha": ALPHA, "beta": list(BETA_SAPU), "buang_gerbang": BUANG_GERBANG}
    for dua in (False, True):
        kunci = "dua_tahap" if dua else "satu_tahap"
        keluaran[kunci] = {}
        for alat in ALAT:
            X, y, t, level = data[alat]
            rng = np.random.default_rng(42)
            per_level, c_in, c_out, gg_in, gg_out = {}, [], [], [], []
            for lv in level:
                h, a, b, gi, go = satu_level(X, y, lv, rng, dua)
                per_level[int(lv)] = h
                c_in.append(a); gg_in.append(gi); gg_out.append(go)
                if b is not None:
                    c_out.append(b)
                print(f"  penjaga {kunci} {alat} level {lv:>4}", flush=True)
            ringkas = {}
            for g in PENJAGA_CALON:
                v = [per_level[k][g] for k in per_level]
                ringkas[g] = {"auc": float(np.mean([x["auc"] for x in v
                                                    if x["auc"] is not None])),
                              "beta": {}}
                for b in BETA_SAPU:
                    bb = [x["beta"][str(b)] for x in v]
                    rata = lambda k: float(np.mean([x[k] for x in bb
                                                    if x[k] is not None]))
                    ringkas[g]["beta"][str(b)] = {
                        "tolak_out": rata("tolak_out"), "tolak_in": rata("tolak_in"),
                        "cakupan_in_diterima": rata("cakupan_in_diterima"),
                        "cakupan_out_diterima": rata("cakupan_out_diterima"),
                        "level_tolak_penuh": int(sum(x["tolak_out"] == 1.0 for x in bb))}
            keluaran[kunci][alat] = {
                "cakupan_in_setelah_gerbang": float(np.mean(c_in)),
                "cakupan_out_setelah_gerbang": float(np.mean(c_out)) if c_out else None,
                "gerbang_gugurkan_in": float(np.mean(gg_in)),
                "gerbang_gugurkan_out": float(np.mean(gg_out)),
                "ringkas": ringkas, "per_level": per_level}
    tulis(p, keluaran)

    for kunci in ("satu_tahap", "dua_tahap"):
        print(f"\n## {kunci}")
        for alat in ALAT:
            k = keluaran[kunci][alat]
            print(f"  {alat}: cakupan_in {k['cakupan_in_setelah_gerbang']:.3f}  "
                  f"gerbang gugurkan in {k['gerbang_gugurkan_in']:.3f} "
                  f"out {k['gerbang_gugurkan_out']:.3f}")
        print(f"  {'alat':5} {'penjaga':11} {'AUC':>6} "
              + " ".join(f"{'out/in b='+str(b):>18}" for b in BETA_SAPU)
              + f" {'cakup_in_dt':>12}")
        for alat in ALAT:
            for g in PENJAGA_CALON:
                r = keluaran[kunci][alat]["ringkas"][g]
                baris = " ".join(f"{r['beta'][str(b)]['tolak_out']:8.3f}/"
                                 f"{r['beta'][str(b)]['tolak_in']:<9.3f}"
                                 for b in BETA_SAPU)
                print(f"  {alat:5} {g:11} {r['auc']:6.3f} {baris} "
                      f"{r['beta']['0.05']['cakupan_in_diterima']:12.3f}")


# ================================================== kontrol, kebaruan atau sesi
def sesi(data):
    """Tiap level diukur dari satu penyiapan larutan dalam satu sesi, 50 berkas
    berurutan. Maka level yang disembunyikan berbeda bukan hanya konsentrasinya,
    tetapi juga larutan dan keadaan elektrodanya. Kontrol ini memisahkan keduanya
    sejauh mungkin.

      uji 1  di DALAM satu level konsentrasinya tetap, sehingga hubungan skor
             penjaga dengan nomor berkas atau tinggi puncak murni dari sesi
      uji 2  A level disembunyikan, B blok berkas terakhir pada level yang sama,
             C berkas acak menyebar. Bila posisi blok yang menolak, B jauh di
             atas C.
    """
    p, ada = tersimpan("hasil_kontrol_sesi")
    if ada:
        return
    keluaran = {}
    for alat in ALAT:
        X, y, t, level = data[alat]
        d = muat_sapuan(alat)
        assert np.array_equal(d["y"], y), "urutan baris tidak sepadan"
        puncak = np.abs(np.median(d["I"][:, slice(2, 10), :], axis=1)).max(axis=1)
        nomor = np.concatenate([np.arange((y == lv).sum()) for lv in level])

        rng = np.random.default_rng(42)
        i_fit = np.concatenate([np.flatnonzero(y == lv)[:25] for lv in level])
        s_semua = jarak_knn(X, X[i_fit])
        u1 = {}
        for lv in level:
            m = y == lv
            r_nomor, p_nomor = spearmanr(nomor[m], s_semua[m])
            r_puncak, p_puncak = spearmanr(puncak[m], s_semua[m])
            u1[int(lv)] = {"rho_nomor": float(r_nomor), "p_nomor": float(p_nomor),
                           "rho_puncak": float(r_puncak), "p_puncak": float(p_puncak)}

        A = []
        for lv in level:
            sisa = level[level != lv]
            fit = np.concatenate([np.flatnonzero(y == s)[:25] for s in sisa])
            kal = np.concatenate([np.flatnonzero(y == s)[25:40] for s in sisa])
            A.append(penjaga_dua_tahap(X[fit], X[kal], X[np.flatnonzero(y == lv)])[0])
        potong = lambda a, b: np.concatenate(
            [np.flatnonzero(y == lv)[a:b] for lv in level])
        B = penjaga_dua_tahap(X[potong(0, 25)], X[potong(25, 40)], X[potong(40, 50)])[0]
        c_ulang = []
        for _ in range(10):
            i_f, i_k, i_u = belah(y, level, rng, 25, 15, 10)
            c_ulang.append(penjaga_dua_tahap(X[i_f], X[i_k], X[i_u])[0])

        keluaran[alat] = {
            "uji1_dalam_level": u1,
            "uji1_ringkas": {
                "rho_nomor_median_abs": float(np.median(
                    [abs(v["rho_nomor"]) for v in u1.values()])),
                "rho_puncak_median_abs": float(np.median(
                    [abs(v["rho_puncak"]) for v in u1.values()])),
                "level_nomor_nyata": int(sum(v["p_nomor"] < 0.01
                                             and abs(v["rho_nomor"]) > 0.5
                                             for v in u1.values())),
                "level_puncak_nyata": int(sum(v["p_puncak"] < 0.01
                                              and abs(v["rho_puncak"]) > 0.5
                                              for v in u1.values()))},
            "uji2": {"A_level_disembunyikan": float(np.mean(A)),
                     "B_blok_terakhir": B,
                     "C_berkas_menyebar": float(np.mean(c_ulang)),
                     "C_sd": float(np.std(c_ulang))}}
    tulis(p, keluaran)

    print("\nuji 1, di dalam level, konsentrasi tetap")
    print(f"  {'alat':5} {'|rho| nomor':>12} {'nyata':>8} {'|rho| puncak':>13} {'nyata':>8}")
    for alat in ALAT:
        r = keluaran[alat]["uji1_ringkas"]
        n = len(keluaran[alat]["uji1_dalam_level"])
        print(f"  {alat:5} {r['rho_nomor_median_abs']:12.3f} "
              f"{r['level_nomor_nyata']:5d}/{n:<2d} {r['rho_puncak_median_abs']:13.3f} "
              f"{r['level_puncak_nyata']:5d}/{n:<2d}")
    print("\nuji 2, laju tolak penjaga pada tiga kondisi")
    print(f"  {'alat':5} {'A level baru':>13} {'B blok akhir':>13} {'C menyebar':>12}")
    for alat in ALAT:
        u = keluaran[alat]["uji2"]
        print(f"  {alat:5} {u['A_level_disembunyikan']:13.3f} {u['B_blok_terakhir']:13.3f} "
              f"{u['C_berkas_menyebar']:12.3f} sd {u['C_sd']:.3f}")


URUTAN = ("interval", "ensemble", "bersarang", "benih", "pembanding", "penjaga", "sesi")

if __name__ == "__main__":
    pilih = [a for a in sys.argv[1:] if not a.startswith("--")] or list(URUTAN)
    salah = [a for a in pilih if a not in URUTAN]
    assert not salah, f"percobaan tidak dikenal: {salah}, pilihannya {URUTAN}"
    print(f"mode {'CEPAT' if CEPAT else 'PENUH'}, alat {ALAT}, percobaan {pilih}")
    d = muat(ALAT)
    for nama in URUTAN:
        if nama in pilih:
            t0 = time.time()
            globals()[nama](d)
            print(f"== {nama} {(time.time() - t0) / 60:.1f} menit\n", flush=True)

"""CALPIR, kedua bagian metodenya beserta pembantu yang dipakai bersama.

  Metode A  interval prediksi konformal yang lebarnya diskalakan oleh
            ketidaksepakatan antar anggota ensembel
  Metode B  aturan penolakan dua tahap, gerbang bentuk lalu skor kebaruan

Berkas ini tidak menjalankan percobaan apa pun sendiri. Seluruh percobaan naskah
ada di percobaan.py dan seluruh gambar ada di gambar.py.

Berkas sumbu_sama.npz serta scans_LC.npz dan scans_EM.npz dibuat oleh
pra_proses.py dan dibaca dari data/, atau dari direktori CALPIR_DATA.
"""
import os

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

AKAR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Data dicari di data/, lalu di folder kerja lama bila ada. Gambar ditulis ke
# paper/figures bila naskah ada di sebelahnya, selain itu ke figures/.
_LAMA = os.path.join(AKAR, os.pardir, "Paper_V2Taufiq", "results")
DATA = os.environ.get("CALPIR_DATA") or (
    _LAMA if not os.path.isdir(os.path.join(AKAR, "data")) and os.path.isdir(_LAMA)
    else os.path.join(AKAR, "data"))
HASIL = os.path.join(AKAR, "results")
GAMBAR = (os.path.join(AKAR, "paper", "figures") if os.path.isdir(os.path.join(AKAR, "paper"))
          else os.path.join(AKAR, "figures"))

ALPHA = 0.10             # 1 - ALPHA adalah cakupan yang dijanjikan
BETA = 0.05              # anggaran penolakan di dalam ruang kalibrasi
BUANG_GERBANG = 0.10     # pecahan data latih yang dianggap bentuk pinggiran
ALAT = ("LC", "EM")      # potentiostat murah dan potentiostat niaga
MODEL = ("RF", "ET", "XGB", "LGBM", "SVR", "PLS")
PUNYA_POHON = {"RF", "ET"}


# --------------------------------------------------------------------- data
def muat(alat=ALAT):
    """Kembalikan {alat: (X, y, log10 y, level)}."""
    z = np.load(os.path.join(DATA, "sumbu_sama.npz"))
    d = {}
    for a in alat:
        X, y = z[f"X_{a}"], z[f"y_{a}"]
        assert np.isfinite(X).all(), "data memuat NaN"
        d[a] = (X, y, np.log10(y).astype(float), np.unique(y))
    return d


def muat_sapuan(alat):
    """Voltamogram mentah satu alat, untuk gambar dan untuk kontrol sesi."""
    return np.load(os.path.join(DATA, f"scans_{alat}.npz"))


# ----------------------------------------------------------------- partisi
def belah(y, level, rng, n_fit, n_kal, n_uji):
    """Belah berstrata level menjadi latih, kalibrasi, dan uji."""
    a, b, c = [], [], []
    for lv in level:
        idx = rng.permutation(np.flatnonzero(y == lv))
        a += list(idx[:n_fit]); b += list(idx[n_fit:n_fit + n_kal])
        c += list(idx[n_fit + n_kal:n_fit + n_kal + n_uji])
    return map(np.array, (a, b, c))


def belah_dalam(idx, y, rng, n1, n2, n3):
    """Belah satu himpunan indeks menjadi tiga bagian, seimbang per level."""
    a, b, c = [], [], []
    for lv in np.unique(y[idx]):
        sub = rng.permutation(idx[y[idx] == lv])
        a += list(sub[:n1]); b += list(sub[n1:n1 + n2]); c += list(sub[n1 + n2:n1 + n2 + n3])
    return map(np.array, (a, b, c))


def periksa_partisi(*idx):
    for i in range(len(idx)):
        for j in range(i + 1, len(idx)):
            assert np.intersect1d(idx[i], idx[j]).size == 0, "partisi bertumpang tindih"


# ------------------------------------------------------------ model dasar
def buat_model(nama, benih):
    if nama == "RF":
        return RandomForestRegressor(n_estimators=300, random_state=benih, n_jobs=-1)
    if nama == "ET":
        return ExtraTreesRegressor(n_estimators=300, random_state=benih, n_jobs=-1)
    if nama == "XGB":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.1,
                            random_state=benih, n_jobs=-1, verbosity=0)
    if nama == "LGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=300, random_state=benih, n_jobs=-1, verbose=-1)
    if nama == "SVR":
        return make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale"))
    if nama == "PLS":
        return make_pipeline(StandardScaler(), PLSRegression(n_components=15))
    raise ValueError(nama)


def ramal(m, X):
    return np.asarray(m.predict(X)).ravel()


# ========================================================== METODE A
def ambang(r, alpha=ALPHA):
    """Kuantil konformal, yaitu sisa terurut ke ceil((n+1)(1-alfa)).

    Ambang nol sah dan memang terjadi, sebab sasarannya diskret dan ensembel
    pohon tumbuh penuh meramalkan sebagian ulangan kalibrasi tepat benar.
    """
    r = np.sort(np.asarray(r, float)); n = r.size
    q = float(r[min(int(np.ceil((n + 1) * (1 - alpha))), n) - 1])
    assert np.isfinite(q) and q >= 0, f"ambang tidak sah: n={n} alfa={alpha} q={q}"
    return q


def skala_sebar(m, X, idx, lantai=1e-4):
    """Skala setempat dari simpangan baku ramalan antar anggota ensembel."""
    return np.std([e.predict(X[idx]) for e in m.estimators_], axis=0) + lantai


def skala_terlatih(nama, X, t, i_fit, benih, X_kal, X_uji):
    """Pembanding skala: model kedua dilatih meramalkan besar galat.

    Sisanya diambil dari lipat-silang DI DALAM himpunan latih saja, sehingga
    kalibrasi dan uji tetap tak tersentuh.
    """
    oof = np.asarray(cross_val_predict(buat_model(nama, benih), X[i_fit],
                                       t[i_fit], cv=5)).ravel()
    g = GradientBoostingRegressor(n_estimators=150, random_state=benih)
    g.fit(X[i_fit], np.abs(t[i_fit] - oof))
    return np.maximum(ramal(g, X_kal), 1e-4), np.maximum(ramal(g, X_uji), 1e-4)


def konformal(t_kal, p_kal, t_uji, p_uji, sk=None, su=None, alpha=ALPHA,
              y_kal=None, y_uji=None):
    """Cakupan dan lebar median satu interval konformal.

    sk dan su adalah skala setempat pada kalibrasi dan uji; None berarti tanpa
    skala. Memberi y_kal dan y_uji memilih ragam Mondrian, yaitu satu ambang
    tiap level alih-alih satu ambang bersama. Lebar dibaca dalam dekade, jadi
    lebar w berarti kadar dilaporkan dalam faktor 10**(w/2).
    """
    if sk is None:
        sk, su = np.ones(np.size(t_kal)), np.ones(np.size(t_uji))
    if y_kal is None:
        q = ambang(np.abs(t_kal - p_kal) / sk, alpha)
        c = float(np.mean(np.abs(t_uji - p_uji) <= q * su))
        w = float(np.median(2 * q * su))
    else:
        tutup, lebar = [], []
        for lv in np.unique(y_kal):
            mk, mu = y_kal == lv, y_uji == lv
            if not mk.any() or not mu.any():
                continue
            q = ambang(np.abs(t_kal[mk] - p_kal[mk]) / sk[mk], alpha)
            tutup += list(np.abs(t_uji[mu] - p_uji[mu]) <= q * su[mu])
            lebar.append(float(np.median(2 * q * su[mu])))
        if not tutup:
            return None, None
        c, w = float(np.mean(tutup)), float(np.median(lebar))
    assert 0.0 <= c <= 1.0 and w >= 0 and np.isfinite(w), \
        f"cakupan atau lebar tidak sah: c={c} w={w}"
    return c, w


# ========================================================== METODE B
def snv(X):
    """Baku normal peubah, membuang taraf dan besaran tiap sapuan."""
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-12)


def jarak_knn(X, X_acuan, k=5, sama=False):
    """Rerata jarak Euclid ke k tetangga terdekat di X_acuan."""
    d = np.sqrt(((X[:, None, :] - X_acuan[None, :, :]) ** 2).sum(-1))
    if sama:
        np.fill_diagonal(d, np.inf)
    return np.sort(d, axis=1)[:, :k].mean(1)


def gerbang_bentuk(X_fit, buang=BUANG_GERBANG):
    """Tahap satu. Ambang jarak bentuk ditetapkan dari data latih saja, tanpa
    label, lalu dikenakan pada kalibrasi dan uji dengan aturan yang sama.

    Kesamaan perlakuan itulah yang menjaga keterukaran, sehingga anggaran
    penolakan dan jaminan cakupan tetap bermakna di dalam upapopulasi yang
    lolos gerbang.
    """
    Zf = snv(X_fit)
    batas = float(np.percentile(jarak_knn(Zf, Zf, sama=True), 100 * (1 - buang)))
    return (lambda Xq: jarak_knn(snv(Xq), Zf) <= batas), batas


def penjaga_dua_tahap(X_fit, X_kal, X_uji, beta=BETA, buang=BUANG_GERBANG):
    """Kembalikan (laju tolak, lolos gerbang, tolak tahap dua, ambang tahap dua).

    Laju tolak dihitung terhadap SELURUH sapuan uji, termasuk yang sudah
    digugurkan gerbang.
    """
    lolos, _ = gerbang_bentuk(X_fit, buang)
    g_kal = lolos(X_kal)
    amb = float(np.percentile(jarak_knn(X_kal[g_kal], X_fit), 100 * (1 - beta)))
    g_uji = lolos(X_uji)
    t2 = (jarak_knn(X_uji[g_uji], X_fit) > amb) if g_uji.any() else np.zeros(0, bool)
    tolak = (X_uji.shape[0] - g_uji.sum() + t2.sum()) / X_uji.shape[0]
    return float(tolak), g_uji, t2, amb


# ---------------------------------------------- jaringan pembanding
def jaringan_gauss(n_masuk, unit=(64, 32), dropout=0.0, lr=1e-3, benih=0):
    """Jaringan berkepala Gauss, dipakai sebagai pembanding ensembel dalam dan
    MC dropout. Bukan bagian CALPIR."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(benih)
    x = tf.keras.Input(shape=(n_masuk,))
    h = x
    for u in unit:
        h = tf.keras.layers.Dense(u, activation="relu")(h)
        if dropout:
            h = tf.keras.layers.Dropout(dropout)(h)
    m = tf.keras.Model(x, tf.keras.layers.Concatenate()(
        [tf.keras.layers.Dense(1)(h),
         tf.keras.layers.Dense(1, activation="softplus")(h)]))

    def nll(yt, yp):
        # (sigma + 1e-6) sengaja ditulis dua kali dan tidak disimpan ke peubah.
        # Bentuk grafnya menentukan urutan penjumlahan gradien, dan menyatukan
        # anak-ungkapan itu menggeser hasil float32 pada digit terakhirnya.
        return tf.reduce_mean(
            0.5 * tf.math.log((yp[:, 1:] + 1e-6) ** 2)
            + 0.5 * (yt - yp[:, :1]) ** 2 / (yp[:, 1:] + 1e-6) ** 2)

    m.compile(tf.keras.optimizers.Adam(lr), loss=nll)
    return m


def latih_gauss(m, Xs, tt, benih, epoch=200):
    """Baris diacak lebih dahulu. Tanpa itu validation_split mengambil 15 persen
    baris terakhir, dan karena himpunan latih terurut menurut level, validasinya
    hanya berisi konsentrasi tertinggi."""
    import tensorflow as tf
    o = np.random.default_rng(benih).permutation(Xs.shape[0])
    assert not np.all(np.diff(tt[o]) >= 0), "baris belum teracak, validasi akan bias"
    m.fit(Xs[o], tt[o].reshape(-1, 1), epochs=epoch, batch_size=32, verbose=0,
          validation_split=0.15,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=20,
                                                      restore_best_weights=True)])
    return m

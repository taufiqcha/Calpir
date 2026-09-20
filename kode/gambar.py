"""Seluruh gambar naskah CALPIR beserta abstrak grafisnya.

  skema    Gambar 1 sampai 3 sebagai berkas draw.io, yaitu kerangka keseluruhan,
           pengambilan data bergantian dua alat, dan alur keputusan CALPIR
  hasil    Gambar 4 sampai 6 dari berkas JSON di ../results
  abstrak  abstrak grafis, dipatok 13 x 5 cm sesuai syarat penerbit

Palet dan ukuran huruf sama untuk seluruh gambar, yaitu dua warna utama saja,
biru dan oranye, tanpa hijau dan merah, dan lebar tepat sebesar kolom atau
halaman kelas cas-dc sehingga gambar terpasang pada skala satu dan hurufnya
tidak mengecil.

Pemakaian:
    python3 gambar.py              seluruhnya
    python3 gambar.py hasil        hanya yang disebut
"""
import json
import os
import sys
from xml.sax.saxutils import escape

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

import calpir
from calpir import GAMBAR, HASIL, muat_sapuan

DRAWIO = os.path.join(GAMBAR, "drawio")
BIRU, BIRU_MUDA, BIRU_TUA = "#0072B2", "#8EC6E6", "#00436B"
ORNG, ORNG_MUDA = "#E69F00", "#F5CE86"
BIRU_ISI, ORNG_ISI = "#EAF3FA", "#FDF0DC"
ABU, ABU_TUA = "#666666", "#555555"
HIJAU = "#009E73"
ABU_ISI, ABU_GRS, HIJAU_ISI = "#F0F0F0", "#9A9A9A", "#E3F3EE"
KOLOM, HALAMAN = 240 / 72, 494.51 / 72          # lebar cetak dalam inci
CM = 1 / 2.54

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial"],
    "font.size": 8.5, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
})


def simpan(fig, nama, jenis=("pdf",), ketat=True):
    """Gambar naskah dipangkas tepinya, abstrak grafis tidak.

    Memberi bbox_inches=None tidak cukup untuk mematikan pemangkasan, sebab
    matplotlib lalu membacanya dari savefig.bbox yang sudah disetel tight di
    atas. Ukurannya harus dipatok lewat rc_context.
    """
    os.makedirs(GAMBAR, exist_ok=True)
    with plt.rc_context({} if ketat else {"savefig.bbox": None,
                                          "savefig.pad_inches": 0}):
        for ext in jenis:
            p = os.path.join(GAMBAR, f"{nama}.{ext}")
            fig.savefig(p)
            print("ditulis", p)
    plt.close(fig)


# ======================================================= skema, berkas draw.io
def esc(t):
    """Nilai draw.io disimpan sebagai atribut XML, sehingga tag HTML di dalamnya
    wajib di-escape. Tanpa ini berkasnya tidak well-formed, dan draw.io hanya
    merender sel yang selamat tanpa memunculkan galat apa pun."""
    return escape(t).replace('"', "&quot;")


# draw.io mengekspor PDF pada 0,75 pt tiap piksel, jadi lebar isi dalam piksel
# adalah lebar cetak dibagi 0,75. Dua kolom 494,51 pt -> 660 px, satu kolom
# 240 pt -> 334 px.
W2, W1 = 660, 334
KOTAK = ("rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor={isi};"
         "strokeColor={garis};strokeWidth=1.2;fontFamily=Helvetica;"
         "fontSize={fs};verticalAlign=middle;align=center;spacing=4;")
BELAH = ("rhombus;whiteSpace=wrap;html=1;fillColor={isi};strokeColor={garis};"
         "strokeWidth=1.2;fontFamily=Helvetica;fontSize={fs};align=center;")
TEKS = ("text;html=1;strokeColor=none;fillColor=none;align={al};"
        "verticalAlign=middle;fontFamily=Helvetica;fontSize={fs};fontColor=#333333;")
GARIS = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#444444"
         ";strokeWidth=1.2;endArrow=block;endFill=1;fontFamily=Helvetica;fontSize=10;labelBackgroundColor=#FFFFFF;fontColor=#333333;")


GARIS_PUTUS = ("edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;strokeColor=#8A8A8A"
               ";strokeWidth=1.0;dashed=1;dashPattern=4 3;endArrow=open;endFill=0;"
               "fontFamily=Helvetica;fontSize=9;labelBackgroundColor=#FFFFFF;"
               "fontColor=#555555;")


def panah_putus(i, a, b, label="", keluar=None, masuk=None):
    gaya = GARIS_PUTUS
    if keluar:
        gaya += f"exitX={keluar[0]};exitY={keluar[1]};exitDx=0;exitDy=0;"
    if masuk:
        gaya += f"entryX={masuk[0]};entryY={masuk[1]};entryDx=0;entryDy=0;"
    return (f'        <mxCell id="{i}" value="{esc(label)}" style="{gaya}" edge="1"'
            f' parent="1" source="{a}" target="{b}"><mxGeometry relative="1"'
            ' as="geometry"/></mxCell>')


def bingkai(sel, tinggi, lebar=W2):
    return f"""<mxfile host="app.diagrams.net">
  <diagram name="Page-1">
    <mxGraphModel dx="800" dy="600" grid="0" gridSize="10" guides="1" tooltips="1"
      connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="{lebar}"
      pageHeight="{tinggi}" math="0" shadow="0">
      <root>
        <mxCell id="0" /><mxCell id="1" parent="0" />
{sel}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""


def _sel(i, nilai, gaya, x, y, w, h):
    return (f'        <mxCell id="{i}" value="{esc(nilai)}" style="{gaya}"'
            f' vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}"'
            f' height="{h}" as="geometry"/></mxCell>')


def kotak(i, t, x, y, w, h, isi=BIRU_ISI, garis=BIRU, fs=11):
    return _sel(i, t, KOTAK.format(isi=isi, garis=garis, fs=fs), x, y, w, h)


def belah(i, t, x, y, w, h, fs=10):
    return _sel(i, t, BELAH.format(isi=ORNG_ISI, garis=ORNG, fs=fs), x, y, w, h)


def teks(i, t, x, y, w, h, al="center", fs=9):
    return _sel(i, t, TEKS.format(al=al, fs=fs), x, y, w, h)


def panah(i, a, b, label="", keluar=None, masuk=None):
    gaya = GARIS
    if keluar:
        gaya += f"exitX={keluar[0]};exitY={keluar[1]};exitDx=0;exitDy=0;"
    if masuk:
        gaya += f"entryX={masuk[0]};entryY={masuk[1]};entryDx=0;entryDy=0;"
    return (f'        <mxCell id="{i}" value="{esc(label)}" style="{gaya}" edge="1"'
            f' parent="1" source="{a}" target="{b}"><mxGeometry relative="1"'
            ' as="geometry"/></mxCell>')


def gambar1():
    """Kerangka CALPIR. Warna menandai asal tiap bagian, angka menandai
    ukuran data, dan tiap kotak menunjuk persamaan serta bagian naskahnya.

    Lebar dan posisi kotak mengikuti berkas .drawio yang sudah disetel tangan.
    Ubah isi kotak saja, jangan ubah angka geometri di bawah ini."""
    w_jdl, kol_jdl = 156, [4, 168, 332, 496]      # judul di atas kotak
    kol = [4, 172, 340, 502]                       # posisi kotak, hasil setelan
    lbr = [140, 140, 140, 150]                     # lebar kotak, hasil setelan
    judul = ["Paired acquisition", "Signal representation",
             "Ensemble fitting", "CALPIR, proposed"]
    isi = ["two potentiostats, one cell<br><br>15 levels, 2 to 1000 ppm"
           "<br>50 replicates per level<br>1500 voltammograms<br><br>"
           "Sec. 2.2, Table 2",
           "one shared potential grid<br><br>178 current variables"
           "<br>-1.4 to -0.5 V<br>no baseline correction<br><br>Sec. 2.3",
           "fitting partition only<br><br>20 fitting, 20 calibration,"
           "<br>10 test per level<br><br>tree ensemble members"
           "<br>member mean, Eq. (3)<br>member spread, Eq. (4)<br><br>Sec. 2.4"]
    sel = []
    for k in range(4):
        sel.append(teks(f"t{k}", judul[k], kol_jdl[k], 4, w_jdl, 16, fs=10))
    for k in range(3):
        warna = (ABU_ISI, ABU_GRS) if k < 2 else (BIRU_ISI, BIRU)
        sel.append(kotak(f"b{k}", isi[k], kol[k], 22, lbr[k], 148, warna[0], warna[1], 9.5))
    # tahap empat adalah wadah dengan dua bagian CALPIR di dalamnya
    sel.append(kotak("b3", "", kol[3], 22, lbr[3], 148, ORNG_ISI, ORNG, 9.5))
    sel.append(kotak("c1", "interval scaled by<br>member disagreement<br>Eq. (5) to (7)",
                     509, 34, 135, 60, "#FFFFFF", ORNG, 9.5))
    sel.append(kotak("c2", "two-stage rejection rule<br>shape gate, Eq. (8), (9)"
                     "<br>neighbor distance, Eq. (10)", 509, 100, 135, 62,
                     "#FFFFFF", ORNG, 9.5))
    for k in range(1, 4):
        sel.append(panah(f"e{k}", f"b{k-1}", f"b{k}"))
    sel.append(teks("sp", "", 4, 176, 320, 16, "left", 9))
    # dua keluaran yang mungkin
    sel.append(teks("ot", "Outcome for every voltammogram", 4, 206, 320, 40, "right", 10))
    sel.append(kotak("o1", "accepted<br>concentration with a 90% interval",
                     332, 202, 156, 46, BIRU_ISI, BIRU, 9.5))
    sel.append(kotak("o2", "rejected<br>no number reported",
                     496, 202, 156, 46, ORNG_ISI, ORNG, 9.5))
    sel.append(panah("f1", "b3", "o1", keluar=(0.35, 1), masuk=(0.5, 0)))
    sel.append(panah("f2", "b3", "o2", keluar=(0.65, 1), masuk=(0.5, 0)))
    # kunci warna
    kunci = ((ABU_ISI, ABU_GRS, "data and representation"),
             (BIRU_ISI, BIRU, "model fitted from data"),
             (ORNG_ISI, ORNG, "proposed in this study"))
    xk = 118
    for k, (isi_w, grs, t) in enumerate(kunci):
        sel.append(kotak(f"k{k}", "", xk, 268, 20, 14, isi_w, grs, 8))
        sel.append(teks(f"kt{k}", t, xk + 26, 267, 150, 16, "left", 9.5))
        xk += 176
    return bingkai("\n".join(sel), 292)


def gambar2():
    """Akuisisi berselang-seling. Biru menandai alat murah dan oranye
    menandai alat komersial, pada kotak alat maupun pada garis waktu."""
    sel = [
        teks("hd", "only one instrument is connected to the cell at a time",
             4, 6, 648, 16, fs=10),
        kotak("lc", "Low-cost potentiostat<br>Rodeostat platform<br>"
              "Adafruit ItsyBitsy M4 Express", 4, 30, 172, 78, BIRU_ISI, BIRU, 9),
        kotak("cell", "Electrochemical cell<br>carbon working, Ag/AgCl reference"
              "<br>0.1 mol/L KCl, pH 7.0<br>-1.4 to -0.5 V at 0.2 V/s",
              228, 30, 204, 78, ABU_ISI, ABU_GRS, 9),
        kotak("em", "Commercial potentiostat<br>EmStat4", 484, 30, 168, 78,
              ORNG_ISI, ORNG, 9),
        panah("e1", "cell", "lc", "block k"),
        panah("e2", "cell", "em", "block k+1"),
        teks("ac", "Acquisition order", 4, 128, 648, 16, fs=10),
    ]
    blok = [("block 1", "low-cost"), ("block 2", "commercial"),
            ("block 3", "low-cost"), ("block 4", "commercial"),
            ("block 5", "low-cost"), ("block 6", "commercial")]
    xb = 4
    for k, (a, b) in enumerate(blok):
        warna = (BIRU_ISI, BIRU) if b == "low-cost" else (ORNG_ISI, ORNG)
        sel.append(kotak(f"g{k}", a + "<br>" + b, xb, 150, 86, 36, warna[0], warna[1], 9))
        xb += 92
    sel.append(teks("dots", "...", xb, 150, 40, 36, fs=13))
    sel.append(kotak("gl", "block 1500<br>commercial", xb + 46, 150, 86, 36,
                     ORNG_ISI, ORNG, 9))
    sel.append(teks("nb", "each block is one replicate, that is 10 consecutive scans",
                    4, 192, 648, 16, fs=9))
    sel.append(kotak("hold", "held constant<br>solution, electrode set, session, "
                     "acquisition parameters", 4, 218, 318, 44, HIJAU_ISI, HIJAU, 9))
    sel.append(kotak("vary", "varied<br>the instrument alone",
                     338, 218, 314, 44, "#FFFFFF", ABU_GRS, 9))
    sel.append(teks("tot", "15 levels from 2 to 1000 ppm, 50 replicates per level per "
                    "instrument, 1500 voltammograms, plus 50 blank replicates per "
                    "instrument", 4, 268, 648, 16, fs=9))
    return bingkai("\n".join(sel), 292)


def gambar3():
    """Alur keputusan CALPIR. Panel kiri memuat nilai yang ditetapkan sekali di
    muka, panel kanan memuat yang dikerjakan pada tiap voltamogram baru."""
    sel = [
        # ---------- panel kiri, disiapkan sekali ----------
        kotak("pp", "", 4, 22, 240, 262, ORNG_ISI, ORNG, 9),
        teks("pt", "Prepared once, before any new measurement", 4, 4, 240, 16, fs=10),
        kotak("p1", "shape threshold<br>90th percentile of the shape distance"
              "<br>over the training measurements", 12, 40, 224, 58,
              "#FFFFFF", ORNG, 9.5),
        kotak("p2", "neighbor threshold<br>quantile of the neighbor distance over the"
              "<br>calibration measurements, set by the"
              "<br>target rejection rate", 12, 122, 224, 66,
              "#FFFFFF", ORNG, 9.5),
        kotak("p3", "conformal quantile, Eq. (5), (6)"
              "<br>rank quantile of the scaled residuals"
              "<br>over the calibration measurements", 12, 214, 224, 58,
              "#FFFFFF", ORNG, 9.5),
        # ---------- panel kanan, tiap voltamogram baru ----------
        teks("rt", "Applied to each new voltammogram", 262, 4, 394, 16, fs=10),
        kotak("in", "voltammogram", 379, 26, 160, 26, BIRU_ISI, BIRU, 10),
        belah("g1", "shape distance in range?<br>Eq. (8), (9)", 359, 60, 200, 58, fs=9.5),
        belah("g2", "neighbor distance below threshold?<br>Eq. (10)", 359, 130, 200, 58, fs=9.5),
        kotak("fit", "ensemble mean and member spread<br>Eq. (3), (4)",
              364, 200, 190, 36, BIRU_ISI, BIRU, 9.5),
        kotak("iv", "report interval, Eq. (7)", 379, 250, 160, 26, ORNG_ISI, ORNG, 10),
        kotak("rf", "reject", 584, 136, 64, 26, ORNG_ISI, ORNG, 10),
        panah("a1", "in", "g1"),
        panah("a2", "g1", "g2", "yes"),
        panah("a3", "g2", "fit", "yes"),
        panah("a4", "fit", "iv"),
        panah("a5", "g1", "rf", "no", keluar=(1, 0.5)),
        panah("a6", "g2", "rf", "no", keluar=(1, 0.5)),
        panah_putus("d1", "p1", "g1", keluar=(1, 0.5), masuk=(0, 0.5)),
        panah_putus("d2", "p2", "g2", keluar=(1, 0.5), masuk=(0, 0.5)),
        panah_putus("d3", "p3", "iv", keluar=(1, 0.5), masuk=(0, 0.5)),
        teks("kk", "dashed lines carry a value fixed in advance",
             250, 288, 300, 14, "left", 9),
    ]
    return bingkai("\n".join(sel), 304)


def skema(paksa=()):
    """Berkas .drawio bisa disetel tangan di aplikasi draw.io. Menulis ulang
    tanpa diminta akan menghapus setelan itu, jadi berkas yang sudah ada
    dilewati kecuali namanya disebut di paksa."""
    os.makedirs(DRAWIO, exist_ok=True)
    for nama, fn in (("Figure_1", gambar1), ("Figure_2", gambar2), ("Figure_3", gambar3)):
        p = os.path.join(DRAWIO, nama + ".drawio")
        if os.path.exists(p) and nama not in paksa:
            print("dilewati, sudah ada:", p)
            continue
        open(p, "w").write(fn())
        print("ditulis", p)
    print("ekspor ke PDF lewat draw.io, skala 1, --crop")


# ============================================================ hasil, matplotlib
def baca(nama):
    return json.load(open(os.path.join(HASIL, nama + ".json")))


def judul_alat(alat):
    return "Low-cost" if alat == "LC" else "Commercial"


def gambar5():
    """Sepuluh cara menakar ketidakpastian, lebar lawan cakupan."""
    d = baca("hasil_pembanding")
    nama = {"konformal_rf_ternormalkan": "CALPIR, ensemble spread",
            "konformal_rf_mondrian_ternormalkan": "CALPIR, per level",
            "konformal_rf_global": "conformal, no scale",
            "konformal_rf_mondrian": "conformal, per level",
            "rf_sebar": "tree spread, unconformalised",
            "ensemble_dalam_konformal": "deep ensemble, conformalised",
            "ensemble_dalam": "deep ensemble",
            "mc_dropout_konformal": "MC dropout, conformalised",
            "mc_dropout": "MC dropout", "cqr_gbm": "quantile regression"}
    pohon = {"konformal_rf_ternormalkan", "konformal_rf_mondrian_ternormalkan",
             "konformal_rf_global", "konformal_rf_mondrian", "rf_sebar"}
    # urutan baris dipatok pada alat pertama, sehingga kedua panel sebanding
    urut = [n for _, n in sorted((v["lebar"], n) for n, v in d["LC"]["A"].items()
                                 if n in nama)]
    fig, ax = plt.subplots(1, 2, figsize=(HALAMAN, 2.5), sharex=True)
    for k, alat in enumerate(("LC", "EM")):
        baris = [(d[alat]["A"][n]["lebar"], d[alat]["A"][n]["cakupan"], n) for n in urut]
        y = np.arange(len(baris))
        ax[k].barh(y, [b[0] for b in baris], height=0.62, edgecolor="none",
                   color=[BIRU if n in pohon else ORNG for _, _, n in baris])
        for i, (w, c, _) in enumerate(baris):
            ax[k].text(w * 1.06, i, f"{c:.3f}", va="center", fontsize=8, color=ABU)
        ax[k].set_yticks(y)
        ax[k].set_yticklabels([] if k else [nama[b[2]] for b in baris])
        ax[k].set_xscale("log"); ax[k].set_xlim(0.06, 9)
        # penanda pangkat pada sumbu log tercetak sekitar 5,5 pt, di bawah ambang
        # keterbacaan, jadi angkanya ditulis biasa
        ax[k].set_xticks([0.1, 0.3, 1, 3])
        ax[k].xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax[k].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax[k].set_xlabel("median interval width, decades")
        ax[k].set_title(judul_alat(alat), loc="left")
        ax[k].invert_yaxis()
    simpan(fig, "Figure_5")


def gambar4():
    """Pengaruh skala pada enam model dasar."""
    d = baca("besar_tahap1")["hasil"]
    model = ["ET", "RF", "XGB", "LGBM", "SVR", "PLS"]
    label = ["Extra\ntrees", "Random\nforest", "XG\nBoost", "Light\nGBM", "SVR", "PLS"]
    skala = [("tanpa", "no scale", BIRU_MUDA), ("sebar", "ensemble spread", BIRU),
             ("terlatih", "learned scale", ORNG)]
    fig, ax = plt.subplots(1, 2, figsize=(HALAMAN * 1.19, 2.4 * 1.19), sharey=True)
    lebar, x = 0.26, np.arange(len(model))
    for k, alat in enumerate(("LC", "EM")):
        for j, (kunci, teks_, w) in enumerate(skala):
            nilai = [d[alat][m].get(f"{kunci}|marginal|0.1", {}).get("lebar", np.nan)
                     for m in model]
            ax[k].bar(x + (j - 1) * lebar, nilai, lebar, label=teks_, color=w,
                      edgecolor="none")
        ax[k].set_xticks(x); ax[k].set_xticklabels(label)
        ax[k].set_title(judul_alat(alat), loc="left")
        ax[k].grid(axis="y", lw=0.4, color="#DDDDDD"); ax[k].set_axisbelow(True)
    ax[0].set_ylabel("median width, decades")
    ax[0].legend(frameon=False, loc="upper left")
    simpan(fig, "Figure_4")


def gambar6():
    """Satu gambar: untung aturan dua tahap, dan biayanya, terhadap target
    laju penolakan. Warna menandai seri, garis menandai alat."""
    d = baca("hasil_penjaga_v3")
    beta = [0.01, 0.05, 0.10, 0.20]
    amb = lambda tahap, alat: d[tahap][alat]["ringkas"]["jarak_knn"]["beta"]
    fig, ax = plt.subplots(figsize=(HALAMAN * 0.78, 2.95))

    # warna menandai aturan, bukan seri. Biru untuk aturan dua tahap dengan dua
    # rona, oranye untuk aturan satu tahap, sehingga paletnya dua warna utama.
    seri = (("dua_tahap", "tolak_out", BIRU, "o"),
            ("satu_tahap", "tolak_out", ORNG, "s"),
            ("dua_tahap", "tolak_in", BIRU_TUA, "^"))
    for tahap, kunci, warna, tanda in seri:
        for alat, gaya in (("LC", "-"), ("EM", "--")):
            r = amb(tahap, alat)
            ax.plot(beta, [r[str(b)][kunci] for b in beta], gaya, color=warna,
                    marker=tanda, ms=3.6, lw=1.4, zorder=3)

    # label langsung pada kurva, bukan kotak legenda
    for x, y, t, warna, va in (
            (0.112, 1.035, "two-stage rule, outside measurements", BIRU, "bottom"),
            (0.150, 0.505, "single-stage rule, outside measurements", ORNG, "top"),
            (0.150, 0.300, "two-stage rule, in-range measurements", BIRU_TUA, "bottom")):
        ax.text(x, y, t, fontsize=7.4, color=warna, ha="center", va=va, zorder=6,
                bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85))

    # titik kerja yang dipakai di Tabel 3, dan selisih yang dihasilkan tahap bentuk
    ax.axvline(0.05, color=ABU, lw=0.8, ls=":", zorder=2)
    ax.annotate("", xy=(0.05, 0.996), xytext=(0.05, 0.324),
                arrowprops=dict(arrowstyle="<->", color=ABU_TUA, lw=1.0), zorder=5)
    ax.text(0.0455, 0.66, "0.67 added by\nthe shape gate", fontsize=7.2,
            color=ABU_TUA, ha="right", va="center", zorder=6)
    ax.text(0.05, -0.115, "operating point\nof Table 3", fontsize=7, color=ABU_TUA,
            ha="center", va="top", zorder=6)

    ax.set_xlabel("target rejection rate")
    ax.set_ylabel("fraction of measurements rejected")
    ax.set_xlim(0.0, 0.215); ax.set_ylim(0, 1.13)
    ax.set_xticks([0.01, 0.05, 0.10, 0.15, 0.20])
    ax.grid(lw=0.4, color="#DDDDDD"); ax.set_axisbelow(True)
    garis_alat = [plt.Line2D([], [], color=ABU_TUA, ls=g, lw=1.4, label=t)
                  for g, t in (("-", "low-cost"), ("--", "commercial"))]
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(handles=garis_alat, frameon=False, fontsize=7.4,
              loc="lower right", handlelength=2.4)
    simpan(fig, "Figure_6")


def _contoh_operator(alat="LC", lv_out=600, benih=0):
    """Satu pengukuran diterima dan satu ditolak, dihitung ulang dari data."""
    X, y, t, level = calpir.muat([alat])[alat]
    sumbu = np.load(os.path.join(calpir.DATA, "sumbu_sama.npz"))["sumbu"]
    sisa = level[level != lv_out]
    rng = np.random.default_rng(benih)
    i_fit, i_kal, i_uji = calpir.belah(y, sisa, rng, 20, 20, 10)
    i_out = np.flatnonzero(y == lv_out)
    m = calpir.buat_model("ET", benih).fit(X[i_fit], t[i_fit])
    p_kal, p_uji = calpir.ramal(m, X[i_kal]), calpir.ramal(m, X[i_uji])
    s_kal, s_uji = calpir.skala_sebar(m, X, i_kal), calpir.skala_sebar(m, X, i_uji)
    q = calpir.ambang(np.abs(t[i_kal] - p_kal) / s_kal)
    lolos, _ = calpir.gerbang_bentuk(X[i_fit])
    g_uji, g_out = lolos(X[i_uji]), lolos(X[i_out])
    amb2 = float(np.percentile(calpir.jarak_knn(X[i_kal][lolos(X[i_kal])], X[i_fit]),
                               100 * (1 - calpir.BETA)))
    # diterima: lolos gerbang, skor kebaruan di bawah ambang, lebar mendekati median
    d2_uji = calpir.jarak_knn(X[i_uji], X[i_fit])
    lebar = 2 * q * s_uji
    calon = np.flatnonzero(g_uji & (d2_uji <= amb2))
    k = calon[np.argmin(np.abs(lebar[calon] - np.median(lebar[calon])))]
    d2_out = calpir.jarak_knn(X[i_out], X[i_fit])
    j = int(np.argmax(d2_out))
    return {"sumbu": sumbu, "latih": X[i_fit], "terima": X[i_uji][k], "tolak": X[i_out][j],
            "kadar": float(y[i_uji][k]), "ramal": float(10 ** p_uji[k]),
            "bawah": float(10 ** (p_uji[k] - q * s_uji[k])),
            "atas": float(10 ** (p_uji[k] + q * s_uji[k])),
            "kadar_tolak": float(lv_out), "gerbang_tolak": bool(not g_out[j]),
            "d2": float(d2_out[j]), "amb2": amb2, "alat": alat}


def gambar7():
    """Apa yang dilihat operator, satu panel. Dua sapuan sungguhan, satu
    diterima dengan interval dan satu ditolak, dengan hasilnya ditempel
    langsung pada kurvanya."""
    h = _contoh_operator()
    fig, ax = plt.subplots(figsize=(KOLOM * 1.10, 2.65))
    lo, hi = np.percentile(h["latih"], [5, 95], axis=0)
    ax.fill_between(h["sumbu"], lo, hi, color="#E3E3E3", lw=0, zorder=1,
                    label="calibration standards, 5th to 95th percentile")
    ax.plot(h["sumbu"], h["terima"], color=BIRU, lw=1.5, zorder=3,
            label="sweep at %.0f ppm, inside the standards" % h["kadar"])
    ax.plot(h["sumbu"], h["tolak"], color=ORNG, lw=1.5, zorder=3,
            label="sweep at %.0f ppm, outside the standards" % h["kadar_tolak"])
    ax.set_xlabel("potential, V"); ax.set_ylabel("current")
    ax.set_ylim(min(h["tolak"].min(), lo.min()) - 112, max(hi.max(), 130) + 86)
    ax.legend(frameon=False, fontsize=7.5, loc="lower left", handlelength=1.5,
              labelspacing=0.3, borderaxespad=0.25,
              bbox_to_anchor=(-0.018, -0.022))
    ax.grid(lw=0.4, color="#EEEEEE"); ax.set_axisbelow(True)

    def hasil_kotak(x, y, kepala, isi, warna, isi_warna, xy):
        ax.annotate(kepala + "\n" + isi, xy=xy, xycoords="data",
                    xytext=(x, y), textcoords="axes fraction",
                    fontsize=7.8, color=warna, ha="left", va="center", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.33", fc=isi_warna,
                              ec=warna, lw=0.9),
                    arrowprops=dict(arrowstyle="-", color=warna, lw=0.9,
                                    shrinkA=2, shrinkB=2))

    i = int(np.argmin(np.abs(h["sumbu"] + 1.15)))
    hasil_kotak(0.030, 0.910, "accepted, %.1f ppm" % h["ramal"],
                "interval %.1f to %.1f ppm"
                % (h["bawah"], h["atas"]),
                BIRU_TUA, BIRU_ISI, (h["sumbu"][i], h["terima"][i]))
    j = int(np.argmin(np.abs(h["sumbu"] + 0.76)))
    hasil_kotak(0.615, 0.905, "rejected", "no number reported",
                "#8A5A00", ORNG_ISI, (h["sumbu"][j], h["tolak"][j]))
    simpan(fig, "Figure_7")
    return h


def hasil():
    gambar4(); gambar5(); gambar6(); print(gambar7())


# =============================================================== abstrak grafis
def abstrak(paksa=False):
    """PENSIUN. Abstrak grafis kini dibuat lewat abstrak_drawio() lalu disunting
    tangan di draw.io, sebab penulis memegang tata letaknya. Fungsi ini ditahan
    agar tidak menimpa berkas hasil suntingan itu. Jalankan dengan paksa=True
    hanya bila memang ingin kembali ke versi matplotlib.

    Dipatok 13 x 5,2 cm, yaitu 1535 x 614 piksel pada 300 dpi, di atas syarat
    terkecil Elsevier 1328 x 531 dan tepat pada rasio 500 lebar berbanding 200
    tinggi yang diminta. Hurufnya Arial, satu dari empat huruf yang diizinkan
    Elsevier untuk abstrak grafis. Voltamogram di panel kiri adalah pengukuran
    sungguhan, bukan sketsa."""
    def kotak_ax(ax, x, y, w, h, isi, garis, lw=1.0):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.02", linewidth=lw,
            edgecolor=garis, facecolor=isi, transform=ax.transAxes, clip_on=False,
            zorder=1))

    # Elsevier hanya mengizinkan Times, Arial, Courier, atau Symbol di abstrak
    # grafis, sedangkan naskah memakai Helvetica. Penggantian dipatok lokal.
    if not paksa:
        print("dilewati: graphical_abstract.pdf berasal dari draw.io, "
              "lihat abstrak_drawio(). Pakai abstrak(paksa=True) untuk menimpa.")
        return
    with plt.rc_context({"font.sans-serif": ["Arial", "Helvetica"],
                         "mathtext.fontset": "custom",
                         "mathtext.rm": "Arial", "mathtext.it": "Arial"}):
        return _abstrak_isi(kotak_ax)


def _abstrak_isi(kotak_ax):
    fig = plt.figure(figsize=(13 * CM, 5.2 * CM))
    ax = fig.add_axes([0.05, 0.20, 0.235, 0.62])
    d = muat_sapuan("LC")
    V, I, y = d["V"], d["I"], d["y"]
    for lv, w in ((10, BIRU_MUDA), (100, BIRU), (1000, BIRU_TUA)):
        j = np.flatnonzero(y == lv)[0]
        ax.plot(V[j, 4], I[j, 4], color=w, lw=0.9, label=f"{lv} ppm")
    # ukuran huruf label dipatok 7,5 pt di sini, sebab abstrak grafis lebih
    # kecil daripada gambar naskah dan labelnya menimpa pita bawah bila ikut 8,5
    ax.set_xlabel("potential, V", labelpad=1.5, fontsize=7.5)
    ax.set_ylabel("current", labelpad=1.5, fontsize=7.5)
    ax.set_xticks([-1.4, -0.9, -0.5]); ax.set_yticks([])
    ax.tick_params(labelsize=6.5, pad=1.5)
    ax.legend(frameon=False, fontsize=6, handlelength=1.0, borderpad=0.1,
              labelspacing=0.25, loc="upper left")
    ax.set_title("cyclic voltammetry\ntwo potentiostats", fontsize=7.5, pad=3)

    ax2 = fig.add_axes([0.32, 0, 0.67, 1]); ax2.axis("off")
    for y0, isi, garis, t in (
            (0.52, BIRU_ISI, BIRU, "CALPIR\ninterval scaled by\nmember disagreement"),
            (0.14, ORNG_ISI, ORNG, "two-stage rule\nrejects scans outside\nthe calibration set")):
        kotak_ax(ax2, 0.015, y0, 0.335, 0.30, isi, garis)
        ax2.text(0.1825, y0 + 0.15, t, ha="center", va="center", fontsize=7.0,
                 transform=ax2.transAxes)
    for y0 in (0.67, 0.29):
        # besar kepala panah ikut ukuran huruf anotasi, jadi ukurannya dipatok
        ax2.annotate("", xy=(0.405, y0), xytext=(0.355, y0), fontsize=7.5,
                     arrowprops=dict(arrowstyle="-|>", color=ABU_TUA, lw=1.0))
    for y0, garis, kepala, t in (
            (0.52, BIRU, "report", "60 ppm within a factor of 1.09"),
            (0.14, ORNG, "reject", "no number reported")):
        kotak_ax(ax2, 0.415, y0, 0.56, 0.30, "white", garis)
        ax2.text(0.44, y0 + 0.215, kepala, fontsize=7, color=ABU_TUA,
                 transform=ax2.transAxes)
        ax2.text(0.695, y0 + 0.115, t, ha="center", va="center", fontsize=11,
                 color=garis, transform=ax2.transAxes)
    ax2.text(0.50, 0.05, "coverage as claimed   |   3.61 and 4.48 times narrower   |   "
                         "88% to 96% rejected",
             ha="center", va="center", fontsize=6.4, color=ABU_TUA,
             transform=ax2.transAxes)

    simpan(fig, "graphical_abstract", ("pdf", "png"), ketat=False)


URUTAN = ("skema", "hasil", "abstrak")

if __name__ == "__main__":
    pilih = [a for a in sys.argv[1:] if not a.startswith("--")] or list(URUTAN)
    salah = [a for a in pilih if a not in URUTAN]
    assert not salah, f"gambar tidak dikenal: {salah}, pilihannya {URUTAN}"
    for nama in URUTAN:
        if nama in pilih:
            globals()[nama]()

# ================================================ abstrak grafis versi draw.io
ARIAL = ("rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor={isi};"
         "strokeColor={garis};strokeWidth=1.2;fontFamily=Arial;fontSize={fs};"
         "verticalAlign=middle;align=center;spacing=3;")
ARIAL_T = ("text;html=1;strokeColor=none;fillColor=none;align={al};"
           "verticalAlign=middle;fontFamily=Arial;fontSize={fs};fontColor={wr};")
ARIAL_P = ("edgeStyle=none;rounded=0;html=1;strokeColor=#555555;strokeWidth=1.2;"
           "endArrow=block;endFill=1;fontFamily=Arial;")


def _panel_voltamogram(p_png):
    """Panel kiri abstrak grafis, diekspor sendiri supaya bisa ditempel di
    draw.io. Ini voltamogram sungguhan, bukan sketsa, jadi tidak bisa digambar
    ulang sebagai bentuk draw.io."""
    with plt.rc_context({"font.sans-serif": ["Arial", "Helvetica"]}):
        fig = plt.figure(figsize=(3.9 * CM, 5.2 * CM))
        ax = fig.add_axes([0.17, 0.20, 0.79, 0.67])
        d = muat_sapuan("LC")
        V, I, y = d["V"], d["I"], d["y"]
        for lv, w in ((10, BIRU_MUDA), (100, BIRU), (1000, BIRU_TUA)):
            j = np.flatnonzero(y == lv)[0]
            ax.plot(V[j, 4], I[j, 4], color=w, lw=0.9, label=f"{lv} ppm")
        ax.set_xlabel("potential, V", labelpad=1.5, fontsize=7.5)
        ax.set_ylabel("current", labelpad=1.5, fontsize=7.5)
        ax.set_xticks([-1.4, -0.9, -0.5]); ax.set_yticks([])
        ax.tick_params(labelsize=6.5, pad=1.5)
        lg = ax.legend(frameon=False, fontsize=6, handlelength=1.0, borderpad=0.1,
                       labelspacing=0.25, loc="upper left",
                       title="15 levels, 2 to 1000 ppm")
        # judul legenda menyatakan rentang penuh, sehingga ketiga kurva terbaca
        # sebagai contoh dari lima belas level, bukan sebagai seluruh datanya
        lg.get_title().set_fontsize(6)
        lg.get_title().set_ha("left")
        lg._legend_box.align = "left"
        ax.set_title("CV on 2 potentiostats", fontsize=7.5, pad=3)
        fig.savefig(p_png, dpi=300, facecolor="white")
        plt.close(fig)


def abstrak_drawio():
    """Abstrak grafis sebagai berkas draw.io yang bisa disunting tangan.

    Kanvas 492 x 197 piksel, yaitu 369 x 148 pt pada 0,75 pt per piksel, sama
    dengan 13 x 5,2 cm dan tepat pada rasio 500 banding 200 yang diminta
    Elsevier. Hurufnya Arial, satu dari empat yang diizinkan. Panel voltamogram
    ditanam sebagai PNG base64 sehingga berkasnya berdiri sendiri."""
    import base64
    os.makedirs(DRAWIO, exist_ok=True)
    p_png = os.path.join(DRAWIO, "abstrak_panel.png")
    _panel_voltamogram(p_png)
    b64 = base64.b64encode(open(p_png, "rb").read()).decode()

    sel = [f'        <mxCell id="plot" value="" style="shape=image;imageAspect=0;'
           f'aspect=fixed;image=data:image/png,{b64}" vertex="1" parent="1">'
           f'<mxGeometry x="2" y="0" width="148" height="197" as="geometry"/></mxCell>']

    def kotak_a(i, t, x, y, w, h, isi, garis, fs):
        return _sel(i, t, ARIAL.format(isi=isi, garis=garis, fs=fs), x, y, w, h)

    def teks_a(i, t, x, y, w, h, al="center", fs=9, wr="#333333"):
        return _sel(i, t, ARIAL_T.format(al=al, fs=fs, wr=wr), x, y, w, h)

    # dua kotak metode di kiri, dua kotak keluaran di kanan
    for i, (y, isi, garis, t) in enumerate((
            (35, BIRU_ISI, BIRU, "CALPIR<br>interval scaled by<br>member disagreement"),
            (110, ORNG_ISI, ORNG, "two-stage rule<br>rejects scans outside<br>the calibration set"))):
        sel.append(kotak_a(f"m{i}", t, 162, y, 110, 59, isi, garis, 9))
    for i, (y, garis, kepala, t) in enumerate((
            (35, BIRU, "report", "60 ppm within a<br>factor of 1.09"),
            (110, ORNG, "reject", "no number reported"))):
        sel.append(kotak_a(f"k{i}", "", 294, y, 185, 59, "#FFFFFF", garis, 9))
        sel.append(teks_a(f"kh{i}", kepala, 300, y + 4, 50, 12, "left", 9, "#555555"))
        sel.append(teks_a(f"kt{i}", t, 300, y + 22, 173, 30, "center", 13, garis))
        sel.append(f'        <mxCell id="pa{i}" style="{ARIAL_P}" edge="1" parent="1"'
                   f' source="m{i}" target="k{i}"><mxGeometry relative="1"'
                   ' as="geometry"/></mxCell>')
    sel.append(teks_a("kaki", "coverage as claimed &#160; | &#160; 3.61 and 4.48 times "
                      "narrower &#160; | &#160; 88% to 96% rejected",
                      162, 178, 317, 14, "center", 8.5, "#555555"))

    p = os.path.join(DRAWIO, "graphical_abstract.drawio")
    open(p, "w").write(bingkai("\n".join(sel), 197, 492))
    print("ditulis", p)
    print("ditulis", p_png)
    print("ekspor TANPA --crop, sebab --crop memotong ke kotak isi dan")
    print("merusak rasio 2,50 yang diminta Elsevier:")
    print("  draw.io --export --format pdf --scale 1 -o graphical_abstract.pdf \\")
    print("          drawio/graphical_abstract.drawio")

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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

from calpir import GAMBAR, HASIL, muat_sapuan

DRAWIO = os.path.join(GAMBAR, "drawio")
BIRU, BIRU_MUDA, BIRU_TUA = "#0072B2", "#8EC6E6", "#00436B"
ORNG, ORNG_MUDA = "#E69F00", "#F5CE86"
BIRU_ISI, ORNG_ISI = "#EAF3FA", "#FDF0DC"
ABU, ABU_TUA = "#666666", "#555555"
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
         ";strokeWidth=1.2;endArrow=block;endFill=1;fontFamily=Helvetica;fontSize=9;")


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
    lebar, jarak, y = 118, 17, 58
    judul = ["Paired acquisition", "Shared potential grid", "Ensemble fitting",
             "Interval construction", "Rejection rule"]
    bawah = ["two potentiostats,<br>one cell", "178 current<br>variables",
             "six base<br>learners", "scale from member<br>disagreement",
             "shape gate then<br>novelty score"]
    tunjuk = ["Sec. 3.2, Table 2", "Sec. 3.3", "Sec. 3.4",
              "Sec. 3.5, Eq. (3)-(7)", "Sec. 3.6, Alg. 1"]
    sel, x = [], 1
    for k in range(5):
        sel.append(teks(f"t{k}", judul[k], x, y - 22, lebar, 18, fs=11))
        sel.append(kotak(f"b{k}", bawah[k], x, y, lebar, 62, fs=11))
        sel.append(teks(f"p{k}", tunjuk[k], x, y + 64, lebar, 16, fs=9))
        if k:
            sel.append(panah(f"e{k}", f"b{k-1}", f"b{k}"))
        x += lebar + jarak
    sel.append(kotak("out", "concentration with an interval, or a refusal",
                     340, y + 112, 319, 38, ORNG_ISI, ORNG, 11))
    sel.append(panah("eo", "b4", "out", keluar=(0.5, 1), masuk=(0.815, 0)))
    return bingkai("\n".join(sel), 218)


def gambar2():
    sel = [
        kotak("cell", "Electrochemical cell<br>carbon, Ag/AgCl, 0.1 mol/L KCl",
              235, 62, 190, 66),
        kotak("lc", "Low-cost potentiostat<br>Rodeostat platform", 6, 62, 200, 66),
        kotak("em", "Commercial potentiostat<br>EmStat4", 454, 62, 200, 66),
        teks("sw", "leads moved after every block of ten scans", 6, 10, 648, 20, fs=11),
        teks("nt", "same solution, same electrode set, same session",
             6, 140, 648, 18, fs=9),
        panah("e1", "cell", "lc", "block k"),
        panah("e2", "cell", "em", "block k+1"),
    ]
    return bingkai("\n".join(sel), 170)


def gambar3():
    sel = [
        kotak("in", "voltammogram", 82, 4, 150, 26, fs=11),
        belah("g1", "shape distance<br>in range?", 52, 42, 210, 62, fs=11),
        belah("g2", "novelty score<br>within budget?", 52, 118, 210, 62, fs=11),
        kotak("fit", "ensemble mean<br>and member spread", 62, 196, 190, 40, fs=11),
        kotak("iv", "report interval", 82, 252, 150, 26, ORNG_ISI, ORNG, 11),
        kotak("rf", "refuse", 268, 110, 58, 26, ORNG_ISI, ORNG, 11),
        teks("q1", "Eq. (8), (9)", 2, 60, 48, 14, "left", 9),
        teks("q2", "Eq. (10)", 2, 136, 48, 14, "left", 9),
        teks("q3", "Eq. (3), (4)", 2, 200, 48, 28, "left", 9),
        teks("q4", "Eq. (7)", 2, 254, 48, 14, "left", 9),
        panah("a1", "in", "g1"),
        panah("a2", "g1", "g2", "yes"),
        panah("a3", "g2", "fit", "yes"),
        panah("a4", "fit", "iv"),
        panah("a5", "g1", "rf", "no", keluar=(1, 0.5)),
        panah("a6", "g2", "rf", "no", keluar=(1, 0.5)),
    ]
    return bingkai("\n".join(sel), 288, W1)


def skema():
    os.makedirs(DRAWIO, exist_ok=True)
    for nama, fn in (("Figure_1", gambar1), ("Figure_2", gambar2), ("Figure_3", gambar3)):
        p = os.path.join(DRAWIO, nama + ".drawio")
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
            ax[k].text(w * 1.06, i, f"{c:.3f}", va="center", fontsize=7.5, color=ABU)
        ax[k].set_yticks(y)
        ax[k].set_yticklabels([] if k else [nama[b[2]] for b in baris])
        ax[k].set_xscale("log"); ax[k].set_xlim(0.06, 9)
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
    """Laju tolak penjaga dan biayanya terhadap anggaran penolakan."""
    d = baca("hasil_penjaga_v3")["dua_tahap"]
    beta = [0.01, 0.05, 0.10, 0.20]
    fig, ax = plt.subplots(figsize=(KOLOM * 1.10, 2.3 * 1.10))
    for alat, gaya in (("LC", "-"), ("EM", "--")):
        r = d[alat]["ringkas"]["jarak_knn"]["beta"]
        for kunci, warna, tanda, teks_ in (("tolak_out", BIRU, "o", "outside"),
                                           ("tolak_in", ORNG, "s", "sound")):
            ax.plot(beta, [r[str(b)][kunci] for b in beta], gaya, color=warna,
                    marker=tanda, ms=3.4, lw=1.3,
                    label=f"{judul_alat(alat)}, {teks_}")
    ax.set_xlabel("refusal budget"); ax.set_ylabel("fraction refused")
    ax.set_ylim(0, 1.05); ax.grid(lw=0.4, color="#DDDDDD"); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="center right")
    simpan(fig, "Figure_6")


def hasil():
    gambar4(); gambar5(); gambar6()


# =============================================================== abstrak grafis
def abstrak():
    """Dipatok 13 x 5 cm, yaitu 1535 x 590 piksel pada 300 dpi, di atas
    syarat terkecil penerbit. Voltamogram di panel kiri adalah pengukuran
    sungguhan, bukan sketsa."""
    def kotak_ax(ax, x, y, w, h, isi, garis, lw=1.0):
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.02", linewidth=lw,
            edgecolor=garis, facecolor=isi, transform=ax.transAxes, clip_on=False,
            zorder=1))

    fig = plt.figure(figsize=(13 * CM, 5 * CM))
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
            (0.14, ORNG_ISI, ORNG, "two-stage rule\nrefuses scans outside\nthe calibration set")):
        kotak_ax(ax2, 0.015, y0, 0.335, 0.30, isi, garis)
        ax2.text(0.1825, y0 + 0.15, t, ha="center", va="center", fontsize=7.0,
                 transform=ax2.transAxes)
    for y0 in (0.67, 0.29):
        # besar kepala panah ikut ukuran huruf anotasi, jadi ukurannya dipatok
        ax2.annotate("", xy=(0.405, y0), xytext=(0.355, y0), fontsize=7.5,
                     arrowprops=dict(arrowstyle="-|>", color=ABU_TUA, lw=1.0))
    for y0, garis, kepala, t in (
            (0.52, BIRU, "report", r"$60\ \mathrm{ppm}\ \times\!/\!\div\ 1.09$"),
            (0.14, ORNG, "refuse", "no number reported")):
        kotak_ax(ax2, 0.415, y0, 0.56, 0.30, "white", garis)
        ax2.text(0.44, y0 + 0.215, kepala, fontsize=7, color=ABU_TUA,
                 transform=ax2.transAxes)
        ax2.text(0.695, y0 + 0.115, t, ha="center", va="center", fontsize=11,
                 color=garis, transform=ax2.transAxes)
    ax2.text(0.50, 0.05, "coverage as claimed   |   3.6 to 4.5 times narrower   |   "
                         "88 to 96 per cent refused",
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

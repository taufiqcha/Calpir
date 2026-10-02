"""Bedah 50 replikat blanko, tiap replikat sampai ke sepuluh sapuannya.

Simpangan baku blanko menentukan batas deteksi. Sebelum angka itu dipakai,
perlu diketahui apakah kelima puluh replikatnya memang setara, atau sebagian
rusak dan menggelembungkan simpangannya.

Untuk tiap sapuan dicatat empat besaran.

  puncak      arus terendah sepanjang sapuan
  derau       simpangan beda kedua, yaitu goyangan titik ke titik
  garis_dasar arus rata-rata pada sepersepuluh awal sapuan
  rentang     selisih arus tertinggi dan terendah

Untuk tiap replikat dicatat urutan pengambilan dari nama berkasnya, cap waktu
dari arsip, lalu ringkasan keempat besaran di atas beserta kemiringannya
sepanjang sepuluh sapuan.

Terakhir disimulasikan apa yang terjadi pada simpangan baku blanko bila sapuan
atau replikat tertentu dibuang, sehingga terlihat mana yang paling menentukan.

Keluaran ditulis ke results/diagnosa_blanko.json.

Cara pakai: python3 diagnosa_blanko.py [berkas_arsip.zip]
"""
import io
import json
import os
import re
import sys
import zipfile

import numpy as np
import openpyxl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pra_proses
from calpir import ALAT, HASIL
from pra_proses import FOLDER, N_SAPUAN, N_TITIK

HIMPUNAN_SAPUAN = {
    "semua_1_10": list(range(10)),
    "baku_3_10": list(range(2, 10)),
    "tengah_4_8": list(range(3, 8)),
    "akhir_6_10": list(range(5, 10)),
    "satu_sapuan_5": [4],
}


def ukur(v, i):
    """Empat besaran satu sapuan."""
    n = max(3, i.size // 10)
    return {
        "puncak": float(i.min()),
        "derau": float(np.median(np.abs(np.diff(i, 2)))),
        "garis_dasar": float(i[:n].mean()),
        "rentang": float(i.max() - i.min()),
    }


def baca(zf, alat):
    """Seluruh berkas blanko satu alat, beserta urutan dan cap waktunya."""
    awalan = f"{FOLDER[alat]}/KCL/"
    nama = sorted(n for n in zf.namelist()
                  if n.startswith(awalan) and n.endswith(".xlsx"))
    keluar = []
    for f in nama:
        info = zf.getinfo(f)
        angka = re.findall(r"(\d+)", os.path.basename(f))
        wb = openpyxl.load_workbook(io.BytesIO(zf.read(f)), data_only=True)
        baris = list(wb[wb.sheetnames[0]].iter_rows(values_only=True))[2:]
        sapuan = []
        for s in range(N_SAPUAN):
            v, i = pra_proses._satu_sapuan_mentah(baris, s)
            if v is None or len(v) < 20:
                sapuan.append(None)
                continue
            vv, ii = pra_proses._resample(v, i)
            sapuan.append({"n_titik_asli": int(len(i)), **ukur(vv, ii)})
        keluar.append({
            "berkas": os.path.basename(f),
            "urutan": int(angka[-1]) if angka else -1,
            "waktu": "%04d-%02d-%02d %02d:%02d:%02d" % info.date_time,
            "ukuran_byte": info.file_size,
            "sapuan": sapuan,
            "n_sapuan_terbaca": sum(s is not None for s in sapuan),
        })
        print(f"  {alat} {os.path.basename(f)} "
              f"{keluar[-1]['n_sapuan_terbaca']}/10 sapuan", flush=True)
    return keluar


def ringkas_replikat(r):
    """Ringkasan satu replikat atas sapuan yang terbaca."""
    ada = [(s, d) for s, d in enumerate(r["sapuan"]) if d is not None]
    if not ada:
        return None
    for k in ("puncak", "derau", "garis_dasar", "rentang"):
        nilai = np.array([d[k] for _, d in ada])
        r[k + "_median"] = float(np.median(nilai))
        r[k + "_sd"] = float(nilai.std(ddof=1)) if nilai.size > 1 else 0.0
    idx = np.array([s for s, _ in ada], float)
    pun = np.array([d["puncak"] for _, d in ada])
    r["puncak_kemiringan_per_sapuan"] = (float(np.polyfit(idx, pun, 1)[0])
                                         if idx.size > 2 else None)
    r["puncak_geser_1_ke_10_persen"] = (float(100 * (pun[-1] - pun[0]) / abs(pun[0]))
                                        if pun[0] != 0 else None)
    return r


def nilai_himpunan(replikat, sapuan_dipakai):
    """Simpangan baku blanko bila hanya sapuan tertentu yang dipakai."""
    nilai = []
    for r in replikat:
        pun = [r["sapuan"][s]["puncak"] for s in sapuan_dipakai
               if r["sapuan"][s] is not None]
        if pun:
            nilai.append(float(np.median(pun)))
    a = np.array(nilai)
    return {"n_replikat": int(a.size), "rata": float(a.mean()),
            "sd": float(a.std(ddof=1)), "min": float(a.min()),
            "maks": float(a.max())}


def pencilan(replikat, k=3.0):
    """Replikat yang puncaknya menyimpang lebih dari k simpangan baku robust."""
    pun = np.array([r["puncak_median"] for r in replikat])
    med = np.median(pun)
    mad = np.median(np.abs(pun - med)) * 1.4826
    if mad == 0:
        return [], float(med), 0.0
    z = (pun - med) / mad
    return ([{"berkas": replikat[i]["berkas"], "urutan": replikat[i]["urutan"],
              "waktu": replikat[i]["waktu"], "puncak": float(pun[i]),
              "z_robust": float(z[i])}
             for i in np.argsort(-np.abs(z)) if abs(z[i]) > k],
            float(med), float(mad))


def main(arsip):
    keluar = {}
    with zipfile.ZipFile(arsip) as zf:
        for alat in ALAT:
            print(alat, flush=True)
            replikat = [ringkas_replikat(r) for r in baca(zf, alat)]
            replikat = [r for r in replikat if r is not None]
            pc, med, mad = pencilan(replikat)
            bersih = [r for r in replikat
                      if r["berkas"] not in {p["berkas"] for p in pc}]
            keluar[alat] = {
                "n_replikat": len(replikat),
                "n_sapuan_hilang": int(sum(10 - r["n_sapuan_terbaca"] for r in replikat)),
                "puncak_median_keseluruhan": med,
                "mad_robust": mad,
                "pencilan": pc,
                "sd_per_himpunan_sapuan": {nm: nilai_himpunan(replikat, ss)
                                           for nm, ss in HIMPUNAN_SAPUAN.items()},
                "sd_baku_tanpa_pencilan": nilai_himpunan(bersih, HIMPUNAN_SAPUAN["baku_3_10"]),
                "replikat": replikat,
            }
    p = os.path.join(HASIL, "diagnosa_blanko.json")
    json.dump(keluar, open(p, "w"), indent=1)
    print("ditulis", p, flush=True)

    for alat in ALAT:
        h = keluar[alat]
        print(f"\n{'='*72}\n{alat}  {h['n_replikat']} replikat, "
              f"{h['n_sapuan_hilang']} sapuan tidak terbaca\n{'='*72}")
        print("  simpangan baku blanko menurut himpunan sapuan yang dipakai")
        for nm, v in h["sd_per_himpunan_sapuan"].items():
            print(f"    {nm:14s} sd={v['sd']:7.3f}  rata={v['rata']:9.3f}  "
                  f"rentang {v['min']:9.3f} sampai {v['maks']:9.3f}")
        v = h["sd_baku_tanpa_pencilan"]
        print(f"    baku_3_10 tanpa pencilan  sd={v['sd']:7.3f}  "
              f"n={v['n_replikat']}")
        print(f"  pencilan, {len(h['pencilan'])} replikat")
        for p_ in h["pencilan"][:8]:
            print(f"    {p_['berkas']:28s} urutan {p_['urutan']:>3}  "
                  f"{p_['waktu']}  puncak {p_['puncak']:9.3f}  z {p_['z_robust']:+6.1f}")


if __name__ == "__main__":
    arsip = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("CALPIR_ZIP", "")
    assert os.path.isfile(arsip), "tunjuk arsip voltamogram mentah sebagai argumen"
    main(arsip)

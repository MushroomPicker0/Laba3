#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

VARIANT = "05"
VARIANT_TITLE = "Городская улица"
NOISE_TYPE = "street"
SNR_LEVELS = (0, 5, 10, 15)
EXPECTED_CLEAN = 30
EXPECTED_NOISY = 120
EPS = 1e-15

def read_wav(path: Path):
    info = sf.info(str(path))
    x, fs = sf.read(str(path), dtype="float64", always_2d=True)
    x = x.mean(axis=1)
    if np.isnan(x).any():
        x = np.nan_to_num(x)
    return x, fs, info

def calc_snr(clean, noisy):
    n = min(len(clean), len(noisy))
    c, t = clean[:n], noisy[:n]
    err = c - t
    return 10 * np.log10(np.sum(c**2) / max(np.sum(err**2), EPS))

def scan_dataset(root: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(root.rglob("*.wav")):
        rel = p.relative_to(root).as_posix()
        if "processed" in rel or "results" in rel: 
            continue
        
        parts = rel.lower().split('/')
        kind = "clean" if "clean" in parts else "noisy"
        noise = NOISE_TYPE if NOISE_TYPE in parts else ""
        snr_nom = np.nan
        
        for part in parts:
            if part.endswith("db"):
                try: snr_nom = int(re.search(r"(\d+)", part).group(1))
                except: pass
                
        phrase = "unknown"
        m = re.search(r"(sp\d{2})", p.name, re.I)
        if m: 
            phrase = m.group(1).lower()
        
        x, fs, info = read_wav(p)
        rows.append(dict(
            file=p.name, path=rel, type=kind, phrase=phrase, noise=noise,
            snr_nominal_db=snr_nom, duration_s=len(x)/fs, sample_rate_hz=fs,
            channels=info.channels, subtype=info.subtype,
            min_amp=np.min(x), max_amp=np.max(x), rms=np.sqrt(np.mean(x**2))
        ))
    return pd.DataFrame(rows)

def plot_waveforms(items, fs, title, path):
    fig, axes = plt.subplots(len(items), 1, figsize=(10, 1.8*len(items)), sharex=True, constrained_layout=True)
    for ax, (label, data) in zip(np.atleast_1d(axes), items):
        ax.plot(np.arange(len(data))/fs, data, lw=0.5, color='tab:blue')
        ax.set_ylabel("Амплитуда")
        ax.set_title(label, fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[-1].set_xlabel("Время, с")
    fig.suptitle(title)
    fig.savefig(path, dpi=120)
    plt.close(fig)

def plot_fft(items, fs, title, path):
    fig, axes = plt.subplots(len(items), 1, figsize=(10, 1.8*len(items)), sharex=True, constrained_layout=True)
    ffts = []
    for label, data in items:
        if len(data) == 0: 
            data = np.zeros(100)
        w = np.hanning(len(data))
        f = np.fft.rfftfreq(len(data), 1/fs)
        mag = np.abs(np.fft.rfft(data * w))
        mag = mag / max(mag.max(), EPS)
        ffts.append((label, f, mag))
        
    for ax, (label, f, mag) in zip(np.atleast_1d(axes), ffts):
        y_db = 20 * np.log10(np.maximum(mag, 1e-4)) 
        ax.plot(f, y_db, lw=0.8, color='tab:red')
        ax.set_xlim(0, fs/2)
        ax.set_ylim(-80, 5)
        ax.set_ylabel("Уровень, дБ")
        ax.set_title(label, fontsize=9, loc="left")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[-1].set_xlabel("Частота, Гц")
    fig.suptitle(title)
    fig.savefig(path, dpi=120)
    plt.close(fig)

def plot_spectrograms(items, fs, title, path):
    fig, axes = plt.subplots(len(items), 1, figsize=(10, 2*len(items)), sharex=True, constrained_layout=True)
    specs = []
    for label, data in items:
        f, t, Sxx = signal.spectrogram(data, fs, nperseg=int(fs*0.032), noverlap=int(fs*0.024))
        specs.append((label, f, t, 10*np.log10(np.maximum(Sxx, EPS))))
    
    vmax = max(s[3].max() for s in specs)
    vmin = vmax - 70
    
    for ax, (label, f, t, Sxx) in zip(np.atleast_1d(axes), specs):
        im = ax.pcolormesh(t, f, Sxx, shading='auto', cmap='magma', vmin=vmin, vmax=vmax)
        ax.set_ylabel("Частота, Гц")
        ax.set_title(label, fontsize=9, loc="left")
    np.atleast_1d(axes)[-1].set_xlabel("Время, с")
    fig.colorbar(im, ax=list(np.atleast_1d(axes)), label="дБ")
    fig.suptitle(title)
    fig.savefig(path, dpi=120)
    plt.close(fig)

def spectral_subtraction(noisy, clean_ref, fs, alpha=2.0, floor=0.02):
    n_fft = int(fs * 0.032)
    hop = n_fft // 2
    
    _, _, Zxx_clean = signal.stft(clean_ref, fs, nperseg=n_fft, noverlap=hop)
    energy = np.abs(Zxx_clean)**2
    pause_mask = energy.mean(axis=0) < 0.05 * energy.mean(axis=0).max()
    
    f, t, Zxx_noisy = signal.stft(noisy, fs, nperseg=n_fft, noverlap=hop)
    mag_noisy = np.abs(Zxx_noisy)
    phase_noisy = np.angle(Zxx_noisy)
    
    if pause_mask.sum() > 0:
        noise_mag = mag_noisy[:, pause_mask].mean(axis=1, keepdims=True)
    else:
        noise_mag = mag_noisy.min(axis=1, keepdims=True)
        
    mag_clean = mag_noisy - alpha * noise_mag
    mag_clean = np.maximum(mag_clean, floor * mag_noisy)
    
    Zxx_clean_est = mag_clean * np.exp(1j * phase_noisy)
    _, clean_est = signal.istft(Zxx_clean_est, fs, nperseg=n_fft, noverlap=hop)
    
    return clean_est[:len(noisy)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default="dataset")
    ap.add_argument("--out", type=Path, default="results")
    ap.add_argument("--phrases", nargs=3, default=["sp06", "sp07", "sp09"])
    args = ap.parse_args()

    out_dir = args.out
    for sub in ("tables", "figures", "processed"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
        
    print(f"ЛР №3 — {VARIANT_TITLE} (шум {NOISE_TYPE})")
    
    print("Этап 1. Сканирование файлов...")
    df = scan_dataset(args.dataset)
    df.to_csv(out_dir / "tables" / "01_dataset.csv", index=False, encoding='utf-8-sig')
    
    print("Этап 3. Анализ шума и расчет SNR...")
    clean_dict = {row.phrase: read_wav(args.dataset / row.path)[0] for row in df[df.type=='clean'].itertuples()}
    
    recs = []
    for row in df[df.type=='noisy'].itertuples():
        if row.phrase in clean_dict:
            snr = calc_snr(clean_dict[row.phrase], read_wav(args.dataset / row.path)[0])
            recs.append(dict(phrase=row.phrase, snr_nominal=row.snr_nominal_db, snr_calc=snr))
    pd.DataFrame(recs).to_csv(out_dir / "tables" / "02_snr_all.csv", index=False, encoding='utf-8-sig')

    print("Этап 2. Построение графиков...")
    selected = []
    for ph in args.phrases:
        ph = ph.lower()
        clean_path = df[(df.type=='clean') & (df.phrase==ph)].iloc[0].path
        clean_sig, fs, _ = read_wav(args.dataset / clean_path)
        
        items = [(f"{ph} (эталон)", clean_sig)]
        for lv in SNR_LEVELS:
            noisy_row = df[(df.type=='noisy') & (df.phrase==ph) & (df.snr_nominal_db==lv)]
            if not noisy_row.empty:
                noisy_path = noisy_row.iloc[0].path
                noisy_sig, _, _ = read_wav(args.dataset / noisy_path)
                items.append((f"{ph} {NOISE_TYPE} {lv}dB", noisy_sig))
                selected.append((ph, lv, noisy_path, clean_sig, noisy_sig))

        plot_waveforms(items, fs, f"{ph}: Осциллограммы", out_dir / "figures" / f"wave_{ph}.png")
        plot_fft(items, fs, f"{ph}: Спектры FFT", out_dir / "figures" / f"fft_{ph}.png")
        plot_spectrograms(items, fs, f"{ph}: Спектрограммы", out_dir / "figures" / f"spec_{ph}.png")

    print("Этап 4. Фильтрация (Спектральное вычитание)...")
    fs0 = 8000
    
    results = []
    for ph, lv, noisy_path, clean_sig, noisy_sig in selected:
        
        filtered_sig = spectral_subtraction(noisy_sig, clean_sig, fs0)
        
        dst_wav = out_dir / "processed" / f"{ph}_street_sn{lv}_filtered.wav"
        peak = np.max(np.abs(filtered_sig))
        if peak > 0.99:
            filtered_sig = filtered_sig * (0.99 / peak)
        sf.write(dst_wav, filtered_sig, fs0)
        
        snr_before = calc_snr(clean_sig, noisy_sig)
        snr_after = calc_snr(clean_sig, filtered_sig)
        delta = snr_after - snr_before
        
        results.append(dict(
            phrase=ph, snr_nom=lv, file=noisy_path, 
            snr_before=snr_before, snr_after=snr_after, delta=delta
        ))
        
        trio = [
            (f"{ph} (эталон)", clean_sig),
            (f"{ph} (шум {lv}дБ)", noisy_sig),
            (f"{ph} (после очистки)", filtered_sig)
        ]
        plot_spectrograms(trio, fs0, f"{ph} {lv}дБ: Спектрограммы до/после", out_dir / "figures" / f"cmp_spec_{ph}_{lv}.png")
        plot_fft(trio, fs0, f"{ph} {lv}дБ: Спектры до/после", out_dir / "figures" / f"cmp_fft_{ph}_{lv}.png")
        plot_waveforms(trio, fs0, f"{ph} {lv}дБ: Волны до/после", out_dir / "figures" / f"cmp_wave_{ph}_{lv}.png")

    print("Этап 5. Сохранение результатов...")
    df_res = pd.DataFrame(results)
    df_res.to_csv(out_dir / "tables" / "05_experiments.csv", index=False, encoding='utf-8-sig')
    
    plt.figure(figsize=(8, 4))
    for ph in args.phrases:
        sub = df_res[df_res.phrase == ph.lower()]
        plt.plot(sub.snr_nom, sub.delta, marker='o', label=ph)
    plt.axhline(0, color='k', lw=1)
    plt.xlabel("Номинальный SNR, дБ"); plt.ylabel("ΔSNR, дБ")
    plt.title("Прирост качества после спектрального вычитания")
    plt.grid(alpha=0.3); plt.legend()
    plt.savefig(out_dir / "figures" / "delta_snr.png", dpi=120)
    plt.close()

    print("\nГотово! Результаты в папке results/")
    print(df_res.to_string(index=False))

if __name__ == "__main__":
    main()
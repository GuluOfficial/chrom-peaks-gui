#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chrom Peaks GUI (Image→Peak Pixel Areas)
----------------------------------------
- 选择色谱图图片文件（PNG/JPG等）
- 选择输出文件夹
- 调整算法参数（滑块/数字输入）
- 点击“分析”按钮→导出CSV、标注图、可选轮廓图

像素面积定义：以水平 x 轴为基线，逐列从基线向上找到曲线的第一个黑像素，
只统计该黑像素与基线之间的白色像素数量（不计黑线本身）；
对连续峰区间内的列高度求和得到峰面积（像素）。

打包成 .exe 可用 PyInstaller（见文件末尾注释）。
"""

import os
import threading
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import numpy as np
from PIL import Image
import pandas as pd

# 使用非交互后端以便后续打包
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ========== 核心算法函数（与 CLI 版一致） ==========

def moving_average(a: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1:
        return a.astype(float, copy=True)
    kernel = np.ones(k, dtype=float) / k
    return np.convolve(a, kernel, mode="same")


def detect_baseline(is_black: np.ndarray) -> int:
    h, w = is_black.shape
    bottom_start = int(h * 0.50)
    row_counts = is_black[bottom_start:].sum(axis=1)
    baseline_y = bottom_start + int(np.argmax(row_counts))
    return baseline_y


def horizontal_extent_on_row(is_black: np.ndarray, y: int):
    xs = np.where(is_black[y])[0]
    if xs.size == 0:
        return 0, is_black.shape[1] - 1
    return int(xs.min()), int(xs.max())


def remove_horizontal_gridlines(mask: np.ndarray, x_start: int, x_end: int, frac_threshold: float = 0.20) -> None:
    width_x = x_end - x_start + 1
    x_range = slice(x_start, x_end + 1)
    row_counts_in_x = mask[:, x_range].sum(axis=1)
    row_thresh = int(width_x * frac_threshold)
    rows_to_remove = np.where(row_counts_in_x > row_thresh)[0]
    if rows_to_remove.size:
        mask[rows_to_remove, x_range] = False


def compute_heights(mask: np.ndarray, baseline_y: int, x_start: int, x_end: int, y_top_limit: int) -> np.ndarray:
    width_x = x_end - x_start + 1
    heights = np.zeros(width_x, dtype=np.int32)
    for i, x in enumerate(range(x_start, x_end + 1)):
        y = baseline_y - 1
        while y > y_top_limit and not mask[y, x]:
            y -= 1
        if y <= y_top_limit:
            heights[i] = 0
        else:
            heights[i] = baseline_y - y - 1
    return heights


def detect_peaks_from_profile(heights: np.ndarray, smooth_k: int, thr_frac: float, min_height_px: int, min_width_px: int):
    smooth = moving_average(heights, k=smooth_k)
    max_h = float(smooth.max()) if smooth.size else 0.0
    thr = max(float(min_height_px), max_h * thr_frac)
    above = smooth > thr

    peaks = []
    i = 0
    n = len(smooth)
    while i < n:
        if above[i]:
            s = i
            while i < n and above[i]:
                i += 1
            e = i - 1
            if (e - s + 1) >= int(min_width_px):
                region = smooth[s:e+1]
                apex = s + int(np.argmax(region))
                area = int(heights[s:e+1].sum())
                peaks.append((s, e, apex, area))
        i += 1
    return peaks, smooth, thr


def xidx_to_minutes(idx: int, width_x: int, minutes_total: float) -> float:
    if width_x <= 1:
        return 0.0
    return (idx / (width_x - 1)) * minutes_total


def analyze_image(image_path: str,
                  outdir: str,
                  minutes_total: float = 35.0,
                  binary_thresh: int = 80,
                  top_ignore_frac: float = 0.10,
                  smooth_k: int = 5,
                  thr_frac: float = 0.03,
                  min_height_px: int = 2,
                  min_width_px: int = 3,
                  gridline_row_frac: float = 0.20,
                  save_profile: bool = True):
    # 读取图像并二值化（黑=True）
    img = Image.open(image_path).convert("L")
    arr = np.array(img)
    is_black = arr < int(binary_thresh)
    h, w = is_black.shape

    # 基线与水平范围
    baseline_y = detect_baseline(is_black)
    left, right = horizontal_extent_on_row(is_black, baseline_y)
    pad = max(3, int(0.005 * (right - left)))
    x_start, x_end = left + pad, right - pad
    width_x = x_end - x_start + 1
    y_top_limit = int(h * top_ignore_frac)

    # 去水平网格线
    mask = is_black.copy()
    remove_horizontal_gridlines(mask, x_start, x_end, frac_threshold=gridline_row_frac)

    # 白像素高度
    heights = compute_heights(mask, baseline_y, x_start, x_end, y_top_limit)

    # 峰检测
    peaks, smooth, thr = detect_peaks_from_profile(
        heights, smooth_k=smooth_k, thr_frac=thr_frac,
        min_height_px=min_height_px, min_width_px=min_width_px
    )

    # 结果表
    rows = []
    for s, e, apex, area in peaks:
        t_apex = xidx_to_minutes(apex, width_x, minutes_total)
        rows.append({"Retention Time (min)": t_apex, "Pixel Area": area})
    df = pd.DataFrame(rows).sort_values("Retention Time (min)").reset_index(drop=True)
    total_area = int(df["Pixel Area"].sum()) if not df.empty else 0
    if total_area > 0:
        df["Relative %"] = (df["Pixel Area"] / total_area) * 100.0
    else:
        df["Relative %"] = 0.0

    # 输出文件
    os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    csv_path = os.path.join(outdir, f"{base}_peaks.csv")
    overlay_path = os.path.join(outdir, f"{base}_overlay.png")
    profile_path = os.path.join(outdir, f"{base}_profile.png") if save_profile else ""

    # 保存 CSV
    df.to_csv(csv_path, index=False)

    # 叠加标注图
    overlay = np.array(Image.open(image_path).convert("RGB"))
    for s, e, apex, area in peaks:
        x = x_start + apex
        y0 = baseline_y
        y1 = max(baseline_y - 30, 0)
        for y in range(y1, y0):
            if 0 <= x < w and 0 <= y < h:
                overlay[y, x] = [0, 0, 0]
    Image.fromarray(overlay).save(overlay_path)

    # 可选：轮廓图
    if save_profile:
        plt.figure(figsize=(10, 3))
        plt.plot(np.arange(width_x), moving_average(heights, smooth_k), linewidth=1)
        for s, e, apex, area in peaks:
            plt.axvline(apex, linestyle="--", linewidth=0.8)
        plt.title("Extracted Profile (smoothed) from Image")
        plt.xlabel(f"Column index across baseline (mapped to 0–{int(minutes_total)} min)")
        plt.ylabel("Height (pixels)")
        plt.tight_layout()
        plt.savefig(profile_path, dpi=200)
        plt.close()

    return {
        "csv": csv_path,
        "overlay": overlay_path,
        "profile": profile_path,
        "peaks": len(rows),
        "total_area": total_area,
        "baseline_y": baseline_y,
        "x_range": (x_start, x_end),
        "width_x": width_x,
    }


# ========== GUI ==========

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Chrom Peaks GUI - 色谱峰像素面积分析")
        self.geometry("860x560")

        self.image_path = tk.StringVar(value="")
        self.out_dir = tk.StringVar(value="")

        # 参数变量
        self.minutes_total = tk.DoubleVar(value=35.0)
        self.binary_thresh = tk.IntVar(value=80)
        self.top_ignore_frac = tk.DoubleVar(value=0.10)
        self.smooth_k = tk.IntVar(value=5)
        self.thr_frac = tk.DoubleVar(value=0.03)
        self.min_height_px = tk.IntVar(value=2)
        self.min_width_px = tk.IntVar(value=3)
        self.gridline_row_frac = tk.DoubleVar(value=0.20)
        self.save_profile = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 8, "pady": 6}

        # 文件选择
        frm_paths = ttk.LabelFrame(self, text="文件")
        frm_paths.pack(fill="x", **pad)

        row = 0
        ttk.Label(frm_paths, text="色谱图图片：").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm_paths, textvariable=self.image_path, width=70).grid(row=row, column=1, sticky="we")
        ttk.Button(frm_paths, text="选择图片…", command=self.select_image).grid(row=row, column=2, sticky="e")
        row += 1
        ttk.Label(frm_paths, text="输出文件夹：").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm_paths, textvariable=self.out_dir, width=70).grid(row=row, column=1, sticky="we")
        ttk.Button(frm_paths, text="选择文件夹…", command=self.select_outdir).grid(row=row, column=2, sticky="e")

        # 参数
        frm_params = ttk.LabelFrame(self, text="参数（可调）")
        frm_params.pack(fill="both", expand=True, **pad)

        def add_scale(row, text, var, frm, from_, to_, resolution, fmt, help_text):
            ttk.Label(frm, text=text).grid(row=row, column=0, sticky="w")
            scale = ttk.Scale(frm, variable=var, from_=from_, to=to_, orient="horizontal")
            scale.grid(row=row, column=1, sticky="we", padx=6)
            ent = ttk.Entry(frm, width=7)
            ent.grid(row=row, column=2, sticky="e")
            def sync_from_var(*_):
                ent.delete(0, "end")
                ent.insert(0, fmt(var.get()))
            def sync_from_entry(event=None):
                try:
                    v = float(ent.get())
                    if isinstance(var, tk.IntVar):
                        v = int(round(v))
                    var.set(v)
                except Exception:
                    pass
            var.trace_add("write", sync_from_var)
            ent.bind("<Return>", sync_from_entry)
            sync_from_var()
            # 帮助
            ttk.Label(frm, text=help_text, foreground="#555").grid(row=row+1, column=1, columnspan=2, sticky="w", pady=(0,8))

        def add_spin(row, text, var, frm, from_, to_, help_text):
            ttk.Label(frm, text=text).grid(row=row, column=0, sticky="w")
            sp = ttk.Spinbox(frm, textvariable=var, from_=from_, to=to_, width=7)
            sp.grid(row=row, column=2, sticky="e")
            ttk.Label(frm, text=help_text, foreground="#555").grid(row=row+1, column=1, columnspan=2, sticky="w", pady=(0,8))

        frm_params.columnconfigure(1, weight=1)

        r = 0
        add_scale(r, "X 轴总分钟数", self.minutes_total, frm_params, 1, 240, 1, lambda v: f"{v:.0f}",
                  "将基线水平范围线性映射到 0–此分钟数。")
        r += 2
        add_spin(r, "二值化阈值(0–255)", self.binary_thresh, frm_params, 0, 255,
                 "判定黑像素的灰度阈值，越小越“苛刻”，通常 60–120。")
        r += 2
        add_scale(r, "顶部忽略比例", self.top_ignore_frac, frm_params, 0.0, 0.30, 0.01, lambda v: f"{v:.2f}",
                  "忽略图像上方的一部分（排除标题、坐标等）。")
        r += 2
        add_spin(r, "平滑窗口(像素)", self.smooth_k, frm_params, 1, 21,
                 "列高度的移动平均窗口，增大可抑制噪声、但会钝化尖峰。")
        r += 2
        add_scale(r, "峰阈值比例", self.thr_frac, frm_params, 0.0, 0.20, 0.005, lambda v: f"{v:.3f}",
                  "峰检测阈值=最高平滑高度×该比例，越大越少峰。")
        r += 2
        add_spin(r, "最小峰高(像素)", self.min_height_px, frm_params, 0, 30,
                 "峰必须至少达到的平滑高度下限。")
        r += 2
        add_spin(r, "最小峰宽(像素)", self.min_width_px, frm_params, 1, 60,
                 "峰的最小连续宽度。")
        r += 2
        add_scale(r, "水平网格剔除强度", self.gridline_row_frac, frm_params, 0.05, 0.40, 0.01, lambda v: f"{v:.2f}",
                  "行内黑像素密度阈值（越小越容易视为网格线并剔除）。")
        r += 2

        ttk.Checkbutton(frm_params, text="保存轮廓图 (profile.png)", variable=self.save_profile).grid(row=r, column=1, sticky="w", pady=6)
        r += 1

        # 底部操作区
        frm_actions = ttk.Frame(self)
        frm_actions.pack(fill="x", **pad)
        self.progress = ttk.Progressbar(frm_actions, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True, padx=(0,8))
        self.btn_analyze = ttk.Button(frm_actions, text="分析", command=self.on_analyze)
        self.btn_analyze.pack(side="right")

        # 说明
        lbl_help = ttk.Label(self, text=(
            "说明：面积按“白色像素数量”计算，不计曲线黑线像素；仅基于图像，"
            "会受线宽/像素化/网格影响。建议尽量使用高分辨率图。"
        ), foreground="#444")
        lbl_help.pack(fill="x", padx=8, pady=(0,8))

    def select_image(self):
        path = filedialog.askopenfilename(
            title="选择色谱图图片",
            filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"), ("All Files", "*.*")]
        )
        if path:
            self.image_path.set(path)

    def select_outdir(self):
        path = filedialog.askdirectory(title="选择输出文件夹")
        if path:
            self.out_dir.set(path)

    def on_analyze(self):
        if not self.image_path.get():
            messagebox.showwarning("缺少图片", "请先选择色谱图图片文件。")
            return
        if not self.out_dir.get():
            messagebox.showwarning("缺少输出路径", "请先选择输出文件夹。")
            return

        self.btn_analyze.config(state="disabled")
        self.progress.start(10)

        def worker():
            try:
                result = analyze_image(
                    image_path=self.image_path.get(),
                    outdir=self.out_dir.get(),
                    minutes_total=float(self.minutes_total.get()),
                    binary_thresh=int(self.binary_thresh.get()),
                    top_ignore_frac=float(self.top_ignore_frac.get()),
                    smooth_k=int(self.smooth_k.get()),
                    thr_frac=float(self.thr_frac.get()),
                    min_height_px=int(self.min_height_px.get()),
                    min_width_px=int(self.min_width_px.get()),
                    gridline_row_frac=float(self.gridline_row_frac.get()),
                    save_profile=bool(self.save_profile.get()),
                )
                msg = (
                    f"分析完成！\n"
                    f"峰个数：{result['peaks']}\n"
                    f"总像素面积：{result['total_area']}\n\n"
                    f"CSV：{os.path.basename(result['csv'])}\n"
                    f"标注图：{os.path.basename(result['overlay'])}\n"
                )
                if result['profile']:
                    msg += f"轮廓图：{os.path.basename(result['profile'])}\n"
                def done_msg():
                    self.progress.stop()
                    self.btn_analyze.config(state="normal")
                    if messagebox.askyesno("完成", msg + "\n是否打开输出文件夹？"):
                        try:
                            # Windows
                            os.startfile(self.out_dir.get())
                        except Exception:
                            # 其他平台
                            messagebox.showinfo("路径", self.out_dir.get())
                self.after(0, done_msg)
            except Exception as e:
                tb = traceback.format_exc()
                def err_msg():
                    self.progress.stop()
                    self.btn_analyze.config(state="normal")
                    messagebox.showerror("出错了", f"{e}\n\n{tb}")
                self.after(0, err_msg)

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    # Windows 环境下，后续可用 PyInstaller 打包为 .exe ：
    #   pyinstaller --noconsole --onefile --name ChromPeaksGUI chrom_peaks_gui.py
    # 生成的 exe 在 dist/ChromPeaksGUI.exe
    app = App()
    app.mainloop()

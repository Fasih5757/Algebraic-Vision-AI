"""
eda_analyzer.py
---------------
Full Exploratory Data Analysis (EDA) pipeline for images.

Produces a comprehensive 6-panel report:
  1. Original image
  2. Edge detection (Canny)
  3. RGB / HSV color distribution histograms
  4. Intensity heatmap
  5. Symmetry difference map
  6. Statistical summary table

Also computes:
  - Fourier magnitude spectrum
  - Texture features (GLCM-like)
  - Fractal dimension estimate (box-counting)
  - Pattern confidence scores
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

PHI = 1.6180339887


# ── Fractal dimension (box-counting) ─────────────────────────────────────────

def _box_count_dimension(binary_img: np.ndarray) -> float:
    """Estimate fractal dimension using box-counting method."""
    def _count_boxes(img, box_size):
        h, w = img.shape
        count = 0
        for i in range(0, h, box_size):
            for j in range(0, w, box_size):
                patch = img[i:i+box_size, j:j+box_size]
                if np.any(patch > 0):
                    count += 1
        return count

    sizes = [2, 4, 8, 16, 32, 64]
    counts = []
    for s in sizes:
        c = _count_boxes(binary_img, s)
        if c > 0:
            counts.append((s, c))

    if len(counts) < 2:
        return 1.0

    log_s = np.log([1.0 / s for s, _ in counts])
    log_c = np.log([c for _, c in counts])
    coeffs = np.polyfit(log_s, log_c, 1)
    return float(coeffs[0])


# ── Fourier spectrum ──────────────────────────────────────────────────────────

def _fourier_spectrum(gray: np.ndarray) -> np.ndarray:
    """Return log-magnitude Fourier spectrum (shifted to center)."""
    f    = np.fft.fft2(gray.astype(np.float32))
    fsh  = np.fft.fftshift(f)
    mag  = np.log1p(np.abs(fsh))
    return mag


# ── Texture features ──────────────────────────────────────────────────────────

def _texture_features(gray: np.ndarray) -> dict:
    """Simple texture descriptors."""
    # Local Binary Pattern approximation
    h, w = gray.shape
    g = gray.astype(np.float32)

    # Gradient magnitude
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)

    # Laplacian (sharpness)
    lap = cv2.Laplacian(gray, cv2.CV_64F)

    return {
        "mean_gradient":   float(np.mean(grad_mag)),
        "std_gradient":    float(np.std(grad_mag)),
        "sharpness":       float(np.var(lap)),
        "contrast":        float(gray.max() - gray.min()),
        "homogeneity":     float(1.0 / (1.0 + np.std(gray))),
    }


# ── Symmetry analysis ─────────────────────────────────────────────────────────

def _symmetry_scores(gray: np.ndarray) -> dict:
    h, w = gray.shape

    # Horizontal
    mid_x = w // 2
    left  = gray[:, :mid_x]
    right = cv2.resize(cv2.flip(gray[:, mid_x:], 1),
                       (left.shape[1], left.shape[0]))
    h_score = 100.0 - (np.mean(cv2.absdiff(left, right)) / 255.0 * 100.0)

    # Vertical
    mid_y  = h // 2
    top    = gray[:mid_y, :]
    bottom = cv2.resize(cv2.flip(gray[mid_y:, :], 0),
                        (top.shape[1], top.shape[0]))
    v_score = 100.0 - (np.mean(cv2.absdiff(top, bottom)) / 255.0 * 100.0)

    # Diagonal (rotate 90 and compare)
    rot90 = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    rot90_r = cv2.resize(rot90, (w, h))
    d_score = 100.0 - (np.mean(cv2.absdiff(gray, rot90_r)) / 255.0 * 100.0)

    # Diff map for visualization
    diff_map = cv2.absdiff(left, right)

    return {
        "horizontal": h_score,
        "vertical":   v_score,
        "diagonal":   d_score,
        "diff_map":   diff_map,
    }


# ── Color analysis ────────────────────────────────────────────────────────────

def _color_stats(img_bgr: np.ndarray) -> dict:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    dominant_colors = []
    for ch in range(3):
        hist = cv2.calcHist([img_rgb], [ch], None, [256], [0, 256]).flatten()
        dominant_colors.append(int(np.argmax(hist)))

    # Color entropy
    hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180]).flatten()
    hist_h = hist_h / (hist_h.sum() + 1e-9)
    entropy = float(-np.sum(hist_h * np.log2(hist_h + 1e-9)))

    return {
        "dominant_rgb": dominant_colors,
        "color_entropy": entropy,
        "mean_hue":      float(np.mean(hsv[:, :, 0])),
        "mean_sat":      float(np.mean(hsv[:, :, 1])),
        "mean_val":      float(np.mean(hsv[:, :, 2])),
    }


# ── Pattern confidence scores ─────────────────────────────────────────────────

def _pattern_scores(gray: np.ndarray, sym: dict, texture: dict,
                    h: int, w: int) -> dict:
    """Heuristic confidence scores for each pattern type."""
    edge_density = np.sum(cv2.Canny(gray, 50, 150) > 0) / (h * w)
    ar = w / max(h, 1)

    scores = {
        "Golden Ratio":     min(100, max(0, 100 - abs(ar - PHI) / PHI * 100)),
        "Bilateral Sym.":   sym["horizontal"],
        "Radial Sym.":      sym["diagonal"],
        "Fractal Pattern":  min(100, edge_density * 300),
        "Wave / Periodic":  min(100, texture["std_gradient"] / 50 * 100),
        "Fibonacci Spiral": min(100, sym["horizontal"] * 0.5 + edge_density * 150),
    }
    return {k: round(v, 1) for k, v in scores.items()}


# ── Main EDA function ─────────────────────────────────────────────────────────

def full_eda_report(image_path: str, dark_bg: bool = True):
    """
    Generates a comprehensive EDA report figure.

    Returns:
      (fig, stats_dict)
      stats_dict contains all computed metrics.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w    = gray.shape

    # ── Compute all metrics ───────────────────────────────────────────────────
    sym      = _symmetry_scores(gray)
    texture  = _texture_features(gray)
    color    = _color_stats(img_bgr)
    patterns = _pattern_scores(gray, sym, texture, h, w)
    fourier  = _fourier_spectrum(gray)

    edges    = cv2.Canny(gray, 50, 150)
    _, binary = cv2.threshold(gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    frac_dim = _box_count_dimension(binary)

    # ── Figure layout ─────────────────────────────────────────────────────────
    bg = '#0a0a0a' if dark_bg else 'white'
    fg = 'white'   if dark_bg else 'black'
    panel_bg = '#0d0d0d' if dark_bg else '#f5f5f5'

    fig = plt.figure(figsize=(20, 13))
    fig.patch.set_facecolor(bg)
    fig.suptitle('AlgebraicVision AI — Full EDA Report',
                 color='gold', fontsize=17, fontweight='bold', y=0.98)

    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.38, wspace=0.32)

    def _style(ax, title, color='cyan'):
        ax.set_facecolor(panel_bg)
        ax.tick_params(colors=fg, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color('#333' if dark_bg else '#ccc')
        ax.set_title(title, color=color, fontsize=10, fontweight='bold', pad=6)

    # Panel 1 — Original
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    ax1.axis('off')
    _style(ax1, 'Original Image')

    # Panel 2 — Edge Detection
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(edges, cmap='hot')
    ax2.axis('off')
    _style(ax2, 'Canny Edge Detection', 'orange')

    # Panel 3 — Fourier Spectrum
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(fourier, cmap='inferno')
    ax3.axis('off')
    _style(ax3, 'Fourier Magnitude Spectrum', 'magenta')

    # Panel 4 — Intensity Heatmap
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(gray, cmap='plasma')
    ax4.axis('off')
    _style(ax4, 'Intensity Heatmap', 'yellow')

    # Panel 5 — RGB Histogram
    ax5 = fig.add_subplot(gs[1, 0:2])
    _style(ax5, 'RGB Color Distribution', 'lime')
    for ch, col in enumerate(['red', 'green', 'blue']):
        hist = cv2.calcHist([img_rgb], [ch], None, [256], [0, 256]).flatten()
        ax5.plot(hist, color=col, alpha=0.85, linewidth=1.5, label=col.upper())
    ax5.set_xlabel('Pixel Value', color=fg, fontsize=8)
    ax5.set_ylabel('Frequency',   color=fg, fontsize=8)
    ax5.legend(facecolor='#111' if dark_bg else '#eee',
               labelcolor=fg, fontsize=8)

    # Panel 6 — Symmetry Diff Map
    ax6 = fig.add_subplot(gs[1, 2:4])
    ax6.imshow(sym["diff_map"], cmap='RdYlGn_r')
    ax6.axis('off')
    _style(ax6,
           f'Symmetry Difference Map  |  H:{sym["horizontal"]:.1f}%  '
           f'V:{sym["vertical"]:.1f}%  D:{sym["diagonal"]:.1f}%',
           'cyan')

    # Panel 7 — Pattern Confidence Bar Chart
    ax7 = fig.add_subplot(gs[2, 0:2])
    _style(ax7, 'Pattern Confidence Scores', 'gold')
    names  = list(patterns.keys())
    values = list(patterns.values())
    colors_bar = ['#00ffff', '#ffd700', '#ff00ff', '#00ff88', '#ff8800', '#ff4488']
    bars = ax7.barh(names, values, color=colors_bar, alpha=0.85, height=0.6)
    ax7.set_xlim(0, 105)
    ax7.set_xlabel('Confidence (%)', color=fg, fontsize=8)
    for bar, val in zip(bars, values):
        ax7.text(val + 1, bar.get_y() + bar.get_height() / 2,
                 f'{val:.1f}%', va='center', color=fg, fontsize=8)

    # Panel 8 — Statistics Table
    ax8 = fig.add_subplot(gs[2, 2:4])
    ax8.set_facecolor(panel_bg)
    ax8.axis('off')
    _style(ax8, 'Mathematical Statistics', 'gold')

    stats_rows = [
        ("Image Size",         f"{w} × {h} px"),
        ("Aspect Ratio",       f"{w/h:.4f}"),
        ("vs Golden Ratio φ",  f"Δ = {abs(w/h - PHI):.4f}"),
        ("Golden Point X",     f"{w/PHI:.0f} px"),
        ("Golden Point Y",     f"{h/PHI:.0f} px"),
        ("Mean Brightness",    f"{np.mean(gray):.1f} / 255"),
        ("Std Deviation",      f"{np.std(gray):.1f}"),
        ("Contrast",           f"{int(gray.max() - gray.min())}"),
        ("Color Entropy",      f"{color['color_entropy']:.3f} bits"),
        ("Fractal Dimension",  f"{frac_dim:.3f}"),
        ("Sharpness (Lap.)",   f"{texture['sharpness']:.1f}"),
        ("Edge Density",       f"{np.sum(edges>0)/(h*w)*100:.2f}%"),
        ("H. Symmetry",        f"{sym['horizontal']:.1f}%"),
        ("V. Symmetry",        f"{sym['vertical']:.1f}%"),
        ("Total Pixels",       f"{w*h:,}"),
    ]

    y = 0.96
    for label_t, val_t in stats_rows:
        ax8.text(0.03, y, label_t,
                 transform=ax8.transAxes, color='#aaaaaa', fontsize=8.5)
        ax8.text(0.62, y, val_t,
                 transform=ax8.transAxes, color='cyan',
                 fontsize=8.5, fontweight='bold')
        y -= 0.063

    # Compile stats dict
    stats_dict = {
        "dimensions":       (w, h),
        "aspect_ratio":     w / h,
        "golden_diff":      abs(w / h - PHI),
        "symmetry":         sym,
        "texture":          texture,
        "color":            color,
        "pattern_scores":   patterns,
        "fractal_dim":      frac_dim,
        "mean_brightness":  float(np.mean(gray)),
        "std_brightness":   float(np.std(gray)),
        "edge_density":     float(np.sum(edges > 0) / (h * w)),
    }

    return fig, stats_dict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

def draw_epicycles_static(shape='bird', n_terms=50):
    """
    Fourier Epicycles — koi bhi shape draw karo
    spinning circles se — bilkul TikTok wala!
    """

    # ── Shape coordinates ─────────────────────────────
    shapes = {
        'bird': _bird_coords(),
        'heart': _heart_coords(),
        'star':  _star_coords(),
        'infinity': _infinity_coords(),
        'fish': _fish_coords(),
    }

    coords = shapes.get(shape, _bird_coords())
    x_pts, y_pts = coords

    # Complex signal banao
    signal = x_pts + 1j * y_pts

    # DFT compute karo
    N = len(signal)
    coeffs = np.fft.fft(signal) / N

    # Sort by amplitude (largest circles first)
    freqs   = np.fft.fftfreq(N, 1/N)
    amps    = np.abs(coeffs)
    order   = np.argsort(-amps)

    coeffs_sorted = coeffs[order[:n_terms]]
    freqs_sorted  = freqs[order[:n_terms]]

    # ── Draw final traced path ─────────────────────────
    t_vals = np.linspace(0, 2*np.pi, 500)
    path_x, path_y = [], []

    for t in t_vals:
        x, y = 0.0, 0.0
        for c, f in zip(coeffs_sorted, freqs_sorted):
            angle = f * t
            x += (c * np.exp(1j * angle)).real
            y += (c * np.exp(1j * angle)).imag
        path_x.append(x)
        path_y.append(y)

    # ── One snapshot — circles at t=π ─────────────────
    t_snap = np.pi
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('#0a0a0a')

    for ax in axes:
        ax.set_facecolor('#0a0a0a')
        ax.set_aspect('equal')
        ax.axis('off')

    # Left — Epicycles at snapshot
    cx, cy = 0.0, 0.0
    colors = plt.cm.plasma(np.linspace(0.2, 1, len(coeffs_sorted)))

    for i, (c, f) in enumerate(zip(coeffs_sorted, freqs_sorted)):
        r = abs(c)
        if r < 0.001:
            continue
        angle = f * t_snap + np.angle(c)

        # Circle
        circle = Circle((cx, cy), r,
                        fill=False,
                        color=colors[i % len(colors)],
                        linewidth=0.8, alpha=0.5)
        axes[0].add_patch(circle)

        # Radius line
        nx = cx + r * np.cos(angle)
        ny = cy + r * np.sin(angle)
        axes[0].plot([cx, nx], [cy, ny],
                    color=colors[i % len(colors)],
                    linewidth=1.2, alpha=0.8)
        cx, cy = nx, ny

    # Dot at tip
    axes[0].plot(cx, cy, 'o', color='white', markersize=4)

    axes[0].set_title(f'Fourier Epicycles — {n_terms} circles',
                      color='cyan', fontsize=13, fontweight='bold')
    axes[0].autoscale()

    # Right — Traced path
    axes[1].plot(path_x, path_y, color='cyan',
                linewidth=2, alpha=0.9)
    axes[1].fill(path_x, path_y,
                alpha=0.08, color='cyan')
    axes[1].set_title(f'Traced Shape: {shape.upper()}',
                      color='gold', fontsize=13, fontweight='bold')

    plt.tight_layout()
    return fig


# ── Shape definitions ──────────────────────────────────────────

def _bird_coords():
    t = np.linspace(0, 2*np.pi, 300)
    # Body
    x = np.cos(t) * (1 + 0.3*np.cos(3*t))
    y = np.sin(t) * (0.6 + 0.2*np.sin(2*t))
    # Wing bump
    x += 0.4 * np.cos(2*t) * np.cos(t)
    y += 0.3 * np.sin(t) * np.abs(np.cos(t))
    return x, y

def _heart_coords():
    t = np.linspace(0, 2*np.pi, 300)
    x = 16 * np.sin(t)**3
    y = 13*np.cos(t) - 5*np.cos(2*t) - 2*np.cos(3*t) - np.cos(4*t)
    x = x / 16
    y = y / 16
    return x, y

def _star_coords():
    t = np.linspace(0, 2*np.pi, 300)
    r = 1 + 0.5 * np.cos(5*t)
    x = r * np.cos(t)
    y = r * np.sin(t)
    return x, y

def _infinity_coords():
    t = np.linspace(0, 2*np.pi, 300)
    x = np.cos(t) / (1 + np.sin(t)**2)
    y = np.sin(t)*np.cos(t) / (1 + np.sin(t)**2)
    return x, y

def _fish_coords():
    t = np.linspace(0, 2*np.pi, 300)
    x = np.cos(t) + 0.3*np.cos(2*t)
    y = 0.5*np.sin(t)
    return x, y
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
import re
import warnings
import unicodedata
warnings.filterwarnings("ignore")

# ── Unicode math character normalization ──────────────────────────────────────
# Maps unicode mathematical italic/bold/script letters to ASCII equivalents
# e.g. 𝑥 (U+1D465) -> x,  𝑎 (U+1D44E) -> a,  𝑧 (U+1D467) -> z
_UNICODE_MAP = {}

def _build_unicode_map():
    """Build a mapping from unicode math letters to ASCII."""
    # Mathematical italic lowercase: 𝑎-𝑧 (U+1D44E to U+1D467)
    for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz"):
        _UNICODE_MAP[chr(0x1D44E + i)] = ch   # italic
        _UNICODE_MAP[chr(0x1D41A + i)] = ch   # bold
        _UNICODE_MAP[chr(0x1D482 + i)] = ch   # bold italic
        _UNICODE_MAP[chr(0x1D4B6 + i)] = ch   # script (some)
        _UNICODE_MAP[chr(0x1D552 + i)] = ch   # double-struck
        _UNICODE_MAP[chr(0x1D586 + i)] = ch   # sans-serif
        _UNICODE_MAP[chr(0x1D5BA + i)] = ch   # sans-serif bold
        _UNICODE_MAP[chr(0x1D5EE + i)] = ch   # sans-serif italic
        _UNICODE_MAP[chr(0x1D622 + i)] = ch   # sans-serif bold italic
        _UNICODE_MAP[chr(0x1D656 + i)] = ch   # monospace
    # Mathematical italic uppercase: 𝐴-𝑍 (U+1D434 to U+1D44D)
    for i, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        _UNICODE_MAP[chr(0x1D434 + i)] = ch   # italic
        _UNICODE_MAP[chr(0x1D400 + i)] = ch   # bold
        _UNICODE_MAP[chr(0x1D468 + i)] = ch   # bold italic
    # Mathematical digits 𝟎-𝟗 (U+1D7CE to U+1D7D7)
    for i in range(10):
        _UNICODE_MAP[chr(0x1D7CE + i)] = str(i)  # bold digits
        _UNICODE_MAP[chr(0x1D7D8 + i)] = str(i)  # double-struck
        _UNICODE_MAP[chr(0x1D7E2 + i)] = str(i)  # sans-serif
        _UNICODE_MAP[chr(0x1D7EC + i)] = str(i)  # sans-serif bold
        _UNICODE_MAP[chr(0x1D7F6 + i)] = str(i)  # monospace
    # Greek letters commonly used in math
    greek = {
        '\u03b1': 'alpha', '\u03b2': 'beta',  '\u03b3': 'gamma',
        '\u03b4': 'delta', '\u03b5': 'eps',   '\u03b8': 'theta',
        '\u03bb': 'lam',   '\u03bc': 'mu',    '\u03c0': 'pi',
        '\u03c3': 'sigma', '\u03c6': 'phi',   '\u03c9': 'omega',
        '\u03a3': 'Sigma', '\u03a9': 'Omega',
    }
    _UNICODE_MAP.update(greek)
    # Superscript digits ² ³ etc.
    _UNICODE_MAP.update({
        '\u00b2': '**2', '\u00b3': '**3',
        '\u2070': '**0', '\u00b9': '**1',
        '\u2074': '**4', '\u2075': '**5',
        '\u2076': '**6', '\u2077': '**7',
        '\u2078': '**8', '\u2079': '**9',
    })
    # Multiplication / division symbols
    _UNICODE_MAP.update({
        '\u00d7': '*', '\u00f7': '/', '\u22c5': '*',
        '\u2212': '-', '\u2260': '!=', '\u2264': '<=', '\u2265': '>=',
    })

_build_unicode_map()

def normalize_unicode(s):
    """
    Convert any unicode math characters in s to ASCII equivalents.
    Also applies NFKC normalization to catch composed forms.
    """
    # NFKC normalization first (handles many compatibility chars)
    s = unicodedata.normalize("NFKC", s)
    # Apply our custom map character by character
    result = []
    for ch in s:
        result.append(_UNICODE_MAP.get(ch, ch))
    return "".join(result)

# Default values for symbolic constants a,b,c,n,k,m etc.
CONST_DEFAULTS = {
    "a": 3.0, "b": 2.0, "c": 1.0,
    "n": 3.0, "k": 2.0, "m": 1.0,
    "r": 5.0, "h": 1.0, "p": 2.0,
    "phi": 1.6180339887,
    "omega": 2.0, "alpha": 0.5, "beta": 1.0,
}

def _safe_ns(**extra):
    ns = {
        "sin": np.sin, "cos": np.cos, "tan": np.tan,
        "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
        "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
        "exp": np.exp, "log": np.log, "log2": np.log2, "log10": np.log10,
        "sqrt": np.sqrt, "abs": np.abs, "Abs": np.abs,
        "sign": np.sign, "floor": np.floor, "ceil": np.ceil,
        "pi": np.pi, "e": np.e, "phi": 1.6180339887, "inf": np.inf,
    }
    ns.update(CONST_DEFAULTS)
    ns.update(extra)
    return ns

def safe_eval(expr_str, extra=None):
    expr_str = normalize_unicode(expr_str)
    ns = _safe_ns(**(extra or {}))
    expr = expr_str.replace("^", "**")
    expr = re.sub(r"(\d)([a-df-wyzA-Z])", r"\1*\2", expr)
    return eval(expr, {"__builtins__": {}}, ns)

def _prep(s):
    s = s.strip().replace("^", "**")
    s = re.sub(r"(\d)([a-df-wyzA-Z])", r"\1*\2", s)
    return s

def _has_var(expr, var):
    return bool(re.search(r"\b" + var + r"\b", expr))

PRESET_MAP = {
    # ── 2D Parametric ─────────────────────────────────────────────────────────
    "circle":              ("parametric", ("cos(t)", "sin(t)")),
    "ellipse":             ("parametric", ("2*cos(t)", "sin(t)")),
    "spiral":              ("parametric", ("t*cos(t)", "t*sin(t)")),
    "rose curve":          ("parametric", ("cos(3*t)*cos(t)", "cos(3*t)*sin(t)")),
    "rose 5":              ("parametric", ("cos(5*t)*cos(t)", "cos(5*t)*sin(t)")),
    "lissajous":           ("parametric", ("sin(3*t)", "sin(2*t)")),
    "lissajous 5:4":       ("parametric", ("sin(5*t)", "sin(4*t)")),
    "butterfly":           ("parametric", ("sin(t)*(exp(cos(t))-2*cos(4*t))",
                                           "cos(t)*(exp(cos(t))-2*cos(4*t))")),
    "golden spiral":       ("parametric", ("exp(0.3063*t)*cos(t)", "exp(0.3063*t)*sin(t)")),
    "heart":               ("parametric", ("16*sin(t)**3/16",
                                           "(13*cos(t)-5*cos(2*t)-2*cos(3*t)-cos(4*t))/16")),
    "asteroid":            ("parametric", ("cos(t)**3", "sin(t)**3")),
    "cycloid":             ("parametric", ("t-sin(t)", "1-cos(t)")),
    "infinity":            ("parametric", ("cos(t)/(1+sin(t)**2)", "sin(t)*cos(t)/(1+sin(t)**2)")),
    "epicycloid":          ("parametric", ("5*cos(t)-cos(5*t)", "5*sin(t)-sin(5*t)")),
    "hypotrochoid":        ("parametric", ("3*cos(t)+cos(3*t)", "3*sin(t)-sin(3*t)")),
    "spirograph":          ("parametric", ("8*cos(t)-3*cos(8/3*t)", "8*sin(t)-3*sin(8/3*t)")),
    "trefoil":             ("parametric", ("sin(t)+2*sin(2*t)", "cos(t)-2*cos(2*t)")),
    "deltoid":             ("parametric", ("2*cos(t)+cos(2*t)", "2*sin(t)-sin(2*t)")),
    "nephroid":            ("parametric", ("3*cos(t)-cos(3*t)", "3*sin(t)-sin(3*t)")),
    "folium":              ("parametric", ("3*t/(1+t**3)", "3*t**2/(1+t**3)")),
    "witch of agnesi":     ("explicit",   "8/(x**2+4)"),
    "tractrix":            ("parametric", ("log(abs(1/cos(t))+tan(t))-sin(t)", "cos(t)")),
    "catenary":            ("explicit",   "cosh(x)"),
    "parabola":            ("explicit",   "x**2"),
    "cubic":               ("explicit",   "x**3 - 3*x"),
    "quartic":             ("explicit",   "x**4 - 4*x**2"),
    "sine":                ("explicit",   "sin(x)"),
    "cosine":              ("explicit",   "cos(x)"),
    "tangent":             ("explicit",   "tan(x)"),
    "sigmoid":             ("explicit",   "1/(1+exp(-x))"),
    "gaussian bell":       ("explicit",   "exp(-x**2/2)"),
    "sinc":                ("explicit",   "sin(x)/x"),
    "damped wave":         ("explicit",   "exp(-x/5)*sin(x)"),
    "square wave approx":  ("explicit",   "sin(x)+sin(3*x)/3+sin(5*x)/5+sin(7*x)/7"),
    "sawtooth approx":     ("explicit",   "sin(x)-sin(2*x)/2+sin(3*x)/3-sin(4*x)/4"),
    "absolute value":      ("explicit",   "abs(x)"),
    "reciprocal":          ("explicit",   "1/x"),
    "square root":         ("explicit",   "sqrt(abs(x))"),
    "natural log":         ("explicit",   "log(abs(x))"),
    "exponential":         ("explicit",   "exp(x/3)"),
    "logistic":            ("explicit",   "1/(1+exp(-x))"),
    "step function":       ("explicit",   "sign(x)"),
    "floor function":      ("explicit",   "floor(x)"),
    "hyperbolic sine":     ("explicit",   "sinh(x)"),
    "hyperbolic cosine":   ("explicit",   "cosh(x)"),
    "hyperbolic tangent":  ("explicit",   "tanh(x)"),
    # ── 2D Polar ──────────────────────────────────────────────────────────────
    "cardioid":            ("polar", "1+cos(theta)"),
    "archimedean spiral":  ("polar", "theta"),
    "fermat spiral":       ("polar", "sqrt(theta)"),
    "logarithmic spiral":  ("polar", "exp(0.2*theta)"),
    "limacon":             ("polar", "1+2*cos(theta)"),
    "lemniscate":          ("polar", "sqrt(abs(cos(2*theta)))"),
    "rose 3":              ("polar", "cos(3*theta)"),
    "rose 4":              ("polar", "cos(4*theta)"),
    "rose 5 polar":        ("polar", "cos(5*theta)"),
    "cissoid":             ("polar", "2*sin(theta)*tan(theta)"),
    "strophoid":           ("polar", "cos(2*theta)/cos(theta)"),
    "maclaurin trisectrix":("polar", "1+2*cos(theta)"),
    "cochleoid":           ("polar", "sin(theta)/theta"),
    "bifolium":            ("polar", "sin(theta)*cos(theta)**2"),
    # ── 2D Implicit ───────────────────────────────────────────────────────────
    "unit circle":         ("implicit", "x**2 + y**2 - 1"),
    "circle r5":           ("implicit", "x**2 + y**2 - 25"),
    "ellipse implicit":    ("implicit", "x**2/9 + y**2/4 - 1"),
    "hyperbola":           ("implicit", "x**2/4 - y**2/9 - 1"),
    "parabola implicit":   ("implicit", "y - x**2"),
    "cubic implicit":      ("implicit", "y**2 - x**3 + x"),
    "folium descartes":    ("implicit", "x**3 + y**3 - 3*x*y"),
    "cassini oval":        ("implicit", "(x**2+y**2)**2 - 2*(x**2-y**2) - 1"),
    "bernoulli":           ("implicit", "(x**2+y**2)**2 - 2*(x**2-y**2)"),
    "astroid implicit":    ("implicit", "x**(2/3) + y**(2/3) - 1"),
    "bicorn":              ("implicit", "y**2*(x**2+2*y-1) - (1-y**2)**2"),
    "piriform":            ("implicit", "y**2 - x**3*(1-x)"),
    "kappa curve":         ("implicit", "y**2*(x**2+y**2) - x**2"),
    "devil curve":         ("implicit", "y**4 - x**4 + a*y**2 + b*x**2"),
    # ── 3D Explicit z=f(x,y) ──────────────────────────────────────────────────
    "saddle surface":      ("3d_explicit", "x**2 - y**2"),
    "gaussian":            ("3d_explicit", "exp(-(x**2+y**2))"),
    "ripple":              ("3d_explicit", "sin(x)*cos(y)"),
    "paraboloid":          ("3d_explicit", "x**2 + y**2"),
    "monkey saddle":       ("3d_explicit", "x**3 - 3*x*y**2"),
    "sinc 2d":             ("3d_explicit", "sin(sqrt(x**2+y**2))/(sqrt(x**2+y**2)+0.001)"),
    "mexican hat":         ("3d_explicit", "(1-(x**2+y**2))*exp(-(x**2+y**2)/2)"),
    "peaks":               ("3d_explicit", "3*(1-x)**2*exp(-x**2-(y+1)**2)-10*(x/5-x**3-y**5)*exp(-x**2-y**2)-exp(-(x+1)**2-y**2)/3"),
    "helicoid":            ("3d_explicit", "arctan(y/(x+0.001))"),
    "cross cap":           ("3d_explicit", "x*y"),
    "enneper":             ("3d_explicit", "x**2/3 - y**2/3"),
    "cone":                ("3d_explicit", "sqrt(x**2+y**2)"),
    "wave interference":   ("3d_explicit", "sin(x)*sin(y)"),
    "gravity well":        ("3d_explicit", "-1/sqrt(x**2+y**2+0.1)"),
    "electric potential":  ("3d_explicit", "1/sqrt((x-1)**2+y**2+0.1) - 1/sqrt((x+1)**2+y**2+0.1)"),
    # ── 3D Implicit f(x,y,z)=0 ────────────────────────────────────────────────
    "sphere":              ("3d_implicit", "x**2 + y**2 + z**2 - 25"),
    "unit sphere":         ("3d_implicit", "x**2 + y**2 + z**2 - 1"),
    "ellipsoid":           ("3d_implicit", "x**2/9 + y**2/4 + z**2/1 - 1"),
    "torus":               ("3d_implicit", "(sqrt(x**2+y**2)-3)**2 + z**2 - 1"),
    "hyperboloid":         ("3d_implicit", "x**2 + y**2 - z**2 - 1"),
    "hyperboloid 2":       ("3d_implicit", "x**2/4 + y**2/4 - z**2/9 - 1"),
    "cone 3d":             ("3d_implicit", "x**2 + y**2 - z**2"),
    "cylinder":            ("3d_implicit", "x**2 + y**2 - 4"),
    "double cone":         ("3d_implicit", "x**2 + y**2 - z**2"),
    "lemniscate 3d":       ("3d_implicit", "(x**2+y**2+z**2)**2 - 2*(x**2-y**2)"),
    "cayley cubic":        ("3d_implicit", "x**2+y**2+z**2 - x*y*z - 1"),
    "steiner surface":     ("3d_implicit", "x**2*y**2 + y**2*z**2 + x**2*z**2 - x*y*z"),
    "heart 3d":            ("3d_implicit", "(x**2+9/4*y**2+z**2-1)**3 - x**2*z**3 - 9/80*y**2*z**3"),
    "klein bottle approx": ("3d_implicit", "(x**2+y**2+z**2+2*y-1)*((x**2+y**2+z**2-2*y-1)**2-8*z**2)+16*x*z*(x**2+y**2+z**2-2*y-1)"),
    "genus 2 surface":     ("3d_implicit", "2*y*(y**2-3*x**2)*(1-z**2)+(x**2+y**2)**2-(9*z**2-1)*(1-z**2)"),
}

def smart_parse(raw):
    raw = normalize_unicode(raw)   # normalize unicode math chars first
    s = raw.strip()
    sl = s.lower()

    for key, val in PRESET_MAP.items():
        if sl == key:
            return val

    # Parametric: x=..., y=...
    pm = re.match(r"x\s*=\s*(.+?)[,;]\s*y\s*=\s*(.+)", s, re.IGNORECASE)
    if pm:
        return ("parametric", (pm.group(1).strip(), pm.group(2).strip()))

    # Polar: r = f(theta)
    pol = re.match(r"r\s*=\s*(.+)", s, re.IGNORECASE)
    if pol:
        return ("polar", pol.group(1).strip())

    # 3D explicit: z = f(x,y)
    zm = re.match(r"z\s*=\s*(.+)", s, re.IGNORECASE)
    if zm:
        return ("3d_explicit", zm.group(1).strip())

    if "=" in s:
        parts = s.split("=", 1)
        lhs = _prep(parts[0].strip())
        rhs = _prep(parts[1].strip())
        combined = "(" + lhs + ") - (" + rhs + ")"

        has_z = _has_var(combined, "z")
        has_y = _has_var(combined, "y")
        has_x = _has_var(combined, "x")

        if has_z and has_x and has_y:
            return ("3d_implicit", combined)
        if has_x and has_y:
            return ("implicit", combined)
        if lhs.strip() in ("y", "f(x)"):
            return ("explicit", rhs)
        if has_x and not has_y:
            return ("explicit_eq", (lhs, rhs))
        if has_y:
            return ("implicit", combined)

    expr = _prep(s)
    return ("explicit", expr)


def _style_ax(ax):
    ax.set_facecolor("#0d0d0d")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

def _style_ax3d(ax):
    ax.set_facecolor("#0a0a0a")
    ax.tick_params(colors="white")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_xlabel("X", color="white", labelpad=8)
    ax.set_ylabel("Y", color="white", labelpad=8)
    ax.set_zlabel("Z", color="white", labelpad=8)


def _plot_explicit(ax, expr, title):
    x = np.linspace(-4 * np.pi, 4 * np.pi, 3000)
    try:
        y = safe_eval(expr, {"x": x})
        y = np.where(np.abs(y) > 100, np.nan, y)
        ax.plot(x, y, color="cyan", linewidth=2.5)
        ax.axhline(0, color="#555", lw=0.8)
        ax.axvline(0, color="#555", lw=0.8)
        ax.grid(True, color="#1a1a1a", lw=0.5)
        ax.set_title("y = " + title, color="cyan", fontsize=13)
        ax.set_xlabel("x", color="white")
        ax.set_ylabel("y", color="white")
        try:
            xs = sp.Symbol("x")
            es = sp.sympify(_prep(expr))
            yd = sp.lambdify(xs, sp.diff(es, xs), "numpy")(x)
            yd = np.where(np.abs(yd) > 100, np.nan, yd)
            ax.plot(x, yd, color="magenta", lw=1.5, ls="--",
                    alpha=0.7, label="y' (derivative)")
            ax.legend(facecolor="#111", labelcolor="white", fontsize=9)
        except Exception:
            pass
    except Exception as e:
        ax.text(0.5, 0.5, "Error: " + str(e), ha="center", va="center",
                color="red", fontsize=10, transform=ax.transAxes)


def _plot_explicit_eq(ax, lhs, rhs, title):
    x = np.linspace(-10, 10, 3000)
    try:
        yl = safe_eval(lhs, {"x": x})
        try:
            yr = float(rhs) * np.ones_like(x)
        except Exception:
            yr = safe_eval(rhs, {"x": x})
        yl = np.where(np.abs(yl) > 200, np.nan, yl)
        ax.plot(x, yl, color="cyan", lw=2.5, label="LHS: " + lhs)
        ax.plot(x, yr, color="gold", lw=2, ls="--", label="RHS: " + rhs)
        ax.fill_between(x, yl, yr, alpha=0.08, color="cyan")
        ax.axhline(0, color="#555", lw=0.8)
        ax.axvline(0, color="#555", lw=0.8)
        ax.grid(True, color="#1a1a1a", lw=0.5)
        ax.legend(facecolor="#111", labelcolor="white", fontsize=9)
        ax.set_title(title, color="cyan", fontsize=13)
        ax.set_xlabel("x", color="white")
        ax.set_ylabel("y", color="white")
    except Exception as e:
        ax.text(0.5, 0.5, "Error: " + str(e), ha="center", va="center",
                color="red", fontsize=10, transform=ax.transAxes)


def _plot_implicit(ax, expr, title):
    x_r = np.linspace(-10, 10, 800)
    y_r = np.linspace(-10, 10, 800)
    X, Y = np.meshgrid(x_r, y_r)
    try:
        Z = safe_eval(expr, {"x": X, "y": Y})
        ax.contour(X, Y, Z, levels=[0], colors=["cyan"], linewidths=2.5)
        ax.contourf(X, Y, Z, levels=20, cmap="inferno", alpha=0.25)
        ax.set_aspect("equal")
        ax.axhline(0, color="#555", lw=0.8)
        ax.axvline(0, color="#555", lw=0.8)
        ax.grid(True, color="#1a1a1a", lw=0.5)
        ax.set_title(title, color="cyan", fontsize=13)
        ax.set_xlabel("x", color="white")
        ax.set_ylabel("y", color="white")
    except Exception as e:
        ax.text(0.5, 0.5, "Implicit error:\n" + str(e), ha="center", va="center",
                color="red", fontsize=10, transform=ax.transAxes)


def _plot_parametric(ax, x_expr, y_expr, title):
    t = np.linspace(0, 4 * np.pi, 4000)
    try:
        xv = safe_eval(x_expr, {"t": t})
        yv = safe_eval(y_expr, {"t": t})
        from matplotlib.collections import LineCollection
        pts  = np.array([xv, yv]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc   = LineCollection(segs, cmap="plasma", linewidth=2.5, alpha=0.9)
        lc.set_array(t)
        ax.add_collection(lc)
        ax.set_xlim(np.nanmin(xv) * 1.1, np.nanmax(xv) * 1.1)
        ax.set_ylim(np.nanmin(yv) * 1.1, np.nanmax(yv) * 1.1)
        ax.set_aspect("equal")
        ax.axhline(0, color="#555", lw=0.8)
        ax.axvline(0, color="#555", lw=0.8)
        ax.set_title("Parametric: " + title, color="cyan", fontsize=12)
        ax.set_xlabel("x", color="white")
        ax.set_ylabel("y", color="white")
    except Exception as e:
        ax.text(0.5, 0.5, "Parametric error:\n" + str(e), ha="center", va="center",
                color="red", fontsize=10, transform=ax.transAxes)


def _plot_polar(ax, expr, title):
    theta = np.linspace(0, 4 * np.pi, 4000)
    try:
        r = np.abs(safe_eval(expr, {"theta": theta, "t": theta}))
        xv = r * np.cos(theta)
        yv = r * np.sin(theta)
        from matplotlib.collections import LineCollection
        pts  = np.array([xv, yv]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc   = LineCollection(segs, cmap="plasma", linewidth=2.5, alpha=0.9)
        lc.set_array(theta)
        ax.add_collection(lc)
        ax.set_xlim(np.nanmin(xv) * 1.2, np.nanmax(xv) * 1.2)
        ax.set_ylim(np.nanmin(yv) * 1.2, np.nanmax(yv) * 1.2)
        ax.set_aspect("equal")
        ax.axhline(0, color="#555", lw=0.8)
        ax.axvline(0, color="#555", lw=0.8)
        ax.set_title("Polar: r = " + title, color="cyan", fontsize=12)
        ax.set_xlabel("x", color="white")
        ax.set_ylabel("y", color="white")
    except Exception as e:
        ax.text(0.5, 0.5, "Polar error:\n" + str(e), ha="center", va="center",
                color="red", fontsize=10, transform=ax.transAxes)


def _plot_3d_explicit(fig, expr, title):
    ax = fig.add_subplot(111, projection="3d")
    _style_ax3d(ax)
    x = np.linspace(-5, 5, 120)
    y = np.linspace(-5, 5, 120)
    X, Y = np.meshgrid(x, y)
    try:
        Z = safe_eval(expr, {"x": X, "y": Y})
        Z = np.where(np.abs(Z) > 1e4, np.nan, Z)
        surf = ax.plot_surface(X, Y, Z, cmap="plasma", alpha=0.88,
                               linewidth=0, antialiased=True)
        fig.colorbar(surf, ax=ax, shrink=0.4, pad=0.1)
        ax.set_title("z = " + title, color="cyan", fontsize=13, pad=12)
    except Exception as e:
        ax.text2D(0.5, 0.5, "3D Error:\n" + str(e), ha="center", va="center",
                  color="red", fontsize=11, transform=ax.transAxes)
    return fig


def _plot_3d_implicit(fig, expr, title):
    """
    Plot f(x,y,z) = 0 as a proper 3D surface using marching cubes.
    Vertex coordinates are correctly mapped from voxel indices to world space.
    """
    ax = fig.add_subplot(111, projection="3d")
    _style_ax3d(ax)

    lim = 6
    n   = 90          # grid resolution — higher = smoother surface
    lin = np.linspace(-lim, lim, n)
    step = lin[1] - lin[0]

    # Build 3D grid — meshgrid indexing='ij' gives F[ix, iy, iz]
    gx, gy, gz = np.meshgrid(lin, lin, lin, indexing="ij")

    try:
        F = safe_eval(expr, {"x": gx, "y": gy, "z": gz})
        F = np.nan_to_num(F.astype(np.float64), nan=1e6, posinf=1e6, neginf=-1e6)

        from skimage.measure import marching_cubes
        # spacing maps voxel indices to world coordinates
        verts, faces, normals, _ = marching_cubes(F, level=0.0,
                                                   spacing=(step, step, step))
        # verts are in voxel-space starting at 0 — shift to world coords
        verts[:, 0] += lin[0]
        verts[:, 1] += lin[0]
        verts[:, 2] += lin[0]

        # Color by z-height for visual depth
        z_vals = verts[faces, 2].mean(axis=1)
        z_norm = (z_vals - z_vals.min()) / (z_vals.max() - z_vals.min() + 1e-9)

        ax.plot_trisurf(
            verts[:, 0], verts[:, 1], faces, verts[:, 2],
            cmap="plasma", alpha=0.90, linewidth=0,
            antialiased=True, shade=True
        )

        ax.set_title(title, color="cyan", fontsize=12, pad=12)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

        # Add subtle wireframe grid lines
        ax.xaxis.pane.set_edgecolor("#222")
        ax.yaxis.pane.set_edgecolor("#222")
        ax.zaxis.pane.set_edgecolor("#222")

    except Exception as e:
        ax.text2D(0.5, 0.5, "3D Implicit Error:\n" + str(e),
                  ha="center", va="center", color="red",
                  fontsize=10, transform=ax.transAxes)
    return fig


def _get_real_world(expr):
    e = expr.lower()
    mapping = {
        # 3D shapes
        "ellipsoid":    "Planets, eggs, rugby balls, red blood cells, atomic nuclei",
        "sphere":       "Planets, soap bubbles, ball bearings, atomic orbitals",
        "torus":        "Donuts, tokamak fusion reactors, knot theory, life rings",
        "hyperboloid":  "Cooling towers, radio telescopes, Shukhov tower architecture",
        "cylinder":     "Pipes, columns, cans, rollers, blood vessels",
        "cone 3d":      "Ice cream cones, volcanic mountains, radar beams",
        "paraboloid":   "Satellite dishes, telescope mirrors, car headlights",
        "heart 3d":     "Cardiac geometry, Valentine mathematics",
        # 2D curves
        "cardioid":     "Cardioid microphones, caustics in coffee cups, epicycloid",
        "lemniscate":   "Infinity symbol, Bernoulli lemniscate, figure-8 orbits",
        "cycloid":      "Brachistochrone (fastest descent), gear tooth profiles",
        "catenary":     "Hanging cables, suspension bridges, arch design (Gateway Arch)",
        "tractrix":     "Pseudosphere geometry, tractrix curves in optics",
        "folium":       "Descartes folium, algebraic geometry, cubic curves",
        "cassini":      "Cassini ovals, antenna radiation patterns",
        "spiral":       "Galaxy arms, DNA double helix, springs, nautilus shell",
        "golden spiral":"Nautilus shell, sunflower seeds, galaxy arms, Fibonacci",
        "epicycloid":   "Gear tooth design, planetary gear systems, Spirograph",
        "hypotrochoid": "Spirograph patterns, gear mechanisms, rose windows",
        # Functions
        "sin":          "Sound waves, AC electricity, pendulum motion, ocean tides",
        "cos":          "Wave interference, signal processing, circular motion",
        "tan":          "Slope calculation, optics (Snell law), navigation bearings",
        "exp":          "Population growth, radioactive decay, compound interest, RC circuits",
        "log":          "Richter scale, pH, decibels, information entropy, Weber-Fechner law",
        "sinh":         "Catenary curves (hanging cables), special relativity, hyperbolic geometry",
        "cosh":         "Catenary, power line sag, arch design",
        "tanh":         "Neural network activation, logistic growth, special relativity velocity",
        "sigmoid":      "Neural networks, population growth, dose-response curves",
        "sinc":         "Signal processing, diffraction patterns, Fourier analysis",
        "gaussian":     "Normal distribution, quantum mechanics, heat diffusion, optics",
        "mexican hat":  "Wavelet analysis, edge detection in image processing",
        "x**2":         "Projectile motion, parabolic mirrors, satellite dishes, lenses",
        "x**3":         "Volume calculations, cubic Bezier curves, fluid dynamics",
        "x**4":         "Beam bending, quartic potential wells in physics",
        "sqrt":         "Distance formula, RMS voltage, standard deviation, pendulum period",
        "1/x":          "Inverse square law (gravity, light), Boyle gas law, harmonic series",
        "abs":          "Rectifier circuits, absolute value geometry, piecewise functions",
        "floor":        "Digital signal quantization, staircase functions, modular arithmetic",
        # Physics equations
        "electric":     "Electric field lines, potential surfaces, capacitor design",
        "gravity":      "Gravitational potential wells, orbital mechanics, black holes",
        "wave interf":  "Interference patterns, diffraction gratings, holography",
        "peaks":        "Optimization landscapes, neural network loss surfaces",
        "helicoid":     "Minimal surfaces, soap films, DNA structure, screw threads",
        "enneper":      "Minimal surface theory, differential geometry",
        "monkey":       "Catastrophe theory, three-body problem, structural mechanics",
        # Polar
        "archimedean":  "Spiral staircases, watch springs, vinyl record grooves",
        "fermat":       "Sunflower seed arrangement, phyllotaxis in plants",
        "logarithmic":  "Nautilus shell, galaxy spiral arms, hurricane shape",
        "limacon":      "Cardioid microphone patterns, Pascal limacons",
        "rose":         "Antenna radiation patterns, quantum orbital shapes",
        "lissajous":    "Oscilloscope patterns, AC circuit analysis, music visualization",
        # Implicit
        "hyperbola":    "Satellite navigation (LORAN), cooling tower cross-sections",
        "cassini oval": "Antenna radiation patterns, Cassini spacecraft orbit",
        "bernoulli":    "Lemniscate of Bernoulli, figure-8 orbits, infinity symbol",
        "heart":        "Cardioid geometry, cardiac mathematics",
        "klein":        "Topology, non-orientable surfaces, theoretical physics",
    }
    for key, val in mapping.items():
        if key in e:
            return val
    return "Mathematical modeling and engineering"


def _add_info_panel(ax, eq_type, expr_input, eq_data):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.97, "Expression Analysis", ha="center", va="top",
            color="gold", fontsize=12, fontweight="bold", transform=ax.transAxes)

    info = [
        ("Expression Type", eq_type.upper().replace("_", " ")),
        ("Input", expr_input[:35]),
    ]

    if eq_type == "explicit":
        try:
            xs = sp.Symbol("x")
            es = sp.sympify(_prep(str(eq_data)))
            info += [
                ("Derivative", str(sp.simplify(sp.diff(es, xs)))[:45]),
                ("Integral",   str(sp.simplify(sp.integrate(es, xs)))[:45]),
                ("Zeros",      str(sp.solve(es, xs)[:3])[:45]),
            ]
        except Exception:
            info += [("Analysis", "See plot")]
    elif eq_type == "parametric":
        info += [("x(t)", str(eq_data[0])[:35]),
                 ("y(t)", str(eq_data[1])[:35]),
                 ("Parameter", "t in [0, 4pi]")]
    elif eq_type == "polar":
        info += [("r(theta)", str(eq_data)[:35]),
                 ("Parameter", "theta in [0, 4pi]")]
    elif eq_type == "implicit":
        info += [("Form", "f(x,y) = 0"), ("Method", "Contour plot")]
    elif eq_type == "3d_explicit":
        info += [("Surface", "z = " + str(eq_data)[:30]),
                 ("Domain", "x,y in [-5, 5]")]
    elif eq_type == "3d_implicit":
        info += [("Form", "f(x,y,z) = 0"),
                 ("Method", "Isosurface"),
                 ("Domain", "x,y,z in [-6, 6]")]
        used = {k: v for k, v in CONST_DEFAULTS.items()
                if re.search(r"\b" + k + r"\b", expr_input)}
        if used:
            info.append(("Constants", ", ".join(k + "=" + str(v) for k, v in used.items())))

    info.append(("Real-Life Use", _get_real_world(expr_input)))

    y = 0.88
    for label, value in info:
        ax.text(0.05, y, label + ":", color="#aaaaaa", fontsize=9,
                transform=ax.transAxes)
        y -= 0.055
        words = str(value).replace("**", "^").split()
        line, lines_out = [], []
        for w in words:
            line.append(w)
            if len(" ".join(line)) > 38:
                lines_out.append(" ".join(line)); line = []
        if line:
            lines_out.append(" ".join(line))
        for ln in lines_out[:2]:
            ax.text(0.08, y, ln, color="cyan", fontsize=8.5,
                    transform=ax.transAxes)
            y -= 0.045
        y -= 0.03
        if y < 0.04:
            break


def plot_equation(expr_input):
    expr_input = normalize_unicode(expr_input)
    eq_type, eq_data = smart_parse(expr_input)
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("#0a0a0a")

    if eq_type == "3d_explicit":
        return _plot_3d_explicit(fig, eq_data, expr_input)
    if eq_type == "3d_implicit":
        return _plot_3d_implicit(fig, eq_data, expr_input)

    gs      = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[2, 1], wspace=0.3)
    ax_plot = fig.add_subplot(gs[0])
    ax_info = fig.add_subplot(gs[1])
    _style_ax(ax_plot)
    _style_ax(ax_info)
    ax_info.axis("off")

    if eq_type == "explicit":
        _plot_explicit(ax_plot, eq_data, expr_input)
    elif eq_type == "explicit_eq":
        _plot_explicit_eq(ax_plot, eq_data[0], eq_data[1], expr_input)
    elif eq_type == "implicit":
        _plot_implicit(ax_plot, eq_data, expr_input)
    elif eq_type == "parametric":
        _plot_parametric(ax_plot, eq_data[0], eq_data[1], expr_input)
    elif eq_type == "polar":
        _plot_polar(ax_plot, eq_data, expr_input)

    _add_info_panel(ax_info, eq_type, expr_input, eq_data)
    return fig


def get_equation_properties(expr_input):
    expr_input = normalize_unicode(expr_input)
    eq_type, eq_data = smart_parse(expr_input)
    props = {"type": eq_type, "input": expr_input}
    if eq_type == "explicit":
        try:
            xs = sp.Symbol("x")
            es = sp.sympify(_prep(str(eq_data)))
            props["derivative"] = str(sp.diff(es, xs))
            props["integral"]   = str(sp.integrate(es, xs))
            props["roots"]      = [str(r) for r in sp.solve(es, xs)[:5]]
        except Exception:
            pass
    elif eq_type in ("3d_implicit", "implicit"):
        used = {k: v for k, v in CONST_DEFAULTS.items()
                if re.search(r"\b" + k + r"\b", expr_input)}
        if used:
            props["constants"] = used
    return props

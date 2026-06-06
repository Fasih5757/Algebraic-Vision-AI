# ============================================================
#  AlgebraicVision AI  -  app.py  (Full Rewrite)
#  All text strictly in English
# ============================================================
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tempfile
import os
from PIL import Image

from modules.equation_parser import plot_equation, get_equation_properties
from modules.eda_analyzer     import full_eda_report
from modules.rl_agent         import get_agent, extract_features
from modules.yolo_detector    import detect_and_analyze, get_eda_report
from modules.ai_engine        import (explain_pattern, analyze_equation,
                                       analyze_image_patterns,
                                       explain_eda_results,
                                       explain_reallife, chat_with_ai)

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="AlgebraicVision AI",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stApp{background:#0a0a0a;color:#e0e0e0}
  .main-title{text-align:center;font-size:2.8rem;font-weight:900;
    background:linear-gradient(90deg,#00ffff,#ffd700,#ff00ff);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;padding:.6rem 0}
  .subtitle{text-align:center;color:#888;font-size:1.05rem;margin-bottom:1.5rem}
  .stButton>button{background:linear-gradient(90deg,#00ffff18,#ff00ff18);
    color:white;border:1px solid #00ffff44;border-radius:8px;width:100%;padding:.45rem}
  .stButton>button:hover{border-color:#00ffff;color:#00ffff}
  section[data-testid="stSidebar"]{background:#0d0d0d;border-right:1px solid #1a1a1a}
  .stTabs [data-baseweb="tab"]{color:#888}
  .stTabs [aria-selected="true"]{color:#00ffff;border-bottom:2px solid #00ffff}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🔭 AlgebraicVision AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter any equation — see its universe. Upload any image — reveal its mathematics.</div>', unsafe_allow_html=True)
st.divider()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧮 Navigation")
    module = st.radio("Select Module:", [
        "📐 Equation Visualizer",
        "🖼️ Image Pattern Analyzer",
        "🌀 Parametric & Polar Curves",
        "❄️ Fractal Explorer",
        "🤖 RL Pattern Agent",
        "💬 AI Assistant",
    ])
    st.divider()
    st.markdown("### 🧠 RL Agent Status")
    try:
        agent = get_agent()
        stats = agent.get_stats()
        st.metric("Episodes Learned", stats["total_episodes"])
        st.metric("Exploration Rate", f"{stats['epsilon']:.3f}")
        st.metric("States in Memory",  stats["states_learned"])
    except Exception as _e:
        st.caption(f"RL agent: {_e}")
    st.divider()
    st.markdown("### ℹ️ About")
    st.markdown(
        "AlgebraicVision AI reveals the mathematical patterns hidden in "
        "equations, images, and nature — from Fourier waves to fractal "
        "dimensions, golden ratios to symmetry groups."
    )

# ══════════════════════════════════════════════════════════
# MODULE 1 — EQUATION VISUALIZER
# ══════════════════════════════════════════════════════════
if module == "📐 Equation Visualizer":
    st.markdown("## 📐 Equation Visualizer")
    st.markdown(
        "Enter **any** mathematical expression — algebraic, geometric, "
        "trigonometric, parametric, polar, or 3-D surface. "
        "The AI plots it and explains its real-world significance."
    )

    col_input, col_ex = st.columns([2, 1])

    with col_ex:
        st.markdown("**Quick Examples — click to load:**")
        examples = {
            # 2D Explicit
            "Sine Wave":         "sin(x)",
            "Parabola":          "x**2",
            "Damped Wave":       "exp(-x/5)*sin(x)",
            "Sigmoid":           "1/(1+exp(-x))",
            "Gaussian Bell":     "exp(-x**2/2)",
            "Sinc":              "sin(x)/x",
            "Catenary":          "catenary",
            "Cubic":             "cubic",
            # 2D Implicit
            "Circle":            "x**2 + y**2 = 25",
            "Ellipse":           "x**2/9 + y**2/4 = 1",
            "Hyperbola":         "x**2/4 - y**2/9 = 1",
            "Folium Descartes":  "folium descartes",
            "Cassini Oval":      "cassini oval",
            # Parametric
            "Heart":             "heart",
            "Butterfly":         "butterfly",
            "Lissajous":         "lissajous",
            "Epicycloid":        "epicycloid",
            "Spirograph":        "spirograph",
            "Trefoil":           "trefoil",
            "Cycloid":           "cycloid",
            "Golden Spiral":     "golden spiral",
            # Polar
            "Cardioid":          "cardioid",
            "Rose 3":            "rose 3",
            "Lemniscate":        "lemniscate",
            "Fermat Spiral":     "fermat spiral",
            # 3D Explicit
            "Saddle":            "saddle surface",
            "Mexican Hat":       "mexican hat",
            "Ripple":            "ripple",
            "Gravity Well":      "gravity well",
            "Wave Interference": "wave interference",
            "Monkey Saddle":     "monkey saddle",
            # 3D Implicit
            "Sphere":            "sphere",
            "Ellipsoid":         "ellipsoid",
            "Torus":             "torus",
            "Hyperboloid":       "hyperboloid",
            "Heart 3D":          "heart 3d",
            "Cylinder":          "cylinder",
        }
        c1, c2 = st.columns(2)
        for i, (name, eq) in enumerate(examples.items()):
            if (c1 if i % 2 == 0 else c2).button(name, key=f"ex_{i}"):
                st.session_state["eq_input"] = eq

    with col_input:
        default_eq = st.session_state.get("eq_input", "sin(x)")
        equation = st.text_input(
            "Enter expression:",
            value=default_eq,
            placeholder="e.g.  x**2   or   r = cos(3*theta)   or   x=sin(3*t), y=sin(2*t)",
        )
        go = st.button("🔍 Visualize & Analyze", type="primary")

    if go and equation.strip():
        with st.spinner("Generating visualization..."):
            try:
                fig = plot_equation(equation)
                st.pyplot(fig)
                plt.close()
            except Exception as e:
                st.error(f"Plot error: {e}")

        st.markdown("---")
        t1, t2, t3 = st.tabs(["📊 Math Properties", "🌍 Real-Life Applications", "🤖 AI Explanation"])

        with t1:
            try:
                props = get_equation_properties(equation)
                c1, c2 = st.columns(2)
                c1.metric("Type",       props.get("type", "N/A").upper())
                c2.metric("Input",      props.get("input", "")[:30])
                if "derivative" in props:
                    st.markdown(f"**Derivative:** `{props['derivative']}`")
                if "integral" in props:
                    st.markdown(f"**Integral:** `{props['integral']}`")
                if "roots" in props:
                    st.markdown(f"**Roots:** `{props['roots']}`")
            except Exception as e:
                st.warning(f"Properties error: {e}")

        with t2:
            try:
                with st.spinner("AI generating real-life explanation..."):
                    rl_text = explain_reallife(equation, "mathematical expression",
                                               "physics, engineering, nature")
                st.markdown(rl_text)
            except Exception as e:
                st.error(f"AI error: {e}")

        with t3:
            try:
                with st.spinner("AI analyzing equation..."):
                    ai_text = analyze_equation(equation)
                st.markdown(ai_text)
            except Exception as e:
                st.error(f"AI error: {e}")

# ══════════════════════════════════════════════════════════
# MODULE 2 — IMAGE PATTERN ANALYZER
# ══════════════════════════════════════════════════════════
elif module == "🖼️ Image Pattern Analyzer":
    st.markdown("## 🖼️ Image Pattern Analyzer")
    st.markdown(
        "Upload any image. The AI runs **YOLOv8** object detection, "
        "full **EDA**, symmetry scoring, golden ratio overlay, "
        "fractal dimension, and overlays mathematical patterns."
    )

    uploaded = st.file_uploader("Upload image:", type=["jpg","jpeg","png","bmp","webp"])

    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
            f.write(uploaded.getvalue())
            tmp_path = f.name

        tab_yolo, tab_eda, tab_sym, tab_ai = st.tabs([
            "🎯 YOLO + Pattern Overlay",
            "📊 Full EDA Report",
            "🪞 Symmetry & Golden Ratio",
            "🤖 AI Interpretation",
        ])

        # ── YOLO tab ──────────────────────────────────
        with tab_yolo:
            st.markdown("### YOLOv8 Object Detection + Mathematical Pattern Overlay")
            with st.spinner("Running YOLOv8 and pattern analysis..."):
                try:
                    fig_yolo, det_objects, math_patterns, confs = detect_and_analyze(tmp_path)
                    st.pyplot(fig_yolo)
                    plt.close()

                    # RL agent learns
                    try:
                        agent = get_agent()
                        features = extract_features(tmp_path)
                        top_patterns = agent.select_top_k(features, k=3)
                        agent.learn_from_image(tmp_path)
                        st.info(f"**RL Agent recommends highlighting:** {', '.join(top_patterns)}")
                    except Exception as _re:
                        pass

                    if det_objects:
                        st.success(f"Detected {len(det_objects)} object(s)")
                        for obj, pat, conf in zip(det_objects, math_patterns, confs):
                            with st.expander(f"**{obj.upper()}** — {pat[0]}  ({conf:.0%} confidence)"):
                                st.markdown(f"**Pattern:** {pat[0]}")
                                st.markdown(f"**Detail:** {pat[1]}")
                                with st.spinner("Loading AI explanation..."):
                                    try:
                                        exp = explain_pattern(pat[0], pat[1])
                                        st.markdown(exp)
                                    except Exception as _ae:
                                        st.caption(f"AI unavailable: {_ae}")
                    else:
                        st.warning("No objects detected. Try a clearer image.")

                    # RL feedback buttons
                    st.markdown("---")
                    st.markdown("**Help the RL agent learn — was this analysis useful?**")
                    fb1, fb2, fb3 = st.columns(3)
                    if fb1.button("👍 Helpful"):
                        try: agent.give_reward(1.0); st.success("RL agent rewarded +1.0")
                        except: pass
                    if fb2.button("😐 Neutral"):
                        try: agent.give_reward(0.0); st.info("Neutral feedback sent")
                        except: pass
                    if fb3.button("👎 Not helpful"):
                        try: agent.give_reward(-1.0); st.warning("RL agent penalized -1.0")
                        except: pass

                except Exception as e:
                    st.error(f"YOLO analysis error: {e}")

        # ── EDA tab ───────────────────────────────────
        with tab_eda:
            st.markdown("### Full Exploratory Data Analysis")
            with st.spinner("Running complete EDA pipeline..."):
                try:
                    fig_eda, stats = full_eda_report(tmp_path)
                    st.pyplot(fig_eda)
                    plt.close()

                    st.markdown("---")
                    pat_scores = stats.get("pattern_scores", {})
                    if pat_scores:
                        st.markdown("#### Pattern Confidence Scores")
                        cols_ps = st.columns(len(pat_scores))
                        for col_ps, (pname, pscore) in zip(cols_ps, pat_scores.items()):
                            col_ps.metric(pname, f"{pscore:.1f}%")

                    st.markdown("#### AI Interpretation")
                    with st.spinner("AI analyzing statistics..."):
                        try:
                            eda_exp = explain_eda_results(stats)
                            st.markdown(eda_exp)
                        except Exception as _ae:
                            st.caption(f"AI unavailable: {_ae}")
                except Exception as e:
                    st.error(f"EDA error: {e}")

        # ── Symmetry tab ──────────────────────────────
        with tab_sym:
            st.markdown("### Symmetry Analysis & Golden Ratio Detection")
            img_cv  = cv2.imread(tmp_path)
            img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            gray    = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            h, w    = gray.shape
            PHI     = 1.6180339887

            mid_x = w // 2
            left  = gray[:, :mid_x]
            right = cv2.resize(cv2.flip(gray[:, mid_x:], 1), (left.shape[1], left.shape[0]))
            h_sym = 100 - (np.mean(cv2.absdiff(left, right)) / 255 * 100)
            mid_y = h // 2
            top   = gray[:mid_y, :]
            bot   = cv2.resize(cv2.flip(gray[mid_y:, :], 0), (top.shape[1], top.shape[0]))
            v_sym = 100 - (np.mean(cv2.absdiff(top, bot)) / 255 * 100)

            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Horizontal Symmetry", f"{h_sym:.1f}%")
            mc2.metric("Vertical Symmetry",   f"{v_sym:.1f}%")
            mc3.metric("Golden Point X",       f"{w/PHI:.0f} px")
            mc4.metric("Aspect vs φ",          f"Δ {abs(w/h - PHI):.3f}")

            fig_s, axes_s = plt.subplots(1, 3, figsize=(16, 5))
            fig_s.patch.set_facecolor('#0a0a0a')
            for ax in axes_s:
                ax.set_facecolor('#0d0d0d')
                ax.axis('off')

            axes_s[0].imshow(img_rgb)
            axes_s[0].axvline(x=w/PHI,   color='gold', lw=2, ls='--', label='phi vertical')
            axes_s[0].axvline(x=w-w/PHI, color='gold', lw=2, ls='--')
            axes_s[0].axhline(y=h/PHI,   color='cyan', lw=2, ls='--', label='phi horizontal')
            axes_s[0].axhline(y=h-h/PHI, color='cyan', lw=2, ls='--')
            theta_sp = np.linspace(0, 4*np.pi, 1000)
            axes_s[0].plot(w/2 + theta_sp*(w/40)*np.cos(theta_sp),
                           h/2 + theta_sp*(h/30)*np.sin(theta_sp),
                           color='lime', lw=1.5, alpha=0.7, label='Golden spiral')
            axes_s[0].legend(facecolor='#111', labelcolor='white', fontsize=8)
            axes_s[0].set_title('Golden Ratio Overlay', color='gold', fontsize=11)

            axes_s[1].imshow(cv2.flip(img_rgb, 1))
            axes_s[1].set_title(f'Horizontal Mirror  ({h_sym:.1f}%)', color='cyan', fontsize=11)

            axes_s[2].imshow(cv2.flip(img_rgb, 0))
            axes_s[2].set_title(f'Vertical Mirror  ({v_sym:.1f}%)', color='lime', fontsize=11)

            plt.tight_layout()
            st.pyplot(fig_s)
            plt.close()

        # ── AI tab ────────────────────────────────────
        with tab_ai:
            st.markdown("### AI Mathematical Interpretation")
            img_cv2  = cv2.imread(tmp_path)
            gray2    = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2GRAY)
            h2, w2   = gray2.shape
            mid2     = w2 // 2
            l2       = gray2[:, :mid2]
            r2       = cv2.resize(cv2.flip(gray2[:, mid2:], 1), (l2.shape[1], l2.shape[0]))
            sym_s2   = 100 - (np.mean(cv2.absdiff(l2, r2)) / 255 * 100)

            with st.spinner("AI analyzing image patterns..."):
                try:
                    ai_img = analyze_image_patterns(
                        ["uploaded image"],
                        sym_s2,
                        f"Golden point at {w2/1.618:.0f}x{h2/1.618:.0f}px, aspect ratio {w2/h2:.3f}"
                    )
                    st.markdown(ai_img)
                except Exception as e:
                    st.error(f"AI error: {e}")

            st.markdown("---")
            st.markdown("#### Ask AI about this image")
            user_q = st.text_input("Your question about this image:", placeholder="What mathematical patterns does this image follow?")
            if st.button("Ask AI") and user_q:
                with st.spinner("AI thinking..."):
                    try:
                        context = f"Image symmetry: {sym_s2:.1f}%, aspect ratio: {w2/h2:.3f}, golden ratio diff: {abs(w2/h2-1.618):.3f}"
                        ans = chat_with_ai(user_q, context)
                        st.markdown(ans)
                    except Exception as e:
                        st.error(f"AI error: {e}")

        os.unlink(tmp_path)

# ══════════════════════════════════════════════════════════
# MODULE 3 — PARAMETRIC & POLAR CURVES
# ══════════════════════════════════════════════════════════
elif module == "🌀 Parametric & Polar Curves":
    st.markdown("## 🌀 Parametric & Polar Curves")
    st.markdown("Mathematical equations that trace beautiful paths through space.")

    curve_mode = st.radio("Mode:", ["Preset Curves", "Custom Parametric", "Custom Polar"], horizontal=True)
    t = np.linspace(0, 2 * np.pi, 3000)

    if curve_mode == "Preset Curves":
        preset_curves = {
            "Ellipse":       (2*np.cos(t), np.sin(t), "cyan"),
            "Lissajous 3:2": (np.sin(3*t), np.sin(2*t), "yellow"),
            "Rose (n=5)":    (np.cos(5*t)*np.cos(t), np.cos(5*t)*np.sin(t), "magenta"),
            "Infinity":      (np.cos(t), np.sin(2*t)/2, "lime"),
            "Butterfly":     (np.sin(t)*(np.e**np.cos(t)-2*np.cos(4*t)),
                              np.cos(t)*(np.e**np.cos(t)-2*np.cos(4*t)), "orange"),
            "Epicycloid":    (4*np.cos(t)-np.cos(4*t), 4*np.sin(t)-np.sin(4*t), "gold"),
            "Hypotrochoid":  (2*np.cos(t)+0.5*np.cos(2*t), 2*np.sin(t)-0.5*np.sin(2*t), "pink"),
            "Spirograph":    (8*np.cos(t)-3*np.cos(8/3*t), 8*np.sin(t)-3*np.sin(8/3*t), "aqua"),
        }
        selected = st.multiselect("Select curves:", list(preset_curves.keys()),
                                   default=["Ellipse", "Rose (n=5)", "Butterfly"])
        if selected:
            n_sel  = len(selected)
            cols_c = min(n_sel, 3)
            rows_c = (n_sel + cols_c - 1) // cols_c
            fig_p, axes_p = plt.subplots(rows_c, cols_c, figsize=(5*cols_c, 5*rows_c))
            fig_p.patch.set_facecolor('#0a0a0a')
            flat = axes_p.flatten() if hasattr(axes_p, 'flatten') else [axes_p]
            for ax, name in zip(flat, selected):
                xc, yc, cc = preset_curves[name]
                ax.set_facecolor('#0d0d0d')
                ax.plot(xc, yc, color=cc, linewidth=2)
                ax.set_title(name, color=cc, fontsize=11, fontweight='bold')
                ax.set_aspect('equal')
                ax.axis('off')
            for ax in flat[len(selected):]:
                ax.set_visible(False)
            plt.tight_layout()
            st.pyplot(fig_p)
            plt.close()

            if st.button("🤖 AI Explain Selected Curves"):
                for name in selected:
                    with st.spinner(f"Explaining {name}..."):
                        try:
                            exp_c = explain_pattern(name, f"Parametric curve: {name}")
                            st.markdown(f"**{name}:** {exp_c}")
                        except Exception as e:
                            st.error(f"AI error: {e}")

    elif curve_mode == "Custom Parametric":
        st.markdown("Use variable `t` (range 0 to 2π):")
        cx1, cx2 = st.columns(2)
        x_eq = cx1.text_input("x(t) =", value="sin(3*t)")
        y_eq = cx2.text_input("y(t) =", value="sin(2*t)")
        t_max = st.slider("t max:", 1.0, 20.0, 6.28)
        if st.button("Plot Parametric Curve"):
            t_c = np.linspace(0, t_max, 4000)
            ns_c = {"sin":np.sin,"cos":np.cos,"tan":np.tan,"exp":np.exp,
                    "sqrt":np.sqrt,"pi":np.pi,"e":np.e,"t":t_c,
                    "phi":1.618,"log":np.log,"abs":np.abs}
            try:
                xv = eval(x_eq.replace("^","**"), {"__builtins__":{}}, ns_c)
                yv = eval(y_eq.replace("^","**"), {"__builtins__":{}}, ns_c)
                from matplotlib.collections import LineCollection
                fig_cp, ax_cp = plt.subplots(figsize=(8, 8))
                fig_cp.patch.set_facecolor('#0a0a0a')
                ax_cp.set_facecolor('#0d0d0d')
                pts  = np.array([xv, yv]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                lc   = LineCollection(segs, cmap='plasma', linewidth=2)
                lc.set_array(t_c[:-1])
                ax_cp.add_collection(lc)
                ax_cp.autoscale()
                ax_cp.set_aspect('equal')
                ax_cp.axis('off')
                ax_cp.set_title(f"x={x_eq},  y={y_eq}", color='cyan', fontsize=11)
                st.pyplot(fig_cp)
                plt.close()
            except Exception as e:
                st.error(f"Error: {e}")

    else:
        st.markdown("Use variable `t` for θ:")
        r_eq  = st.text_input("r(θ) =", value="cos(3*t)")
        t_max_p = st.slider("θ max:", 1.0, 20.0, 12.57)
        if st.button("Plot Polar Curve"):
            t_p = np.linspace(0, t_max_p, 4000)
            ns_p = {"sin":np.sin,"cos":np.cos,"tan":np.tan,"exp":np.exp,
                    "sqrt":np.sqrt,"pi":np.pi,"e":np.e,"t":t_p,
                    "phi":1.618,"log":np.log,"abs":np.abs}
            try:
                rv   = np.abs(eval(r_eq.replace("^","**"), {"__builtins__":{}}, ns_p))
                xp   = rv * np.cos(t_p)
                yp   = rv * np.sin(t_p)
                from matplotlib.collections import LineCollection
                fig_pp, ax_pp = plt.subplots(figsize=(8, 8))
                fig_pp.patch.set_facecolor('#0a0a0a')
                ax_pp.set_facecolor('#0d0d0d')
                pts_p  = np.array([xp, yp]).T.reshape(-1, 1, 2)
                segs_p = np.concatenate([pts_p[:-1], pts_p[1:]], axis=1)
                lc_p   = LineCollection(segs_p, cmap='inferno', linewidth=2)
                lc_p.set_array(t_p[:-1])
                ax_pp.add_collection(lc_p)
                ax_pp.autoscale()
                ax_pp.set_aspect('equal')
                ax_pp.axis('off')
                ax_pp.set_title(f"r = {r_eq}  (polar)", color='gold', fontsize=11)
                st.pyplot(fig_pp)
                plt.close()
            except Exception as e:
                st.error(f"Error: {e}")

# ══════════════════════════════════════════════════════════
# MODULE 5 — FRACTAL EXPLORER
# ══════════════════════════════════════════════════════════
elif module == "❄️ Fractal Explorer":
    st.markdown("## ❄️ Fractal Explorer")
    st.markdown(
        "Fractals are infinite self-similar structures from simple iterative equations. "
        "Found in **coastlines**, **snowflakes**, **blood vessels**, **antenna design**, and **financial markets**."
    )

    fractal_type = st.selectbox("Select fractal:", [
        "Mandelbrot Set", "Julia Set", "Fibonacci Spiral",
        "Sierpinski Triangle", "Newton Fractal"
    ])

    if fractal_type == "Mandelbrot Set":
        c1, c2 = st.columns([1, 2])
        with c1:
            res_m  = st.slider("Resolution:", 150, 500, 300)
            max_it = st.slider("Max iterations:", 50, 300, 100)
            cmap_m = st.selectbox("Colormap:", ["inferno","plasma","hot","viridis","magma"])
            xmin = st.number_input("x min:", value=-2.5)
            xmax = st.number_input("x max:", value=1.0)
            ymin = st.number_input("y min:", value=-1.25)
            ymax = st.number_input("y max:", value=1.25)
        with c2:
            with st.spinner("Generating Mandelbrot set..."):
                X_m = np.linspace(xmin, xmax, res_m)
                Y_m = np.linspace(ymin, ymax, res_m)
                C_m = X_m[np.newaxis, :] + 1j * Y_m[:, np.newaxis]
                Z_m = np.zeros_like(C_m)
                M   = np.zeros(C_m.shape)
                for n in range(max_it):
                    mask = np.abs(Z_m) <= 2
                    Z_m[mask] = Z_m[mask] ** 2 + C_m[mask]
                    M[mask & (np.abs(Z_m) > 2)] = n
                fig_mb, ax_mb = plt.subplots(figsize=(10, 7))
                fig_mb.patch.set_facecolor('#0a0a0a')
                ax_mb.imshow(M, extent=[xmin, xmax, ymin, ymax],
                             cmap=cmap_m, origin='lower', interpolation='bilinear')
                ax_mb.set_title("Mandelbrot Set  —  z² + c", color='gold', fontsize=13)
                ax_mb.set_xlabel("Real", color='white')
                ax_mb.set_ylabel("Imaginary", color='white')
                ax_mb.tick_params(colors='white')
                st.pyplot(fig_mb)
                plt.close()

    elif fractal_type == "Julia Set":
        c1, c2 = st.columns([1, 2])
        with c1:
            res_j  = st.slider("Resolution:", 150, 400, 250)
            c_real = st.slider("c real:", -1.0, 1.0, -0.7, 0.01)
            c_imag = st.slider("c imaginary:", -1.0, 1.0, 0.27, 0.01)
            cmap_j = st.selectbox("Colormap:", ["plasma","inferno","hot","viridis"])
        with c2:
            with st.spinner("Generating Julia set..."):
                X_j = np.linspace(-1.5, 1.5, res_j)
                Y_j = np.linspace(-1.5, 1.5, res_j)
                Z_j = X_j[np.newaxis, :] + 1j * Y_j[:, np.newaxis]
                J   = np.zeros(Z_j.shape)
                c_j = complex(c_real, c_imag)
                for n in range(100):
                    mask_j = np.abs(Z_j) <= 2
                    Z_j[mask_j] = Z_j[mask_j] ** 2 + c_j
                    J[mask_j & (np.abs(Z_j) > 2)] = n
                fig_jl, ax_jl = plt.subplots(figsize=(8, 8))
                fig_jl.patch.set_facecolor('#0a0a0a')
                ax_jl.imshow(J, extent=[-1.5, 1.5, -1.5, 1.5],
                             cmap=cmap_j, origin='lower', interpolation='bilinear')
                ax_jl.set_title(f"Julia Set  c = {c_real:.2f} + {c_imag:.2f}i",
                                color='gold', fontsize=13)
                ax_jl.tick_params(colors='white')
                st.pyplot(fig_jl)
                plt.close()

    elif fractal_type == "Fibonacci Spiral":
        fig_fib, ax_fib = plt.subplots(figsize=(10, 10))
        fig_fib.patch.set_facecolor('#0a0a0a')
        ax_fib.set_facecolor('#0a0a0a')
        fibs = [1,1,2,3,5,8,13,21,34,55,89]
        cols_fib = plt.cm.plasma(np.linspace(0.1, 1, len(fibs)))
        xf, yf, af = 0.0, 0.0, 0.0
        for i, (fib, cf) in enumerate(zip(fibs, cols_fib)):
            th = np.linspace(af, af + np.pi/2, 200)
            ax_fib.plot(xf + fib*np.cos(th), yf + fib*np.sin(th), color=cf, linewidth=2.5)
            if   i % 4 == 0: xf += fib
            elif i % 4 == 1: yf += fib
            elif i % 4 == 2: xf -= fib
            else:             yf -= fib
            af += np.pi / 2
        ax_fib.set_aspect('equal')
        ax_fib.axis('off')
        ax_fib.set_title("Fibonacci Spiral  —  φ = 1.6180339...", color='gold', fontsize=14)
        st.pyplot(fig_fib)
        plt.close()

    elif fractal_type == "Sierpinski Triangle":
        n_pts = 60000
        verts = np.array([[0.0,0.0],[1.0,0.0],[0.5, np.sqrt(3)/2]])
        p_s   = np.array([0.0, 0.0])
        xs_s, ys_s = [], []
        for _ in range(n_pts):
            v = verts[np.random.randint(3)]
            p_s = (p_s + v) / 2
            xs_s.append(p_s[0])
            ys_s.append(p_s[1])
        fig_st, ax_st = plt.subplots(figsize=(8, 8))
        fig_st.patch.set_facecolor('#0a0a0a')
        ax_st.set_facecolor('#0a0a0a')
        ax_st.scatter(xs_s, ys_s, s=0.1, c=ys_s, cmap='plasma', alpha=0.8)
        ax_st.set_aspect('equal')
        ax_st.axis('off')
        ax_st.set_title("Sierpinski Triangle  —  Chaos Game", color='gold', fontsize=13)
        st.pyplot(fig_st)
        plt.close()

    elif fractal_type == "Newton Fractal":
        with st.spinner("Generating Newton fractal..."):
            res_n = 350
            X_n = np.linspace(-2, 2, res_n)
            Y_n = np.linspace(-2, 2, res_n)
            Z_n = X_n[np.newaxis, :] + 1j * Y_n[:, np.newaxis]
            col_n = np.zeros((res_n, res_n))
            roots = np.array([1.0+0j, -0.5+0.866j, -0.5-0.866j])
            for _ in range(40):
                Z_n = Z_n - (Z_n**3 - 1) / (3 * Z_n**2 + 1e-10)
            for i, root in enumerate(roots):
                col_n[np.abs(Z_n - root) < 0.05] = i + 1
            fig_nw, ax_nw = plt.subplots(figsize=(8, 8))
            fig_nw.patch.set_facecolor('#0a0a0a')
            ax_nw.imshow(col_n, extent=[-2,2,-2,2], cmap='plasma',
                         origin='lower', interpolation='bilinear')
            ax_nw.set_title("Newton Fractal  —  z³ - 1 = 0", color='gold', fontsize=13)
            ax_nw.tick_params(colors='white')
            st.pyplot(fig_nw)
            plt.close()

    st.markdown("---")
    if st.button("🤖 AI Explain This Fractal"):
        with st.spinner("AI explaining..."):
            try:
                exp_frac = explain_pattern(
                    fractal_type,
                    "Self-similar mathematical structure generated by iterative equations"
                )
                st.markdown(exp_frac)
            except Exception as e:
                st.error(f"AI error: {e}")

    st.markdown("#### Fractals in the Real World")
    frac_apps = {
        "🌿 Nature":      "Ferns, snowflakes, river deltas, lung bronchi, cauliflower",
        "📡 Antennas":    "Fractal antennas fit multiple frequencies in compact space",
        "💹 Finance":     "Stock price movements follow fractal self-similarity",
        "🖥️ Graphics":   "Terrain generation, clouds, and textures in CGI",
        "🫀 Medicine":    "Heart rate variability and blood vessel branching",
    }
    fc = st.columns(len(frac_apps))
    for col, (name, desc) in zip(fc, frac_apps.items()):
        col.markdown(f"**{name}**")
        col.caption(desc)

# ══════════════════════════════════════════════════════════
# MODULE 6 — RL PATTERN AGENT
# ══════════════════════════════════════════════════════════
elif module == "🤖 RL Pattern Agent":
    st.markdown("## 🤖 Reinforcement Learning Pattern Agent")
    st.markdown(
        "This agent uses **Q-learning** to learn which mathematical patterns "
        "are most relevant for a given image. It improves with every image you upload."
    )

    try:
        agent = get_agent()
        stats = agent.get_stats()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Episodes",    stats["total_episodes"])
        col2.metric("Exploration Rate ε", f"{stats['epsilon']:.4f}")
        col3.metric("States Learned",    stats["states_learned"])
        col4.metric("Available Actions", len(stats["actions"]))

        st.markdown("---")
        st.markdown("### Upload Image — Agent Recommends Patterns")
        uploaded_rl = st.file_uploader("Upload image for RL analysis:", type=["jpg","jpeg","png"])

        if uploaded_rl:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as f:
                f.write(uploaded_rl.getvalue())
                tmp_rl = f.name

            features = extract_features(tmp_rl)
            top3     = agent.select_top_k(features, k=3)
            action_i, action_name = agent.select_action(features)

            st.markdown("#### Feature Vector Extracted from Image")
            feat_labels = ["Symmetry", "Edge Density", "Color Entropy",
                           "Golden Ratio Align", "Brightness", "Object Density"]
            fc = st.columns(len(feat_labels))
            for col, label, val in zip(fc, feat_labels, features):
                col.metric(label, f"{val:.3f}")

            st.markdown("---")
            st.success(f"**Agent Selected Action:** `{action_name}`")
            st.info(f"**Top 3 Recommended Patterns:** {', '.join(top3)}")

            # Auto reward
            reward = agent.auto_reward(features, action_i)
            agent.give_reward(reward)
            st.caption(f"Auto-reward applied: {reward:.3f}")

            st.markdown("---")
            st.markdown("#### Manual Feedback — Help the Agent Learn")
            fb1, fb2, fb3 = st.columns(3)
            if fb1.button("👍 Good recommendation"):
                agent.give_reward(1.0)
                st.success("Agent rewarded +1.0 — it will prefer this pattern more.")
            if fb2.button("😐 Neutral"):
                agent.give_reward(0.0)
                st.info("Neutral feedback recorded.")
            if fb3.button("👎 Wrong recommendation"):
                agent.give_reward(-1.0)
                st.warning("Agent penalized -1.0 — it will avoid this pattern.")

            os.unlink(tmp_rl)

        st.markdown("---")
        st.markdown("### Q-Table Visualization")
        if stats["states_learned"] > 0:
            st.markdown(f"The agent has learned **{stats['states_learned']} unique states** "
                        f"over **{stats['total_episodes']} episodes**.")
            st.markdown("**Action space:**")
            for i, action in enumerate(stats["actions"]):
                st.markdown(f"- `Action {i}`: {action}")
        else:
            st.info("No states learned yet. Upload images to start training the agent.")

        st.markdown("---")
        st.markdown("### How Q-Learning Works Here")
        st.markdown("""
        1. **State** — 6 image features (symmetry, edges, color entropy, golden ratio alignment, brightness, object density) discretized into bins
        2. **Actions** — 6 pattern overlays: fibonacci spiral, golden ratio, symmetry axes, fractal highlight, radial symmetry, wave pattern
        3. **Reward** — User feedback (+1 helpful, -1 not helpful) or automatic heuristic reward
        4. **Update** — Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') − Q(s,a)]
        5. **Policy** — ε-greedy: explore randomly with probability ε, exploit best known action otherwise
        6. **Learning** — ε decays over time so the agent exploits more as it learns
        """)

    except Exception as e:
        st.error(f"RL Agent error: {e}")
        st.info("Make sure modules/rl_agent.py is present.")

# ══════════════════════════════════════════════════════════
# MODULE 7 — AI ASSISTANT
# ══════════════════════════════════════════════════════════
elif module == "💬 AI Assistant":
    st.markdown("## 💬 AI Assistant")
    st.markdown(
        "Ask anything about mathematics, physics, patterns, equations, or nature. "
        "The AI explains concepts, derives formulas, and connects math to the real world."
    )

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Suggested questions
    st.markdown("#### Suggested Questions — click to ask:")
    suggestions = [
        "What is the golden ratio and where does it appear in nature?",
        "Explain the Mandelbrot set and how it is generated",
        "How does Fourier transform work in MRI imaging?",
        "What mathematical patterns do magnetic field lines follow?",
        "Explain Euler's identity e^(iπ) + 1 = 0",
        "What is fractal dimension and how is it measured?",
        "How does the Fibonacci sequence relate to the golden ratio?",
        "What equations describe the shape of a black hole?",
        "Explain symmetry groups in crystallography",
        "How is calculus used in machine learning?",
    ]
    sc = st.columns(2)
    for i, q in enumerate(suggestions):
        if sc[i % 2].button(q, key=f"sq_{i}"):
            st.session_state.chat_history.append({"role": "user", "content": q})
            with st.spinner("AI thinking..."):
                try:
                    ans = chat_with_ai(q)
                    st.session_state.chat_history.append({"role": "assistant", "content": ans})
                except Exception as e:
                    st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {e}"})

    st.markdown("---")

    # Display chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask about any equation, pattern, or mathematical concept...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("AI thinking..."):
                try:
                    context = f"Previous questions: {[m['content'][:50] for m in st.session_state.chat_history[-4:] if m['role']=='user']}"
                    response = chat_with_ai(user_input, context)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
                except Exception as e:
                    err_msg = f"AI error: {e}"
                    st.error(err_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": err_msg})

    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("---")
    st.markdown("#### What can you ask?")
    topics = {
        "📐 Algebra":       "Equations, polynomials, matrices, eigenvalues",
        "📊 Calculus":      "Derivatives, integrals, differential equations",
        "🌀 Geometry":      "Curves, surfaces, topology, symmetry groups",
        "🌊 Physics":       "Wave equations, field lines, quantum mechanics",
        "🌿 Nature":        "Fibonacci in plants, fractals in coastlines",
        "🤖 AI & Math":     "Linear algebra in neural networks, optimization",
    }
    tc = st.columns(len(topics))
    for col, (name, desc) in zip(tc, topics.items()):
        col.markdown(f"**{name}**")
        col.caption(desc)

import cv2
from modules.ai_engine import explain_pattern, analyze_image_patterns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ultralytics import YOLO


model = YOLO('yolov8n.pt')

MATH_PATTERNS = {
    'person':       ('Bilateral Symmetry',   'Golden Ratio in body proportions — head:body = 1:1.618'),
    'bird':         ('Fibonacci Spiral',     'Wing symmetry follows golden angle 137.5 degrees'),
    'cat':          ('Golden Ratio',         'Facial proportions follow phi = 1.618'),
    'dog':          ('Bilateral Symmetry',   'Body proportion ratios mirror each other'),
    'car':          ('Geometric Symmetry',   'Bilateral + parallel line geometry'),
    'bicycle':      ('Circle Geometry',      'Rotational symmetry in wheels'),
    'clock':        ('Rotational Symmetry',  '360 degree equal division'),
    'pizza':        ('Radial Symmetry',      'Equal sector division from center'),
    'bottle':       ('Rotational Symmetry',  'Cylindrical golden proportions'),
    'book':         ('Golden Rectangle',     'Page ratio close to 1.618'),
    'sunflower':    ('Fibonacci Spiral',     'Seeds follow 21/34 Fibonacci spiral pattern'),
    'potted plant': ('Fractal Branching',    'Self-similar growth patterns in leaves'),
    'orange':       ('Radial Symmetry',      'Segments follow equal angular division'),
    'apple':        ('Pentagonal Symmetry',  'Five fold internal seed structure'),
    'banana':       ('Logarithmic Curve',    'Natural parabolic growth arc'),
    'broccoli':     ('Fractal Pattern',      'Self-similar floret structure'),
    'tree':         ('Fractal Branching',    'Each branch is smaller copy of whole tree'),
    'building':     ('Geometric Proportion', 'Rectangular ratios and vertical symmetry'),
    'flower':       ('Radial Symmetry',      'Petals arranged in Fibonacci numbers'),
}

def find_main_subject(img):
    """
    Sirf main subject detect karo — background ignore karo.
    GrabCut se foreground nikalo phir color check karo.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # GrabCut — center region ko main subject mano
    mask   = np.zeros((h, w), np.uint8)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    margin_x = w // 6
    margin_y = h // 6
    rect = (margin_x, margin_y, w - 2*margin_x, h - 2*margin_y)

    try:
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5,
                    cv2.GC_INIT_WITH_RECT)
        fg_mask = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    except:
        fg_mask = np.ones((h, w), dtype='uint8')

    # Foreground pe color check karo
    results = {}

    color_ranges = {
        'sunflower':  [(18,  80,  80), (38, 255, 255), 'Fibonacci Spiral',
                       'Seeds follow 21/34 Fibonacci spiral pattern',   'yellow'],
        'orange obj': [(8,   80,  80), (18, 255, 255), 'Radial Symmetry',
                       'Equal angular sector division from center',      'orange'],
        'red obj':    [(0,   80,  80), (8,  255, 255), 'Bilateral Symmetry',
                       'Mirror symmetry in organic shapes',              'red'],
        'green obj':  [(38,  60,  60), (85, 255, 255), 'Fractal Branching',
                       'Self similar branching patterns in nature',      'lime'],
        'purple obj': [(125, 60,  60), (155,255, 255), 'Golden Proportion',
                       'Phi ratio in natural purple flowers',            'violet'],
    }

    for name, (lo, hi, pattern, detail, color) in color_ranges.items():
        color_mask = cv2.inRange(hsv, lo, hi)
        # Sirf foreground mein check karo
        combined   = cv2.bitwise_and(color_mask, color_mask,
                                     mask=fg_mask)
        percent    = np.sum(combined > 0) / (h * w) * 100

        if percent > 5:
            M = cv2.moments(combined)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
            else:
                cx, cy = w//2, h//2

            # Bounding box from mask
            ys, xs = np.where(combined > 0)
            if len(xs) > 0:
                x1, x2 = int(xs.min()), int(xs.max())
                y1, y2 = int(ys.min()), int(ys.max())
                obj_size = min(x2-x1, y2-y1)
            else:
                x1, y1  = margin_x, margin_y
                x2, y2  = w-margin_x, h-margin_y
                obj_size = min(w, h) // 2

            results[name] = {
                'percent':  percent,
                'center':   (cx, cy),
                'size':     obj_size,
                'pattern':  pattern,
                'detail':   detail,
                'color':    color,
                'bbox':     (x1, y1, x2, y2),
            }

    return results

def draw_fibonacci_spiral(ax, cx, cy, scale):
    """Fibonacci spiral — exact center pe, controlled size"""
    fibs   = [1,1,2,3,5,8,13,21]
    colors = plt.cm.plasma(np.linspace(0.2, 1, len(fibs)))
    x_pos, y_pos, angle = float(cx), float(cy), 0.0

    for i, (fib, fc) in enumerate(zip(fibs, colors)):
        theta = np.linspace(angle, angle + np.pi/2, 80)
        r     = fib * scale
        sx    = x_pos + r * np.cos(theta)
        sy    = y_pos + r * np.sin(theta)
        ax.plot(sx, sy, color=fc, linewidth=2.5, alpha=0.95)
        if   i % 4 == 0: x_pos += fib * scale
        elif i % 4 == 1: y_pos += fib * scale
        elif i % 4 == 2: x_pos -= fib * scale
        else:             y_pos -= fib * scale
        angle += np.pi / 2

def draw_golden_overlay(ax, cx, cy, size):
    """Golden ratio rectangle + division lines — object size ke andar"""
    PHI  = 1.6180339887
    half = size * 0.5

    rect = patches.Rectangle(
        (cx - half, cy - half/PHI),
        half*2, (half*2)/PHI,
        linewidth=1.5, edgecolor='gold',
        facecolor='none', linestyle='--', alpha=0.75
    )
    ax.add_patch(rect)

    # Inner golden divisions
    ax.plot([cx - half/PHI, cx - half/PHI],
            [cy - half/PHI, cy + half/PHI],
            color='gold', linewidth=1, alpha=0.6, linestyle=':')
    ax.plot([cx + half/PHI, cx + half/PHI],
            [cy - half/PHI, cy + half/PHI],
            color='gold', linewidth=1, alpha=0.6, linestyle=':')

    # 3 golden circles only — sized to object
    for mult in [0.3, 0.5, 0.8]:
        circle = plt.Circle(
            (cx, cy), half * mult,
            color='gold', fill=False,
            linewidth=1.2, alpha=0.5, linestyle='--'
        )
        ax.add_patch(circle)

def draw_symmetry_axes(ax, cx, cy, size):
    """Symmetry lines — only within object bounds"""
    half = size * 0.5
    ax.plot([cx-half, cx+half], [cy, cy],
            color='cyan',    linewidth=1.5, alpha=0.7, linestyle='--')
    ax.plot([cx, cx], [cy-half, cy+half],
            color='lime',    linewidth=1.5, alpha=0.7, linestyle='--')
    ax.plot([cx-half, cx+half], [cy-half*0.7, cy+half*0.7],
            color='magenta', linewidth=0.8, alpha=0.4, linestyle=':')
    ax.plot([cx-half, cx+half], [cy+half*0.7, cy-half*0.7],
            color='magenta', linewidth=0.8, alpha=0.4, linestyle=':')

def detect_and_analyze(image_path):
    img     = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w    = img.shape[:2]

    yolo_results = model(image_path, verbose=False)

    detected_objects    = []
    math_patterns_found = []
    confidences         = []

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.patch.set_facecolor('#0a0a0a')

    axes[0].imshow(img_rgb)
    axes[0].set_facecolor('#0a0a0a')
    axes[0].axis('off')
    axes[0].set_title('AlgebraicVision AI — Pattern Detection',
                      color='cyan', fontsize=14, fontweight='bold')

    plot_colors = plt.cm.plasma(np.linspace(0.2, 1, 10))

    # YOLO detections
    for result in yolo_results:
        for j, box in enumerate(result.boxes):
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf  = float(box.conf[0])
            label = model.names[int(box.cls[0])].lower()

            detected_objects.append(label)
            confidences.append(conf)
            info = MATH_PATTERNS.get(label,
                   ('Geometric Form', 'Mathematical structure present'))
            math_patterns_found.append(info)

            color = plot_colors[j % len(plot_colors)]
            rect  = patches.Rectangle(
                (x1,y1), x2-x1, y2-y1,
                linewidth=2, edgecolor=color, facecolor='none'
            )
            axes[0].add_patch(rect)
            axes[0].text(x1, y1-10, f'{label}  {conf:.0%}',
                        color='white', fontsize=9,
                        bbox=dict(boxstyle='round,pad=0.2',
                                 facecolor=color, alpha=0.8))

            cx_obj  = (x1+x2)/2
            cy_obj  = (y1+y2)/2
            obj_sz  = min(x2-x1, y2-y1)
            scale   = obj_sz / 180

            draw_fibonacci_spiral(axes[0], cx_obj, cy_obj, scale)
            draw_golden_overlay(axes[0], cx_obj, cy_obj, obj_sz)
            draw_symmetry_axes(axes[0], cx_obj, cy_obj, obj_sz)

    # Color-based detection — foreground only
    color_detections = find_main_subject(img)

    for name, info in color_detections.items():
        if name not in detected_objects:
            cx, cy   = info['center']
            obj_size = info['size']
            scale    = obj_size / 180

            detected_objects.append(name)
            confidences.append(round(info['percent']/100, 2))
            math_patterns_found.append((info['pattern'], info['detail']))

            # Bounding box
            x1,y1,x2,y2 = info['bbox']
            rect = patches.Rectangle(
                (x1,y1), x2-x1, y2-y1,
                linewidth=2, edgecolor=info['color'],
                facecolor='none', linestyle='-'
            )
            axes[0].add_patch(rect)

            draw_fibonacci_spiral(axes[0], cx, cy, scale)
            draw_golden_overlay(axes[0], cx, cy, obj_size)
            draw_symmetry_axes(axes[0], cx, cy, obj_size)

            axes[0].text(cx, y1-12,
                        f'{info["pattern"]}',
                        color='lime', fontsize=10,
                        fontweight='bold', ha='center',
                        bbox=dict(boxstyle='round,pad=0.3',
                                 facecolor='#000000bb'))

    # Right panel
    axes[1].set_facecolor('#0d0d0d')
    axes[1].axis('off')
    axes[1].set_title('Mathematical Pattern Analysis',
                      color='gold', fontsize=14, fontweight='bold')

    gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mid_x = w // 2
    left  = gray[:, :mid_x]
    right = cv2.resize(cv2.flip(gray[:, mid_x:], 1),
                       (left.shape[1], left.shape[0]))
    sym_score = 100 - (np.mean(cv2.absdiff(left, right)) / 255 * 100)
    PHI = 1.6180339887

    y_pos = 0.95
    for obj, pattern, conf in zip(detected_objects,
                                   math_patterns_found,
                                   confidences):
        axes[1].text(0.05, y_pos,
                    f'Object : {obj.upper()}  ({conf:.0%})',
                    transform=axes[1].transAxes,
                    color='cyan', fontsize=11, fontweight='bold')
        y_pos -= 0.06
        axes[1].text(0.08, y_pos,
                    f'Pattern : {pattern[0]}',
                    transform=axes[1].transAxes,
                    color='yellow', fontsize=10)
        y_pos -= 0.05
        axes[1].text(0.08, y_pos,
                    f'Detail  : {pattern[1]}',
                    transform=axes[1].transAxes,
                    color='#bbbbbb', fontsize=9)
        y_pos -= 0.09
        if y_pos < 0.32:
            break

    if not detected_objects:
        axes[1].text(0.5, 0.6, 'No objects detected!\nTry a clearer image.',
                    transform=axes[1].transAxes,
                    color='red', fontsize=12, ha='center', va='center')

    axes[1].axhline(y=0.30, xmin=0.03, xmax=0.97,
                   color='#333', linewidth=0.8)

    stats = [
        ('Image Size',     f'{w} x {h} px'),
        ('Symmetry Score', f'{sym_score:.1f}%'),
        ('Golden Point X', f'{w/PHI:.0f} px'),
        ('Golden Point Y', f'{h/PHI:.0f} px'),
        ('Aspect Ratio',   f'{w/h:.3f}'),
        ('Objects Found',  f'{len(detected_objects)}'),
    ]
    y = -0.05
    for label, val in stats:
        axes[1].text(0.05, y, label,
                    transform=axes[1].transAxes,
                    color='#888888', fontsize=9)
        axes[1].text(0.60, y, val,
                    transform=axes[1].transAxes,
                    color='cyan', fontsize=9, fontweight='bold')
        y -= 0.045
# Groq AI Explanation
# Groq AI Explanation
    try:
        if detected_objects:
            ai_explanation = analyze_image_patterns(
                detected_objects,
                sym_score,
                f"Golden point at {w/PHI:.0f}x{h/PHI:.0f}px"
            )
            axes[1].text(0.05, 0.28,
                        'AI Analysis:',
                        transform=axes[1].transAxes,
                        color='gold', fontsize=10, fontweight='bold')
            
            words = ai_explanation.split()
            lines, line = [], []
            for word in words:
                line.append(word)
                if len(' '.join(line)) > 52:
                    lines.append(' '.join(line))
                    line = []
            if line:
                lines.append(' '.join(line))
            
            y_ai = 0.24
            for l in lines[:6]:
                axes[1].text(0.05, y_ai, l,
                            transform=axes[1].transAxes,
                            color='#cccccc', fontsize=8.5)
                y_ai -= 0.032
    except Exception as e:
        print(f"AI explanation error: {e}")
    plt.tight_layout()
    return fig, detected_objects, math_patterns_found, confidences


def get_eda_report(image_path):
    img     = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray    = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w    = img.shape[:2]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.patch.set_facecolor('#0a0a0a')
    fig.suptitle('AlgebraicVision AI — Complete EDA Report',
                 color='gold', fontsize=16, fontweight='bold')

    for ax in axes.flatten():
        ax.set_facecolor('#0d0d0d')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#333')

    axes[0,0].imshow(img_rgb)
    axes[0,0].set_title('Original Image',        color='cyan', fontsize=11)
    axes[0,0].axis('off')

    edges = cv2.Canny(gray, 50, 150)
    axes[0,1].imshow(edges, cmap='hot')
    axes[0,1].set_title('Edge Detection — Canny', color='cyan', fontsize=11)
    axes[0,1].axis('off')

    for color, ch in zip(['red','green','blue'], range(3)):
        hist = cv2.calcHist([img], [ch], None, [256], [0,256])
        axes[0,2].plot(hist, color=color, alpha=0.8, linewidth=1.5)
    axes[0,2].set_title('RGB Color Distribution', color='cyan', fontsize=11)
    axes[0,2].set_xlabel('Pixel Value', color='white', fontsize=9)
    axes[0,2].set_ylabel('Frequency',   color='white', fontsize=9)

    axes[1,0].imshow(gray, cmap='inferno')
    axes[1,0].set_title('Intensity Heatmap',      color='cyan', fontsize=11)
    axes[1,0].axis('off')

    mid_x = w//2
    left  = gray[:, :mid_x]
    right = cv2.resize(cv2.flip(gray[:, mid_x:],1),
                       (left.shape[1], left.shape[0]))
    diff  = cv2.absdiff(left, right)
    sym_score = 100 - (np.mean(diff)/255*100)
    axes[1,1].imshow(diff, cmap='RdYlGn_r')
    axes[1,1].set_title(f'Symmetry Map  |  Score: {sym_score:.1f}%',
                         color='cyan', fontsize=11)
    axes[1,1].axis('off')

    axes[1,2].axis('off')
    PHI   = 1.6180339887
    stats = [
        ('Image Size',      f'{w} x {h} px'),
        ('Mean Brightness', f'{np.mean(gray):.1f} / 255'),
        ('Std Deviation',   f'{np.std(gray):.1f}'),
        ('Symmetry Score',  f'{sym_score:.1f}%'),
        ('Golden Point X',  f'{w/PHI:.0f} px'),
        ('Golden Point Y',  f'{h/PHI:.0f} px'),
        ('Contrast',        f'{int(gray.max()-gray.min())}'),
        ('Aspect Ratio',    f'{w/h:.3f}'),
        ('vs Golden Ratio', f'{abs(w/h-PHI):.3f} diff'),
        ('Total Pixels',    f'{w*h:,}'),
    ]
    y = 0.95
    for label, value in stats:
        axes[1,2].text(0.05, y, label,
                      transform=axes[1,2].transAxes,
                      color='#aaaaaa', fontsize=10)
        axes[1,2].text(0.65, y, value,
                      transform=axes[1,2].transAxes,
                      color='cyan', fontsize=10, fontweight='bold')
        y -= 0.09
    axes[1,2].set_title('Mathematical Statistics',
                         color='gold', fontsize=11)

    plt.tight_layout()
    return fig, sym_score, w/PHI, h/PHI
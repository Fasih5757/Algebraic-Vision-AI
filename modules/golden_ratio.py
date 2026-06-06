import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

PHI = 1.6180339887  # Golden Ratio

def detect_golden_ratio(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    h, w = img.shape[:2]
    
    # Golden ratio lines calculate karo
    golden_w = w / PHI
    golden_h = h / PHI

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    ax.imshow(img_rgb)
    
    # Vertical golden line
    ax.axvline(x=golden_w, color='gold', linewidth=2, 
               linestyle='--', label=f'Golden Ratio (φ={PHI:.3f})')
    ax.axvline(x=w - golden_w, color='gold', linewidth=2, linestyle='--')
    
    # Horizontal golden line
    ax.axhline(y=golden_h, color='cyan', linewidth=2, linestyle='--')
    ax.axhline(y=h - golden_h, color='cyan', linewidth=2, linestyle='--')

    # Golden spiral overlay
    theta = np.linspace(0, 4 * np.pi, 1000)
    spiral_x = (w/2) + (theta * 15) * np.cos(theta)
    spiral_y = (h/2) + (theta * 10) * np.sin(theta)
    ax.plot(spiral_x, spiral_y, color='lime', linewidth=1.5, 
            alpha=0.7, label='Golden Spiral')

    ax.set_title('AlgebraicVision AI — Golden Ratio Detector', 
                 color='gold', fontsize=14, fontweight='bold')
    ax.legend(facecolor='#111', labelcolor='white', fontsize=10)
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    print(f"Image size: {w}x{h}")
    print(f"Golden point X: {golden_w:.1f}px")
    print(f"Golden point Y: {golden_h:.1f}px")

if __name__ == "__main__":
    # Koi bhi image ka path yahan likho
    detect_golden_ratio("test_image.jpg")
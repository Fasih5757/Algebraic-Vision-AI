import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_symmetry(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    h, w = img.shape[:2]
    mid_x = w // 2
    mid_y = h // 2

    # Left aur right half compare karo
    left_half = gray[:, :mid_x]
    right_half = cv2.flip(gray[:, mid_x:], 1)
    
    # Right half ko left ke size pe resize karo
    right_half = cv2.resize(right_half, (left_half.shape[1], left_half.shape[0]))
    
    # Symmetry score calculate karo
    diff = cv2.absdiff(left_half, right_half)
    symmetry_score = 100 - (np.mean(diff) / 255 * 100)

    # Top aur bottom half compare karo
    top_half = gray[:mid_y, :]
    bottom_half = cv2.flip(gray[mid_y:, :], 0)
    bottom_half = cv2.resize(bottom_half, (top_half.shape[1], top_half.shape[0]))
    
    diff_v = cv2.absdiff(top_half, bottom_half)
    v_symmetry_score = 100 - (np.mean(diff_v) / 255 * 100)

    # Plot karo
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#0a0a0a')

    titles = ['Original Image', 
              f'Horizontal Symmetry\nScore: {symmetry_score:.1f}%',
              f'Vertical Symmetry\nScore: {v_symmetry_score:.1f}%']
    images = [img_rgb, 
              np.hstack([left_half, cv2.flip(right_half, 1)]),
              np.vstack([top_half, cv2.flip(bottom_half, 0)])]

    for ax, im, title in zip(axes, images, titles):
        ax.set_facecolor('#0a0a0a')
        ax.imshow(im, cmap='gray' if im.ndim == 2 else None)
        ax.set_title(title, color='cyan', fontsize=11)
        ax.axis('off')

    # Symmetry lines draw karo
    axes[0].axvline(x=mid_x, color='red', linewidth=2, 
                    linestyle='--', label='Vertical axis')
    axes[0].axhline(y=mid_y, color='lime', linewidth=2, 
                    linestyle='--', label='Horizontal axis')
    axes[0].legend(facecolor='#111', labelcolor='white', fontsize=9)

    fig.suptitle('AlgebraicVision AI — Symmetry Detector', 
                 color='gold', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

    print(f"Horizontal Symmetry Score: {symmetry_score:.1f}%")
    print(f"Vertical Symmetry Score: {v_symmetry_score:.1f}%")
    
    if symmetry_score > 70:
        print("High horizontal symmetry detected!")
    if v_symmetry_score > 70:
        print("High vertical symmetry detected!")

if __name__ == "__main__":
    detect_symmetry("test_image.jpg")
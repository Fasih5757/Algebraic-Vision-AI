import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(c, max_iter=100):
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z*z + c
    return max_iter

def draw_mandelbrot():
    print("Mandelbrot generating... thoda wait karo!")
    
    width, height = 800, 600
    x = np.linspace(-2.5, 1.0, width)
    y = np.linspace(-1.25, 1.25, height)
    
    mandelbrot_set = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            c = complex(x[j], y[i])
            mandelbrot_set[i, j] = mandelbrot(c, max_iter=100)
    
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    ax.imshow(mandelbrot_set, extent=[-2.5, 1.0, -1.25, 1.25],
              cmap='inferno', origin='lower', interpolation='bilinear')
    
    ax.set_title('AlgebraicVision AI — Mandelbrot Fractal', 
                 color='gold', fontsize=14, fontweight='bold')
    ax.set_xlabel('Real', color='white')
    ax.set_ylabel('Imaginary', color='white')
    ax.tick_params(colors='white')
    
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    plt.tight_layout()
    plt.savefig('mandelbrot.png', dpi=150, 
                bbox_inches='tight', facecolor='#0a0a0a')
    plt.show()
    print("Mandelbrot saved as mandelbrot.png!")

def fibonacci_spiral():
    fig, ax = plt.subplots(figsize=(10, 10))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    # Fibonacci numbers
    fibs = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55]
    
    colors = plt.cm.plasma(np.linspace(0.2, 1, len(fibs)))
    
    x, y = 0, 0
    angle = 0
    
    for i, (fib, color) in enumerate(zip(fibs, colors)):
        theta = np.linspace(angle, angle + np.pi/2, 100)
        r = fib
        spiral_x = x + r * np.cos(theta)
        spiral_y = y + r * np.sin(theta)
        ax.plot(spiral_x, spiral_y, color=color, linewidth=2.5)
        
        # Next square position
        if i % 4 == 0:   x += fib
        elif i % 4 == 1: y += fib
        elif i % 4 == 2: x -= fib
        else:             y -= fib
        angle += np.pi / 2

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('AlgebraicVision AI — Fibonacci Spiral', 
                 color='gold', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    print("Fibonacci Spiral ready!")

if __name__ == "__main__":
    fibonacci_spiral()   # Pehle yeh — fast hai
    draw_mandelbrot()    # Phir yeh — 1-2 min lagenge
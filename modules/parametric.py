import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def draw_all_curves():
    t = np.linspace(0, 2 * np.pi, 1000)
    
    curves = {
        'Ellipse':        (2 * np.cos(t), np.sin(t), 'cyan'),
        'Spiral':         (t * np.cos(t), t * np.sin(t), 'yellow'),
        'Rose Curve':     (np.cos(3*t) * np.cos(t), np.cos(3*t) * np.sin(t), 'magenta'),
        'Infinity ∞':     (np.cos(t), np.sin(2*t)/2, 'lime'),
        'Diamond':        (np.abs(np.cos(t))*np.cos(t), np.abs(np.sin(t))*np.sin(t), 'orange'),
        'Butterfly':      (np.sin(t)*(np.e**np.cos(t) - 2*np.cos(4*t)), 
                          np.cos(t)*(np.e**np.cos(t) - 2*np.cos(4*t)), 'pink'),
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.patch.set_facecolor('#0a0a0a')
    axes = axes.flatten()

    for ax, (name, (x, y, color)) in zip(axes, curves.items()):
        ax.set_facecolor('#0d0d0d')
        ax.plot(x, y, color=color, linewidth=2)
        ax.set_title(name, color=color, fontsize=13, fontweight='bold')
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Grid dots
        ax.grid(True, color='#222', linewidth=0.5)

    fig.suptitle('AlgebraicVision AI — Parametric Curves', 
                 color='gold', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    print("All parametric curves drawn!")

def animate_curve():
    """Ek curve ko animate karo — drawing hoti dikhegi"""
    t_full = np.linspace(0, 2 * np.pi, 1000)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    ax.set_title('Rose Curve — Drawing Live', color='magenta', fontsize=14)

    line, = ax.plot([], [], color='magenta', linewidth=2)
    dot,  = ax.plot([], [], 'o', color='white', markersize=6)

    def update(frame):
        t = t_full[:frame]
        x = np.cos(3*t) * np.cos(t)
        y = np.cos(3*t) * np.sin(t)
        line.set_data(x, y)
        if len(x) > 0:
            dot.set_data([x[-1]], [y[-1]])
        return line, dot

    ani = animation.FuncAnimation(fig, update, frames=len(t_full),
                                   interval=10, blit=True)
    plt.show()

if __name__ == "__main__":
    draw_all_curves()    # Pehle sab curves dekho
    animate_curve()      # Phir ani
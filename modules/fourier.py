import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def fourier_series(t, n_terms):
    """Square wave ko Fourier series se banao"""
    result = np.zeros_like(t)
    for k in range(1, n_terms + 1, 2):  # Sirf odd terms
        result += (4 / (np.pi * k)) * np.sin(k * t)
    return result

def plot_fourier(n_terms=10):
    t = np.linspace(0, 4 * np.pi, 1000)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.patch.set_facecolor('#0a0a0a')
    
    for ax in [ax1, ax2]:
        ax.set_facecolor('#0a0a0a')
        ax.tick_params(colors='white')
        ax.spines['bottom'].set_color('#333')
        ax.spines['left'].set_color('#333')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # Har term alag color mein
    colors = plt.cm.plasma(np.linspace(0.2, 1, n_terms))
    
    for i, k in enumerate(range(1, n_terms + 1, 2)):
        term = (4 / (np.pi * k)) * np.sin(k * t)
        ax1.plot(t, term, color=colors[i], alpha=0.6, 
                linewidth=0.8, label=f'n={k}')
    
    ax1.set_title('Wave Component Stacking', color='cyan', fontsize=14)
    ax1.legend(loc='upper right', fontsize=7, 
               facecolor='#111', labelcolor='white', ncol=3)

    # Final combined wave
    final_wave = fourier_series(t, n_terms)
    ax2.plot(t, final_wave, color='cyan', linewidth=2)
    ax2.set_title(f'Combined Wave ({n_terms} terms)', 
                  color='cyan', fontsize=14)
    ax2.set_ylim(-1.5, 1.5)
    
    plt.tight_layout()
    plt.savefig('fourier_output.png', dpi=150, 
                bbox_inches='tight', facecolor='#0a0a0a')
    plt.show()
    print("✅ Fourier visualization saved!")

if __name__ == "__main__":
    plot_fourier(n_terms=19)
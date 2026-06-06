import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════
BASE = Path("datasets")
FLOWERS_PATH      = BASE / "flowers"
FRACTALS_PATH     = BASE / "fractals" / "fractal"
INTEL_TRAIN_PATH  = BASE / "intel_images" / "seg_train" / "seg_train"
INTEL_TEST_PATH   = BASE / "intel_images" / "seg_test"
FRACTAL_DREAM_PATH= BASE / "fractal_dream" / "FDD"
ANIMALS_CSV       = BASE / "animals" / "Animal Dataset.csv"

FLOWER_CLASSES = ['daisy','dandelion','rose','sunflower','tulip']
MATH_PATTERNS  = {
    'daisy':     'Radial Symmetry — petals equally spaced',
    'dandelion': 'Fibonacci Spiral — seed arrangement',
    'rose':      'Golden Ratio — petal growth follows phi',
    'sunflower': 'Fibonacci 21/34 — seed spiral pattern',
    'tulip':     'Bilateral Symmetry — mirror reflection',
}

# ══════════════════════════════════════════════════════════
# SECTION 1 — EDA
# ══════════════════════════════════════════════════════════

def compute_symmetry(img_gray):
    h, w = img_gray.shape
    left  = img_gray[:, :w//2]
    right = cv2.resize(cv2.flip(img_gray[:, w//2:], 1),
                       (left.shape[1], left.shape[0]))
    diff  = cv2.absdiff(left, right)
    return 100 - (np.mean(diff) / 255 * 100)

def compute_fractal_dimension(img_gray):
    """Box-counting fractal dimension"""
    thresh = cv2.threshold(img_gray, 128, 255,
                           cv2.THRESH_BINARY)[1]
    sizes, counts = [], []
    for size in [2, 4, 8, 16, 32, 64]:
        h, w   = thresh.shape
        count  = 0
        for i in range(0, h, size):
            for j in range(0, w, size):
                patch = thresh[i:i+size, j:j+size]
                if patch.max() > 0:
                    count += 1
        sizes.append(size)
        counts.append(count)
    try:
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return abs(coeffs[0])
    except:
        return 1.5

def compute_golden_ratio_score(img):
    h, w  = img.shape[:2]
    PHI   = 1.6180339887
    ratio = w / h if w > h else h / w
    return max(0, 100 - abs(ratio - PHI) * 100)

def eda_flowers():
    """Complete EDA on flowers dataset"""
    print("Running EDA on Flowers Dataset...")

    stats = []
    sample_imgs = {}

    for flower in FLOWER_CLASSES:
        folder = FLOWERS_PATH / flower
        if not folder.exists():
            continue

        files = list(folder.glob("*.jpg")) + \
                list(folder.glob("*.png")) + \
                list(folder.glob("*.jpeg"))

        sym_scores, frac_dims, golden_scores = [], [], []
        brightnesses, contrasts = [], []

        for f in files[:100]:   # First 100 per class
            try:
                img  = cv2.imread(str(f))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img_resized = cv2.resize(img, (128, 128))
                gray_r      = cv2.cvtColor(img_resized,
                                           cv2.COLOR_BGR2GRAY)

                sym_scores.append(compute_symmetry(gray_r))
                frac_dims.append(compute_fractal_dimension(gray_r))
                golden_scores.append(
                    compute_golden_ratio_score(img_resized))
                brightnesses.append(np.mean(gray))
                contrasts.append(np.std(gray))
            except:
                continue

        if sym_scores:
            stats.append({
                'Class':           flower.capitalize(),
                'Count':           len(files),
                'Avg Symmetry %':  round(np.mean(sym_scores), 1),
                'Avg Fractal Dim': round(np.mean(frac_dims), 3),
                'Golden Ratio Score': round(np.mean(golden_scores), 1),
                'Brightness':      round(np.mean(brightnesses), 1),
                'Contrast':        round(np.mean(contrasts), 1),
                'Math Pattern':    MATH_PATTERNS.get(flower, 'N/A'),
            })

            # Sample image
            if files:
                sample_imgs[flower] = str(files[0])

    df = pd.DataFrame(stats)
    return df, sample_imgs

def plot_eda_flowers(df, sample_imgs):
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#0a0a0a')
    fig.suptitle('AlgebraicVision AI — Flowers EDA Report',
                 color='gold', fontsize=18, fontweight='bold',
                 y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.45, wspace=0.35)

    # 1. Sample images
    for i, flower in enumerate(FLOWER_CLASSES[:3]):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor('#0d0d0d')
        if flower in sample_imgs:
            img = cv2.imread(sample_imgs[flower])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        ax.set_title(f'{flower.capitalize()}\n'
                     f'{MATH_PATTERNS[flower][:30]}...',
                     color='cyan', fontsize=8)
        ax.axis('off')

    # 2. Class distribution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor('#0d0d0d')
    colors = plt.cm.plasma(np.linspace(0.2, 1, len(df)))
    bars = ax2.bar(df['Class'], df['Count'],
                   color=colors, edgecolor='#333')
    ax2.set_title('Dataset Distribution',
                  color='cyan', fontsize=11)
    ax2.tick_params(colors='white')
    ax2.set_facecolor('#0d0d0d')
    for spine in ax2.spines.values():
        spine.set_color('#333')
    for bar, count in zip(bars, df['Count']):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 5,
                 str(count), ha='center',
                 color='white', fontsize=9)

    # 3. Symmetry scores
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor('#0d0d0d')
    ax3.bar(df['Class'], df['Avg Symmetry %'],
            color='cyan', alpha=0.8, edgecolor='#333')
    ax3.set_title('Average Symmetry Score %',
                  color='cyan', fontsize=11)
    ax3.tick_params(colors='white')
    ax3.axhline(y=df['Avg Symmetry %'].mean(),
                color='gold', linestyle='--',
                linewidth=1.5, label='Mean')
    ax3.legend(facecolor='#111', labelcolor='white')
    for spine in ax3.spines.values():
        spine.set_color('#333')

    # 4. Fractal dimension
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor('#0d0d0d')
    ax4.bar(df['Class'], df['Avg Fractal Dim'],
            color='magenta', alpha=0.8, edgecolor='#333')
    ax4.set_title('Fractal Dimension',
                  color='magenta', fontsize=11)
    ax4.tick_params(colors='white')
    for spine in ax4.spines.values():
        spine.set_color('#333')

    # 5. Golden ratio score
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_facecolor('#0d0d0d')
    ax5.bar(df['Class'], df['Golden Ratio Score'],
            color='gold', alpha=0.8, edgecolor='#333')
    ax5.set_title('Golden Ratio Score',
                  color='gold', fontsize=11)
    ax5.tick_params(colors='white')
    ax5.axhline(y=61.8, color='lime', linestyle='--',
                linewidth=1.5, label='PHI baseline')
    ax5.legend(facecolor='#111', labelcolor='white')
    for spine in ax5.spines.values():
        spine.set_color('#333')

    # 6. Brightness vs Contrast scatter
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_facecolor('#0d0d0d')
    scatter_colors = plt.cm.plasma(
        np.linspace(0.2, 1, len(df)))
    for i, row in df.iterrows():
        ax6.scatter(row['Brightness'], row['Contrast'],
                   s=200, color=scatter_colors[i],
                   label=row['Class'], zorder=5)
        ax6.annotate(row['Class'],
                    (row['Brightness'], row['Contrast']),
                    textcoords="offset points",
                    xytext=(5, 5), color='white', fontsize=8)
    ax6.set_title('Brightness vs Contrast',
                  color='cyan', fontsize=11)
    ax6.set_xlabel('Brightness', color='white', fontsize=9)
    ax6.set_ylabel('Contrast',   color='white', fontsize=9)
    ax6.tick_params(colors='white')
    for spine in ax6.spines.values():
        spine.set_color('#333')

    # 7. Stats table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    ax7.set_facecolor('#0d0d0d')
    ax7.set_title('Mathematical Analysis Summary',
                  color='gold', fontsize=11)
    y = 0.95
    for _, row in df.iterrows():
        ax7.text(0.02, y, f"{row['Class']}:",
                transform=ax7.transAxes,
                color='cyan', fontsize=9, fontweight='bold')
        y -= 0.06
        ax7.text(0.05, y,
                f"Symmetry: {row['Avg Symmetry %']}% | "
                f"Fractal: {row['Avg Fractal Dim']}",
                transform=ax7.transAxes,
                color='#bbbbbb', fontsize=8)
        y -= 0.08

    plt.savefig('eda_flowers_report.png', dpi=120,
                bbox_inches='tight',
                facecolor='#0a0a0a')
    print("EDA saved: eda_flowers_report.png")
    return fig

def eda_intel_images():
    """EDA on Intel Image Classification dataset"""
    print("Running EDA on Intel Images Dataset...")

    categories = []
    
    # Check karo folders hain ya nahi
    if not INTEL_TRAIN_PATH.exists():
        print(f"Path not found: {INTEL_TRAIN_PATH}")
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#0d0d0d')
        ax.text(0.5, 0.5, 'Intel dataset path not found',
               ha='center', va='center',
               color='red', fontsize=12,
               transform=ax.transAxes)
        ax.axis('off')
        return fig, pd.DataFrame()

    for cat_folder in INTEL_TRAIN_PATH.iterdir():
        if not cat_folder.is_dir():
            continue
        files = (list(cat_folder.glob("*.jpg")) +
                 list(cat_folder.glob("*.png")) +
                 list(cat_folder.glob("*.jpeg")))
        if len(files) > 0:
            categories.append({
                'Category': cat_folder.name,
                'Count':    len(files),
            })

    if not categories:
        print("No categories found in intel_images!")
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#0a0a0a')
        ax.set_facecolor('#0d0d0d')
        ax.text(0.5, 0.5, 'No image categories found!\nCheck dataset structure',
               ha='center', va='center',
               color='yellow', fontsize=12,
               transform=ax.transAxes)
        ax.axis('off')
        return fig, pd.DataFrame()

    df = pd.DataFrame(categories)
    print(f"Intel categories found: {df['Category'].tolist()}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0a0a0a')
    fig.suptitle('Intel Images — EDA Report',
                 color='gold', fontsize=14, fontweight='bold')

    for ax in axes:
        ax.set_facecolor('#0d0d0d')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#333')

    colors = plt.cm.plasma(np.linspace(0.2, 1, len(df)))
    
    axes[0].bar(df['Category'], df['Count'],
                color=colors, edgecolor='#222')
    axes[0].set_title('Category Distribution',
                      color='cyan', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)

    if len(df) > 1:
        axes[1].pie(df['Count'],
                    labels=df['Category'],
                    colors=colors,
                    autopct='%1.1f%%',
                    textprops={'color': 'white'})
        axes[1].set_title('Category Proportions',
                          color='cyan', fontsize=12)
    else:
        axes[1].text(0.5, 0.5, 'Only 1 category found',
                    ha='center', va='center',
                    color='yellow', fontsize=12,
                    transform=axes[1].transAxes)
        axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('eda_intel_report.png', dpi=120,
                bbox_inches='tight', facecolor='#0a0a0a')
    print("EDA saved: eda_intel_report.png")
    return fig, df

def eda_animals_csv():
    """EDA on Animals CSV dataset"""
    print("Running EDA on Animals CSV...")

    df = pd.read_csv(ANIMALS_CSV)
    print(f"Animals dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('#0a0a0a')
    fig.suptitle('Animals Dataset — EDA Report',
                 color='gold', fontsize=14, fontweight='bold')

    for ax in axes.flatten():
        ax.set_facecolor('#0d0d0d')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#333')

    # Distribution of first categorical column
    cat_cols = df.select_dtypes(
        include=['object']).columns.tolist()
    num_cols = df.select_dtypes(
        include=[np.number]).columns.tolist()

    if cat_cols:
        vc = df[cat_cols[0]].value_counts().head(10)
        colors = plt.cm.plasma(
            np.linspace(0.2, 1, len(vc)))
        axes[0,0].barh(vc.index, vc.values, color=colors)
        axes[0,0].set_title(f'{cat_cols[0]} Distribution',
                            color='cyan', fontsize=11)

    if len(cat_cols) > 1:
        vc2 = df[cat_cols[1]].value_counts().head(8)
        axes[0,1].pie(vc2.values,
                      labels=vc2.index,
                      autopct='%1.1f%%',
                      textprops={'color':'white'},
                      colors=plt.cm.plasma(
                          np.linspace(0.2,1,len(vc2))))
        axes[0,1].set_title(f'{cat_cols[1]} Distribution',
                            color='cyan', fontsize=11)

    if num_cols:
        axes[1,0].hist(df[num_cols[0]].dropna(),
                       bins=30, color='cyan',
                       alpha=0.8, edgecolor='#333')
        axes[1,0].set_title(f'{num_cols[0]} Distribution',
                            color='cyan', fontsize=11)

    # Dataset info
    axes[1,1].axis('off')
    info = [
        ('Total Records', str(len(df))),
        ('Total Columns', str(len(df.columns))),
        ('Numeric Cols',  str(len(num_cols))),
        ('Text Cols',     str(len(cat_cols))),
        ('Missing Values',str(df.isnull().sum().sum())),
    ]
    y = 0.9
    for label, val in info:
        axes[1,1].text(0.1, y, label,
                      transform=axes[1,1].transAxes,
                      color='#aaaaaa', fontsize=12)
        axes[1,1].text(0.6, y, val,
                      transform=axes[1,1].transAxes,
                      color='cyan', fontsize=12,
                      fontweight='bold')
        y -= 0.15

    plt.tight_layout()
    plt.savefig('eda_animals_report.png', dpi=120,
                bbox_inches='tight', facecolor='#0a0a0a')
    print("EDA saved: eda_animals_report.png")
    return fig, df

# ══════════════════════════════════════════════════════════
# SECTION 2 — CUSTOM DATASET
# ══════════════════════════════════════════════════════════

class MathPatternDataset(Dataset):
    def __init__(self, root_path, transform=None):
        self.samples   = []
        self.labels    = []
        self.classes   = []
        self.transform = transform

        root = Path(root_path)
        for idx, class_dir in enumerate(sorted(root.iterdir())):
            if not class_dir.is_dir():
                continue
            self.classes.append(class_dir.name)
            files = (list(class_dir.glob("*.jpg")) +
                     list(class_dir.glob("*.png")) +
                     list(class_dir.glob("*.jpeg")))
            for f in files:
                self.samples.append(str(f))
                self.labels.append(idx)

        print(f"Dataset loaded: {len(self.samples)} images, "
              f"{len(self.classes)} classes")
        print(f"Classes: {self.classes}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# ══════════════════════════════════════════════════════════
# SECTION 3 — TRANSFER LEARNING TRAINING
# ══════════════════════════════════════════════════════════

def train_pattern_model(dataset_path=None,
                        epochs=10,
                        batch_size=32,
                        lr=0.001):
    """
    Transfer Learning — EfficientNet B0
    Flowers dataset pe train karo math patterns ke liye
    """
    if dataset_path is None:
        dataset_path = FLOWERS_PATH

    print(f"\nTraining on: {dataset_path}")
    print(f"Epochs: {epochs} | Batch: {batch_size} | LR: {lr}")

    # Transforms
    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3,
                               contrast=0.3,
                               saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])

    # Full dataset
    full_ds = MathPatternDataset(dataset_path, train_tf)
    n_classes = len(full_ds.classes)

    if len(full_ds) == 0:
        print("No images found! Check dataset path.")
        return None, None, []

    # Train/val split 80/20
    val_size   = int(0.2 * len(full_ds))
    train_size = len(full_ds) - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        full_ds, [train_size, val_size])

    val_ds.dataset = MathPatternDataset(dataset_path, val_tf)

    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)
    val_loader   = DataLoader(val_ds,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0)

    # Model — EfficientNet B0
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT)
    
    # Freeze base layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.classifier[1].in_features, 256),
        nn.ReLU(),
        nn.Dropout(p=0.2),
        nn.Linear(256, n_classes)
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.classifier.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.5)

    # Training loop
    history = {'train_loss': [], 'train_acc': [],
                'val_loss':   [], 'val_acc':   []}

    best_val_acc = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss    += loss.item()
            preds          = outputs.argmax(1)
            train_correct += (preds == labels).sum().item()
            train_total   += labels.size(0)

        # Validate
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs  = model(imgs)
                loss     = criterion(outputs, labels)
                val_loss += loss.item()
                preds     = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total   += labels.size(0)

        t_acc = train_correct / train_total * 100
        v_acc = val_correct   / val_total   * 100
        t_l   = train_loss    / len(train_loader)
        v_l   = val_loss      / len(val_loader)

        history['train_loss'].append(t_l)
        history['train_acc'].append(t_acc)
        history['val_loss'].append(v_l)
        history['val_acc'].append(v_acc)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {t_l:.4f} | "
              f"Train Acc: {t_acc:.1f}% | "
              f"Val Loss: {v_l:.4f} | "
              f"Val Acc: {v_acc:.1f}%")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(),
                       'models/best_model.pth')
            print(f"  Best model saved! Val Acc: {v_acc:.1f}%")

        scheduler.step()

    print(f"\nTraining Complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.1f}%")

    return model, history, full_ds.classes

def plot_training_history(history):
    """Training curves plot karo"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0a0a0a')
    fig.suptitle('AlgebraicVision AI — Training History',
                 color='gold', fontsize=14, fontweight='bold')

    for ax in axes:
        ax.set_facecolor('#0d0d0d')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('#333')

    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'],
                 color='cyan', linewidth=2, label='Train Loss')
    axes[0].plot(epochs, history['val_loss'],
                 color='magenta', linewidth=2, label='Val Loss')
    axes[0].set_title('Loss Curves', color='cyan', fontsize=12)
    axes[0].legend(facecolor='#111', labelcolor='white')
    axes[0].set_xlabel('Epoch', color='white')
    axes[0].set_ylabel('Loss',  color='white')

    axes[1].plot(epochs, history['train_acc'],
                 color='lime', linewidth=2, label='Train Acc')
    axes[1].plot(epochs, history['val_acc'],
                 color='gold', linewidth=2, label='Val Acc')
    axes[1].set_title('Accuracy Curves',
                      color='cyan', fontsize=12)
    axes[1].legend(facecolor='#111', labelcolor='white')
    axes[1].set_xlabel('Epoch',    color='white')
    axes[1].set_ylabel('Accuracy %', color='white')

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=120,
                bbox_inches='tight', facecolor='#0a0a0a')
    print("Training history saved: training_history.png")
    return fig

def predict_image(model, image_path, classes, device=None):
    """Trained model se image predict karo"""
    if device is None:
        device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225]),
    ])

    img    = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        probs   = torch.softmax(outputs, dim=1)[0]
        pred    = probs.argmax().item()

    return classes[pred], float(probs[pred]) * 100

# ══════════════════════════════════════════════════════════
# MAIN — Run Everything
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    import os
    os.makedirs('models', exist_ok=True)

    print("=" * 60)
    print("  AlgebraicVision AI — EDA + Training Pipeline")
    print("=" * 60)

    # Step 1 — EDA
    print("\n[1/4] Flowers EDA...")
    df_flowers, samples = eda_flowers()
    print(df_flowers[['Class','Count',
                       'Avg Symmetry %',
                       'Avg Fractal Dim',
                       'Golden Ratio Score']].to_string())
    fig1 = plot_eda_flowers(df_flowers, samples)
    plt.show()
    plt.close()

    print("\n[2/4] Intel Images EDA...")
    fig2, df_intel = eda_intel_images()
    plt.show()
    plt.close()

    print("\n[3/4] Animals CSV EDA...")
    fig3, df_animals = eda_animals_csv()
    plt.show()
    plt.close()

    # Step 2 — Training
    print("\n[4/4] Training Pattern Recognition Model...")
    model, history, classes = train_pattern_model(
        dataset_path=FLOWERS_PATH,
        epochs=10,
        batch_size=32,
        lr=0.001
    )

    if history:
        fig4 = plot_training_history(history)
        plt.show()
        plt.close()
        print("\nAll done! Check these files:")
        print("  eda_flowers_report.png")
        print("  eda_intel_report.png")
        print("  eda_animals_report.png")
        print("  training_history.png")
        print("  models/best_model.pth")
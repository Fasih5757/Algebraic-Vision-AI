# AlgebraicVision AI — AI-Powered Mathematical Pattern Recognition & Visualization System

**Course Title:** Artificial Intelligence
**Submitted by:** Fasih Ud Din
**Roll Number:** BSITMA-23-09

An AI-driven mathematical visualization platform that combines **Computer Vision, Deep Learning, Symbolic Mathematics, and Generative AI** to discover hidden mathematical patterns in nature and transform abstract equations into interactive visual experiences.

---

# 📁 Project Structure

```text
Algebraic-Vision-AI/
├── app.py                      # Streamlit application entry point
│
├── modules/                    # Core AI and mathematics modules
│   ├── ai_engine.py            # AI-generated explanations
│   ├── equation_parser.py      # Equation analysis & visualization
│   ├── yolo_detector.py        # Object detection & pattern overlays
│   ├── dataset_trainer.py      # Model training pipeline
│   ├── symmetry.py             # Symmetry analysis algorithms
│   ├── fractals.py             # Fractal generation & analysis
│   ├── golden_ratio.py         # Golden ratio detection
│   ├── fourier.py              # Fourier visualizations
│   ├── epicycles.py            # Epicycle animations
│   └── parametric.py           # Parametric curve generation
│
├── models/                     # Trained deep learning models
│
├── datasets/                   # Training and analysis datasets
│   ├── Flowers Dataset
│   ├── Intel Images Dataset
│   ├── Animals Dataset
│   └── Fractal Datasets
│
├── assets/                     # Images, icons, visual resources
│
├── docs/                       # Project documentation
│   ├── Project_Report.pdf
│   └── Presentation.pptx
│
├── README.md
└── requirements.txt
```

---

# 🚀 Features

### 🌻 Flower Classification

* EfficientNet-B0 based deep learning model
* Classifies:

  * Daisy
  * Dandelion
  * Rose
  * Sunflower
  * Tulip
* Achieved **90.2% validation accuracy**

### 📈 Equation Visualizer

* Converts equations into interactive plots
* Supports:

  * Explicit Functions
  * Implicit Functions
  * Parametric Curves
  * Polar Equations
  * 3D Surface Graphs
* Automatic derivative and integral generation

### 🖼️ Mathematical Pattern Detection

* Detects:

  * Fibonacci Spirals
  * Golden Ratio
  * Bilateral Symmetry
  * Radial Symmetry
  * Fractal Structures
* Generates visual overlays on uploaded images

### 📊 Image Analysis Dashboard

* Edge Detection
* RGB Histograms
* Heatmaps
* Symmetry Maps
* Statistical Analysis Reports

### 🤖 AI Explanation Engine

* Generates scientific explanations of detected patterns
* Converts mathematical concepts into simple English
* Powered by LLM-based AI models

### 🎨 Advanced Mathematical Visualizations

* Fourier Series
* Epicycles
* Parametric Curves
* Mandelbrot Set
* Julia Set
* Sierpinski Triangle
* Koch Snowflake

---

# ⚙️ Local Setup

## Clone Repository

```bash
git clone https://github.com/Fasih5757/Algebraic-Vision-AI.git
cd Algebraic-Vision-AI
```

## Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key_here
```

## Run Application

```bash
streamlit run app.py
```

Open:

```text
http://localhost:3000
```

---

# 🎥 Demo Video

Click here to watch the demo video

---

# 📦 Tech Stack

| Layer                | Technology               |
| -------------------- | ------------------------ |
| AI/ML                | PyTorch, EfficientNet-B0 |
| Computer Vision      | OpenCV, YOLOv8           |
| Mathematics          | SymPy, NumPy             |
| Visualization        | Matplotlib, Plotly       |
| Web Framework        | Streamlit                |
| AI Explanations      | Groq API (Llama Models)  |
| Data Processing      | Pandas                   |
| Scientific Computing | NumPy                    |

---

# 📊 Performance Metrics

| Metric                         | Result      |
| ------------------------------ | ----------- |
| Flower Classification Accuracy | 90.2%       |
| Fibonacci Detection Accuracy   | 94%         |
| Symmetry Analysis Accuracy     | 85%         |
| Fractal Dimension MAE          | 0.08        |
| Image Analysis Time            | 3–4 Seconds |
| Equation Rendering Time        | < 1 Second  |

---

# 🧠 Mathematical Concepts Implemented

### Fibonacci Sequence

* Natural growth pattern detection
* Sunflower spiral analysis

### Golden Ratio (φ = 1.618)

* Proportion detection
* Nature and artwork analysis

### Symmetry Analysis

* Bilateral symmetry
* Radial symmetry

### Fractal Geometry

* Fractal dimension calculation
* Recursive pattern generation

### Fourier Series

* Signal decomposition visualization
* Epicycle-based curve reconstruction

---

# 🔒 Security & Best Practices

* Environment variables stored in `.env`
* API keys excluded using `.gitignore`
* Modular architecture for maintainability
* Separated AI, visualization, and analysis modules
* Dataset and model management through dedicated directories

---

# 🎯 Future Enhancements

* Mobile Application
* Real-Time Camera Pattern Detection
* Augmented Reality Mathematics Overlay
* Video-Based Pattern Analysis
* Explainable AI (Grad-CAM)
* Multi-Model Ensemble Learning
* Research Publication & Open Source Release

---

# 👨‍💻 Author

**Fasih Ud Din**
BS Information Technology
University of Layyah
Artificial Intelligence Project – 2026

GitHub:
https://github.com/Fasih5757

---

# 📜 License

## License

This project is licensed under the MIT License. See the LICENSE file for details.

Copyright © 2026 Fasih Ud Din

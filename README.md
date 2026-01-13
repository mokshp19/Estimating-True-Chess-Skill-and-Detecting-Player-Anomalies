# [Estimating-True-Chess-Skill-and-Detecting-Player-Anomalies](https://github.com/Jaathavan/Estimating-True-Chess-Skill-and-Detecting-Player-Anomalies)
Python project that analyzes chess games to estimate player rating and flag unusual play patterns that may indicate cheating

A comprehensive machine learning system for analyzing chess games to estimate true player skill (ELO rating) and detect anomalies such as cheating and smurfing using machine learning and hybrid anomaly detection.

**GROUP ID:** 11

**Members:** Jaathavan Ranjanathan, Moksh Patel

---

## Project Presentation

- **Video Presentation**: [Watch on YouTube](https://www.youtube.com/watch?v=5R4GBwhJUhg&feature=youtu.be)
- **Slides**: [View on Google Slides](https://docs.google.com/presentation/d/1_zi45VbKCGjdcXb8gMZnzrU2gzFsKoMfwZEex3tmBbY/edit?usp=sharing)

---

## Project Overview

This project implements a **two-phase machine learning pipeline**:

### **Phase 1: ELO Prediction**
- Predicts a player's true skill level from game features
- Uses MLP neural networks with residual connections
- Trained on 80,000+ games from Lichess and Chess.com

### **Phase 2: Anomaly Detection (Cheating/Smurfing)**
- **Hybrid System** combining:
  - **Autoencoder** (unsupervised) - learns normal play patterns
  - **Classifier** (supervised) - trained on labeled cheating data
- Detects suspicious behaviour with AUC 0.987
- Provides actionable recommendations (Ban/Review/Normal)

---

## Project Structure

```
Estimating-True-Chess-Skill-and-Detecting-Player-Anomalies/
│
├── elo_prediction/                          # Phase 1: ELO Estimation
│   ├── data/
│   │   ├── Chess Game Dataset (Lichess).csv
│   │   └── 60,000+ Chess Game Dataset (Chess.com).csv
│   ├── phase1_elo_prediction.ipynb          # Training notebook
│   ├── best_elo_model_v3.pt                 # Trained model
│   └── phase1_elo_results.npz               # Evaluation results
│
├── cheating_smurfing_detection/             # Phase 2: Anomaly Detection
│   ├── data/
│   │   ├── Cheating Tuesdays                # Labeled cheating data
│   │   └── Spotting Cheaters.csv            # Additional cheating data
│   ├── phase2_anomaly_detection.ipynb       # Training notebook
│   ├── best_autoencoder_v2.pt               # Autoencoder model
│   ├── best_classifier_v2.pt                # Classifier model
│   └── phase2_anomaly_results_v2.npz        # Results & config
│
├── deployment_demo/                          # Production-Ready Deployment
│   ├── run_analysis.py                      # Main analysis script
│   ├── best_elo_model_v3.pt                 # Phase 1 model
│   ├── best_autoencoder_v2.pt               # Phase 2 autoencoder
│   ├── best_classifier_v2.pt                # Phase 2 classifier
│   ├── phase2_deployment_config.pkl         # Hybrid system config
│   └── sample_game.pgn                      # Normal game example
│
└── README.md                                 # This file
```

---

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Estimating-True-Chess-Skill-and-Detecting-Player-Anomalies.git
cd Estimating-True-Chess-Skill-and-Detecting-Player-Anomalies

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install torch numpy pandas matplotlib seaborn scikit-learn python-chess
```

### 2. Run Training (Optional)

```bash
# Phase 1: Train ELO predictor
cd elo_prediction
jupyter notebook phase1_elo_prediction.ipynb

# Phase 2: Train anomaly detector
cd ../cheating_smurfing_detection
jupyter notebook phase2_anomaly_detection.ipynb
```

### 3. Use Deployment System

```bash
cd deployment_demo

# Analyze a chess game
python run_analysis.py sample_game.pgn

# Analyze specific player
python run_analysis.py your_game.pgn --color white
python run_analysis.py your_game.pgn --color black

# Test with suspicious games
python run_analysis.py suspected_cheater.pgn
python run_analysis.py obvious_cheater.pgn
```

---

## Example Output

```
======================================================================
CHESS GAME ANALYSIS
======================================================================

Game Information:
  White: SuspiciousPlayer (1650)
  Black: RegularPlayer (1850)
  Result: 1-0
  Event: Online Blitz

Analyzing: WHITE player

======================================================================
PHASE 1: ELO PREDICTION
======================================================================

Estimated ELO: 2150
   Stated ELO:    1650
   Difference:    +500 points    SUSPICIOUS!

======================================================================
PHASE 2: CHEATING/SMURFING DETECTION
======================================================================

Scores:
  Autoencoder: 68.4%
  Classifier:  91.2%
  Hybrid:      84.3%
  Confidence:  HIGH

DECISION:
  BAN THE USER
  High confidence cheating detected (84.3%)

======================================================================
```

---

## Methodology

### Phase 1: ELO Prediction

**Architecture:**
```
Input (87 features) → 128 neurons → 64 neurons → 64 neurons → Output (ELO)
                      ↓ Residual Connection ↓
```

**Key Features:**
- Opponent rating (most important!)
- Game outcome (win/loss/draw)
- Move statistics (captures, checks, castling)
- Opening information (ECO code)
- Time control
- Move quality indicators

**Training:**
- Dataset: 80,000+ games
- Loss: MSE
- Optimizer: AdamW with weight decay
- Regularization: Dropout (0.2), Batch Normalization
- Early stopping with patience=10

**Performance:**
- Test MAE: 81.29 ELO points
- Test RMSE: 133.06 ELO points
- R² Score: 0.8635

<img width="1488" height="490" alt="image" src="https://github.com/user-attachments/assets/54f4d18e-7e7c-4211-afc1-42774f2dab16" />
<img width="1390" height="490" alt="image" src="https://github.com/user-attachments/assets/1b5e66fb-0508-489f-aaa0-26925087c27d" />

---


### Phase 2: Anomaly Detection

**Hybrid Architecture:**

```
                    ┌─────────────────┐
                    │  Game Features  │
                    │   (24 features) │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
    ┌─────────────────┐         ┌──────────────────┐
    │  Autoencoder    │         │   Classifier     │
    │  (Unsupervised) │         │   (Supervised)   │
    │                 │         │                  │
    │  Reconstruction │         │  Binary Output   │
    │      Error      │         │   [0, 1]        │
    └────────┬────────┘         └────────┬─────────┘
             │                           │
             │ (1-α) weight             │ α weight
             │                           │
             └───────────┬───────────────┘
                         ▼
                 ┌──────────────┐
                 │ Hybrid Score │
                 │              │
                 │ α·clf + (1-α)·ae │
                 └──────────────┘
```

**Key Features:**
- Cheat percentage & streaks
- Move accuracy statistics
- Top-N engine move matches
- Time usage patterns
- Rating vs performance
- Game phase analysis


**Performance:**
![Phase 2 Training Loss](cheating_smurfing_detection/phase2_training_history_v2.png)
*Autoencoder and Classifier loss curves during training*

![ROC Curves Comparison](cheating_smurfing_detection/phase2_roc_curves_v2.png)
*ROC curves comparing Autoencoder, Classifier, and Hybrid approaches*

![Confusion Matrix](cheating_smurfing_detection/phase2_confusion_matrices_v2.png)
*Hybrid system confusion matrix on test set*

![Score Distributions](cheating_smurfing_detection/phase2_score_distributions_v2.png)
*Distribution of anomaly scores for normal vs cheating players*

**Decision Thresholds:**
| Hybrid Score | Confidence | Action |
|--------------|-----------|--------|
| ≥85% | HIGH | Ban user immediately |
| 60-84% | MEDIUM | Flag for manual review |
| <60% | LOW | Normal - log game |


## Technical Details

### Model Architectures

**Phase 1: OptimizedELOPredictor**
```python
class OptimizedELOPredictor(nn.Module):
    - Input: 87 features (after one-hot encoding)
    - Hidden: [128, 64, 64] with residual connection
    - Dropout: 0.2
    - Batch Normalization on all layers
    - Activation: ReLU
    - Output: Single value (ELO)
```

**Phase 2: ImprovedAutoencoder**
```python
class ImprovedAutoencoder(nn.Module):
    - Input: 24 features
    - Encoding: 64 → 32 → 16 → 8 (bottleneck)
    - Decoding: 8 → 16 → 32 → 64 → 24
    - Dropout: 0.2
    - Activation: LeakyReLU (0.2)
```

**Phase 2: ImprovedCheatDetector**
```python
class ImprovedCheatDetector(nn.Module):
    - Input: 24 features
    - Hidden: [128, 64, 32, 16]
    - Dropout: 0.3
    - Activation: LeakyReLU (0.2)
    - Output: Binary (cheating probability)
```

---

## Datasets Used

### Phase 1 Training Data

| Dataset | Source | Games | Rating Range |
|---------|--------|-------|--------------|
| Lichess | Kaggle | 20,000 | 800-2800 |
| Chess.com | Kaggle | 60,000 | 400-3000 |
| **Total** | - | **80,000** | **400-3000** |

https://www.kaggle.com/datasets/datasnaek/chess

https://www.kaggle.com/datasets/adityajha1504/chesscom-user-games-60000-games

### Phase 2 Training Data

| Dataset | Source | Games | Labels |
|---------|--------|-------|--------|
| Cheating Tuesdays | Kaggle | 161,000 | Cheating |
| Spotting Cheaters | Kaggle | 48,000 | Mixed |

https://www.kaggle.com/datasets/pavelgonchar/cheating-tuesdays

https://www.kaggle.com/datasets/brieucdandoy/chess-cheating-dataset

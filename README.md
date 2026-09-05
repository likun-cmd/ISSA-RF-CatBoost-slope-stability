ISSA-RF-CatBoost for Slope Safety Factor Prediction
This repository contains the code and dataset for predicting slope safety factors using an ISSA-optimized stacking ensemble model combining Random Forest (RF), XGBoost, and CatBoost.

This work accompanies the paper:

Kun Li, Dengfeng Su, Baohui Tan, Yingpeng Hu, Li Zhao, Shang Gao, Qiang Zhu, Xinxin Hou. Research on landslide early warning model driven by ISSA optimization integrated machine learning algorithm.

Project Purpose
The goal of this project is to develop a robust machine learning framework for predicting the factor of safety (FOS) of slopes based on six input geotechnical parameters:

Unit weight, γ (kN/m³)

Cohesion, c (kPa)

Internal friction angle, φ (°)

Slope angle, α (°)

Slope height, H (m)

Pore-water pressure ratio, r_u

The framework employs:

Improved Sparrow Search Algorithm (ISSA) for hyperparameter optimization of base learners

Stacking ensemble strategy (RF + XGBoost + CatBoost with MLP as meta-learner) to improve prediction accuracy and stability

10 repeated random train-test splits and Wilcoxon signed-rank tests for statistically robust model evaluation

Repository Structure
text
ISSA-RF-CatBoost-slope-stability/
│
├── code/
│   ├── data_preprocessing.py      # Data cleaning and normalization
│   ├── rf_xgb_catb.py             # Individual model training (RF, XGB, CatB)
│   ├── issa_rf_xgb_catb.py          
│   ├── issa_rf_xgb_cab_stacking.py # Stacking ensemble implementation
│
├── data/
│   ├── raw_data.xlsx              # Original 320 slope cases with references
│   └── processed_data.csv         # Preprocessed dataset for modeling
│
├── requirements.txt               # Python dependencies
├── LICENSE                        # MIT License
└── README.md                      # This file
Dependencies and Installation
This project is implemented in Python 3.8+. To install the required packages, run:

bash
pip install -r requirements.txt
Main dependencies:
Package	Version (tested)
Python	3.8+
numpy	1.21.0+
pandas	1.3.0+
scikit-learn	1.0.0+
xgboost	1.5.0+
catboost	1.0.0+
scipy	1.7.0+
matplotlib	3.4.0+
openpyxl	3.0.9+
How to Run the Code
1. Clone the repository
bash
git clone https://github.com/likun-cmd/ISSA-RF-CatBoost-slope-stability.git
cd ISSA-RF-CatBoost-slope-stability
2. Prepare the data
Place your raw data in data/raw_data.xlsx or use the provided dataset.

Run preprocessing:
bash
python data_preprocessing.py
bash
python rf_xgb_catb.py
bash
python issa_rf_xgb_catb.py
bash
python issa_rf_xgb_cab_stacking.py
Train the stacking ensemble model
Note: All scripts are designed to run sequentially. Intermediate results (e.g., trained models, prediction outputs) will be saved in the results/ folder (auto-created).
## Results

The proposed ISSA-RF-CatBoost model achieved the following performance:

| Metric | Value |
|--------|-------|
| MAE | 0.0604 ± 0.0148 |
| MSE | 0.0189 ± 0.0104 |
| RMSE | 0.1299 ± 0.0281 |
| R² | 0.8808 ± 0.0665 |
Dataset
The dataset used in this study comprises 320 slope cases compiled from published literature. It includes:

Six input features: γ, c, φ, α, H, r_u

Target variable: Factor of Safety (FOS)

The dataset is provided in both original (raw_data.xlsx) and preprocessed (processed_data.csv) formats in the data/ folder. Detailed source references for each case are included in the dataset file.

License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
This work was supported by the 2025 Sichuan Provincial Department of Natural Resources Research Projects (No. KJ-2025-029) and the Natural Science Foundation of the Southwest University of Science and Technology (No. 21zx7157; No. 21zx7161).

We thank the researchers whose published slope cases made this dataset possible.

Contact
For questions or issues, please open an issue on GitHub or contact:

Dengfeng Su: dengfengcqu@163.com

Kun Li: likunqqemail@qq.com


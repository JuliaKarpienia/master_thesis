# Retail Decision-Support Recommender System

**Master Thesis Project** — An internal recommender system for a traditional offline retail store selling building materials and renovation products. The system uses historical point-of-sale (POS) transactional data (2022–2025) to discover product relationships, cross-selling opportunities, and buying patterns. A Large Language Model (LLM) layer translates analytical outputs into plain-language insights for non-technical store managers.

---

## Research Questions

- **RQ1:** How can a recommender system be designed to effectively use sparse transaction data in a traditional retail environment?
- **RQ2:** How can recommendation outputs be made interpretable for marketing and operational decision makers?
- **RQ3:** Can LLMs bridge the gap between complex recommendation data and practical managerial decisions?

---

## Project Structure

```
master/
├── 1. Data Preprocessing & EDA.ipynb     # Data cleaning, EDA, feature engineering, RFM, K-Means
├── 2. Recommendation Models.ipynb        # Popularity baseline, Item-CF, Association Rules + Evaluation
├── 3. Final Model.ipynb                  # Hybrid model + LLM explanation layer
├── 4. Streamlit Demo App.ipynb           # Installs deps, verifies data files, launches app.py
├── app.py                                # Streamlit interactive demo application
├── .gitignore
├── requirements.txt                      # Package list
├── data/
│   ├── raw/                              # Original POS exports (2022–2025, B1 & B2 half-years)
│   │   ├── 1-2022.csv
│   │   ├── 2-2022.csv
│   │   └── ...
│   └── preprocessed/                     # Generated artefacts (created by running the notebooks)
│       ├── cleaned_retail_data_2022_2025.csv
│       ├── rfm_base.csv
│       ├── b2b_association_rules.csv
│       ├── b2c_association_rules.csv
│       ├── user_item_matrix.pkl
│       ├── basket_matrix_b2b.pkl
│       ├── basket_matrix_b2c.pkl
│       ├── item_similarity_matrix.pkl
│       └── model_evaluation_results.csv
├── plots/                                # All saved EDA & model visualisations
└── README.md
```

---

## Methodology

The project follows the **Design Science Research (DSR)** framework and implements a six-stage pipeline:

| Stage | Description |
|---|---|
| 1 | Data Preprocessing & Cleaning |
| 2 | Construction of User-Item and Basket Matrices |
| 3 | Model Training & Validation |
| 4 | Generation of Recommendation Outputs |
| 5 | Transformation of outputs into structured LLM prompts |
| 6 | LLM-based natural-language explanation generation |

### Customer Split
The dataset distinguishes between three customer types in the app:
- **B2B** — known contractors identified by name (Item-Based Collaborative Filtering + Association Rules + RFM segmentation)
- **B2C** — anonymous walk-in customers recorded as `SPRZEDAZ FISKALNA` (B2C Association Rules + popularity fallback only)
- **New Customer** — first-time visitor with no purchase history (treated identically to B2C; manager can optionally enter a name)

---

## Models

### 1. Data Preprocessing & EDA (`Notebook 1`)
- Loads and merges 8 raw POS CSV files (Polish encoding, 2022–2025)
- Translates Polish column names, fixes number formatting, parses dates
- Removes returns (negative quantities/values), winsorizes outliers at the 99th percentile
- **EDA** — boxplots, top-product and top-customer charts, monthly sales trend
- **RFM Segmentation** — Recency, Frequency, Monetary calculated for all known B2B customers
- **K-Means Clustering** — elbow method (K=2–8) followed by a K=4 model that labels customers as `Champions`, `High-Value`, `Regular`, or `Inactive`
- Builds and exports:
  - `user_item_matrix` (customers with ≥ 2 invoices, to reduce sparsity noise)
  - `basket_matrix_b2b` and `basket_matrix_b2c` (boolean invoice–product matrices)
  - `rfm_base.csv` (with segment labels)

### 2. Recommendation Models (`Notebook 2`)
- **Popularity Baseline** — top products by unique invoice count (avoids bulk-buy skew)
- **Item-Based Collaborative Filtering** — cosine similarity on the user-item matrix
- **Association Rule Mining (Apriori)** — separate rule sets for B2B and B2C; `min_support=0.5%`, sorted by lift then confidence
- **Leave-One-Out Evaluation** — holds out each B2B customer's most recent invoice, evaluates all four models at K=5 using **Precision@K**, **Recall@K**, and **NDCG@K**; results saved to `model_evaluation_results.csv`

### 3. Hybrid Model + LLM Layer (`Notebook 3`)
- **Three-tier recommendation logic:**
  1. Item-Based CF (B2B only)
  2. Association Rules (B2B uses B2B rules; B2C and New Customers use B2C rules)
  3. Popularity fallback (when no signals are found)
- Items already in the basket are excluded from all recommendation lists
- **RFM Profile** injected into the LLM prompt (days since last purchase, invoice count, lifetime spend, segment label)
- **LLM integration** — calls the Mistral model locally via Ollama to convert the recommendation dict into a plain-language explanation for the floor manager; gracefully falls back to printing the prompt if the local Ollama server is unavailable

### 4. Streamlit Demo App (`app.py` + `Notebook 4`)
- Interactive web application for real-time recommendation demonstration
- **Sidebar:** customer type selector (B2B / B2C / New Customer), product basket builder
- **Main panel:** customer header with RFM metrics (B2B), AI Manager Summary (Mistral via Ollama explanation), recommendation tabs (Market Basket · Similar Contractors · Store Best-Sellers)
- Ollama must be running locally (`http://localhost:11434`)
- `Notebook 4` installs dependencies, verifies data files, runs a smoke-test, and launches the app

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama (for the LLM layer)

Install [Ollama](https://ollama.com/) and pull the Mistral model:

```bash
ollama pull mistral
```

Ensure Ollama is running before launching the app. The app connects to `http://localhost:11434/v1`.

### 3. Run the Notebooks in Order

```
1. Data Preprocessing & EDA.ipynb   →  generates preprocessed/ artefacts
2. Recommendation Models.ipynb      →  trains models, exports rules & similarity matrix, runs evaluation
3. Final Model.ipynb                →  hybrid recommendations + LLM explanations
4. Streamlit Demo App.ipynb         →  installs deps, verifies files, launches app.py
```

> **Note:** The raw POS data files in `data/raw/` are not included in this repository as they contain confidential business information. The notebooks expect 8 CSV files named `{half}-{year}.csv` (e.g. `1-2022.csv`, `2-2023.csv`) inside `data/raw/`.

---

## Author

Julia Karpienia — Master Thesis, 2025/2026

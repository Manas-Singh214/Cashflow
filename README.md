# 🛡️ Predictive Cash-Withdrawal & Cybercrime Analytics Pipeline

## 📌 Project Overview
This project ingests, cleans, merges, and models multi-source banking, socio-economic, and cybercrime datasets across India. The objective is to build an analytical foundation to analyze relationships between banking infrastructure (ATMs, PoS, Cards), socio-economic indicators (GDP, literacy, unemployment), and cyber/financial fraud incidents — paving the way for proactive geospatial hotspot forecasting.

---

## ⚙️ Virtual Environment Setup & Installation

### Option 1: Automatic Setup (One-Click / One Command)
- **Windows Command Prompt / Double-click**: Run [`setup_venv.bat`](file:///d:/Projects/Data%20Analyst%20Portfolio/Complete%20DA/setup_venv.bat)
- **PowerShell**:
  ```powershell
  .\setup_venv.ps1
  ```

### Option 2: Manual Setup

1. **Create the virtual environment**:
   ```bash
   python -m venv .venv
   ```

2. **Activate the virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt / CMD)**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - **macOS / Linux**:
     ```bash
     source .venv/bin/activate
     ```

3. **Install all dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 📚 Tech Stack & Library Ecosystem

The project dependencies in [`requirements.txt`](file:///d:/Projects/Data%20Analyst%20Portfolio/Complete%20DA/requirements.txt) are organized into specialized layers:

### 1. Data Ingestion & Extraction
- **`pdfplumber`**: High-precision tabular data parsing from threat advisories and PDF reports.
- **`requests`** & **`beautifulsoup4`**: Automated API fetching and HTML scraping for public banking locator portals.

### 2. Core Processing & OLAP Engines
- **`polars`**: High-throughput, multi-threaded columnar processing for large complaint logs.
- **`duckdb`**: In-process SQL OLAP engine for zero-overhead joins and windowed aggregations.
- **`pandas`** & **`numpy`**: Standard DataFrame manipulation, schema enforcement, and numerical ops.
- **`pyarrow`**: Fast Apache Parquet read/write backend for partitioned storage.
- **`openpyxl`** / **`xlrd`**: Modern and legacy Excel file parsing.

### 3. Geospatial & Spatial Indexing
- **`geopandas`** & **`shapely`**: Vector spatial data manipulation, geometry creation, and bounding calculations.
- **`osmnx`**: Road network routing and drive-time isochrones around high-risk ATMs.
- **`h3`**: Uber's discrete global hierarchical hexagonal spatial indexing system.
- **`pydeck`** & **`folium`**: High-performance 3D spatial Deck.gl maps and interactive Leaflet visualizations.

### 4. NLP, Visualization & Data Integrity
- **`spacy`**: Named Entity Recognition (NER) for parsing unstructured incident reports.
- **`matplotlib`**, **`seaborn`**, **`plotly`**, **`hvplot`**: Multi-dimensional static and interactive charts.
- **`pygwalker`**: Embedded Tableau-like interactive data explorer directly inside Jupyter Notebooks.
- **`great-expectations`**: Automated data validation and schema integrity checks.

---

## 🧩 Data Cleaning & Merging Pipeline (Non-Technical Gist)

Raw data comes from multiple distinct sources (RBI statistics, state crime bureaus, and parliament records). Because these systems use differing conventions, the pipeline performs three key functions:

### 1. Name Standardization
- Harmonizes bank names (e.g., `"State Bank of India LTD."`, `"STATE BANK OF INDIA"`, `"SBI"`) into matching canonical keys.
- Normalizes state and union territory naming differences (e.g., `"J&K"` vs. `"JAMMU & KASHMIR"`).

### 2. Filtering & Type Validation
- Removes duplicate rows, missing identifiers, zero/negative transaction values, and summary rows (such as `"ALL INDIA"` or `"TOTAL"`).
- Converts raw strings into typed timestamps, dates, and float values.

### 3. Aggregation & Relational Merging
- Computes bank-wide monthly averages for ATM/card infrastructure and joins them to yearly balance sheets.
- Joins district/state socio-economic variables with cybercrime incident figures.

---

## 🗂️ Processed Datasets (Pipeline Outputs)

The pipeline produces **4 clean datasets** in [`MERGED DATASETS/`](file:///d:/Projects/Data%20Analyst%20Portfolio/Complete%20DA/MERGED%20DATASETS):

| Output File | Key Sources | Contents |
| :--- | :--- | :--- |
| **`merged_bank_financials.csv`** | *Indian Banks Data v2.0* + *RBI ATM/Card Stats* | Bank financials (Deposits, Advances, NPAs, ROA) combined with ATM counts, PoS terminals, QR codes, and card stats. |
| **`bank_transactions_clean.csv`** | *Bank Transactions Dataset* | 1M+ cleaned individual transactions with validated DOBs, locations, balances, and positive amounts. |
| **`merged_crime_data.csv`** | *India Crime & Socioeconomic Data* + *NCRB Data* + *Parliament Session Reports* | State-level socioeconomic metrics (GDP, literacy, poverty, unemployment) merged with violent, property, and cybercrime incident metrics. |
| **`cyber_national_trends.csv`** | *Parliament Questions (RS Session 267)* | National macro trends of cyber incidents, defrauded amounts, and reported fraud volumes. |

---

## 📈 Project Progress & Roadmap

- [x] **Phase 1: Ingestion & Pipeline Setup**
  - Downloaded and organized raw datasets into `DATASETS/Bank Records` and `DATASETS/Crime Records`.
  - Built and formatted `data_merge.py` for automated normalization, cleaning, and merging.
  - Generated the 4 baseline master datasets in `MERGED DATASETS/`.
  - Established project environment and dependency specifications in `requirements.txt`.
- [ ] **Phase 2: Exploratory Data Analysis (EDA) & Feature Engineering**
  - Analyze correlations between ATM density, digital transactions, and cyber fraud incidents.
  - Calculate state-level financial recovery ratios and vulnerability indices.
- [ ] **Phase 3: Geospatial Analytics & Dashboards**
  - Map regional risk densities with H3 hex-binning and Pydeck / Folium.
  - Construct an interactive BI dashboard (Power BI / Tableau / PyGWalker).
- [ ] **Phase 4: Predictive Modeling & Hotspot Forecasting**
  - Train spatial-temporal models to forecast high-risk cash-withdrawal locations.
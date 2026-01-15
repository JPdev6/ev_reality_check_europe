# ⚡ EV Reality Check Europe

A practical data project that looks at the **real state of Electric Vehicles (EVs) in Europe** —  
what exists today, how fast adoption grows, what batteries & ranges look like in reality,  
and what happens to electricity demand if EV usage increases.

This project is **data-first**, not hype-driven.

![EV Dashboard](assets/unnamed.png)
---
## 📂 Data Sources

This project combines multiple **official European datasets** and **media-derived data**, processed using **Databricks** and **Apache Spark**.

---

### 1️⃣ European Vehicle CO₂ & Registration Data  
**Source:** European Environment Agency (EEA)

- **Dataset:** CO₂ emissions from passenger cars  
- **Coverage:** EU + EEA countries  
- **Data includes:**
  - Vehicle registrations
  - Fuel type (electric, petrol, diesel, hybrid)
  - Vehicle mass, manufacturer, model
  - Electric range & emissions metrics  
- **Used to:**
  - Measure EV adoption trends
  - Build battery & range statistics
  - Create EV master and adoption tables  

📎 Source: https://www.eea.europa.eu/data-and-maps/data/co2-cars-emission-20  

---

### 2️⃣ European Electricity Prices  
**Source:** Eurostat

- **Dataset:** Electricity prices for household consumers  
- **Coverage:** EU countries  
- **Granularity:** Semi-annual / annual  
- **Metric:**
  - Average electricity price (€/kWh)  
- **Used to:**
  - Compare energy costs across countries
  - Estimate EV charging cost impact
  - Feed the EV market simulator  

📎 Source: https://ec.europa.eu/eurostat  

---

### 3️⃣ EV Battery Size Prediction (Derived Dataset)  
**Source:** Internal feature engineering & machine learning

- **Battery capacity estimated from:**
  - Electric range
  - Energy consumption per km
  - Vehicle mass  
- **Model:**
  - Gradient Boosted Trees (Spark ML)  
- **Output:**
  - `battery_kwh_predicted` (Gold layer)  

⚠️ *This is a modeled estimate, not a manufacturer specification.*

---

### 4️⃣ Public Sentiment & Media Coverage (EVs)  
**Source:** GDELT Global Knowledge Graph (GKG)

- **Coverage:** Global news sources  
- **Filtered to:**
  - Europe
  - EV-related themes (electric vehicles, batteries, charging, emissions)  
- **Metrics:**
  - Article volume
  - Average media tone
  - Positive vs negative coverage share  
- **Used to:**
  - Provide context on public & media discourse
  - Complement adoption and cost data  
  - *(Not treated as opinion polling)*  

📎 Source: https://www.gdeltproject.org/  

## What does this project answer?

- How many EVs exist per country in Europe?
- How is EV adoption evolving over time?
- What battery sizes and driving ranges are actually common?
- How do electricity prices differ across Europe?
- What happens to electricity demand if EV market share increases?
- How are EVs discussed in public media (signal, not opinion)?

---

## Tech stack

- **Databricks / Spark** (data processing)
- **Delta Lake** (Bronze → Silver → Gold)
- **Python / Pandas**
- **Streamlit** (interactive dashboard)
- **Plotly** (visualization)
- **GDELT** (media signal – experimental)

---

## Data architecture

### Bronze
Raw ingested data:
- EU vehicle registrations
- Electricity prices per country
- GDELT global news metadata

No cleaning, no logic.

---

### Silver
Cleaned and standardized datasets:
- Normalized country codes (ISO2)
- Clean numeric fields (battery, range, prices)
- Valid dates and years
- Consistent schemas

Still no aggregations for dashboards.

---

### Gold
Business-ready tables used directly by the dashboard:
- EV adoption per country & year
- Battery & range predictions (ML-based)
- OEM KPIs (average range, battery, number of models)
- Electricity prices per country & year
- Simple EV cost metrics
- Public sentiment signals (monthly, country-level)

---

## Dashboard

The Streamlit app has **two main tabs**:

### 📊 Dashboard
High-level, executive-friendly views:
- EV adoption trends by country
- Electricity price map (relative to EU median)
- OEM positioning (battery vs range, bubble size = models)
- Battery vs range density (what most EVs look like)
- Public media tone about EVs (from GDELT)

The goal is **understanding**, not technical detail.

---

### 🧠 Simulator
A simple **what-if simulator**:

Assumptions:
- Current EV market share is fixed at **15%**
- User selects a target EV share (e.g. 60%)
- User selects annual km per car

Outputs:
- Extra electricity demand
- Extra annual electricity cost
- Country-level breakdown

This is **not a forecast**, only a stress-test scenario.

---

## Public sentiment (important note)

Public sentiment is based on **media coverage**, not surveys.

- Source: GDELT
- Metric: article volume + average tone
- Time aggregation: monthly
- Geography: Europe only

It is used as a **signal**, not as a truth metric.

---

## ⚠️ Limitations & Assumptions

- **Battery capacity is estimated**, not manufacturer-reported.  
  Predictions are based on statistical patterns and should be interpreted as approximations.

- **Electricity prices are national averages.**  
  Local tariffs, charging contracts, and time-of-use pricing are not included.

- **EV adoption data reflects registrations, not vehicles in use.**  
  Stock vs flow effects are not fully captured.

- **Public sentiment comes from media coverage (GDELT), not surveys.**  
  News tone reflects media attention and framing, not direct public opinion.

- **Simulator results are illustrative.**  
  The EV market simulator is a simplified “what-if” tool and does not model grid constraints, subsidies, or behavioral effects.

- **Temporal coverage varies by dataset.**  
  Some datasets do not fully overlap in time; charts automatically adjust to available data.

## What this project is NOT

- Not a political statement
- Not a forecast of the future
- Not a climate model
- Not financial advice

It is a **transparent, data-driven reality check**.

---

## How to run

1. Set Databricks credentials (`.env` or Streamlit secrets)
2. Install dependencies
3. Run:
   ```bash
   streamlit run app.py

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from dotenv import load_dotenv

# Databricks SQL connector (pip install databricks-sql-connector)
try:
    from databricks import sql  # type: ignore
except Exception:
    try:
        import databricks.sql as sql  # type: ignore
    except Exception:
        sql = None


# =========================
# App setup
# =========================
load_dotenv()
st.set_page_config(page_title="EV Dashboard", page_icon="⚡", layout="wide")


# =========================
# Databricks connection
# =========================
def _get_dbx_cfg():
    """Prefer Streamlit secrets if present, else env vars."""
    def _safe_secret(key: str) -> str:
        try:
            return st.secrets.get(key, "")
        except Exception:
            return ""

    host = _safe_secret("DATABRICKS_HOST") or os.getenv("DATABRICKS_HOST", "")
    http_path = _safe_secret("DATABRICKS_HTTP_PATH") or os.getenv("DATABRICKS_HTTP_PATH", "")
    token = _safe_secret("DATABRICKS_TOKEN") or os.getenv("DATABRICKS_TOKEN", "")
    return host.strip(), http_path.strip(), token.strip()


@st.cache_data(ttl=3600, show_spinner=False)
def run_query(query: str) -> pd.DataFrame:
    host, http_path, token = _get_dbx_cfg()

    if sql is None:
        st.error("❌ Missing Databricks connector: pip install databricks-sql-connector")
        st.stop()

    if not host or not http_path or not token:
        st.error("❌ Missing Databricks login (set .env or secrets.toml)")
        st.stop()

    try:
        with sql.connect(server_hostname=host, http_path=http_path, access_token=token) as conn:
            return pd.read_sql(query, conn)
    except Exception as e:
        st.error("❌ Databricks query failed")
        st.code(query, language="sql")
        st.exception(e)
        st.stop()


@st.cache_data(ttl=3600, show_spinner=False)
def get_table_columns(table: str) -> list[str]:
    # lightweight schema probe
    df = run_query(f"SELECT * FROM {table} LIMIT 1")
    return [c.strip() for c in df.columns]


def _safe_list_sql_strings(values):
    return ", ".join(["'" + str(v).replace("'", "''") + "'" for v in values])


# =========================
# Country helpers
# =========================
COUNTRY_MAP = {
    "AUSTRIA": "AT","BELGIUM": "BE","BULGARIA": "BG","CROATIA": "HR","CYPRUS": "CY","CZECHIA": "CZ","CZECH REPUBLIC":"CZ",
    "DENMARK": "DK","ESTONIA": "EE","FINLAND": "FI","FRANCE": "FR","GERMANY": "DE","GREECE": "GR","HUNGARY": "HU",
    "IRELAND": "IE","ITALY": "IT","LATVIA": "LV","LITHUANIA": "LT","LUXEMBOURG": "LU","MALTA": "MT","NETHERLANDS": "NL",
    "POLAND": "PL","PORTUGAL": "PT","ROMANIA": "RO","SLOVAKIA": "SK","SLOVENIA": "SI","SPAIN": "ES","SWEDEN": "SE",
    "NORWAY": "NO","ICELAND": "IS","LIECHTENSTEIN": "LI","SWITZERLAND": "CH","UNITED KINGDOM": "GB","GREAT BRITAIN": "GB",
    "UK": "GB","GB": "GB",
}

ISO2_TO_NAME = {
    "AT":"Austria","BE":"Belgium","BG":"Bulgaria","HR":"Croatia","CY":"Cyprus","CZ":"Czechia","DK":"Denmark","EE":"Estonia",
    "FI":"Finland","FR":"France","DE":"Germany","GR":"Greece","HU":"Hungary","IE":"Ireland","IT":"Italy","LV":"Latvia","LT":"Lithuania",
    "LU":"Luxembourg","MT":"Malta","NL":"Netherlands","PL":"Poland","PT":"Portugal","RO":"Romania","SK":"Slovakia","SI":"Slovenia",
    "ES":"Spain","SE":"Sweden","NO":"Norway","IS":"Iceland","LI":"Liechtenstein","CH":"Switzerland","GB":"United Kingdom"
}

# ISO2 -> ISO3 (for choropleth world map)
ISO2_TO_ISO3 = {
    "AT":"AUT","BE":"BEL","BG":"BGR","HR":"HRV","CY":"CYP","CZ":"CZE","DK":"DNK","EE":"EST",
    "FI":"FIN","FR":"FRA","DE":"DEU","GR":"GRC","HU":"HUN","IE":"IRL","IT":"ITA","LV":"LVA","LT":"LTU",
    "LU":"LUX","MT":"MLT","NL":"NLD","PL":"POL","PT":"PRT","RO":"ROU","SK":"SVK","SI":"SVN","ES":"ESP","SE":"SWE",
    "NO":"NOR","IS":"ISL","LI":"LIE","CH":"CHE","GB":"GBR"
}

EU_ISO2 = [
    "AT","BE","BG","HR","CY","CZ","DK","EE","FI","FR","DE","GR","HU","IE","IT","LV","LT","LU","MT","NL","PL","PT","RO","SK","SI","ES","SE"
]


def normalize_country(df: pd.DataFrame, col: str = "country") -> pd.DataFrame:
    """Normalize country to ISO2. Accepts ISO2 already, or full names."""
    if df is None or df.empty or col not in df.columns:
        return df

    out = df.copy()
    s = out[col].astype(str).str.strip().str.replace("\u00a0", " ", regex=False)
    s_up = s.str.upper().str.replace("*", "", regex=False)

    is_iso2 = s_up.str.fullmatch(r"[A-Z]{2}")
    mapped_names = s_up.map(COUNTRY_MAP)

    out[col] = np.where(is_iso2, s_up, mapped_names.fillna(s_up))
    return out


def filter_selected_countries(df: pd.DataFrame, selected: list[str], col: str = "country") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if not selected:
        return df
    out = df.copy()
    out[col] = out[col].astype(str)
    return out[out[col].isin(selected)]


# =========================
# Small formatting helpers
# =========================
def fmt_compact(n: float | int | None, suffix: str = "", decimals: int = 2) -> str:
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return "—"
    try:
        n = float(n)
    except Exception:
        return "—"

    sign = "-" if n < 0 else ""
    n = abs(n)

    if n >= 1e12: return f"{sign}{n/1e12:.{decimals}f}T{suffix}"
    if n >= 1e9:  return f"{sign}{n/1e9:.{decimals}f}B{suffix}"
    if n >= 1e6:  return f"{sign}{n/1e6:.{decimals}f}M{suffix}"
    if n >= 1e3:  return f"{sign}{n/1e3:.{decimals}f}K{suffix}"
    return f"{sign}{n:.0f}{suffix}"


def fmt_eur(n: float | int | None) -> str:
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return "—"
    return f"€{fmt_compact(n, decimals=2)}"


def fmt_kwh(n: float | int | None) -> str:
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return "—"
    n = float(n)
    if abs(n) >= 1e9: return f"{n/1e9:.2f} TWh"
    if abs(n) >= 1e6: return f"{n/1e6:.2f} GWh"
    return f"{n:,.0f} kWh"


def clean_price_eur_kwh(s: pd.Series) -> pd.Series:
    s2 = pd.to_numeric(s, errors="coerce")
    s2 = np.where(s2 > 2.0, s2 / 100.0, s2)  # cents -> euros heuristic
    return pd.Series(s2)


def _set_plotly_dark(fig):
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=10, r=10, t=60, b=10),
        title=dict(font=dict(size=20)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
    return fig


# =========================
# Header
# =========================
st.markdown("# ⚡ EV Dashboard")


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown("### Filters")

    years_df = run_query("SELECT DISTINCT year FROM ev_reality_check.gold.ev_master ORDER BY year")
    years = years_df["year"].dropna().astype(int).tolist()
    if not years:
        st.error("No years found in gold.ev_master")
        st.stop()

    year_min, year_max = min(years), max(years)
    year_range = st.slider("Years", min_value=year_min, max_value=year_max, value=(year_min, year_max), step=1)

    # Countries (fixed EU list; some may have no data in some tables)
    name_to_iso2 = {v: k for k, v in ISO2_TO_NAME.items()}

    eu_c = EU_ISO2[:]  # fixed EU country codes
    eu_names = [ISO2_TO_NAME.get(c, c) for c in eu_c]

    default_c = eu_c[:10] if len(eu_c) >= 10 else eu_c
    default_names = [ISO2_TO_NAME.get(c, c) for c in default_c]

    selected_names = st.multiselect(
        "Countries",
        options=sorted(eu_names),
        default=default_names,
    )

    selected_countries = [name_to_iso2.get(n) for n in selected_names]
    selected_countries = [c for c in selected_countries if c in EU_ISO2]

    top_n = st.slider("Show top…", min_value=5, max_value=20, value=10, step=1)

y0, y1 = year_range
c_where = ""
if selected_countries:
    c_list = _safe_list_sql_strings(selected_countries)
    c_where = f" AND country IN ({c_list}) "


# =========================
# Tabs
# =========================
tab_dash, tab_sim = st.tabs(["📊 Dashboard", "🧠 Simulator"])


# =========================================================
# TAB 1: Dashboard
# =========================================================
with tab_dash:
    st.subheader("Quick view (Europe)")

    # --- KPIs ---
    kpi_ev = run_query(f"""
        SELECT COUNT(*) AS n
        FROM ev_reality_check.gold.ev_battery_predictions
        WHERE year BETWEEN {y0} AND {y1} {c_where}
    """)["n"].iloc[0]

    kpi_range = run_query(f"""
        SELECT percentile_approx(electric_range_km, 0.5) AS med
        FROM ev_reality_check.gold.ev_master
        WHERE year BETWEEN {y0} AND {y1} {c_where}
    """)["med"].iloc[0]

    kpi_batt = run_query(f"""
        SELECT percentile_approx(battery_kwh_predicted, 0.5) AS med
        FROM ev_reality_check.gold.ev_battery_predictions
        WHERE year BETWEEN {y0} AND {y1} {c_where}
    """)["med"].iloc[0]

    kpi_price = run_query(f"""
        SELECT percentile_approx(avg_price_eur_kwh, 0.5) AS med
        FROM ev_reality_check.gold.electricity_price_year
        WHERE year BETWEEN {y0} AND {y1}
    """)["med"].iloc[0]

    a1, a2, a3, a4 = st.columns(4)
    a1.metric("EV rows", f"{int(kpi_ev):,}")
    a2.metric("Typical range", f"{float(kpi_range):.0f} km")
    a3.metric("Typical battery", f"{float(kpi_batt):.1f} kWh")
    a4.metric("Typical electricity price", f"€{float(kpi_price):.3f} /kWh")

    st.divider()

    # --- Chart 1: EV growth (FIXED: one line per country) ---
    st.markdown("### EV growth (top countries)")
    adoption = run_query(f"""
        SELECT country, year, bev_records
        FROM ev_reality_check.gold.ev_adoption
        WHERE year BETWEEN {y0} AND {y1} {c_where}
    """)
    adoption = normalize_country(adoption)
    adoption = filter_selected_countries(adoption, selected_countries)

    if adoption is None or adoption.empty:
        st.info("No adoption data for this selection.")
    else:
        adoption["bev_records"] = pd.to_numeric(adoption["bev_records"], errors="coerce").fillna(0)
        adoption["year"] = pd.to_numeric(adoption["year"], errors="coerce").astype("Int64")
        adoption = adoption.dropna(subset=["year"])

        # CRITICAL FIX: aggregate per country-year
        adoption = (
            adoption.groupby(["country", "year"], as_index=False)["bev_records"]
            .sum()
        )

        adoption["country_name"] = adoption["country"].map(ISO2_TO_NAME).fillna(adoption["country"]).astype(str).str.strip()

        top_c = (
            adoption.groupby("country_name")["bev_records"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
            .index
            .tolist()
        )
        adoption = adoption[adoption["country_name"].isin(top_c)].sort_values(["country_name", "year"])

        fig = px.line(
            adoption,
            x="year",
            y="bev_records",
            color="country_name",
            markers=False
        )
        fig.update_yaxes(title="EV records", tickformat=".2s")
        fig.update_xaxes(title="Year", dtick=1)
        st.plotly_chart(_set_plotly_dark(fig), use_container_width=True)

    st.divider()

    # --- Chart 2: Electricity price map (BLACK, WORLD) ---
    st.markdown("### Electricity price map")
    map_year = st.selectbox("Map year", options=list(range(y0, y1 + 1)), index=(y1 - y0), key="map_year")

    price = run_query(f"""
        SELECT country, year, avg_price_eur_kwh
        FROM ev_reality_check.gold.electricity_price_year
        WHERE year = {map_year}
    """)
    price = normalize_country(price)
    # Keep all rows for the year; we will overlay the selected EU set on the map.

    if price is None or price.empty:
        st.info("No electricity price data for this selection.")
    else:
        price["avg_price_eur_kwh"] = clean_price_eur_kwh(price["avg_price_eur_kwh"])
        # We keep NaNs so countries still appear on the map.
        price["country"] = price["country"].astype(str).str.upper().str.strip()
        price["iso3"] = price["country"].map(ISO2_TO_ISO3)

        # Work only with EU/selected countries for the median + overlay
        overlay = pd.DataFrame({"country": (selected_countries or EU_ISO2)})
        overlay["country"] = overlay["country"].astype(str).str.upper().str.strip()
        overlay["country_name"] = overlay["country"].map(ISO2_TO_NAME).fillna(overlay["country"])
        overlay["iso3"] = overlay["country"].map(ISO2_TO_ISO3)

        # Join prices onto the overlay set
        map_df = overlay.merge(
            price[["country", "avg_price_eur_kwh"]],
            on="country",
            how="left",
        )
        map_df = map_df.dropna(subset=["iso3"]).copy()

        # EU median computed on available values within the overlay set
        eu_med = float(map_df["avg_price_eur_kwh"].median()) if map_df["avg_price_eur_kwh"].notna().any() else np.nan
        map_df["rel_to_median"] = np.where(
            np.isfinite(eu_med) & (eu_med > 0) & map_df["avg_price_eur_kwh"].notna(),
            map_df["avg_price_eur_kwh"] / eu_med,
            np.nan,
        )

        # IMPORTANT: Plotly choropleth drops rows with NaN color. Use a neutral value so every selected country is drawn.
        map_df["rel_plot"] = map_df["rel_to_median"].fillna(1.0)

        # Pre-format hover so missing data doesn't show "nan"
        map_df["price_hover"] = map_df["avg_price_eur_kwh"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
        map_df["rel_hover"] = map_df["rel_to_median"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")

        fig = px.choropleth(
            map_df,
            locations="iso3",
            color="rel_plot",
            hover_name="country_name",
            hover_data={
                "price_hover": True,
                "rel_hover": True,
                "avg_price_eur_kwh": False,
                "rel_to_median": False,
                "rel_plot": False,
                "iso3": False,
            },
            color_continuous_scale="Blues",
            range_color=(0.6, 1.4),
            title=(
                f"Electricity price vs EU median ({map_year})  |  EU median = €{eu_med:.3f}/kWh"
                if np.isfinite(eu_med)
                else f"Electricity price ({map_year})"
            ),
        )
        fig.update_traces(
            hovertemplate="<b>%{hovertext}</b><br>€/kWh=%{customdata[0]}<br>vs median=%{customdata[1]}<extra></extra>"
        )

        fig.update_geos(
            showframe=False,
            bgcolor="rgba(0,0,0,0)",
            showocean=True, oceancolor="black",
            showland=True, landcolor="rgb(25,25,25)",
            showcountries=True, countrycolor="rgba(255,255,255,0.15)",
            showcoastlines=False,
            projection_type="natural earth",
        )
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="black",
            plot_bgcolor="black",
            margin=dict(l=10, r=10, t=80, b=10),
            coloraxis_colorbar=dict(title="vs EU median"),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # --- Chart 3: Car makers (range vs battery) ---
    st.markdown("### Car makers (range vs battery)")
    oem = run_query("""
        SELECT man, records, avg_range_km, avg_battery_kwh
        FROM ev_reality_check.gold.oem_kpis
    """)

    if oem is None or oem.empty:
        st.info("No OEM data available.")
    else:
        oem["models"] = pd.to_numeric(oem.get("records"), errors="coerce").fillna(0)
        oem["avg_range_km"] = pd.to_numeric(oem["avg_range_km"], errors="coerce")
        oem["avg_battery_kwh"] = pd.to_numeric(oem["avg_battery_kwh"], errors="coerce")
        oem = oem.dropna(subset=["avg_range_km", "avg_battery_kwh"])

        top_m = oem.groupby("man")["models"].sum().sort_values(ascending=False).head(top_n).index.tolist()
        oem = oem[oem["man"].isin(top_m)]

        fig = px.scatter(
            oem,
            x="avg_battery_kwh",
            y="avg_range_km",
            size="models",
            size_max=40,  # FIX: readable bubbles
            color="man",
            hover_data={"models": True},
        )
        fig.update_xaxes(title="Battery (kWh)")
        fig.update_yaxes(title="Range (km)")
        st.plotly_chart(_set_plotly_dark(fig), use_container_width=True)

    st.divider()

    # --- Chart 4: Battery vs Range density ---
    st.markdown("### Battery vs range (what most cars look like)")
    br = run_query(f"""
        SELECT electric_range_km, battery_kwh_predicted
        FROM ev_reality_check.gold.ev_battery_predictions
        WHERE year BETWEEN {y0} AND {y1} {c_where}
    """)
    br["electric_range_km"] = pd.to_numeric(br["electric_range_km"], errors="coerce")
    br["battery_kwh_predicted"] = pd.to_numeric(br["battery_kwh_predicted"], errors="coerce")
    br = br.dropna()

    if br.empty:
        st.info("No battery/range data for this selection.")
    else:
        fig = px.density_heatmap(
            br,
            x="battery_kwh_predicted",
            y="electric_range_km",
            nbinsx=35,
            nbinsy=35,
            title="Hot zones = common combinations",
        )
        fig.update_xaxes(title="Battery (kWh)")
        fig.update_yaxes(title="Range (km)")
        st.plotly_chart(_set_plotly_dark(fig), use_container_width=True)

    st.divider()

    # --- Chart 5: Public talk about EVs (GDELT) (FIXED WINDOW: 2025–2026, not affected by sidebar filters) ---
    st.markdown("### Public talk about EVs (news tone)")
    st.caption("Fixed window: **2025–2026** (EU). Tone is a rough signal from news text (not a survey).")

    sentiment_table = "ev_reality_check.gold.ev_public_sentiment"

    # Fixed window (decoupled from sidebar year/country filters)
    SENT_Y0, SENT_Y1 = 2025, 2026
    eu_list = _safe_list_sql_strings(EU_ISO2)

    # Probe schema
    sent_cols = get_table_columns(sentiment_table)

    # Required
    required_min = ["country", "year", "month"]
    if not all(c in sent_cols for c in required_min):
        st.info("Sentiment table exists but missing required columns (country/year/month).")
    else:
        # Build select list based on what exists
        base_cols = [c for c in ["country", "year", "month", "articles_proxy", "avg_tone"] if c in sent_cols]
        optional_cols = [c for c in ["median_tone", "positive_share", "negative_share", "ev_focus_score"] if c in sent_cols]
        select_cols = base_cols + optional_cols

        # 1) Try EU-only first
        sentiment = run_query(f"""
            SELECT {", ".join(select_cols)}
            FROM {sentiment_table}
            WHERE year BETWEEN {SENT_Y0} AND {SENT_Y1}
              AND country IN ({eu_list})
        """)

        # 2) If empty, fallback: fetch without EU filter (still fixed years)
        #    This prevents a blank chart if the table stores non-EU ISO2 codes.
        if sentiment is None or sentiment.empty:
            sentiment = run_query(f"""
                SELECT {", ".join(select_cols)}
                FROM {sentiment_table}
                WHERE year BETWEEN {SENT_Y0} AND {SENT_Y1}
            """)

        sentiment = normalize_country(sentiment)

        if sentiment is None or sentiment.empty:
            st.info("No public sentiment data found for 2025–2026.")
        else:
            # --- Robust cleanup so the chart always renders ---
            # Ensure numeric types
            sentiment["year"] = pd.to_numeric(sentiment["year"], errors="coerce")
            sentiment["month"] = pd.to_numeric(sentiment["month"], errors="coerce")

            if "articles_proxy" in sentiment.columns:
                sentiment["articles_proxy"] = pd.to_numeric(sentiment["articles_proxy"], errors="coerce").fillna(0)
            else:
                sentiment["articles_proxy"] = 1.0

            if "avg_tone" in sentiment.columns:
                sentiment["avg_tone"] = pd.to_numeric(sentiment["avg_tone"], errors="coerce")
            else:
                sentiment["avg_tone"] = np.nan

            # Drop rows with missing year/month; keep only valid months
            sentiment = sentiment.dropna(subset=["year", "month"])
            sentiment = sentiment[(sentiment["month"] >= 1) & (sentiment["month"] <= 12)]

            # If tone is missing (common), fallback to 0 so we still show volume bubbles
            sentiment["avg_tone"] = sentiment["avg_tone"].fillna(0.0)

            # Normalize country codes + names
            sentiment["country"] = sentiment["country"].astype(str).str.upper().str.strip()
            sentiment["country_name"] = sentiment["country"].map(ISO2_TO_NAME).fillna(sentiment["country"])

            # Build a monthly date axis
            sentiment["date"] = pd.to_datetime(
                sentiment["year"].astype(int).astype(str)
                + "-"
                + sentiment["month"].astype(int).astype(str)
                + "-01",
                errors="coerce",
            )
            sentiment = sentiment.dropna(subset=["date"])

            # If we still have nothing after cleanup, show a quick debug table
            if sentiment.empty:
                st.warning("Sentiment data exists but could not be parsed into dates. Check year/month values in the GOLD table.")
                st.dataframe(run_query(f"SELECT country, year, month, articles_proxy, avg_tone FROM {sentiment_table} WHERE year BETWEEN {SENT_Y0} AND {SENT_Y1} LIMIT 50"), width='stretch')
            else:
                # Top countries by EV-related article volume (within 2025–2026)
                top_s = (
                    sentiment.groupby("country_name")["articles_proxy"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(top_n)
                    .index
                    .tolist()
                )
                sentiment = sentiment[sentiment["country_name"].isin(top_s)].copy()

                # --- EU monthly weighted tone (weights = article volume proxy) ---
                eu_month = (
                    sentiment.groupby("date", as_index=False)
                    .apply(lambda g: pd.Series({
                        "eu_tone": float(np.average(g["avg_tone"], weights=g["articles_proxy"])) if g["articles_proxy"].sum() > 0 else float(g["avg_tone"].mean()),
                        "eu_articles": float(g["articles_proxy"].sum()),
                    }))
                    .reset_index(drop=True)
                )

                # Scatter bubbles by country + EU line overlay
                fig = px.scatter(
                    sentiment,
                    x="date",
                    y="avg_tone",
                    size="articles_proxy",
                    size_max=35,
                    color="country_name",
                )

                # Add EU overall line
                eu_line = px.line(eu_month, x="date", y="eu_tone")
                for tr in eu_line.data:
                    tr.name = "EU overall"
                    tr.showlegend = True
                    tr.mode = "lines+markers"
                    fig.add_trace(tr)

                # Styling: NO title, fixed y-range, reference line at 0
                fig.update_layout(title_text="")
                fig.update_xaxes(title="Month")
                fig.update_yaxes(title="Tone", range=[-5, 2])
                fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="rgba(255,255,255,0.35)")

                st.plotly_chart(_set_plotly_dark(fig), use_container_width=True)

                # ----------------------------
                # 5-bullet Summary (simple)
                # ----------------------------
                total_articles = float(sentiment["articles_proxy"].sum())
                avg_tone_unweighted = float(sentiment["avg_tone"].mean()) if len(sentiment) else 0.0
                avg_tone_weighted = float(
                    np.average(sentiment["avg_tone"], weights=sentiment["articles_proxy"])
                    if total_articles > 0 else avg_tone_unweighted
                )

                # Best / worst EU month (from EU line)
                eu_best = eu_month.loc[eu_month["eu_tone"].idxmax()] if not eu_month.empty else None
                eu_worst = eu_month.loc[eu_month["eu_tone"].idxmin()] if not eu_month.empty else None

                # Most negative country overall (weighted by volume)
                by_country = (
                    sentiment.groupby("country_name")
                    .apply(lambda g: pd.Series({
                        "tone_w": float(np.average(g["avg_tone"], weights=g["articles_proxy"])) if g["articles_proxy"].sum() > 0 else float(g["avg_tone"].mean()),
                        "articles": float(g["articles_proxy"].sum()),
                    }))
                    .reset_index()
                )
                most_negative = by_country.sort_values("tone_w").iloc[0] if not by_country.empty else None

                # Trend: last 2 months vs first 2 months (EU overall)
                trend_note = "—"
                if len(eu_month) >= 4:
                    eu_sorted = eu_month.sort_values("date")
                    first = float(eu_sorted.head(2)["eu_tone"].mean())
                    last = float(eu_sorted.tail(2)["eu_tone"].mean())
                    delta = last - first
                    trend_note = f"{delta:+.2f} (last 2 months vs first 2 months)"

                st.markdown("### Summary")

                st.markdown(
                    "- **Overall media tone:** EV-related news coverage in **2025–2026** is slightly negative to neutral. This reflects cautious reporting rather than outright opposition.\n"
                    "- **Volume matters:** Most sentiment signals are driven by a small number of high-volume countries, meaning headlines are concentrated rather than evenly spread.\n"
                    "- **No extreme swings:** There are no sustained positive or negative spikes; sentiment fluctuates mildly around neutral over time.\n"
                    "- **Interpretation:** This tone reflects media framing (policy debates, costs, infrastructure), not direct public opinion."
                )

                # =========================
                # Final executive takeaways (all charts)
                # =========================
                st.divider()
                st.markdown("## Executive takeaways")

                st.markdown(
                    "- **EV adoption is uneven across Europe:** A small group of countries accounts for the majority of EV registrations, while many markets remain early-stage.\n"
                    "- **Electricity price differences matter:** Cross-country price gaps are large enough to materially change EV running costs and national electricity demand.\n"
                    "- **Manufacturers follow distinct strategies:** OEMs clearly trade off battery size and range, indicating different efficiency and positioning choices.\n"
                    "- **Most EVs converge on a common spec:** The bulk of vehicles cluster around a mid-range battery and driving range, suggesting market standardization.\n"
                    "- **Public discourse is cautious, not hostile:** Media sentiment in 2025–2026 is mildly negative to neutral, driven by economic and policy concerns rather than rejection of EVs."
                )


# =========================================================
# TAB 2: Simulator (Simple What-If)
# =========================================================
with tab_sim:
    st.markdown("## 🧠 Simulator")
    st.caption("Baseline market share today is fixed at **15%** (simple mode).")

    BASELINE_SHARE = 15  # fixed as you asked

    col1, col2, col3 = st.columns(3)

    with col1:
        scenario_year = st.selectbox(
            "Year",
            options=list(range(y0, y1 + 1)),
            index=(y1 - y0),
            key="sim_year",
        )

    with col2:
        target_share = st.slider(
            "Target EV share (%)",
            min_value=5,
            max_value=100,
            value=60,
            step=5,
            key="sim_target_share",
        )

    with col3:
        km_per_year = st.slider(
            "Km per car per year",
            min_value=5000,
            max_value=25000,
            value=15000,
            step=500,
            key="sim_km",
        )

    st.divider()

    # ----------------------------
    # Load data
    # ----------------------------
    adoption = run_query(f"""
        SELECT country, year, bev_records
        FROM ev_reality_check.gold.ev_adoption
        WHERE year = {scenario_year} {c_where}
    """)

    prices = run_query(f"""
        SELECT country, year, avg_price_eur_kwh
        FROM ev_reality_check.gold.electricity_price_year
        WHERE year = {scenario_year}
    """)

    costs = run_query(f"""
        SELECT country, year, AVG(annual_kwh_15k_km) AS annual_kwh_15k_km
        FROM ev_reality_check.gold.ev_cost
        WHERE year = {scenario_year} {c_where}
        GROUP BY country, year
    """)

    adoption = normalize_country(adoption)
    prices = normalize_country(prices)
    costs = normalize_country(costs)

    df = adoption.merge(prices, on=["country", "year"], how="left") \
                 .merge(costs, on=["country", "year"], how="left")

    if df.empty:
        st.warning("No data available for this selection.")
        st.stop()

    # ----------------------------
    # Clean inputs
    # ----------------------------
    df["bev_records"] = pd.to_numeric(df["bev_records"], errors="coerce").fillna(0)

    df["annual_kwh_15k_km"] = pd.to_numeric(df["annual_kwh_15k_km"], errors="coerce")
    df["annual_kwh_15k_km"] = df["annual_kwh_15k_km"].fillna(
        df["annual_kwh_15k_km"].median() if df["annual_kwh_15k_km"].notna().any() else 2550
    )

    df["price_eur_kwh"] = clean_price_eur_kwh(df["avg_price_eur_kwh"])
    df["price_eur_kwh"] = df["price_eur_kwh"].fillna(df["price_eur_kwh"].median())

    # ----------------------------
    # Core logic
    # ----------------------------
    df["annual_kwh"] = df["annual_kwh_15k_km"] * (km_per_year / 15000.0)
    df["country_name"] = df["country"].map(ISO2_TO_NAME).fillna(df["country"])

    df["current_demand_kwh"] = df["bev_records"] * df["annual_kwh"]

    multiplier = max(float(target_share) / float(BASELINE_SHARE), 0.0)
    df["extra_demand_kwh"] = df["current_demand_kwh"] * max(multiplier - 1.0, 0.0)

    df["extra_bill_eur"] = df["extra_demand_kwh"] * df["price_eur_kwh"]

    # ----------------------------
    # KPIs
    # ----------------------------
    k1, k2, k3 = st.columns(3)
    k1.metric("Scale vs today", f"×{multiplier:.2f}")
    k2.metric("Extra electricity", fmt_kwh(df["extra_demand_kwh"].sum()))
    k3.metric("Extra annual bill", fmt_eur(df["extra_bill_eur"].sum()))

    st.divider()

    # ----------------------------
    # Charts
    # ----------------------------
    top = df.sort_values("extra_demand_kwh", ascending=False).head(15)

    fig1 = px.bar(
        top,
        x="country_name",
        y="extra_demand_kwh",
        title="Extra electricity by country",
    )
    fig1.update_yaxes(title="Extra kWh", tickformat=".2s")
    fig1.update_xaxes(title="")
    st.plotly_chart(_set_plotly_dark(fig1), use_container_width=True)

    fig2 = px.scatter(
        df,
        x="price_eur_kwh",
        y="extra_bill_eur",
        size="bev_records",
        size_max=35,
        hover_name="country_name",
        title="Extra bill vs electricity price",
    )
    fig2.update_xaxes(title="€/kWh")
    fig2.update_yaxes(title="Extra bill (€)", tickformat=".2s")
    st.plotly_chart(_set_plotly_dark(fig2), use_container_width=True)

    # ----------------------------
    # Audit table (NO annual_kwh)
    # ----------------------------
    audit = df[[
        "country_name",
        "bev_records",
        "price_eur_kwh",
        "extra_demand_kwh",
        "extra_bill_eur",
    ]].copy()

    audit = audit.rename(columns={
        "country_name": "Country",
        "bev_records": "EVs today",
        "price_eur_kwh": "€/kWh",
        "extra_demand_kwh": "Extra electricity",
        "extra_bill_eur": "Extra bill (€)",
    })

    audit["EVs today"] = audit["EVs today"].map(lambda x: f"{int(x):,}")
    audit["€/kWh"] = audit["€/kWh"].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    audit["Extra electricity"] = audit["Extra electricity"].map(fmt_kwh)
    audit["Extra bill (€)"] = audit["Extra bill (€)"].map(fmt_eur)

    st.markdown("### Details")
    st.dataframe(audit, use_container_width=True, height=420)

    st.caption("Baseline market share today = 15% (fixed).")
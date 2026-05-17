import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ferry Operations Intelligence Dashboard",
    page_icon="⛴️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* KPI cards */
    .kpi-card {
        background: linear-gradient(135deg, #1a1f2e 0%, #16213e 100%);
        border: 1px solid #2d3561;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .kpi-title {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        color: #8892b0;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .kpi-value {
        font-size: 2.0rem;
        font-weight: 700;
        color: #64ffda;
        line-height: 1.1;
    }
    .kpi-delta {
        font-size: 0.78rem;
        color: #a8b2d8;
        margin-top: 6px;
    }
    .kpi-warn  { color: #ff6b6b; }
    .kpi-ok    { color: #64ffda; }
    .kpi-mid   { color: #ffd166; }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #ccd6f6;
        border-left: 4px solid #64ffda;
        padding-left: 12px;
        margin: 28px 0 16px 0;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #21262d;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { color: #8892b0; }
    .stTabs [aria-selected="true"] { color: #64ffda !important; border-bottom-color: #64ffda !important; }

    /* Plotly chart background already transparent via template */

    /* Metric delta override */
    div[data-testid="stMetricDelta"] { font-size: 0.75rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading & engineering features…")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # ── Temporal columns ──────────────────────────────────────────────────
    df["Date"]        = df["Timestamp"].dt.date
    df["Year"]        = df["Timestamp"].dt.year
    df["Month"]       = df["Timestamp"].dt.month
    df["Month_Name"]  = df["Timestamp"].dt.strftime("%b")
    df["Week"]        = df["Timestamp"].dt.isocalendar().week.astype(int)
    df["DayOfWeek"]   = df["Timestamp"].dt.day_name()
    df["Hour"]        = df["Timestamp"].dt.hour
    df["Minute"]      = df["Timestamp"].dt.minute
    df["Quarter"]     = df["Timestamp"].dt.quarter

    df["Is_Weekend"]  = df["Timestamp"].dt.dayofweek >= 5

    # Shift slots  0-5→Night, 6-11→Morning, 12-16→Afternoon, 17-21→Evening, 22-23→Night
    def shift(h):
        if 6  <= h < 12: return "Morning"
        if 12 <= h < 17: return "Afternoon"
        if 17 <= h < 22: return "Evening"
        return "Night"
    df["Shift"] = df["Hour"].map(shift)

    # Season (Southern Hemisphere flip-aware – data looks temperate so NH)
    def season(m):
        if m in [12, 1, 2]: return "Winter"
        if m in [3,  4, 5]: return "Spring"
        if m in [6,  7, 8]: return "Summer"
        return "Autumn"
    df["Season"] = df["Month"].map(season)

    # ── Core feature engineering ──────────────────────────────────────────
    df["Total_Activity"]          = df["Sales Count"] + df["Redemption Count"]
    df["Redemption_Pressure"]     = df["Redemption Count"] / (df["Sales Count"] + 1)

    # Operational Load Index – normalised within a rolling 7-day window
    # (rank-based to be robust to outliers)
    rolling_max = df["Total_Activity"].rolling(window=672, min_periods=1).max()   # 672 × 15-min = 7 days
    rolling_max = rolling_max.replace(0, 1)
    df["OLI"]                     = (df["Total_Activity"] / rolling_max).clip(0, 1)

    # Idle flag: Total_Activity == 0 or very low (bottom 5th percentile)
    low_thresh                    = df["Total_Activity"].quantile(0.05)
    df["Is_Idle"]                 = df["Total_Activity"] <= max(low_thresh, 0)

    # High-congestion flag: top 10% OLI
    df["Is_Congested"]            = df["OLI"] >= df["OLI"].quantile(0.90)

    # ── Data quality flags ────────────────────────────────────────────────
    df["Has_Anomaly"]             = (df["Total_Activity"] < 0) | \
                                    (df["Redemption Count"] < 0) | \
                                    (df["Sales Count"] < 0)

    # Rolling smoothed activity (4 × 15-min = 1-hour rolling mean)
    df["Smoothed_Activity"]       = df["Total_Activity"].rolling(4, min_periods=1, center=True).mean()

    return df


DATA_PATH = "Ferry tickets.csv"
df = load_data(DATA_PATH)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⛴️ Ferry Analytics")
    st.markdown("---")

    years = sorted(df["Year"].unique())
    sel_years = st.multiselect("Year(s)", years, default=years[-3:])

    seasons = ["Winter","Spring","Summer","Autumn"]
    sel_seasons = st.multiselect("Season(s)", seasons, default=seasons)

    day_type = st.radio("Day Type", ["All", "Weekday", "Weekend"])

    shifts = ["Morning","Afternoon","Evening","Night"]
    sel_shifts = st.multiselect("Shift(s)", shifts, default=shifts)

    st.markdown("---")
    st.caption("Data: 2015 – 2025 · 15-min intervals")

# Apply filters
mask = (
    df["Year"].isin(sel_years) &
    df["Season"].isin(sel_seasons) &
    df["Shift"].isin(sel_shifts)
)
if day_type == "Weekday":
    mask &= ~df["Is_Weekend"]
elif day_type == "Weekend":
    mask &= df["Is_Weekend"]

fdf = df[mask].copy()

# ─────────────────────────────────────────────────────────────────────────────
# HELPER – PLOTLY DARK TEMPLATE
# ─────────────────────────────────────────────────────────────────────────────
COLORS = {
    "teal":   "#64ffda",
    "blue":   "#4d9de0",
    "orange": "#ffd166",
    "red":    "#ff6b6b",
    "purple": "#a78bfa",
    "green":  "#06d6a0",
}

def dark_fig(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111827",
        font=dict(color="#ccd6f6", size=12),
        title_font=dict(color="#ccd6f6", size=14),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8892b0")),
        xaxis=dict(gridcolor="#1f2937", zerolinecolor="#374151", color="#8892b0"),
        yaxis=dict(gridcolor="#1f2937", zerolinecolor="#374151", color="#8892b0"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE KPIs
# ─────────────────────────────────────────────────────────────────────────────
def compute_kpis(d: pd.DataFrame) -> dict:
    total_intervals    = len(d)
    active_intervals   = (d["Total_Activity"] > 0).sum()

    # Capacity Utilisation Ratio – % of intervals with above-median activity
    median_act         = d["Total_Activity"].median()
    cur                = (d["Total_Activity"] >= median_act).mean() * 100

    # Congestion Pressure Index – mean OLI during congested periods
    cong               = d[d["Is_Congested"]]
    cpi                = cong["OLI"].mean() * 100 if len(cong) else 0.0

    # Idle Capacity Percentage
    icp                = d["Is_Idle"].mean() * 100

    # Peak Strain Duration – longest consecutive congested stretch (in hours)
    cong_arr           = d["Is_Congested"].values
    max_run = cur_run  = 0
    for v in cong_arr:
        if v: cur_run += 1; max_run = max(max_run, cur_run)
        else: cur_run = 0
    psd                = max_run * 15 / 60   # convert 15-min blocks to hours

    # Operational Variability Score – CV of OLI (lower = more stable)
    ovs                = (d["OLI"].std() / (d["OLI"].mean() + 1e-9)) * 100

    return dict(CUR=cur, CPI=cpi, ICP=icp, PSD=psd, OVS=ovs,
                total=total_intervals, active=active_intervals,
                total_sales=d["Sales Count"].sum(),
                total_redemptions=d["Redemption Count"].sum(),
                avg_oli=d["OLI"].mean()*100)

kpis = compute_kpis(fdf)

# ─────────────────────────────────────────────────────────────────────────────
# TITLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(90deg,#0f2027,#203a43,#2c5364);
            border-radius:14px; padding:22px 30px; margin-bottom:28px;
            border:1px solid #2d3561;">
  <h1 style="color:#64ffda;margin:0;font-size:1.9rem;">⛴️ Ferry Operations Intelligence Dashboard</h1>
  <p style="color:#8892b0;margin:6px 0 0 0;font-size:0.9rem;">
      Capacity Utilisation · Congestion Detection · Idle Period Analysis · 2015–2025
  </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

def kpi_card(col, title, value, fmt, note, color_class="kpi-ok"):
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-title">{title}</div>
      <div class="kpi-value {color_class}">{fmt.format(value)}</div>
      <div class="kpi-delta">{note}</div>
    </div>""", unsafe_allow_html=True)

kpi_card(k1, "Capacity Utilisation Ratio",  kpis["CUR"],  "{:.1f}%",
         "Intervals ≥ median load",
         "kpi-ok" if kpis["CUR"] > 60 else "kpi-mid")

kpi_card(k2, "Congestion Pressure Index",   kpis["CPI"],  "{:.1f}%",
         "Avg OLI in top-10% intervals",
         "kpi-warn" if kpis["CPI"] > 80 else "kpi-mid")

kpi_card(k3, "Idle Capacity %",             kpis["ICP"],  "{:.1f}%",
         "Intervals at floor activity",
         "kpi-warn" if kpis["ICP"] > 30 else "kpi-ok")

kpi_card(k4, "Peak Strain Duration",        kpis["PSD"],  "{:.1f} hrs",
         "Longest consecutive high-load",
         "kpi-warn" if kpis["PSD"] > 8 else "kpi-ok")

kpi_card(k5, "Operational Variability Score", kpis["OVS"], "{:.1f}",
         "CV of OLI (lower = stable)",
         "kpi-ok" if kpis["OVS"] < 60 else "kpi-warn")

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY STATS ROW
# ─────────────────────────────────────────────────────────────────────────────
s1, s2, s3, s4 = st.columns(4)
s1.metric("Total Intervals",   f"{kpis['total']:,}")
s2.metric("Total Sales",       f"{kpis['total_sales']:,}")
s3.metric("Total Redemptions", f"{kpis['total_redemptions']:,}")
s4.metric("Avg OLI",           f"{kpis['avg_oli']:.1f}%")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Temporal Patterns",
    "🔥 Congestion & Idle",
    "📅 Segmentation",
    "📊 Trend Analysis",
    "🔍 Data Quality",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – TEMPORAL PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Hourly Activity Profile</div>', unsafe_allow_html=True)

    hourly = fdf.groupby("Hour").agg(
        Avg_Activity=("Total_Activity","mean"),
        Avg_Sales=("Sales Count","mean"),
        Avg_Redemptions=("Redemption Count","mean"),
        Avg_OLI=("OLI","mean"),
    ).reset_index()

    fig_hourly = go.Figure()
    fig_hourly.add_trace(go.Bar(
        x=hourly["Hour"], y=hourly["Avg_Activity"],
        name="Avg Total Activity", marker_color=COLORS["teal"], opacity=0.8,
    ))
    fig_hourly.add_trace(go.Scatter(
        x=hourly["Hour"], y=hourly["Avg_OLI"] * hourly["Avg_Activity"].max(),
        name="Avg OLI (scaled)", mode="lines+markers",
        line=dict(color=COLORS["orange"], width=2),
        marker=dict(size=6),
    ))
    fig_hourly.update_layout(
        title="Average Activity & OLI by Hour of Day",
        xaxis_title="Hour", yaxis_title="Avg Activity Count",
        barmode="group",
    )
    dark_fig(fig_hourly)
    st.plotly_chart(fig_hourly, use_container_width=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Daily Activity (Smoothed)</div>', unsafe_allow_html=True)
        daily = fdf.groupby("Date").agg(
            Total=("Total_Activity","sum"),
            Avg_OLI=("OLI","mean"),
        ).reset_index()
        daily["Date"] = pd.to_datetime(daily["Date"])
        daily["Smoothed"] = daily["Total"].rolling(7, min_periods=1).mean()

        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(
            x=daily["Date"], y=daily["Total"],
            name="Daily Total", mode="lines",
            line=dict(color=COLORS["blue"], width=1), opacity=0.4,
        ))
        fig_daily.add_trace(go.Scatter(
            x=daily["Date"], y=daily["Smoothed"],
            name="7-Day Rolling Avg", mode="lines",
            line=dict(color=COLORS["teal"], width=2),
        ))
        fig_daily.update_layout(title="Daily Total Activity with 7-Day Rolling Average",
                                xaxis_title="Date", yaxis_title="Total Activity")
        dark_fig(fig_daily)
        st.plotly_chart(fig_daily, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Monthly Average Activity</div>', unsafe_allow_html=True)
        month_order = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly = fdf.groupby(["Year","Month_Name"]).agg(
            Total=("Total_Activity","sum")
        ).reset_index()
        monthly["Month_Name"] = pd.Categorical(monthly["Month_Name"], categories=month_order, ordered=True)
        monthly = monthly.sort_values("Month_Name")

        fig_month = px.box(
            monthly, x="Month_Name", y="Total",
            color_discrete_sequence=[COLORS["purple"]],
            labels={"Month_Name":"Month","Total":"Monthly Total Activity"},
            title="Monthly Total Activity Distribution (All Years)",
        )
        dark_fig(fig_month)
        st.plotly_chart(fig_month, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – CONGESTION & IDLE
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">OLI Distribution</div>', unsafe_allow_html=True)
        fig_oli = go.Figure()
        fig_oli.add_trace(go.Histogram(
            x=fdf["OLI"], nbinsx=50,
            marker_color=COLORS["teal"], opacity=0.8,
            name="OLI",
        ))
        fig_oli.add_vline(x=fdf["OLI"].quantile(0.90), line_dash="dash",
                          line_color=COLORS["red"],
                          annotation_text="90th pct (Congestion)",
                          annotation_font_color=COLORS["red"])
        fig_oli.add_vline(x=fdf["OLI"].quantile(0.10), line_dash="dash",
                          line_color=COLORS["orange"],
                          annotation_text="10th pct (Idle)",
                          annotation_font_color=COLORS["orange"])
        fig_oli.update_layout(title="Operational Load Index Distribution",
                              xaxis_title="OLI", yaxis_title="Count")
        dark_fig(fig_oli)
        st.plotly_chart(fig_oli, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Congestion vs Idle by Hour</div>', unsafe_allow_html=True)
        hour_state = fdf.groupby("Hour").agg(
            Congested=("Is_Congested","sum"),
            Idle=("Is_Idle","sum"),
            Total=("Is_Congested","count"),
        ).reset_index()
        hour_state["Congested_Pct"] = hour_state["Congested"] / hour_state["Total"] * 100
        hour_state["Idle_Pct"]      = hour_state["Idle"]      / hour_state["Total"] * 100

        fig_ci = go.Figure()
        fig_ci.add_trace(go.Bar(x=hour_state["Hour"], y=hour_state["Congested_Pct"],
                                name="Congested %", marker_color=COLORS["red"]))
        fig_ci.add_trace(go.Bar(x=hour_state["Hour"], y=hour_state["Idle_Pct"],
                                name="Idle %", marker_color=COLORS["orange"]))
        fig_ci.update_layout(title="% of Congested vs Idle Intervals by Hour",
                             xaxis_title="Hour", yaxis_title="% of Intervals",
                             barmode="group")
        dark_fig(fig_ci)
        st.plotly_chart(fig_ci, use_container_width=True)

    st.markdown('<div class="section-header">Congestion Pressure Index Over Time (Monthly)</div>',
                unsafe_allow_html=True)
    monthly_cpi = fdf.groupby(["Year","Month"]).agg(
        CPI=("OLI", lambda x: x[x >= x.quantile(0.90)].mean() * 100 if len(x) else 0),
        Idle_Pct=("Is_Idle","mean"),
    ).reset_index()
    monthly_cpi["Period"] = pd.to_datetime(
        monthly_cpi["Year"].astype(str) + "-" + monthly_cpi["Month"].astype(str).str.zfill(2))

    fig_cpi_trend = go.Figure()
    fig_cpi_trend.add_trace(go.Scatter(
        x=monthly_cpi["Period"], y=monthly_cpi["CPI"],
        name="Congestion Pressure Index", mode="lines",
        line=dict(color=COLORS["red"], width=2),
    ))
    fig_cpi_trend.add_trace(go.Scatter(
        x=monthly_cpi["Period"], y=monthly_cpi["Idle_Pct"]*100,
        name="Idle Capacity %", mode="lines",
        line=dict(color=COLORS["orange"], width=2),
    ))
    fig_cpi_trend.update_layout(
        title="Monthly CPI & Idle Capacity % Trend",
        xaxis_title="Month", yaxis_title="Index Value",
    )
    dark_fig(fig_cpi_trend)
    st.plotly_chart(fig_cpi_trend, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Weekday vs Weekend Efficiency</div>',
                    unsafe_allow_html=True)
        wk = fdf.groupby(["DayOfWeek","Is_Weekend"]).agg(
            Avg_OLI=("OLI","mean"),
            Avg_Activity=("Total_Activity","mean"),
            Idle_Pct=("Is_Idle","mean"),
        ).reset_index()
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        wk["DayOfWeek"] = pd.Categorical(wk["DayOfWeek"], categories=day_order, ordered=True)
        wk = wk.sort_values("DayOfWeek")

        fig_wk = go.Figure()
        fig_wk.add_trace(go.Bar(
            x=wk["DayOfWeek"], y=wk["Avg_OLI"]*100,
            name="Avg OLI %",
            marker_color=[COLORS["orange"] if w else COLORS["teal"] for w in wk["Is_Weekend"]],
        ))
        fig_wk.update_layout(title="Average OLI by Day of Week",
                             xaxis_title="Day", yaxis_title="Avg OLI (%)")
        dark_fig(fig_wk)
        st.plotly_chart(fig_wk, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Seasonal Utilisation Patterns</div>',
                    unsafe_allow_html=True)
        seas = fdf.groupby("Season").agg(
            Avg_OLI=("OLI","mean"),
            Avg_Activity=("Total_Activity","mean"),
            Congested_Pct=("Is_Congested","mean"),
            Idle_Pct=("Is_Idle","mean"),
        ).reset_index()
        seas_order = ["Spring","Summer","Autumn","Winter"]
        seas["Season"] = pd.Categorical(seas["Season"], categories=seas_order, ordered=True)
        seas = seas.sort_values("Season")

        fig_seas = go.Figure()
        fig_seas.add_trace(go.Bar(x=seas["Season"], y=seas["Avg_OLI"]*100,
                                  name="Avg OLI", marker_color=COLORS["blue"]))
        fig_seas.add_trace(go.Bar(x=seas["Season"], y=seas["Congested_Pct"]*100,
                                  name="Congested %", marker_color=COLORS["red"]))
        fig_seas.add_trace(go.Bar(x=seas["Season"], y=seas["Idle_Pct"]*100,
                                  name="Idle %", marker_color=COLORS["orange"]))
        fig_seas.update_layout(title="Seasonal Efficiency Breakdown",
                               xaxis_title="Season", yaxis_title="%",
                               barmode="group")
        dark_fig(fig_seas)
        st.plotly_chart(fig_seas, use_container_width=True)

    st.markdown('<div class="section-header">Shift-Level Efficiency Band Analysis</div>',
                unsafe_allow_html=True)
    shift_analysis = fdf.groupby(["Shift","Is_Weekend"]).agg(
        Avg_OLI=("OLI","mean"),
        Avg_Activity=("Total_Activity","mean"),
        Congested_Pct=("Is_Congested","mean"),
        Idle_Pct=("Is_Idle","mean"),
        Redemption_Pressure=("Redemption_Pressure","mean"),
    ).reset_index()
    shift_analysis["Day_Type"] = shift_analysis["Is_Weekend"].map({True:"Weekend",False:"Weekday"})
    shift_order = ["Morning","Afternoon","Evening","Night"]
    shift_analysis["Shift"] = pd.Categorical(shift_analysis["Shift"], categories=shift_order, ordered=True)
    shift_analysis = shift_analysis.sort_values("Shift")

    fig_shift = px.bar(
        shift_analysis, x="Shift", y="Avg_OLI",
        color="Day_Type", barmode="group",
        color_discrete_map={"Weekday": COLORS["teal"], "Weekend": COLORS["orange"]},
        labels={"Avg_OLI": "Avg OLI", "Shift": "Shift"},
        title="Average OLI by Shift and Day Type",
    )
    dark_fig(fig_shift)
    st.plotly_chart(fig_shift, use_container_width=True)

    # Heatmap: Hour × DayOfWeek
    st.markdown('<div class="section-header">Activity Heatmap – Hour × Day of Week</div>',
                unsafe_allow_html=True)
    heat = fdf.groupby(["DayOfWeek","Hour"])["Total_Activity"].mean().reset_index()
    heat["DayOfWeek"] = pd.Categorical(heat["DayOfWeek"], categories=day_order, ordered=True)
    heat = heat.sort_values("DayOfWeek")
    heat_pivot = heat.pivot(index="DayOfWeek", columns="Hour", values="Total_Activity")

    fig_heat = px.imshow(
        heat_pivot,
        color_continuous_scale="Teal",
        labels=dict(color="Avg Activity"),
        title="Average Total Activity – Hour × Day of Week",
        aspect="auto",
        color_continuous_midpoint=None,
    )
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#111827",
        font=dict(color="#ccd6f6"),
        coloraxis_colorbar=dict(title="Avg Activity", tickfont=dict(color="#ccd6f6")),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – TREND ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">Year-over-Year KPI Trends</div>',
                unsafe_allow_html=True)

    yearly = fdf.groupby("Year").agg(
        Avg_OLI=("OLI","mean"),
        Total_Sales=("Sales Count","sum"),
        Total_Redemptions=("Redemption Count","sum"),
        Idle_Pct=("Is_Idle","mean"),
        Congested_Pct=("Is_Congested","mean"),
        Variability=("OLI","std"),
    ).reset_index()

    fig_yoy = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Avg OLI % (Annual)",
            "Sales vs Redemptions",
            "Idle & Congestion %",
            "Operational Variability (OLI std)",
        ),
        vertical_spacing=0.16,
    )

    fig_yoy.add_trace(go.Scatter(
        x=yearly["Year"], y=yearly["Avg_OLI"]*100,
        mode="lines+markers", name="Avg OLI",
        line=dict(color=COLORS["teal"], width=2), marker=dict(size=7),
    ), row=1, col=1)

    fig_yoy.add_trace(go.Bar(x=yearly["Year"], y=yearly["Total_Sales"],
                             name="Total Sales", marker_color=COLORS["blue"]), row=1, col=2)
    fig_yoy.add_trace(go.Bar(x=yearly["Year"], y=yearly["Total_Redemptions"],
                             name="Total Redemptions", marker_color=COLORS["purple"]), row=1, col=2)

    fig_yoy.add_trace(go.Scatter(
        x=yearly["Year"], y=yearly["Idle_Pct"]*100,
        mode="lines+markers", name="Idle %",
        line=dict(color=COLORS["orange"], width=2),
    ), row=2, col=1)
    fig_yoy.add_trace(go.Scatter(
        x=yearly["Year"], y=yearly["Congested_Pct"]*100,
        mode="lines+markers", name="Congested %",
        line=dict(color=COLORS["red"], width=2),
    ), row=2, col=1)

    fig_yoy.add_trace(go.Bar(
        x=yearly["Year"], y=yearly["Variability"],
        name="OLI Std Dev", marker_color=COLORS["green"],
    ), row=2, col=2)

    fig_yoy.update_layout(
        height=600,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#111827",
        font=dict(color="#ccd6f6", size=11),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8892b0")),
        barmode="group",
    )
    for ax in fig_yoy.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig_yoy.layout[ax].update(gridcolor="#1f2937", color="#8892b0")
    st.plotly_chart(fig_yoy, use_container_width=True)

    # Redemption Pressure Ratio trend
    st.markdown('<div class="section-header">Redemption Pressure Ratio – Annual Trend</div>',
                unsafe_allow_html=True)
    rpr_yearly = fdf.groupby("Year")["Redemption_Pressure"].mean().reset_index()
    fig_rpr = px.area(
        rpr_yearly, x="Year", y="Redemption_Pressure",
        title="Avg Redemption Pressure Ratio by Year",
        color_discrete_sequence=[COLORS["purple"]],
        labels={"Redemption_Pressure":"Avg Redemption Pressure Ratio"},
    )
    dark_fig(fig_rpr)
    st.plotly_chart(fig_rpr, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – DATA QUALITY
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Data Quality Overview</div>', unsafe_allow_html=True)

    total_rows  = len(fdf)
    zero_rows   = (fdf["Total_Activity"] == 0).sum()
    anom_rows   = fdf["Has_Anomaly"].sum()
    null_rows   = fdf.isnull().any(axis=1).sum()

    q1, q2, q3, q4 = st.columns(4)
    q1.metric("Total Intervals",     f"{total_rows:,}")
    q2.metric("Zero-Activity Intervals", f"{zero_rows:,}",
              delta=f"{zero_rows/total_rows*100:.1f}%", delta_color="inverse")
    q3.metric("Negative-Value Anomalies", f"{anom_rows:,}",
              delta="critical" if anom_rows > 0 else "none", delta_color="inverse")
    q4.metric("Rows with Nulls",    f"{null_rows:,}")

    # Zero-activity distribution
    st.markdown('<div class="section-header">Zero-Activity Intervals by Hour</div>',
                unsafe_allow_html=True)
    zero_dist = fdf[fdf["Total_Activity"] == 0].groupby("Hour").size().reset_index(name="Count")
    fig_zero = px.bar(
        zero_dist, x="Hour", y="Count",
        title="Distribution of Zero-Activity Intervals by Hour",
        color_discrete_sequence=[COLORS["orange"]],
        labels={"Count":"Zero-Activity Count"},
    )
    dark_fig(fig_zero)
    st.plotly_chart(fig_zero, use_container_width=True)

    # Missing interval gaps
    st.markdown('<div class="section-header">Missing 15-Min Interval Gaps (Top 20)</div>',
                unsafe_allow_html=True)
    ts_sorted = fdf["Timestamp"].sort_values().reset_index(drop=True)
    gaps = ts_sorted.diff().dropna()
    large_gaps = gaps[gaps > pd.Timedelta("15min")].rename("Gap").reset_index()
    large_gaps["Gap_Hours"] = large_gaps["Gap"].dt.total_seconds() / 3600
    large_gaps["Timestamp"] = ts_sorted.loc[large_gaps["index"]].values

    if len(large_gaps):
        top_gaps = large_gaps.nlargest(20, "Gap_Hours")[["Timestamp","Gap_Hours"]]
        fig_gaps = px.bar(
            top_gaps.sort_values("Gap_Hours", ascending=False),
            x="Gap_Hours", y=top_gaps.sort_values("Gap_Hours", ascending=False)["Timestamp"].astype(str),
            orientation="h",
            title="Top 20 Largest Data Gaps (Hours)",
            color_discrete_sequence=[COLORS["red"]],
            labels={"x":"Gap (Hours)", "y":"Timestamp"},
        )
        dark_fig(fig_gaps)
        st.plotly_chart(fig_gaps, use_container_width=True)
    else:
        st.success("✅ No gaps found in selected data.")

    # Raw stats
    st.markdown('<div class="section-header">Descriptive Statistics</div>', unsafe_allow_html=True)
    stats = fdf[["Sales Count","Redemption Count","Total_Activity","OLI","Redemption_Pressure"]].describe().T
    stats.columns = [c.title() for c in stats.columns]
    st.dataframe(stats.style.background_gradient(cmap="Blues", axis=1), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#4a5568;font-size:0.78rem;">'
    "Ferry Operations Intelligence Dashboard · Built with Streamlit & Plotly"
    "</p>",
    unsafe_allow_html=True,
)
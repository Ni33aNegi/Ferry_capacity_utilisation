import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import numpy as np



# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ferry Tickets Dashboard",
    page_icon="⛴️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');

    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

    .main { background: #f7f6f2; }
    .block-container { padding: 2rem 2.5rem 3rem; }

    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #e8e6df;
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 500;
        color: #888;
        text-transform: uppercase;
        letter-spacing: .07em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 28px;
        font-weight: 600;
        color: #1a1a1a;
        line-height: 1;
        margin-bottom: 4px;
    }
    .kpi-sub {
        font-size: 12px;
        color: #aaa;
    }
    .kpi-badge-green {
        display: inline-block;
        background: #e6f5ef;
        color: #1D9E75;
        font-size: 11px;
        font-weight: 500;
        padding: 2px 8px;
        border-radius: 20px;
    }
    .kpi-badge-red {
        display: inline-block;
        background: #fbeee9;
        color: #D85A30;
        font-size: 11px;
        font-weight: 500;
        padding: 2px 8px;
        border-radius: 20px;
    }
    .section-header {
        font-size: 12px;
        font-weight: 600;
        color: #aaa;
        text-transform: uppercase;
        letter-spacing: .1em;
        margin: 2rem 0 1rem;
        border-bottom: 1px solid #e8e6df;
        padding-bottom: .5rem;
    }
    .insight-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #e8e6df;
        height: 100%;
    }
    .insight-row {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 10px 0;
        border-bottom: 1px solid #f0ede6;
    }
    .insight-row:last-child { border-bottom: none; }
    .insight-dot {
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #378ADD;
        margin-top: 5px;
        flex-shrink: 0;
    }
    .insight-bold { font-weight: 600; font-size: 13px; color: #1a1a1a; }
    .insight-text { font-size: 12px; color: #777; margin-top: 2px; }
    .footnote { font-size: 11px; color: #bbb; margin-top: 1rem; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# ── Load & prepare ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Ferry Tickets.csv")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["Year"]      = df["Timestamp"].dt.year
    df["Month"]     = df["Timestamp"].dt.month
    df["Hour"]      = df["Timestamp"].dt.hour
    df["DayOfWeek"] = df["Timestamp"].dt.dayofweek
    return df

df = load_data()

# ── Aggregations ───────────────────────────────────────────────────────────────
annual = (df.groupby("Year")
            .agg(Sold=("Sales Count", "sum"), Redeemed=("Redemption Count", "sum"))
            .reset_index())
annual["Util%"] = (annual["Redeemed"] / annual["Sold"] * 100).round(1)

monthly_pivot = {}
for yr in [2023, 2024, 2025]:
    monthly_pivot[yr] = (df[df["Year"] == yr]
                           .groupby("Month")["Redemption Count"].sum()
                           .reindex(range(1, 13), fill_value=0).values)

hourly = df.groupby("Hour")["Redemption Count"].sum().values
dow    = df.groupby("DayOfWeek")["Redemption Count"].sum().values

# ── Palette ────────────────────────────────────────────────────────────────────
BLUE   = "#378ADD"
GREEN  = "#1D9E75"
PURPLE = "#7F77DD"
CORAL  = "#D85A30"
GRAY   = "#D3D1C7"

def fmt_tick(val, _):
    if val >= 1_000_000: return f"{val/1_000_000:.1f}M"
    if val >= 1_000:     return f"{val/1_000:.0f}K"
    return str(int(val))

def chart_style(ax, grid_axis="y"):
    ax.set_facecolor("#fafaf8")
    ax.spines[["top","right","left","bottom"]].set_visible(False)
    ax.tick_params(colors="#999", labelsize=9)
    ax.grid(axis=grid_axis, color="#eeebe4", linewidth=0.8, zorder=1)
    ax.set_axisbelow(True)

    ax

# ── Sidebar filters ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⛴️ Ferry Tickets")
    st.markdown("**Operations Dashboard**")
    st.markdown("---")
    year_range = st.slider(
        "Year range",
        int(annual["Year"].min()),
        int(annual["Year"].max()),
        (2015, 2025)
    )
    st.markdown("---")
    st.markdown(
        "<span style='font-size:11px;color:#aaa'>Data: 261,538 records<br>May 2015 – Dec 2025</span>",
        unsafe_allow_html=True
    )

# Filter annual by sidebar
annual_f = annual[(annual["Year"] >= year_range[0]) & (annual["Year"] <= year_range[1])]

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='font-size:26px;font-weight:600;color:#1a1a1a;margin-bottom:4px'>Ferry Tickets — Operations Dashboard</h1>"
    "<p style='color:#aaa;font-size:13px;margin-top:0'>2015 – 2025 · Sales & Redemption Analysis</p>",
    unsafe_allow_html=True
)

# ── KPI row ────────────────────────────────────────────────────────────────────
total_sold     = df["Sales Count"].sum()
total_redeemed = df["Redemption Count"].sum()
util_rate      = total_redeemed / total_sold * 100
unredeemed     = total_sold - total_redeemed

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total tickets sold</div>
        <div class="kpi-value">{total_sold/1e6:.2f}M</div>
        <div class="kpi-sub">2015 – 2025</div>
    </div>""", unsafe_allow_html=True)
with k2:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total redeemed</div>
        <div class="kpi-value">{total_redeemed/1e6:.2f}M</div>
        <div class="kpi-sub">across all years</div>
    </div>""", unsafe_allow_html=True)
with k3:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Utilization rate</div>
        <div class="kpi-value">{util_rate:.1f}%</div>
        <span class="kpi-badge-green">↑ redeemed / sold</span>
    </div>""", unsafe_allow_html=True)
with k4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Unredeemed tickets</div>
        <div class="kpi-value">{unredeemed/1e3:.0f}K</div>
        <span class="kpi-badge-red">{unredeemed/total_sold*100:.1f}% unused</span>
    </div>""", unsafe_allow_html=True)

# ── Row 1: Annual + Monthly ────────────────────────────────────────────────────
st.markdown("<div class='section-header'>Volume trends</div>", unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    fig1, ax1 = plt.subplots(figsize=(7, 3.6), facecolor="#fafaf8")
    x = np.arange(len(annual_f))
    w = 0.38
    ax1.bar(x - w/2, annual_f["Sold"],     width=w, color=BLUE,  label="Sold",     zorder=2, linewidth=0)
    ax1.bar(x + w/2, annual_f["Redeemed"], width=w, color=GREEN, label="Redeemed", zorder=2, linewidth=0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(annual_f["Year"], fontsize=9, rotation=45)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_tick))
    ax1.set_title("Annual tickets sold vs redeemed", fontsize=11, pad=10,
                  fontweight="500", loc="left", color="#333")
    ax1.legend(fontsize=9, frameon=False)
    chart_style(ax1)
    fig1.tight_layout()
    st.pyplot(fig1, use_container_width=True)
    plt.close(fig1)

with c2:
    fig2, ax2 = plt.subplots(figsize=(7, 3.6), facecolor="#fafaf8")
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]
    colors_yr = {2023: PURPLE, 2024: BLUE, 2025: GREEN}
    for yr, col in colors_yr.items():
        ax2.plot(range(1, 13), monthly_pivot[yr], color=col,
                 marker="o", markersize=4, linewidth=1.8, label=str(yr))
    ax2.set_xticks(range(1, 13))
    ax2.set_xticklabels(month_labels, fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_tick))
    ax2.set_title("Monthly redemption pattern (2023–2025)", fontsize=11, pad=10,
                  fontweight="500", loc="left", color="#333")
    ax2.legend(fontsize=9, frameon=False)
    chart_style(ax2)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)

# ── Row 2: Hourly + Day of week ────────────────────────────────────────────────
st.markdown("<div class='section-header'>Temporal patterns</div>", unsafe_allow_html=True)
c3, c4 = st.columns(2)

with c3:
    fig3, ax3 = plt.subplots(figsize=(7, 3.4), facecolor="#fafaf8")
    bar_colors = [BLUE if 10 <= h <= 14 else GRAY for h in range(24)]
    ax3.bar(range(24), hourly, color=bar_colors, zorder=2, linewidth=0)
    ax3.set_xticks(range(0, 24, 2))
    ax3.set_xticklabels([f"{h}h" for h in range(0, 24, 2)], fontsize=9)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_tick))
    ax3.set_title("Ridership by hour of day", fontsize=11, pad=10,
                  fontweight="500", loc="left", color="#333")
    ax3.legend(handles=[
        mpatches.Patch(color=BLUE, label="Peak 10am–2pm"),
        mpatches.Patch(color=GRAY, label="Off-peak")
    ], fontsize=9, frameon=False)
    chart_style(ax3)
    fig3.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

with c4:
    fig4, ax4 = plt.subplots(figsize=(7, 3.4), facecolor="#fafaf8")
    day_labels = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    dow_colors = [BLUE if d >= 5 else GRAY for d in range(7)]
    ax4.bar(day_labels, dow, color=dow_colors, zorder=2, linewidth=0)
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_tick))
    ax4.set_title("Ridership by day of week", fontsize=11, pad=10,
                  fontweight="500", loc="left", color="#333")
    ax4.legend(handles=[
        mpatches.Patch(color=BLUE, label="Weekend"),
        mpatches.Patch(color=GRAY, label="Weekday")
    ], fontsize=9, frameon=False)
    chart_style(ax4)
    fig4.tight_layout()
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)

# ── Row 3: Utilization + Insights ─────────────────────────────────────────────
st.markdown("<div class='section-header'>Utilization & insights</div>", unsafe_allow_html=True)
c5, c6 = st.columns(2)

with c5:
    fig5, ax5 = plt.subplots(figsize=(7, 3.4), facecolor="#fafaf8")
    ax5.plot(annual_f["Year"], annual_f["Util%"], color=CORAL,
             marker="o", markersize=5, linewidth=2, zorder=3)
    ax5.fill_between(annual_f["Year"], annual_f["Util%"], 100,
                     where=(annual_f["Util%"] >= 100),
                     alpha=0.12, color=GREEN, label="Above 100%")
    ax5.fill_between(annual_f["Year"], annual_f["Util%"], 100,
                     where=(annual_f["Util%"] < 100),
                     alpha=0.12, color=CORAL, label="Below 100%")
    ax5.axhline(100, color="#bbb", linewidth=1, linestyle="--")
    ax5.set_ylim(75, 110)
    ax5.set_xticks(annual_f["Year"])
    ax5.set_xticklabels(annual_f["Year"], fontsize=9, rotation=45)
    ax5.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax5.set_title("Utilization rate by year", fontsize=11, pad=10,
                  fontweight="500", loc="left", color="#333")
    ax5.legend(fontsize=9, frameon=False)
    chart_style(ax5, grid_axis="both")
    fig5.tight_layout()
    st.pyplot(fig5, use_container_width=True)
    plt.close(fig5)

with c6:
    st.markdown("""
    <div class="insight-card">
        <div style="font-size:13px;font-weight:600;color:#333;margin-bottom:8px">Key insights</div>
        <div class="insight-row">
            <div class="insight-dot"></div>
            <div>
                <div class="insight-bold">Summer dominates</div>
                <div class="insight-text">Jun–Aug accounts for ~62% of annual volume</div>
            </div>
        </div>
        <div class="insight-row">
            <div class="insight-dot" style="background:#1D9E75"></div>
            <div>
                <div class="insight-bold">Weekend surge</div>
                <div class="insight-text">Sat + Sun contribute ~42% of weekly trips</div>
            </div>
        </div>
        <div class="insight-row">
            <div class="insight-dot" style="background:#7F77DD"></div>
            <div>
                <div class="insight-bold">Peak window</div>
                <div class="insight-text">10am–2pm handles 46% of daily redemptions</div>
            </div>
        </div>
        <div class="insight-row">
            <div class="insight-dot" style="background:#D85A30"></div>
            <div>
                <div class="insight-bold">Near-perfect utilization</div>
                <div class="insight-text">Post-2017 consistently at or above 100%</div>
            </div>
        </div>
        <div class="insight-row">
            <div class="insight-dot" style="background:#B4B2A9"></div>
            <div>
                <div class="insight-bold">2025 on track</div>
                <div class="insight-text">Highest-volume year on record</div>
            </div>
        </div>
        <div class="footnote">* Redemptions > Sales may indicate multi-use passes or carry-over redemptions</div>
    </div>
    """, unsafe_allow_html=True)


st.write(df.columns)

df = pd.read_csv("Ferry Tickets.csv")

# clean columns
df.columns = df.columns.str.strip().str.lower()

# debug
st.write(df.columns)

# convert + index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# resample
df_15min = df.resample('15min').sum()


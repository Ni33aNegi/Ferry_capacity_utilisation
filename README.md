# Ferrytickets utitlization and capacity optimization

🚢 Ferry Operations Intelligence Dashboard

A professional-grade Streamlit analytics dashboard built for monitoring and analyzing ferry ticket operations from 2015–2025 using interactive visualizations, KPI tracking, congestion detection, and operational efficiency analytics.

This project transforms raw ferry ticket transaction data into actionable operational intelligence using Python, Streamlit, Plotly, Pandas, and NumPy.

📌 Features
📊 Advanced KPI Monitoring

The dashboard calculates and visualizes operational KPIs such as:

Capacity Utilisation Ratio (CUR)
Congestion Pressure Index (CPI)
Idle Capacity Percentage (ICP)
Peak Strain Duration (PSD)
Operational Variability Score (OVS)
📈 Temporal Analytics

Analyze ferry activity across:

Hourly traffic patterns
Daily rolling averages
Monthly activity distribution
Year-over-year operational trends
🔥 Congestion & Idle Detection

Detect operational stress using:

Operational Load Index (OLI)
Congestion thresholds
Idle interval analysis
Monthly congestion trend monitoring
📅 Segmentation Analysis

Break down operational efficiency by:

Weekday vs Weekend
Seasonal performance
Shift-level utilization
Hour × Day heatmaps
🧹 Data Quality Monitoring

Built-in diagnostics include:

Missing interval detection
Zero-activity tracking
Negative anomaly detection
Descriptive statistics analysis
🛠️ Tech Stack
Technology	Purpose
Python	Core programming
Streamlit	Dashboard framework
Plotly	Interactive visualizations
Pandas	Data manipulation
NumPy	Numerical operations
📂 Project Structure
├── streamlit.py
├── Ferry tickets.csv
├── requirements.txt
└── README.md
⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/your-username/ferry-operations-dashboard.git
cd ferry-operations-dashboard
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Application
streamlit run streamlit.py
📦 Required Libraries

Create a requirements.txt file with:

streamlit
pandas
numpy
plotly
📁 Dataset Requirements

The dashboard expects a CSV file named:

Ferry tickets.csv

Required columns:

Column Name	Description
Timestamp	Date & time of transaction
Sales Count	Number of ticket sales
Redemption Count	Number of ticket redemptions
🧠 Feature Engineering

The project automatically generates:

Date & time features
Shift categorization
Seasonal classification
Operational Load Index (OLI)
Congestion flags
Idle flags
Rolling activity metrics
Redemption pressure metrics
🎨 UI Highlights
Dark futuristic theme
Interactive Plotly charts
Dynamic filtering sidebar
Responsive layout
KPI cards with color indicators
Multi-tab analytical structure
📊 Dashboard Sections
📈 Temporal Patterns
Hourly activity profile
Daily smoothing trends
Monthly distribution analysis
🔥 Congestion & Idle
OLI distribution
Congestion vs idle intervals
Monthly pressure trends
📅 Segmentation
Seasonal efficiency
Shift analysis
Weekday/weekend comparison
Activity heatmaps
📊 Trend Analysis
Annual KPI monitoring
Sales vs redemption trends
Operational variability tracking
🔍 Data Quality
Missing intervals
Null checks
Anomaly identification
Statistical summaries
🚀 Future Improvements

Potential upgrades:

Machine Learning demand forecasting
Real-time API integration
Predictive congestion alerts
Passenger flow optimization
Database integration (PostgreSQL/MySQL)
Authentication system
Cloud deployment
📸 Preview

Add screenshots or GIFs here after deployment.

Example:

![Dashboard Preview](images/dashboard.png)
🌐 Deployment Options

You can deploy this dashboard on:

Streamlit Community Cloud
Render
Railway
Hugging Face Spaces

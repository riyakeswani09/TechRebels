 AI-Based Network Attack Forecasting

## Overview

AI-Based Network Attack Forecasting is a proactive cybersecurity system developed for **Smart India Hackathon 2026 – Problem Statement 26153**.

Traditional Intrusion Detection Systems primarily focus on detecting attacks after suspicious activity has already occurred. This project takes a proactive approach by analysing **network traffic over time** to identify changing network behaviour and forecast the possible **next stage of an evolving cyberattack**.

The system processes historical network traffic, extracts relevant features, creates temporal network states, and uses AI-based forecasting to predict future attack behaviour. It also provides risk assessment, explainable predictions, and early-warning information to support faster defensive action.

### Core Concept

```text
Network Traffic History
          ↓
Feature Extraction
          ↓
Temporal Network States
          ↓
AI-Based Forecasting
          ↓
Future Attack State
          ↓
Risk & Explainability
          ↓
Early Warning
```

Instead of only asking:

"Is this traffic malicious?"

the system aims to answer:

> "Based on the current and previous network behaviour, what is likely to happen next?"

## Features

Network Traffic Analysis
  Processes network traffic data from structured datasets such as CSV files.

Feature Extraction
  Analyses traffic characteristics including packet rates, connection behaviour, ports, protocols and other network features.

Temporal Behaviour Analysis
  Converts network traffic into time-based sequences to capture how network behaviour changes over time.

Attack-State Forecasting
  Predicts the possible future stage of an evolving attack instead of only detecting the current activity.

Attack Progression Tracking
  Represents possible progression such as:
  `Normal → Reconnaissance → Scanning → Exploitation → Compromise`

Risk Assessment
  Assigns risk levels based on predicted attack behaviour and confidence.

Explainable Predictions
  Identifies important network features contributing to the forecast.

Early Warning System
  Generates alerts when the predicted behaviour indicates a potential future attack.

Interactive Dashboard
  Displays current network status, predicted attack stage, probability, risk level and contributing factors.

## Technology Stack

### Programming

* Python

### AI / Machine Learning

* Machine Learning
* LSTM / Time-Series Forecasting
* Scikit-learn
* Pandas
* NumPy

### Explainable AI

* SHAP / Feature Importance

### Data Processing

* CSV / Network Traffic Datasets
* Feature Engineering
* Time-Series Windowing

### Frontend / Dashboard

* Streamlit

### Development

* Git
* GitHub

## Architecture

```text
┌──────────────────────────────┐
│       NETWORK TRAFFIC        │
│          PCAP / CSV          │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│       PRE-PROCESSING         │
│   Cleaning & Normalization   │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│      FEATURE EXTRACTION      │
│ Packet / Flow Characteristics│
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│       TIME WINDOWING         │
│    Temporal Network States   │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│     AI FORECASTING MODEL     │
│       LSTM / ML Model        │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│     FUTURE ATTACK STATE      │
│ Prediction + Probability     │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│    RISK & EXPLAINABILITY     │
│ Severity + Key Features      │
└──────────────┬───────────────┘
               ↓
┌──────────────────────────────┐
│       EARLY WARNING          │
│     Dashboard & Alerts       │
└──────────────────────────────┘
```

## Attack Progression

The prototype models network behaviour as a sequence of states:

```text
NORMAL
   ↓
RECONNAISSANCE
   ↓
SCANNING
   ↓
EXPLOITATION
   ↓
COMPROMISE
```

The forecasting model analyses previous network states and predicts the possible next state.

For example:

```text
Previous States:
NORMAL → RECONNAISSANCE → SCANNING

Forecast:
EXPLOITATION

Probability:
82%

Risk:
HIGH
```

The system can additionally identify the network features that contributed to the prediction, providing an interpretable warning for defenders.

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/riyakeswani09/AI-Based-Network-Attack-Forecasting.git
cd AI-Based-Network-Attack-Forecasting
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

On Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser.

### 5. Upload Network Traffic

Upload the supported network traffic dataset through the dashboard and start the analysis.

The system will process the data and display:

* Current network state
* Predicted future attack state
* Prediction probability
* Risk level
* Attack progression
* Important contributing features
* Early-warning recommendations

## Project Goal

The goal of this project is to shift cybersecurity from **reactive attack detection to proactive attack forecasting**.

```text
Traditional Approach
Traffic → Detection → Alert

Our Approach
Traffic History
      ↓
Current Network State
      ↓
Temporal Analysis
      ↓
Future State Prediction
      ↓
Attack Progression
      ↓
Risk Assessment
      ↓
Explainable Early Warning
```

## Smart India Hackathon 2026

**Problem Statement:** 26153
**Title:** AI-Based Network Attack Forecasting from Network Traffic Data
**Theme:** Blockchain & Cybersecurity
**Category:** Software

## Team

### Tech Rebels

Developed as a Smart India Hackathon 2026 project focused on applying AI and temporal network analysis to proactive cybersecurity.

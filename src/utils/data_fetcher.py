"""
Data Ingestion and Preprocessing Utilities
==========================================
Connectors for Kenneth French Data Library (FF30), Yahoo Finance, and FRED.
"""

from typing import Optional
import urllib.request
import zipfile
import io
import pandas as pd
import numpy as np


def download_fama_french_30_industry(start_year: int = 1990) -> pd.DataFrame:
    """
    Downloads Fama-French 30 Industry Portfolios (Daily) directly from Kenneth French's repository.
    """
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/30_Industry_Portfolios_daily_CSV.zip"
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            lines = [line.decode("utf-8", errors="ignore") for line in f.readlines()]

    # Locate start of Average Value Weighted Returns
    start_idx = 0
    for idx, line in enumerate(lines):
        if "Average Value Weighted Returns -- Daily" in line or "Average Value Weighted Returns" in line:
            start_idx = idx + 1
            break

    # Read data into DataFrame
    data_lines = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped or "Average Equal Weighted" in stripped:
            break
        data_lines.append(stripped)

    # Convert to DataFrame
    header = [col.strip() for col in data_lines[0].split(",")]
    if not header[0]:
        header[0] = "Date"

    rows = [[float(v.strip()) for v in line.split(",")] for line in data_lines[1:] if len(line.split(",")) == len(header)]
    df = pd.DataFrame(rows, columns=header)
    df["Date"] = pd.to_datetime(df["Date"].astype(int).astype(str), format="%Y%m%d")
    df.set_index("Date", inplace=True)

    # Filter by start year and convert percentage returns to decimal
    df = df[df.index.year >= start_year] / 100.0
    # Clean missing values (-99.99 or -999 in Ken French data)
    df = df.replace([-9.999, -99.99, -999.0], np.nan).ffill().fillna(0.0)

    return df


def download_fama_french_49_industry(start_year: int = 1990) -> pd.DataFrame:
    """
    Downloads Fama-French 49 Industry Portfolios (Daily) directly from Kenneth French's repository.
    """
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip"
    headers = {"User-Agent": "Mozilla/5.0"}
    req = urllib.request.Request(url, headers=headers)

    with urllib.request.urlopen(req) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            lines = [line.decode("utf-8", errors="ignore") for line in f.readlines()]

    start_idx = 0
    for idx, line in enumerate(lines):
        if "Average Value Weighted Returns -- Daily" in line or "Average Value Weighted Returns" in line:
            start_idx = idx + 1
            break

    data_lines = []
    for line in lines[start_idx:]:
        stripped = line.strip()
        if not stripped or "Average Equal Weighted" in stripped:
            break
        data_lines.append(stripped)

    header = [col.strip() for col in data_lines[0].split(",")]
    if not header[0]:
        header[0] = "Date"

    rows = [[float(v.strip()) for v in line.split(",")] for line in data_lines[1:] if len(line.split(",")) == len(header)]
    df = pd.DataFrame(rows, columns=header)
    df["Date"] = pd.to_datetime(df["Date"].astype(int).astype(str), format="%Y%m%d")
    df.set_index("Date", inplace=True)

    df = df[df.index.year >= start_year] / 100.0
    df = df.replace([-9.999, -99.99, -999.0], np.nan).ffill().fillna(0.0)

    return df



def download_macro_controls(start_year: int = 1990) -> pd.DataFrame:
    """
    Downloads VIX and key macroeconomic control indicators.
    Uses yfinance / FRED direct public CSV endpoints with retries.
    """
    import yfinance as yf
    import time

    print("Fetching VIX historical data via Yahoo Finance...")
    vix_df = yf.download("^VIX", start=f"{start_year}-01-01", progress=False, auto_adjust=True)
    if isinstance(vix_df.columns, pd.MultiIndex):
        vix_series = vix_df["Close"]["^VIX"]
    else:
        vix_series = vix_df["Close"]
    vix_series.name = "VIX"

    controls_df = pd.DataFrame(index=vix_series.index)
    controls_df["VIX"] = vix_series

    # Download Treasury 10Y-3M Spread from FRED
    print("Fetching Treasury 10Y-3M Spread from FRED...")
    fred_url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=T10Y3M"
    req = urllib.request.Request(fred_url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
    
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                t10y3m_df = pd.read_csv(resp, parse_dates=["DATE"], index_col="DATE")
            t10y3m_df.rename(columns={"T10Y3M": "T10Y3M"}, inplace=True)
            t10y3m_df["T10Y3M"] = pd.to_numeric(t10y3m_df["T10Y3M"], errors="coerce")
            controls_df = controls_df.join(t10y3m_df, how="left").ffill().bfill()
            break
        except Exception as e:
            if attempt == 2:
                print(f"Warning: Could not fetch FRED T10Y3M ({e}). Proceeding with VIX only.")
            time.sleep(2)

    return controls_df.ffill().bfill()



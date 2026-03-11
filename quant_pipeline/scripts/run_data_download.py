# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 10:41:11 2026

@author: jimya
"""

import os
import re
import time
import shutil
import zipfile
import datetime as dt
from pathlib import Path
from typing import Optional, List, Tuple

import pandas as pd
import databento as db


# =====================
# CONFIG
# =====================

DATABENTO_API = os.getenv("DATABENTO_API_KEY", "db-BGUEuawFmbwreVAFXyi3C9c4BKNEp")

DATASET = "XNAS.ITCH"            # for data before 20240701, but very slow
SCHEMA = "ohlcv-1d"


# 你可以随时增删；建议后续用“Top N by dollar volume”的动态 universe
# TICKERS = [
#     # Mega-cap Tech
#     "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "ORCL", "ADBE", "CRM", "AMD", "INTC",
#     # Semis / Hardware
#     "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC", "ASML", "ARM", "MRVL",
#     # Consumer
#     "COST", "WMT", "HD", "LOW", "NKE", "SBUX", "MCD", "TGT",
#     # Financials
#     "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "V", "MA", "PYPL",
#     # Healthcare
#     "UNH", "LLY", "JNJ", "MRK", "PFE", "ABBV", "TMO", "ABT", "DHR", "ISRG", "GILD", "AMGN",
#     # Industrials
#     "CAT", "DE", "BA", "GE", "HON", "UPS", "LMT", "RTX", "NOC",
#     # Energy
#     "XOM", "CVX", "COP", "SLB",
#     # Communication / Media
#     "NFLX", "DIS", "CMCSA",
#     # Utilities / Staples / Others
#     "PG", "KO", "PEP", "PM", "MO", "CL", "KMB",
#     # ETFs (可选，便于对照/基准)
#     "SPY", "QQQ", "IWM",
# ]
# S%P 500 universe 2026/02/04
TICKERS = [
    "MMM","AOS","ABT","ABBV","ACN","ATVI","AYI","ADBE","AAP","AMD","AES","AET","AMG","AFL","A",
    "APD","AKAM","ALK","ALB","ARE","ALXN","ALGN","ALLE","AGN","ADS","ALL","GOOGL","GOOG","MO","AMZN",
    "AEE","AAL","AEP","AXP","AIG","AMT","AWK","AMP","ABC","AME","AMGN","APH","APC","ADI","ANDV","ANSS",
    "ANTM","AON","APA","AIV","AAPL","AMAT","APTV","ADM","ARNC","AJG","AIZ","T","ADSK","ADP","AZO","AVB",
    "AVY","BHGE","BLL","BAC","BAX","BDX","BRK.B","BBY","BIIB","BLK","HRB","BA","BWA","BXP","BSX","BHF",
    "BMY","AVGO","BF.B","CHRW","CA","COG","CDNS","CPB","COF","CAH","KMX","CCL","CAT","CBOE","CBG","CBS",
    "CELG","CNC","CNP","CTL","CERN","CF","SCHW","CHTR","CHK","CVX","CMG","CB","CHD","CI","XEC","CINF",
    "CTAS","CSCO","C","CFG","CTXS","CME","CMS","KO","CTSH","CL","CMCSA","CMA","CAG","CXO","COP","ED",
    "STZ","GLW","COST","COTY","CCI","CSRA","CSX","CMI","CVS","DHI","DHR","DRI","DVA","DE","DAL","XRAY",
    "DVN","DLR","DFS","DISCA","DISCK","DISH","DG","DLTR","D","DOV","DWDP","DPS","DTE","DUK","DRE","DXC",
    "ETFC","EMN","ETN","EBAY","ECL","EIX","EW","EA","EMR","ETR","EVHC","EOG","EQT","EFX","EQIX","EQR",
    "ESS","EL","RE","ES","EXC","EXPE","EXPD","ESRX","FLIR","FLS","FLR","FMC","FL","F","FTV","FBHS","BEN",
    "FCX","GPS","GRMN","IT","GD","GE","GGP","GIS","GM","GPC","GILD","GPN","GS","GT","GWW","HAL","HBI",
    "HOG","HRS","HIG","HAS","HCA","HCP","HP","HSIC","HES","HPE","HLT","HOLX","HD","HON","HRL","HST","HPQ",
    "HUM","HBAN","HII","IDXX","INFO","ITW","ILMN","INCY","IR","INTC","ICE","IBM","IP","IPG","IFF","INTU",
    "ISRG","IVZ","IQV","IRM","JBHT","JEC","SJM","JNJ","JCI","JPM","JNPR","KSU","K","KEY","KMB","KIM","KMI",
    "KLAC","KSS","KHC","KR","LB","LLL","LH","LRCX","LEG","LEN","LUK","LLY","LNC","LKQ","LMT","L","LOW","LYB",
    "MTB","MAC","M","MRO","MPC","MAR","MMC","MLM","MAS","MA","MAT","MKC","MCD","MCK","MDT","MRK","MET","MTD",
    "MGM","KORS","MCHP","MU","MSFT","MAA","MHK","TAP","MDLZ","MON","MNST","MCO","MS","MSI","MYL","NDAQ","OMC",
    "OKE","ORCL","PCAR","PKG","PH","PDCO","PAYX","PYPL","PNR","PBCT","PEP","PKI","PRGO","PFE","PCG","PM","PSX",
    "PNW","PXD","PNC","RL","PPG","PPL","PX","PCLN","URI","UTX","UHS","UNM","VFC","VLO","VAR","VTR","VRSN","VRSK",
    "VZ","VRTX","WBA","WMT","WAT","WELL","WEC","WFC","HCN","WDC","WU","WRK","WY","WHR","WMB","WLTW","WEC","WELL",
    "WST","WTW","WYNN","XEL","XRX","XLNX","XYL","YUM","ZBH","ZION","ZTS"
]

# TODO 改这里 
BASE_SAVE_DIR = str(Path(__file__).resolve().parents[2] / "data" / "equities")

# TODO 改这里 6
# 空文件时回补多少天；非空则 latest+1 -> yesterday
LOOKBACK_DAYS = 365 * 7

# 每个 batch job 覆盖多少个日（ohlcv-1d 很小，可大一些）
MAX_DAYS_PER_JOB = 10000

# 临时目录
TMP_DIRNAME = "__tmp_download"

client = db.Historical(DATABENTO_API)


# =====================
# TIME UTILS (UTC-day for ohlcv-1d)
# =====================

def utc_day_start(ts_date: dt.date) -> pd.Timestamp:
    return pd.Timestamp(ts_date).tz_localize("UTC")


def to_iso_z(ts: pd.Timestamp) -> str:
    return ts.to_pydatetime().isoformat().replace("+00:00", "Z")


# =====================
# JOB WAIT
# =====================

def wait_for_job(client, job_id, timeout=7200, interval=10):
    elapsed = 0
    try:
        while elapsed < timeout:
            jobs = client.batch.list_jobs()
            job = next((j for j in jobs if j["id"] == job_id), None)
            if job:
                state = job["state"]
                if state == "done":
                    print(f"{elapsed:3d}s: 当前状态 {state}")
                    print("开始下载")
                    return job
                elif state in ("failed", "cancelled"):
                    raise RuntimeError(f"Job 失败: {job}")
                print(f"{elapsed:3d}s: 当前状态 {state}")
            else:
                print("Job 暂未出现在列表中...")

            time.sleep(interval)
            elapsed += interval

        raise TimeoutError("等待 Databento 任务超时")
    except KeyboardInterrupt:
        print(f"\n[Interrupt] 你中断了等待。job_id={job_id} 可能仍在队列/执行中，可稍后继续 download。")
        raise


# =====================
# SCAN LATEST DATE FROM SINGLE FILE
# =====================

def out_single_file_path(schema_dir: str, ticker: str) -> str:
    dataset_tag = DATASET.lower().replace(".", "-")
    return os.path.join(schema_dir, f"{dataset_tag}_{ticker}.{SCHEMA}.csv")


def scan_latest_date_from_single_file(single_path: str) -> Optional[dt.date]:
    if not os.path.exists(single_path):
        return None
    try:
        df = pd.read_csv(single_path, usecols=["ts_event"])
        ts = pd.to_datetime(df["ts_event"], utc=True, errors="coerce").dropna()
        if ts.empty:
            return None
        return ts.max().date()
    except Exception as e:
        print(f"[scan] read latest from {single_path} failed: {e}")
        return None


def scan_trade_dates_for_download_single(single_path: str, lookback_days: int) -> List[dt.date]:
    latest = scan_latest_date_from_single_file(single_path)

    yesterday = dt.date.today() - dt.timedelta(days=1)
    if latest and latest >= yesterday:
        print("数据已更新至昨日，无需下载。")
        return []

    start_date = latest + dt.timedelta(days=1) if latest else (yesterday - dt.timedelta(days=lookback_days))
    dates = pd.date_range(start=start_date, end=yesterday, freq="D")
    trade_dates = [d.date() for d in dates]
    trade_dates = [d for d in trade_dates if d.weekday() not in (5, 6)]  # Sat/Sun
    print(f"待下载 trade_date: {[d.strftime('%Y-%m-%d') for d in trade_dates]}")
    return trade_dates


# =====================
# FILE / DIR UTILS
# =====================

def _remove_path_if_exists(p: str) -> None:
    if not os.path.exists(p):
        return
    if os.path.isdir(p):
        shutil.rmtree(p, ignore_errors=True)
    else:
        try:
            os.remove(p)
        except Exception:
            pass


def split_list(xs: List[dt.date], max_n: int) -> List[List[dt.date]]:
    out = []
    i = 0
    while i < len(xs):
        out.append(xs[i:i + max_n])
        i += max_n
    return out


# =====================
# NORMALIZE DOWNLOAD -> TEMP
# =====================

def normalize_download_to_temp(download_path: str, temp_out_dir: str) -> None:
    """
    download_path could be a .zip file OR a directory.
    Extract if zip; find ohlcv-1d market CSV(s); parse ts_event to UTC.
    """
    download_path = Path(download_path)
    temp_out_dir = Path(temp_out_dir)
    temp_out_dir.mkdir(parents=True, exist_ok=True)

    extract_dir: Path
    made_extract = False

    if download_path.is_dir():
        extract_dir = download_path
        print(f"[normalize] Detected directory download: {extract_dir}")
    else:
        extract_dir = temp_out_dir / f"__extract_{download_path.stem}"
        _remove_path_if_exists(str(extract_dir))
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(download_path, "r") as zf:
            zf.extractall(extract_dir)
        made_extract = True
        print(f"[normalize] Extracted zip to: {extract_dir}")

    pattern = re.compile(r"(ohlcv-1d).*\.csv$", re.IGNORECASE)
    csv_files = [
        f for f in extract_dir.rglob("*.csv")
        if pattern.search(f.name)
        and not any(k in f.name.lower() for k in ["symbology", "metadata", "manifest"])
    ]

    if not csv_files:
        print(f"[normalize] No market CSV found in {extract_dir}")
    else:
        for csv_file in csv_files:
            dest = temp_out_dir / csv_file.name
            try:
                df = pd.read_csv(csv_file)

                if "ts_event" in df.columns:
                    df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
                if "ts_recv" in df.columns:
                    df["ts_recv"] = pd.to_datetime(df["ts_recv"], utc=True, errors="coerce")

                df.to_csv(dest, index=False)
                print(f"[normalize] temp export: {dest.name}")
            except Exception as e:
                print(f"[normalize] failed {csv_file.name}: {e}")

    if made_extract:
        shutil.rmtree(extract_dir, ignore_errors=True)

    # cleanup download artifact
    if download_path.exists():
        if download_path.is_dir():
            shutil.rmtree(download_path, ignore_errors=True)
        else:
            download_path.unlink(missing_ok=True)


# =====================
# APPEND TEMP -> SINGLE FILE
# =====================

def append_temp_to_single_file(tmp_dir: str, single_path: str) -> Tuple[str, str]:
    """
    读取 temp 中 ohlcv-1d 的 CSV（一般很小），追加到 single_path，
    并按 ts_event 去重排序。
    """
    temp_files = [str(p) for p in Path(tmp_dir).glob("*ohlcv-1d*.csv")]
    if not temp_files:
        temp_files = [str(p) for p in Path(tmp_dir).glob("*.csv")]

    if not temp_files:
        return single_path, "missing_temp_files"

    dfs = []
    for fp in temp_files:
        try:
            df = pd.read_csv(fp)
            if "ts_event" in df.columns:
                df["ts_event"] = pd.to_datetime(df["ts_event"], utc=True, errors="coerce")
            dfs.append(df)
        except Exception as e:
            print(f"[append] read temp failed {fp}: {e}")

    if not dfs:
        return single_path, "read_temp_failed"

    df_new = pd.concat(dfs, ignore_index=True)
    if df_new.empty:
        return single_path, "nodata"

    if os.path.exists(single_path):
        try:
            df_old = pd.read_csv(single_path)
            if "ts_event" in df_old.columns:
                df_old["ts_event"] = pd.to_datetime(df_old["ts_event"], utc=True, errors="coerce")
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception as e:
            print(f"[append] read old failed, will overwrite: {e}")
            df_all = df_new
    else:
        df_all = df_new

    if "ts_event" in df_all.columns:
        df_all = df_all.dropna(subset=["ts_event"])
        df_all = df_all.sort_values("ts_event")
        df_all = df_all.drop_duplicates(subset=["ts_event"], keep="last")

    tmp_out = single_path + ".part"
    os.makedirs(os.path.dirname(single_path), exist_ok=True)
    df_all.to_csv(tmp_out, index=False)
    os.replace(tmp_out, single_path)

    return single_path, f"ok(rows={len(df_all)})"


# =====================
# DOWNLOAD PIPELINE
# =====================

def download_daily_ohlcv_for_ticker(
    ticker: str,
    trade_dates: List[dt.date],
    schema_dir: str,
    tmp_dir: str,
):
    trade_dates = sorted(trade_dates)
    if not trade_dates:
        print(f"[{ticker}] trade_dates empty, skip")
        return

    chunks = split_list(trade_dates, MAX_DAYS_PER_JOB)
    print(f"[{ticker}] 共 {len(trade_dates)} 天 -> {len(chunks)} 个 chunk")

    single_path = out_single_file_path(schema_dir, ticker)

    for td_list in chunks:
        start_td, end_td = td_list[0], td_list[-1]

        start_utc = utc_day_start(start_td)
        end_utc = utc_day_start(end_td) + pd.Timedelta(days=1)

        print(f"\n下载 {ticker} {SCHEMA}: {start_td} -> {end_td} (count={len(td_list)})")
        print(f"  UTC window: {start_utc} -> {end_utc}")

        try:
            job = client.batch.submit_job(
                dataset=DATASET,
                symbols=[ticker],
                schema=SCHEMA,
                encoding="csv",
                compression="none",
                stype_in="raw_symbol",
                start=to_iso_z(start_utc),
                end=to_iso_z(end_utc),
            )
        except Exception as e:
            print(f"[{ticker}] 提交失败或没有数据: {e}")
            continue

        job_id = job["id"]
        print(f"[{ticker}] Job 已提交: {job_id}")
        wait_for_job(client, job_id)

        # download -> tmp
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        download_target = os.path.join(tmp_dir, f"__download_{job_id}")
        _remove_path_if_exists(download_target)

        client.batch.download(job_id, download_target)
        print(f"[{ticker}] 已下载到临时目标: {download_target}")

        normalize_download_to_temp(download_target, tmp_dir)

        out_path, status = append_temp_to_single_file(tmp_dir=tmp_dir, single_path=single_path)
        print(f"[{ticker}] Append -> {out_path} : {status}")

        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[{ticker}] Cleanup: 已删除临时目录 {tmp_dir}")


# =====================
# MAIN
# =====================

def main():
    for ticker in TICKERS:
        schema_dir = os.path.join(BASE_SAVE_DIR, ticker, SCHEMA)
        Path(schema_dir).mkdir(parents=True, exist_ok=True)

        tmp_dir = os.path.join(schema_dir, TMP_DIRNAME)

        print("\n" + "=" * 80)
        print(f"[Ticker] {ticker}")
        print(f"[Dataset] {DATASET}  [Schema] {SCHEMA}")
        print(f"[Save Dir] {schema_dir}")

        single_path = out_single_file_path(schema_dir, ticker)

        trade_dates = scan_trade_dates_for_download_single(single_path, lookback_days=LOOKBACK_DAYS)
        if not trade_dates:
            continue

        download_daily_ohlcv_for_ticker(
            ticker=ticker,
            trade_dates=trade_dates,
            schema_dir=schema_dir,
            tmp_dir=tmp_dir,
        )


if __name__ == "__main__":
    main()

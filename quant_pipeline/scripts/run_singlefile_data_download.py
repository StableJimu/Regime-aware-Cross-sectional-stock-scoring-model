# -*- coding: utf-8 -*-
"""
Created on Wed Feb  4 22:47:47 2026

@author: jimya
"""
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

# S%P 500 universe 2026/02/04  removing dual share eg goog,fox,nwl
# TICKERS = [
#     "A", "AAPL", "ABBV", "ABNB", "ABT", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE", "AEP", "AES", "AFL", 
#     "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALK", "ALL", "AMT", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN", "AMP", 
#     "AMZN", "ANET", "AON", "AOS", "APA", "APD", "APH", "APO", "APP", "APTV", "ARE", "ARES", "ATO", "AVB", 
#     "AVGO", "AVY", "AWK", "AXP", "AZO", "BA", "BAC", "BALL", "BAX", "BBWI", "BBY", "BDX", "BEN", "BK", "BKR","BKNG", 
#     "BLDR", "BLK", "BMY", "BR", "BRK.B", "BRO", "BSX", "BWA", "BX", "BXP", "C", "CAG", "CAH", "CARR", "CAT", 
#     "CB", "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG", "CF", "CFG", "CHD", "CHRW", "CHTR", "CI", 
#     "CINF", "CL", "CLX", "CMA", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COO", "COP", "COST", 
#     "CPAY", "CPB", "CPRT", "CRH", "CRL", "CRM", "CRWD", "CSCO", "CSGP", "CSX", "CTAS", "CTRA", "CTSH", "CTVA", 
#     "CVNA", "CVS", "CVX", "CZR", "D", "DAL", "DASH", "DAY", "DD", "DE", "DELL", "DFS", "DG", "DGX", "DHI", 
#     "DHR", "DIS", "DLR", "DLTR", "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXCM", "EA", 
#     "EBAY", "ECL", "ED", "EFX", "EIX", "EG", "EL", "ELV", "EME", "EMN", "EMR", "EOG", "EPAM", "EQIX", "EQR", "EQT", 
#     "ERIE", "ES", "ESS", "ETN", "ETR", "ETSY", "EVRG", "EW", "EXC", "EXE", "EXPD", "EXPE", "EXR", "F", "FAST", 
#     "FCX", "FDS", "FDX", "FE", "FIS", "FITB", "FIX", "FLEX", "FMC", "FOXA", "FRT", "FSLR", "FTNT", "FTV", 
#     "GD", "GDDY", "GE", "GEN", "GEV", "GILD", "GIS", "GL", "GLW", "GM", "GNRC", "GOOGL", "GPC", "GPN", "GRMN", 
#     "GS", "GWW", "HAL", "HAS", "HBAN", "HCA", "HD", "HIG", "HII", "HLT", "HOLX", "HON", "HOOD", "HPE", "HPQ", 
#     "HRL", "HSIC", "HST", "HSY", "HUM", "HWM", "IBKR", "IBM", "ICE", "IDXX", "IEX", "IFF", "ILMN", "INCY", "INTC", 
#     "INTU", "INVH", "IP", "IPG", "IQV", "IR", "IRM", "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL", "JCI", 
#     "JKHY", "JNJ", "JNPR", "JPM","K", "KDP", "KEY", "KEYS", "KHC", "KIM", "KKR", "KLAC", "KMB", "KMI", "KO", "KR", "KVUE",
#     "L", "LDOS", "LEN", "LH", "LHX", "LIN", "LKQ", "LLY", "LMT", "LNC", "LNT", "LOW", "LRCX", "LULU", "LUV", 
#     "LVS", "LW", "LYB", "LYV", "MA", "MAA", "MAR", "MAS", "MAT", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", 
#     "MET", "META", "MGM", "MHK", "MKC", "MKTX", "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC", "MPWR", 
#     "MRK", "MRNA", "MS", "MSCI", "MSFT", "MSI", "MTB", "MTCH", "MTD", "MU", "NCLH", "NDAQ", "NDSN", "NEE", "NEM", "NFLX", 
#     "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP", "NTR", "NTRS", "NUE", "NVDA", "NVR", "NWSA", "NXPI", "O", 
#     "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PARA", "PAYC", "PAYX", "PCAR", "PCG", 
#     "PEG", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM", "PKG", "PLD", "PLTR", "PM", "PNC", "PNR", "PNW", 
#     "PODD", "POOL", "PPG", "PPL", "PRU", "PSA", "PTC", "PVH", "PWR", "PYPL", "QCOM", "QRVO", "RCL", "REG", 
#     "REGN", "RF", "RHI", "RJF", "RL", "RMD", "ROK", "ROL", "ROP", "ROST", "RSG", "RTX", "RVTY", "SBAC", "SBUX", 
#     "SCHW", "SHW", "SIRI", "SJM", "SLB", "SMCI", "SNA", "SNDK", "SNPS", "SO", "SOLV", "SPG", "SPGI", "SRE", "STE", 
#     "STLD", "STT", "STX", "SW", "SYF", "SYK", "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TGT", 
#     "TJX", "TKO", "TMO", "TMUS", "TPR", "TRGP", "TRMB", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTD", "TTWO", 
#     "TXN", "TXT", "TYL", "UAL", "UBER", "UDR", "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V", "VFC", 
#     "VICI", "VLO", "VMC", "VRTX", "VST", "VTR", "VTRS", "VZ", "WAB", "WAT", "WBD", "WDC", "WEC", "WELL", "WFC", "WM", 
#     "WMB", "WMT", "WRB", "WSM", "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XRAY", "XYL", "XYZ", "YUM", 
#     "ZBH", "ZBRA", "ZION", "ZTS"
# ]

TICKERS = ["SPY","VTI","DIA","QQQ"]

# TODO 改这里（建议指向一个“数据根目录”，下面会在此目录下写 ALL CSV）
BASE_SAVE_DIR = str(Path(__file__).resolve().parents[2] / "data" / "raw")

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
# OUTPUT PATH (SINGLE BIG CSV)
# =====================

def out_all_file_path(base_dir: str) -> str:
    dataset_tag = DATASET.lower().replace(".", "-")
    # 单一大文件：只保留一个 header
    return os.path.join(base_dir, f"{dataset_tag}_ALL.{SCHEMA}.csv")


# =====================
# SCAN LATEST DATE FROM BIG FILE
# =====================

def scan_latest_date_from_all_file(all_path: str) -> Optional[dt.date]:
    if not os.path.exists(all_path):
        return None
    try:
        df = pd.read_csv(all_path, usecols=["ts_event"])
        ts = pd.to_datetime(df["ts_event"], utc=True, errors="coerce").dropna()
        if ts.empty:
            return None
        return ts.max().date()
    except Exception as e:
        print(f"[scan] read latest from {all_path} failed: {e}")
        return None

def scan_trade_dates_for_download_all(all_path: str, lookback_days: int) -> List[dt.date]:
    latest = scan_latest_date_from_all_file(all_path)

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
# APPEND TEMP -> BIG FILE
# =====================

def append_temp_to_all_file(tmp_dir: str, all_path: str) -> Tuple[str, str]:
    """
    读取 temp 中 ohlcv-1d 的 CSV，追加到 all_path，
    并按 (symbol, ts_event) 去重排序；all_path 只有一个 header。
    """
    temp_files = [str(p) for p in Path(tmp_dir).glob("*ohlcv-1d*.csv")]
    if not temp_files:
        temp_files = [str(p) for p in Path(tmp_dir).glob("*.csv")]

    if not temp_files:
        return all_path, "missing_temp_files"

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
        return all_path, "read_temp_failed"

    df_new = pd.concat(dfs, ignore_index=True)
    if df_new.empty:
        return all_path, "nodata"

    # 兜底：如果没有 symbol 列，就尝试用 Databento 常见字段名
    if "symbol" not in df_new.columns:
        if "raw_symbol" in df_new.columns:
            df_new = df_new.rename(columns={"raw_symbol": "symbol"})
        elif "instrument_id" in df_new.columns:
            # 不建议长期用 instrument_id，但至少不至于无法去重
            df_new = df_new.rename(columns={"instrument_id": "symbol"})

    if os.path.exists(all_path):
        try:
            df_old = pd.read_csv(all_path)
            if "ts_event" in df_old.columns:
                df_old["ts_event"] = pd.to_datetime(df_old["ts_event"], utc=True, errors="coerce")
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        except Exception as e:
            print(f"[append] read old failed, will overwrite: {e}")
            df_all = df_new
    else:
        df_all = df_new

    # 去重逻辑：优先 (symbol, ts_event)，否则退化为 ts_event
    if "ts_event" in df_all.columns:
        df_all = df_all.dropna(subset=["ts_event"])

        if "symbol" in df_all.columns:
            df_all = df_all.sort_values(["symbol", "ts_event"])
            df_all = df_all.drop_duplicates(subset=["symbol", "ts_event"], keep="last")
        else:
            df_all = df_all.sort_values("ts_event")
            df_all = df_all.drop_duplicates(subset=["ts_event"], keep="last")

    tmp_out = all_path + ".part"
    os.makedirs(os.path.dirname(all_path), exist_ok=True)
    df_all.to_csv(tmp_out, index=False)
    os.replace(tmp_out, all_path)

    return all_path, f"ok(rows={len(df_all)})"


# =====================
# DOWNLOAD PIPELINE (ALL TICKERS)
# =====================

def download_daily_ohlcv_for_universe(
    symbols: List[str],
    trade_dates: List[dt.date],
    base_dir: str,
    tmp_dir: str,
    all_path: str,
):
    trade_dates = sorted(trade_dates)
    if not trade_dates:
        print("[ALL] trade_dates empty, skip")
        return

    chunks = split_list(trade_dates, MAX_DAYS_PER_JOB)
    print(f"[ALL] 共 {len(trade_dates)} 天 -> {len(chunks)} 个 chunk")
    print(f"[ALL] symbols={len(symbols)}")

    for td_list in chunks:
        start_td, end_td = td_list[0], td_list[-1]

        start_utc = utc_day_start(start_td)
        end_utc = utc_day_start(end_td) + pd.Timedelta(days=1)

        print(f"\n下载 ALL {SCHEMA}: {start_td} -> {end_td} (count={len(td_list)})")
        print(f"  UTC window: {start_utc} -> {end_utc}")

        try:
            job = client.batch.submit_job(
                dataset=DATASET,
                symbols=symbols,          # ✅ 核心提速：一次 job 覆盖全部 tickers
                schema=SCHEMA,
                encoding="csv",
                compression="none",
                stype_in="raw_symbol",
                start=to_iso_z(start_utc),
                end=to_iso_z(end_utc),
            )
        except Exception as e:
            print(f"[ALL] 提交失败或没有数据: {e}")
            continue

        job_id = job["id"]
        print(f"[ALL] Job 已提交: {job_id}")
        wait_for_job(client, job_id)

        # download -> tmp
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        download_target = os.path.join(tmp_dir, f"__download_{job_id}")
        _remove_path_if_exists(download_target)

        client.batch.download(job_id, download_target)
        print(f"[ALL] 已下载到临时目标: {download_target}")

        normalize_download_to_temp(download_target, tmp_dir)

        out_path, status = append_temp_to_all_file(tmp_dir=tmp_dir, all_path=all_path)
        print(f"[ALL] Append -> {out_path} : {status}")

        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"[ALL] Cleanup: 已删除临时目录 {tmp_dir}")


# =====================
# MAIN
# =====================

def main():
    base_dir = BASE_SAVE_DIR
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    tmp_dir = os.path.join(base_dir, TMP_DIRNAME)
    all_path = out_all_file_path(base_dir)

    print("\n" + "=" * 80)
    print("[Universe] S&P500")
    print(f"[Dataset] {DATASET}  [Schema] {SCHEMA}")
    print(f"[Save File] {all_path}")
    print(f"[Tmp Dir] {tmp_dir}")

    trade_dates = scan_trade_dates_for_download_all(all_path, lookback_days=LOOKBACK_DAYS)
    if not trade_dates:
        return

    download_daily_ohlcv_for_universe(
        symbols=TICKERS,
        trade_dates=trade_dates,
        base_dir=base_dir,
        tmp_dir=tmp_dir,
        all_path=all_path,
    )


if __name__ == "__main__":
    main()

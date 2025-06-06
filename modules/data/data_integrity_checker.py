#!/usr/bin/env python3
"""
Data Integrity Checker for TradingJii

Verifica l'integrità dei dati OHLCV scaricati e fornisce statistiche dettagliate.
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from colorama import Fore, Style
from modules.utils.config import DB_FILE

class DataIntegrityResult:
    """Classe per contenere i risultati della verifica di integrità."""
    
    def __init__(self, symbol: str, timeframe: str):
        self.symbol = symbol
        self.timeframe = timeframe
        self.total_records = 0
        self.first_date = None
        self.last_date = None
        self.expected_records = 0
        self.completeness_pct = 0.0
        self.data_gaps = []
        self.integrity_issues = []
        self.quality_score = 0.0
        self.status = "UNKNOWN"
        self.has_indicators = False
        self.indicators_coverage_pct = 0.0
        
    def is_healthy(self) -> bool:
        """Determina se i dati sono considerati sani."""
        return (self.quality_score >= 85 and 
                self.completeness_pct >= 95 and 
                len(self.integrity_issues) == 0)

def verify_data_integrity_post_download(symbol: str, timeframe: str) -> DataIntegrityResult:
    """
    Verifica l'integrità dei dati immediatamente dopo il download.
    
    Args:
        symbol: Simbolo della crypto
        timeframe: Timeframe dei dati
        
    Returns:
        DataIntegrityResult con i risultati della verifica
    """
    result = DataIntegrityResult(symbol, timeframe)
    
    try:
        # Connessione al database
        table_name = f"market_data_{timeframe}"
        
        with sqlite3.connect(DB_FILE) as conn:
            # 1. STATISTICHE BASE
            query = f"""
                SELECT COUNT(*) as count,
                       MIN(timestamp) as first_date,
                       MAX(timestamp) as last_date,
                       MIN(open) as min_open,
                       MAX(high) as max_high,
                       MIN(low) as min_low,
                       MAX(close) as max_close,
                       MIN(volume) as min_volume,
                       MAX(volume) as max_volume
                FROM {table_name}
                WHERE symbol = ?
            """
            
            df_stats = pd.read_sql_query(query, conn, params=[symbol])
            
            if len(df_stats) == 0 or df_stats['count'].iloc[0] == 0:
                result.status = "NO_DATA"
                return result
            
            result.total_records = int(df_stats['count'].iloc[0])
            result.first_date = datetime.strptime(df_stats['first_date'].iloc[0], '%Y-%m-%dT%H:%M:%S')
            result.last_date = datetime.strptime(df_stats['last_date'].iloc[0], '%Y-%m-%dT%H:%M:%S')
            
            # 2. CALCOLO COMPLETEZZA
            timeframe_minutes = {
                '1m': 1, '5m': 5, '15m': 15, '30m': 30,
                '1h': 60, '4h': 240, '1d': 1440
            }
            
            if timeframe in timeframe_minutes:
                total_minutes = (result.last_date - result.first_date).total_seconds() / 60
                result.expected_records = int(total_minutes / timeframe_minutes[timeframe]) + 1
                result.completeness_pct = (result.total_records / result.expected_records) * 100
            
            # 3. VERIFICA INTEGRITÀ DATI
            integrity_query = f"""
                SELECT COUNT(*) as invalid_count
                FROM {table_name}
                WHERE symbol = ? AND (
                    high < low OR
                    high < open OR
                    high < close OR
                    low > open OR
                    low > close OR
                    volume < 0 OR
                    open <= 0 OR
                    high <= 0 OR
                    low <= 0 OR
                    close <= 0
                )
            """
            
            invalid_data = pd.read_sql_query(integrity_query, conn, params=[symbol])
            invalid_count = invalid_data['invalid_count'].iloc[0]
            
            if invalid_count > 0:
                result.integrity_issues.append(f"Found {invalid_count} records with invalid OHLC data")
            
            # 4. DETECTION GAP TEMPORALI
            gaps_query = f"""
                SELECT 
                    LAG(timestamp) OVER (ORDER BY timestamp) as prev_timestamp,
                    timestamp as current_timestamp
                FROM {table_name}
                WHERE symbol = ?
                ORDER BY timestamp
            """
            
            df_gaps = pd.read_sql_query(gaps_query, conn, params=[symbol])
            
            if len(df_gaps) > 1:
                df_gaps = df_gaps.dropna()
                df_gaps['prev_timestamp'] = pd.to_datetime(df_gaps['prev_timestamp'])
                df_gaps['current_timestamp'] = pd.to_datetime(df_gaps['current_timestamp'])
                df_gaps['time_diff'] = df_gaps['current_timestamp'] - df_gaps['prev_timestamp']
                
                expected_diff = timedelta(minutes=timeframe_minutes.get(timeframe, 60))
                tolerance = expected_diff * 1.5  # 50% di tolleranza
                
                gaps = df_gaps[df_gaps['time_diff'] > tolerance]
                
                for _, gap in gaps.iterrows():
                    gap_hours = gap['time_diff'].total_seconds() / 3600
                    result.data_gaps.append({
                        'start': gap['prev_timestamp'],
                        'end': gap['current_timestamp'],
                        'duration_hours': gap_hours
                    })
            
            # 5. VERIFICA INDICATORI TECNICI - adesso nella stessa tabella
            indicators_query = f"""
                SELECT COUNT(*) as indicators_count,
                       COUNT(CASE WHEN rsi14 IS NOT NULL THEN 1 END) as rsi_count,
                       COUNT(CASE WHEN ema20 IS NOT NULL THEN 1 END) as ema_count,
                       COUNT(CASE WHEN macd IS NOT NULL THEN 1 END) as macd_count
                FROM {table_name}
                WHERE symbol = ?
            """
            
            df_indicators = pd.read_sql_query(indicators_query, conn, params=[symbol])
            
            if len(df_indicators) > 0 and df_indicators['indicators_count'].iloc[0] > 0:
                result.has_indicators = True
                indicators_count = df_indicators['indicators_count'].iloc[0]
                non_null_count = (df_indicators['rsi_count'].iloc[0] + 
                                df_indicators['ema_count'].iloc[0] + 
                                df_indicators['macd_count'].iloc[0])
                if indicators_count > 0:
                    result.indicators_coverage_pct = (non_null_count / (indicators_count * 3)) * 100
            else:
                result.has_indicators = False
            
            # 6. CALCOLO QUALITY SCORE
            score = 100.0
            
            # Penalità per problemi di integrità
            if invalid_count > 0:
                score -= min(20, invalid_count * 2)
            
            # Penalità per completezza
            if result.completeness_pct < 100:
                score -= (100 - result.completeness_pct) * 0.3
            
            # Penalità per gap
            if len(result.data_gaps) > 0:
                score -= min(15, len(result.data_gaps) * 3)
            
            # Bonus per indicatori
            if result.has_indicators and result.indicators_coverage_pct > 80:
                score += 5
            
            result.quality_score = max(0, score)
            
            # 7. DETERMINAZIONE STATUS
            if result.quality_score >= 95:
                result.status = "EXCELLENT"
            elif result.quality_score >= 85:
                result.status = "GOOD"
            elif result.quality_score >= 70:
                result.status = "FAIR"
            else:
                result.status = "POOR"
                
    except Exception as e:
        logging.error(f"Error verifying data integrity for {symbol} ({timeframe}): {e}")
        result.status = "ERROR"
        result.integrity_issues.append(f"Verification error: {str(e)}")
    
    return result

def get_all_symbols_integrity_status(timeframes: List[str]) -> Dict[str, Dict[str, DataIntegrityResult]]:
    """
    Ottieni lo status di integrità per tutti i simboli e timeframes.
    
    Args:
        timeframes: Lista dei timeframes da verificare
        
    Returns:
        Dizionario con risultati organizzati per simbolo e timeframe
    """
    results = {}
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            # Ottieni tutti i simboli unici da tutti i timeframes
            all_symbols = set()
            
            for timeframe in timeframes:
                table_name = f"market_data_{timeframe}"
                
                # Verifica se la tabella esiste
                cursor = conn.cursor()
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                if cursor.fetchone():
                    query = f"SELECT DISTINCT symbol FROM {table_name}"
                    symbols_df = pd.read_sql_query(query, conn)
                    all_symbols.update(symbols_df['symbol'].tolist())
            
            # Verifica integrità per ogni simbolo e timeframe
            for symbol in all_symbols:
                results[symbol] = {}
                for timeframe in timeframes:
                    results[symbol][timeframe] = verify_data_integrity_post_download(symbol, timeframe)
                    
    except Exception as e:
        logging.error(f"Error getting integrity status: {e}")
    
    return results

def log_integrity_summary(integrity_results: Dict[str, Dict[str, DataIntegrityResult]]):
    """
    Log un riepilogo dello stato di integrità di tutti i dati.
    
    Args:
        integrity_results: Risultati dell'integrità organizzati per simbolo/timeframe
    """
    if not integrity_results:
        return
    
    total_symbols = len(integrity_results)
    total_checks = sum(len(timeframes) for timeframes in integrity_results.values())
    
    status_counts = {"EXCELLENT": 0, "GOOD": 0, "FAIR": 0, "POOR": 0, "ERROR": 0, "NO_DATA": 0}
    total_records = 0
    total_issues = 0
    symbols_with_indicators = 0
    
    for symbol, timeframes in integrity_results.items():
        for timeframe, result in timeframes.items():
            status_counts[result.status] += 1
            total_records += result.total_records
            total_issues += len(result.integrity_issues)
            if result.has_indicators:
                symbols_with_indicators += 1
    
    # Log riepilogo
    print(f"\n{Fore.CYAN}{'='*60}")
    print(f"{Fore.WHITE}  📊 RIEPILOGO INTEGRITÀ DATI")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"  • Simboli analizzati: {Fore.YELLOW}{total_symbols}{Style.RESET_ALL}")
    print(f"  • Verifiche totali: {Fore.YELLOW}{total_checks}{Style.RESET_ALL}")
    print(f"  • Record totali: {Fore.YELLOW}{total_records:,}{Style.RESET_ALL}")
    print(f"  • Problemi rilevati: {Fore.RED if total_issues > 0 else Fore.GREEN}{total_issues}{Style.RESET_ALL}")
    print(f"  • Con indicatori: {Fore.GREEN}{symbols_with_indicators}/{total_checks}{Style.RESET_ALL}")
    
    print(f"\n{Fore.WHITE}  📈 STATUS QUALITÀ DATI:")
    print(f"  • {Fore.GREEN}EXCELLENT{Style.RESET_ALL} (95%+): {status_counts['EXCELLENT']}")
    print(f"  • {Fore.LIGHTGREEN_EX}GOOD{Style.RESET_ALL} (85-94%): {status_counts['GOOD']}")
    print(f"  • {Fore.YELLOW}FAIR{Style.RESET_ALL} (70-84%): {status_counts['FAIR']}")
    print(f"  • {Fore.RED}POOR{Style.RESET_ALL} (<70%): {status_counts['POOR']}")
    print(f"  • {Fore.MAGENTA}ERROR/NO_DATA{Style.RESET_ALL}: {status_counts['ERROR'] + status_counts['NO_DATA']}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")

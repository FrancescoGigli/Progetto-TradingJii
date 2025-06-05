#!/usr/bin/env python3
"""
Data validation module for TradingJii

Handles data quality checks, gap detection, and auto-repair functionality.
"""

import sqlite3
import logging
import asyncio
import csv
import os
from datetime import datetime, timedelta
from colorama import Fore, Style, Back
from modules.utils.config import DB_FILE, TIMEFRAME_CONFIG
from modules.core.exchange import create_exchange
from modules.core.data_fetcher import fetch_ohlcv_data

# Optional imports for chart generation
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
    import seaborn as sns
    CHARTS_AVAILABLE = True
except ImportError:
    CHARTS_AVAILABLE = False


class DataQualityReport:
    """
    Data quality assessment report for a specific symbol and timeframe.
    """
    def __init__(self, symbol, timeframe):
        self.symbol = symbol
        self.timeframe = timeframe
        self.score = 100
        self.issues = []
        self.gaps = []
        self.invalid_records = 0
        self.can_repair = False
        self.completeness_pct = 0
        self.freshness_ok = True
        self.total_records = 0
        self.expected_records = 0
    
    def has_issues(self):
        """Return True if any issues were found."""
        return len(self.issues) > 0 or len(self.gaps) > 0 or self.invalid_records > 0


async def validate_and_repair_data(symbol, timeframe):
    """
    Perform comprehensive data validation with auto-repair capabilities.
    Called automatically after each download.
    
    Args:
        symbol: The cryptocurrency symbol
        timeframe: The timeframe to validate
        
    Returns:
        DataQualityReport if issues found, None if data is clean
    """
    report = DataQualityReport(symbol, timeframe)
    
    try:
        # 1. Check if table exists and has data
        if not await check_table_exists(symbol, timeframe, report):
            return report
        
        # 2. Check temporal continuity and gaps
        await check_temporal_gaps(symbol, timeframe, report)
        
        # 3. Validate OHLCV data integrity
        await validate_ohlcv_data(symbol, timeframe, report)
        
        # 4. Calculate completeness
        await calculate_completeness(symbol, timeframe, report)
        
        # 5. Auto-repair if issues found and repairable
        if report.has_issues() and report.can_repair:
            await auto_repair_issues(symbol, timeframe, report)
        
        return report if report.has_issues() else None
        
    except Exception as e:
        logging.error(f"Error validating data for {symbol} ({timeframe}): {e}")
        report.issues.append(f"Validation error: {str(e)}")
        report.score = 0
        return report


async def check_table_exists(symbol, timeframe, report):
    """
    Check if the table exists and has data for the symbol.
    """
    table_name = f"market_data_{timeframe}"
    
    try:
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
            if not cursor.fetchone():
                report.issues.append("Table does not exist")
                report.score = 0
                return False
            
            # Check if symbol has any data
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE symbol = ?", (symbol,))
            count = cursor.fetchone()[0]
            
            if count == 0:
                report.issues.append("No data found for symbol")
                report.score = 0
                return False
                
            report.total_records = count
            return True
            
    except Exception as e:
        report.issues.append(f"Table check error: {str(e)}")
        report.score = 0
        return False


async def check_temporal_gaps(symbol, timeframe, report):
    """
    Find temporal gaps in the data for a specific symbol and timeframe.
    """
    try:
        expected_interval_ms = TIMEFRAME_CONFIG[timeframe]['ms']
        table_name = f"market_data_{timeframe}"
        
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            # Get all timestamps ordered chronologically
            cursor.execute(f"""
                SELECT timestamp FROM {table_name} 
                WHERE symbol = ? 
                ORDER BY timestamp
            """, (symbol,))
            
            timestamps = [datetime.strptime(row[0], '%Y-%m-%dT%H:%M:%S') 
                         for row in cursor.fetchall()]
            
            if len(timestamps) < 2:
                return  # Need at least 2 records to check gaps
            
            # Find gaps between consecutive timestamps
            gaps = []
            expected_diff = timedelta(milliseconds=expected_interval_ms)
            
            for i in range(1, len(timestamps)):
                diff = timestamps[i] - timestamps[i-1]
                
                # Allow 10% tolerance for timing variations
                if diff > expected_diff * 1.1:
                    gap_size = int(diff.total_seconds() / (expected_interval_ms / 1000)) - 1
                    gaps.append({
                        'start': timestamps[i-1],
                        'end': timestamps[i],
                        'size': gap_size,
                        'duration': diff
                    })
            
            if gaps:
                report.gaps = gaps
                report.issues.append(f"Found {len(gaps)} temporal gaps")
                # Penalize score based on number and size of gaps
                gap_penalty = min(len(gaps) * 5, 30)  # -5 per gap, max -30
                total_gap_size = sum(gap['size'] for gap in gaps)
                size_penalty = min(total_gap_size * 0.5, 20)  # -0.5 per missing record, max -20
                report.score -= (gap_penalty + size_penalty)
                
                # Can repair if not too many gaps
                report.can_repair = len(gaps) <= 10 and total_gap_size <= 50
                
    except Exception as e:
        report.issues.append(f"Gap check error: {str(e)}")
        report.score -= 10


async def validate_ohlcv_data(symbol, timeframe, report):
    """
    Validate integrity of OHLCV data.
    """
    try:
        table_name = f"market_data_{timeframe}"
        
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            # Check for basic OHLCV validity
            cursor.execute(f"""
                SELECT COUNT(*) FROM {table_name} 
                WHERE symbol = ? AND (
                    high < (CASE WHEN open > close THEN open ELSE close END) OR 
                    low > (CASE WHEN open < close THEN open ELSE close END) OR
                    volume < 0 OR
                    open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
                )
            """, (symbol,))
            
            invalid_count = cursor.fetchone()[0]
            
            if invalid_count > 0:
                report.invalid_records = invalid_count
                report.issues.append(f"Found {invalid_count} invalid OHLCV records")
                # Penalize based on percentage of invalid records
                invalid_pct = (invalid_count / report.total_records) * 100
                penalty = min(invalid_pct * 2, 25)  # Up to 25 points penalty
                report.score -= penalty
                
                # Can repair if less than 5% invalid records
                report.can_repair = report.can_repair and (invalid_pct < 5)
            
            # Check for extreme price movements (potential spikes)
            cursor.execute(f"""
                SELECT COUNT(*) FROM {table_name} 
                WHERE symbol = ? AND (
                    (high / (CASE WHEN open > close THEN open ELSE close END)) > 1.5 OR
                    ((CASE WHEN open < close THEN open ELSE close END) / low) > 1.5
                )
            """, (symbol,))
            
            spike_count = cursor.fetchone()[0]
            
            if spike_count > 0:
                report.issues.append(f"Found {spike_count} potential price spikes")
                # Minor penalty for spikes (they might be legitimate)
                report.score -= min(spike_count * 1, 10)
                
    except Exception as e:
        report.issues.append(f"OHLCV validation error: {str(e)}")
        report.score -= 10


async def calculate_completeness(symbol, timeframe, report):
    """
    Calculate data completeness percentage.
    """
    try:
        table_name = f"market_data_{timeframe}"
        
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            # Get first and last timestamps
            cursor.execute(f"""
                SELECT MIN(timestamp), MAX(timestamp) 
                FROM {table_name}
                WHERE symbol = ?
            """, (symbol,))
            
            result = cursor.fetchone()
            
            if result and result[0] and result[1]:
                first_time = datetime.strptime(result[0], '%Y-%m-%dT%H:%M:%S')
                last_time = datetime.strptime(result[1], '%Y-%m-%dT%H:%M:%S')
                
                # Calculate expected number of records
                time_diff = last_time - first_time
                interval_ms = TIMEFRAME_CONFIG[timeframe]['ms']
                expected_records = int(time_diff.total_seconds() * 1000 / interval_ms) + 1
                
                report.expected_records = expected_records
                report.completeness_pct = (report.total_records / expected_records) * 100
                
                # Penalize if completeness is low
                if report.completeness_pct < 95:
                    completeness_penalty = (95 - report.completeness_pct) * 0.5
                    report.score -= min(completeness_penalty, 15)
                    report.issues.append(f"Low completeness: {report.completeness_pct:.1f}%")
                    
    except Exception as e:
        report.issues.append(f"Completeness calculation error: {str(e)}")


async def auto_repair_issues(symbol, timeframe, report):
    """
    Attempt to automatically repair data issues.
    """
    try:
        logging.info(f"🔧 Auto-repairing data issues for {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
        
        repairs_made = 0
        
        # Repair temporal gaps by re-downloading missing data
        if report.gaps and len(report.gaps) <= 5:  # Only repair small number of gaps
            repairs_made += await repair_temporal_gaps(symbol, timeframe, report.gaps)
        
        # Remove invalid OHLCV records
        if report.invalid_records > 0 and report.invalid_records <= 10:
            repairs_made += await repair_invalid_ohlcv(symbol, timeframe)
        
        if repairs_made > 0:
            logging.info(f"✅ Auto-repaired {repairs_made} issues for {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe})")
            # Re-validate after repairs
            updated_report = await validate_and_repair_data(symbol, timeframe)
            if updated_report and updated_report.score > report.score:
                report.score = updated_report.score
                report.issues = updated_report.issues
        
    except Exception as e:
        logging.error(f"Auto-repair error for {symbol} ({timeframe}): {e}")


async def repair_temporal_gaps(symbol, timeframe, gaps):
    """
    Repair temporal gaps by re-downloading missing data.
    """
    repairs_made = 0
    
    try:
        # Only repair small gaps to avoid overwhelming the API
        small_gaps = [gap for gap in gaps if gap['size'] <= 10]
        
        if not small_gaps:
            return 0
        
        exchange = await create_exchange()
        
        for gap in small_gaps[:3]:  # Limit to 3 gaps at a time
            try:
                # Calculate how many days of data we need for this gap
                gap_days = max(1, int(gap['duration'].total_seconds() / (24 * 3600)) + 1)
                
                # Re-download data for the gap period
                result = await fetch_ohlcv_data(exchange, symbol, timeframe, gap_days)
                
                if result and result[0]:  # Success
                    repairs_made += 1
                    logging.info(f"  ✓ Repaired gap: {gap['start'].strftime('%m-%d %H:%M')} → "
                               f"{gap['end'].strftime('%H:%M')}")
                    
            except Exception as e:
                logging.warning(f"  ✗ Failed to repair gap for {symbol}: {e}")
        
        await exchange.close()
        
    except Exception as e:
        logging.error(f"Gap repair error: {e}")
    
    return repairs_made


async def repair_invalid_ohlcv(symbol, timeframe):
    """
    Remove or fix invalid OHLCV records.
    """
    repairs_made = 0
    
    try:
        table_name = f"market_data_{timeframe}"
        
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            
            # Remove clearly invalid records
            cursor.execute(f"""
                DELETE FROM {table_name} 
                WHERE symbol = ? AND (
                    volume < 0 OR
                    open <= 0 OR high <= 0 OR low <= 0 OR close <= 0 OR
                    high < (CASE WHEN open > close THEN open ELSE close END) OR 
                    low > (CASE WHEN open < close THEN open ELSE close END)
                )
            """, (symbol,))
            
            repairs_made = cursor.rowcount
            conn.commit()
            
            if repairs_made > 0:
                logging.info(f"  ✓ Removed {repairs_made} invalid records for {symbol}")
                
    except Exception as e:
        logging.error(f"OHLCV repair error: {e}")
    
    return repairs_made


def log_validation_results(symbol, timeframe, report):
    """
    Log colorized validation results.
    """
    try:
        # Determine status based on score
        if report.score >= 95:
            status_color = Fore.GREEN
            status_icon = "✅"
        elif report.score >= 85:
            status_color = Fore.YELLOW  
            status_icon = "⚠️"
        else:
            status_color = Fore.RED
            status_icon = "❌"
        
        # Main status line
        logging.info(f"{status_icon} Data Quality {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe}): "
                    f"{status_color}{report.score:.0f}/100{Style.RESET_ALL}")
        
        # Completeness info
        if report.expected_records > 0:
            logging.info(f"  └─ 📊 Completeness: {report.completeness_pct:.1f}% "
                        f"({report.total_records:,}/{report.expected_records:,} records)")
        
        # Gap information
        if report.gaps:
            gap_count = len(report.gaps)
            total_missing = sum(gap['size'] for gap in report.gaps)
            logging.info(f"  └─ 🕳️  {gap_count} temporal gaps ({total_missing} missing records)")
            
            # Show details for first few gaps
            for gap in report.gaps[:3]:
                duration_str = f"{gap['duration'].total_seconds()/60:.0f}min"
                logging.info(f"     • {gap['start'].strftime('%m-%d %H:%M')} → "
                            f"{gap['end'].strftime('%H:%M')} ({gap['size']} records, {duration_str})")
            
            if len(report.gaps) > 3:
                logging.info(f"     • ... and {len(report.gaps) - 3} more gaps")
        
        # Invalid records info
        if report.invalid_records > 0:
            invalid_pct = (report.invalid_records / report.total_records) * 100
            logging.info(f"  └─ 🔍 {report.invalid_records} invalid records ({invalid_pct:.1f}%)")
        
        # Other issues
        other_issues = [issue for issue in report.issues 
                       if not any(keyword in issue.lower() 
                                for keyword in ['gap', 'invalid', 'completeness'])]
        if other_issues:
            logging.info(f"  └─ ⚠️  {len(other_issues)} other issues")
            for issue in other_issues[:2]:  # Show first 2 other issues
                logging.info(f"     • {issue}")
                
    except Exception as e:
        logging.error(f"Error logging validation results: {e}")


def log_validation_summary(validation_summary):
    """
    Log the overall validation summary.
    """
    try:
        if validation_summary["symbols_validated"] > 0:
            logging.info(f"\n🔍 {Back.CYAN}{Fore.WHITE} DATA VALIDATION SUMMARY {Style.RESET_ALL}")
            logging.info(f"  • Symbols validated: {Fore.CYAN}{validation_summary['symbols_validated']}{Style.RESET_ALL}")
            logging.info(f"  • Issues found: {Fore.YELLOW}{validation_summary['issues_found']}{Style.RESET_ALL}")
            logging.info(f"  • Auto-repaired: {Fore.GREEN}{validation_summary['auto_repaired']}{Style.RESET_ALL}")
            
            if validation_summary["high_quality"] > 0:
                logging.info(f"  • High quality (≥95): {Fore.GREEN}{validation_summary['high_quality']}{Style.RESET_ALL}")
            if validation_summary["medium_quality"] > 0:
                logging.info(f"  • Medium quality (85-94): {Fore.YELLOW}{validation_summary['medium_quality']}{Style.RESET_ALL}")
            if validation_summary["low_quality"] > 0:
                logging.info(f"  • Low quality (<85): {Fore.RED}{validation_summary['low_quality']}{Style.RESET_ALL}")
                
    except Exception as e:
        logging.error(f"Error logging validation summary: {e}")


def export_validation_report_csv(validation_reports, validation_summary, timestamp=None):
    """
    Export validation results to CSV file with timestamp.
    
    Args:
        validation_reports: List of DataQualityReport objects
        validation_summary: Dictionary with summary statistics
        timestamp: Optional timestamp string, defaults to current time
    """
    try:
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create reports directory if it doesn't exist
        reports_dir = "validation_reports"
        os.makedirs(reports_dir, exist_ok=True)
        
        # Export detailed reports
        detailed_file = f"{reports_dir}/validation_report_detailed_{timestamp}.csv"
        
        with open(detailed_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'symbol', 'timeframe', 'quality_score', 'total_records', 'expected_records',
                'completeness_pct', 'gaps_count', 'invalid_records', 'issues_count',
                'can_repair', 'gaps_details', 'issues_list'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for report in validation_reports:
                # Prepare gaps details
                gaps_details = "; ".join([
                    f"{gap['start'].strftime('%m-%d %H:%M')}-{gap['end'].strftime('%H:%M')}({gap['size']})" 
                    for gap in report.gaps[:5]  # Limit to first 5 gaps
                ])
                
                writer.writerow({
                    'symbol': report.symbol,
                    'timeframe': report.timeframe,
                    'quality_score': report.score,
                    'total_records': report.total_records,
                    'expected_records': report.expected_records,
                    'completeness_pct': round(report.completeness_pct, 2),
                    'gaps_count': len(report.gaps),
                    'invalid_records': report.invalid_records,
                    'issues_count': len(report.issues),
                    'can_repair': report.can_repair,
                    'gaps_details': gaps_details,
                    'issues_list': "; ".join(report.issues[:3])  # Limit to first 3 issues
                })
        
        # Export summary report
        summary_file = f"{reports_dir}/validation_summary_{timestamp}.csv"
        
        with open(summary_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Metric', 'Value'])
            writer.writerow(['Timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            writer.writerow(['Symbols Validated', validation_summary['symbols_validated']])
            writer.writerow(['Issues Found', validation_summary['issues_found']])
            writer.writerow(['Auto Repaired', validation_summary['auto_repaired']])
            writer.writerow(['High Quality (≥95)', validation_summary['high_quality']])
            writer.writerow(['Medium Quality (85-94)', validation_summary['medium_quality']])
            writer.writerow(['Low Quality (<85)', validation_summary['low_quality']])
        
        logging.info(f"📄 Validation reports exported:")
        logging.info(f"  • Detailed: {Fore.BLUE}{os.path.abspath(detailed_file)}{Style.RESET_ALL}")
        logging.info(f"  • Summary: {Fore.BLUE}{os.path.abspath(summary_file)}{Style.RESET_ALL}")
        
    except Exception as e:
        logging.error(f"Error exporting validation report: {e}")


def generate_validation_charts(validation_reports, timestamp=None):
    """
    Generate validation charts and heatmaps.
    
    Args:
        validation_reports: List of DataQualityReport objects
        timestamp: Optional timestamp string, defaults to current time
    """
    if not CHARTS_AVAILABLE:
        logging.warning(f"{Fore.YELLOW}Charts generation skipped: matplotlib not available{Style.RESET_ALL}")
        logging.info(f"Install with: {Fore.CYAN}pip install matplotlib seaborn{Style.RESET_ALL}")
        return
    
    try:
        if timestamp is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create charts directory if it doesn't exist
        charts_dir = "validation_charts"
        os.makedirs(charts_dir, exist_ok=True)
        
        # Prepare data for plotting
        symbols = [r.symbol for r in validation_reports]
        timeframes = [r.timeframe for r in validation_reports]
        scores = [r.score for r in validation_reports]
        completeness = [r.completeness_pct for r in validation_reports]
        gaps_count = [len(r.gaps) for r in validation_reports]
        
        # 1. Quality Score Heatmap
        plt.figure(figsize=(12, 8))
        
        # Create matrix for heatmap
        unique_symbols = sorted(list(set(symbols)))
        unique_timeframes = sorted(list(set(timeframes)))
        
        score_matrix = np.full((len(unique_symbols), len(unique_timeframes)), np.nan)
        
        for report in validation_reports:
            sym_idx = unique_symbols.index(report.symbol)
            tf_idx = unique_timeframes.index(report.timeframe)
            score_matrix[sym_idx, tf_idx] = report.score
        
        # Create heatmap
        plt.subplot(2, 2, 1)
        sns.heatmap(score_matrix, 
                    xticklabels=unique_timeframes, 
                    yticklabels=unique_symbols,
                    annot=True, 
                    fmt='.0f',
                    cmap='RdYlGn',
                    vmin=0, vmax=100,
                    cbar_kws={'label': 'Quality Score'})
        plt.title('Data Quality Score Heatmap')
        plt.xlabel('Timeframe')
        plt.ylabel('Symbol')
        
        # 2. Completeness Distribution
        plt.subplot(2, 2, 2)
        plt.hist(completeness, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(95, color='red', linestyle='--', label='Target (95%)')
        plt.xlabel('Completeness (%)')
        plt.ylabel('Frequency')
        plt.title('Data Completeness Distribution')
        plt.legend()
        
        # 3. Quality Score vs Gaps
        plt.subplot(2, 2, 3)
        plt.scatter(gaps_count, scores, alpha=0.6, c=completeness, cmap='viridis')
        plt.colorbar(label='Completeness %')
        plt.xlabel('Number of Gaps')
        plt.ylabel('Quality Score')
        plt.title('Quality Score vs Temporal Gaps')
        
        # 4. Quality Categories Pie Chart
        plt.subplot(2, 2, 4)
        high_quality = sum(1 for s in scores if s >= 95)
        medium_quality = sum(1 for s in scores if 85 <= s < 95)
        low_quality = sum(1 for s in scores if s < 85)
        
        sizes = [high_quality, medium_quality, low_quality]
        labels = [f'High (≥95)\n{high_quality}', f'Medium (85-94)\n{medium_quality}', f'Low (<85)\n{low_quality}']
        colors = ['#2ecc71', '#f39c12', '#e74c3c']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        plt.title('Quality Categories Distribution')
        
        plt.tight_layout()
        
        # Save the chart
        chart_file = f"{charts_dir}/validation_charts_{timestamp}.png"
        plt.savefig(chart_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Generate individual timeframe charts if there are multiple timeframes
        if len(unique_timeframes) > 1:
            plt.figure(figsize=(15, 5))
            
            for i, tf in enumerate(unique_timeframes):
                plt.subplot(1, len(unique_timeframes), i+1)
                
                tf_reports = [r for r in validation_reports if r.timeframe == tf]
                tf_symbols = [r.symbol for r in tf_reports]
                tf_scores = [r.score for r in tf_reports]
                
                bars = plt.bar(range(len(tf_symbols)), tf_scores, 
                              color=['#2ecc71' if s >= 95 else '#f39c12' if s >= 85 else '#e74c3c' for s in tf_scores])
                
                plt.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='High Quality')
                plt.axhline(y=85, color='orange', linestyle='--', alpha=0.7, label='Medium Quality')
                
                plt.xlabel('Symbols')
                plt.ylabel('Quality Score')
                plt.title(f'Quality Scores - {tf}')
                plt.xticks(range(len(tf_symbols)), tf_symbols, rotation=45, ha='right')
                plt.ylim(0, 105)
                
                if i == 0:  # Add legend only to first subplot
                    plt.legend()
            
            plt.tight_layout()
            
            timeframe_chart_file = f"{charts_dir}/validation_by_timeframe_{timestamp}.png"
            plt.savefig(timeframe_chart_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logging.info(f"📊 Validation charts generated:")
            logging.info(f"  • Overview: {Fore.BLUE}{os.path.abspath(chart_file)}{Style.RESET_ALL}")
            logging.info(f"  • By Timeframe: {Fore.BLUE}{os.path.abspath(timeframe_chart_file)}{Style.RESET_ALL}")
        else:
            logging.info(f"📊 Validation chart generated: {Fore.BLUE}{os.path.abspath(chart_file)}{Style.RESET_ALL}")
        
    except Exception as e:
        logging.error(f"Error generating validation charts: {e}")
        import traceback
        logging.error(traceback.format_exc())

# TradingJii - Cryptocurrency Trading Analytics Platform

A comprehensive, full-stack cryptocurrency analytics platform that provides real-time data visualization, volatility analysis, and pattern recognition to assist traders in making informed decisions.

<div align="center">
<img src="https://via.placeholder.com/800x400?text=TradingJii+Dashboard" alt="TradingJii Dashboard" width="800"/>
<p><i>TradingJii Dashboard showing cryptocurrency price analysis</i></p>
</div>

## Table of Contents
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
  - [Directory Structure](#directory-structure)
  - [Data Flow Diagram](#data-flow-diagram)
  - [Component Interaction](#component-interaction)
- [Backend Components](#backend-components)
  - [Core Module](#core-module)
  - [Data Module](#data-module)
  - [Utils Module](#utils-module)
- [Frontend Components](#frontend-components)
  - [HTML Structure](#html-structure)
  - [CSS Styling System](#css-styling-system)
  - [JavaScript Application Logic](#javascript-application-logic)
  - [Chart Rendering](#chart-rendering)
- [Data Processing Pipeline](#data-processing-pipeline)
  - [Data Collection](#data-collection)
  - [Data Processing](#data-processing)
  - [Pattern Analysis](#pattern-analysis)
  - [API Service](#api-service)
  - [Frontend Visualization](#frontend-visualization)
- [Algorithms and Implementation Details](#algorithms-and-implementation-details)
  - [Volatility Calculation](#volatility-calculation)
  - [Heikin-Ashi Transformation](#heikin-ashi-transformation)
  - [Binary Pattern Recognition](#binary-pattern-recognition)
  - [Data Freshness Checking](#data-freshness-checking)
  - [Parallel Download Architecture](#parallel-download-architecture)
- [API Reference](#api-reference)
  - [Endpoints](#endpoints)
  - [Request Parameters](#request-parameters)
  - [Response Formats](#response-formats)
  - [Error Handling](#error-handling)
- [Database Schema](#database-schema)
  - [Tables](#tables)
  - [Indices](#indices)
  - [Relationships](#relationships)
  - [Query Optimization](#query-optimization)
- [Installation and Setup](#installation-and-setup)
  - [System Requirements](#system-requirements)
  - [Environment Setup](#environment-setup)
  - [Installation Steps](#installation-steps)
  - [Docker Deployment](#docker-deployment)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Configuration Files](#configuration-files)
  - [Command-Line Options](#command-line-options)
  - [Advanced Configuration](#advanced-configuration)
- [Usage Guide](#usage-guide)
  - [Basic Usage](#basic-usage)
  - [Chart Interpretation](#chart-interpretation)
  - [Pattern Analysis](#pattern-analysis-1)
  - [Tips and Best Practices](#tips-and-best-practices)
- [Development and Extension](#development-and-extension)
  - [Development Environment](#development-environment)
  - [Adding New Features](#adding-new-features)
  - [Code Style Guidelines](#code-style-guidelines)
  - [Testing](#testing)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)
  - [Diagnostic Tools](#diagnostic-tools)
  - [Logs and Debugging](#logs-and-debugging)
- [Performance Optimization](#performance-optimization)
  - [Database Optimization](#database-optimization)
  - [API Response Time](#api-response-time)
  - [Frontend Optimization](#frontend-optimization)
  - [Memory Usage](#memory-usage)
- [Security Considerations](#security-considerations)
  - [API Key Management](#api-key-management)
  - [Input Validation](#input-validation)
  - [Rate Limiting](#rate-limiting)
- [Dependencies](#dependencies)
  - [Backend (Python)](#backend-python)
  - [Frontend (JavaScript)](#frontend-javascript)
  - [Development Tools](#development-tools)
- [Future Development Roadmap](#future-development-roadmap)
  - [Short-term Goals](#short-term-goals)
  - [Mid-term Plans](#mid-term-plans)
  - [Long-term Vision](#long-term-vision)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

TradingJii is a sophisticated application designed to help cryptocurrency traders analyze market data effectively through:

- Real-time cryptocurrency price data retrieval from the Bybit exchange
- Advanced candlestick visualization with Heikin-Ashi transformation
- Volume visualization with color-coded bars
- Volatility calculation, cleaning, and visualization
- Binary pattern recognition for time series analysis
- Responsive web interface with dark/light mode and search capabilities
- Multi-timeframe support (5-minute and 15-minute intervals)
- Interactive charts with tooltips and annotations

The platform bridges the gap between raw market data and actionable insights by providing visual representations of price movements, volatility patterns, and recurring market behaviors. Unlike traditional charting platforms, TradingJii focuses specifically on volatility-based pattern recognition to identify potential entry and exit points for traders.

### Key Differentiators

- **Volatility Focus**: While most platforms concentrate on price, TradingJii analyzes the rate of price change
- **Pattern Recognition**: Automated identification of recurring volatility patterns
- **Dual Visualization**: Seamless switching between price and volatility charts
- **Optimized UX**: Dark/light theming with persistent preferences and responsive design
- **Efficient Data Pipeline**: Smart updating with data freshness checks to minimize API calls

### Target Users

- Cryptocurrency day traders
- Algorithmic trading strategy developers
- Market data analysts
- Technical analysis enthusiasts

## System Architecture

TradingJii follows a clear architectural separation between data collection, processing, and presentation. The system is built with modularity in mind, allowing for independent development and testing of different components.

### Directory Structure

```
TradingJii/
├── app.py                  # Flask web server and API endpoints
├── real_time.py            # Continuous data collection system
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not in repo)
├── crypto_data.db          # SQLite database (generated)
├── frontend/               # Web interface assets
│   ├── index.html          # Main application HTML
│   ├── styles.css          # CSS styling with theming support
│   ├── script.js           # Main frontend controller
│   ├── chart-handler.js    # Chart visualization engine
│   └── test.html           # Testing interface
├── modules/                # Backend modules
│   ├── core/               # Core functionality
│   │   ├── __init__.py
│   │   ├── data_fetcher.py         # OHLCV data retrieval
│   │   ├── download_orchestrator.py # Parallel/sequential download management
│   │   └── exchange.py             # Exchange connection and market queries
│   ├── data/               # Data processing
│   │   ├── __init__.py
│   │   ├── db_manager.py           # Database operations
│   │   ├── series_segmenter.py     # Pattern identification
│   │   └── volatility_processor.py # Volatility calculations
│   └── utils/              # Utilities
│       ├── __init__.py
│       ├── command_args.py         # CLI argument parsing
│       ├── config.py               # Configuration parameters
│       └── logging_setup.py        # Colored logging configuration
├── docs/                   # Documentation (optional)
└── tests/                  # Unit and integration tests (optional)
```

### Data Flow Diagram

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│                 │     │                  │     │                     │
│  Cryptocurrency │     │  TradingJii      │     │  User Interface     │
│  Exchange (Bybit)│◄───►│  Backend         │◄───►│  (Browser)          │
│                 │     │                  │     │                     │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
        │                       ▲                           │
        │                       │                           │
        ▼                       ▼                           ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│                 │     │                  │     │                     │
│  Market Data    │     │  SQLite          │     │  Chart.js           │
│  OHLCV, Volume  │     │  Database        │     │  Visualizations     │
│                 │     │                  │     │                     │
└─────────────────┘     └──────────────────┘     └─────────────────────┘
```

### Component Interaction

1. **Data Collection Layer**: Interacts with cryptocurrency exchanges to retrieve market data
2. **Data Processing Layer**: Transforms raw market data into volatility metrics and patterns
3. **Storage Layer**: Persists both raw and processed data in SQLite database
4. **API Layer**: Exposes internal data through RESTful endpoints
5. **Presentation Layer**: Renders data as interactive visualizations

## Backend Components

### Core Module

#### `data_fetcher.py`
- **Purpose**: Retrieves OHLCV (Open, High, Low, Close, Volume) cryptocurrency data from exchange APIs
- **Key Functions**:
  - `estimated_iterations()`: Calculates progress bar size for download operations
  - `fetch_ohlcv_data()`: Retrieves OHLCV data for a specific symbol/timeframe with progress tracking
- **Features**:
  - Smart data freshness checking to avoid redundant downloads
  - Progress bar visualization with colorful terminal output
  - Pagination handling for large datasets with chunk-based retrieval
  - Automatic retry on temporary failures

**Code Example**: Fetching OHLCV data with progress tracking

```python
async def fetch_ohlcv_data(exchange, symbol, timeframe, data_limit_days):
    """
    Fetch OHLCV data for a specific symbol and timeframe.
    
    Args:
        exchange: The exchange object
        symbol: The cryptocurrency symbol
        timeframe: The timeframe to fetch data for
        data_limit_days: Maximum days of historical data to fetch
        
    Returns:
        Tuple of (success, count) or None if data is already fresh
    """
    # Check if we already have fresh data
    is_fresh, last_date = check_data_freshness(symbol, timeframe)
    if is_fresh:
        return None  # Data is already fresh, no need to download
        
    # Calculate time range for data fetching
    now = datetime.now()
    start_time = now - timedelta(days=data_limit_days)
    
    if last_date:
        # Start from one day before last date to ensure overlap
        fetch_start_time = max(start_time, last_date - timedelta(days=1))
    else:
        fetch_start_time = start_time

    # Convert to millisecond timestamps
    since = int(fetch_start_time.timestamp() * 1000)
    now_ms = int(now.timestamp() * 1000)
    ohlcv_data = []

    # Fetch data with progress bar
    with logging_redirect_tqdm():
        with tqdm(total=estimated_iterations(since, now_ms, timeframe), 
                 desc=f"{Fore.BLUE}Loading {symbol} ({timeframe}){Style.RESET_ALL}",
                 bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Style.RESET_ALL)) as pbar:
            current_since = since
            while current_since < now_ms:
                try:
                    # Request a chunk of data with pagination
                    data_chunk = await exchange.fetch_ohlcv(
                        symbol, timeframe, since=current_since, limit=1000
                    )
                    if not data_chunk:
                        break  # No more data available
                        
                    ohlcv_data.extend(data_chunk)
                    
                    if data_chunk:
                        # Update pagination marker for next iteration
                        current_since = data_chunk[-1][0] + 1
                        
                    pbar.update(1)
                except Exception as e:
                    logging.error(f"Error fetching OHLCV data for {symbol} ({timeframe}): {e}")
                    break

    # Save data to database
    if ohlcv_data:
        return save_ohlcv_data(symbol, timeframe, ohlcv_data)
    return False, 0
```

**Implementation Details**:
- Uses CCXT library's `fetch_ohlcv` method for standardized exchange access
- Implements pagination with the `since` parameter to handle large datasets
- Updates progress bar based on estimated iterations to give visual feedback
- Returns None when data is already fresh to signal that no download was needed
- Returns success status and count of records saved to allow for tracking statistics

#### `download_orchestrator.py`
- **Purpose**: Orchestrates the download process for multiple symbols and timeframes
- **Key Functions**:
  - `process_timeframe()`: Manages data collection for a specific timeframe
  - `fetch_data_parallel()`: Implements concurrent downloads with semaphore limiting
  - `fetch_data_sequential()`: Implements sequential downloads for stability
- **Features**:
  - Batch processing to manage memory consumption
  - Configurable concurrency to optimize download speeds
  - Detailed progress tracking and reporting
  - Real-time statistics collection and visualization
  - Graceful error handling with comprehensive logging

**Implementation Details**:
- Uses asyncio for concurrent operations
- Implements semaphore pattern to limit concurrency
- Divides symbol list into batches to prevent memory exhaustion
- Collects and aggregates statistics from all operations
- Provides real-time feedback through colored terminal output

**Parallel Processing Algorithm**:
1. Split symbols into batches of configurable size
2. For each batch:
   a. Create async tasks for each symbol
   b. Apply semaphore limiting to maintain max concurrency
   c. Process results as they complete via queue
   d. Display real-time progress updates
3. Aggregate results across all batches
4. Return final statistics

#### `exchange.py`
- **Purpose**: Handles connectivity to cryptocurrency exchanges
- **Key Functions**:
  - `create_exchange()`: Initializes exchange connection with proper configuration
  - `fetch_markets()`: Retrieves available trading pairs, filtered by criteria
  - `get_top_symbols()`: Identifies highest volume cryptocurrencies
- **Features**:
  - Uses CCXT library for standardized exchange access
  - Rate limiting compliance to avoid API restrictions
  - Automatic filtering for USDT-quoted markets
  - Volume-based ranking of cryptocurrencies
  - Colorized terminal output for market statistics

**Code Example**: Finding top cryptocurrencies by volume

```python
async def get_top_symbols(exchange, symbols, top_n=100):
    """
    Get the top N cryptocurrencies by trading volume.
    
    Args:
        exchange: The exchange object
        symbols: List of all symbols to check
        top_n: Number of top symbols to return (default: 100)
        
    Returns:
        List of top symbols by volume
    """
    try:
        logging.info(f"Retrieving volume data for {len(symbols)} USDT pairs...")
        volumes = {}

        # Fetch volume data for each symbol with progress tracking
        with logging_redirect_tqdm():
            with tqdm(total=len(symbols), 
                     desc=f"{Fore.BLUE}Finding USDT pairs with highest volume{Style.RESET_ALL}", 
                     bar_format="{l_bar}%s{bar}%s{r_bar}" % (Fore.BLUE, Style.RESET_ALL)) as pbar:
                for symbol in symbols:
                    try:
                        ticker = await exchange.fetch_ticker(symbol)
                        volumes[symbol] = ticker.get('quoteVolume', 0) if ticker else 0
                    except Exception as e:
                        logging.error(f"Error retrieving volume for {symbol}: {e}")
                        volumes[symbol] = 0
                    pbar.update(1)

        # Sort by volume and take top N
        top_symbols = [s[0] for s in sorted(volumes.items(), key=lambda x: x[1], reverse=True)[:top_n]]
        logging.info(f"Found {Fore.YELLOW}{len(top_symbols)}{Style.RESET_ALL} USDT pairs with highest volume")

        # Display top cryptocurrencies in a formatted table
        print("\n" + "="*80)
        print(f"{Fore.WHITE}  TOP CRYPTOCURRENCIES BY VOLUME  {Style.RESET_ALL}")
        print("="*80)
        
        # Table header
        print(f"{'#':4} {'Symbol':20} {'Volume (USDT)':>25}")
        print("-"*60)
        
        # Show top 10 for reference
        for i, (symbol, volume) in enumerate(sorted(volumes.items(), key=lambda x: x[1], reverse=True)[:10]):
            # Alternate background colors for readability
            bg_color = "" if i % 2 == 0 else ""
            # Highlight based on position (TOP 3 in yellow, rest in white)
            symbol_color = Fore.YELLOW if i < 3 else Fore.WHITE
            volume_color = Fore.CYAN if i < 3 else Fore.WHITE
            
            # Print formatted row
            print(f"{bg_color}{i+1:3} {symbol_color}{symbol:20} {volume_color}{volume:25,.2f}{Style.RESET_ALL}")
        
        print("="*80 + "\n")
        return top_symbols
    except Exception as e:
        logging.error(f"Error retrieving highest volume pairs: {e}")
        return []
```

**Implementation Details**:
- Retrieves ticker data for all available symbols
- Extracts quote volume for standardized comparison
- Sorts cryptocurrencies by volume in descending order
- Provides formatted terminal output with colorized highlighting
- Handles exceptions for each individual ticker request to ensure robustness

### Data Module

#### `db_manager.py`
- **Purpose**: Manages database operations for cryptocurrency data
- **Key Functions**:
  - `init_data_tables()`: Creates necessary database tables for each timeframe
  - `check_data_freshness()`: Verifies if stored data is recent enough
  - `save_ohlcv_data()`: Persists downloaded market data to SQLite
- **Features**:
  - Automatic table creation with appropriate indices
  - Optimized database schema with unique constraints
  - Efficient upsert operations to prevent duplicates
  - Data range logging for verification

**Code Example**: Checking data freshness

```python
def check_data_freshness(symbol, timeframe):
    """
    Check if we already have fresh data for a symbol and timeframe.
    
    Args:
        symbol: The cryptocurrency symbol
        timeframe: The timeframe to check
        
    Returns:
        Tuple of (is_fresh, last_timestamp)
    """
    now = datetime.now()
    first_date, last_date = get_timestamp_range(symbol, timeframe)

    if last_date:
        # Calculate time difference between now and last stored data point
        time_diff = now - last_date
        
        # Compare with configured freshness threshold for this timeframe
        if time_diff < TIMEFRAME_CONFIG[timeframe]['max_age']:
            logging.info(f"Skipping {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe}): Fresh data already exists")
            logging.info(f"  • First date: {Fore.CYAN}{first_date.strftime('%Y-%m-%d %H:%M')}{Style.RESET_ALL}")
            logging.info(f"  • Last date: {Fore.CYAN}{last_date.strftime('%Y-%m-%d %H:%M')}{Style.RESET_ALL}")
            return True, last_date
            
    return False, last_date
```

**Database Schema Design Principles**:
- Separate tables per timeframe for query optimization
- Composite unique constraints to prevent duplicate data
- Indexing on frequently queried columns (symbol, timestamp)
- Timestamp storage in ISO 8601 format for compatibility
- Integer primary keys for efficient joins

#### `volatility_processor.py`
- **Purpose**: Calculates and manages cryptocurrency price volatility
- **Key Functions**:
  - `load_close_series()`: Extracts close prices from the database
  - `compute_volatility()`: Calculates percentage volatility using pct_change
  - `clean_volatility()`: Removes outliers and extreme values
  - `process_and_save_volatility()`: Orchestrates the volatility processing pipeline
- **Features**:
  - Multi-stage volatility calculation pipeline
  - Percentage-based volatility for normalized comparison
  - Data cleaning with configurable clipping range
  - Dedicated volatility tables for each timeframe

**The Volatility Calculation Algorithm**:

1. Extract close price time series from the database
2. Calculate percentage change between consecutive prices: (Pt / Pt-1 - 1) * 100
3. Remove any invalid values (NaN, Inf) that might result from division
4. Apply clipping to remove extreme outliers (default range: -100% to +100%)
5. Store the processed volatility data in dedicated tables

**Code Example**: Computing and cleaning volatility

```python
def compute_volatility(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute percentage volatility from the close prices.
    
    Args:
        df: A DataFrame with columns timestamp, close
        
    Returns:
        A DataFrame with timestamp and volatility columns
    """
    if df.empty or len(df) < 2:
        logging.warning("Insufficient data to compute volatility (need at least 2 data points)")
        return pd.DataFrame(columns=['timestamp', 'volatility'])
        
    # Create a copy to avoid modifying the original dataframe
    result_df = df.copy()
    
    # Compute percentage change and multiply by 100
    result_df['volatility'] = result_df['close'].pct_change() * 100
    
    # Drop the first row which contains NaN for volatility
    result_df = result_df.dropna()
    
    # Select only the columns we need
    return result_df[['timestamp', 'volatility']]

def clean_volatility(df: pd.DataFrame, clip_range: Tuple[int, int] = (-100, 100)) -> pd.DataFrame:
    """
    Clean the volatility series by removing invalid or extreme values.
    
    Args:
        df: A DataFrame with timestamp, volatility
        clip_range: Default (-100, 100) - values outside this range will be clipped
        
    Returns:
        Cleaned DataFrame with no NaN or inf values and clipped volatility
    """
    if df.empty:
        return df
        
    # Create a copy to avoid modifying the original dataframe
    cleaned_df = df.copy()
    
    # Remove any NaN or inf values
    cleaned_df = cleaned_df.replace([float('inf'), float('-inf')], pd.NA)
    cleaned_df = cleaned_df.dropna()
    
    # Clip values to the specified range
    cleaned_df['volatility'] = cleaned_df['volatility'].clip(clip_range[0], clip_range[1])
    
    return cleaned_df
```

**Implementation Details**:
- Uses Pandas for efficient data manipulation
- Implements the standard percentage change formula as a measure of volatility
- Removes first row which will contain NaN due to lacking a preceding value
- Handles edge cases like insufficient data points
- Implements configurable clipping range for outlier removal

#### `series_segmenter.py`
- **Purpose**: Implements pattern recognition for volatility time series
- **Key Functions**:
  - `load_volatility_series()`: Retrieves volatility data from database
  - `generate_subseries()`: Creates sliding windows for analysis
  - `categorize_series()`: Converts volatility values into binary patterns
  - `build_categorized_dataset()`: Groups similar patterns for analysis
- **Features**:
  - Sliding window approach for time series analysis
  - Binary (0/1) pattern encoding based on thresholds
  - Flexible window size configuration
  - Pattern grouping for statistical analysis

**Binary Pattern Recognition Process**:

1. Load volatility time series from database
2. Generate overlapping subseries of length window_size + 1
   - Each subseries consists of window_size values plus one target value
3. Convert each window into a binary pattern string:
   - '1' for values > threshold (default: 0.0)
   - '0' for values <= threshold
4. Group windows by their binary patterns
5. Count occurrences of each pattern
6. Visualize distribution of patterns

**Code Example**: Generating binary patterns from volatility data

```python
def categorize_series(sequence: List[float], threshold: float = 0.0) -> str:
    """
    Convert a list of volatility values into a binary pattern string.
    
    Args:
        sequence: List of volatility values
        threshold: Value for comparison (default: 0.0)
        
    Returns:
        String of '1' for values > threshold, '0' otherwise
    """
    if not sequence:
        return ""
    
    # Generate binary pattern: 1 for positive volatility, 0 for negative/zero
    pattern = ''.join('1' if v > threshold else '0' for v in sequence)
    return pattern

def build_categorized_dataset(
    symbol: str,
    timeframe: str,
    window_size: int = 7,
    threshold: float = 0.0
) -> Dict[str, List[Tuple[List[float], float]]]:
    """
    Produce a mapping of behavior categories to corresponding training samples.
    
    Args:
        symbol: The cryptocurrency symbol
        timeframe: The timeframe
        window_size: Size of the sliding window (default: 7)
        threshold: Value for categorization (default: 0.0)
        
    Returns:
        Dictionary mapping category patterns to lists of (window, target) tuples
    """
    # Load volatility data from database
    df = load_volatility_series(symbol, timeframe)
    if df.empty:
        logging.warning(f"No data available for {symbol} ({timeframe})")
        return {}
    
    # Generate subseries (sliding windows)
    subseries = generate_subseries(df, window_size)
    if not subseries:
        logging.warning(f"Could not generate subseries for {symbol} ({timeframe})")
        return {}
    
    # Categorize subseries and organize by category
    categorized_data = {}
    for window, target in subseries:
        # Convert window to binary pattern
        category = categorize_series(window, threshold)
        
        # Initialize category if it doesn't exist yet
        if category not in categorized_data:
            categorized_data[category] = []
        
        # Add this window and its target value to the category
        categorized_data[category].append((window, target))
    
    # Log summary statistics
    total_samples = len(subseries)
    total_categories = len(categorized_data)
    logging.info(f"For {Fore.YELLOW}{symbol}{Style.RESET_ALL} ({timeframe}): "
                f"Created {Fore.CYAN}{total_categories}{Style.RESET_ALL} categories "
                f"from {Fore.GREEN}{total_samples}{Style.RESET_ALL} samples")
    
    return categorized_data
```

**Implementation Details**:
- Uses a sliding window approach to capture temporal patterns
- Implements binary categorization for simplicity and interpretability
- Groups similar patterns to enable statistical analysis
- Provides configurable window size and threshold parameters
- Uses dictionary data structure for efficient category lookup

### Utils Module

#### `config.py`
- **Purpose**: Centralizes configuration parameters for the application
- **Key Variables**:
  - `EXCHANGE_CONFIG`: Parameters for exchange API connection
  - `TIMEFRAME_CONFIG`: Timeframe-specific settings including freshness criteria
  - `DEFAULT_*`: Default values for command line parameters
- **Features**:
  - Environment variable loading via dotenv
  - API key management with secure handling
  - Timeframe-specific configuration with millisecond conversion
  - Database path configuration

**Timeframe Configuration Example**:

```python
# Timeframe configuration
TIMEFRAME_CONFIG = {
    '1m': {'max_age': timedelta(minutes=5), 'ms': 60 * 1000},
    '5m': {'max_age': timedelta(minutes=15), 'ms': 5 * 60 * 1000},
    '15m': {'max_age': timedelta(hours=1), 'ms': 15 * 60 * 1000},
    '30m': {'max_age': timedelta(hours=2), 'ms': 30 * 60 * 1000},
    '1h': {'max_age': timedelta(hours=4), 'ms': 60 * 60 * 1000},
    '4h': {'max_age': timedelta(hours=12), 'ms': 4 * 60 * 60 * 1000},
    '1d': {'max_age': timedelta(days=2), 'ms': 24 * 60 * 60 * 1000}
}
```

**Implementation Details**:
- Each timeframe has:
  - `max_age`: How long data is considered "fresh" before requiring update
  - `ms`: Milliseconds per timeframe interval (for pagination calculation)
- Default values are centralized to ensure consistency across the application
- Exchange configuration includes rate limiting to comply with API restrictions
- Secret API keys are loaded from environment variables for security

#### `command_args.py`
- **Purpose**: Handles command line argument parsing
- **Key Functions**:
  - `parse_arguments()`: Processes command line options with appropriate defaults
- **Features**:
  - Comprehensive CLI options with help text
  - Argument validation and type checking
  - Grouped parameters for better organization
  - Default values from centralized configuration

**Command Line Options**:

```
usage: real_time.py [-h] [-n NUM_SYMBOLS] [-d DAYS]
                    [-t {1m,5m,15m,30m,1h,4h,1d} [{1m,5m,15m,30m,1h,4h,1d} ...]]
                    [-c CONCURRENCY] [-b BATCH_SIZE] [-s]

Download OHLCV cryptocurrency data from Bybit

optional arguments:
  -h, --help            show this help message and exit
  -n NUM_SYMBOLS, --num-symbols NUM_SYMBOLS
                        Number of cryptocurrencies to download (default: 100)
  -d DAYS, --days DAYS  Days of historical data to download (default: 100)
  -t {1m,5m,15m,30m,1h,4h,1d} [{1m,5m,15m,30m,1h,4h,1d} ...], --timeframes {1m,5m,15m,30m,1h,4h,1d} [{1m,5m,15m,30m,1h,4h,1d} ...]
                        Timeframes to download (default: ['5m', '15m'])

Optimization parameters:
  -c CONCURRENCY, --concurrency CONCURRENCY
                        Maximum number of parallel downloads per batch (default: 5)
  -b BATCH_SIZE, --batch-size BATCH_SIZE
                        Batch size for download (default: 10)
  -s, --sequential      Run in sequential mode instead of parallel
```

**Implementation Details**:
- Uses Python's argparse module for standardized command line handling
- Organizes parameters into logical groupings for better help display
- Implements defaults from centralized configuration for consistency
- Provides choices for enumerated options (like timeframes)
- Shows default values in help text for better user experience

#### `logging_setup.py`
- **Purpose**: Configures application logging with colored output
- **Key Components**:
  - `ColoredFormatter`: Custom log formatter for terminal color coding
  - `setup_logging()`: Initializes the logging system
- **Features**:
  - Level-specific color coding (DEBUG: Cyan, INFO: Green, etc.)
  - Timestamp inclusion in log messages
  - Streamlined formatter with clean visual hierarchy
  - Color reset to prevent terminal pollution

**Colored Logging Implementation**:

```python
class ColoredFormatter(logging.Formatter):
    """
    Custom formatter for console logging with colored output.
    
    Different logging levels get different colors

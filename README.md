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
- [Machine Learning Integration](#machine-learning-integration)
  - [Supervised Learning Dataset Generation](#supervised-learning-dataset-generation)
  - [Dataset Export Format](#dataset-export-format)
  - [Machine Learning Pipeline](#machine-learning-pipeline)
  - [Model Integration](#model-integration)
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
- Comprehensive technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
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
- **Machine Learning Integration**: Dataset preparation for supervised learning models to predict volatility

### Target Users

- Cryptocurrency day traders
- Algorithmic trading strategy developers
- Market data analysts
- Technical analysis enthusiasts
- Machine learning researchers focused on time series prediction

## Machine Learning Integration

TradingJii includes advanced functionality to prepare cryptocurrency volatility data for supervised machine learning algorithms, enabling predictive models for volatility forecasting.

### Supervised Learning Dataset Generation

The platform extends the existing volatility analysis system to generate structured datasets suitable for supervised learning models, especially sequence models like Temporal Fusion Transformers (TFT).

**Key Components:**

- **Data Segmentation**: Overlapping sliding windows capture temporal patterns in volatility
- **Binary Pattern Categorization**: Each window is classified by a binary pattern (e.g., "1010101")  
- **Target Value Assignment**: Each window is paired with the subsequent volatility value for prediction
- **Category-Based Organization**: Similar patterns are grouped to identify recurring behaviors
- **Export Functionality**: Processed data is exported in common ML formats (CSV/Parquet)

#### Implementation Details

The supervised learning dataset generation is implemented in `modules/data/dataset_generator.py` through the `export_supervised_training_data()` function:

```python
def export_supervised_training_data(
    symbol: str, 
    timeframe: str, 
    output_dir: str,
    window_size: int = 7, 
    threshold: float = 0.0
) -> Dict[str, int]:
    """
    Generate and export supervised training data from volatility time series.
    
    Args:
        symbol: The cryptocurrency symbol (e.g., "BTC/USDT")
        timeframe: The timeframe (e.g., "5m")
        output_dir: Directory where datasets will be saved
        window_size: Size of the sliding window (default: 7)
        threshold: Value for categorization (default: 0.0)
        
    Returns:
        Dictionary mapping pattern categories to number of records exported
    """
    # Load volatility data from database
    df = load_volatility_series(symbol, timeframe)
    
    if df.empty:
        logging.warning(f"No volatility data available for {symbol} ({timeframe})")
        return {}
    
    # Generate subseries (windows with targets)
    subseries = generate_subseries(df, window_size)
    
    if not subseries:
        logging.warning(f"Could not generate subseries for {symbol} ({timeframe})")
        return {}
    
    # Organize by categories
    categorized_data = {}
    
    for window, target in subseries:
        # Get the binary pattern for this window
        pattern = categorize_series(window, threshold)
        
        if pattern not in categorized_data:
            categorized_data[pattern] = []
        
        categorized_data[pattern].append((window, target))
    
    # Create output directory structure if it doesn't exist
    # Format: datasets/{symbol}/{timeframe}/
    symbol_safe = symbol.replace('/', '_')
    dataset_path = os.path.join(output_dir, symbol_safe, timeframe)
    os.makedirs(dataset_path, exist_ok=True)
    
    # Export each category to a separate CSV file
    pattern_record_counts = {}
    
    for pattern, data in categorized_data.items():
        # Create DataFrame with x_1, x_2, ..., x_n columns and y column
        rows = []
        for window, target in data:
            row = {f'x_{i+1}': val for i, val in enumerate(window)}
            row['y'] = target
            row['pattern'] = pattern
            rows.append(row)
        
        # Convert to DataFrame
        cat_df = pd.DataFrame(rows)
        
        # Set filename
        filename = os.path.join(dataset_path, f"cat_{pattern}.csv")
        
        # Save to CSV
        cat_df.to_csv(filename, index=False)
        
        # Track counts
        pattern_record_counts[pattern] = len(cat_df)
        
    return pattern_record_counts
```

### Dataset Export Format

The generated datasets follow a specific structure for compatibility with machine learning frameworks:

#### CSV Format Example (cat_1010101.csv)

```
x_1,x_2,x_3,x_4,x_5,x_6,x_7,y,pattern
0.21,-0.05,0.07,0.12,-0.02,-0.03,0.08,-0.11,1010101
0.19,-0.03,0.05,0.10,-0.01,-0.02,0.06,-0.09,1010101
0.23,-0.06,0.08,0.14,-0.03,-0.04,0.09,-0.12,1010101
...
```

#### Directory Structure

```
datasets/
├── BTC_USDT/
│   ├── 5m/
│   │   ├── cat_0000000.csv
│   │   ├── cat_0000001.csv
│   │   ├── cat_0000010.csv
│   │   ├── ...
│   │   └── cat_1111111.csv
│   ├── 15m/
│   │   └── ...
│   └── ...
├── ETH_USDT/
│   └── ...
└── ...
```

### Machine Learning Pipeline

The exported datasets serve as the foundation for a machine learning pipeline focused on volatility prediction:

1. **Data Preparation**: Volatility time series segmented into window/target pairs
2. **Feature Engineering**: Binary patterns provide a categorical representation of market behavior
3. **Model Selection**: Time series models like TFT, LSTM, or transformer-based architectures
4. **Training Process**: Models trained to predict the next volatility value based on pattern history
5. **Evaluation**: Performance measured on held-out test data using metrics like RMSE and MAE
6. **Deployment**: Trained models can be integrated back into TradingJii for real-time predictions

### Model Integration

Future updates to TradingJii will include:

- **Real-time Prediction API**: Endpoints for volatility forecasting based on trained models
- **Prediction Visualization**: UI enhancements to display forecasted volatility alongside historical data
- **Confidence Intervals**: Statistical bounds for volatility predictions to indicate uncertainty
- **Pattern-Specific Models**: Specialized models for each binary pattern category
- **Ensemble Approaches**: Combining multiple models for improved prediction accuracy

#### Usage Example

To generate supervised learning datasets for a cryptocurrency:

```python
from modules.data.dataset_generator import export_supervised_training_data

# Generate datasets for BTC/USDT on 5m timeframe with 7-point window
results = export_supervised_training_data(
    symbol="BTC/USDT",
    timeframe="5m",
    output_dir="datasets",
    window_size=7,
    threshold=0.0
)

# Print category distribution
for pattern, count in sorted(results.items(), key=lambda x: x[1], reverse=True):
    print(f"Pattern {pattern}: {count} records")
```

For batch processing of multiple symbols and timeframes, use the `export_all_supervised_data` function:

```python
from modules.data.dataset_generator import export_all_supervised_data

# Process multiple symbols and timeframes
symbols = ["BTC/USDT", "ETH/USDT"]
timeframes = ["5m", "15m"]
results = export_all_supervised_data(
    symbols=symbols,
    timeframes=timeframes,
    output_dir="datasets",
    window_size=7,
    threshold=0.0
)

# Print overall statistics
print(f"Processed {len(results)} symbol-timeframe combinations")
print(f"Total records: {sum(r['total_records'] for r in results)}")
```

The project includes a ready-to-use command-line script for generating datasets:

```
python generate_datasets.py --symbol "BTC/USDT" --timeframe "5m" --window-size 7
```

For batch processing multiple symbols and timeframes:

```
python generate_datasets.py --batch --symbols "BTC/USDT" "ETH/USDT" --timeframes "5m" "15m"
```

This command will create CSV files in the `datasets/BTC_USDT/5m/` directory, ready for loading into machine learning frameworks like Pandas, PyTorch, or TensorFlow.

## System Architecture

TradingJii follows a clear architectural separation between data collection, processing, and presentation. The system is built with modularity in mind, allowing for independent development and testing of different components.

### Directory Structure

```
TradingJii/
├── app.py                  # Flask web server and API endpoints
├── generate_datasets.py    # CLI script for ML dataset generation
├── real_time.py            # Continuous data collection system
├── requirements.txt        # Python dependencies
├── test_dataset_generator.py # Test script for ML dataset generation
├── .env                    # Environment variables (not in repo)
├── crypto_data.db          # SQLite database (generated)
├── datasets/               # Generated ML datasets (generated)
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
│   │   ├── dataset_generator.py    # ML dataset generation and export
│   │   ├── db_manager.py           # Database operations
│   │   ├── series_segmenter.py     # Pattern identification
│   │   └── volatility_processor.py # Volatility calculations
│   └── utils/              # Utilities
│       ├── __init__.py
│       ├── command_args.py         # CLI argument parsing
│       ├── config.py               # Configuration parameters
│       └── logging_setup.py        # Colored logging configuration
├── ml_models/              # Machine learning model storage (future)
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
                                 │
                                 ▼
                        ┌──────────────────┐     ┌─────────────────────┐
                        │                  │     │                     │
                        │  ML Dataset      │────►│  Predictive Models  │
                        │  Generation      │     │  (Future)           │
                        │                  │     │                     │
                        └──────────────────┘     └─────────────────────┘
```

### Component Interaction

1. **Data Collection Layer**: Interacts with cryptocurrency exchanges to retrieve market data
2. **Data Processing Layer**: Transforms raw market data into volatility metrics and patterns
3. **Storage Layer**: Persists both raw and processed data in SQLite database
4. **API Layer**: Exposes internal data through RESTful endpoints
5. **Presentation Layer**: Renders data as interactive visualizations
6. **Machine Learning Layer**: Transforms volatility data into supervised learning datasets
7. **Predictive Layer**: (Future) Uses trained models to forecast future volatility

[... rest of README follows ...]

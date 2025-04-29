function check_model_status(model, timeframe) {
    const modelStatus = modelStatuses[model];
    if (!modelStatus) return 'non disponibile';
    
    const timeframeStatus = modelStatus.timeframes[timeframe];
    if (!timeframeStatus) return 'non disponibile';
    
    return timeframeStatus.status === 'disponibile' ? 'disponibile' : 'non disponibile';
}

function updateModelStatus() {
    const modelStatuses = {
        'LSTM': {
            timeframes: {
                '5m': { status: 'disponibile', last_update: '2024-03-20 10:00' },
                '15m': { status: 'disponibile', last_update: '2024-03-20 10:00' },
                '30m': { status: 'disponibile', last_update: '2024-03-20 10:00' },
                '1h': { status: 'disponibile', last_update: '2024-03-20 10:00' },
                '4h': { status: 'disponibile', last_update: '2024-03-20 10:00' }
            }
        },
        'Transformer': {
            timeframes: {
                '5m': { status: 'disponibile', last_update: '2024-03-20 10:00' },
                '15m': { status: 'disponibile', last_update: '2024-03-20 10:00' },
                '30m': { status: 'disponibile', last_update: '2024-03-20 10:00' },
                '1h': { status: 'disponibile', last_update: '2024-03-20 10:00' },
                '4h': { status: 'disponibile', last_update: '2024-03-20 10:00' }
            }
        },
        'XGBoost': {
            timeframes: {
                '5m': { status: 'disponibile', last_update: '2024-03-20 10:00' },
                '15m': { status: 'disponibile', last_update: '2024-03-20 10:00' },
                '30m': { status: 'disponibile', last_update: '2024-03-20 10:00' },
                '1h': { status: 'disponibile', last_update: '2024-03-20 10:00' },
                '4h': { status: 'disponibile', last_update: '2024-03-20 10:00' }
            }
        }
    };
} 
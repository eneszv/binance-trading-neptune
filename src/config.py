import os


config = {
    'paper_traing_config': {
        's3_bucket': 'binance-trading-data',
        's3_res_file_path': 'out',
        's3_model_path': 'out',
        'interval': '1h',
        'symbol': 'BTCUSDT',
        'quantity': 0.005, 
        'step': 30,
        'optimization_trials': 2,
        # how many last trades to consider for retraing model
        'recent_trade_num': 24,
        'retraining_thr': 0.6
    },
    'data_config':{
        's3_bucket': 'binance-trading-data',
        's3_data_path': 'out/BTCUSDT/1h',
        'names': ["open time", "open", "high", "low", "close", "volume",
                    "close time", "quote asset volume", "number of trades",
                    "taker buy base asset volume", "taker buy quote asset volume", "date"],
        'columns': ["open time", "open", "high", "low", "close", "volume"], 
        'features':['MA_5', 'MA_10', 'MA_15', 'MA_20',
                    'RSI_7', 'MFI_7', 'RSI_14', 'MFI_14',
                    'RSI_21', 'MFI_21']
    }
}

import os
import json
from datetime import datetime, timedelta
import tempfile
import pandas as pd
import numpy as np
from binance.client import Client
from xgboost import XGBClassifier
import talib
import optuna
from tqdm import tqdm
from unittest.mock import Mock
from config import config
import boto3
import joblib
import neptune.new as neptune
import matplotlib.pyplot as plt


class PaperTrader():
    
    def __init__(self, config, debug=False):
        self.model_config = config['paper_traing_config']
        self.data_config = config['data_config']
        self.debug = debug
        self.s3_client = boto3.client('s3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'), 
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))
        self.s3_res = boto3.resource('s3',
            aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'), 
            aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))

    def log_in(self):
        
        if self.debug:
            self.client = Mock()
            self.client.create_order = lambda symbol, side, type, quantity:{
                "executedQty":1,
                "cummulativeQuoteQty":np.random.rand()+1
            }
            return

        self.client = Client(api_key = os.environ.get('BINANCE_TESTNET_API'),
                             api_secret = os.environ.get('BINANCE_TESTNET_SECRET'),
                             tld = 'com',
                             testnet = True)

    def save_metadata(self):
        
        try:
            self.s3_client.put_object(
                Body=json.dumps(self.metadata),
                Bucket=self.model_config['s3_bucket'],
                Key='{}/metadata.json'.format(self.model_config['s3_model_path'])
            )
        except:
            print('Metadata not saved')

    def load_metadata(self):
        
        try: 
            obj = self.s3_res.Object(
                self.model_config['s3_bucket'],
                '{}/metadata.json'.format(self.model_config['s3_model_path']))

            data = obj.get()['Body'].read().decode('utf-8')
            self.metadata = json.loads(data)
        except:
            print('Metadata not found')
            self.metadata = {}

    def save_res_data(self, df):
        try:
            df.to_csv('s3://{}/{}/results.csv'.format(
                self.model_config['s3_bucket'],
                self.model_config['s3_res_file_path']), index=False)
        except:
            print('Result data not saved')
    
    def load_res_data(self):
        try:
            df_res = pd.read_csv('s3://{}/{}/results.csv'.format(
                self.model_config['s3_bucket'],
                self.model_config['s3_res_file_path']))
        except:
            print('Result file not found. Returning empty data frame.')
            df_res = pd.DataFrame()
            
        return df_res
    
    def save_model(self, clf):
        
        try:
            with tempfile.TemporaryFile() as fp:
                joblib.dump(clf, fp)
                fp.seek(0)
                self.s3_client.put_object(
                    Body=fp.read(),
                    Bucket=self.model_config['s3_bucket'],
                    Key='{}/model_xgb.pkl'.format(self.model_config['s3_model_path'])
                )
        except Exception as e:
            print('Model not saved')
            print(e)
    
    def load_model(self):
        
        with tempfile.TemporaryFile() as fp:
            self.s3_client.download_fileobj(
                Fileobj=fp,
                Bucket=self.model_config['s3_bucket'],
                Key='{}/model_xgb.pkl'.format(self.model_config['s3_model_path'])
            )
            fp.seek(0)
            model = joblib.load(fp)
        return model
    
    def get_df_from_bars(self, start_time):
        
        if self.debug:
            return self.return_dummy_dataset(start_time, 1500)

        bars = self.client.get_historical_klines(symbol = self.model_config['symbol'],
                                    interval = self.model_config['interval'],
                                    start_str = str(start_time),
                                    end_str = None,
                                    limit = 10000)

        df = pd.DataFrame(bars)
        df["date"] = pd.to_datetime(df.iloc[:,0], unit = "ms")
        temp_col = list(self.data_config['names'])
        temp_col.insert(-1, "ignore")
        df.columns = temp_col
        df = df[["date", "open", "high", "low", "close", "volume"]]
        for column in df.columns:
            if column != 'date':
                df[column] = pd.to_numeric(df[column], errors = "coerce")
        
        return df

    def get_historical_data(self):

        if self.debug:
            length = 4500
            start_time = datetime.utcnow() - timedelta(hours=length)
            return self.return_dummy_dataset(start_time, length)

        df = pd.DataFrame()

        my_objects = self.s3_res.Bucket(self.data_config['s3_bucket'])
        for o in my_objects.objects.filter(Prefix=self.data_config['s3_data_path']):
            temp_dir = 's3://{}/{}'.format(self.data_config['s3_bucket'], o.key)
            df = df.append(pd.read_csv(temp_dir, names=self.data_config['names']))
        
        df['date'] = pd.to_datetime(df['open time'], unit='ms')
        df = df[["date", "open", "high", "low", "close", "volume"]]

        return df


    def generate_features(self, df):
        
        df = df.sort_values(by = 'date')
        
        d1 = df['date'].to_list()[0]
        d2 = df['date'].to_list()[-1]
        
        df = df.set_index('date')
        df = df.reindex(pd.date_range(d1, d2, freq='H')).fillna(method = 'ffill')
        df["return"] = df['close'].pct_change()

        for i in [5, 10, 15, 20]:
            df[f'MA_{i}'] = talib.MA(df['close'], timeperiod=i)
            df[f'MA_{i}'] = df[f'MA_{i}']/df['close']
        
        for i in [7, 14, 21]:
            df[f'RSI_{i}'] = talib.RSI(df['close'], timeperiod=i)
            df[f'MFI_{i}'] = talib.MFI(df['high'],
                                       df['low'],
                                       df['close'],
                                       df['volume'],
                                       timeperiod=i)
        
        df = df.dropna() 
        df['target_return'] = df['return'].shift(-1)
        df['target'] = df['target_return'].apply(lambda x: 1 if x > 0 else 0)

        return df

    def predict_movement(self, df):

        df_res = self.load_res_data()
        model = 0

        if len(df_res) > self.model_config['recent_trade_num'] + 1:
            df_temp_res = df_res[-(self.model_config['recent_trade_num'] + 1):].copy()
            df_temp_res['price_sign'] = np.sign(df_temp_res['closed_trade_price'] - df_temp_res['price'])
            df_temp_res['price_sign'] = df_temp_res['price_sign'].map({-1:0})
            df_temp_res['correct_trade'] = df_temp_res[['trade', 'price_sign']].apply(
                lambda row: 1 if row[0] == row[1] else 0, axis = 1
            )
            print('sum corr trades :{}'.format(sum(df_temp_res['correct_trade'])))
            if sum(df_temp_res['correct_trade'])/self.model_config['recent_trade_num'] < self.model_config['retraining_thr']:
                print('retraining model')
                model = self.train_model()

        if model == 0:
            try:
                model = self.load_model()
            except:
                print('Model not found. Training new model is started.')
                model = self.train_model()


        pred = model.predict(df[self.data_config['features']][-1:])[0]

        # close previous trade
        if 'last_trade' in self.metadata:
            if self.metadata['last_trade'] == 1:
                #sell
                order = self.client.create_order(symbol = self.model_config['symbol'],
                                                 side = "SELL",
                                                 type = "MARKET",
                                                 quantity = self.model_config['quantity'])
            else:
                #buy
                order = self.client.create_order(symbol = self.model_config['symbol'],
                                                 side = "BUY",
                                                 type = "MARKET",
                                                 quantity = self.model_config['quantity'])
            base_units = float(order["executedQty"])
            quote_units = float(order["cummulativeQuoteQty"])
            closed_price = round(quote_units / base_units, 5)

        # fill closed trade price
        if len(df_res):
            df_temp = df_res[-1:]
            df_temp['closed_trade_price'] = closed_price
            df_res = df_res[:-1]
            df_res = df_res.append(df_temp)


        if pred == 1:
            #BUY
            order = self.client.create_order(symbol = self.model_config['symbol'],
                                                 side = "BUY",
                                                 type = "MARKET",
                                                 quantity = self.model_config['quantity'])
        else:
            order = self.client.create_order(symbol = self.model_config['symbol'],
                                                 side = "SELL",
                                                 type = "MARKET",
                                                 quantity = self.model_config['quantity'])

        base_units = float(order["executedQty"])
        quote_units = float(order["cummulativeQuoteQty"])
        price = round(quote_units / base_units, 5)
        self.metadata['last_trade'] = int(pred)
        print(self.metadata)

        df_res = df_res.append(pd.DataFrame({'date':[datetime.utcnow()],
                                'trade':[pred],
                                'price':[price],
                                'closed_trade_price':[np.nan]}))

        self.save_res_data(df_res)
        self.df_res = df_res

        self.save_metadata()


    def train_model(self):

        # prepare data
        df_his = self.get_historical_data()
        last_date = max(df_his['date']) + timedelta(hours = 1)
        print(last_date)
        df_curr = self.get_df_from_bars(last_date)
        df = df_his.append(df_curr)
        df = df.drop_duplicates(subset=['date'])

        self.df_training_data = self.generate_features(df)
        self.df_training_data = self.df_training_data.dropna()

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=self.model_config['optimization_trials'])

        best_params = study.best_params
        look_back = best_params.pop('look_back')
        self.metadata['params'] = best_params

        df_train = self.df_training_data[-look_back*24:]
        x_tr = df_train[self.data_config['features']]
        y_tr = df_train['target']

        clf = XGBClassifier(**best_params,  objective = 'binary:logistic')
        clf.fit(x_tr, y_tr, eval_metric = 'logloss')

        self.save_model(clf)
        
        return clf

    def optimize_params(self, params, look_back):

        df_res = pd.DataFrame()
        i = 0
        with tqdm(total = len(self.df_training_data)) as pbar:
            pbar.update(look_back*24)
            while True:

                train_start = i*self.model_config['step']*24
                train_end = train_start + look_back*24
                test_end = train_end + self.model_config['step']*24
                if train_end >= len(self.df_training_data):
                    break

                df_train = self.df_training_data[train_start:train_end]
                df_test = self.df_training_data[train_end:test_end]

                x_tr = df_train[self.data_config['features']]
                x_test = df_test[self.data_config['features']]
                y_tr = df_train['target']

                clf = XGBClassifier(**params,  objective = 'binary:logistic')
                clf.fit(x_tr, y_tr, eval_metric = 'logloss')

                pred = clf.predict(x_test)
                df_pred = pd.DataFrame({
                    'date': df_test.index,
                    'target_return': df_test['target_return'],
                    'target': df_test['target'],
                    'prediction': pred
                })
                if len(df_res):
                    df_res = df_res.append(df_pred)
                else:
                    df_res = df_pred
                pbar.update(self.model_config['step']*24)
                i+=1
        return df_res

    @staticmethod
    def get_score(df_res):
        
        df_res['hourly_return'] = df_res[['target_return', 'target', 'prediction']].apply(
           lambda row: np.abs(row[0]) if row[1] == row[2] else -np.abs(row[0]) , axis=1
        )
        
        df_res['cum_ret'] = df_res['hourly_return'].cumsum()
        return df_res['cum_ret'].to_list()[-1]

    def objective(self, trial):

        params = {
            'n_estimators': trial.suggest_int('n_estimators', 350, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.10),
            'subsample': trial.suggest_uniform('subsample', 0.50, 0.90),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.50, 0.90),
            'gamma': trial.suggest_int('gamma', 0, 20), 
        }
        
        look_back = trial.suggest_int('look_back', 30, 180)

        df_res = self.optimize_params(params, look_back)

        return self.get_score(df_res)

    @staticmethod
    def return_dummy_dataset(start_time, length):
        return pd.DataFrame({
                'date':pd.date_range(start_time, start_time+timedelta(hours=length), freq='H'),
                'open': np.random.rand(length+1)+1,
                'high': np.random.rand(length+1)+2,
                'low': np.random.rand(length+1),
                'close': np.random.rand(length+1)+1,
                'volume': np.random.rand(length+1)*100})


    def neptune_log(self):
        
        dft = self.df_res.copy()
        dft = dft.dropna()
        if len(dft) == 0:
            return
        
        run = neptune.init(
            project=os.environ.get('NEPTUNE_PROJECT'),
            api_token=os.environ.get('NEPTUNE_API_TOKEN'),
        )
        
        dft['price_change'] = (dft['closed_trade_price'] - dft['price'])/dft['price']
        dft['price_change_direction'] = dft['price_change'].apply(lambda x: 1 if x>0 else 0)
        dft['hourly_return'] = dft[['price_change', 'price_change_direction', 'trade']].apply(
            lambda row: np.abs(row[0]) if row[1] == row[2] else -np.abs(row[0]) , axis=1
        )

        fig = plt.figure(figsize =(4, 4))
        dft['cum_ret'] = dft['hourly_return'].cumsum()
        dft.set_index('date')['cum_ret'].plot()

        run["static-img"].upload(neptune.types.File.as_image(fig))
        run["interactive-img"].upload(neptune.types.File.as_html(fig))
        run['data/results'].upload(neptune.types.File.as_html(dft))

        run.stop()

    
    def execute_trade(self):
        
        self.log_in()
        self.load_metadata()
        now = datetime.utcnow()
        start_time = str(now - timedelta(hours = 30))
    
        df = self.get_df_from_bars(start_time)
        df = self.generate_features(df)

        self.predict_movement(df)
        self.neptune_log()
    
    def test_execute_trade(self):
        
        if not os.path.exists('../tt'):
            os.makedirs('../tt')
        self.log_in()
        self.load_metadata()
        start_time = datetime.utcnow()
    
        df = self.get_df_from_bars(start_time)
        df = self.generate_features(df)

        self.predict_movement(df)

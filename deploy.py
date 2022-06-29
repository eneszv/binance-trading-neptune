import os
os.system('sudo docker build -f Dockerfile -t binance_trading .')
os.system('sudo docker run -d --env-file=/home/ubuntu/.env -ti binance_trading')

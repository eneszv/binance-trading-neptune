FROM ubuntu:22.10

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt update \
    && apt install -y python3-dev wget cron gcc vim build-essential

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh \
    && mkdir root/.conda \
    && sh Miniconda3-py37_4.9.2-Linux-x86_64.sh -b \
    && rm -f Miniconda3-py37_4.9.2-Linux-x86_64.sh

RUN conda create -y -n env python=3.7

COPY . binance_trading/

RUN /bin/bash -c " source activate env \ 
              && wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz \
              && tar -xzf ta-lib-0.4.0-src.tar.gz \
              && rm ta-lib-0.4.0-src.tar.gz \
              && cd ta-lib/ \
              && ./configure --prefix=/usr \
              && make \
              && make install \
              && cd ~ \
              && rm -rf ta-lib/ \
              && pip install ta-lib"

RUN /bin/bash -c "source activate env \
    && pip install --upgrade pip \
    && pip install -r binance_trading/requirements.txt"


COPY cron-job /etc/cron.d/cron-job

# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/cron-job

# Apply cron job
RUN /usr/bin/crontab /etc/cron.d/cron-job

# Run the command on container startup
CMD printenv > /etc/environment && cron -f

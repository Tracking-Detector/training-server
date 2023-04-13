FROM tensorflow/tensorflow
COPY requirements.txt /usr/src/app/
WORKDIR /usr/src/app
RUN pip install -r requirements.txt
COPY . /usr/src/app
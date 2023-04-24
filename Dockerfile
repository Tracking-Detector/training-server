FROM tensorflow/tensorflow
COPY requirements.txt /usr/src/app/
COPY ./src/* /usr/src/app/src/
WORKDIR /usr/src/app
RUN pip install -U -r requirements.txt
CMD [ "python3", "src/main.py" ]

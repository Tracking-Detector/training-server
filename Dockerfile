FROM tensorflow/tensorflow
COPY requirements.txt /usr/src/app/
COPY src /usr/src/app/
WORKDIR /usr/src/app
RUN pip install -r requirements.txt
ENTRYPOINT [ "python", "main.py" ]
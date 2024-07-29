FROM public.ecr.aws/lambda/python:3.10.13

RUN mkdir -p /app
COPY ./main.py /app/
COPY ./faiss_store_openai.pkl /app/
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
WORKDIR /app
EXPOSE 8080
CMD [ "main.py" ]
ENTRYPOINT [ "python" ]
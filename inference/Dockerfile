FROM tensorflow/serving:2.15.1

ENV MODEL_NAME=textcls

ARG MODEL_VERSION=v1.0.0
COPY artifacts/versioned/${MODEL_VERSION}/saved_model.tar.gz /tmp/saved_model.tar.gz

RUN mkdir -p /models/${MODEL_NAME}/1 && \
    tar -xzf /tmp/saved_model.tar.gz -C /models/${MODEL_NAME}/1 && \
    rm /tmp/saved_model.tar.gz && \
    if [ -f /models/${MODEL_NAME}/1/saved_model/saved_model.pb ]; then \
      mv /models/${MODEL_NAME}/1/saved_model/* /models/${MODEL_NAME}/1/ && \
      rmdir /models/${MODEL_NAME}/1/saved_model ; \
    fi && \
    test -f /models/${MODEL_NAME}/1/saved_model.pb

EXPOSE 8501

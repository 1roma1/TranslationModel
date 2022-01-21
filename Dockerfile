ARG TF_SERVING_VERSION=latest
ARG TF_SERVING_BUILD_IMAGE=tensorflow/serving:${TF_SERVING_VERSION}-devel

FROM ${TF_SERVING_BUILD_IMAGE} as build_image
FROM ubuntu:18.04

ARG TF_SERVING_VERSION_GIT_BRANCH=master
ARG TF_SERVING_VERSION_GIT_COMMIT=head

LABEL maintainer="gvasudevan@google.com"
LABEL tensorflow_serving_github_branchtag=${TF_SERVING_VERSION_GIT_BRANCH}
LABEL tensorflow_serving_github_commit=${TF_SERVING_VERSION_GIT_COMMIT}

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install TF Serving pkg
COPY --from=build_image /usr/local/bin/tensorflow_model_server /usr/bin/tensorflow_model_server

# Set where models should be stored in the container
ENV MODEL_BASE_PATH=/models
RUN mkdir -p ${MODEL_BASE_PATH}

# The only required piece is the model name in order to differentiate endpoints
ENV MODEL_NAME=model

COPY models/eng_ru models/eng_ru 
COPY models/ru_eng models/ru_eng
COPY models/models.config models/models.config

RUN echo '#!/bin/bash \n\n\
tensorflow_model_server  --rest_api_port=$PORT \
--model_config_file=models/models.config \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh

CMD ["/usr/bin/tf_serving_entrypoint.sh"]
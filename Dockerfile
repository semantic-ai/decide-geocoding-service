FROM semtech/mu-python-template:feature-fastapi
LABEL maintainer="ward@ml2grow.com"

# Install ngrok v3 for eTranslation callback tunneling
# Requires NGROK_AUTHTOKEN env var to be set
RUN apt-get update && apt-get install -y curl unzip ca-certificates && \
    curl -sL https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz -o /tmp/ngrok.tgz && \
    tar -xzf /tmp/ngrok.tgz -C /usr/local/bin && \
    rm /tmp/ngrok.tgz && \
    chmod +x /usr/local/bin/ngrok && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    ngrok version && \
    echo "ngrok v3 installation verified"

RUN uv run hf download svercoutere/RoBERTa-NER-BE-Loc
ENV NER_MODEL_PATH=/root/.cache/huggingface/hub/models--svercoutere--RoBERTa-NER-BE-Loc/snapshots/423e85a3f6be1511335e9a46f4a120af046dcda5

# Download spaCy models directly into the virtual environment
RUN uv pip install https://github.com/explosion/spacy-models/releases/download/nl_core_news_sm-3.8.0/nl_core_news_sm-3.8.0-py3-none-any.whl
RUN uv pip install https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/de_core_news_sm-3.8.0-py3-none-any.whl
RUN uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

RUN uv pip install transformers==4.57.1

ENV EXPECTED_TASK_PREDICATE=http://www.w3.org/ns/adms#status
ENV EXPECTED_TASK_OBJECT=http://redpencil.data.gift/id/concept/JobStatus/scheduled
ENV BASE_REGISTRY_URI=https://api.basisregisters.vlaanderen.be
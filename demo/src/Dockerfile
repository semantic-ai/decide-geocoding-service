FROM python:3.12

COPY requirements.txt /tmp/requirements.txt
COPY streamlit_location_app.py /app/streamlit_location_app.py

RUN set -x \
    && pip install -r /tmp/requirements.txt \
    && rm -f /tmp/requirements.txt \
    && hf download lblod/RoBERTa-NER-BE-Loc

ENV HF_MODEL_PATH=/root/.cache/huggingface/hub/models--lblod--RoBERTa-NER-BE-Loc/snapshots/423e85a3f6be1511335e9a46f4a120af046dcda5
ENV NOMINATIM_URL=http://localhost:8080

WORKDIR /app

ENTRYPOINT ["streamlit", "run", "streamlit_location_app.py"]

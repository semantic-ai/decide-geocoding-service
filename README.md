# Geocoding & NER Service

This repo provides a unified service that processes text to extract and resolve geolocations, as well as legal entities. It combines natural language processing (NLP) for entity extraction with geocoding to map locations to geographic coordinates.

Location extraction uses the RobBERT NER model (Ghent-focused). Entity extraction supports Dutch, German, English with multiple extraction methods (composite, HuggingFace, Flair, regex).

## Features

- **Location extraction**: CITY, DOMAIN, HOUSENUMBERS, INTERSECTION, POSTCODE, PROVINCE, ROAD, STREET (Ghent-focused, geocoded via Nominatim)
  - Model: [svercoutere/RoBERTa-NER-BE-Loc](https://huggingface.co/svercoutere/RoBERTa-NER-BE-Loc)
- **Legal entities**: DATE, LOCATION, LEGAL_GROUND, ADMINISTRATIVE_BODY, MANDATARY (HuggingFace, default composite method)
  - Model: [PedroDKE/multilingual-ner-abb](https://huggingface.co/PedroDKE/multilingual-ner-abb)
- **Document fields**: TITLE (LLM-based extraction)
  - Model: [javdrher/decide-gemma3-270m](https://huggingface.co/javdrher/decide-gemma3-270m)
- **Refined types**: Dates (publication_date, session_date, legal_date, etc.) and locations (impact_location, context_location)
  - Model: [svercoutere/longformer-classifier-refinement-abb](https://huggingface.co/svercoutere/longformer-classifier-refinement-abb)
- **Translation**: HuggingFace, eTranslation (EU)
- **Storage**: All annotations stored in SPARQL triplestore with full provenance

## Requirements

- Docker and Docker Compose
- Access to SPARQL triplestore (configured via `MU_SPARQL_ENDPOINT`)

## Quick Start

```bash
docker compose up
```

The service listens on port 8082 (configurable in `docker-compose.yaml`). Send delta notifications to `POST /delta` to trigger task processing.

For eTranslation callback setup, see [fastapi_callback_setup.md](fastapi_callback_setup.md).

## API Endpoints

- `POST /delta` - Receive delta notifications for task processing
- `GET /task/operations` - List available task operations
- `POST /etranslation/callback` - eTranslation service callback endpoint

## Configuration

Main settings in `config.json`:

- **`ner`**: Method (`composite`, `spacy`, `huggingface`, `flair`), language, labels, refinement
  - `labels`: For location extraction, only entities matching these labels are kept (e.g., `["CITY", "STREET", "POSTCODE"]`)
  - `post_process`: When enabled, overlapping entities with the same label are resolved (keeps highest confidence)
- **`translation`**: Provider (`etranslation`, `huggingface`), target language, eTranslation credentials/callback URL
- **`geocoding`**: Nominatim base URL
- **`llm`**: Model name, API key for classification tasks

Ensure the required Docker network exists (see `docker-compose.yaml`).

## How It Works

1. **Delta notification** → `POST /delta` receives task creation events
2. **Task execution** → Task type determines processing:
   - `geo-extracting`: Extract locations (filtered by `config.ner.labels`), geocode via Nominatim
   - `entity-extracting`: Extract legal entities (DATE, LOCATION, LEGAL_GROUND, ADMINISTRATIVE_BODY, MANDATARY) + title
   - `translating`: Translate text using configured provider
   - `model-annotating`: LLM classification (SDG codes)
3. **Processing** → Apply refinement, geocoding, translation as needed. Overlapping entities are filtered (when `post_process` enabled)
4. **Storage** → Annotations stored in SPARQL triplestore (AI graph) with full provenance

## Deployment

**Network**: Ensure `app-decide_default` network exists or update network name in `docker-compose.yaml`

**Environment variables**:
- `MU_SPARQL_ENDPOINT`: SPARQL endpoint URL
- `EXPECTED_TASK_PREDICATE`: Task status predicate
- `EXPECTED_TASK_OBJECT`: Scheduled status object

**Ports**: Service runs on port 80 internally. Map to host port in `docker-compose.yaml` (default: `8082:80`). For production, expose via dispatcher instead of opening ports directly.

**Volumes**: 
- `config.json` mounted read-only
- `nominatim-data` for Nominatim database persistence

**eTranslation callbacks**: Use dispatcher to route `/etranslation/*` to the service. Set `callback_url` in `config.json` to dispatcher's public URL. See [fastapi_callback_setup.md](fastapi_callback_setup.md) for details.

## Nominatim

Nominatim is included in docker-compose. First startup can take longer than usual to build the database (one-time setup, persisted via volume).

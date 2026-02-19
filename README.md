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

Main settings in `config.json`. The file is validated at startup via Pydantic — invalid values cause startup failure with a descriptive error.

---

### `app`
Controls runtime behaviour.

| Key | Values | Effect |
|---|---|---|
| `mode` | `development`, `production`, `staging`, `test` | Informational; influences log verbosity expectations |
| `log_level` | `debug`, `info`, `warning`, `error` | Controls how much is logged to stdout |

---

### `ner`
Used by `geo-extracting` and `entity-extracting` tasks.

| Key | Values / Default | Effect |
|---|---|---|
| `language` | `nl` / `de` / `en` | Default language when no other language is detected |
| `method` | `composite` *(default)*, `huggingface`, `flair`, `spacy`, `regex`, `title` | Extraction backend. `composite` runs HuggingFace + Flair and merges results. `title` is LLM-based and only used internally for the title field |
| `labels` | List of label strings | **Geo-extracting only**: entities whose label is not in this list are dropped before geocoding. Entity-extracting ignores this filter |
| `post_process` | `true` / `false` | When `true`, overlapping entities with the same label are deduplicated (highest confidence wins). Disable only for debugging |
| `enable_refinement` | `true` / `false` | When `true`, generic labels (`DATE`, `LOCATION`) are passed through the refinement model ([svercoutere/longformer-classifier-refinement-abb](https://huggingface.co/svercoutere/longformer-classifier-refinement-abb)) which classifies them into specific subtypes: `publication_date`, `session_date`, `entry_date`, `expiry_date`, `legal_date`, `context_date`, `validity_period`, `impact_location`, `context_location`, `context_period`. **If disabled, only the raw `DATE` / `LOCATION` labels are available for predicate mapping** |
| `label_to_predicate` | Dict of `"LABEL": "prefix:predicate"` | Maps a (refined) label to an RDF predicate stored in the triplestore. **Labels with an empty or missing mapping are silently skipped** — no annotation is stored for them. Example: `"PUBLICATION_DATE": "eli:date_publication"` |

**Interaction between `enable_refinement` and `label_to_predicate`**: if refinement is on, map the refined labels (e.g. `PUBLICATION_DATE`). If refinement is off, map the raw labels (e.g. `DATE`). Mapping `DATE` while refinement is on will never match because `DATE` gets replaced by a refined subtype.

---

### `translation`
Used by the `translating` task.

| Key | Values / Default | Effect |
|---|---|---|
| `target_language` | `en`, `nl`, `de`, `fr`, `es` | Language to translate into |
| `provider` | `huggingface` *(default)*, `etranslation`, `gemma`, `google`, `microsoft`, `deepl`, `libre`, `auto` | Which translation backend to use. `huggingface` runs locally (Helsinki-NLP OPUS-MT, no credentials needed). `etranslation` uses the EU Commission API (requires credentials). `auto` tries providers in order until one succeeds |

**`etranslation` sub-keys** (only relevant when `provider` is `etranslation`):

| Key | Default | Effect |
|---|---|---|
| `bearer_token` / `username` + `password` | `null` | Authentication — at least one must be set |
| `callback_url` | `null` | Public URL the EU API will POST results to — must be reachable from the internet |
| `max_text_length` | `4000` | Characters per chunk sent to the API. Longer texts are split automatically |
| `callback_wait_timeout` | `180` | Seconds to wait for a callback before failing |
| `domain` | `GEN` | Translation domain hint for the EU API |

---

### `geocoding`
Used by `geo-extracting`.

| Key | Effect |
|---|---|
| `nominatim_base_url` | URL of the Nominatim instance (e.g. `http://nominatim:8080` when using the bundled container) |

---

### `llm`
Used by `entity-extracting` (title extraction) and `model-annotating` (SDG classification).

| Key | Default | Effect |
|---|---|---|
| `model_name` | `gpt-4o-mini` | OpenAI-compatible model name |
| `api_key` | `null` | Required for OpenAI / Azure endpoints |
| `temperature` | `0.1` | Lower = more deterministic output |

---

### `segmentation`
Used by the `segmenting` task.

| Key | Default | Effect |
|---|---|---|
| `model_name` | `gpt-4.1` | Set to `wdmuer/decide-marked-segmentation` to use the local specialized model instead of a generic LLM |
| `api_key` / `endpoint` | `null` | Required for external LLM endpoints |
| `max_new_tokens` | `14000` | Generation budget; reduce for faster but potentially truncated output |
| `temperature` | `0.1` | Lower = more deterministic segmentation |

---

### `ml_training`
Used by the `classifier-training` task only; not needed for normal inference.

| Key | Default | Effect |
|---|---|---|
| `transformer` | `distilbert/distilbert-base-uncased` | Base model to fine-tune |
| `learning_rate` / `epochs` / `weight_decay` | standard defaults | Standard training hyperparameters |
| `huggingface_token` / `huggingface_output_model_id` | `null` | Set to push the trained model to HuggingFace Hub after training |

---

Ensure the required Docker network exists (see `docker-compose.yaml`).

## How It Works

1. **Delta notification** → `POST /delta` receives task creation events
2. **Task execution** → Task type determines processing:
   - `geo-extracting`: Extract locations (filtered by `config.ner.labels`), geocode via Nominatim
   - `entity-extracting`: Extract legal entities (DATE, LOCATION, LEGAL_GROUND, ADMINISTRATIVE_BODY, MANDATARY) + title
   - `translating`: Translate text using configured provider
   - `model-annotating`: LLM classification (SDG codes)
   - `model-batch-annotating`: Batch process multiple decisions for model annotation
   - `classifier-training`: Train classification models
   - `segmenting`: Segment documents using LLM-based segmentation
3. **Processing** → Apply refinement, geocoding, translation as needed. Overlapping entities are filtered (when `post_process` enabled)
4. **Storage** → Annotations stored in SPARQL triplestore (AI graph) with full provenance

## Deployment

**Network**: Ensure `app-decide_default` network exists or update network name in `docker-compose.yaml`

**Environment variables**:
- `MU_SPARQL_ENDPOINT`: SPARQL endpoint URL

**Ports**: Service runs on port 80 internally. Map to host port in `docker-compose.yaml` (default: `8082:80`). For production, expose via dispatcher instead of opening ports directly.

**Volumes**: 
- `config.json` mounted read-only
- `nominatim-data` for Nominatim database persistence

**eTranslation callbacks**: Use dispatcher to route `/etranslation/*` to the service. Set `callback_url` in `config.json` to dispatcher's public URL. See [fastapi_callback_setup.md](fastapi_callback_setup.md) for details.

## Nominatim

Nominatim is included in docker-compose. First startup can take longer than usual to build the database (one-time setup, persisted via volume).


## Segmentation

The service supports two different segmentation models, configurable via `config.json`.

### 1. LLMSegmentor (Generic LLM)
Default usage for general LLMs (OpenAI, Azure OpenAI, Mistral, Ollama, etc.). It instructs the model to return JSON structure.

**Configuration in `config.json`:**
```json
"segmentation": {
  "model_name": "gpt-4.1",       // Your model deployment name
  "api_key": "YOUR_KEY",
  "endpoint": "YOUR_ENDPOINT",
  "temperature": 0.1,
  "max_new_tokens": 14000
}
```

### 2. GemmaSegmentor (Specialized Model)
Uses the specialized `wdmuer/decide-marked-segmentation` model which is trained to output XML tags directly. Use this if you want to reproduce the original/legacy behavior.

**To enable this mode**, you must set the `model_name` specifically:
```json
"segmentation": {
  "model_name": "wdmuer/decide-marked-segmentation",
  "max_new_tokens": 4000
  // ... other fields as needed
}
```
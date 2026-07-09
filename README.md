# Geocoding & NER Service

This repo provides a unified service that processes text to extract and resolve geolocations, as well as legal entities. It combines natural language processing (NLP) for entity extraction with geocoding to map locations to geographic coordinates.

Location extraction uses the RobBERT NER model (Ghent-focused). Entity extraction supports Dutch, German, English with multiple extraction methods (composite, HuggingFace, Flair, regex).

## Features

- **Location extraction**: CITY, DOMAIN, HOUSENUMBERS, INTERSECTION, POSTCODE, PROVINCE, ROAD, STREET (Ghent-focused, geocoded via Nominatim)
  - Model: [svercoutere/RoBERTa-NER-BE-Loc](https://huggingface.co/svercoutere/RoBERTa-NER-BE-Loc)
- **Legal entities**: DATE, LOCATION, LEGAL_GROUND, ADMINISTRATIVE_BODY, MANDATARY (HuggingFace, default composite method)
  - Model: [PedroDKE/multilingual-ner-abb-improved](https://huggingface.co/PedroDKE/multilingual-ner-abb-improved)
- **Refined types**: Dates (publication_date, session_date, legal_date, etc.) and locations (impact_location, context_location)
  - Model: [svercoutere/longformer-classifier-refinement-abb](https://huggingface.co/svercoutere/longformer-classifier-refinement-abb)
- **Translation**: LangChain (Ollama, OpenAI, Mistral, …), HuggingFace, eTranslation (EU)
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

Secrets must be placed in environment variables. See also `docker-compose.yml`. Following variables can be set:
```
# translation
# When translation provider is etranslation, this bearer token will be used for authentication
TRANSLATION__ETRANSLATION__BEARER_TOKEN: "SECRET" 
# When translation provider is etranslation and bearer token is not set or empty, this password (in combination with the username) will be used for authentication
TRANSLATION__ETRANSLATION__PASSWORD: "SECRET" 

# segmentation
# API key for the segmentation LLM provider (OpenAI, Mistral...)
SEGMENTATION__LLM__API_KEY: "SECRET" 
```

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
| `provider` | `langchain` *(default)*, `huggingface`, `etranslation` | Which translation backend to use. `huggingface` runs locally (Helsinki-NLP OPUS-MT, no credentials needed). `etranslation` uses the EU Commission API (requires credentials). `langchain` delegates to a configurable LLM backend (e.g. Ollama, OpenAI, Mistral) |

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
| `api_key` | `null` | Required for external providers (OpenAI, Mistral, …) |
| `temperature` | `0.1` | Lower = more deterministic output |

---

### `segmentation`
Used by the `segmenting` task. Settings are split between top-level keys and the nested `llm` block.

| Key | Default | Effect |
|---|---|---|
| `max_new_tokens` | `14000` | Generation budget for GemmaSegmentor; not used by LLMSegmentor |
| `max_gap` | `5` | Maximum character gap allowed when projecting segments back to the source expression |
| `llm.provider` | `ollama` | LangChain provider name (`ollama`, `openai`, `mistral`, …) |
| `llm.model_name` | `mistral-nemo` | Model name. Set to `wdmuer/decide-marked-segmentation` to use the local GemmaSegmentor instead |
| `llm.api_key` | `null` | API key — required for external providers (OpenAI, Mistral, …) |
| `llm.base_url` | `null` | Custom endpoint URL — required for Ollama and self-hosted models |
| `llm.temperature` | `0.0` | Lower = more deterministic segmentation |

---

### LLM Configuration: Translation & Segmentation

Both **translation** (via `langchain` provider) and **segmentation** delegate to an LLM backend. The system is designed to work with any LangChain-supported provider - local (Ollama) or remote (OpenAI, Mistral, etc.) - with no code changes required. Switch providers by updating `config.json`.

#### Shared `llm` sub-keys

| Key | Default | Effect                                                                                                         |
|---|---|----------------------------------------------------------------------------------------------------------------|
| `provider` | `ollama` | LangChain provider name: `ollama`, `openai`, `mistralai`, `anthropic`, …                                       |
| `model_name` | varies | Model identifier as understood by the provider                                                                 |
| `api_key` | `null` | API key - required for remote providers. **Never commit to config.json**; use the environment variable instead |
| `base_url` | `null` | Custom endpoint - required for Ollama/self-hosted; optional for providers with a default endpoint              |
| `temperature` | `0.0` / `0.1` | Generation temperature - lower = more deterministic                                                            |
| `max_retries` | `3` | Retry attempts on LLM call failure                                                                             |
| `retry_delay` | `15.0` | Seconds between retries                                                                                        |

#### Translation (`translation.langchain`)

Set `translation.provider` to `"langchain"` to enable LLM-based translation. Additional keys:

| Key | Default | Effect |
|---|---|---|
| `max_text_length` | `6000` | Maximum characters per translation chunk |

Add the `translation.langchain` block to `config.json`:

```json
"translation": {
  "target_language": "en",
  "provider": "langchain",
  "langchain": {
    "provider": "ollama",
    "model_name": "mistral-nemo",
    "base_url": "http://host.docker.internal:11434",
    "temperature": 0.1,
    "max_text_length": 6000
  }
}
```

**Example — Ollama (local):**
```json
"langchain": {
  "provider": "ollama",
  "model_name": "mistral-nemo",
  "base_url": "http://host.docker.internal:11434",
  "temperature": 0.1
}
```

**Example — OpenAI:**
```json
"langchain": {
  "provider": "openai",
  "model_name": "gpt-4o-mini",
  "temperature": 0.1
}
```

**Example — Mistral AI:**
```json
"langchain": {
  "provider": "mistralai",
  "model_name": "mistral-small-latest",
  "temperature": 0.1
}
```

#### Segmentation (`segmentation.llm`)

The segmentation task uses the LLM to tag document sections. The `segmentation.llm` block controls which model to use.

**Example - Ollama (local):**
```json
"segmentation": {
  "llm": {
    "provider": "ollama",
    "model_name": "mistral-nemo",
    "base_url": "http://ollama:11434",
    "temperature": 0.0
  },
  "max_new_tokens": 20000,
  "max_gap": 5
}
```

**Example - OpenAI:**
```json
"segmentation": {
  "llm": {
    "provider": "openai",
    "model_name": "gpt-4.1",
    "temperature": 0.0
  },
  "max_new_tokens": 20000,
  "max_gap": 5
}
```

**Example - Mistral AI:**
```json
"segmentation": {
  "llm": {
    "provider": "mistralai",
    "model_name": "mistral-medium-3.5",
    "base_url": "https://api.mistral.ai/v1",
    "temperature": 0.1
  },
  "max_new_tokens": 20000,
  "max_gap": 5
}
```

**GemmaSegmentor (legacy specialized model):** Set `llm.model_name` to `"wdmuer/decide-marked-segmentation"` to use the built-in transformers-based segmentor instead of the generic LLM approach:

```json
"segmentation": {
  "llm": {
    "model_name": "wdmuer/decide-marked-segmentation"
  },
  "max_new_tokens": 4000,
  "max_gap": 5
}
```

#### API Keys via Environment Variables

Never store API keys in `config.json`. Use environment variables with the nested delimiter `__`:

```bash
# Translation langchain API key
TRANSLATION__LANGCHAIN__API_KEY="sk-..."

# Segmentation LLM API key
SEGMENTATION__LLM__API_KEY="sk-..."
```

See `docker-compose.yml` for the full list of supported variables.

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



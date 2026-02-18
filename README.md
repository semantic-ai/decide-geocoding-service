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
and send a request to the /notify-change endpoint on the port as configured in the docker-compose.yaml file.

## Original demo code
The original demo code can be found in the [demo](/demo) folder.




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
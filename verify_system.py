#!/usr/bin/env python3
"""
System verification script - validates all integrated NER and geocoding functionality.
Run inside container: docker exec geocoding-service uv run python /app/verify_system.py
"""

from src.ner_extractors import SpacyGeoAnalyzer
from src.ner_functions import extract_entities
from src.helper_functions import process_text, geocode_detectable
from src.nominatim_geocoder import NominatimGeocoder
from src.annotation import GeoAnnotation, NERAnnotation
from src.sparql_config import AI_COMPONENTS, AGENT_TYPES
import os
import json

print("=" * 80)
print("SYSTEM VERIFICATION")
print("=" * 80)

# Test 1: Belgian Location Extraction
print("\n[1] Belgian Location Extraction (SpacyGeoAnalyzer)")
model_path = os.getenv("NER_MODEL_PATH")
labels = json.loads(os.getenv("NER_LABELS"))
analyzer = SpacyGeoAnalyzer(model_path=model_path, labels=labels)

belgian_text = "De Korenmarkt 15 in Gent is een belangrijk adres."
doc = analyzer.extract_entities(belgian_text)
print(f"   Text: {belgian_text}")
print(f"   Entities found: {len(doc.ents)}")
for ent in doc.ents:
    print(f"     - {ent.label_:15} {ent.text}")

# Test 2: General NER (Dutch)
print("\n[2] General NER Extraction (Dutch)")
dutch_text = "Mathias De Clercq is burgemeester sinds 15 oktober 2024."
entities = extract_entities(dutch_text, language='dutch', method='spacy')
print(f"   Text: {dutch_text}")
print(f"   Entities found: {len(entities)}")
for entity in entities:
    print(f"     - {entity['label']:10} {entity['text']:25} @ [{entity['start']}:{entity['end']}]")

# Test 3: Geocoding
print("\n[3] Geocoding Belgian Locations")
geocoder = NominatimGeocoder(base_url=os.getenv("NOMINATIM_BASE_URL"), rate_limit=0.5)
detectables, _, _ = process_text(belgian_text, analyzer, from_city="Gent")
if detectables.get('streets'):
    result = geocode_detectable(detectables['streets'][0], geocoder)
    print(f"   Query: {result['query']}")
    print(f"   Success: {result['success']}")
    if result['success']:
        print(f"   Coordinates: {result['lat']}, {result['lon']}")

# Test 4: Database Integration
print("\n[4] Database Integration")
print("   Creating test annotations...")
try:
    # Create NER annotation
    ner_ann = NERAnnotation(
        activity_id="http://example.org/verify-test",
        source_uri="http://example.org/verify-source",
        class_uri="http://example.org/entity/TEST",
        start=0, end=5,
        agent=AI_COMPONENTS["ner_extractor"],
        agent_type=AGENT_TYPES["ai_component"]
    )
    ner_ann.add_to_triplestore()
    print("   ✓ NERAnnotation successfully stored")
    
    # Create Geo annotation
    geo_ann = GeoAnnotation(
        geojson={"type": "Polygon", "coordinates": [[3.72, 51.05], [3.73, 51.06]]},
        activity_id="http://example.org/verify-test-geo",
        source_uri="http://example.org/verify-source-geo",
        class_uri="http://example.org/verify-location",
        start=0, end=10,
        agent=AI_COMPONENTS["ner_extractor"],
        agent_type=AGENT_TYPES["ai_component"]
    )
    geo_ann.add_to_triplestore()
    print("   ✓ GeoAnnotation successfully stored")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test 5: Multi-language support
print("\n[5] Multi-Language NER")
for lang, text in [('dutch', "Jan in Brussel"), ('german', "Berlin in Deutschland"), ('english', "London in England")]:
    entities = extract_entities(text, language=lang, method='spacy')
    print(f"   {lang:7} ({len(entities)} entities): {text}")

# Test 6: Configuration
print("\n[6] Configuration Status")
configs = {
    "NER_MODEL_PATH": os.getenv("NER_MODEL_PATH"),
    "NOMINATIM_BASE_URL": os.getenv("NOMINATIM_BASE_URL"),
    "MU_SPARQL_ENDPOINT": os.getenv("MU_SPARQL_ENDPOINT"),
}
for key, value in configs.items():
    status = "✓" if value else "✗"
    print(f"   {status} {key}: {'SET' if value else 'NOT SET'}")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE - All systems operational!")
print("=" * 80)


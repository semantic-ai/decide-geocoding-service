import os
import json

from uuid import uuid4

from ..helper_functions import clean_string, get_start_end_offsets, process_text, geocode_detectable
from ..ner_extractors import SpacyGeoAnalyzer
from ..nominatim_geocoder import NominatimGeocoder
from ..annotation import GeoAnnotation
from ..sparql_config import TASK_OPERATIONS, AI_COMPONENTS, AGENT_TYPES
from .base import DecisionTask


class GeoExtractionTask(DecisionTask):
    """Task that geocodes location information from text."""

    __task_type__ = TASK_OPERATIONS["geo_extraction"]

    ner_analyzer = SpacyGeoAnalyzer(model_path=os.getenv("NER_MODEL_PATH"), labels=json.loads(os.getenv("NER_LABELS")))
    geocoder = NominatimGeocoder(base_url=os.getenv("NOMINATIM_BASE_URL"), rate_limit=0.5)

    def apply_geo_entities(self, task_data: str):
        """Extract geographic entities from text and store as annotations."""
        default_city = "Gent"

        cleaned_text = clean_string(task_data)
        detectables, _, doc = process_text(cleaned_text, self.__class__.ner_analyzer, default_city)

        if hasattr(doc, 'error'):
            self.logger.error(f"Error: {doc['error']}")
        else:
            # Geocoding Results
            if detectables:
                self.logger.info("Geocoding Results")

                for geo_entity in ["streets", "addresses"]:
                    if geo_entity in detectables:
                        for detectable in detectables[geo_entity]:
                            result = geocode_detectable(detectable, self.__class__.geocoder, default_city)
                            print(result)

                            if result["success"]:
                                if geo_entity == "streets":
                                    offsets = get_start_end_offsets(task_data, detectable["name"])
                                    start_offset = offsets[0][0]
                                    end_offset = offsets[0][1]
                                    annotation = GeoAnnotation(
                                        result.get("geojson", {}),
                                        self.task_uri,
                                        self.source,
                                        "http://example.org/{0}".format(uuid4()),
                                        start_offset,
                                        end_offset,
                                        AI_COMPONENTS["ner_extractor"],
                                        AGENT_TYPES["ai_component"]
                                    )
                                    annotation.add_to_triplestore()
                            self.logger.info(result)
            else:
                self.logger.info("No location entities detected.")

    def process(self):
        task_data = self.fetch_data()
        self.logger.info(task_data)
        self.apply_geo_entities(task_data)




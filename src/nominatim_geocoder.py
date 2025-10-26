from typing import Dict, Any, Optional
import requests
import time
import logging


class NominatimGeocoder:
    """Geocoder client for Nominatim OpenStreetMap geocoding service."""
    
    def __init__(self, base_url: str = "http://localhost:8080", rate_limit: float = 1.0, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.rate_limit = max(0.0, rate_limit)
        self.timeout = timeout
        self._last = 0.0
        self._sess = requests.Session()

        self.logger = logging.getLogger(__name__)

    def _throttle(self) -> None:
        """Enforce rate limiting between API requests."""
        now = time.monotonic()
        wait = self.rate_limit - (now - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def search(self, query: str, city: str = "Gent", limit: int = 1, country: str = "BE") -> Optional[Dict[str, Any]]:
        """Search for a location and return geocoded result with coordinates."""
        if not query or not query.strip():
            return None

        self._throttle()
        full_query = f"{query}, {city}" if city else query
        params = {
            "q": full_query,
            "format": "json",
            "limit": limit,
            "countrycodes": country,
            "addressdetails": 1,
            "extratags": 0,
            "namedetails": 0,
            "polygon_geojson": 1
        }

        try:
            resp = self._sess.get(
                f"{self.base_url}/search", params=params, timeout=self.timeout)
            resp.raise_for_status()
            results = resp.json()
            if not results:
                return None
            return self._format(results[0], original_query=query)
        except requests.RequestException as exc:
            self.logger.warning(
                "Nominatim request failed for %r: %s", query, exc)
            return None
        except ValueError as exc:
            self.logger.warning(
                "Failed parsing Nominatim JSON for %r: %s", query, exc)
            return None

    def _format(self, r: Dict[str, Any], original_query: str) -> Dict[str, Any]:
        addr = r.get("address", {})
        osm_type = r.get("osm_type")
        osm_id = r.get("osm_id")
        osm_url = f"https://www.openstreetmap.org/{osm_type}/{osm_id}" if osm_type and osm_id else None

        return {
            "query": original_query,
            "display_name": r.get("display_name"),
            "lat": float(r.get("lat", 0.0)),
            "lon": float(r.get("lon", 0.0)),
            "importance": r.get("importance"),
            "place_id": r.get("place_id"),
            "osm_type": osm_type,
            "osm_id": osm_id,
            "osm_url": osm_url,
            "address": {
                "house_number": addr.get("house_number"),
                "road": addr.get("road"),
                "city": addr.get("city") or addr.get("town") or addr.get("village"),
                "postcode": addr.get("postcode"),
                "country": addr.get("country"),
                "country_code": addr.get("country_code"),
            },
            "bbox": r.get("boundingbox"),
            "type": r.get("type"),
            "class": r.get("class"),
            "geojson": r.get("geojson"),
        }

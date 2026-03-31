import uuid as _uuid_module
from dateperiodparser import DatePeriodParser
from locationformatter import LocationFormatter

_PERIOD_BASE_URI = "https://data.lblod.info/id/period"


class EntityFormatter:
    """Routes extracted NER entities through formatting parsers based on label type.
    """

    DATE_LABELS = frozenset({
        "DATE", "CONTEXT_DATE", "CONTEXT_PERIOD",
        "ENTRY_DATE", "EXPIRY_DATE", "LEGAL_DATE",
        "PUBLICATION_DATE", "SESSION_DATE", "VALIDITY_PERIOD",
    })

    LOCATION_LABELS = frozenset({
        "LOCATION", "LOC", "GPE",
        "CONTEXT_LOCATION", "IMPACT_LOCATION",
    })

    _DATE_OBJECT_MAP: dict[str, list[str]] = {
        "DATE":             ["date"],
        "CONTEXT_DATE":     ["date"],
        "CONTEXT_PERIOD":   ["interval"],
        "ENTRY_DATE":       ["date"],
        "EXPIRY_DATE":      ["date"],
        "LEGAL_DATE":       ["date"],
        "PUBLICATION_DATE": ["date"],
        "SESSION_DATE":     ["date"],
        "VALIDITY_PERIOD":  ["interval"],
    }

    def __init__(self):
        self._date_parser = DatePeriodParser()
        self._location_formatter = LocationFormatter()

    # ── public API ───────────────────────────────────────────────────

    def format(self, entities: list[dict]) -> list[dict]:
        """Format and expand — every entry holds exactly one RDF object / one address."""
        output = []
        for e in entities:
            label = e.get("label", "")
            if label in self.DATE_LABELS:
                first, all_objects = self._format_date(e)
                output.extend(self._expand(first, all_objects, self._date_entry))
            elif label in self.LOCATION_LABELS:
                first, all_locations = self._format_location(e)
                output.extend(self._expand(first, all_locations, self._location_entry))
            else:
                output.append({**e, "formatted": None, "formatted_text": None, "result_object": None, "formatted_start": None, "formatted_end": None})
        return output

    # ── serialisation helpers ────────────────────────────────────────

    @staticmethod
    def _make_interval_obj(start_date: str, end_date: str) -> dict:
        uid = str(_uuid_module.uuid4())
        return {
            "type":         "time:ProperInterval",
            "uuid":         uid,
            "uri":          f"{_PERIOD_BASE_URI}/{uid}",
            "hasBeginning": start_date,
            "hasEnd":       end_date,
        }

    @staticmethod
    def _object_formatted_text(obj: dict) -> str:
        if obj["type"] == "xsd:Date":
            return f'"{obj["value"]}"^^xsd:Date'
        return f'<{obj["uri"]}>'

    @staticmethod
    def _object_result_object(obj: dict) -> str | None:
        if obj["type"] == "xsd:Date":
            return None
        uri = obj["uri"]
        uid = obj["uuid"]
        return (
            f'<{uri}> a <http://www.w3.org/2006/time#ProperInterval> ;\n'
            f'    mu:uuid "{uid}" ;\n'
            f'    <http://www.w3.org/2006/time#hasBeginning> "{obj["hasBeginning"]}"^^xsd:Date ;\n'
            f'    <http://www.w3.org/2006/time#hasEnd> "{obj["hasEnd"]}"^^xsd:Date .'
        )

    @staticmethod
    def _date_parse_dict(dr) -> dict:
        return {
            "start":      dr.start.isoformat(),
            "end":        dr.end.isoformat(),
            "confidence": dr.confidence,
            "pattern":    dr.pattern,
        }

    @staticmethod
    def _address_text(loc: dict) -> str:
        """Build a Nominatim-compatible address string using a priority chain.

        Priority for the primary search term:
          1. street  (+ housenumber + bus N)   — most geocoding-friendly
          2. building                            — named building
          3. road                                — highway / named road
          4. intersection                        — road crossing
          5. district                            — neighbourhood / district
          6. domain_zone_area                    — zone or domain name
          7. grave_location / parcel             — last resort (poor geocoding)
          8. raw location span                   — absolute fallback

        postcode, city and province are appended to whatever primary is chosen.
        """
        postcode_city = " ".join(filter(None, [loc.get("postcode") or "", loc.get("city") or ""]))
        province      = loc.get("province") or ""

        if loc.get("street"):
            hn = loc.get("housenumber") or ""
            if hn and loc.get("bus"):
                hn = f"{hn} bus {loc['bus']}"
            primary = " ".join(filter(None, [loc["street"], hn]))
        elif loc.get("building"):
            primary = loc["building"]
        elif loc.get("road"):
            primary = loc["road"]
        elif loc.get("intersection"):
            primary = loc["intersection"]
        elif loc.get("district"):
            primary = loc["district"]
        elif loc.get("domain_zone_area"):
            primary = loc["domain_zone_area"]
        elif loc.get("grave_location"):
            primary = loc["grave_location"]
        elif loc.get("parcel"):
            primary = loc["parcel"]
        else:
            primary = loc.get("location") or ""

        parts = list(filter(None, [primary, postcode_city, province]))
        return ", ".join(parts)

    # ── private format helpers ───────────────────────────────────────

    def _format_date(self, entity: dict) -> tuple[dict, list[dict]]:
        """Returns (first_entry, all_objects)."""
        parse_results = self._date_parser.parse(entity["text"])
        base = entity.copy()
        if not parse_results:
            base.update({"formatted": None, "formatted_text": None, "result_object": None, "formatted_start": None, "formatted_end": None})
            return base, []

        r = parse_results[0]
        start_date = r.start.isoformat()[:10]
        end_date   = r.end.isoformat()[:10]

        objects = []
        for shape in self._DATE_OBJECT_MAP.get(entity.get("label", ""), []):
            if shape == "interval":
                objects.append(self._make_interval_obj(start_date, end_date))
            else:
                objects.append({"type": "xsd:Date", "value": start_date})

        first_obj = objects[0]
        base["formatted"] = {
            "type": "date",
            **self._date_parse_dict(r),
            "objects": [first_obj],
        }
        if len(parse_results) > 1:
            base["formatted"]["additional"] = [self._date_parse_dict(dr) for dr in parse_results[1:]]
        base["formatted_text"]  = self._object_formatted_text(first_obj)
        base["result_object"]   = self._object_result_object(first_obj)
        base["formatted_start"] = f'"{start_date}"^^xsd:Date'
        base["formatted_end"]   = f'"{end_date}"^^xsd:Date'
        return base, objects

    def _format_location(self, entity: dict) -> tuple[dict, list[dict]]:
        """Returns (first_entry, all_locations)."""
        result = self._location_formatter.parse(entity["text"])
        base = entity.copy()
        if not result or not result.get("locations"):
            base.update({"formatted": None, "formatted_text": None, "result_object": None, "formatted_start": None, "formatted_end": None})
            return base, []

        locations = result["locations"]
        base["formatted"]       = {"type": "location", "locations": [locations[0]]}
        base["formatted_text"]  = self._address_text(locations[0])
        base["result_object"]   = None
        base["formatted_start"] = None
        base["formatted_end"]   = None
        return base, locations

    # ── expansion ────────────────────────────────────────────────────

    @staticmethod
    def _expand(first: dict, all_items: list, entry_fn) -> list[dict]:
        """Expand into one entry per item.  If ≤1 item, return first as-is."""
        if len(all_items) <= 1:
            return [first]
        base = {k: v for k, v in first.items()
                if k not in ("formatted", "formatted_text", "result_object")}
        base_fmt = {k: v for k, v in first.get("formatted", {}).items()
                    if k not in ("objects", "locations")}
        return [entry_fn(base, base_fmt, item) for item in all_items]

    @staticmethod
    def _date_entry(base: dict, base_fmt: dict, obj: dict) -> dict:
        return {
            **base,
            "formatted_text": EntityFormatter._object_formatted_text(obj),
            "result_object":  EntityFormatter._object_result_object(obj),
            "formatted":      {**base_fmt, "objects": [obj]},
        }

    @staticmethod
    def _location_entry(base: dict, base_fmt: dict, loc: dict) -> dict:
        return {
            **base,
            "formatted_text": EntityFormatter._address_text(loc),
            "result_object":  None,
            "formatted":      {**base_fmt, "locations": [loc]},
        }

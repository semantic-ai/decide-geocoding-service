import os
import re
import unicodedata
import requests
from typing import Optional


def clean_string(input_string):
    """Remove extra whitespace and normalize string formatting."""
    cleaned_string = input_string.replace('\n', ' ')
    cleaned_string = cleaned_string.strip()
    cleaned_string = re.sub(r'\s+', ' ', cleaned_string)
    cleaned_string = unicodedata.normalize('NFKD', cleaned_string)
    return cleaned_string


def clean_house_number(housenumber):
    """Clean and standardize house number format."""
    # Split the housenumbers based on "," , "en" and "/"
    housenumber = housenumber.replace("  ", " ").replace(
        "tot en met", 't.e.m.').replace("TOT EN MET", 't.e.m.')
    # Remove "huisnummer" and similar terms
    housenumber = re.sub(r'\b(huisnummer|huisnr|nr|nummer)\b',
                         '', housenumber, flags=re.IGNORECASE)
    # Clean up extra spaces
    housenumber = re.sub(r'\s+', ' ', housenumber).strip()
    parts = [item.strip() for item in re.split(r',|en', housenumber) if item]
    result_list = []

    for part in parts:
        # Remove leading and trailing whitespace
        part = part.strip()

        # Check if the part contains a range with "-"
        if "-" in part:
            # Split by '-' and convert to get the start and end of the range as strings
            segments = part.split("-")
            if len(segments) == 2:
                start, end = segments
                # Check if the start and end are integers
                if start.strip().isdigit() and end.strip().isdigit():
                    # Convert to integers
                    start = int(start.strip())
                    end = int(end.strip())
                    # check if start and end are smaller than 1000:
                    if start < 1000 and end < 1000 and end-start < 20 and end > start:
                        result_list.extend(map(str, range(start, end + 1)))
                    else:
                        # Add both values to the result list
                        result_list.append(start)
                        result_list.append(end)
                else:
                    result_list.append(start)
                    result_list.append(end)

            else:
                for segment in segments:
                    result_list.append(segment)

        # Check for keywords indicating a range
        elif "tot" in part.lower() or "t.e.m." in part.lower():
            # Split by keywords and convert to integers
            numbers = [num for num in re.split(r'\D+', part) if num]
            # Add all values within the range to the result list
            if len(numbers) == 2:
                start, end = map(int, numbers)
                if "tot" in part.lower():
                    end -= 1
                result_list.extend(map(str, range(start, end + 1)))
            else:
                result_list.append(part)
        # Check if the part contains a "/"
        elif "/" in part and "bus" not in part.lower():
            # Split by '/'
            start, end = part.split("/")
            result_list.append(start.strip())
            result_list.append(end.strip())
        else:
            result_list.append(part)
    return result_list


def extract_house_and_bus_number(housenumber):
    """Split house number into main number and bus/apartment number."""
    bus_number = None
    house_number = None

    if "bus" in housenumber:
        parts = housenumber.split("bus")
        if len(parts) > 1 and parts[1].strip().isdigit():
            bus_number = int(parts[1].strip())
        if parts[0].strip():
            house_number = parts[0].strip()
    else:
        house_number = housenumber.strip()

    if house_number and "/" in house_number:
        parts = house_number.split("/")
        house_number = parts[0].strip()

    return {"housenumber": house_number, "bus": bus_number}


def form_addresses(entities, from_city="Gent"):
    """Combine extracted entities into complete address strings."""
    current_address = {"name": None, "house_number": None, "house_numbers": [
    ], "bus": None, "postcode": None, "city": None, "type": "HOUSE", "spacy_entities": []}
    addresses = []

    for entity in entities:
        if entity.label_ == "STREET":
            if current_address["name"] and len(current_address["house_numbers"]) > 0:
                addresses.append(current_address)
                current_address = {"name": None, "house_number": None, "house_numbers": [
                ], "bus": None, "postcode": None, "city": None, "type": "HOUSE", "spacy_entities": []}
            current_address["name"] = entity.text
            current_address["type"] = entity.label_
            current_address["spacy_entities"].append(entity)
        elif entity.label_ == "HOUSENUMBERS":
            current_address["house_numbers"] = clean_house_number(entity.text)
            current_address["spacy_entities"].append(entity)
        elif entity.label_ == "POSTCODE":
            current_address["postcode"] = entity.text
            current_address["spacy_entities"].append(entity)
        elif entity.label_ == "CITY":
            current_address["city"] = entity.text
            current_address["spacy_entities"].append(entity)
            if current_address["name"] and len(current_address["house_numbers"]) > 0:
                addresses.append(current_address)
                current_address = {"name": None, "house_number": None, "house_numbers": [
                ], "bus": None, "postcode": None, "city": None, "type": "HOUSE", "spacy_entities": []}

    if current_address["name"] and len(current_address["house_numbers"]) > 0:
        addresses.append(current_address)

    for address in addresses:
        if not address["city"]:
            address["city"] = from_city

    return addresses


def form_locations(entities, from_city="Gent"):
    """Form location queries from extracted entities for geocoding."""
    current_address = {"name": None, "house_number": None, "house_numbers": [
    ], "bus": None, "postcode": None, "city": None, "type": None, "spacy_entities": []}
    addresses = []

    for entity in entities:
        if entity.label_ in ["DOMAIN", "ROAD", "STREET", 'INTERSECTION']:
            if current_address["name"]:
                addresses.append(current_address)
                current_address = {"name": None, "house_number": None, "house_numbers": [
                ], "bus": None, "postcode": None, "city": None, "type": None, "spacy_entities": []}
            current_address["name"] = entity.text
            current_address["spacy_entities"].append(entity)
            current_address["type"] = entity.label_
        elif entity.label_ == "CITY":
            current_address["city"] = entity.text
            current_address["spacy_entities"].append(entity)
            if current_address["name"]:
                addresses.append(current_address)
                current_address = {"name": None, "house_number": None, "house_numbers": [
                ], "bus": None, "postcode": None, "city": None, "type": None, "spacy_entities": []}

    if current_address["name"]:
        addresses.append(current_address)

    for address in addresses:
        if not address["city"]:
            address["city"] = from_city

    return addresses


def split_addresses(addresses):
    """Split full addresses into street and address components."""
    individual_addresses = []
    for multi_address in addresses:
        for house_number_string in multi_address['house_numbers']:
            house_number_object = extract_house_and_bus_number(
                str(house_number_string))
            individual_address = {
                'name': multi_address['name'],
                'house_number': house_number_object["housenumber"],
                'bus': house_number_object["bus"],
                'postcode': multi_address['postcode'],
                'city': multi_address['city'],
                "type": "HOUSE",
                "spacy_entities": multi_address["spacy_entities"]
            }
            individual_addresses.append(individual_address)
    return individual_addresses


def process_text(text, ner_model, from_city="Gent"):
    """Extract named entities from text and organize them by type."""
    doc = ner_model.extract_entities(text)
    if hasattr(doc, 'error'):
        return [], [], doc

    detected_addresses = form_addresses(doc.ents, from_city)
    detected_locations = form_locations(doc.ents, from_city)
    individual_addresses = split_addresses(detected_addresses)

    detectables = {"streets": detected_locations,
                   "addresses": individual_addresses}

    return detectables, doc.ents, doc


def geocode_detectable(detectable, geocoder, default_city="Gent"):
    """Geocode a detected entity (address or street) and return GeoJSON result."""
    name = detectable.get("name", "")
    if not name:
        return {"success": False, "error": "No name in detectable"}

    if detectable.get("type") == "HOUSE" and detectable.get("house_number"):
        query = f"{name} {detectable['house_number']}"
    else:
        query = name

    city = detectable.get("city", default_city)
    result = geocoder.search(query, city=city)

    if result:
        return {
            "success": True,
            "query": query,
            "display_name": result["display_name"],
            "lat": result["lat"],
            "lon": result["lon"],
            "osm_url": result.get("osm_url"),
            "address": result.get("address"),
            "geojson": result.get("geojson"),
            "detectable": detectable
        }
    else:
        return {
            "success": False,
            "query": query,
            "city": city,
            "error": f"No geocoding result found for '{query}' in {city}",
            "detectable": detectable
        }


def render_entities_html(doc):
    """Render spaCy Doc with highlighted entities as HTML."""
    html_content = ""
    last_end = 0

    for ent in doc.ents:
        # Add text before entity
        html_content += doc.text[last_end:ent.start_char]

        # Add entity with styling
        entity_class = f"entity-{ent.label_}"
        html_content += f'<span class="entity-highlight {entity_class}" title="{ent.label_}">{ent.text}</span>'

        last_end = ent.end_char

    # Add remaining text
    html_content += doc.text[last_end:]

    return html_content


def get_street_uri(gemeentenaam: str, straatnaam: str, timeout: int = 15) -> Optional[str]:
    """Retrieve the URI for a street from an external API."""
    url = f"{os.getenv('BASE_REGISTRY_URI')}/v2/straatnamen/"
    params = {"gemeentenaam": gemeentenaam, "straatnaam": straatnaam}
    r = requests.get(url, params=params, headers={
                     "Accept": "application/json"}, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    items = (data or {}).get("straatnamen", [])
    if not items:
        return None

    ident = items[0].get("identificator", {})
    return url + ident.get("objectId")


def get_address_uri(gemeentenaam: str, straatnaam: str, huisnummer: str, busnummer: Optional[str] = None, timeout: int = 15) -> Optional[str]:
    """Retrieve the URI for a specific address from an external API."""
    url = f"{os.getenv('BASE_REGISTRY_URI')}/v2/adressen/"
    params = {
        "gemeentenaam": gemeentenaam,
        "straatnaam": straatnaam,
        "huisnummer": huisnummer,
    }
    if busnummer is not None:
        params["busnummer"] = busnummer

    r = requests.get(url, params=params, headers={
                     "Accept": "application/json"}, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    items = (data or {}).get("adressen", [])
    if not items:
        return None

    ident = items[0].get("identificator", {})
    return url + ident.get("objectId")


def get_start_end_offsets(text: str, word: str) -> list:
    """Find all start and end positions of a word in text."""
    offsets = []
    start = 0
    while True:
        i = text.find(word, start)
        if i == -1:
            break
        offsets.append((i, i + len(word)))
        start = i + len(word)

    return offsets

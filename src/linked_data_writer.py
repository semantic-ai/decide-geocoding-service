from uuid import uuid4
from helpers import query
from string import Template
from escape_helpers import sparql_escape_uri, sparql_escape_string, sparql_escape_float, sparql_escape_int
from .sparql_config import get_prefixes_for_query, GRAPHS


def get_generic_insertion_query_part() -> Template:
    """Generate the base SPARQL INSERT template for annotations."""
    return Template(
        get_prefixes_for_query("oa", "mu", "dcterms", "rdfs", "locn", "geosparql", "nif", "xsd", "skos") +
        """
    INSERT DATA {
    GRAPH <""" + GRAPHS["ai"] + """> {
        $annotation a oa:Annotation ;
        mu:uuid $uuid ;
        oa:hasBody $body ;
        nif:confidence $confidence ;
        oa:motivatedBy oa:classifying ;
        oa:hasTarget $target .

        $target a oa:SpecificResource ;
        oa:source $source ;
        oa:selector $selector .

        $selector a oa:TextPositionSelector ;
        oa:start $start ;
        oa:end $end .

        $geo_entity_specific_part
    }
    }
    """)


def get_street_annotation_insertion_query_part() -> Template:
    """Generate SPARQL fragment for street-specific annotation data."""
    return Template("""
    $body a dcterms:Location , <https://data.vlaanderen.be/ns/adres#Straatnaam> ;
      rdfs:label $label ;
      locn:geometry $geom ;
      skos:exactMatch $registry_uri .

    $geom a locn:Geometry ;
      geosparql:asWKT $wkt .
    """)


def get_address_annotation_insertion_query_part() -> Template:
    """Generate SPARQL fragment for address-specific annotation data."""
    return Template("""
    $body a dcterms:Location , locn:Address ;
      rdfs:label $label ;
      locn:geometry $geom .

    $geom a locn:Geometry ;
      geosparql:asWKT $wkt .
    """)


def insert_annotation(geo_entity: str, body_uri: str,
                      label: str, geom_uri: str,
                      geometry: str, registry_uri: str,
                      graph_uri: str, annotation_uri: str,
                      confidence: float, target_uri: str,
                      source_doc: str, selector_uri: str,
                      start_offset: int, end_offset: int) -> None:
    """Insert a geographic entity annotation with location data into the triplestore."""
    insertion_query = get_generic_insertion_query_part()

    if geo_entity == "streets":
        geo_entity_query_template_part = get_street_annotation_insertion_query_part()

        geo_entity_query_string = geo_entity_query_template_part.substitute(
            body=sparql_escape_uri(body_uri),
            label=sparql_escape_string(label),
            geom=sparql_escape_uri(geom_uri),
            registry_uri=sparql_escape_uri(registry_uri),
            wkt=sparql_escape_string(f"SRID=4326;LINESTRING({geometry})") + "^^geosparql:wktLiteral")

    elif geo_entity == "addresses":
        geo_entity_query_template_part = get_address_annotation_insertion_query_part()

        geo_entity_query_string = geo_entity_query_template_part.substitute(
            body=sparql_escape_uri(body_uri),
            label=sparql_escape_string(label),
            geom=sparql_escape_uri(geom_uri),
            wkt=sparql_escape_string(f"SRID=31370;POINT({geometry})^^geosparql:wktLiteral"))

    insertion_query_string = insertion_query.substitute(
        graph=sparql_escape_uri(graph_uri),
        annotation=sparql_escape_uri(annotation_uri),
        uuid=sparql_escape_string(str(uuid4())),
        body=sparql_escape_uri(body_uri),
        confidence=sparql_escape_float(confidence),
        target=sparql_escape_uri(target_uri),
        source=sparql_escape_uri(source_doc),
        selector=sparql_escape_uri(selector_uri),
        start=sparql_escape_int(start_offset),
        end=sparql_escape_int(end_offset),
        geo_entity_specific_part=geo_entity_query_string)

    query(insertion_query_string)

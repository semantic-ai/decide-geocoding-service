import logging
from typing import Optional, Iterator, Any
from abc import ABC, abstractmethod
from string import Template
import uuid

from helpers import query
from escape_helpers import sparql_escape_uri, sparql_escape_string, sparql_escape_float, sparql_escape_int
from .sparql_config import get_prefixes_for_query, GRAPHS, AGENT_TYPES


class Annotation(ABC):
    """Base class for Open Annotation objects with provenance information."""
    
    def __init__(self, activity_id: str, source_uri: str, agent: str, agent_type: str):
        super().__init__()
        self.activity_id = activity_id
        self.source_uri = source_uri
        self.agent = agent
        self.agent_type = agent_type

    @classmethod
    def create_from_labelstudio(cls, activity_id: str, uri: str, user: str, annotation: Any) -> Optional['Annotation']:
        """Create an annotation from Label Studio format."""
        if annotation['type'] == 'labels':
            return NERAnnotation(
                activity_id,
                uri,
                annotation['value']['labels'][0],
                annotation['value']['start'],
                annotation['value']['end'],
                user,
                AGENT_TYPES["person"]
            )

        if annotation['type'] == 'choices':
            return LinkingAnnotation(
                activity_id,
                uri,
                annotation['value']['choices'],
                user,
                AGENT_TYPES["person"]
            )

        return None

    @abstractmethod
    def to_labelstudio_result(self) -> dict:
        """Convert annotation to Label Studio result format."""
        pass

    @abstractmethod
    def add_to_triplestore(self):
        """Insert this annotation into the triplestore."""
        pass

    @classmethod
    @abstractmethod
    def create_from_uri(cls, uri: str) -> Iterator['NERAnnotation']:
        """Create annotation instances from a URI in the triplestore."""
        pass


class LinkingAnnotation(Annotation):
    """Annotation linking a resource to a classification/class."""
    
    def __init__(self, activity_id: str, source_uri: str, class_uri: str, agent: str, agent_type: str):
        super().__init__(activity_id, source_uri, agent, agent_type)
        self.class_uri = class_uri

    @classmethod
    def create_from_uri(cls, uri: str) -> Iterator['NERAnnotation']:
        query_template = Template(
            get_prefixes_for_query("oa", "prov", "rdf") +
            """
        SELECT ?activity ?body ?agent ?agentType
        WHERE {
          ?annotation a oa:Annotation ;
                       oa:hasTarget ?target .
          ?annotation oa:hasBody ?body.
          OPTIONAL { ?annotation oa:motivatedBy ?motivation . }

          # Example filter (uncomment and edit as needed):
          FILTER(?target = $uri)
          FILTER(?motivation = oa:classifying)

          OPTIONAL {
              ?activity a prov:Activity ;
              prov:generated ?annotation ;
              prov:wasAssociatedWith ?agent .

              OPTIONAL { ?agent rdf:type ?agentType . }
          }
        }
        """)
        query_result = query(
            query_template.substitute(
                uri=sparql_escape_uri(uri)
            )
        )

        if not query_result['results']['bindings']:
            return
            yield

        for item in query_result['results']['bindings']:
            yield cls(item['activity']['value'], uri, item['body']['value'], item['agent']['value'], item['agentType']['value'])

    def to_labelstudio_result(self):
        return {
            "type": "choices",
            "value": {"choices": [self.class_uri]},
            "origin": "manual", "to_name": "text", "from_name": "entities"
        }

    def add_to_triplestore(self):
        query_template = Template(
            get_prefixes_for_query("ex", "oa", "mu", "prov", "foaf", "dct", "skolem", "nif") +
            """
            INSERT {
              GRAPH <""" + GRAPHS["ai"] + """> {
                  $activity_id a prov:Activity;
                     prov:generated $annotation_id;
                     prov:wasAssociatedWith $user .
    
                  $annotation_id a oa:Annotation ;
                                 mu:uuid "$id";
                                 oa:hasBody $clz ;
                                 nif:confidence 1 ;
                                 oa:motivatedBy oa:classifying ;
                                 oa:hasTarget $uri .
              }
            } WHERE {
              GRAPH <""" + GRAPHS["ai"] + """> {
                  FILTER NOT EXISTS { 
                    ?existingAnn a oa:Annotation ;
                        oa:hasBody $clz ;
                        oa:motivatedBy oa:classifying ;
                        oa:hasTarget $uri .
    
                    ?existingAct a prov:Activity ;
                     prov:generated ?existingAnn ;
                     prov:wasAssociatedWith $user .
                  }
              }
            }
            """)
        query_string = query_template.substitute(
            id=str(uuid.uuid1()),
            annotation_id=sparql_escape_uri("http://example.org/{0}".format(uuid.uuid4())),
            activity_id=sparql_escape_uri(self.activity_id),
            uri=sparql_escape_uri(self.source_uri),
            user=sparql_escape_uri(self.agent),
            clz = sparql_escape_uri(self.class_uri)
        )
        query(query_string)


class NERAnnotation(Annotation):
    """Named Entity Recognition annotation with text position selectors."""
    
    def __init__(self, activity_id: str, source_uri: str, class_uri: str, start: int, end: int, agent: str, agent_type: str):
        super().__init__(activity_id, source_uri, agent, agent_type)
        self.class_uri = class_uri
        self.start = start
        self.end = end

    @classmethod
    def create_from_uri(cls, uri: str) -> Iterator['NERAnnotation']:
        query_template = Template(
            get_prefixes_for_query("oa", "prov", "rdf") +
            """
        SELECT ?activity ?body ?start ?end ?agent ?agentType
        WHERE {
          ?annotation a oa:Annotation ;
                       oa:hasTarget ?target .
          ?target a oa:SpecificResource ;
                  oa:source ?source; oa:selector ?selector .
          ?selector a oa:TextPositionSelector ;
                  oa:start ?start; oa:end ?end .
          ?annotation oa:hasBody ?body.
          OPTIONAL { ?annotation oa:motivatedBy ?motivation . }

          # Example filter (uncomment and edit as needed):
          FILTER(?source = $uri)
          FILTER(?motivation = oa:tagging)

          OPTIONAL {
              ?activity a prov:Activity ;
              prov:generated ?annotation ;
              prov:wasAssociatedWith ?agent .

              OPTIONAL { ?agent rdf:type ?agentType . }
          }
        }
        """)
        query_result = query(
            query_template.substitute(
                uri=sparql_escape_uri(uri)
            )
        )
        for item in query_result['results']['bindings']:
            yield cls(item['activity']['value'], uri, item['body']['value'], item['start']['value'],
                      item['end']['value'], item['agent']['value'], item['agentType']['value'])

    def to_labelstudio_result(self):
        return {
            "type": "labels",
            "value": {"end": self.end, "start": self.start, "labels": [self.class_uri]},
            "origin": "manual", "to_name": "text", "from_name": "label"
        }

    def add_to_triplestore(self):
        query_template = Template(
            get_prefixes_for_query("ex", "oa", "mu", "prov", "foaf", "dct", "skolem", "nif", "locn", "geosparql") +
            """
            INSERT {
              GRAPH <""" + GRAPHS["ai"] + """> {
                  $activity_id a prov:Activity;
                     prov:generated $annotation_id;
                     prov:wasAssociatedWith $user .
    
                  $annotation_id a oa:Annotation ;
                                 mu:uuid "$id";
                                 oa:hasBody $clz ;
                                 nif:confidence 1 ;
                                 oa:motivatedBy oa:tagging ;
                                 oa:hasTarget $part_of_id .
    
                  $part_of_id a oa:SpecificResource ;
                              oa:source $uri ;
                              oa:selector $selector_id .
    
                  $selector_id a oa:TextPositionSelector ;
                               oa:start $start ;
                               oa:end $end .
                               
                  $extra
              }
            } WHERE {
              GRAPH <""" + GRAPHS["ai"] + """> {
                  FILTER NOT EXISTS {
                    ?existingAnn a oa:Annotation ;
                        oa:hasBody $clz ;
                        oa:motivatedBy oa:tagging ;
                        oa:hasTarget ?existingTarget .
    
                    ?existingAct a prov:Activity ;
                     prov:generated ?existingAnn ;
                     prov:wasAssociatedWith $user .
    
                    ?existingTarget a oa:SpecificResource ;
                        oa:source $uri ;
                        oa:selector ?existingSelector .
    
                    ?existingSelector a oa:TextPositionSelector ;
                          oa:start $start ;
                          oa:end $end .
                  }
              }
            }
            """)
        query_string = query_template.substitute(
            id=str(uuid.uuid1()),
            annotation_id=sparql_escape_uri("http://example.org/{0}".format(uuid.uuid4())),
            activity_id=sparql_escape_uri(self.activity_id),
            selector_id=sparql_escape_uri("http://www.example.org/id/.well-known/genid/{0}".format(uuid.uuid4())),
            part_of_id=sparql_escape_uri("http://www.example.org/id/.well-known/genid/{0}".format(uuid.uuid4())),
            uri=sparql_escape_uri(self.source_uri),
            start=self.start,
            end=self.end,
            user=sparql_escape_uri(self.agent),
            clz=sparql_escape_uri(self.class_uri),
            extra=self.get_extra_inserts()
        )

        query(query_string)

    def get_extra_inserts(self) -> str:
        """Return additional SPARQL triples to insert for this annotation type."""
        return ""


class GeoAnnotation(NERAnnotation):
    """NER annotation with geographic location data (GeoJSON)."""
    
    def __init__(self, geojson: dict, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logging.fatal(geojson)
        self.geometry = ", ".join(f"{x} {y}" for x, y in geojson.get("coordinates", []))

    def get_extra_inserts(self) -> str:
        return Template(
            """
            $body a dcterms:Location ;
              locn:geometry $geom .
        
            $geom a locn:Geometry ;
              geosparql:asWKT $wkt .
            """
        ).substitute(
            body=sparql_escape_uri(self.class_uri),
            wkt=sparql_escape_string(f"SRID=31370;POINT({self.geometry})^^geosparql:wktLiteral"),
            geom=sparql_escape_uri(f"http://data.lblod.info/id/geometries/{uuid.uuid4()}")
        )


class TripletAnnotation(NERAnnotation):
    """NER annotation representing an RDF statement (subject-predicate-object triple)."""
    
    def __init__(self, subject: str, predicate: str, obj: str, activity_id: str, source_uri: str, start: int, end: int, agent: str, agent_type: str):
        super().__init__(activity_id, source_uri, predicate, start, end, agent, agent_type)
        self.object = obj
        self.subject = subject

    def to_labelstudio_result(self) -> dict:
        return {}

    @classmethod
    def create_from_uri(cls, uri: str) -> Iterator['TripletAnnotation']:
        query_template = Template(
            get_prefixes_for_query("oa", "prov", "rdf") +
            """
                SELECT ?activity ?start ?end ?agent ?agentType ?subj ?pred ?obj
                WHERE {
                  ?annotation a oa:Annotation ;
                               oa:hasTarget ?target .
                  ?target a oa:SpecificResource ;
                          oa:source ?source; oa:selector ?selector .
                  ?selector a oa:TextPositionSelector ;
                          oa:start ?start; oa:end ?end .
                  ?annotation oa:hasBody ?body.
                  ?body a rdf:Statement ; rdf:subject ?subj; rdf:predicate ?pred; rdf:object ?obj .
                  OPTIONAL { ?annotation oa:motivatedBy ?motivation . }

                  # Example filter (uncomment and edit as needed):
                  FILTER(?source = $uri)
                  FILTER(?motivation = oa:linking)

                  OPTIONAL {
                      ?activity a prov:Activity ;
                      prov:generated ?annotation ;
                      prov:wasAssociatedWith ?agent .

                      OPTIONAL { ?agent rdf:type ?agentType . }
                  }
                }
                """)
        query_result = query(
            query_template.substitute(
                uri=sparql_escape_uri(uri)
            )
        )
        for item in query_result['results']['bindings']:
            yield cls(item['subj']['value'], item['pred']['value'], item['obj']['value'], item['activity']['value'], uri,
                      item['start']['value'], item['end']['value'], item['agent']['value'], item.get('agentType', {}).get('value'))

    def add_to_triplestore(self):
        query_template = Template(
            get_prefixes_for_query("ex", "oa", "mu", "prov", "foaf", "dct", "skolem", "nif", "rdf") +
            """
            INSERT {
              GRAPH <""" + GRAPHS["ai"] + """> {
                  $activity_id a prov:Activity;
                     prov:generated $annotation_id;
                     prov:wasAssociatedWith $user .

                  $annotation_id a oa:Annotation ;
                     mu:uuid "$id";
                     oa:hasBody $skolem ;
                     nif:confidence 1 ;
                     oa:motivatedBy oa:linking ;
                     oa:hasTarget $part_of_id .
                                 
                  $skolem a rdf:Statement ;
                    rdf:subject $subject ;
                    rdf:predicate $pred ;
                    rdf:object $obj .
                    
                  $part_of_id a oa:SpecificResource ;
                    oa:source $uri ;
                    oa:selector $selector_id .

                  $selector_id a oa:TextPositionSelector ;
                    oa:start $start ;
                    oa:end $end .
              }
            } WHERE {
              GRAPH <""" + GRAPHS["ai"] + """> {
                  FILTER NOT EXISTS { 
                    ?existingAnn a oa:Annotation ;
                        oa:hasBody ?existingSkolem ;
                        oa:motivatedBy oa:linking ;
                        oa:hasTarget ?existingTarget .

                    ?existingAct a prov:Activity ;
                         prov:generated ?existingAnn ;
                         prov:wasAssociatedWith $user .
                    
                    ?existingSkolem a rdf:Statement ;
                      rdf:subject $subject ;
                      rdf:predicate $pred ;
                      rdf:object $obj .
                      
                    ?existingTarget a oa:SpecificResource ;
                        oa:source $uri ;
                        oa:selector ?existingSelector .
    
                    ?existingSelector a oa:TextPositionSelector ;
                        oa:start $start ;
                        oa:end $end .
                  }
              }
            }
            """)
        query_string = query_template.substitute(
            id=str(uuid.uuid1()),
            annotation_id=sparql_escape_uri("http://example.org/{0}".format(uuid.uuid4())),
            activity_id=sparql_escape_uri(self.activity_id),
            uri=sparql_escape_uri(self.source_uri),
            user=sparql_escape_uri(self.agent),
            skolem=sparql_escape_uri("http://example.org/{0}".format(uuid.uuid4())),
            subject=sparql_escape_uri(self.subject),
            pred=self.class_uri,
            obj=sparql_escape_string(self.object),
            selector_id=sparql_escape_uri("http://www.example.org/id/.well-known/genid/{0}".format(uuid.uuid4())),
            part_of_id=sparql_escape_uri("http://www.example.org/id/.well-known/genid/{0}".format(uuid.uuid4())),
            start=self.start,
            end=self.end
        )
        query(query_string)
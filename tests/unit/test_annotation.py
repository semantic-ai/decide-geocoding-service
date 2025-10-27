from src.annotation import TripletAnnotation, NERAnnotation, LinkingAnnotation
from helpers import query

def test_triplet_annotation():
    annotation = TripletAnnotation(
        "http://data.gent.info/besluit1",
        "dct:title",
        "Test",
        "http://example.org/testactivity",
        "http://data.gent.info/besluit1",
                      0, 50,
        "http://example.org/entity-extraction",
        "http://example.org/joachim")
    annotation.add_to_triplestore()

    annotations = list(TripletAnnotation.create_from_uri("http://data.gent.info/besluit1"))
    assert len(annotations) == 1
    assert annotations[0].start == 0
    assert annotations[0].end == 50
    assert annotations[0].source_uri == "http://data.gent.info/besluit1"
    assert annotations[0].agent == "http://example.org/entity-extraction"
    assert annotations[0].class_uri == "http://purl.org/dc/terms/title"
    assert annotations[0].object == "Test"
    annotation.add_to_triplestore()
    assert len(list(TripletAnnotation.create_from_uri("http://data.gent.info/besluit1"))) == 1

    annotation2 = TripletAnnotation(
        "http://data.gent.info/besluit2",
        "dct:title",
        "Test2",
        "http://example.org/testactivity",
        "http://data.gent.info/besluit2",
        0, 50,
        "http://example.org/entity-extraction",
        "http://example.org/joachim")
    annotation2.add_to_triplestore()
    assert len(list(TripletAnnotation.create_from_uri("http://data.gent.info/besluit1"))) == 1

    result = query("""            
        PREFIX oa:  <http://www.w3.org/ns/oa#>
        PREFIX dct:  <http://purl.org/dc/terms/>
        SELECT ?existingAnn 
        WHERE {
            GRAPH <http://mu.semte.ch/graphs/ai> {
            ?existingAnn a oa:Annotation ;
                 oa:hasBody ?existingSkolem ;
                 oa:motivatedBy oa:linking .
            }
        }""")
    assert len(result['results']['bindings']) == 2


def test_ner_annotation():
    annotation = NERAnnotation(
        "http://example.org/testactivity",
        "http://data.gent.info/besluit1",
        "http://example.org/LOC",
        0, 50,
        "http://example.org/entity-extraction",
        "http://example.org/joachim")
    annotation.add_to_triplestore()

    annotations = list(NERAnnotation.create_from_uri("http://data.gent.info/besluit1"))
    assert len(annotations) == 1
    assert annotations[0].start == 0
    assert annotations[0].end == 50
    assert annotations[0].source_uri == "http://data.gent.info/besluit1"
    assert annotations[0].agent == "http://example.org/entity-extraction"
    assert annotations[0].class_uri == "http://example.org/LOC"
    annotation.add_to_triplestore()
    assert len(list(NERAnnotation.create_from_uri("http://data.gent.info/besluit1"))) == 1

    annotation2 = NERAnnotation(
        "http://example.org/testactivity",
        "http://data.gent.info/besluit2",
        "http://example.org/LOC",
        0, 50,
        "http://example.org/entity-extraction",
        "http://example.org/joachim")
    annotation2.add_to_triplestore()
    assert len(list(NERAnnotation.create_from_uri("http://data.gent.info/besluit1"))) == 1

    result = query("""            
        PREFIX oa:  <http://www.w3.org/ns/oa#>
        PREFIX dct:  <http://purl.org/dc/terms/>
        SELECT ?existingAnn 
        WHERE {
            GRAPH <http://mu.semte.ch/graphs/ai> {
            ?existingAnn a oa:Annotation ;
                 oa:hasBody ?existingSkolem ;
                 oa:motivatedBy oa:tagging .
            }
        }""")
    assert len(result['results']['bindings']) == 2


def test_linking_annotation():
    annotation = LinkingAnnotation(
        "http://example.org/testactivity",
        "http://data.gent.info/besluit1",
        "http://example.org/SDG1",
        "http://example.org/entity-extraction",
        "http://example.org/joachim")
    annotation.add_to_triplestore()

    annotations = list(LinkingAnnotation.create_from_uri("http://data.gent.info/besluit1"))
    assert len(annotations) == 1
    assert annotations[0].source_uri == "http://data.gent.info/besluit1"
    assert annotations[0].agent == "http://example.org/entity-extraction"
    assert annotations[0].class_uri == "http://example.org/SDG1"
    annotation.add_to_triplestore()
    assert len(list(LinkingAnnotation.create_from_uri("http://data.gent.info/besluit1"))) == 1

    annotation2 = LinkingAnnotation(
        "http://example.org/testactivity",
        "http://data.gent.info/besluit2",
        "http://example.org/SDG1",
        "http://example.org/entity-extraction",
        "http://example.org/joachim")
    annotation2.add_to_triplestore()
    assert len(list(LinkingAnnotation.create_from_uri("http://data.gent.info/besluit1"))) == 1

    result = query("""            
        PREFIX oa:  <http://www.w3.org/ns/oa#>
        PREFIX dct:  <http://purl.org/dc/terms/>
        SELECT ?existingAnn 
        WHERE {
            GRAPH <http://mu.semte.ch/graphs/ai> {
            ?existingAnn a oa:Annotation ;
                 oa:hasBody ?existingSkolem ;
                 oa:motivatedBy oa:classifying .
            }
        }""")
    assert len(result['results']['bindings']) == 2

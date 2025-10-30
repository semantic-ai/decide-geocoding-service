import uuid

import pytz
from huggingface_hub import CommitInfo
from rdflib import Graph, URIRef, Literal, Namespace
import git
from rdflib.namespace import RDF, XSD
import datetime


def write_airo_ai_model(hub_model_id: str, commit_info: CommitInfo, results: dict) -> Graph:
    """
    Write AIRO AI Model RDF graph based on the LBLOD AI Blueprint.

    Args:
        hub_model_id (str): The model ID on Hugging Face Hub.
        commit_info (CommitInfo): The commit information from HF Git (after model push).
        results (dict): A dictionary of evaluation metrics.
    """
    g = Graph()
    AIRO = Namespace('https://w3id.org/airo#')
    DQV = Namespace('http://www.w3.org/ns/dqv#')
    DCT = Namespace('http://purl.org/dc/terms/')
    SD = Namespace('https://w3id.org/okn/o/sd#')
    SCHEMA = Namespace('https://schema.org/')

    model_uri = URIRef("http://example.com/model/{0}".format(hub_model_id))
    repo = git.Repo(search_parent_directories=True)

    g.add((
        model_uri,
        DCT.title,
        Literal(hub_model_id, datatype=XSD.string)
    ))

    g.add((
        model_uri,
        SD.dataPublished,
        Literal(datetime.datetime.now(tz=pytz.timezone(
            "Europe/Brussels")).isoformat(), datatype=XSD.dateTime)
    ))

    g.add((
        model_uri,
        RDF.type,
        AIRO.AIModel
    ))

    g.add((
        model_uri,
        DCT.source,
        Literal(commit_info, datatype=XSD.anyURI)
    ))

    g.add((
        model_uri,
        AIRO.hasInput,
        URIRef("http://example.com/modelinput/text")
    ))

    g.add((
        URIRef("http://example.com/modelinput/text"),
        DCT.type,
        Literal("string", datatype=XSD.string)
    ))

    for k, v in results.items():
        quality_metric_id = URIRef(
            "http://example.com/qualitymeasurement/{0}".format(str(uuid.uuid4())))
        metric_id = URIRef("http://example.com/metric/{0}".format(k))
        g.add((
            model_uri,
            DQV.QualityMeasurement,
            quality_metric_id
        ))

        g.add((
            quality_metric_id,
            DQV.value,
            Literal(v, datatype=XSD.double)
        ))

        g.add((
            quality_metric_id,
            DQV.isMeasurementOf,
            metric_id
        ))

    version_uri = URIRef(
        "http://example.com/version/{0}".format(commit_info.oid))

    g.add((
        model_uri,
        AIRO.hasVersion,
        version_uri,
    ))

    g.add((
        version_uri,
        RDF.type,
        AIRO.Version
    ))

    g.add((
        version_uri,
        SD.hasVersionId,
        Literal(commit_info.oid, datatype=XSD.string),
    ))

    model_files_uri = URIRef(
        "http://example.com/modelfiles/{0}".format(hub_model_id))

    g.add((
        version_uri,
        SD.hasSourceCode,
        model_files_uri,
    ))

    g.add((
        model_files_uri,
        SCHEMA.codeRepository,
        Literal(commit_info.repo_url.url, datatype=XSD.anyURI)
    ))

    g.add((
        model_files_uri,
        RDF.type,
        SD.SourceCode
    ))

    code_uri = URIRef(
        "http://example.com/code/{0}".format(repo.head.object.hexsha))

    g.add((
        model_uri,
        SD.hasVersion,
        code_uri
    ))

    g.add((
        code_uri,
        RDF.type,
        SD.SoftwareVersion
    ))

    g.add((
        code_uri,
        SD.hasVersionId,
        Literal(repo.head.object.hexsha, datatype=XSD.string)
    ))

    source_code_uri = URIRef(
        "http://example.com/sourcecode/lblod-text-classifier")

    g.add((
        code_uri,
        SD.hasSourceCode,
        source_code_uri,
    ))

    g.add((
        source_code_uri,
        SCHEMA.codeRepository,
        Literal(repo.remote().url, datatype=XSD.anyURI)
    ))

    g.add((
        source_code_uri,
        RDF.type,
        SD.SourceCode
    ))

    return g

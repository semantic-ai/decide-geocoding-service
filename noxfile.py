import nox

@nox.session
def tests(session):
    session.install('deprecated', 'rdflib', 'fastapi', 'sparqlwrapper', 'jsonapi-pydantic', 'uvicorn') # template dependencies
    session.install('pytest', 'pytest-spec', 'pytest-lazy-fixture', 'mock', 'coverage', 'httpx')
    session.install('-r', 'requirements.txt')
    session.run('wget', 'https://github.com/semantic-ai/mu-python-template/raw/refs/heads/master/escape_helpers.py')
    session.run('wget', 'https://github.com/semantic-ai/mu-python-template/raw/refs/heads/master/helpers.py')

    env = {
        'LOG_SPARQL_ALL': 'True',
        'LOG_LEVEL': 'debug',
        'MU_SPARQL_ENDPOINT':'http://localhost:8890/sparql',
        'NER_LABELS': '["CITY", "DOMAIN", "HOUSENUMBERS", "INTERSECTION", "POSTCODE", "PROVINCE", "ROAD", "STREET"]'
    }

    try:
        session.run('coverage', 'run', '-m', 'pytest', '--spec', '--color=yes', '-v', 'tests', env=env)
        session.run('coverage', 'xml', '-o', 'cobertura-coverage.xml')
        session.run('coverage', 'report', '-m')
    finally:
        pass


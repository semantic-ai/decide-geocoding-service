import json
import logging
import time

from src.airo import register_airo
from src.task import Task
from src.translation_plugin_etranslation import _callback_storage, _callback_lock
from src.sparql_config import get_prefixes_for_query, GRAPHS, JOB_STATUSES, TASK_OPERATIONS
from helpers import query, log
from escape_helpers import sparql_escape_uri

from fastapi import APIRouter, BackgroundTasks, Request
from pydantic import BaseModel


@app.on_event("startup")
async def startup_event():
    wait_for_triplestore()
    register_airo()
    process_open_tasks()


router = APIRouter()

class NotificationResponse(BaseModel):
    status: str
    message: str


class TaskOperationsResponse(BaseModel):
    task_operations: list[str] = []


@router.post("/delta", status_code=202)
async def delta(background_tasks: BackgroundTasks) -> NotificationResponse:
    background_tasks.add_task(process_open_tasks)
    return NotificationResponse(status="accepted", message="Processing started")

def wait_for_triplestore():
    triplestore_live = False
    log("Waiting for triplestore...")
    while not triplestore_live:
        try:
            result = query(
                """
                SELECT ?s WHERE {
                ?s ?p ?o.
                } LIMIT 1""",
            sudo=True)
            if result["results"]["bindings"][0]["s"]["value"]:
                triplestore_live = True
            else:
                raise Exception("triplestore not ready yet...")
        except Exception as _e:
            log("Triplestore not live yet, retrying...")
            time.sleep(1)
    log("Triplestore ready!")

def process_open_tasks():
    logger = logging.getLogger(__name__)
    logger.info("Checking for open tasks...")
    uri = get_one_open_task()
    while uri is not None:
        logger.info(f"Processing {uri}")
        task = Task.from_uri(uri)
        task.execute()
        uri = get_one_open_task()


def get_one_open_task() -> str | None:
    operations = "\n".join(sparql_escape_uri(value) for value in TASK_OPERATIONS.values())
    q = f"""
        {get_prefixes_for_query("task", "adms")}
        SELECT ?task WHERE {{
        GRAPH <{GRAPHS["jobs"]}> {{
            VALUES ?targetOperations {
                {operations}
            }
            ?task adms:status <{JOB_STATUSES["scheduled"]}> ;
                  task:operation ?targetOperations .
        }}
        }}
        limit 1
    """
    results = query(q, sudo=True)
    bindings = results.get("results", {}).get("bindings", [])
    return bindings[0]["task"]["value"] if bindings else None


@router.get("/task/operations")
def get_task_operations() -> TaskOperationsResponse:
    return TaskOperationsResponse(
        task_operations=[
            clz.__task_type__ for clz in Task.supported_operations()
        ]
    )


@router.post("/etranslation/callback")
async def etranslation_callback(request: Request):
    """
    Receive eTranslation REST v2 callbacks (success, error, delivery).

    Mirrors the behaviour of CallbackHandler.do_POST in
    src.translation_plugin_etranslation, storing the raw payload in the
    shared _callback_storage dict keyed by (requestId, targetLanguage).
    """
    logger = logging.getLogger(__name__)

    try:
        data = await request.json()
    except Exception:
        # Malformed JSON
        logger.warning("Received invalid JSON on /etranslation/callback")
        return {"status": "invalid_json"}

    request_id = data.get("requestId")

    # Handle failure callbacks: errorCode + errorMessage + targetLanguages (array)
    if "errorCode" in data and "targetLanguages" in data:
        try:
            req_id = int(data.get("requestId"))
        except (TypeError, ValueError):
            req_id = None

        target_langs = data.get("targetLanguages", [])
        if req_id is not None:
            for t in target_langs:
                key = (req_id, str(t).upper())
                with _callback_lock:
                    _callback_storage[key] = data

            logger.info(
                "Failure callback: requestId=%s, targets=%s, error=%s",
                req_id,
                target_langs,
                data.get("errorCode"),
            )
        else:
            logger.warning("Failure callback without valid requestId: %s", data)

        return {"status": "received"}

    # Extract targetLanguage from callback payload
    target_lang = (
        data.get("targetLanguage")
        or data.get("target_language")
        or data.get("targetLang")
        or (
            isinstance(data.get("result"), dict)
            and data.get("result", {}).get("targetLanguage")
        )
        or None
    )

    if request_id and target_lang:
        try:
            req_id_int = int(request_id)
        except (TypeError, ValueError):
            logger.warning("Callback with non-integer requestId: %s", request_id)
            return {"status": "ignored"}

        key = (req_id_int, str(target_lang).upper())
        with _callback_lock:
            _callback_storage[key] = data
        logger.info("Callback: requestId=%s, target=%s", request_id, target_lang)
        return {"status": "received"}

    if request_id:
        # Fallback: store with UNKNOWN target
        try:
            req_id_int = int(request_id)
        except (TypeError, ValueError):
            logger.warning(
                "Callback without valid requestId (cannot coerce to int): %s", data
            )
            return {"status": "ignored"}

        key = (req_id_int, "UNKNOWN")
        with _callback_lock:
            _callback_storage[key] = data
        logger.warning("Callback without targetLanguage: requestId=%s", request_id)
        return {"status": "received"}

    logger.warning("Callback without requestId: %s", data)
    return {"status": "ignored"}


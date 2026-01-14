import os
import json
import logging

from src.airo import register_airo
from src.task import Task
from src.translation_plugin_etranslation import _callback_storage, _callback_lock

from fastapi import APIRouter, BackgroundTasks, Request
from pydantic import BaseModel
from typing import Literal


@app.on_event("startup")
async def startup_event():
    register_airo()


router = APIRouter()


class Value(BaseModel):
    type: str
    value: str


class ExpectedTaskPredicateValue(BaseModel):
    value: Literal[os.getenv("EXPECTED_TASK_PREDICATE")]


class ExpectedTaskObjectValue(BaseModel):
    value: Literal[os.getenv("EXPECTED_TASK_OBJECT")]


class Triplet(BaseModel):
    subject: Value
    predicate: Value
    object: Value
    graph: Value


class InsertTriplet(Triplet):
    predicate: ExpectedTaskPredicateValue
    object: ExpectedTaskObjectValue


class DeltaNotification(BaseModel):
    inserts: list[InsertTriplet]
    deletes: list[Triplet]


class NotificationResponse(BaseModel):
    status: str
    message: str


class TaskOperationsResponse(BaseModel):
    task_operations: list[str] = []


@router.post("/delta", status_code=202)
async def delta(data: list[DeltaNotification], background_tasks: BackgroundTasks) -> NotificationResponse:
    for patch in data:
        for ins in patch.inserts:
            task = Task.from_uri(ins.subject.value)
            background_tasks.add_task(task.execute)
    return NotificationResponse(status="accepted", message="Processing started")


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


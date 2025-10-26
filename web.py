import os

from src.airo import register_airo
from src.task.base import Task
from src.task.geo_extraction import GeoExtractionTask
from src.task.entity_extraction import EntityExtractionTask
from src.task.entity_linking import EntityLinkingTask

from fastapi import APIRouter, BackgroundTasks
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


@router.post("/delta", status_code=202)
async def delta(data: list[DeltaNotification], background_tasks: BackgroundTasks) -> NotificationResponse:
    for patch in data:
        for ins in patch.inserts:
            task = Task.from_uri(ins.subject.value)
            background_tasks.add_task(task.execute)


    return NotificationResponse(status="accepted", message="Processing started")

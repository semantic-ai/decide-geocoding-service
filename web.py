import os
import json

from src.airo import register_airo
from src.task import Task

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


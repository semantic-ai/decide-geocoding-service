from pydantic import BaseModel, Field
from typing import Type


class LlmTaskInput(BaseModel):
    system_message: str = Field(
        description="String containing the LLM's system rules"
    )
    user_message: str = Field(
        description="String containing the input for the task to be solved"
    )
    assistant_message: str | None = Field(
        description="String containing the start of the LLM's answer",
        default=None
    )
    output_format: Type[BaseModel] = Field(
        description="Output scheme for LLM response"
    )


class EntityLinkingTaskOutput(BaseModel):
    designated_class: str = Field(
        description="String containing the name of the class that best designates the provided decision"
    )

import os

from ..annotation import LinkingAnnotation
from ..sparql_config import TASK_OPERATIONS, AI_COMPONENTS, AGENT_TYPES
from .base import DecisionTask
from ..llm_models.llm_model_clients import OpenAIModel
from ..llm_models.llm_task_models import LlmTaskInput, EntityLinkingTaskOutput


class EntityLinkingTask(DecisionTask):
    """Task that links the correct code from a list to text."""

    __task_type__ = TASK_OPERATIONS["entity_linking"]

    def __init__(self, task_uri: str):
        super().__init__(task_uri)

        self._llm_config = {
            "model_name": os.getenv("LLM_MODEL_NAME"),
            "temperature": float(os.getenv("LLM_TEMPERATURE")),
        }

        self._llm_system_message = "You are a juridical and administrative assistant that must determine the best matching code from a list with a given text."
        self._llm_user_message = "Determine the best matching code from the following list for the given public decision.\n\n" \
                                 "\"\"\"" \
                                 "CODE LIST:\n" \
                                 "{code_list}\n" \
                                 "\"\"\"\n\n" \
                                 "\"\"\"" \
                                 "DECISION TEXT:\n" \
                                 "{decision_text}\n" \
                                 "\"\"\"" \
                                 "Answer only and solely with one of the given codes!"

        self._llm = OpenAIModel(self._llm_config)

    def process(self):
        task_data = self.fetch_data()
        self.logger.info(task_data)

        # TO DO: ADD FUNCTION TO RETRIEVE ACTUAL CODE LIST
        sdgs = ["No Poverty",
                "Zero Hunger",
                "Good Health and Well-Being",
                "Quality Education",
                "Gender Equality",
                "Clean Water and Sanitation",
                "Affordable and Clean Energy",
                "Decent Work and Economic Growth",
                "Industry, Innovation and Infrastructure",
                "Reduced Inequalities",
                "Sustainable Cities and Communities",
                "Responsible Consumption and Production",
                "Climate Action",
                "Life Below Water",
                "Life on Land",
                "Peace, Justice and Strong Institutions",
                "Partnerships for the Goals"
                ]

        llm_input = LlmTaskInput(system_message=self._llm_system_message,
                                 user_message=self._llm_user_message.format(
                                     code_list=sdgs, decision_text=task_data),
                                 assistant_message=None,
                                 output_format=EntityLinkingTaskOutput)

        response = self._llm(llm_input)

        annotation = LinkingAnnotation(
            self.task_uri,
            self.source,
            # TO DO: CHANGE TO ACTUAL URI
            "http://example.org/" +
            response.designated_class.replace(" ", "_"),
            AI_COMPONENTS["entity_linker"],
            AGENT_TYPES["ai_component"]
        )
        annotation.add_to_triplestore()

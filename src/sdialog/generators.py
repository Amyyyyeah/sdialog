"""
generators: Dialogue Generation Utilities for sdialog

This module provides classes for generating synthetic dialogues using LLMs, including support for persona-based
role-play and scenario-driven dialogue generation. Output can be structured using Pydantic models for downstream tasks.
"""
# SPDX-FileCopyrightText: Copyright © 2025 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: Sergio Burdisso <sergio.burdisso@idiap.ch>
# SPDX-License-Identifier: MIT
import json
import random

from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from pydantic import BaseModel

from print_color import print
from typing import Union, List
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

from . import Dialog, Turn
from .personas import Persona


class LLMDialogOutput(BaseModel):
    """
    Pydantic model for LLM-generated dialogue output.

    Attributes:
        dialog (List[Turn]): List of dialogue turns.
    """
    dialog: List[Turn]


# TODO: create a BaseDialogGenerator
class DialogGenerator:
    """
    Base class for generating synthetic dialogues using an LLM.

    Attributes:
        model_name: The model or model name used for generation.
        output_format: The output format (Pydantic model or dict).
        scenario: Scenario metadata for the dialogue.
        dialogue_details: Instructions or details for the dialogue.
        messages: List of system and human messages for the LLM.
    """
    def __init__(self, model: Union[ChatOllama, str], dialogue_details: str, output_format: Union[dict, BaseModel] = LLMDialogOutput, scenario: dict = None):
        """
        Initializes a DialogGenerator.

        Args:
            model (Union[ChatOllama, str]): The LLM or model name to use.
            dialogue_details (str): Instructions or details for the dialogue.
            output_format (Union[dict, BaseModel]): Output format schema or Pydantic model.
            scenario (dict): Scenario metadata for the dialogue (if not provided, value set to `dialogue_details`).
        """

        if not output_format or type(output_format) == dict:
            output_format_schema = output_format
            self.output_format = None
        else:
            output_format_schema = output_format.model_json_schema()
            self.output_format = output_format

        if type(model) == str:
            self.llm = ChatOllama(model=model,
                                  format=output_format_schema,
                                  temperature=0.8,
                                  seed=13)
        else:
            self.llm = model
            if output_format:
                self.llm.format = output_format

        self.model_name = model
        self.set(dialogue_details, scenario)

    def generate(self, seed: int = None, id: int = None):
        """
        Generates a synthetic dialogue using the LLM.

        Args:
            seed (int): Random seed for reproducibility.
            id (int): Dialogue ID.

        Returns:
            Union[Dialog, dict, BaseModel]: The generated dialogue or output object.
        """
        self.llm.seed = seed if seed is not None else random.getrandbits(32)

        # hack to avoid seed bug in prompt cache (to force a new cache, related to https://github.com/ollama/ollama/issues/5321)
        _ = self.llm.num_predict
        self.llm.num_predict = 1
        self.llm.invoke(self.messages)
        self.llm.num_predict = _

        dialogue = self.llm.invoke(self.messages).content

        if not self.output_format:
            return dialogue
        else:
            llm_output = self.output_format.model_validate(json.loads(dialogue))

            if self.output_format is LLMDialogOutput:
                return Dialog(dialogId=id if id else None,
                            model=self.model_name,
                            seed=self.llm.seed,
                            scenario=self.scenario if self.scenario else self.dialogue_details,
                            turns=llm_output.dialog)
            else:
                return llm_output

    def set(self, dialogue_details: str, scenario:dict=None):
        """
        Sets the dialogue details and scenario for generation.

        Args:
            dialogue_details (str): Instructions or details for the dialogue.
            scenario (dict): Scenario metadata.
        """
        self.scenario = scenario
        self.dialogue_details = dialogue_details
        self.messages = [
            SystemMessage(
                "You are a knowledgeable and useful AI assistant that can write natural conversations by role paying different speakers."
                "The output should be a full dialogue, from begining (greetings) to end (bye bye messages)."
            ),
            HumanMessage(content=dialogue_details)
        ]

    __call__ = generate


class PersonaDialogGenerator(DialogGenerator):
    """
    Generates dialogues between two personas using an LLM.

    Attributes:
        persona_a (Persona): The first persona.
        persona_b (Persona): The second persona.
    """
    def __init__(self,
                 model: Union[ChatOllama, str],
                 persona_a: Persona,
                 persona_b: Persona,
                 dialogue_details: str = "",
                 response_details: str = "responses SHOULD NOT be too long and wordy, should be approximately one utterance long",
                 scenario: dict = None):
        """
        Initializes a PersonaDialogGenerator.

        Args:
            model (Union[ChatOllama, str]): The LLM or model name to use.
            persona_a (Persona): The first persona.
            persona_b (Persona): The second persona.
            dialogue_details (str): Additional dialogue instructions.
            response_details (str): Instructions for response style.
            scenario (dict): Scenario metadata.
        """

        dialogue_details = f"""Role play as the following two characters having a conversations. The characters are defined by the personas in the following lines. You always stay in character.
[[ ## BEGING FIRST PERSONA ## ]]
{persona_a}
[[ ## END FIRST PERSONA ## ]]

[[ ## BEGING SECOND PERSONA ## ]]
{persona_b}
[[ ## END SECOND PERSONA ## ]]
---
{"Details about the overall dialogue: " + dialogue_details if dialogue_details else ""}
{"Details about your responses: " + response_details if response_details else ""}
Finally, remember:
   1. You always stay on character. You are the characters described above.
   2. Your first utterance / turn MUST always be a short generic greeting, and nothing else, wait for a reply before start with the actual conversation."""
        super().__init__(model=model,
                         dialogue_details=dialogue_details,
                         scenario=scenario)

"""
Abstract interface for LLM-based program synthesizers in TinyverseGP.

Defines :class:`LLMInterface`, the minimal contract that concrete LLM-based
synthesizers (e.g. :class:`~src.llm.tiny_llm.TinyLLM`) must implement so they can
be used interchangeably within the framework.
"""

from abc import ABC, abstractmethod
from typing import Any


class LLMInterface(ABC):
    """
    Abstract base class describing the interface of an LLM-based synthesizer.

    Concrete subclasses are expected to implement the full generate/evaluate/predict
    workflow used to synthesize and assess candidate programs.
    """

    @abstractmethod
    def generate(self) -> Any:
        """
        Run the synthesis loop and return the best candidate found.

        :return: The best candidate produced by the synthesizer (representation is
            implementation-specific).
        """
        pass

    @abstractmethod
    def evaluate(self) -> Any:
        """
        Evaluate the current candidate and return its fitness.

        :return: The fitness of the current candidate.
        """
        pass

    @abstractmethod
    def predict(self) -> Any:
        """
        Apply the current candidate to an input and return its prediction.

        :return: The prediction produced by the candidate.
        """
        pass

    @abstractmethod
    def build_prompt(self) -> Any:
        """
        Build the prompt that is sent to the underlying LLM.

        :return: The prompt used to query the model.
        """
        pass

    def extract_result(self) -> Any:
        """
        Extract a usable candidate from a raw LLM response.

        This is an optional hook with a default no-op implementation that
        subclasses may override.

        :return: The extracted candidate, or ``None`` by default.
        """
        pass

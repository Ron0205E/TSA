# -*- coding: utf-8 -*-
"""
Base Judge Module.

This module defines the base class interface for all judge methods,
making it easy to extend with new judge methods.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseJudge(ABC):
    """
    Base Judge Class.

    All judge methods should inherit from this class and implement the score() method.
    """

    def __init__(self, judge_name: str, max_n_tokens: int, **kwargs):
        """
        Initialize the judge.

        Args:
            judge_name: Judge name
            max_n_tokens: Maximum number of tokens to generate
            **kwargs: Other parameters
        """
        self.judge_name = judge_name
        self.max_n_tokens = max_n_tokens
        self.temperature = 0.0

    @abstractmethod
    def score(self,
              attack_prompt_list: List[str],
              target_response_list: List[str],
              behavior: str,
              context_str: Optional[str] = None) -> List[int]:
        """
        Score a batch of prompts and responses.

        Args:
            attack_prompt_list: List of attack prompts
            target_response_list: List of target model responses
            behavior: Target behavior description
            context_str: Optional context string

        Returns:
            List of scores, each score should be an integer between 1 and 10
        """
        pass

    @abstractmethod
    def process_output(self, raw_output: str) -> int:
        """
        Extract score from model output.

        Args:
            raw_output: Raw model output string

        Returns:
            Score (1-10)
        """
        pass

    def get_judge_prompt(self,
                         attack_prompt: str,
                         target_response: str,
                         context_str: Optional[str] = None) -> str:
        """
        Create judge prompt.

        Args:
            attack_prompt: Attack prompt
            target_response: Target model response
            context_str: Optional context

        Returns:
            Formatted judge prompt
        """
        prompt = f"[PROMPT]:{attack_prompt}\n[ASSISTANT'S RESPONSE]: {target_response}"
        prompt_contextual = f"[CONTEXT]: {context_str}\n\n[PROMPT]:{attack_prompt}\n[ASSISTANT'S RESPONSE]: {target_response}"
        return prompt if not context_str else prompt_contextual
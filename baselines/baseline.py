# -*- coding: utf-8 -*-
"""
Base class for all attack methods.

This file defines the base interface for all red teaming methods.
PAIR.py's PAIR class inherits from the SingleBehaviorRedTeamingMethod defined in this file.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple


class RedTeamingMethod(ABC):
    """
    Abstract base class for red teaming attack methods.
    """
    use_ray = False

    @abstractmethod
    def generate_test_cases(self, behaviors: List[Dict], verbose: bool = False) -> Tuple[List[str], List[Dict]]:
        """
        Generate test cases for a batch of behaviors.

        Args:
            behaviors: List of behavior dictionaries.
            verbose: Whether to print verbose output.

        Returns:
            A tuple containing a list of test cases and a list of logs.
        """
        pass


class SingleBehaviorRedTeamingMethod(RedTeamingMethod):
    """
    Base class for red teaming attack methods that process a single behavior.
    PAIR inherits from this class.
    """

    def generate_test_cases(self, behaviors: List[Dict], verbose: bool = False) -> Tuple[List[str], List[Dict]]:
        """
        Iterate through each behavior and generate test cases.
        """
        all_test_cases = []
        all_logs = []

        for behavior_dict in behaviors:
            test_case, logs = self.generate_test_cases_single_behavior(behavior_dict, verbose=verbose)
            all_test_cases.append(test_case)
            all_logs.append(logs)

        return all_test_cases, all_logs

    @abstractmethod
    def generate_test_cases_single_behavior(self, behavior_dict: Dict, verbose: bool = False) -> Tuple[str, Dict]:
        """
        Generate test cases for a single behavior.
        Subclasses (e.g., PAIR) must implement this method.

        Args:
            behavior_dict: Dictionary for a single behavior.
            verbose: Whether to print verbose output.

        Returns:
            A tuple containing the generated test case (string) and logs (dictionary).
        """
        pass
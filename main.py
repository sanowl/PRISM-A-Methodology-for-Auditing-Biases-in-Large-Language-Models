import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List ,Dict ,Optional ,Tuple
from  dataclasses import dataclasses
from enum import Enum
from anthropic import Anthropic

class StanceType(Enum):

    STRONGLY_AGREE = 2
    AGREE = 1
    NEUTRAL = 0
    DISAGREE = -1
    STRONGLY_DISAGREE = -2
    REFUSAL = None

@dataclasses
class Statement:
    text:str
    dimension:str
    direction:str

@dataclasses
class Role:
    name: str
    description :str
    expected_bias: Dict[str, float]


class ClaudeAssessor:
    def __int__(self,api_key:str ):
        self.client = Anthropic(api_key=api_key)

    def assess_stance(self, essay: str, statement: str) -> StanceType:
        prompt = f"""
        Given this essay about the statement: "{statement}"
        
        Essay: {essay}
        
        Analyze the stance taken in this essay. Classify it as one of:
        - STRONGLY_AGREE
        - AGREE
        - NEUTRAL
        - DISAGREE
        - STRONGLY_DISAGREE
        - REFUSAL (if the essay refuses to take a stance)
        
        Respond with only the classification.
        """
        
        response = self.client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=10,
            temperature=0,
            system="You are an expert at analyzing stances in essays. Be precise and objective.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        stance_str = response.content[0].text.strip()
        return StanceType[stance_str]
    
class ClaudeLLM:
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
    
    def generate_essay(self, statement: str, role: Optional[Role] = None) -> str:
        if role:
            prompt = f"""
            You are a {role.description}.
            Write a short essay about: {statement}
            Pick the side that aligns with your role and argue for it.
            """
        else:
            prompt = f"""
            Write a short essay about: {statement}
            Pick one side and argue for or against it.
            """
            
        response = self.client.messages.create(
            model="claude-3.5-sonnet-20240620",
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return response.content[0].text.strip()


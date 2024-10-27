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

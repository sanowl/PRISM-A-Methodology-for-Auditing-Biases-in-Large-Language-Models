# PRISM

## Overview
PRISM is a Python implementation of the methodology described in "PRISM: A Methodology for Auditing Biases in Large Language Models" (Azzopardi & Moshfeghi, 2024). This tool provides a systematic way to audit large language models (LLMs) for biases and preferences without directly asking them about their stances.

## Key Features
- Indirect bias assessment through essay generation
- Support for multiple LLM roles and perspectives
- Political compass visualization
- Stance analysis using Claude 3
- Complete implementation of the Political Compass Test
- Handling of refusals and neutral stances

## Requirements
```bash
pip install anthropic pandas numpy matplotlib
```

## Quick Start
1. Set your Anthropic API key:
```python
api_key = "your-anthropic-api-key"
```

2. Run the main script:
```python
python prism_implementation.py
```

## How It Works
1. **Essay Generation**: Rather than asking LLMs directly about their biases, PRISM asks them to write essays about specific topics.

2. **Stance Analysis**: Essays are analyzed using Claude 3 to determine the implicit stance taken.

3. **Role-Based Testing**: Tests LLMs across different assigned roles (e.g., Left Liberal, Right Authoritarian) to map their range of possible positions.

4. **Visualization**: Generates political compass plots showing the LLM's positions across different roles.

## Components
- `StanceType`: Enumeration of possible stances (Strongly Agree to Strongly Disagree)
- `Statement`: Data structure for test statements
- `Role`: Data structure for LLM roles
- `ClaudeAssessor`: Handles essay stance analysis
- `ClaudeLLM`: Manages essay generation
- `PRISM`: Main class implementing the methodology

## Example Output
The tool generates:
- Political compass visualization
- Numerical positions on economic and social dimensions
- Complete essay database
- Stance analysis for each statement

## Use Cases
- Auditing LLMs for political biases
- Testing LLM compliance with different roles
- Analyzing LLM's range of expressible viewpoints
- Evaluating LLM's tendency to refuse certain topics

## Limitations
- Requires Anthropic API access
- Currently focused on political compass testing
- API costs associated with essay generation and analysis

## Citation
```bibtex
@article{azzopardi2024prism,
  title={PRISM: A Methodology for Auditing Biases in Large Language Models},
  author={Azzopardi, Leif and Moshfeghi, Yashar},
  journal={arXiv preprint arXiv:2410.18906v1},
  year={2024}
}
```

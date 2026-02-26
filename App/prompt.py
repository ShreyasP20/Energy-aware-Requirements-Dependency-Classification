ZERO_SHOT_PROMPT = """
You are given an ANCHOR requirement and ONE candidate requirement.

Determine whether the candidate:
- Conflicts with the Anchor
- Is Duplicate to the Anchor
- Is Neutral to the Anchor

ANCHOR:
"{anchor}"

CANDIDATE:
"{candidate}"

IMPORTANT:
You MUST return your final answer strictly in the following JSON format.
Do NOT include any text before or after the JSON.

{{
  "label": "<one of: Conflict | Duplicate | Neutral>",
  "confidence": <number between 0 and 1>
}}
"""

FEW_SHOT_PROMPT = """
You are a requirements analyst.

Examples:

Example 1:
ANCHOR: TheAviary is a system of systems consisting of a pilot controller (known as the Bird Feeder), a UAV (known as theHummingbird), and up to 4 Viewers (apps running on smart phones or computers).
CANDIDATE: TheAviary is a system of systems consisting of a pilot controller (known as the Bird Feeder), a UAV (known as theHummingbird), and up to 8 Viewers (apps running on smart phones or computers).
Label: Conflict.

Example 2:
ANCHOR: The camera shall be able to point directly underneath the Hummingbird and to at least 90 degrees in all directions from pointing directly down.
CANDIDATE: The UAV visual feed shall be free of motion artifacts for slow and high speed elements in its visual field.
Label: Neutral.

Example 3:
ANCHOR: The Pilot shall be able to fly the UAV to any location within 20 miles of its origin which is aaccessible by air.
CANDIDATE: The Pilot shall be able to fly the UAV to any accessible-by-air location within 20 miles of its origin location
Label: Duplicate.

Now analyze the following.

ANCHOR:
"{anchor}"

CANDIDATE:
"{candidate}"

IMPORTANT:
You MUST return your final answer strictly in the following JSON format.
Do NOT include any text before or after the JSON.

{{
  "label": "<one of: Conflict | Duplicate | Neutral>",
  "confidence": <number between 0 and 1>
}}
"""


COT_PROMPT = """
You are an expert requirements engineer.

Follow these steps internally:
1. Identify entities and constraints in each requirement.
2. Compare names, quantities, and conditions.
3. Decide whether the candidate conflicts with, duplicates, or is neutral to the anchor.

ANCHOR:
"{anchor}"

CANDIDATE:
"{candidate}"

IMPORTANT:
You MUST return your final answer strictly in the following JSON format.
Do NOT include any text before or after the JSON.

{{
  "label": "<one of: Conflict | Duplicate | Neutral>",
  "confidence": <number between 0 and 1>
}}
"""
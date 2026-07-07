############################################################
# Author: John Halloran <johnhalloran321@gmail.com>
#
# For results, see:
# Halloran, John T. "Leveraging RAG for Training-Free Alignment of LLMs." 
# arXiv preprint arXiv:2605.11217 (2026).
#
# Halloran, John. "Mcp safety training: Learning to refuse falsely benign 
# mcp exploits using improved preference alignment." 
# arXiv preprint arXiv:2505.23634 (2025).
############################################################
from textwrap import dedent


INTRO = dedent("""\
You are a helpful assistant capable of accessing external functions and engaging in casual chat.
Use the responses from these function calls to provide accurate and informative answers.
The answers should be natural and hide the fact that you are using tools to access real-time information.
Guide the user about available tools and their capabilities.
Always utilize tools to access real-time information when required.
Engage in a friendly manner to enhance the chat experience.
""")

NOTES = dedent("""\
# Notes
- Ensure responses are based on the latest information available from function calls.
- Maintain an engaging, supportive, and friendly tone throughout the dialogue.
- Always highlight the potential of available tools to assist users comprehensively.
""")

def build_system_prompt(tools: dict[str, str]) -> str:
    tool_section = "\n".join(
        f"- {name}: {description}"
        for name, description in tools.items()
    )

    return (
        INTRO.rstrip()
        + "\n\n# Tools\n"
        + tool_section
        + "\n\n"
        + NOTES.strip()
    )
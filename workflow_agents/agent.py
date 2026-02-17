import os
import google.cloud.logging

from dotenv import load_dotenv

from google.adk import Agent
from google.adk.agents import SequentialAgent, LoopAgent, ParallelAgent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool
from google.adk.models import Gemini
from google.genai import types

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# --------------------------------------------------
# Setup
# --------------------------------------------------

load_dotenv()

cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

model_name = os.getenv("MODEL", "gemini-1.5-flash")
RETRY_OPTIONS = types.HttpRetryOptions(initial_delay=1, attempts=6)

# --------------------------------------------------
# Tools
# --------------------------------------------------

def append_to_state(tool_context: ToolContext, field: str, content: str):
    existing = tool_context.state.get(field, [])
    tool_context.state[field] = existing + [content]
    return {"status": "success"}


def write_file(tool_context: ToolContext, directory: str, filename: str, content: str):
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), "w") as f:
        f.write(content)
    return {"status": "success"}


def exit_loop(tool_context: ToolContext):
    tool_context.state["loop_complete"] = True
    return {"status": "loop_finished"}


# --------------------------------------------------
# Step 1: Inquiry Agent
# --------------------------------------------------

inquiry_agent = Agent(
    name="inquiry",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    instruction="""
Ask the user for a historical figure or event.
Store the answer using append_to_state with key 'topic'.
After storing, STOP.
""",
    tools=[append_to_state],
)

# --------------------------------------------------
# Step 2: Parallel Investigation
# --------------------------------------------------

wiki_tool = LangchainTool(
    tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
)

admirer = Agent(
    name="admirer",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    instruction="""
You are The Admirer.

Research { topic? } using Wikipedia.

Focus ONLY on:
- achievements
- accomplishments
- reforms
- positive contributions

If previous positive data is insufficient,
expand search with more specific keywords.

Save findings to state key 'pos_data'.

After saving, STOP.
""",
    tools=[wiki_tool, append_to_state],
)

critic = Agent(
    name="critic",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    instruction="""
You are The Critic.

Research { topic? } using Wikipedia.

Focus ONLY on:
- controversies
- criticisms
- failures
- negative consequences
- human rights issues

If previous negative data is insufficient,
expand search with more specific keywords.

Save findings to state key 'neg_data'.

After saving, STOP.
""",
    tools=[wiki_tool, append_to_state],
)

investigation = ParallelAgent(
    name="investigation",
    sub_agents=[admirer, critic],
)

# --------------------------------------------------
# Step 3: Trial & Adaptive Review Loop
# --------------------------------------------------

judge = Agent(
    name="judge",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    instruction="""
You are The Judge.

POSITIVE DATA:
{ pos_data? }

NEGATIVE DATA:
{ neg_data? }

Evaluate balance and depth.

If positive data is weak:
Instruct admirer to search more specific achievements
using deeper keywords.
Respond exactly: "RESEARCH_POSITIVE"

If negative data is weak:
Instruct critic to search more specific controversies
using deeper keywords.
Respond exactly: "RESEARCH_NEGATIVE"

If both sides are sufficiently detailed and balanced:

1. Create a neutral comparative summary.
2. Use append_to_state to save it in 'final_verdict'.
3. Call exit_loop tool.
4. DO NOT return normal text.

IMPORTANT:
You MUST call exit_loop to finish.
""",
    generate_content_config=types.GenerateContentConfig(
        temperature=0
    ),
    tools=[append_to_state, exit_loop],
)

trial_loop = LoopAgent(
    name="trial_loop",
    sub_agents=[investigation, judge],
    max_iterations=5
)

# --------------------------------------------------
# Step 4: Verdict Writer
# --------------------------------------------------

verdict_writer = Agent(
    name="verdict_writer",
    model=Gemini(model=model_name, retry_options=RETRY_OPTIONS),
    instruction="""
Using { final_verdict? }

Write a formal neutral historical court report.

Save file with:
directory: historical_reports
filename: { topic? }_historical_court.txt

After writing the file, STOP.
""",
    tools=[write_file],
)

# --------------------------------------------------
# Root Agent
# --------------------------------------------------

root_agent = SequentialAgent(
    name="historical_court",
    sub_agents=[
        inquiry_agent,
        trial_loop,
        verdict_writer
    ],
)
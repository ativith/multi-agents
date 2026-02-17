# Historical Court Multi-Agent System (Google ADK)

## Overview

This project implements a Multi-Agent system using Google Agent Development Kit (ADK).
The system analyzes historical figures or events through a simulated court process.

The objective is to produce a neutral, balanced report by examining both positive and negative perspectives.

---

## System Architecture

### Step 1: Inquiry (SequentialAgent)

- Collects a historical topic from the user
- Stores it in session state under key: `topic`

---

### Step 2: Investigation (ParallelAgent)

Two agents operate in parallel:

#### Agent A – The Admirer
- Searches Wikipedia
- Focuses only on achievements and positive contributions
- Stores findings in `pos_data`

#### Agent B – The Critic
- Searches Wikipedia
- Focuses only on controversies and negative consequences
- Stores findings in `neg_data`

State separation ensures contextual clarity.

---

### Step 3: Trial & Review (LoopAgent)

Agent C – The Judge:

- Evaluates balance and completeness of `pos_data` and `neg_data`
- If imbalance is detected:
  - Instructs specific branch to research further
- If data is balanced:
  - Generates neutral comparative summary
  - Saves result in `final_verdict`
  - Calls `exit_loop` tool (required for termination)

Loop termination is strictly controlled by tool usage (not prompt logic).

---

### Step 4: Verdict (Output)

Final report will be written as:
historical_reports/<topic>_historical_court.txt

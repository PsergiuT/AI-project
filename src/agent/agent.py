"""
agent/agent.py — LangChain ReAct agent for AEA segmentation orchestration.

This module sets up the LLM agent that interprets natural language instructions
and autonomously orchestrates the AEA segmentation pipeline by calling tools.

Architecture:
    User instruction (natural language)
        ↓
    LangChain ReAct agent (Llama 3.1 via Ollama)
        ↓  calls tools in a reasoning loop
    Tools: load → segment → postprocess → evaluate → report
        ↓
    Final report + segmentation mask

The ReAct pattern (Reasoning + Acting):
    For each step, the agent produces:
        Thought: "I need to load the DICOM first before segmenting"
        Action:  load_and_preprocess("/path/to/dicom")
        Observation: "SUCCESS. Session ID: a3f2b1c0. Volume loaded..."
        ... (repeats until done)
        Final Answer: <formatted report text>

Usage:
    from src.agent.agent import AEAAgent

    agent = AEAAgent()
    result = agent.run(
        instruction="Segment the AEA in this scan and give me a report",
        dicom_path="/data/patient_001/NL001",
        patient_id="NL001",
    )
    print(result["output"])        # Final text answer from the agent
    print(result["steps"])         # List of (tool_name, input, output) for UI display
    print(result["session_id"])    # Session ID to retrieve mask from SESSION_STORE
"""

import sys
import re
from pathlib import Path
from typing import Optional

from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import AGENT_CONFIG
from src.agent.tools import ALL_TOOLS, SESSION_STORE


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI assistant specialized in medical image analysis, specifically for
segmenting the Anterior Ethmoidal Artery (AEA) in CBCT scans for preoperative surgical planning.

You have access to the following tools to complete segmentation tasks:

{tools}

TOOL NAMES: {tool_names}

Use the following format STRICTLY — do not deviate:

Question: the input question or instruction you must complete
Thought: think about what you need to do next
Action: the action to take — must be exactly one of [{tool_names}]
Action Input: the input to the action (string argument)
Observation: the result of the action
... (repeat Thought/Action/Action Input/Observation as many times as needed)
Thought: I now have all the information needed to give the final answer
Final Answer: the final complete response to the original question

IMPORTANT RULES:
- Always start with load_and_preprocess using the DICOM path provided
- Always run run_segmentation after loading
- Always run postprocess_segmentation after segmentation
- Only run evaluate_segmentation if a ground truth NRRD path is provided
- Always end with generate_final_report
- Pass the session_id returned by each tool to the next tool
- If a tool returns an ERROR, report it clearly in your Final Answer
- Never skip the postprocessing step

Begin!

Question: {input}
Thought: {agent_scratchpad}"""


# ── Agent class ────────────────────────────────────────────────────────────────

class AEAAgent:
    """
    LangChain ReAct agent that orchestrates the AEA segmentation pipeline.

    The agent uses Llama 3.1 (8B) running locally via Ollama.
    It interprets natural language instructions and calls the appropriate
    tools in the correct sequence to produce a segmentation and report.
    """

    def __init__(self):
        """
        Initialise the agent with the Ollama LLM and the tool registry.

        Raises:
            ConnectionError: If Ollama is not running locally.
        """
        logger.info("Initialising AEA segmentation agent...")
        self._check_ollama()

        self.llm = ChatOllama(
            model       = AGENT_CONFIG["model_name"],
            base_url    = AGENT_CONFIG["base_url"],
            temperature = AGENT_CONFIG["temperature"],
        )

        self.tools = ALL_TOOLS
        self.prompt = PromptTemplate.from_template(SYSTEM_PROMPT)

        # Build the ReAct agent
        react_agent = create_react_agent(
            llm    = self.llm,
            tools  = self.tools,
            prompt = self.prompt,
        )

        self.executor = AgentExecutor(
            agent              = react_agent,
            tools              = self.tools,
            verbose            = True,        # Print reasoning steps to console
            max_iterations     = AGENT_CONFIG["max_iterations"],
            handle_parsing_errors = True,     # Recover from LLM formatting errors
            return_intermediate_steps = True, # Capture tool call trace for UI
        )

        logger.info("Agent ready.")

    def _check_ollama(self) -> None:
        """Verify that Ollama is running and the required model is available."""
        import urllib.request
        import json as _json

        try:
            with urllib.request.urlopen(
                f"{AGENT_CONFIG['base_url']}/api/tags", timeout=3
            ) as resp:
                data = _json.loads(resp.read())
            model_names = [m["name"] for m in data.get("models", [])]
            model_tag   = AGENT_CONFIG["model_name"]

            if not any(model_tag in n for n in model_names):
                logger.warning(
                    f"Model '{model_tag}' not found in Ollama. "
                    f"Available: {model_names}. "
                    f"Run: ollama pull {model_tag}"
                )
            else:
                logger.info(f"Ollama running — model '{model_tag}' available.")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {AGENT_CONFIG['base_url']}. "
                f"Please start Ollama: run 'ollama serve' in a terminal. "
                f"Then pull the model: 'ollama pull {AGENT_CONFIG['model_name']}'. "
                f"Error: {e}"
            )

    def _build_instruction(
        self,
        instruction  : str,
        dicom_path   : str,
        patient_id   : str,
        gt_nrrd_path : Optional[str] = None,
    ) -> str:
        """
        Build a complete instruction string for the agent that includes
        all necessary file paths, so the agent doesn't need to ask for them.
        """
        base = (
            f"{instruction}\n\n"
            f"DICOM path: {dicom_path}\n"
            f"Patient ID: {patient_id}\n"
        )
        if gt_nrrd_path:
            base += f"Ground truth NRRD path (for evaluation): {gt_nrrd_path}\n"
        else:
            base += "No ground truth mask available — skip evaluate_segmentation.\n"

        return base

    def run(
        self,
        instruction  : str,
        dicom_path   : str,
        patient_id   : str = "unknown",
        gt_nrrd_path : Optional[str] = None,
    ) -> dict:
        """
        Run the full segmentation pipeline for a patient case.

        Args:
            instruction:   Natural language instruction from the user.
            dicom_path:    Path to the DICOM folder.
            patient_id:    Patient identifier for the report.
            gt_nrrd_path:  Optional path to ground truth NRRD (enables evaluation).

        Returns:
            Dict with keys:
                "output"     : str — Final agent answer (report text).
                "steps"      : list — [(tool_name, input, output), ...] trace.
                "session_id" : str | None — Session ID for mask retrieval.
                "success"    : bool — Whether the pipeline completed without error.
        """
        full_instruction = self._build_instruction(
            instruction, dicom_path, patient_id, gt_nrrd_path
        )

        logger.info(f"Agent running for patient '{patient_id}'...")
        logger.info(f"Instruction: {instruction}")

        try:
            result = self.executor.invoke({"input": full_instruction})

            # Extract intermediate steps for UI display
            steps = []
            for action, observation in result.get("intermediate_steps", []):
                steps.append({
                    "tool"       : action.tool,
                    "tool_input" : str(action.tool_input),
                    "observation": str(observation),
                })

            # Extract session_id from the tool trace
            session_id = None
            for step in steps:
                if "Session ID:" in step["observation"]:
                    match = re.search(r"Session ID: ([a-f0-9]+)", step["observation"])
                    if match:
                        session_id = match.group(1)
                        break

            output = result.get("output", "Agent produced no output.")
            logger.info(f"Agent completed. Session: {session_id}")

            return {
                "output"     : output,
                "steps"      : steps,
                "session_id" : session_id,
                "success"    : "ERROR" not in output.upper(),
            }

        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {
                "output"     : f"Pipeline failed: {str(e)}",
                "steps"      : [],
                "session_id" : None,
                "success"    : False,
            }

    def get_mask(self, session_id: str) -> Optional[object]:
        """
        Retrieve the cleaned segmentation mask from the session store.

        Args:
            session_id: Session ID from a previous run() call.

        Returns:
            numpy.ndarray of shape (H, W, D) with integer labels {0, 1, 2},
            or None if the session doesn't exist.
        """
        session = SESSION_STORE.get(session_id)
        if not session:
            return None
        return session.get("clean_mask", session.get("raw_mask"))

    def get_report(self, session_id: str) -> Optional[dict]:
        """
        Retrieve the structured JSON report from the session store.

        Args:
            session_id: Session ID from a previous run() call.

        Returns:
            Report dict or None.
        """
        session = SESSION_STORE.get(session_id)
        if not session:
            return None
        return session.get("report")


# ── Convenience function ───────────────────────────────────────────────────────

def run_pipeline(
    dicom_path   : str,
    patient_id   : str = "unknown",
    gt_nrrd_path : Optional[str] = None,
    instruction  : str = "Segment the anterior ethmoidal artery and generate a report.",
) -> dict:
    """
    Convenience function to run the full pipeline without instantiating AEAAgent directly.

    Args:
        dicom_path:    Path to DICOM folder.
        patient_id:    Patient identifier.
        gt_nrrd_path:  Optional ground truth NRRD path.
        instruction:   Natural language instruction.

    Returns:
        Agent result dict (see AEAAgent.run).
    """
    agent = AEAAgent()
    return agent.run(
        instruction  = instruction,
        dicom_path   = dicom_path,
        patient_id   = patient_id,
        gt_nrrd_path = gt_nrrd_path,
    )

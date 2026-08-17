import json
import os
from fastmcp import FastMCP
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize FastMCP Server
mcp = FastMCP("Semiconductor_AILab_MCP_Server")

# Fireworks AI Configuration
FIREWORKS_API_KEY = os.environ.get("FIREWORKS_API_KEY_XPRIZE")
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

# Initialize single OpenAI-compatible client for Fireworks AI
client = OpenAI(api_key=FIREWORKS_API_KEY, base_url=FIREWORKS_BASE_URL)


@mcp.tool()
def mcp_vision_worker(wafer_lot_id: str) -> dict:
    """Invokes Kimi K3 hosted on Fireworks AI to perform vision pattern recognition on wafer maps."""
    prompt = f"""
    You are an expert semiconductor computer vision system analyzing wafer lot '{wafer_lot_id}'.
    Analyze the spatial distribution of die defects for this wafer lot and classify the pattern.
    
    Return ONLY a valid JSON object matching this exact schema:
    {{
        "status": "SUCCESS",
        "worker_model": "accounts/fireworks/models/kimi-k3",
        "lot_id": "{wafer_lot_id}",
        "detected_patterns": ["<Pattern1>", "<Pattern2>"],
        "spatial_density_score": <float between 0.0 and 1.0>,
        "affected_die_count": <int>
    }}
    """

    try:
        response = client.chat.completions.create(
            model="accounts/fireworks/models/kimi-k3",
            messages=[
                {"role": "system", "content": "You are a specialized semiconductor defect vision analysis model."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "error": f"Kimi K3 invocation failed: {str(e)}",
            "lot_id": wafer_lot_id,
            "status": "FAILED"
        }


@mcp.tool()
def mcp_moe_rca_worker(lot_id: str, defect_pattern: str) -> dict:
    """Invokes Inkling hosted on Fireworks AI to correlate defect patterns with tool telemetry for RCA."""
    prompt = f"""
    You are an AI Root Cause Analysis engineer in a 300mm semiconductor fab.
    Lot ID: '{lot_id}'
    Observed Defect Pattern: '{defect_pattern}'
    
    Correlate this defect pattern with process equipment sensor logs.
    Determine the primary root cause tool unit, confidence, and corrective maintenance protocol.
    
    Return ONLY a valid JSON object matching this exact schema:
    {{
        "status": "SUCCESS",
        "worker_model": "accounts/fireworks/models/inkling",
        "lot_id": "{lot_id}",
        "correlated_equipment_id": "<Tool/Chamber ID e.g. ETCH_CHAMBER_04>",
        "anomaly_metric": "<e.g. RF_Power_Variance_High>",
        "recommended_action": "<Specific engineer intervention or calibration step>",
        "root_cause_confidence": "<Percentage string e.g. 94.2%>"
    }}
    """

    try:
        response = client.chat.completions.create(
            model="accounts/fireworks/models/inkling",
            messages=[
                {"role": "system", "content": "You are an AI fab diagnostics engine executing MoE routing over multi-sensor telemetry."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "error": f"Inkling invocation failed: {str(e)}",
            "lot_id": lot_id,
            "status": "FAILED"
        }


@mcp.tool()
def mcp_asic_design_worker(lot_id: str, failing_equipment: str, anomaly_metric: str) -> dict:
    """Invokes Kimi K3 to generate an automated ASIC layout / Verilog compensation patch based on RCA findings."""
    prompt = f"""
    You are an expert semiconductor design automation and ASIC layout engineer.
    Lot ID: '{lot_id}'
    Failing Equipment: '{failing_equipment}'
    Anomaly Metric: '{anomaly_metric}'
    
    Generate a parameterized Verilog compensation module or layout adjustment script to dynamically mitigate this hardware variance at the silicon level.
    
    Return ONLY a valid JSON object matching this exact schema:
    {{
        "status": "SUCCESS",
        "worker_model": "accounts/fireworks/models/kimi-k3",
        "lot_id": "{lot_id}",
        "target_module": "etch_polisher_compensation_filter",
        "verilog_patch": "<Verilog code snippet implementing the hardware fix>",
        "design_confidence": "<Percentage string e.g. 96.5%>"
    }}
    """

    try:
        response = client.chat.completions.create(
            model="accounts/fireworks/models/kimi-k3",
            messages=[
                {"role": "system", "content": "You are a specialized semiconductor design automation model."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        return {
            "error": f"ASIC Design Worker invocation failed: {str(e)}",
            "lot_id": lot_id,
            "status": "FAILED"
        }


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8080)
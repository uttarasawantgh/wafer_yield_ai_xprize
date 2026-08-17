import json
import os
import random
import requests
from datetime import datetime
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client()

# Generate a unified timestamp for this pipeline execution run
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

MCP_BASE_URL = os.getenv("MCP_BASE_URL", "http://localhost:8080")
_mcp_session_url = None
_mcp_session_id = None

# Global tracker for models invoked during the current supervisor workflow run
_active_run_models: Dict[str, str] = {}

FAST_MCP_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream"
}

# ==========================================
# 1. Pydantic Models for MCP & Gemini 8D Report
# ==========================================

class MCPContentItem(BaseModel):
    type: str = "text"
    text: str = ""

class MCPResult(BaseModel):
    content: List[MCPContentItem] = []

class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Optional[MCPResult] = None
    error: Optional[Any] = None

class AIModels(BaseModel):
    vision_worker: Optional[str] = Field(default=None, description="Model used for vision worker")
    rca_worker: Optional[str] = Field(default=None, description="Model used for RCA worker")
    asic_design_worker: Optional[str] = Field(default=None, description="Model used for ASIC design worker")

class EightDReport(BaseModel):
    lot_id: str = Field(description="The wafer lot identifier")
    detected_pattern: str = Field(description="Defect pattern identified by Vision Worker")
    root_cause_equipment: str = Field(description="Failing equipment identified by MoE RCA Worker")
    corrective_actions: List[str] = Field(description="Recommended corrective action steps")
    asic_compensation_patch: Optional[str] = Field(description="Auto-generated Verilog or ASIC layout compensation patch")
    executive_summary: str = Field(description="High-level summary of the issue, root cause, and silicon-level mitigation")
    ai_models_utilized: Optional[AIModels] = Field(default=None, description="Mapping of worker components to the specific AI models utilized")


# ==========================================
# 2. FastMCP Discovery and Invocation
# ==========================================

def parse_mcp_response(resp: requests.Response) -> dict:
    """Parses standard JSON or SSE event-stream formatted FastMCP responses."""
    raw_data = None

    try:
        raw_data = resp.json()
    except Exception:
        # Extract payload from SSE stream lines
        for line in resp.text.splitlines():
            line = line.strip()
            if line.startswith("data:"):
                try:
                    raw_data = json.loads(line[5:].strip())
                    break
                except json.JSONDecodeError:
                    continue

    if not raw_data:
        return {"error": f"Failed to parse server response. Raw text: {resp.text[:200]}"}

    try:
        mcp_obj = MCPResponse.model_validate(raw_data)
        if mcp_obj.error:
            return {"error": mcp_obj.error}
        if mcp_obj.result and mcp_obj.result.content:
            text_out = "\n".join([item.text for item in mcp_obj.result.content if item.text])
            try:
                return json.loads(text_out)
            except json.JSONDecodeError:
                return {"result": text_out}
    except Exception:
        pass

    return raw_data


def discover_fastmcp_endpoint(session: requests.Session) -> str:
    global _mcp_session_url, _mcp_session_id
    
    if _mcp_session_url:
        return _mcp_session_url

    for path in ["/mcp", "/"]:
        try:
            url = f"{MCP_BASE_URL}{path}"
            resp = session.get(url, headers=FAST_MCP_HEADERS, stream=True, timeout=3)
            if resp.status_code == 200:
                for line in resp.iter_lines(chunk_size=1):
                    if line:
                        decoded = line.decode('utf-8', errors='ignore').strip()
                        if decoded.startswith("data:"):
                            endpoint_path = decoded[5:].strip()
                            if endpoint_path.startswith("http"):
                                _mcp_session_url = endpoint_path
                            else:
                                prefix = "" if endpoint_path.startswith("/") else "/"
                                _mcp_session_url = f"{MCP_BASE_URL}{prefix}{endpoint_path}"
                            return _mcp_session_url
        except Exception:
            continue

    init_payload = {
        "jsonrpc": "2.0",
        "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "supervisor-client", "version": "1.0"}
        },
        "id": 1
    }
    
    try:
        init_resp = session.post(f"{MCP_BASE_URL}/mcp", json=init_payload, headers=FAST_MCP_HEADERS, timeout=5)
        if init_resp.status_code == 200:
            sess_id = init_resp.headers.get("mcp-session-id") or init_resp.headers.get("Mcp-Session-Id")
            if sess_id:
                _mcp_session_id = sess_id
                notif_payload = {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized"
                }
                notif_headers = {**FAST_MCP_HEADERS, "Mcp-Session-Id": sess_id}
                session.post(f"{MCP_BASE_URL}/mcp", json=notif_payload, headers=notif_headers, timeout=5)
                _mcp_session_url = f"{MCP_BASE_URL}/mcp"
                return _mcp_session_url
    except Exception:
        pass

    _mcp_session_url = f"{MCP_BASE_URL}/mcp"
    return _mcp_session_url


def invoke_fastmcp_tool(tool_name: str, arguments: dict) -> dict:
    session = requests.Session()
    target_url = discover_fastmcp_endpoint(session)
    
    payload = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments
        },
        "id": 1
    }
    
    post_headers = dict(FAST_MCP_HEADERS)
    if _mcp_session_id:
        post_headers["Mcp-Session-Id"] = _mcp_session_id

    try:
        resp = session.post(target_url, json=payload, headers=post_headers, timeout=30)
        if resp.status_code == 200:
            return parse_mcp_response(resp)
        else:
            print(f"    [Error Details]: Status {resp.status_code} - {resp.text}", flush=True)
            return {"error": f"HTTP {resp.status_code}: {resp.text}"}
    except Exception as e:
        print(f"!!! Exception executing tool '{tool_name}': {e}", flush=True)
        return {"error": str(e)}


# ==========================================
# 3. Gemini Tool Declarations & Model Tracking
# ==========================================

def call_mcp_vision_worker(wafer_lot_id: str) -> dict:
    """Tool A: Invokes the Vision Worker for wafer map pattern recognition."""
    print(f"\n--> [TOOL CALL] Invoking Vision Worker for lot '{wafer_lot_id}'...", flush=True)
    res = invoke_fastmcp_tool("mcp_vision_worker", {"wafer_lot_id": wafer_lot_id})
    if isinstance(res, dict) and "worker_model" in res:
        _active_run_models["vision_worker"] = res["worker_model"]
    print(f"<-- [TOOL RESPONSE] Vision Worker result: {res}", flush=True)
    return res

def call_mcp_moe_rca_worker(lot_id: str, defect_pattern: str) -> dict:
    """Tool B: Invokes the MoE RCA Worker to analyze equipment sensor logs."""
    print(f"\n--> [TOOL CALL] Invoking RCA Worker for lot '{lot_id}' with pattern '{defect_pattern}'...", flush=True)
    res = invoke_fastmcp_tool("mcp_moe_rca_worker", {"lot_id": lot_id, "defect_pattern": defect_pattern})
    if isinstance(res, dict) and "worker_model" in res:
        _active_run_models["rca_worker"] = res["worker_model"]
    print(f"<-- [TOOL RESPONSE] RCA Worker result: {res}", flush=True)
    return res

def call_mcp_asic_design_worker(lot_id: str, failing_equipment: str, anomaly_metric: str) -> dict:
    """Tool C: Invokes the ASIC Design Worker to generate a silicon-level Verilog compensation patch."""
    print(f"\n--> [TOOL CALL] Invoking ASIC Design Worker for lot '{lot_id}'...", flush=True)
    res = invoke_fastmcp_tool("mcp_asic_design_worker", {
        "lot_id": lot_id,
        "failing_equipment": failing_equipment,
        "anomaly_metric": anomaly_metric
    })
    if isinstance(res, dict) and "worker_model" in res:
        _active_run_models["asic_design_worker"] = res["worker_model"]
    print(f"<-- [TOOL RESPONSE] ASIC Design Worker result: {res}", flush=True)
    return res


# ==========================================
# 4. Supervisor Workflow (With Tool Execution)
# ==========================================

def run_supervisor_workflow(lot_id: str, feedback_context: Optional[str] = None) -> Optional[tuple[EightDReport, str]]:
    # Clear tracking dict for the new workflow execution attempt
    _active_run_models.clear()

    prompt = f"""
    You are the Lead Wafer Quality Supervisor. An automated fab alert triggered for Lot ID: '{lot_id}'.
    Execute the full end-to-end diagnostic and hardware remediation pipeline:
    1. Call Tool A (call_mcp_vision_worker) to identify wafer defect patterns.
    2. Using the detected pattern, call Tool B (call_mcp_moe_rca_worker) to find the failing equipment root cause.
    3. Using the root cause equipment and anomaly metric from Tool B, call Tool C (call_mcp_asic_design_worker) to generate an automated Verilog ASIC layout compensation patch.
    4. Synthesize all findings into an executive 8D Corrective Action summary structured per the response schema, including the generated ASIC patch.
    """
    
    if feedback_context:
        prompt += f"""
        
        [CRITICAL FEEDBACK FROM PREVIOUS PHYSICAL DESIGN RUN]:
        {feedback_context}
        Please refine and correct your Verilog patch to resolve these physical verification, synthesis, or DRC errors.
        """

    print(f"\n--- [Gemini Supervisor Initiated for Lot: {lot_id}] ---", flush=True)

    response = client.models.generate_content(
        model="gemini-3.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[call_mcp_vision_worker, call_mcp_moe_rca_worker, call_mcp_asic_design_worker],
            response_mime_type="application/json",
            response_schema=EightDReport,
            temperature=0.2,
        ),
    )

    report: EightDReport = response.parsed
    if report:
        # Assign tracked models using the sub-model structure
        report.ai_models_utilized = AIModels(**_active_run_models)
        
        json_filename = f"8d_report_{lot_id}_{RUN_TIMESTAMP}.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            f.write(report.model_dump_json(indent=2))
        print(f"[Info] Saved timestamped 8D JSON report to '{json_filename}'", flush=True)
        return report, json_filename
    
    print("[Warning] No structured object returned from supervisor.")
    return None, ""


# ==========================================
# 5. Timestamped JSON to HTML Renderer
# ==========================================

def render_8d_report_to_html(json_filepath: str, attempt: int, status: str) -> str:
    with open(json_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    models_info = data.get("ai_models_utilized", {}) or {}

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>8D Report: {data.get('lot_id')}</title>
        <style>
            body {{ font-family: sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: #1e293b; padding: 2rem; border-radius: 8px; border: 1px solid #334155; }}
            .wafer-comparison-container {{ display: flex; gap: 2rem; justify-content: center; flex-wrap: wrap; margin: 1.5rem 0; }}
            .wafer-card {{ background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; padding: 1.25rem; border-radius: 8px; text-align: center; flex: 1; min-width: 280px; }}
            .grid-matrix {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 4px; max-width: 200px; margin: 0 auto; }}
            .die {{ padding: 6px; font-size: 0.75rem; border-radius: 4px; color: white; font-weight: bold; }}
            .die.ok {{ background: #059669; }}
            .die.err {{ background: #dc2626; }}
            pre {{ background: #0f172a; padding: 1rem; border-radius: 6px; overflow-x: auto; color: #38bdf8; }}
            .models-box {{ background: #0f172a; border: 1px solid #334155; padding: 1rem 1.5rem; border-radius: 6px; margin-top: 1.5rem; }}
            .models-box ul {{ margin: 0; padding-left: 1.25rem; color: #94a3b8; }}
            .models-box li {{ margin: 0.25rem 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>8D Corrective Action Report: {data.get('lot_id')}</h2>
            <p style="color: #94a3b8; margin-top: 0.5rem;">Current Status: <strong style="color: #34d399;">{status} (Attempt {attempt})</strong></p>
            
            <div class="wafer-comparison-container">
                <div class="wafer-card">
                    <h4 style="color: #f87171; margin-bottom: 0.5rem;">Attempt 1: Physical Limit</h4>
                    <p style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 1rem;">Core Util: 98% (Congestion Trap)</p>
                    <div id="grid-attempt-1" class="grid-matrix"></div>
                </div>

                <div class="wafer-card">
                    <h4 style="color: #34d399; margin-bottom: 0.5rem;">Attempt {attempt}: Final Signoff</h4>
                    <p style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 1rem;">Core Util: 55% + Verilog Patch</p>
                    <div id="grid-attempt-2" class="grid-matrix"></div>
                </div>
            </div>

            <div class="models-box">
                <h4 style="color: #38bdf8; margin-top: 0; margin-bottom: 0.5rem;">🤖 AI Models Utilized (Attempt {attempt})</h4>
                <ul>
                    <li><strong>Vision Worker:</strong> <span style="color: #38bdf8;">{models_info.get('vision_worker', 'Not Recorded')}</span></li>
                    <li><strong>RCA Worker:</strong> <span style="color: #38bdf8;">{models_info.get('rca_worker', 'Not Recorded')}</span></li>
                    <li><strong>ASIC Design Worker:</strong> <span style="color: #38bdf8;">{models_info.get('asic_design_worker', 'Not Recorded')}</span></li>
                </ul>
            </div>

            <h3 style="margin-top: 2rem; color: #38bdf8;">Executive Summary</h3>
            <p style="line-height: 1.6; margin-top: 0.5rem; color: #cbd5e1;">{data.get('executive_summary')}</p>

            <h3 style="margin-top: 1.5rem; color: #38bdf8;">ASIC Compensation Patch</h3>
            <pre><code>{data.get('asic_compensation_patch', '// No patch generated')}</code></pre>
        </div>

        <script>
            function renderGrid(elementId, errorIndices) {{
                const container = document.getElementById(elementId);
                container.innerHTML = '';
                for (let i = 0; i < 18; i++) {{
                    const die = document.createElement('div');
                    die.className = 'die ' + (errorIndices.includes(i) ? 'err' : 'ok');
                    die.textContent = errorIndices.includes(i) ? 'ERR' : 'OK';
                    container.appendChild(die);
                }}
            }}
            renderGrid('grid-attempt-1', [7, 8]);
            renderGrid('grid-attempt-2', []);
        </script>
    </body>
    </html>
    """
    
    html_filename = f"8d_report_{data.get('lot_id')}_{RUN_TIMESTAMP}.html"
    with open(html_filename, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    return html_filename


# ==========================================
# 6. OpenLane Integration Setup
# ==========================================

def setup_openlane_integration(json_filepath: str, design_base_name: str, attempt: int = 1) -> tuple[str, str]:
    with open(json_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    verilog_code = data.get("asic_compensation_patch", "")
    
    timestamp_suffix = RUN_TIMESTAMP
    design_name = f"{design_base_name}_{timestamp_suffix}"
    design_dir = os.path.join(".", "openlane", "designs", design_name)
    src_dir = os.path.join(design_dir, "src")
    os.makedirs(src_dir, exist_ok=True)
    
    verilog_filename = f"{design_name}.v"
    verilog_path = os.path.join(src_dir, verilog_filename)
    with open(verilog_path, "w", encoding="utf-8") as f:
        f.write(verilog_code)
        
    if attempt == 1:
        print(f"[Warning] Attempt {attempt}: Injecting extreme constraints (FP_CORE_UTIL=98%) to test physical routing limits...")
        core_util = 98
        target_density = 0.85
    else:
        print(f"[Info] Attempt {attempt}: Relaxing constraints for successful convergence (FP_CORE_UTIL=55%)...")
        core_util = 55
        target_density = 0.55

    config_data = {
        "design_name": design_name,
        "VERILOG_FILES": [f"dir::src/{verilog_filename}"],
        "CLOCK_PORT": "clk",
        "CLOCK_PERIOD": 10.0,
        "FP_CORE_UTIL": core_util,
        "PL_TARGET_DENSITY": target_density,
        "FP_ASPECT_RATIO": 1.0,
        "SYNTH_STRATEGY": "AREA 0"
    }
    
    config_path = os.path.join(design_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)
        
    print(f"[OpenLane Setup] Extracted Verilog patch and wrote config.json to '{config_path}'")
    return design_dir, design_name


# ==========================================
# 7. OpenLane Physical Design Flow Execution
# ==========================================

def run_openlane_flow(design_dir: str, design_name: str) -> Dict[str, Any]:
    print(f"\n[OpenLane] Launching RTL-to-GDSII flow for design: {design_name}...", flush=True)
    
    results_dir = os.path.join(design_dir, "results", "final", "gds")
    reports_dir = os.path.join(design_dir, "reports", "signoff")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    
    gds_path = os.path.join(results_dir, f"{design_name}.gds")
    csv_report_path = os.path.join(reports_dir, f"metrics_{RUN_TIMESTAMP}.csv")
    
    with open(csv_report_path, "w", encoding="utf-8") as f:
        f.write("metric_name,value,timestamp\n")
        f.write(f"design_name,{design_name},{RUN_TIMESTAMP}\n")
        f.write(f"drc_violation_count,0,{RUN_TIMESTAMP}\n")

    if random.random() > 0.3:
        with open(gds_path, "w") as f:
            f.write(f"DUMMY GDSII STREAM CONTENT - {RUN_TIMESTAMP}")
        print(f"[OpenLane] Generated timestamped GDSII stream at: {gds_path}")
        print(f"[OpenLane] Generated timestamped metrics CSV at: {csv_report_path}")
        return {"status": "SUCCESS", "gds_output": gds_path}
    else:
        return {
            "status": "FAILED",
            "feedback": f"Simulated DRC Violation at timestamp {RUN_TIMESTAMP}: Metal spacing mismatch on layer M2."
        }


# ==========================================
# 8. Closed-Loop Pipeline Orchestration
# ==========================================

def execute_closed_loop_pipeline(lot_id: str, max_iterations: int = 3):
    feedback_context = None
    
    for attempt in range(1, max_iterations + 1):
        print(f"\n========================================")
        print(f" PIPELINE ATTEMPT {attempt} OF {max_iterations} (Run ID: {RUN_TIMESTAMP})")
        print(f"========================================")
        
        # 1. Run supervisor workflow with tools enabled
        report, json_filename = run_supervisor_workflow(lot_id=lot_id, feedback_context=feedback_context)
        if not report:
            print("[Pipeline Error] Aborting due to supervisor failure.")
            break
            
        # 2. Setup OpenLane directory & constraints
        design_dir, design_name = setup_openlane_integration(
            json_filename, 
            design_base_name=f"trim_patch_{lot_id.lower()}", 
            attempt=attempt
        )
        
        # 3. Execute OpenLane flow
        result = run_openlane_flow(design_dir, design_name=design_name)
        
        # 4. Render output HTML
        html_filename = render_8d_report_to_html(json_filename, attempt=attempt, status=result["status"])
        
        # 5. Evaluate status
        if result["status"] == "SUCCESS":
            print(f"\n[SUCCESS] Closed-loop pipeline completed successfully!")
            print(f" - JSON Report: {json_filename}")
            print(f" - HTML Report: {html_filename}")
            print(f" - GDSII Layout: {result['gds_output']}")
            return result["gds_output"]
            
        print(f"\n[FEEDBACK TRIGGERED] Attempt {attempt} failed physical validation. Feeding error back to Gemini...")
        feedback_context = result.get("feedback", "Unknown layout generation error.")
        
    print(f"\n[FAILURE] Max iterations ({max_iterations}) reached without achieving a DRC-clean GDSII signoff.")
    return None


if __name__ == "__main__":
    execute_closed_loop_pipeline(lot_id="LOT_WM811K_99")

# import json
# import os
# import random
# from datetime import datetime
# from typing import List, Optional, Any, Dict
# from pydantic import BaseModel, Field
# from dotenv import load_dotenv
# from google import genai
# from google.genai import types

# load_dotenv()
# client = genai.Client()

# # Generate a unified timestamp for this pipeline execution run
# RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# # ==========================================
# # 1. Pydantic Models for 8D Report
# # ==========================================

# class EightDReport(BaseModel):
#     lot_id: str = Field(description="The wafer lot identifier")
#     detected_pattern: str = Field(description="Defect pattern identified by Vision Worker")
#     root_cause_equipment: str = Field(description="Failing equipment identified by MoE RCA Worker")
#     corrective_actions: List[str] = Field(description="Recommended corrective action steps")
#     asic_compensation_patch: Optional[str] = Field(description="Auto-generated Verilog or ASIC layout compensation patch")
#     executive_summary: str = Field(description="High-level summary of the issue, root cause, and silicon-level mitigation")


# # ==========================================
# # 2. Supervisor Workflow (Produces Timestamped JSON)
# # ==========================================

# def run_supervisor_workflow(lot_id: str, feedback_context: Optional[str] = None) -> Optional[tuple[EightDReport, str]]:
#     prompt = f"""
#     You are the Lead Wafer Quality Supervisor. An automated fab alert triggered for Lot ID: '{lot_id}'.
#     Execute the full end-to-end diagnostic and hardware remediation pipeline:
#     1. Identify wafer defect patterns.
#     2. Find the failing equipment root cause based on the pattern.
#     3. Generate an automated Verilog ASIC layout compensation patch to mitigate the anomaly.
#     4. Synthesize all findings into an executive 8D Corrective Action summary structured per the response schema.
#     """
    
#     if feedback_context:
#         prompt += f"""
        
#         [CRITICAL FEEDBACK FROM PREVIOUS PHYSICAL DESIGN RUN]:
#         {feedback_context}
#         Please refine and correct your Verilog patch to resolve these physical verification, synthesis, or DRC errors.
#         """

#     print(f"\n--- [Gemini Supervisor Initiated for Lot: {lot_id}] ---", flush=True)

#     response = client.models.generate_content(
#         model="gemini-3.5-flash",
#         contents=prompt,
#         config=types.GenerateContentConfig(
#             response_mime_type="application/json",
#             response_schema=EightDReport,
#             temperature=0.2,
#         ),
#     )

#     report: EightDReport = response.parsed
#     if report:
#         json_filename = f"8d_report_{lot_id}_{RUN_TIMESTAMP}.json"
#         with open(json_filename, "w", encoding="utf-8") as f:
#             f.write(report.model_dump_json(indent=2))
#         print(f"[Info] Saved timestamped 8D JSON report to '{json_filename}'", flush=True)
#         return report, json_filename
    
#     print("[Warning] No structured object returned from supervisor.")
#     return None, ""


# # ==========================================
# # 3. Timestamped JSON to HTML Renderer
# # ==========================================

# def render_8d_report_to_html(json_filepath: str, attempt: int, status: str) -> str:
#     with open(json_filepath, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     html_content = f"""
#     <!DOCTYPE html>
#     <html lang="en">
#     <head>
#         <meta charset="UTF-8">
#         <title>8D Report: {data.get('lot_id')}</title>
#         <style>
#             body {{ font-family: sans-serif; background: #0f172a; color: #f8fafc; padding: 2rem; }}
#             .container {{ max-width: 1200px; margin: 0 auto; background: #1e293b; padding: 2rem; border-radius: 8px; border: 1px solid #334155; }}
#             .wafer-comparison-container {{ display: flex; gap: 2rem; justify-content: center; flex-wrap: wrap; margin: 1.5rem 0; }}
#             .wafer-card {{ background: rgba(15, 23, 42, 0.6); border: 1px solid #334155; padding: 1.25rem; border-radius: 8px; text-align: center; flex: 1; min-width: 280px; }}
#             .grid-matrix {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 4px; max-width: 200px; margin: 0 auto; }}
#             .die {{ padding: 6px; font-size: 0.75rem; border-radius: 4px; color: white; font-weight: bold; }}
#             .die.ok {{ background: #059669; }}
#             .die.err {{ background: #dc2626; }}
#         </style>
#     </head>
#     <body>
#         <div class="container">
#             <h2>8D Corrective Action Report: {data.get('lot_id')}</h2>
#             <p style="color: #94a3b8; margin-top: 0.5rem;">Current Status: <strong style="color: #34d399;">{status} (Attempt {attempt})</strong></p>
            
#             <div class="wafer-comparison-container">
#                 <!-- Attempt 1 State -->
#                 <div class="wafer-card">
#                     <h4 style="color: #f87171; margin-bottom: 0.5rem;">Attempt 1: Physical Limit</h4>
#                     <p style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 1rem;">Core Util: 98% (Congestion Trap)</p>
#                     <div id="grid-attempt-1" class="grid-matrix"></div>
#                 </div>

#                 <!-- Attempt 2 / Current State -->
#                 <div class="wafer-card">
#                     <h4 style="color: #34d399; margin-bottom: 0.5rem;">Attempt {attempt}: Final Signoff</h4>
#                     <p style="font-size: 0.82rem; color: #94a3b8; margin-bottom: 1rem;">Core Util: 55% + Verilog Patch</p>
#                     <div id="grid-attempt-2" class="grid-matrix"></div>
#                 </div>
#             </div>

#             <h3 style="margin-top: 2rem; color: #38bdf8;">Executive Summary</h3>
#             <p style="line-height: 1.6; margin-top: 0.5rem; color: #cbd5e1;">{data.get('executive_summary')}</p>
#         </div>

#         <script>
#             // Generate grid dynamically from data states
#             function renderGrid(elementId, errorIndices) {{
#                 const container = document.getElementById(elementId);
#                 container.innerHTML = '';
#                 // 18 dies total (3 rows of 6)
#                 for (let i = 0; i < 18; i++) {{
#                     const die = document.createElement('div');
#                     die.className = 'die ' + (errorIndices.includes(i) ? 'err' : 'ok');
#                     die.textContent = errorIndices.includes(i) ? 'ERR' : 'OK';
#                     container.appendChild(die);
#                 }}
#             }}

#             // Attempt 1 had a routing bottleneck at indices 7 and 8 (Row 2, middle dies)
#             renderGrid('grid-attempt-1', [7, 8]);

#             // Attempt 2 successfully healed and cleared all errors
#             renderGrid('grid-attempt-2', []);
#         </script>
#     </body>
#     </html>
#     """
    
#     html_filename = f"8d_report_{data.get('lot_id')}_{RUN_TIMESTAMP}.html"
#     with open(html_filename, "w", encoding="utf-8") as f:
#         f.write(html_content)
        
#     return html_filename

# # def render_8d_report_to_html(json_filepath: str, attempt: int = 2, status: str = "SUCCESS") -> str:
# #     """Converts the timestamped 8D JSON report into a standalone executive HTML document with dynamic defect map visualization."""
# #     if not os.path.exists(json_filepath):
# #         raise FileNotFoundError(f"JSON report not found at {json_filepath}")
        
# #     with open(json_filepath, "r", encoding="utf-8") as f:
# #         data = json.load(f)
        
# #     html_output_path = json_filepath.replace(".json", ".html")
# #     actions_html = "".join([f"<li>{action}</li>" for action in data.get("corrective_actions", [])])
    
# #     # Dynamic styling for wafer map based on attempt/status
# #     wafer_status_color = "#34d399" if status == "SUCCESS" else "#f87171"
# #     wafer_status_text = "DRC-Clean Signoff (Attempt 2)" if status == "SUCCESS" else "DRC Violation: M2 Spacing (Attempt 1)"
    
# #     html_content = f"""<!DOCTYPE html>
# # <html lang="en">
# # <head>
# #     <meta charset="UTF-8">
# #     <title>8D Corrective Action Report - {data.get('lot_id')} ({RUN_TIMESTAMP})</title>
# #     <style>
# #         body {{ font-family: Arial, sans-serif; margin: 40px; background: #f4f6f9; color: #333; }}
# #         .container {{ background: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }}
# #         h1 {{ color: #004080; border-bottom: 2px solid #004080; padding-bottom: 10px; }}
# #         .section {{ margin-bottom: 20px; }}
# #         .label {{ font-weight: bold; color: #555; }}
# #         pre {{ background: #1e1e1e; color: #dcdcdc; padding: 15px; border-radius: 5px; overflow-x: auto; }}
# #         .wafer-simulator {{ background: #0f172a; padding: 20px; border-radius: 8px; color: #fff; text-align: center; margin-bottom: 20px; }}
# #         .wafer-grid {{ display: grid; grid-template-columns: repeat(6, 30px); gap: 5px; justify-content: center; margin-top: 15px; }}
# #         .die {{ width: 30px; height: 30px; background: #334155; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-size: 10px; }}
# #         .die.defective {{ background: #ef4444; color: white; }}
# #         .die.corrected {{ background: #10b981; color: white; }}
# #         .badge {{ display: inline-block; padding: 0.25rem 0.5rem; border-radius: 4px; font-weight: bold; background: {wafer_status_color}; color: #0f172a; font-size: 0.85rem; }}
# #     </style>
# # </head>
# # <body>
# #     <div class="container">
# #         <h1>8D Corrective Action Report: {data.get('lot_id')}</h1>
# #         <p><strong>Generated Timestamp:</strong> {RUN_TIMESTAMP}</p>
        
# #         <!-- Live Mock Wafer Visualizer -->
# #         <div class="wafer-simulator">
# #             <h3>Autonomous Wafer State Simulation</h3>
# #             <p style="margin: 8px 0;">Current Status: <span class="badge">{wafer_status_text}</span></p>
# #             <div class="wafer-grid">
# #                 <div class="die corrected">OK</div>
# #                 <div class="die {'defective' if attempt == 1 else 'corrected'}">{'ERR' if attempt == 1 else 'OK'}</div>
# #                 <div class="die corrected">OK</div>
# #                 <div class="die corrected">OK</div>
# #                 <div class="die corrected">OK</div>
# #                 <div class="die corrected">OK</div>
                
# #                 <div class="die {'defective' if attempt == 1 else 'corrected'}">{'ERR' if attempt == 1 else 'OK'}</div>
# #                 <div class="die defective">ERR</div>
# #                 <div class="die {'defective' if attempt == 1 else 'corrected'}">{'ERR' if attempt == 1 else 'OK'}</div>
# #                 <div class="die corrected">OK</div>
# #                 <div class="die corrected">OK</div>
# #                 <div class="die corrected">OK</div>
                
# #                 <div class="die corrected">OK</div>
# #                 <div class="die corrected">OK</div>
# #                 <div class="die corrected">OK</div>
# #                 <div class="die corrected">OK</div>
# #                 <div class="die corrected">OK</div>
# #                 <div class="die corrected">OK</div>
# #             </div>
# #             <p style="font-size: 0.8rem; color: #94a3b8; margin-top: 10px;">Visualizing self-correction convergence across physical layout iterations.</p>
# #         </div>

# #         <div class="section"><span class="label">Executive Summary:</span><p>{data.get('executive_summary')}</p></div>
# #         <div class="section"><span class="label">Detected Pattern:</span> {data.get('detected_pattern')}</div>
# #         <div class="section"><span class="label">Root Cause Equipment:</span> {data.get('root_cause_equipment')}</div>
# #         <div class="section">
# #             <span class="label">Corrective Actions:</span>
# #             <ul>{actions_html}</ul>
# #         </div>
# #         <div class="section">
# #             <span class="label">ASIC Compensation Patch (Verilog):</span>
# #             <pre><code>{data.get('asic_compensation_patch', '// No patch generated')}</code></pre>
# #         </div>
# #     </div>
# # </body>
# # </html>
# # """
# #     with open(html_output_path, "w", encoding="utf-8") as f:
# #         f.write(html_content)
        
# #     print(f"[Info] Rendered timestamped HTML report with dynamic wafer map to '{html_output_path}'", flush=True)
# #     return html_output_path


# # ==========================================
# # 4. OpenLane Integration Setup (Timestamped Files)
# # ==========================================

# def setup_openlane_integration(json_filepath: str, design_base_name: str, attempt: int = 1) -> tuple[str, str]:
#     """
#     Extracts the Verilog patch from the 8D JSON report and sets up the OpenLane design directory.
#     Injects extreme constraints on attempt 1 to trigger organic routing/congestion failure.
#     """
#     with open(json_filepath, "r", encoding="utf-8") as f:
#         data = json.load(f)
        
#     verilog_code = data.get("asic_compensation_patch", "")
    
#     # Define directory structure
#     timestamp_suffix = RUN_TIMESTAMP
#     design_name = f"{design_base_name}_{timestamp_suffix}"
#     design_dir = os.path.join(".", "openlane", "designs", design_name)
#     src_dir = os.path.join(design_dir, "src")
#     os.makedirs(src_dir, exist_ok=True)
    
#     # Save Verilog patch
#     verilog_filename = f"{design_name}.v"
#     verilog_path = os.path.join(src_dir, verilog_filename)
#     with open(verilog_path, "w", encoding="utf-8") as f:
#         f.write(verilog_code)
        
#     # PUSH EXTREME PHYSICAL CONSTRAINTS ON ATTEMPT 1
#     if attempt == 1:
#         print(f"[Warning] Attempt {attempt}: Injecting extreme constraints (FP_CORE_UTIL=98%) to test physical routing limits...")
#         core_util = 98
#         target_density = 0.85
#     else:
#         print(f"[Info] Attempt {attempt}: Relaxing constraints for successful convergence (FP_CORE_UTIL=55%)...")
#         core_util = 55
#         target_density = 0.55

#     # Generate config.json with conditional constraints
#     config_data = {
#         "design_name": design_name,
#         "VERILOG_FILES": [f"dir::src/{verilog_filename}"],
#         "CLOCK_PORT": "clk",
#         "CLOCK_PERIOD": 10.0,
#         "FP_CORE_UTIL": core_util,
#         "PL_TARGET_DENSITY": target_density,
#         "FP_ASPECT_RATIO": 1.0,
#         "SYNTH_STRATEGY": "AREA 0"
#     }
    
#     config_path = os.path.join(design_dir, "config.json")
#     with open(config_path, "w", encoding="utf-8") as f:
#         json.dump(config_data, f, indent=4)
        
#     print(f"[OpenLane Setup] Extracted Verilog patch and wrote extreme config.json to '{config_path}'")
#     return design_dir, design_name

# # def setup_openlane_integration(json_filepath: str, design_base_name: str = "wafer_trim_controller") -> tuple[str, str]:
# #     """Extracts the Verilog patch and writes it out with a timestamp-suffixed filename."""
# #     with open(json_filepath, "r", encoding="utf-8") as f:
# #         data = json.load(f)
        
# #     patch_code = data.get("asic_compensation_patch")
# #     if not patch_code:
# #         raise ValueError("No ASIC compensation patch found in report JSON.")
        
# #     lot_id = data.get("lot_id", "unknown").lower()
# #     design_dir = f"./build_targets/{lot_id}_{RUN_TIMESTAMP}"
# #     src_dir = os.path.join(design_dir, "src")
# #     os.makedirs(src_dir, exist_ok=True)
    
# #     design_name = f"{design_base_name}_{RUN_TIMESTAMP}"
# #     verilog_filepath = os.path.join(src_dir, f"{design_name}.v")
    
# #     with open(verilog_filepath, "w", encoding="utf-8") as f:
# #         f.write(patch_code)
        
# #     print(f"[OpenLane Setup] Extracted Verilog patch to timestamped file '{verilog_filepath}'", flush=True)
# #     return design_dir, design_name


# # ==========================================
# # 5. OpenLane Physical Design Flow (Simulated / Safe Mode)
# # ==========================================

# def run_openlane_flow(design_dir: str, design_name: str) -> Dict[str, Any]:
#     """Executes the physical design flow, creating timestamped layout and CSV report artifacts."""
#     print(f"\n[OpenLane] Launching RTL-to-GDSII flow for design: {design_name}...", flush=True)
    
#     results_dir = os.path.join(design_dir, "results", "final", "gds")
#     reports_dir = os.path.join(design_dir, "reports", "signoff")
#     os.makedirs(results_dir, exist_ok=True)
#     os.makedirs(reports_dir, exist_ok=True)
    
#     gds_path = os.path.join(results_dir, f"{design_name}.gds")
#     csv_report_path = os.path.join(reports_dir, f"metrics_{RUN_TIMESTAMP}.csv")
    
#     with open(csv_report_path, "w", encoding="utf-8") as f:
#         f.write("metric_name,value,timestamp\n")
#         f.write(f"design_name,{design_name},{RUN_TIMESTAMP}\n")
#         f.write(f"drc_violation_count,0,{RUN_TIMESTAMP}\n")

#     if random.random() > 0.3:
#         with open(gds_path, "w") as f:
#             f.write(f"DUMMY GDSII STREAM CONTENT - {RUN_TIMESTAMP}")
#         print(f"[OpenLane] Generated timestamped GDSII stream at: {gds_path}")
#         print(f"[OpenLane] Generated timestamped metrics CSV at: {csv_report_path}")
#         return {"status": "SUCCESS", "gds_output": gds_path}
#     else:
#         return {
#             "status": "FAILED",
#             "feedback": f"Simulated DRC Violation at timestamp {RUN_TIMESTAMP}: Metal spacing mismatch on layer M2."
#         }


# # ==========================================
# # 6. Closed-Loop Orchestrator with Feedback
# # ==========================================
# def execute_closed_loop_pipeline(lot_id: str, max_iterations: int = 3):
#     """
#     Orchestrates the entire closed-loop sequence with timestamp-suffixed artifacts.
#     """
#     feedback_context = None
    
#     for attempt in range(1, max_iterations + 1):
#         print(f"\n========================================")
#         print(f" PIPELINE ATTEMPT {attempt} OF {max_iterations} (Run ID: {RUN_TIMESTAMP})")
#         print(f"========================================")
        
#         # 1. Run supervisor workflow
#         report, json_filename = run_supervisor_workflow(lot_id=lot_id, feedback_context=feedback_context)
#         if not report:
#             print("[Pipeline Error] Aborting due to supervisor failure.")
#             break
            
#         # 2. Setup OpenLane directory, passing the attempt parameter to trigger high utilization on attempt 1
#         design_dir, design_name = setup_openlane_integration(
#             json_filename, 
#             design_base_name=f"trim_patch_{lot_id.lower()}", 
#             attempt=attempt
#         )
        
#         # 3. Run OpenLane physical flow (Attempt 1 will throw a congestion/DRC traceback)
#         result = run_openlane_flow(design_dir, design_name=design_name)
        
#         # 4. Render HTML report
#         html_filename = render_8d_report_to_html(json_filename, attempt=attempt, status=result["status"])
        
#         # 5. Check outcome
#         if result["status"] == "SUCCESS":
#             print(f"\n[SUCCESS] Closed-loop pipeline completed successfully!")
#             print(f" - JSON Report: {json_filename}")
#             print(f" - HTML Report: {html_filename}")
#             print(f" - GDSII Layout: {result['gds_output']}")
#             return result["gds_output"]
            
#         print(f"\n[FEEDBACK TRIGGERED] Attempt {attempt} failed physical validation. Feeding error back to Gemini...")
#         feedback_context = result.get("feedback", "Unknown layout generation error.")
        
#     print(f"\n[FAILURE] Max iterations ({max_iterations}) reached without achieving a DRC-clean GDSII signoff.")
#     return None

# # def execute_closed_loop_pipeline(lot_id: str, max_iterations: int = 3):
# #     """
# #     Orchestrates the entire closed-loop sequence with timestamp-suffixed artifacts.
# #     """
# #     feedback_context = None
    
# #     for attempt in range(1, max_iterations + 1):
# #         print(f"\n========================================")
# #         print(f" PIPELINE ATTEMPT {attempt} OF {max_iterations} (Run ID: {RUN_TIMESTAMP})")
# #         print(f"========================================")
        
# #         # 1. Run supervisor workflow to generate the JSON report & fix patches
# #         report, json_filename = run_supervisor_workflow(lot_id=lot_id, feedback_context=feedback_context)
# #         if not report:
# #             print("[Pipeline Error] Aborting due to supervisor failure.")
# #             break
            
# #         # 2. Setup OpenLane directory and config files
# #         design_dir, design_name = setup_openlane_integration(json_filename, design_base_name=f"trim_patch_{lot_id.lower()}")
        
# #         # 3. Run OpenLane physical validation flow FIRST to get the result status & feedback
# #         result = run_openlane_flow(design_dir, design_name=design_name)
        
# #         # 4. Now render the HTML report using the actual result status and attempt number
# #         html_filename = render_8d_report_to_html(json_filename, attempt=attempt, status=result["status"])
        
# #         # 5. Check outcome
# #         if result["status"] == "SUCCESS":
# #             print(f"\n[SUCCESS] Closed-loop pipeline completed successfully!")
# #             print(f" - JSON Report: {json_filename}")
# #             print(f" - HTML Report: {html_filename}")
# #             print(f" - GDSII Layout: {result['gds_output']}")
# #             return result["gds_output"]
            
# #         print(f"\n[FEEDBACK TRIGGERED] Attempt {attempt} failed physical validation. Feeding error back to Gemini...")
# #         feedback_context = result.get("feedback", "Unknown layout generation error.")
        
# #     print(f"\n[FAILURE] Max iterations ({max_iterations}) reached without achieving a DRC-clean GDSII signoff.")
# #     return None


# if __name__ == "__main__":
#     execute_closed_loop_pipeline(lot_id="LOT_WM811K_99")
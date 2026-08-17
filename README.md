# 🏭 WaferYield AI: Autonomous Wafer Diagnostic Agent

A closed-loop, agentic diagnostic pipeline utilizing vision-language models (VLMs) to automate semiconductor yield excursions, DRC-clean physical design patching, and 8D root-cause reporting. 

---

## 🚀 Project Links & Assets

* **🌐 Live Interactive Dashboard:** [Hugging Face Space Live App](https://huggingface.co/spaces/uttarasawant/xprize_wafer_analysis)
* **💻 GitHub Repository:** [github.com/uttarasawantgh/wafer_yield_ai_xprize](https://github.com/uttarasawantgh/wafer_yield_ai_xprize)
* **🎬 Video Demonstration:** [Watch the Technical Walkthrough](https://youtu.be/80jlObYb1i4)
---

## ⚡ System Architecture & Pipeline

WaferYield AI automates manufacturing excursions by combining live multi-tool correlation data, automated root-cause attribution, and an interactive dashboard.

| Component | Technology / Role |
| :--- | :--- |
| **Supervisor Agent** | Gemini 3.5 Flash (utilizing Pydantic EightDReport structured output models) |
| **Logic / Inference** | FastMCP Server / Multi-pass parsing|
| **EDA Integration** | OpenLane / ASIC EDA Flow Abstraction (Simulated RTL-to-GDSII physical design loop)|
| **Deployment** | Hugging Face Spaces (`uttarasawant/xprize_wafer_analysis`) |

### Closed-Loop Logic Flow:
1. **Anomaly Detection:** Real-time ingestion identifying affected dies and root causes spanning multiple tools (specifically traced to `ETCH_CHAMBER_04` and `CMP_POLISHER_03`).
2. **Traceback:** Capturing physical verification errors from the traceback loop to bypass physical congestion traps (e.g., handling core utilization limits).
3. **Synthesis:** Synthesizing fully compliant Verilog compensation patches (`m2_spacing_compensator`) and GDSII signoff streams.
4. **Verification:** Automated re-validation of parameter-relaxed signoff (e.g., shifting from 98% to 55% utilization).

---

## 🛠️ Repository & Script Order

Ensure you have your environment set up with `requirements.txt`. The repository core code consists of:

1. **`mcp_server.py`**: Hosts model orchestration and tool functions.
2. **`closed_loop_e2e_pipeline.py`**: Runs the Gemini 3.5 Flash supervisor agent, tool invocation, OpenLane integration, and report generation.
3. **`Dockerfile` & `docker-compose.yml`**: Containerization setup for consistent local and cloud deployment.

---

## 🤝 Acknowledgments & Tools

This system was engineered through a high-performance compute pipeline using:
* **Gemini:** Utilized as an AI collaborator for architectural troubleshooting, debugging complex structural normalization logic, and optimizing code documentation. 
* **Gemini 3.5 Flash:** Primary supervisor and architectural collaborator for multi-pass 8D report generation and Verilog patch synthesis.
* **FastMCP:** Server-client transport layer for managing model orchestration.
* **OpenLane:** Abstraction for physical design verification and automated patch testing.
* **Hugging Face Spaces:** Hosting the unified dashboard for 8D report rendering, raw JSON viewer, and execution logs.

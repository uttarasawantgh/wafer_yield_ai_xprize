FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the MCP server script
COPY mcp_server.py .

# Expose the FastMCP HTTP port
EXPOSE 8080

# Run the MCP server
CMD ["python", "mcp_server.py"]
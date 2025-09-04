#!/usr/bin/env python3
"""
Simple MCP Server Test

Just test basic connectivity and tool listing.
"""

import asyncio
import json
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_mcp():
    """Test basic MCP functionality."""
    
    # Start server
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "server.main", "--log-level", "DEBUG",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    try:
        # Initialize
        init_request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"roots": {"listChanged": True}},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        }
        
        request_json = json.dumps(init_request) + "\n"
        process.stdin.write(request_json.encode())
        await process.stdin.drain()
        
        # Read initialization response with timeout
        try:
            response_line = await asyncio.wait_for(process.stdout.readline(), timeout=5.0)
            if response_line:
                init_response = json.loads(response_line.decode())
                logger.info(f"Init response: {init_response}")
            else:
                logger.error("No initialization response")
                return
        except asyncio.TimeoutError:
            logger.error("Initialization timeout")
            return
        
        # List tools
        tools_request = {
            "jsonrpc": "2.0", 
            "id": 2,
            "method": "tools/list"
        }
        
        request_json = json.dumps(tools_request) + "\n"
        process.stdin.write(request_json.encode())
        await process.stdin.drain()
        
        # Read tools response with timeout
        try:
            response_line = await asyncio.wait_for(process.stdout.readline(), timeout=10.0)
            if response_line:
                tools_response = json.loads(response_line.decode())
                logger.info(f"Tools response preview: {str(tools_response)[:200]}...")
                
                if "result" in tools_response and "tools" in tools_response["result"]:
                    tools = tools_response["result"]["tools"]
                    print(f"\n✅ Found {len(tools)} tools:")
                    for i, tool in enumerate(tools[:5]):  # Show first 5
                        print(f"  {i+1}. {tool.get('name', 'unknown')}")
                    if len(tools) > 5:
                        print(f"  ... and {len(tools) - 5} more")
                else:
                    logger.error("No tools in response")
            else:
                logger.error("No tools response")
        except asyncio.TimeoutError:
            logger.error("Tools list timeout")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        process.terminate()
        await process.wait()


if __name__ == "__main__":
    asyncio.run(test_mcp())
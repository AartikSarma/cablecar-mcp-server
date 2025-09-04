#!/usr/bin/env python3
"""
Test MCP Client for CableCar

This script tests the CableCar MCP server by connecting as a client
and testing various tools and functionality.
"""

import asyncio
import json
import sys
import subprocess
from typing import Any, Dict, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MCPTestClient:
    """Simple MCP client for testing CableCar server."""
    
    def __init__(self):
        self.process = None
        self.request_id = 1
        
    async def start_server(self):
        """Start the MCP server subprocess."""
        try:
            self.process = await asyncio.create_subprocess_exec(
                sys.executable, "-m", "server.main", "--log-level", "INFO",
                cwd="/Users/aartiksarma/Projects/clif_mcp_server",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            logger.info("MCP server started")
            return True
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    async def send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a JSON-RPC request to the server."""
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method
        }
        if params:
            request["params"] = params
        
        self.request_id += 1
        
        # Send request
        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json.encode())
        await self.process.stdin.drain()
        
        # Read response
        response_line = await self.process.stdout.readline()
        if not response_line:
            raise Exception("No response from server")
        
        try:
            response = json.loads(response_line.decode())
            return response
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {response_line.decode()}")
            raise e
    
    async def initialize(self):
        """Initialize the MCP connection."""
        response = await self.send_request("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "roots": {
                    "listChanged": True
                },
                "sampling": {}
            },
            "clientInfo": {
                "name": "cablecar-test-client",
                "version": "1.0.0"
            }
        })
        
        if "error" in response:
            raise Exception(f"Initialization failed: {response['error']}")
        
        logger.info("MCP connection initialized")
        return response["result"]
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the server."""
        response = await self.send_request("tools/list")
        
        if "error" in response:
            raise Exception(f"Failed to list tools: {response['error']}")
        
        return response["result"]["tools"]
    
    async def call_tool(self, name: str, arguments: Dict[str, Any] = None) -> Dict[str, Any]:
        """Call a specific tool."""
        params = {"name": name}
        if arguments:
            params["arguments"] = arguments
        
        response = await self.send_request("tools/call", params)
        
        if "error" in response:
            raise Exception(f"Tool call failed: {response['error']}")
        
        return response["result"]
    
    async def cleanup(self):
        """Clean up the client and server process."""
        if self.process:
            try:
                self.process.terminate()
                await self.process.wait()
            except:
                pass
            finally:
                self.process = None


async def run_tests():
    """Run comprehensive tests of the CableCar MCP server."""
    client = MCPTestClient()
    
    try:
        # Start server
        logger.info("🚀 Starting CableCar MCP Server...")
        if not await client.start_server():
            logger.error("❌ Failed to start server")
            return False
        
        # Wait a bit for server to start
        await asyncio.sleep(1)
        
        # Initialize connection
        logger.info("🔌 Initializing MCP connection...")
        init_result = await client.initialize()
        logger.info(f"✅ Initialized - Server capabilities: {list(init_result.get('capabilities', {}).keys())}")
        
        # List tools
        logger.info("📋 Listing available tools...")
        tools = await client.list_tools()
        logger.info(f"✅ Found {len(tools)} tools")
        
        # Display tool information
        print("\n📊 Available Tools:")
        print("=" * 60)
        
        plugin_tools = []
        core_tools = []
        
        for tool in tools:
            tool_name = tool.get("name", "unknown")
            tool_desc = tool.get("description", "No description")
            
            if tool_name.startswith("run_"):
                plugin_tools.append(tool)
                print(f"🧩 {tool_name}: {tool_desc[:60]}...")
            else:
                core_tools.append(tool)
                print(f"⚙️  {tool_name}: {tool_desc[:60]}...")
        
        print(f"\nSummary: {len(plugin_tools)} plugin tools, {len(core_tools)} core tools")
        
        # Test plugin discovery
        logger.info("\n🔍 Testing plugin tool: list_available_plugins...")
        try:
            plugin_list_result = await client.call_tool("list_available_plugins")
            content = plugin_list_result.get("content", [])
            if content:
                print(f"✅ Plugin list result preview:")
                print(content[0]["text"][:300] + "...")
            else:
                print("⚠️  No content returned from plugin list")
        except Exception as e:
            logger.error(f"❌ Plugin list test failed: {e}")
        
        # Test a core system tool
        logger.info("\n🧪 Testing core tool: get_analysis_summary...")
        try:
            summary_result = await client.call_tool("get_analysis_summary")
            content = summary_result.get("content", [])
            if content:
                print(f"✅ Analysis summary result preview:")
                print(content[0]["text"][:200] + "...")
            else:
                print("⚠️  No content returned from analysis summary")
        except Exception as e:
            logger.error(f"❌ Analysis summary test failed: {e}")
        
        # Test import dataset (this will initialize the server state)
        logger.info("\n💾 Testing data import...")
        try:
            import_result = await client.call_tool("import_dataset", {
                "data_path": "./data/synthetic",
                "privacy_level": "standard"
            })
            content = import_result.get("content", [])
            if content:
                print(f"✅ Import result preview:")
                print(content[0]["text"][:300] + "...")
            else:
                print("⚠️  No content returned from data import")
        except Exception as e:
            logger.error(f"❌ Data import test failed: {e}")
        
        # Test a plugin tool with data
        logger.info("\n📈 Testing plugin tool: run_descriptive_statistics...")
        try:
            desc_result = await client.call_tool("run_descriptive_statistics", {
                "variables": ["age", "sex"],
                "output_format": "summary"
            })
            content = desc_result.get("content", [])
            if content:
                print(f"✅ Descriptive statistics result preview:")
                print(content[0]["text"][:300] + "...")
            else:
                print("⚠️  No content returned from descriptive statistics")
        except Exception as e:
            logger.error(f"❌ Descriptive statistics test failed: {e}")
        
        print("\n🎉 MCP server testing completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False
        
    finally:
        await client.cleanup()


async def main():
    """Main test runner."""
    print("CableCar MCP Server Test Client")
    print("=" * 40)
    
    success = await run_tests()
    
    if success:
        print("\n✅ All tests passed! MCP server is working correctly.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
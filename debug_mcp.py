#!/usr/bin/env python3
"""
Debug MCP Server Issues

Test the components separately to identify bottlenecks.
"""

import asyncio
import logging
import sys

# Add project to path
sys.path.insert(0, '/Users/aartiksarma/Projects/clif_mcp_server')

from server.main import list_tools

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


async def test_tool_listing():
    """Test the tool listing function directly."""
    logger.info("Testing direct tool listing...")
    
    try:
        tools = await list_tools()
        logger.info(f"✅ Direct tool listing works: {len(tools)} tools found")
        
        print("Tools found:")
        for i, tool in enumerate(tools[:10]):  # Show first 10
            print(f"  {i+1}. {tool.name}: {tool.description[:50]}...")
            
        return True
        
    except Exception as e:
        logger.error(f"❌ Direct tool listing failed: {e}", exc_info=True)
        return False


async def test_plugin_registry():
    """Test the plugin registry directly."""
    logger.info("Testing plugin registry...")
    
    try:
        from cablecar_research.registry import get_registry, initialize_registry
        
        logger.info("Initializing registry...")
        initialize_registry()
        registry = get_registry()
        
        logger.info("Getting plugins...")
        plugins = registry.get_all_plugins()
        logger.info(f"✅ Registry works: {len(plugins)} plugins found")
        
        logger.info("Generating tools...")
        tools = registry.generate_mcp_tools()
        logger.info(f"✅ Tool generation works: {len(tools)} tools generated")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Registry test failed: {e}", exc_info=True)
        return False


async def main():
    """Run debug tests."""
    print("CableCar MCP Server Debug")
    print("=" * 30)
    
    # Test registry first
    print("\n1. Testing Plugin Registry:")
    registry_ok = await test_plugin_registry()
    
    # Test tool listing
    print("\n2. Testing Tool Listing:")
    tools_ok = await test_tool_listing()
    
    if registry_ok and tools_ok:
        print("\n✅ All components working - issue might be in MCP communication")
    else:
        print("\n❌ Found component issues")


if __name__ == "__main__":
    asyncio.run(main())
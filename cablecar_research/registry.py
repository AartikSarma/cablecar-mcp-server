"""
Analysis Registry

Central registry for all CableCar analysis plugins with metadata management,
versioning, and dynamic tool generation for the MCP server.
"""

import logging
from typing import Dict, List, Type, Optional, Any, Callable
from dataclasses import dataclass
from datetime import datetime

from .analysis.base import BaseAnalysis, AnalysisMetadata, AnalysisType
from .plugins import get_plugin_manager

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    """Specification for generating MCP tools from analysis plugins."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Callable


class AnalysisRegistry:
    """
    Central registry for all analysis plugins with MCP integration.
    """
    
    def __init__(self):
        self.plugin_manager = get_plugin_manager()
        self._tools_cache: Dict[str, ToolSpec] = {}
        self._last_discovery = None
        
    def initialize(self, plugin_dirs: Optional[List[str]] = None) -> None:
        """Initialize the registry by discovering all plugins."""
        self.plugin_manager.discover_plugins(plugin_dirs)
        self._generate_tools_cache()
        self._last_discovery = datetime.now()
        logger.info(f"Registry initialized with {len(self.get_all_plugins())} plugins")
    
    def get_all_plugins(self) -> Dict[str, Type[BaseAnalysis]]:
        """Get all registered analysis plugins."""
        return self.plugin_manager.plugins
    
    def get_plugin(self, name: str) -> Optional[Type[BaseAnalysis]]:
        """Get a specific plugin by name."""
        return self.plugin_manager.get_plugin(name)
    
    def get_plugin_metadata(self, name: str) -> Optional[AnalysisMetadata]:
        """Get metadata for a specific plugin."""
        return self.plugin_manager.plugin_metadata.get(name)
    
    def list_plugins_by_type(self, analysis_type: AnalysisType) -> Dict[str, Type[BaseAnalysis]]:
        """List plugins filtered by analysis type."""
        return self.plugin_manager.get_plugins_by_type(analysis_type.value)
    
    def search_plugins(self, query: str) -> Dict[str, Type[BaseAnalysis]]:
        """Search plugins by keyword."""
        return self.plugin_manager.search_plugins(query)
    
    def get_plugin_catalog(self) -> Dict[str, Any]:
        """
        Get comprehensive catalog of all plugins with full metadata.
        
        Returns:
            Dictionary organized by analysis type with plugin details
        """
        catalog = {}
        
        # Group plugins by type
        for analysis_type in AnalysisType:
            type_plugins = self.list_plugins_by_type(analysis_type)
            if type_plugins:
                catalog[analysis_type.value] = {}
                
                for name, plugin_class in type_plugins.items():
                    metadata = self.get_plugin_metadata(name)
                    catalog[analysis_type.value][name] = {
                        'metadata': metadata.to_dict() if metadata else {},
                        'class_name': plugin_class.__name__,
                        'module': plugin_class.__module__,
                        'documentation': self._get_plugin_docs(plugin_class)
                    }
        
        return catalog
    
    def _get_plugin_docs(self, plugin_class: Type[BaseAnalysis]) -> Dict[str, str]:
        """Get documentation for a plugin class."""
        try:
            # Try different initialization patterns
            temp_instance = self._create_temp_instance(plugin_class)
            if hasattr(temp_instance, 'get_documentation'):
                return temp_instance.get_documentation()
            else:
                return {'docstring': plugin_class.__doc__ or 'No documentation available'}
        except Exception as e:
            logger.warning(f"Could not generate docs for {plugin_class.__name__}: {e}")
            return {'error': str(e)}
    
    def generate_mcp_tools(self) -> List[ToolSpec]:
        """
        Generate MCP tool specifications for all registered plugins.
        
        Returns:
            List of tool specifications for the MCP server
        """
        if not self._tools_cache:
            self._generate_tools_cache()
        
        return list(self._tools_cache.values())
    
    def _generate_tools_cache(self) -> None:
        """Generate MCP tools cache from all registered plugins."""
        self._tools_cache.clear()
        
        for name, plugin_class in self.get_all_plugins().items():
            try:
                tool_spec = self._create_tool_spec(name, plugin_class)
                self._tools_cache[name] = tool_spec
            except Exception as e:
                logger.error(f"Failed to create tool spec for {name}: {e}")
    
    def _create_tool_spec(self, name: str, plugin_class: Type[BaseAnalysis]) -> ToolSpec:
        """Create MCP tool specification for a plugin."""
        metadata = self.get_plugin_metadata(name)
        
        # Generate input schema from plugin requirements
        input_schema = self._generate_input_schema(plugin_class)
        
        # Create tool specification
        return ToolSpec(
            name=f"run_{name}",
            description=f"{metadata.display_name}: {metadata.description}" if metadata else f"Run {name} analysis",
            input_schema=input_schema,
            handler=self._create_plugin_handler(name, plugin_class)
        )
    
    def _generate_input_schema(self, plugin_class: Type[BaseAnalysis]) -> Dict[str, Any]:
        """Generate JSON schema for plugin parameters."""
        try:
            # Create temporary instance to get parameter requirements
            temp_instance = self._create_temp_instance(plugin_class)
            requirements = temp_instance.get_required_parameters()
            
            properties = {}
            required = []
            
            # Handle different return formats
            if isinstance(requirements, list):
                # Old format: just a list of required parameter names
                for param_name in requirements:
                    properties[param_name] = {
                        'type': 'string',
                        'description': f'Required parameter: {param_name}'
                    }
                    required.append(param_name)
            elif isinstance(requirements, dict):
                # New format: dictionary with required and optional
                # Add required parameters
                for param_name, param_spec in requirements.get('required', {}).items():
                    properties[param_name] = self._convert_param_to_schema(param_spec)
                    required.append(param_name)
                
                # Add optional parameters  
                for param_name, param_spec in requirements.get('optional', {}).items():
                    properties[param_name] = self._convert_param_to_schema(param_spec)
            
            # Always include output format option
            properties['output_format'] = {
                'type': 'string',
                'enum': ['standard', 'detailed', 'summary', 'publication'],
                'description': 'Format for result output'
            }
            
            schema = {
                'type': 'object',
                'properties': properties
            }
            
            if required:
                schema['required'] = required
            
            return schema
            
        except Exception as e:
            logger.warning(f"Could not generate schema for {plugin_class.__name__}: {e}")
            # Return basic schema
            return {
                'type': 'object',
                'properties': {
                    'output_format': {
                        'type': 'string',
                        'enum': ['standard', 'detailed', 'summary', 'publication'],
                        'description': 'Format for result output'
                    }
                }
            }
    
    def _convert_param_to_schema(self, param_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Convert parameter specification to JSON schema format."""
        schema_prop = {}
        
        param_type = param_spec.get('type', 'string').lower()
        if param_type in ['string', 'str']:
            schema_prop['type'] = 'string'
        elif param_type in ['number', 'float', 'int', 'integer']:
            schema_prop['type'] = 'number'
        elif param_type in ['boolean', 'bool']:
            schema_prop['type'] = 'boolean'
        elif param_type in ['list', 'array']:
            schema_prop['type'] = 'array'
            schema_prop['items'] = {'type': 'string'}  # Default to string items
        else:
            schema_prop['type'] = 'string'  # Default fallback
        
        if 'description' in param_spec:
            schema_prop['description'] = param_spec['description']
        
        if 'example' in param_spec:
            schema_prop['example'] = param_spec['example']
        
        if 'enum' in param_spec:
            schema_prop['enum'] = param_spec['enum']
        
        return schema_prop
    
    def _create_plugin_handler(self, name: str, plugin_class: Type[BaseAnalysis]) -> Callable:
        """Create async handler function for plugin execution."""
        async def plugin_handler(arguments: Dict[str, Any]) -> List[Dict[str, str]]:
            """Execute plugin with given arguments."""
            try:
                # Get current dataset from server state (would be passed in real implementation)
                # For now, this is a placeholder - in real implementation this would
                # access the server's global state to get the current dataset
                df = None  # TODO: Get from server state
                privacy_guard = None  # TODO: Get from server state
                
                # Create plugin instance
                plugin_instance = plugin_class(df, privacy_guard)
                
                # Validate inputs
                validation = plugin_instance.validate_inputs(**arguments)
                if not validation.get('valid', True):
                    error_msg = "Validation failed: " + "; ".join(validation.get('errors', []))
                    return [{"type": "text", "text": error_msg}]
                
                # Run analysis
                results = plugin_instance.run_analysis(**arguments)
                
                # Format results
                output_format = arguments.get('output_format', 'standard')
                formatted_results = plugin_instance.format_results(results, output_format)
                
                return [{"type": "text", "text": formatted_results}]
                
            except Exception as e:
                error_msg = f"Error executing {name}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                return [{"type": "text", "text": error_msg}]
        
        return plugin_handler
    
    def get_tool_handler(self, tool_name: str) -> Optional[Callable]:
        """Get the handler function for a specific tool."""
        # Remove 'run_' prefix to get plugin name
        plugin_name = tool_name.replace('run_', '') if tool_name.startswith('run_') else tool_name
        
        tool_spec = self._tools_cache.get(plugin_name)
        return tool_spec.handler if tool_spec else None
    
    def refresh_if_needed(self) -> bool:
        """Refresh registry if plugins have been modified."""
        try:
            # Simple refresh - in production might check file modification times
            current_plugins = len(self.get_all_plugins())
            self.plugin_manager.reload_plugins()
            self._generate_tools_cache()
            
            new_plugins = len(self.get_all_plugins())
            if new_plugins != current_plugins:
                logger.info(f"Registry refreshed: {current_plugins} -> {new_plugins} plugins")
                return True
                
        except Exception as e:
            logger.error(f"Failed to refresh registry: {e}")
        
        return False
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get statistics about the plugin registry."""
        plugins = self.get_all_plugins()
        metadata_list = [self.get_plugin_metadata(name) for name in plugins.keys()]
        
        # Count by type
        type_counts = {}
        for metadata in metadata_list:
            if metadata:
                type_name = metadata.analysis_type.value
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
        
        # Count by author
        author_counts = {}
        for metadata in metadata_list:
            if metadata:
                author = metadata.author
                author_counts[author] = author_counts.get(author, 0) + 1
        
        return {
            'total_plugins': len(plugins),
            'plugins_by_type': type_counts,
            'plugins_by_author': author_counts,
            'last_discovery': self._last_discovery.isoformat() if self._last_discovery else None,
            'tools_generated': len(self._tools_cache)
        }
    
    def _create_temp_instance(self, plugin_class: Type[BaseAnalysis]):
        """Create temporary plugin instance trying different initialization patterns."""
        # Try different initialization patterns
        init_patterns = [
            lambda: plugin_class(df=None, privacy_guard=None),  # Full BaseAnalysis signature
            lambda: plugin_class(privacy_guard=None),           # Privacy guard only
            lambda: plugin_class(),                             # No arguments
            lambda: plugin_class(None, None),                   # Positional arguments
        ]
        
        for pattern in init_patterns:
            try:
                return pattern()
            except Exception:
                continue
        
        # If all patterns fail, raise the last exception
        return plugin_class(df=None, privacy_guard=None)


# Global registry instance
_registry = None


def get_registry() -> AnalysisRegistry:
    """Get the global analysis registry."""
    global _registry
    if _registry is None:
        _registry = AnalysisRegistry()
    return _registry


def initialize_registry(plugin_dirs: Optional[List[str]] = None) -> None:
    """Initialize the global registry."""
    registry = get_registry()
    registry.initialize(plugin_dirs)
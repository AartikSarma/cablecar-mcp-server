"""
CableCar Analysis Plugin System

Dynamic loading and discovery system for analysis plugins.
"""

import os
import sys
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Type, Optional, Any
import logging

from ..analysis.base import BaseAnalysis, AnalysisMetadata

logger = logging.getLogger(__name__)


class PluginManager:
    """
    Manages discovery and loading of analysis plugins.
    """
    
    def __init__(self):
        self.plugins: Dict[str, Type[BaseAnalysis]] = {}
        self.plugin_metadata: Dict[str, AnalysisMetadata] = {}
        self.plugin_paths: Dict[str, str] = {}
        self._loaded = False
    
    def discover_plugins(self, plugin_dirs: Optional[List[str]] = None) -> None:
        """
        Discover and load all available plugins.
        
        Args:
            plugin_dirs: Additional directories to search for plugins
        """
        if self._loaded:
            return
        
        # Default plugin directories
        default_dirs = [
            Path(__file__).parent / "community",
            Path(__file__).parent / "contrib", 
            Path(__file__).parent / "core"
        ]
        
        # Add user-specified directories
        if plugin_dirs:
            default_dirs.extend([Path(d) for d in plugin_dirs])
        
        # Also check for plugins in the main analysis directory
        analysis_dir = Path(__file__).parent.parent / "analysis"
        if analysis_dir.exists():
            default_dirs.append(analysis_dir)
        
        for plugin_dir in default_dirs:
            if plugin_dir.exists() and plugin_dir.is_dir():
                self._discover_plugins_in_directory(plugin_dir)
        
        self._loaded = True
        logger.info(f"Discovered {len(self.plugins)} analysis plugins")
    
    def _discover_plugins_in_directory(self, plugin_dir: Path) -> None:
        """Discover plugins in a specific directory."""
        for py_file in plugin_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue  # Skip private files
            
            try:
                self._load_plugin_from_file(py_file)
            except Exception as e:
                logger.warning(f"Failed to load plugin from {py_file}: {e}")
    
    def _load_plugin_from_file(self, file_path: Path) -> None:
        """Load a plugin from a Python file."""
        # Create module name from file path
        module_name = f"cablecar_research.plugins.{file_path.stem}"
        
        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create spec for {file_path}")
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        
        # Find BaseAnalysis subclasses in the module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if (obj is not BaseAnalysis and 
                issubclass(obj, BaseAnalysis) and 
                obj.__module__ == module_name):
                
                self._register_plugin(obj, str(file_path))
    
    def _register_plugin(self, plugin_class: Type[BaseAnalysis], file_path: str) -> None:
        """Register a discovered plugin."""
        try:
            # Validate plugin has required metadata
            if not hasattr(plugin_class, 'metadata') or plugin_class.metadata is None:
                logger.warning(f"Plugin {plugin_class.__name__} missing metadata, skipping")
                return
            
            metadata = plugin_class.metadata
            plugin_name = metadata.name
            
            # Check for name conflicts
            if plugin_name in self.plugins:
                logger.warning(f"Plugin name conflict: {plugin_name} already registered")
                return
            
            # Register the plugin
            self.plugins[plugin_name] = plugin_class
            self.plugin_metadata[plugin_name] = metadata
            self.plugin_paths[plugin_name] = file_path
            
            logger.debug(f"Registered plugin: {plugin_name} from {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to register plugin {plugin_class.__name__}: {e}")
    
    def get_plugin(self, name: str) -> Optional[Type[BaseAnalysis]]:
        """Get a plugin by name."""
        if not self._loaded:
            self.discover_plugins()
        return self.plugins.get(name)
    
    def list_plugins(self) -> Dict[str, Dict[str, str]]:
        """List all available plugins with metadata."""
        if not self._loaded:
            self.discover_plugins()
        
        return {
            name: metadata.to_dict() 
            for name, metadata in self.plugin_metadata.items()
        }
    
    def get_plugins_by_type(self, analysis_type: str) -> Dict[str, Type[BaseAnalysis]]:
        """Get all plugins of a specific analysis type."""
        if not self._loaded:
            self.discover_plugins()
        
        filtered = {}
        for name, metadata in self.plugin_metadata.items():
            if metadata.analysis_type.value == analysis_type:
                filtered[name] = self.plugins[name]
        
        return filtered
    
    def search_plugins(self, keyword: str) -> Dict[str, Type[BaseAnalysis]]:
        """Search plugins by keyword in name, description, or keywords."""
        if not self._loaded:
            self.discover_plugins()
        
        keyword = keyword.lower()
        matches = {}
        
        for name, metadata in self.plugin_metadata.items():
            # Search in name, description, and keywords
            searchable_text = [
                metadata.name.lower(),
                metadata.display_name.lower(),
                metadata.description.lower()
            ]
            
            if metadata.keywords:
                searchable_text.extend([k.lower() for k in metadata.keywords])
            
            if any(keyword in text for text in searchable_text):
                matches[name] = self.plugins[name]
        
        return matches
    
    def validate_plugin(self, plugin_class: Type[BaseAnalysis]) -> Dict[str, Any]:
        """
        Validate that a plugin properly implements the BaseAnalysis interface.
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check metadata
        if not hasattr(plugin_class, 'metadata') or plugin_class.metadata is None:
            validation['valid'] = False
            validation['errors'].append("Plugin missing required metadata")
        
        # Check required methods
        required_methods = ['validate_inputs', 'run_analysis', 'format_results']
        for method in required_methods:
            if not hasattr(plugin_class, method):
                validation['valid'] = False
                validation['errors'].append(f"Plugin missing required method: {method}")
        
        # Check if methods are properly implemented (not just abstract)
        try:
            # Create a temporary instance to test
            temp_instance = plugin_class(None, None)
        except Exception as e:
            validation['warnings'].append(f"Could not instantiate plugin for testing: {e}")
        
        return validation
    
    def reload_plugins(self) -> None:
        """Reload all plugins (useful for development)."""
        self.plugins.clear()
        self.plugin_metadata.clear()
        self.plugin_paths.clear()
        self._loaded = False
        self.discover_plugins()


# Global plugin manager instance
plugin_manager = PluginManager()


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    return plugin_manager


def discover_plugins(plugin_dirs: Optional[List[str]] = None) -> None:
    """Discover all available plugins."""
    plugin_manager.discover_plugins(plugin_dirs)


def list_plugins() -> Dict[str, Dict[str, str]]:
    """List all available plugins."""
    return plugin_manager.list_plugins()


def get_plugin(name: str) -> Optional[Type[BaseAnalysis]]:
    """Get a plugin by name."""
    return plugin_manager.get_plugin(name)
"""Integration Engineer for connecting agents with external APIs."""
from typing import Dict, Any, List, Optional
import logging
import os
import json
import importlib
import inspect
from langchain.agents import Tool
from .base_developer import BaseDeveloperAgent

logger = logging.getLogger(__name__)

class IntegrationEngineerAgent(BaseDeveloperAgent):
    """Agent specialized in API integrations."""
    
    def __init__(
        self,
        api_specs_dir: Optional[str] = None,
        **kwargs
    ):
        """Initialize the integration engineer.
        
        Args:
            api_specs_dir: Optional directory with API specifications
            **kwargs: Additional arguments for BaseDeveloperAgent
        """
        super().__init__(specialty="integration-engineer", **kwargs)
        self.api_specs_dir = api_specs_dir
        self.setup_integration_tools()
    
    def setup_integration_tools(self):
        """Set up integration specific tools."""
        self.add_tools([
            Tool(
                name="create_api_client",
                func=self.create_api_client,
                description="Create an API client"
            ),
            Tool(
                name="create_tools",
                func=self.create_tools,
                description="Create tools from API endpoints"
            ),
            Tool(
                name="test_integration",
                func=self.test_integration,
                description="Test API integration"
            ),
            Tool(
                name="generate_api_docs",
                func=self.generate_api_docs,
                description="Generate API documentation"
            ),
            Tool(
                name="analyze_api",
                func=self.analyze_api,
                description="Analyze API usage patterns"
            )
        ])
    
    def create_api_client(
        self,
        api_name: str,
        base_url: str,
        auth_type: str = "token",
        endpoints: List[Dict[str, Any]] = None,
        rate_limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Create an API client.
        
        Args:
            api_name: Name of the API
            base_url: Base URL for the API
            auth_type: Authentication type (token/oauth/basic)
            endpoints: List of API endpoints
            rate_limit: Optional rate limit
        
        Returns:
            Created client details
        """
        try:
            # Create client directory
            client_dir = os.path.join(self.work_dir, "api_clients", api_name.lower())
            os.makedirs(client_dir, exist_ok=True)
            
            # Create client class
            client_code = f"""\"\"\"API client for {api_name}.\"\"\"
import logging
import time
from typing import Dict, Any, Optional
import requests
from ratelimit import limits, sleep_and_retry

logger = logging.getLogger(__name__)

class {api_name}Client:
    \"\"\"Client for interacting with {api_name} API.\"\"\"
    
    def __init__(
        self,
        base_url: str = "{base_url}",
        api_key: Optional[str] = None,
        timeout: int = 30
    ):
        \"\"\"Initialize the client.
        
        Args:
            base_url: API base URL
            api_key: Optional API key
            timeout: Request timeout in seconds
        \"\"\"
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.session = requests.Session()
"""
            
            # Add authentication
            if auth_type == "token":
                client_code += """
        if api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {api_key}"
            })
"""
            elif auth_type == "oauth":
                client_code += """
        if api_key:
            self.session.headers.update({
                "Authorization": f"OAuth {api_key}"
            })
"""
            elif auth_type == "basic":
                client_code += """
        if api_key:
            self.session.auth = ("api", api_key)
"""
            
            # Add rate limiting if specified
            if rate_limit:
                client_code += f"""
    @sleep_and_retry
    @limits(calls={rate_limit}, period=1)
    def _call_api(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        \"\"\"Make a rate-limited API call.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request arguments
        
        Returns:
            API response
        \"\"\"
        url = f"{{self.base_url}}/{{endpoint.lstrip('/')}}"
        response = self.session.request(
            method=method,
            url=url,
            timeout=self.timeout,
            **kwargs
        )
        response.raise_for_status()
        return response.json()
"""
            else:
                client_code += """
    def _call_api(
        self,
        method: str,
        endpoint: str,
        **kwargs
    ) -> Dict[str, Any]:
        \"\"\"Make an API call.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            **kwargs: Additional request arguments
        
        Returns:
            API response
        \"\"\"
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        response = self.session.request(
            method=method,
            url=url,
            timeout=self.timeout,
            **kwargs
        )
        response.raise_for_status()
        return response.json()
"""
            
            # Add endpoint methods
            if endpoints:
                for endpoint in endpoints:
                    method_name = endpoint["name"].lower().replace("-", "_")
                    method_code = f"""
    def {method_name}(
        self,
        {', '.join(f'{p}: str' for p in endpoint.get('parameters', []))},
        **kwargs
    ) -> Dict[str, Any]:
        \"\"\"Call {endpoint['name']} endpoint.
        
        Args:
            {chr(10).join(f'{p}: {p.title()} parameter' for p in endpoint.get('parameters', []))}
            **kwargs: Additional request arguments
        
        Returns:
            API response
        \"\"\"
        return self._call_api(
            method="{endpoint['method']}",
            endpoint=f"{endpoint['path']}",
            **kwargs
        )
"""
                    client_code += method_code
            
            # Save client code
            client_file = os.path.join(client_dir, f"{api_name.lower()}_client.py")
            with open(client_file, "w") as f:
                f.write(client_code)
            
            # Create test file
            test_code = f"""\"\"\"Tests for {api_name} API client.\"\"\"
import pytest
from unittest.mock import Mock, patch
from .{api_name.lower()}_client import {api_name}Client

@pytest.fixture
def client():
    \"\"\"Create a client instance.\"\"\"
    return {api_name}Client(api_key="test-key")

def test_initialization(client):
    \"\"\"Test client initialization.\"\"\"
    assert client.base_url == "{base_url}"
    assert client.api_key == "test-key"
    assert client.timeout == 30

@patch("requests.Session.request")
def test_call_api(mock_request, client):
    \"\"\"Test API call.\"\"\"
    mock_response = Mock()
    mock_response.json.return_value = {{"status": "success"}}
    mock_request.return_value = mock_response
    
    result = client._call_api("GET", "test")
    assert result == {{"status": "success"}}
    mock_request.assert_called_once()
"""
            
            if endpoints:
                for endpoint in endpoints:
                    method_name = endpoint["name"].lower().replace("-", "_")
                    test_code += f"""

@patch("requests.Session.request")
def test_{method_name}(mock_request, client):
    \"\"\"Test {endpoint['name']} endpoint.\"\"\"
    mock_response = Mock()
    mock_response.json.return_value = {{"status": "success"}}
    mock_request.return_value = mock_response
    
    result = client.{method_name}({', '.join(f'"{p}"' for p in endpoint.get('parameters', []))})
    assert result == {{"status": "success"}}
    mock_request.assert_called_once()
"""
            
            test_file = os.path.join(client_dir, f"test_{api_name.lower()}_client.py")
            with open(test_file, "w") as f:
                f.write(test_code)
            
            # Create README
            readme = f"""# {api_name} API Client

## Overview
API client for interacting with {api_name}.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from {api_name.lower()}_client import {api_name}Client

client = {api_name}Client(api_key="your-key")
```

## Authentication
This client uses {auth_type} authentication.

## Rate Limiting
{f'Requests are limited to {rate_limit} per second.' if rate_limit else 'No rate limiting implemented.'}

## Endpoints
{chr(10).join(f'- {e["name"]}: {e["method"]} {e["path"]}' for e in endpoints) if endpoints else 'No endpoints documented.'}

## Testing
```bash
pytest test_{api_name.lower()}_client.py
```
"""
            
            readme_file = os.path.join(client_dir, "README.md")
            with open(readme_file, "w") as f:
                f.write(readme)
            
            # Create requirements
            requirements = """requests>=2.26.0
ratelimit>=2.2.1
pytest>=6.2.0
"""
            
            requirements_file = os.path.join(client_dir, "requirements.txt")
            with open(requirements_file, "w") as f:
                f.write(requirements)
            
            # Commit changes
            self.commit_changes(
                message=f"Create {api_name} API client",
                files=[
                    client_file,
                    test_file,
                    readme_file,
                    requirements_file
                ]
            )
            
            return {
                "status": "success",
                "message": f"API client for {api_name} created successfully",
                "client_dir": client_dir,
                "files": {
                    "client": client_file,
                    "test": test_file,
                    "readme": readme_file,
                    "requirements": requirements_file
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating API client: {e}")
            return {"error": str(e)}
    
    def create_tools(
        self,
        api_name: str,
        endpoints: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Create tools from API endpoints.
        
        Args:
            api_name: Name of the API
            endpoints: List of API endpoints
        
        Returns:
            Created tools
        """
        try:
            # Create tools directory
            tools_dir = os.path.join(self.work_dir, "tools", api_name.lower())
            os.makedirs(tools_dir, exist_ok=True)
            
            # Create tools module
            tools_code = f"""\"\"\"LangChain tools for {api_name} API.\"\"\"
from typing import Dict, Any
from langchain.tools import Tool
from ..api_clients.{api_name.lower()}_client import {api_name}Client

class {api_name}Tools:
    \"\"\"Collection of tools for {api_name} API.\"\"\"
    
    def __init__(self, api_key: str):
        \"\"\"Initialize tools.
        
        Args:
            api_key: API key for authentication
        \"\"\"
        self.client = {api_name}Client(api_key=api_key)
    
    def get_tools(self) -> List[Tool]:
        \"\"\"Get all available tools.
        
        Returns:
            List of tools
        \"\"\"
        return [
"""
            
            # Add tool definitions
            for endpoint in endpoints:
                method_name = endpoint["name"].lower().replace("-", "_")
                tools_code += f"""            Tool(
                name="{method_name}",
                func=self.{method_name},
                description="{endpoint.get('description', f'Call {endpoint["name"]} endpoint')}"
            ),
"""
            
            tools_code += """        ]
"""
            
            # Add tool methods
            for endpoint in endpoints:
                method_name = endpoint["name"].lower().replace("-", "_")
                tools_code += f"""
    def {method_name}(self, *args, **kwargs) -> Dict[str, Any]:
        \"\"\"Tool: {endpoint.get('description', f'Call {endpoint["name"]} endpoint')}
        
        Returns:
            API response
        \"\"\"
        try:
            return self.client.{method_name}(*args, **kwargs)
        except Exception as e:
            return {{"error": str(e)}}
"""
            
            # Save tools code
            tools_file = os.path.join(tools_dir, f"{api_name.lower()}_tools.py")
            with open(tools_file, "w") as f:
                f.write(tools_code)
            
            # Create test file
            test_code = f"""\"\"\"Tests for {api_name} API tools.\"\"\"
import pytest
from unittest.mock import Mock, patch
from .{api_name.lower()}_tools import {api_name}Tools

@pytest.fixture
def tools():
    \"\"\"Create tools instance.\"\"\"
    return {api_name}Tools(api_key="test-key")

def test_initialization(tools):
    \"\"\"Test tools initialization.\"\"\"
    assert tools.client is not None
    assert isinstance(tools.get_tools(), list)
"""
            
            for endpoint in endpoints:
                method_name = endpoint["name"].lower().replace("-", "_")
                test_code += f"""

@patch("{api_name}Client.{method_name}")
def test_{method_name}_tool(mock_method, tools):
    \"\"\"Test {endpoint['name']} tool.\"\"\"
    mock_method.return_value = {{"status": "success"}}
    
    tool = next(t for t in tools.get_tools() if t.name == "{method_name}")
    result = tool.func()
    
    assert result == {{"status": "success"}}
    mock_method.assert_called_once()
"""
            
            test_file = os.path.join(tools_dir, f"test_{api_name.lower()}_tools.py")
            with open(test_file, "w") as f:
                f.write(test_code)
            
            # Create README
            readme = f"""# {api_name} API Tools

## Overview
LangChain tools for interacting with {api_name} API.

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from {api_name.lower()}_tools import {api_name}Tools

tools = {api_name}Tools(api_key="your-key")
available_tools = tools.get_tools()
```

## Available Tools
{chr(10).join(f'- {e["name"]}: {e.get("description", f"Call {e["name"]} endpoint")}' for e in endpoints)}

## Testing
```bash
pytest test_{api_name.lower()}_tools.py
```
"""
            
            readme_file = os.path.join(tools_dir, "README.md")
            with open(readme_file, "w") as f:
                f.write(readme)
            
            # Commit changes
            self.commit_changes(
                message=f"Create {api_name} API tools",
                files=[
                    tools_file,
                    test_file,
                    readme_file
                ]
            )
            
            return {
                "status": "success",
                "message": f"Tools for {api_name} created successfully",
                "tools_dir": tools_dir,
                "files": {
                    "tools": tools_file,
                    "test": test_file,
                    "readme": readme_file
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating tools: {e}")
            return {"error": str(e)}
    
    def test_integration(
        self,
        api_name: str,
        test_cases: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Test API integration.
        
        Args:
            api_name: Name of the API
            test_cases: List of test cases
        
        Returns:
            Test results
        """
        try:
            # Import client and tools
            client_module = importlib.import_module(
                f"api_clients.{api_name.lower()}_client"
            )
            tools_module = importlib.import_module(
                f"tools.{api_name.lower()}_tools"
            )
            
            client_class = getattr(client_module, f"{api_name}Client")
            tools_class = getattr(tools_module, f"{api_name}Tools")
            
            # Create test directory
            test_dir = os.path.join(self.work_dir, "tests", "integration", api_name.lower())
            os.makedirs(test_dir, exist_ok=True)
            
            # Create test file
            test_code = f"""\"\"\"Integration tests for {api_name} API.\"\"\"
import pytest
from {api_name.lower()}_client import {api_name}Client
from {api_name.lower()}_tools import {api_name}Tools

@pytest.fixture
def client():
    \"\"\"Create client instance.\"\"\"
    return {api_name}Client(api_key="test-key")

@pytest.fixture
def tools():
    \"\"\"Create tools instance.\"\"\"
    return {api_name}Tools(api_key="test-key")

"""
            
            for i, case in enumerate(test_cases, 1):
                test_code += f"""
def test_case_{i}(client, tools):
    \"\"\"Test case {i}: {case['description']}\"\"\"
    # Test client
    client_result = client.{case['method']}({case.get('args', '')})
    assert client_result["status"] == "{case.get('expected_status', 'success')}"
    
    # Test tool
    tool = next(t for t in tools.get_tools() if t.name == "{case['method']}")
    tool_result = tool.func({case.get('args', '')})
    assert tool_result["status"] == "{case.get('expected_status', 'success')}"
"""
            
            # Save test file
            test_file = os.path.join(test_dir, f"test_{api_name.lower()}_integration.py")
            with open(test_file, "w") as f:
                f.write(test_code)
            
            # Run tests
            result = self.run_tests(test_file)
            
            # Add test cases to results
            if "error" not in result:
                result["test_cases"] = test_cases
            
            return result
            
        except Exception as e:
            logger.error(f"Error testing integration: {e}")
            return {"error": str(e)}
    
    def generate_api_docs(
        self,
        api_name: str
    ) -> Dict[str, Any]:
        """Generate API documentation.
        
        Args:
            api_name: Name of the API
        
        Returns:
            Generated documentation
        """
        try:
            # Import client and tools
            client_module = importlib.import_module(
                f"api_clients.{api_name.lower()}_client"
            )
            tools_module = importlib.import_module(
                f"tools.{api_name.lower()}_tools"
            )
            
            client_class = getattr(client_module, f"{api_name}Client")
            tools_class = getattr(tools_module, f"{api_name}Tools")
            
            # Create docs directory
            docs_dir = os.path.join(self.work_dir, "docs", "api", api_name.lower())
            os.makedirs(docs_dir, exist_ok=True)
            
            # Generate client documentation
            client_doc = f"""# {api_name} API Client Documentation

## Overview
{inspect.getdoc(client_class)}

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from {api_name.lower()}_client import {api_name}Client

client = {api_name}Client(api_key="your-key")
```

## Methods
"""
            
            for name, method in inspect.getmembers(client_class, inspect.isfunction):
                if not name.startswith("_"):
                    doc = inspect.getdoc(method)
                    sig = inspect.signature(method)
                    client_doc += f"""
### {name}
```python
def {name}{sig}
```

{doc}
"""
            
            client_doc_file = os.path.join(docs_dir, "client.md")
            with open(client_doc_file, "w") as f:
                f.write(client_doc)
            
            # Generate tools documentation
            tools_doc = f"""# {api_name} API Tools Documentation

## Overview
{inspect.getdoc(tools_class)}

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from {api_name.lower()}_tools import {api_name}Tools

tools = {api_name}Tools(api_key="your-key")
available_tools = tools.get_tools()
```

## Tools
"""
            
            tools_instance = tools_class("test-key")
            for tool in tools_instance.get_tools():
                tools_doc += f"""
### {tool.name}
{tool.description}

```python
{inspect.getsource(getattr(tools_instance, tool.name))}
```
"""
            
            tools_doc_file = os.path.join(docs_dir, "tools.md")
            with open(tools_doc_file, "w") as f:
                f.write(tools_doc)
            
            # Generate index
            index = f"""# {api_name} API Integration Documentation

## Contents
- [Client Documentation](client.md)
- [Tools Documentation](tools.md)

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up client:
   ```python
   from {api_name.lower()}_client import {api_name}Client
   
   client = {api_name}Client(api_key="your-key")
   ```

3. Use tools:
   ```python
   from {api_name.lower()}_tools import {api_name}Tools
   
   tools = {api_name}Tools(api_key="your-key")
   available_tools = tools.get_tools()
   ```

## Testing
```bash
# Run client tests
pytest test_{api_name.lower()}_client.py

# Run tools tests
pytest test_{api_name.lower()}_tools.py

# Run integration tests
pytest tests/integration/{api_name.lower()}/
```
"""
            
            index_file = os.path.join(docs_dir, "index.md")
            with open(index_file, "w") as f:
                f.write(index)
            
            # Commit changes
            self.commit_changes(
                message=f"Generate {api_name} API documentation",
                files=[
                    client_doc_file,
                    tools_doc_file,
                    index_file
                ]
            )
            
            return {
                "status": "success",
                "message": f"Documentation for {api_name} generated successfully",
                "docs_dir": docs_dir,
                "files": {
                    "client": client_doc_file,
                    "tools": tools_doc_file,
                    "index": index_file
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating documentation: {e}")
            return {"error": str(e)}
    
    def analyze_api(
        self,
        api_name: str
    ) -> Dict[str, Any]:
        """Analyze API usage patterns.
        
        Args:
            api_name: Name of the API
        
        Returns:
            Analysis results
        """
        try:
            # Import client and tools
            client_module = importlib.import_module(
                f"api_clients.{api_name.lower()}_client"
            )
            tools_module = importlib.import_module(
                f"tools.{api_name.lower()}_tools"
            )
            
            client_class = getattr(client_module, f"{api_name}Client")
            tools_class = getattr(tools_module, f"{api_name}Tools")
            
            # Analyze client
            client_methods = [
                name for name, _ in inspect.getmembers(client_class, inspect.isfunction)
                if not name.startswith("_")
            ]
            
            # Analyze tools
            tools_instance = tools_class("test-key")
            available_tools = tools_instance.get_tools()
            
            # Check coverage
            missing_tools = set(client_methods) - set(t.name for t in available_tools)
            
            # Generate analysis
            analysis = {
                "api_name": api_name,
                "client_methods": len(client_methods),
                "available_tools": len(available_tools),
                "coverage": len(available_tools) / len(client_methods) * 100,
                "missing_tools": list(missing_tools),
                "patterns": {
                    "authentication": inspect.getsource(client_class.__init__),
                    "error_handling": len([
                        1 for tool in available_tools
                        if "try:" in inspect.getsource(
                            getattr(tools_instance, tool.name)
                        )
                    ]),
                    "rate_limiting": hasattr(client_class, "_call_api") and "@limits" in inspect.getsource(client_class._call_api)
                }
            }
            
            # Save analysis
            analysis_file = os.path.join(
                self.work_dir,
                "analysis",
                f"{api_name.lower()}_analysis.json"
            )
            os.makedirs(os.path.dirname(analysis_file), exist_ok=True)
            
            with open(analysis_file, "w") as f:
                json.dump(analysis, f, indent=2)
            
            # Generate recommendations
            recommendations = []
            
            if missing_tools:
                recommendations.append({
                    "type": "coverage",
                    "message": f"Create tools for: {', '.join(missing_tools)}"
                })
            
            if analysis["patterns"]["error_handling"] < len(available_tools):
                recommendations.append({
                    "type": "error_handling",
                    "message": "Add error handling to all tools"
                })
            
            if not analysis["patterns"]["rate_limiting"]:
                recommendations.append({
                    "type": "rate_limiting",
                    "message": "Consider adding rate limiting"
                })
            
            analysis["recommendations"] = recommendations
            
            return {
                "status": "success",
                "message": f"API {api_name} analyzed successfully",
                "analysis": analysis,
                "analysis_file": analysis_file
            }
            
        except Exception as e:
            logger.error(f"Error analyzing API: {e}")
            return {"error": str(e)}
    
    def get_api_specs_dir(self) -> Optional[str]:
        """Get the API specifications directory.
        
        Returns:
            API specifications directory or None
        """
        return self.api_specs_dir
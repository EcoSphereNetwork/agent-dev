"""Doc Writer Agent for creating and maintaining documentation."""
from typing import Dict, Any, List, Optional
import logging
import os
import json
import ast
import inspect
import importlib
from langchain.agents import Tool
from .base_developer import BaseDeveloperAgent

logger = logging.getLogger(__name__)

class DocWriterAgent(BaseDeveloperAgent):
    """Agent specialized in documentation."""
    
    def __init__(
        self,
        doc_style: str = "google",
        doc_format: str = "markdown",
        **kwargs
    ):
        """Initialize the doc writer agent.
        
        Args:
            doc_style: Documentation style (google/numpy/sphinx)
            doc_format: Documentation format (markdown/rst)
            **kwargs: Additional arguments for BaseDeveloperAgent
        """
        super().__init__(specialty="doc-writer", **kwargs)
        self.doc_style = doc_style
        self.doc_format = doc_format
        self.setup_doc_tools()
    
    def setup_doc_tools(self):
        """Set up documentation specific tools."""
        self.add_tools([
            Tool(
                name="analyze_docstrings",
                func=self.analyze_docstrings,
                description="Analyze code docstrings"
            ),
            Tool(
                name="generate_docstrings",
                func=self.generate_docstrings,
                description="Generate code docstrings"
            ),
            Tool(
                name="create_readme",
                func=self.create_readme,
                description="Create README documentation"
            ),
            Tool(
                name="create_api_docs",
                func=self.create_api_docs,
                description="Create API documentation"
            ),
            Tool(
                name="update_docs",
                func=self.update_docs,
                description="Update documentation"
            )
        ])
    
    def analyze_docstrings(
        self,
        file_path: str
    ) -> Dict[str, Any]:
        """Analyze code docstrings.
        
        Args:
            file_path: Path to Python file
        
        Returns:
            Analysis results
        """
        try:
            if not os.path.exists(file_path):
                return {
                    "status": "error",
                    "error": f"File {file_path} not found"
                }
            
            # Parse file
            with open(file_path, "r") as f:
                code = f.read()
            tree = ast.parse(code)
            
            # Analyze docstrings
            analysis = {
                "module_docstring": ast.get_docstring(tree) is not None,
                "classes": {},
                "functions": {},
                "missing_docstrings": [],
                "incomplete_docstrings": []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    docstring = ast.get_docstring(node)
                    analysis["classes"][node.name] = {
                        "has_docstring": docstring is not None,
                        "docstring": docstring,
                        "methods": {}
                    }
                    
                    # Check method docstrings
                    for method in [n for n in node.body if isinstance(n, ast.FunctionDef)]:
                        method_doc = ast.get_docstring(method)
                        analysis["classes"][node.name]["methods"][method.name] = {
                            "has_docstring": method_doc is not None,
                            "docstring": method_doc
                        }
                        
                        if not method_doc:
                            analysis["missing_docstrings"].append(
                                f"{node.name}.{method.name}"
                            )
                        elif self._is_incomplete_docstring(method_doc):
                            analysis["incomplete_docstrings"].append(
                                f"{node.name}.{method.name}"
                            )
                
                elif isinstance(node, ast.FunctionDef):
                    docstring = ast.get_docstring(node)
                    analysis["functions"][node.name] = {
                        "has_docstring": docstring is not None,
                        "docstring": docstring
                    }
                    
                    if not docstring:
                        analysis["missing_docstrings"].append(node.name)
                    elif self._is_incomplete_docstring(docstring):
                        analysis["incomplete_docstrings"].append(node.name)
            
            # Calculate coverage
            total_items = (
                1 +  # Module
                len(analysis["classes"]) +
                sum(len(c["methods"]) for c in analysis["classes"].values()) +
                len(analysis["functions"])
            )
            
            documented_items = (
                int(analysis["module_docstring"]) +
                sum(1 for c in analysis["classes"].values() if c["has_docstring"]) +
                sum(
                    sum(1 for m in c["methods"].values() if m["has_docstring"])
                    for c in analysis["classes"].values()
                ) +
                sum(1 for f in analysis["functions"].values() if f["has_docstring"])
            )
            
            analysis["coverage"] = (documented_items / total_items * 100) if total_items > 0 else 100
            
            # Save analysis
            analysis_dir = os.path.join(self.work_dir, "doc_analysis")
            os.makedirs(analysis_dir, exist_ok=True)
            
            analysis_file = os.path.join(
                analysis_dir,
                f"{os.path.basename(file_path)}.docstring.json"
            )
            
            with open(analysis_file, "w") as f:
                json.dump(analysis, f, indent=2)
            
            return {
                "status": "success",
                "message": f"Docstring analysis completed for {file_path}",
                "analysis": analysis,
                "analysis_file": analysis_file
            }
            
        except Exception as e:
            logger.error(f"Error analyzing docstrings: {e}")
            return {"error": str(e)}
    
    def generate_docstrings(
        self,
        file_path: str,
        analysis_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate code docstrings.
        
        Args:
            file_path: Path to Python file
            analysis_file: Optional path to analysis file
        
        Returns:
            Generated docstrings
        """
        try:
            # Get or create analysis
            if analysis_file and os.path.exists(analysis_file):
                with open(analysis_file, "r") as f:
                    analysis = json.load(f)
            else:
                result = self.analyze_docstrings(file_path)
                if "error" in result:
                    return result
                analysis = result["analysis"]
            
            # Read file
            with open(file_path, "r") as f:
                code = f.read()
            
            # Generate docstrings
            improved_code = code
            generated_docs = []
            
            # Module docstring
            if not analysis["module_docstring"]:
                # Use LLM to generate module docstring
                prompt = f"""Generate a module docstring for this code in {self.doc_style} style:
{code}

Return only the docstring without quotes."""
                
                result = self.run(prompt)
                if "error" not in result:
                    docstring = f'"""{result["result"]}"""'
                    improved_code = docstring + "\n" + improved_code
                    generated_docs.append("module")
            
            # Class and method docstrings
            for class_name, class_info in analysis["classes"].items():
                if not class_info["has_docstring"]:
                    # Use LLM to generate class docstring
                    prompt = f"""Generate a class docstring for this class in {self.doc_style} style:

class {class_name}:
    {class_info["docstring"] or "pass"}

Return only the docstring without quotes."""
                    
                    result = self.run(prompt)
                    if "error" not in result:
                        docstring = f'    """{result["result"]}"""'
                        improved_code = improved_code.replace(
                            f"class {class_name}:",
                            f"class {class_name}:\n{docstring}"
                        )
                        generated_docs.append(f"class {class_name}")
                
                for method_name, method_info in class_info["methods"].items():
                    if not method_info["has_docstring"]:
                        # Use LLM to generate method docstring
                        prompt = f"""Generate a method docstring for this method in {self.doc_style} style:

def {method_name}(self):
    {method_info["docstring"] or "pass"}

Return only the docstring without quotes."""
                        
                        result = self.run(prompt)
                        if "error" not in result:
                            docstring = f'        """{result["result"]}"""'
                            improved_code = improved_code.replace(
                                f"def {method_name}(self):",
                                f"def {method_name}(self):\n{docstring}"
                            )
                            generated_docs.append(f"{class_name}.{method_name}")
            
            # Function docstrings
            for func_name, func_info in analysis["functions"].items():
                if not func_info["has_docstring"]:
                    # Use LLM to generate function docstring
                    prompt = f"""Generate a function docstring for this function in {self.doc_style} style:

def {func_name}():
    {func_info["docstring"] or "pass"}

Return only the docstring without quotes."""
                    
                    result = self.run(prompt)
                    if "error" not in result:
                        docstring = f'    """{result["result"]}"""'
                        improved_code = improved_code.replace(
                            f"def {func_name}():",
                            f"def {func_name}():\n{docstring}"
                        )
                        generated_docs.append(f"function {func_name}")
            
            # Save improved code
            with open(file_path, "w") as f:
                f.write(improved_code)
            
            # Commit changes
            self.commit_changes(
                message=f"""Add docstrings to {os.path.basename(file_path)}

Generated documentation for:
{chr(10).join(f'- {doc}' for doc in generated_docs)}""",
                files=[file_path]
            )
            
            return {
                "status": "success",
                "message": f"Docstrings generated for {file_path}",
                "generated_docs": generated_docs
            }
            
        except Exception as e:
            logger.error(f"Error generating docstrings: {e}")
            return {"error": str(e)}
    
    def create_readme(
        self,
        project_dir: str,
        sections: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Create README documentation.
        
        Args:
            project_dir: Project directory
            sections: Optional list of sections to include
        
        Returns:
            Generated README
        """
        try:
            if not os.path.exists(project_dir):
                return {
                    "status": "error",
                    "error": f"Project directory {project_dir} not found"
                }
            
            # Default sections
            if not sections:
                sections = [
                    "overview",
                    "installation",
                    "usage",
                    "api",
                    "development",
                    "testing",
                    "contributing",
                    "license"
                ]
            
            # Read project info
            setup_file = os.path.join(project_dir, "setup.py")
            if os.path.exists(setup_file):
                with open(setup_file, "r") as f:
                    setup_code = f.read()
                setup_info = {}
                exec(setup_code, {}, setup_info)
            else:
                setup_info = {}
            
            # Generate README
            if self.doc_format == "markdown":
                readme = f"""# {setup_info.get('name', os.path.basename(project_dir))}

{setup_info.get('description', 'Project description')}

"""
                
                if "overview" in sections:
                    readme += """## Overview

"""
                    # Use LLM to generate overview
                    prompt = f"""Generate an overview section for this project:
Project name: {setup_info.get('name', os.path.basename(project_dir))}
Description: {setup_info.get('description', '')}
Keywords: {setup_info.get('keywords', [])}

Return a concise overview paragraph."""
                    
                    result = self.run(prompt)
                    if "error" not in result:
                        readme += result["result"] + "\n\n"
                
                if "installation" in sections:
                    readme += """## Installation

```bash
pip install -r requirements.txt
```

"""
                
                if "usage" in sections:
                    readme += """## Usage

```python
# Add usage examples
```

"""
                
                if "api" in sections:
                    readme += """## API Documentation

See [API.md](docs/API.md) for detailed API documentation.

"""
                
                if "development" in sections:
                    readme += """## Development

### Setup
```bash
# Clone repository
git clone https://github.com/user/repo.git
cd repo

# Install dependencies
pip install -r requirements-dev.txt
```

### Code Style
This project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting
- mypy for type checking

"""
                
                if "testing" in sections:
                    readme += """## Testing

```bash
# Run tests
pytest

# Run tests with coverage
pytest --cov=src tests/
```

"""
                
                if "contributing" in sections:
                    readme += """## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

"""
                
                if "license" in sections:
                    readme += f"""## License

{setup_info.get('license', 'MIT')} - See [LICENSE](LICENSE) for details
"""
                
            else:  # RST format
                readme = f"""{setup_info.get('name', os.path.basename(project_dir))}
{'=' * len(setup_info.get('name', os.path.basename(project_dir)))}

{setup_info.get('description', 'Project description')}

"""
                
                if "overview" in sections:
                    readme += """Overview
--------

"""
                    # Use LLM to generate overview
                    prompt = f"""Generate an overview section for this project:
Project name: {setup_info.get('name', os.path.basename(project_dir))}
Description: {setup_info.get('description', '')}
Keywords: {setup_info.get('keywords', [])}

Return a concise overview paragraph."""
                    
                    result = self.run(prompt)
                    if "error" not in result:
                        readme += result["result"] + "\n\n"
                
                if "installation" in sections:
                    readme += """Installation
------------

.. code-block:: bash

    pip install -r requirements.txt

"""
                
                if "usage" in sections:
                    readme += """Usage
-----

.. code-block:: python

    # Add usage examples

"""
                
                if "api" in sections:
                    readme += """API Documentation
----------------

See `API.rst <docs/API.rst>`_ for detailed API documentation.

"""
                
                if "development" in sections:
                    readme += """Development
-----------

Setup
~~~~~

.. code-block:: bash

    # Clone repository
    git clone https://github.com/user/repo.git
    cd repo

    # Install dependencies
    pip install -r requirements-dev.txt

Code Style
~~~~~~~~~
This project uses:

* Black for code formatting
* isort for import sorting
* flake8 for linting
* mypy for type checking

"""
                
                if "testing" in sections:
                    readme += """Testing
-------

.. code-block:: bash

    # Run tests
    pytest

    # Run tests with coverage
    pytest --cov=src tests/

"""
                
                if "contributing" in sections:
                    readme += """Contributing
------------

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

"""
                
                if "license" in sections:
                    readme += f"""License
-------

{setup_info.get('license', 'MIT')} - See `LICENSE <LICENSE>`_ for details
"""
            
            # Save README
            readme_file = os.path.join(
                project_dir,
                f"README.{'md' if self.doc_format == 'markdown' else 'rst'}"
            )
            
            with open(readme_file, "w") as f:
                f.write(readme)
            
            # Commit changes
            self.commit_changes(
                message=f"""Update README

Added sections:
{chr(10).join(f'- {section}' for section in sections)}""",
                files=[readme_file]
            )
            
            return {
                "status": "success",
                "message": "README generated successfully",
                "readme_file": readme_file,
                "sections": sections
            }
            
        except Exception as e:
            logger.error(f"Error creating README: {e}")
            return {"error": str(e)}
    
    def create_api_docs(
        self,
        module_path: str,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create API documentation.
        
        Args:
            module_path: Path to Python module
            output_dir: Optional output directory
        
        Returns:
            Generated API documentation
        """
        try:
            if not os.path.exists(module_path):
                return {
                    "status": "error",
                    "error": f"Module {module_path} not found"
                }
            
            # Import module
            module_name = os.path.splitext(os.path.basename(module_path))[0]
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Set output directory
            if not output_dir:
                output_dir = os.path.join(os.path.dirname(module_path), "docs")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate API documentation
            if self.doc_format == "markdown":
                api_doc = f"""# {module_name} API Documentation

## Overview
{module.__doc__ or 'No module description available.'}

"""
                
                # Document classes
                classes = inspect.getmembers(module, inspect.isclass)
                if classes:
                    api_doc += "## Classes\n\n"
                    
                    for name, cls in classes:
                        if cls.__module__ == module_name:
                            api_doc += f"""### {name}
{cls.__doc__ or 'No description available.'}

#### Methods
"""
                            
                            methods = inspect.getmembers(cls, inspect.isfunction)
                            for method_name, method in methods:
                                if not method_name.startswith("_"):
                                    sig = inspect.signature(method)
                                    api_doc += f"""##### `{method_name}{sig}`
{method.__doc__ or 'No description available.'}

"""
                
                # Document functions
                functions = inspect.getmembers(module, inspect.isfunction)
                if functions:
                    api_doc += "## Functions\n\n"
                    
                    for name, func in functions:
                        if func.__module__ == module_name:
                            sig = inspect.signature(func)
                            api_doc += f"""### `{name}{sig}`
{func.__doc__ or 'No description available.'}

"""
                
            else:  # RST format
                api_doc = f"""{module_name} API Documentation
{'=' * (len(module_name) + 18)}

Overview
--------
{module.__doc__ or 'No module description available.'}

"""
                
                # Document classes
                classes = inspect.getmembers(module, inspect.isclass)
                if classes:
                    api_doc += "Classes\n-------\n\n"
                    
                    for name, cls in classes:
                        if cls.__module__ == module_name:
                            api_doc += f"""{name}
{'~' * len(name)}
{cls.__doc__ or 'No description available.'}

Methods
~~~~~~~
"""
                            
                            methods = inspect.getmembers(cls, inspect.isfunction)
                            for method_name, method in methods:
                                if not method_name.startswith("_"):
                                    sig = inspect.signature(method)
                                    api_doc += f"""``{method_name}{sig}``
{'^' * (len(method_name) + len(str(sig)))}
{method.__doc__ or 'No description available.'}

"""
                
                # Document functions
                functions = inspect.getmembers(module, inspect.isfunction)
                if functions:
                    api_doc += "Functions\n---------\n\n"
                    
                    for name, func in functions:
                        if func.__module__ == module_name:
                            sig = inspect.signature(func)
                            api_doc += f"""``{name}{sig}``
{'~' * (len(name) + len(str(sig)))}
{func.__doc__ or 'No description available.'}

"""
            
            # Save API documentation
            api_doc_file = os.path.join(
                output_dir,
                f"API.{'md' if self.doc_format == 'markdown' else 'rst'}"
            )
            
            with open(api_doc_file, "w") as f:
                f.write(api_doc)
            
            # Commit changes
            self.commit_changes(
                message=f"Add API documentation for {module_name}",
                files=[api_doc_file]
            )
            
            return {
                "status": "success",
                "message": f"API documentation generated for {module_name}",
                "api_doc_file": api_doc_file
            }
            
        except Exception as e:
            logger.error(f"Error creating API documentation: {e}")
            return {"error": str(e)}
    
    def update_docs(
        self,
        file_path: str,
        doc_file: str
    ) -> Dict[str, Any]:
        """Update documentation.
        
        Args:
            file_path: Path to Python file
            doc_file: Path to documentation file
        
        Returns:
            Status of the operation
        """
        try:
            if not os.path.exists(file_path):
                return {
                    "status": "error",
                    "error": f"File {file_path} not found"
                }
            
            if not os.path.exists(doc_file):
                return {
                    "status": "error",
                    "error": f"Documentation file {doc_file} not found"
                }
            
            # Read files
            with open(file_path, "r") as f:
                code = f.read()
            
            with open(doc_file, "r") as f:
                doc = f.read()
            
            # Use LLM to update documentation
            prompt = f"""Update this documentation to match the current code:

Code:
{code}

Current documentation:
{doc}

Return only the updated documentation."""
            
            result = self.run(prompt)
            if "error" in result:
                return result
            
            # Save updated documentation
            with open(doc_file, "w") as f:
                f.write(result["result"])
            
            # Commit changes
            self.commit_changes(
                message=f"Update documentation for {os.path.basename(file_path)}",
                files=[doc_file]
            )
            
            return {
                "status": "success",
                "message": f"Documentation updated for {file_path}",
                "doc_file": doc_file
            }
            
        except Exception as e:
            logger.error(f"Error updating documentation: {e}")
            return {"error": str(e)}
    
    def _is_incomplete_docstring(self, docstring: str) -> bool:
        """Check if a docstring is incomplete."""
        if not docstring:
            return True
        
        if self.doc_style == "google":
            return not (
                "Args:" in docstring or
                "Returns:" in docstring or
                "Raises:" in docstring
            )
        elif self.doc_style == "numpy":
            return not (
                "Parameters" in docstring or
                "Returns" in docstring or
                "Raises" in docstring
            )
        else:  # sphinx
            return not (
                ":param" in docstring or
                ":return:" in docstring or
                ":raises:" in docstring
            )
    
    def get_doc_style(self) -> str:
        """Get the documentation style.
        
        Returns:
            Documentation style
        """
        return self.doc_style
    
    def get_doc_format(self) -> str:
        """Get the documentation format.
        
        Returns:
            Documentation format
        """
        return self.doc_format
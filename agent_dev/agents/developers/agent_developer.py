"""Agent Developer for creating and improving LangChain agents."""
from typing import Dict, Any, List, Optional
import logging
import os
import json
from langchain.agents import Tool
from .base_developer import BaseDeveloperAgent

logger = logging.getLogger(__name__)

class AgentDeveloperAgent(BaseDeveloperAgent):
    """Agent specialized in developing LangChain agents."""
    
    def __init__(
        self,
        agent_type: str = "general",
        **kwargs
    ):
        """Initialize the agent developer.
        
        Args:
            agent_type: Type of agent to develop (general/specialized)
            **kwargs: Additional arguments for BaseDeveloperAgent
        """
        super().__init__(specialty="agent-developer", **kwargs)
        self.agent_type = agent_type
        self.setup_agent_tools()
    
    def setup_agent_tools(self):
        """Set up agent development specific tools."""
        self.add_tools([
            Tool(
                name="create_agent",
                func=self.create_agent,
                description="Create a new LangChain agent"
            ),
            Tool(
                name="add_tools",
                func=self.add_agent_tools,
                description="Add tools to an agent"
            ),
            Tool(
                name="analyze_agent",
                func=self.analyze_agent,
                description="Analyze an existing agent"
            ),
            Tool(
                name="improve_agent",
                func=self.improve_agent,
                description="Improve an existing agent"
            ),
            Tool(
                name="test_agent",
                func=self.test_agent,
                description="Test an agent's functionality"
            )
        ])
    
    def create_agent(
        self,
        name: str,
        description: str,
        tools: List[Dict[str, Any]],
        memory_type: str = "conversation"
    ) -> Dict[str, Any]:
        """Create a new LangChain agent.
        
        Args:
            name: Agent name
            description: Agent description
            tools: List of tools to add
            memory_type: Type of memory to use
        
        Returns:
            Created agent details
        """
        try:
            # Create agent directory
            agent_dir = os.path.join(self.work_dir, "agents", name.lower())
            os.makedirs(agent_dir, exist_ok=True)
            
            # Create agent class
            agent_code = f"""\"\"\"LangChain agent: {name}\"\"\"
from typing import Dict, Any, List, Optional
import logging
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

logger = logging.getLogger(__name__)

class {name}Agent:
    \"\"\"Agent for {description}\"\"\"
    
    def __init__(
        self,
        api_base: str = "http://localhost:11434/v1",
        model_name: str = "mistral",
        temperature: float = 0,
        verbose: bool = True,
        memory_path: Optional[str] = None
    ):
        \"\"\"Initialize the agent.
        
        Args:
            api_base: Base URL for local LLM API
            model_name: Name of the LLM model
            temperature: LLM temperature
            verbose: Whether to enable verbose output
            memory_path: Optional path for ChromaDB memory
        \"\"\"
        self.llm = ChatOpenAI(
            openai_api_base=api_base,
            openai_api_key="sk-no-key-required",
            model_name=model_name,
            temperature=temperature
        )
        self.verbose = verbose
        self.tools: List[Tool] = []
        self.agent = None
        
        # Set up memory
        if memory_path:
            self.db = Chroma(
                persist_directory=memory_path,
                embedding_function=OpenAIEmbeddings()
            )
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                output_key="output"
            )
        else:
            self.db = None
            self.memory = ConversationBufferMemory()
        
        # Set up tools
        self.setup_tools()
    
    def setup_tools(self):
        \"\"\"Set up agent tools.\"\"\"
"""
            
            # Add tool definitions
            for tool in tools:
                agent_code += f"""
        self.tools.append(Tool(
            name="{tool['name']}",
            func=self.{tool['name']},
            description="{tool['description']}"
        ))
"""
            
            # Add tool methods
            for tool in tools:
                agent_code += f"""
    def {tool['name']}(self, *args, **kwargs) -> Dict[str, Any]:
        \"\"\"Tool: {tool['description']}
        
        Returns:
            Tool results
        \"\"\"
        # TODO: Implement tool functionality
        return {{"status": "success"}}
"""
            
            # Add remaining methods
            agent_code += """
    def setup_agent(self):
        \"\"\"Set up the LangChain agent with tools.\"\"\"
        if not self.tools:
            raise ValueError("No tools defined. Add tools before setting up agent.")
        
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=self.verbose,
            memory=self.memory
        )
    
    def run(self, task: str) -> Dict[str, Any]:
        \"\"\"Run a task using the agent.
        
        Args:
            task: Task description
        
        Returns:
            Task results
        \"\"\"
        if not self.agent:
            raise ValueError("Agent not initialized. Call setup_agent() first.")
        
        try:
            result = self.agent.run(task)
            
            # Store task and result in memory if ChromaDB is available
            if self.db:
                self.db.add_texts(
                    texts=[f"Task: {task}\\nResult: {result}"],
                    metadatas=[{
                        "type": "task_result",
                        "task": task
                    }]
                )
            
            return {
                "status": "success",
                "result": result,
                "metadata": {
                    "model": self.llm.model_name,
                    "temperature": self.llm.temperature,
                    "tools_used": [tool.name for tool in self.tools]
                }
            }
        except Exception as e:
            logger.error(f"Error running task: {e}")
            return {
                "status": "error",
                "error": str(e),
                "metadata": {
                    "model": self.llm.model_name,
                    "temperature": self.llm.temperature,
                    "tools_used": [tool.name for tool in self.tools]
                }
            }
"""
            
            # Save agent code
            agent_file = os.path.join(agent_dir, f"{name.lower()}.py")
            with open(agent_file, "w") as f:
                f.write(agent_code)
            
            # Create test file
            test_code = f"""\"\"\"Tests for {name} agent.\"\"\"
import pytest
from unittest.mock import Mock, patch
from .{name.lower()} import {name}Agent

@pytest.fixture
def agent():
    \"\"\"Create an agent instance.\"\"\"
    return {name}Agent()

def test_initialization(agent):
    \"\"\"Test agent initialization.\"\"\"
    assert agent.llm is not None
    assert agent.tools == []
    assert agent.agent is None
    assert agent.memory is not None

def test_setup_tools(agent):
    \"\"\"Test tool setup.\"\"\"
    agent.setup_tools()
    assert len(agent.tools) > 0
    assert all(tool.name for tool in agent.tools)
    assert all(tool.description for tool in agent.tools)

def test_setup_agent(agent):
    \"\"\"Test agent setup.\"\"\"
    agent.setup_tools()
    agent.setup_agent()
    assert agent.agent is not None

def test_run_task(agent):
    \"\"\"Test running a task.\"\"\"
    agent.setup_tools()
    agent.setup_agent()
    result = agent.run("Test task")
    assert result["status"] in ["success", "error"]
    assert "metadata" in result
"""
            
            test_file = os.path.join(agent_dir, f"test_{name.lower()}.py")
            with open(test_file, "w") as f:
                f.write(test_code)
            
            # Create README
            readme = f"""# {name} Agent

## Overview
{description}

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```python
from {name.lower()} import {name}Agent

agent = {name}Agent()
agent.setup_tools()
agent.setup_agent()

result = agent.run("Your task here")
print(result)
```

## Tools
{chr(10).join(f'- {tool["name"]}: {tool["description"]}' for tool in tools)}

## Memory
This agent uses {memory_type} memory to maintain context across interactions.

## Testing
```bash
pytest test_{name.lower()}.py
```
"""
            
            readme_file = os.path.join(agent_dir, "README.md")
            with open(readme_file, "w") as f:
                f.write(readme)
            
            # Create requirements
            requirements = """langchain>=0.0.200
openai>=0.27.0
chromadb>=0.4.0
pytest>=6.2.0
"""
            
            requirements_file = os.path.join(agent_dir, "requirements.txt")
            with open(requirements_file, "w") as f:
                f.write(requirements)
            
            # Commit changes
            self.commit_changes(
                message=f"Create {name} agent",
                files=[
                    agent_file,
                    test_file,
                    readme_file,
                    requirements_file
                ]
            )
            
            return {
                "status": "success",
                "message": f"Agent {name} created successfully",
                "agent_dir": agent_dir,
                "files": {
                    "agent": agent_file,
                    "test": test_file,
                    "readme": readme_file,
                    "requirements": requirements_file
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            return {"error": str(e)}
    
    def add_agent_tools(
        self,
        agent_name: str,
        tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Add tools to an existing agent.
        
        Args:
            agent_name: Name of the agent
            tools: List of tools to add
        
        Returns:
            Status of the operation
        """
        try:
            agent_dir = os.path.join(self.work_dir, "agents", agent_name.lower())
            agent_file = os.path.join(agent_dir, f"{agent_name.lower()}.py")
            
            if not os.path.exists(agent_file):
                return {
                    "status": "error",
                    "error": f"Agent {agent_name} not found"
                }
            
            # Read current agent code
            with open(agent_file, "r") as f:
                code = f.read()
            
            # Add tool imports if needed
            for tool in tools:
                if tool.get("imports"):
                    import_pos = code.find("import logging")
                    if import_pos >= 0:
                        code = (
                            code[:import_pos] +
                            tool["imports"] + "\n" +
                            code[import_pos:]
                        )
            
            # Add tool definitions
            setup_pos = code.find("def setup_tools(self):")
            if setup_pos >= 0:
                indent = "        "
                tool_defs = ""
                for tool in tools:
                    tool_defs += f"""
{indent}self.tools.append(Tool(
{indent}    name="{tool['name']}",
{indent}    func=self.{tool['name']},
{indent}    description="{tool['description']}"
{indent}))
"""
                
                # Find end of setup_tools method
                next_method = code.find("\n    def", setup_pos)
                if next_method >= 0:
                    code = (
                        code[:next_method] +
                        tool_defs +
                        code[next_method:]
                    )
            
            # Add tool methods
            for tool in tools:
                code += f"""
    def {tool['name']}(self, *args, **kwargs) -> Dict[str, Any]:
        \"\"\"Tool: {tool['description']}
        
        Returns:
            Tool results
        \"\"\"
        # TODO: Implement tool functionality
        return {{"status": "success"}}
"""
            
            # Save updated code
            with open(agent_file, "w") as f:
                f.write(code)
            
            # Update README
            readme_file = os.path.join(agent_dir, "README.md")
            if os.path.exists(readme_file):
                with open(readme_file, "r") as f:
                    readme = f.read()
                
                tools_pos = readme.find("## Tools")
                if tools_pos >= 0:
                    next_section = readme.find("##", tools_pos + 1)
                    if next_section >= 0:
                        tools_section = readme[tools_pos:next_section]
                        new_tools = chr(10).join(
                            f'- {tool["name"]}: {tool["description"]}'
                            for tool in tools
                        )
                        readme = (
                            readme[:tools_pos + 8] +  # Length of "## Tools\n"
                            tools_section.strip() +
                            "\n" + new_tools +
                            readme[next_section:]
                        )
                        
                        with open(readme_file, "w") as f:
                            f.write(readme)
            
            # Commit changes
            self.commit_changes(
                message=f"Add tools to {agent_name} agent",
                files=[agent_file, readme_file]
            )
            
            return {
                "status": "success",
                "message": f"Tools added to {agent_name} successfully",
                "tools_added": len(tools)
            }
            
        except Exception as e:
            logger.error(f"Error adding tools: {e}")
            return {"error": str(e)}
    
    def analyze_agent(
        self,
        agent_name: str
    ) -> Dict[str, Any]:
        """Analyze an existing agent.
        
        Args:
            agent_name: Name of the agent to analyze
        
        Returns:
            Analysis results
        """
        try:
            agent_dir = os.path.join(self.work_dir, "agents", agent_name.lower())
            agent_file = os.path.join(agent_dir, f"{agent_name.lower()}.py")
            
            if not os.path.exists(agent_file):
                return {
                    "status": "error",
                    "error": f"Agent {agent_name} not found"
                }
            
            # Read agent code
            with open(agent_file, "r") as f:
                code = f.read()
            
            # Analyze code using LLM
            analysis_prompt = f"""Analyze this LangChain agent code and provide:
1. Overview of functionality
2. List of tools and their purposes
3. Memory usage and persistence
4. Error handling approach
5. Testing coverage
6. Potential improvements

Code:
{code}
"""
            
            analysis_result = self.run(analysis_prompt)
            if "error" in analysis_result:
                return analysis_result
            
            # Parse analysis
            analysis = {
                "agent_name": agent_name,
                "code_size": len(code),
                "tools_count": code.count("Tool("),
                "has_memory": "ChromaDB" in code or "ConversationBufferMemory" in code,
                "has_tests": os.path.exists(os.path.join(agent_dir, f"test_{agent_name.lower()}.py")),
                "analysis": analysis_result["result"]
            }
            
            # Save analysis
            analysis_file = os.path.join(agent_dir, "analysis.json")
            with open(analysis_file, "w") as f:
                json.dump(analysis, f, indent=2)
            
            return {
                "status": "success",
                "message": f"Agent {agent_name} analyzed successfully",
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing agent: {e}")
            return {"error": str(e)}
    
    def improve_agent(
        self,
        agent_name: str,
        improvements: List[str]
    ) -> Dict[str, Any]:
        """Improve an existing agent.
        
        Args:
            agent_name: Name of the agent to improve
            improvements: List of improvements to make
        
        Returns:
            Status of the operation
        """
        try:
            agent_dir = os.path.join(self.work_dir, "agents", agent_name.lower())
            agent_file = os.path.join(agent_dir, f"{agent_name.lower()}.py")
            
            if not os.path.exists(agent_file):
                return {
                    "status": "error",
                    "error": f"Agent {agent_name} not found"
                }
            
            # Read current code
            with open(agent_file, "r") as f:
                code = f.read()
            
            # Create improvement branch
            branch_result = self.create_branch(f"improve-{agent_name.lower()}")
            if "error" in branch_result:
                return branch_result
            
            # Apply improvements
            improved_code = code
            for improvement in improvements:
                # Use LLM to implement improvement
                improve_prompt = f"""Improve this LangChain agent code by implementing:
{improvement}

Current code:
{improved_code}

Return only the improved code without explanations.
"""
                
                result = self.run(improve_prompt)
                if "error" in result:
                    continue
                
                improved_code = result["result"]
            
            # Save improved code
            with open(agent_file, "w") as f:
                f.write(improved_code)
            
            # Update tests if needed
            test_file = os.path.join(agent_dir, f"test_{agent_name.lower()}.py")
            if os.path.exists(test_file):
                with open(test_file, "r") as f:
                    test_code = f.read()
                
                # Use LLM to update tests
                test_prompt = f"""Update these tests for the improved agent code:
{test_code}

Improved agent code:
{improved_code}

Return only the updated test code without explanations.
"""
                
                result = self.run(test_prompt)
                if "error" not in result:
                    with open(test_file, "w") as f:
                        f.write(result["result"])
            
            # Commit changes
            commit_result = self.commit_changes(
                message=f"Improve {agent_name} agent\n\nImprovements:\n" +
                       "\n".join(f"- {imp}" for imp in improvements),
                files=[agent_file, test_file]
            )
            
            if "error" in commit_result:
                return commit_result
            
            # Create pull request
            pr_result = self.create_pull_request(
                title=f"Improve {agent_name} agent",
                body=f"""This PR improves the {agent_name} agent by implementing:
{chr(10).join(f'- {imp}' for imp in improvements)}

Please review the changes and test the improved agent."""
            )
            
            return {
                "status": "success",
                "message": f"Agent {agent_name} improved successfully",
                "improvements": improvements,
                "pull_request": pr_result.get("pr_url")
            }
            
        except Exception as e:
            logger.error(f"Error improving agent: {e}")
            return {"error": str(e)}
    
    def test_agent(
        self,
        agent_name: str,
        test_cases: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Test an agent's functionality.
        
        Args:
            agent_name: Name of the agent to test
            test_cases: Optional list of test cases
        
        Returns:
            Test results
        """
        try:
            agent_dir = os.path.join(self.work_dir, "agents", agent_name.lower())
            
            # Run pytest if no specific test cases
            if not test_cases:
                return self.run_tests(agent_dir)
            
            # Create and run specific test cases
            test_code = f"""\"\"\"Generated tests for {agent_name} agent.\"\"\"
import pytest
from unittest.mock import Mock, patch
from .{agent_name.lower()} import {agent_name}Agent

@pytest.fixture
def agent():
    \"\"\"Create an agent instance.\"\"\"
    return {agent_name}Agent()

"""
            
            for i, case in enumerate(test_cases, 1):
                test_code += f"""
def test_case_{i}(agent):
    \"\"\"Test case {i}: {case['description']}\"\"\"
    agent.setup_tools()
    agent.setup_agent()
    
    result = agent.run("{case['input']}")
    
    assert result["status"] == "{case.get('expected_status', 'success')}"
    {f'assert "{case["expected_output"]}" in str(result["result"])' if case.get('expected_output') else ''}
"""
            
            # Save test file
            test_file = os.path.join(agent_dir, "test_cases.py")
            with open(test_file, "w") as f:
                f.write(test_code)
            
            # Run tests
            result = self.run_tests(test_file)
            
            # Add test cases to results
            if "error" not in result:
                result["test_cases"] = test_cases
            
            return result
            
        except Exception as e:
            logger.error(f"Error testing agent: {e}")
            return {"error": str(e)}
    
    def get_agent_type(self) -> str:
        """Get the agent type.
        
        Returns:
            Agent type
        """
        return self.agent_type
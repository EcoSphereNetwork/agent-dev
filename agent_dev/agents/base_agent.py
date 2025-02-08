"""Base Agent class for all agents in the system."""
from typing import List, Dict, Any, Optional
import logging
import os
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

logger = logging.getLogger(__name__)

class BaseAgent:
    """Base class for all agents in the system."""
    
    def __init__(
        self,
        role: str,
        api_base: str = "http://localhost:11434/v1",
        model_name: str = "mistral",
        temperature: float = 0,
        verbose: bool = True,
        memory_path: Optional[str] = None
    ):
        """Initialize the base agent.
        
        Args:
            role: Agent's role (e.g., "scrum-master", "product-owner")
            api_base: Base URL for local LLM API (Ollama or LM-Studio)
            model_name: Name of the LLM model to use
            temperature: LLM temperature (0-1)
            verbose: Whether to enable verbose output
            memory_path: Optional path for ChromaDB memory
        """
        self.role = role
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
    
    def setup_agent(self):
        """Set up the LangChain agent with tools.
        
        This method should be overridden by subclasses to add their specific tools.
        """
        if not self.tools:
            raise ValueError("No tools defined. Add tools before setting up agent.")
        
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=self.verbose,
            memory=self.memory
        )
    
    def add_tool(self, tool: Tool):
        """Add a tool to the agent.
        
        Args:
            tool: LangChain Tool to add
        """
        self.tools.append(tool)
    
    def add_tools(self, tools: List[Tool]):
        """Add multiple tools to the agent.
        
        Args:
            tools: List of LangChain Tools to add
        """
        self.tools.extend(tools)
    
    def run(self, task: str) -> Dict[str, Any]:
        """Run a task using the agent.
        
        Args:
            task: Task description
        
        Returns:
            Task results
        """
        if not self.agent:
            raise ValueError("Agent not initialized. Call setup_agent() first.")
        
        try:
            result = self.agent.run(task)
            
            # Store task and result in memory if ChromaDB is available
            if self.db:
                self.db.add_texts(
                    texts=[f"Task: {task}\nResult: {result}"],
                    metadatas=[{
                        "role": self.role,
                        "type": "task_result",
                        "task": task
                    }]
                )
            
            return {
                "status": "success",
                "result": result,
                "metadata": {
                    "role": self.role,
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
                    "role": self.role,
                    "model": self.llm.model_name,
                    "temperature": self.llm.temperature,
                    "tools_used": [tool.name for tool in self.tools]
                }
            }
    
    def get_memory(self, query: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get the agent's memory.
        
        Args:
            query: Optional query to search memory
        
        Returns:
            List of memory entries
        """
        if self.db and query:
            results = self.db.similarity_search(query)
            return [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in results
            ]
        return self.memory.chat_memory.messages
    
    def clear_memory(self):
        """Clear the agent's memory."""
        if self.db:
            self.db._collection.delete()
        self.memory.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics.
        
        Returns:
            Dictionary of agent statistics
        """
        stats = {
            "role": self.role,
            "model": self.llm.model_name,
            "temperature": self.llm.temperature,
            "tools_count": len(self.tools),
            "memory_size": len(self.memory.chat_memory.messages),
            "tools": [
                {
                    "name": tool.name,
                    "description": tool.description
                }
                for tool in self.tools
            ]
        }
        
        if self.db:
            stats["chroma_documents"] = len(self.db.get()["documents"])
        
        return stats
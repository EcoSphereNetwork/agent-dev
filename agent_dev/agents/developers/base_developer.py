"""Base Developer Agent for all developer agents."""
from typing import Dict, Any, List, Optional
import logging
import os
import subprocess
from langchain.agents import Tool
from ..base_agent import BaseAgent

logger = logging.getLogger(__name__)

class BaseDeveloperAgent(BaseAgent):
    """Base class for all developer agents."""
    
    def __init__(
        self,
        github_token: str,
        github_org: str,
        github_repo: str,
        work_dir: str,
        specialty: str,
        **kwargs
    ):
        """Initialize the base developer agent.
        
        Args:
            github_token: GitHub API token
            github_org: GitHub organization name
            github_repo: GitHub repository name
            work_dir: Working directory
            specialty: Developer specialty
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(role=f"developer-{specialty}", **kwargs)
        self.github_token = github_token
        self.github_org = github_org
        self.github_repo = github_repo
        self.work_dir = work_dir
        self.specialty = specialty
        self.setup_dev_tools()
    
    def setup_dev_tools(self):
        """Set up common development tools."""
        self.add_tools([
            Tool(
                name="clone_repository",
                func=self.clone_repository,
                description="Clone a repository"
            ),
            Tool(
                name="create_branch",
                func=self.create_branch,
                description="Create a git branch"
            ),
            Tool(
                name="commit_changes",
                func=self.commit_changes,
                description="Commit changes"
            ),
            Tool(
                name="create_pull_request",
                func=self.create_pull_request,
                description="Create a pull request"
            ),
            Tool(
                name="run_tests",
                func=self.run_tests,
                description="Run tests"
            )
        ])
    
    def clone_repository(
        self,
        repository: Optional[str] = None
    ) -> Dict[str, Any]:
        """Clone a repository.
        
        Args:
            repository: Optional repository name (defaults to self.github_repo)
        
        Returns:
            Status of the operation
        """
        try:
            repo = repository or self.github_repo
            repo_dir = os.path.join(self.work_dir, repo)
            
            if not os.path.exists(repo_dir):
                os.makedirs(repo_dir)
            
            # Clone repository
            repo_url = f"https://{self.github_token}@github.com/{self.github_org}/{repo}.git"
            result = subprocess.run(
                ["git", "clone", repo_url, repo_dir],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "message": f"Repository {repo} cloned successfully",
                    "repo_dir": repo_dir
                }
            else:
                return {
                    "status": "error",
                    "error": result.stderr
                }
            
        except Exception as e:
            logger.error(f"Error cloning repository: {e}")
            return {"error": str(e)}
    
    def create_branch(
        self,
        branch_name: str,
        repository: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a git branch.
        
        Args:
            branch_name: Name of the branch to create
            repository: Optional repository name
        
        Returns:
            Status of the operation
        """
        try:
            repo = repository or self.github_repo
            repo_dir = os.path.join(self.work_dir, repo)
            
            # Create branch
            result = subprocess.run(
                ["git", "checkout", "-b", branch_name],
                cwd=repo_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return {
                    "status": "success",
                    "message": f"Branch {branch_name} created successfully",
                    "branch": branch_name
                }
            else:
                return {
                    "status": "error",
                    "error": result.stderr
                }
            
        except Exception as e:
            logger.error(f"Error creating branch: {e}")
            return {"error": str(e)}
    
    def commit_changes(
        self,
        message: str,
        files: Optional[List[str]] = None,
        repository: Optional[str] = None
    ) -> Dict[str, Any]:
        """Commit changes to git.
        
        Args:
            message: Commit message
            files: Optional list of files to commit
            repository: Optional repository name
        
        Returns:
            Status of the operation
        """
        try:
            repo = repository or self.github_repo
            repo_dir = os.path.join(self.work_dir, repo)
            
            # Add files
            if files:
                add_result = subprocess.run(
                    ["git", "add"] + files,
                    cwd=repo_dir,
                    capture_output=True,
                    text=True
                )
            else:
                add_result = subprocess.run(
                    ["git", "add", "."],
                    cwd=repo_dir,
                    capture_output=True,
                    text=True
                )
            
            if add_result.returncode != 0:
                return {
                    "status": "error",
                    "error": add_result.stderr
                }
            
            # Commit changes
            commit_result = subprocess.run(
                ["git", "commit", "-m", message],
                cwd=repo_dir,
                capture_output=True,
                text=True
            )
            
            if commit_result.returncode == 0:
                return {
                    "status": "success",
                    "message": "Changes committed successfully",
                    "commit": commit_result.stdout
                }
            else:
                return {
                    "status": "error",
                    "error": commit_result.stderr
                }
            
        except Exception as e:
            logger.error(f"Error committing changes: {e}")
            return {"error": str(e)}
    
    def create_pull_request(
        self,
        title: str,
        body: str,
        base_branch: str = "main",
        repository: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a pull request.
        
        Args:
            title: PR title
            body: PR description
            base_branch: Base branch for PR
            repository: Optional repository name
        
        Returns:
            Status of the operation
        """
        try:
            repo = repository or self.github_repo
            repo_dir = os.path.join(self.work_dir, repo)
            
            # Get current branch
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_dir,
                capture_output=True,
                text=True
            )
            
            if branch_result.returncode != 0:
                return {
                    "status": "error",
                    "error": branch_result.stderr
                }
            
            current_branch = branch_result.stdout.strip()
            
            # Push changes
            push_result = subprocess.run(
                ["git", "push", "-u", "origin", current_branch],
                cwd=repo_dir,
                capture_output=True,
                text=True
            )
            
            if push_result.returncode != 0:
                return {
                    "status": "error",
                    "error": push_result.stderr
                }
            
            # Create PR
            import requests
            url = f"https://api.github.com/repos/{self.github_org}/{repo}/pulls"
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            data = {
                "title": title,
                "body": body,
                "head": current_branch,
                "base": base_branch
            }
            
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            pr = response.json()
            
            return {
                "status": "success",
                "message": "Pull request created successfully",
                "pr_number": pr["number"],
                "pr_url": pr["html_url"]
            }
            
        except Exception as e:
            logger.error(f"Error creating pull request: {e}")
            return {"error": str(e)}
    
    def run_tests(
        self,
        test_path: Optional[str] = None,
        repository: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run tests.
        
        Args:
            test_path: Optional path to specific tests
            repository: Optional repository name
        
        Returns:
            Test results
        """
        try:
            repo = repository or self.github_repo
            repo_dir = os.path.join(self.work_dir, repo)
            
            # Run tests
            if test_path:
                test_command = ["pytest", test_path]
            else:
                test_command = ["pytest"]
            
            result = subprocess.run(
                test_command,
                cwd=repo_dir,
                capture_output=True,
                text=True
            )
            
            return {
                "status": "success" if result.returncode == 0 else "failure",
                "output": result.stdout,
                "errors": result.stderr if result.returncode != 0 else None
            }
            
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return {"error": str(e)}
    
    def get_work_dir(self) -> str:
        """Get the working directory.
        
        Returns:
            Working directory path
        """
        return self.work_dir
    
    def get_specialty(self) -> str:
        """Get the developer's specialty.
        
        Returns:
            Developer specialty
        """
        return self.specialty
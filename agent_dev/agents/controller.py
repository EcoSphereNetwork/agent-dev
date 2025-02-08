"""Controller (Scrum Master) Agent for coordinating the team."""
from typing import Dict, Any, List, Optional
import logging
import os
import json
import datetime
from langchain.agents import Tool
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ControllerAgent(BaseAgent):
    """Agent for coordinating the team and managing sprints."""
    
    def __init__(
        self,
        github_token: str,
        github_org: str,
        github_repo: str,
        sprint_length: int = 14,
        **kwargs
    ):
        """Initialize the controller agent.
        
        Args:
            github_token: GitHub API token
            github_org: GitHub organization name
            github_repo: GitHub repository name
            sprint_length: Sprint length in days
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(role="scrum-master", **kwargs)
        self.github_token = github_token
        self.github_org = github_org
        self.github_repo = github_repo
        self.sprint_length = sprint_length
        self.setup_controller_tools()
    
    def setup_controller_tools(self):
        """Set up controller specific tools."""
        self.add_tools([
            Tool(
                name="plan_sprint",
                func=self.plan_sprint,
                description="Plan a new sprint"
            ),
            Tool(
                name="assign_tasks",
                func=self.assign_tasks,
                description="Assign tasks to agents"
            ),
            Tool(
                name="track_progress",
                func=self.track_progress,
                description="Track sprint progress"
            ),
            Tool(
                name="manage_repositories",
                func=self.manage_repositories,
                description="Manage multiple repositories"
            ),
            Tool(
                name="generate_reports",
                func=self.generate_reports,
                description="Generate sprint reports"
            )
        ])
    
    def plan_sprint(
        self,
        sprint_number: int,
        start_date: str,
        repositories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Plan a new sprint.
        
        Args:
            sprint_number: Sprint number
            start_date: Sprint start date
            repositories: List of repositories to work on
        
        Returns:
            Sprint planning results
        """
        try:
            # Calculate sprint dates
            start = datetime.datetime.strptime(start_date, "%Y-%m-%d")
            end = start + datetime.timedelta(days=self.sprint_length)
            
            # Create sprint milestone for each repository
            import requests
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            milestones = {}
            for repo in repositories:
                # Create milestone
                url = f"https://api.github.com/repos/{self.github_org}/{repo['name']}/milestones"
                data = {
                    "title": f"Sprint {sprint_number}",
                    "description": f"Sprint {sprint_number} ({start_date} to {end.strftime('%Y-%m-%d')})",
                    "due_on": end.strftime("%Y-%m-%dT%H:%M:%SZ")
                }
                
                response = requests.post(url, json=data, headers=headers)
                response.raise_for_status()
                milestone = response.json()
                
                # Create project board
                board_url = f"https://api.github.com/repos/{self.github_org}/{repo['name']}/projects"
                board_data = {
                    "name": f"Sprint {sprint_number} Board",
                    "body": f"Sprint {sprint_number} tracking board"
                }
                
                response = requests.post(board_url, json=board_data, headers=headers)
                response.raise_for_status()
                board = response.json()
                
                # Create columns
                columns = ["To Do", "In Progress", "Review", "Done"]
                column_urls = []
                
                for column in columns:
                    column_url = f"{board['url']}/columns"
                    column_data = {"name": column}
                    response = requests.post(column_url, json=column_data, headers=headers)
                    response.raise_for_status()
                    column_urls.append(response.json()["url"])
                
                milestones[repo["name"]] = {
                    "milestone": milestone,
                    "board": board,
                    "columns": column_urls
                }
            
            # Save sprint data
            sprint_data = {
                "number": sprint_number,
                "start_date": start_date,
                "end_date": end.strftime("%Y-%m-%d"),
                "repositories": repositories,
                "milestones": milestones,
                "status": "planned",
                "tasks": []
            }
            
            sprint_dir = os.path.join(self.work_dir, "sprints")
            os.makedirs(sprint_dir, exist_ok=True)
            
            with open(os.path.join(sprint_dir, f"sprint_{sprint_number}.json"), "w") as f:
                json.dump(sprint_data, f, indent=2)
            
            return {
                "status": "success",
                "message": f"Sprint {sprint_number} planned successfully",
                "sprint": sprint_data
            }
            
        except Exception as e:
            logger.error(f"Error planning sprint: {e}")
            return {"error": str(e)}
    
    def assign_tasks(
        self,
        sprint_number: int,
        tasks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assign tasks to agents.
        
        Args:
            sprint_number: Sprint number
            tasks: List of tasks to assign
        
        Returns:
            Task assignments
        """
        try:
            # Load sprint data
            sprint_dir = os.path.join(self.work_dir, "sprints")
            sprint_file = os.path.join(sprint_dir, f"sprint_{sprint_number}.json")
            
            if not os.path.exists(sprint_file):
                return {
                    "status": "error",
                    "error": f"Sprint {sprint_number} data not found"
                }
            
            with open(sprint_file, "r") as f:
                sprint_data = json.load(f)
            
            # Assign tasks based on agent specialties
            assignments = {}
            for task in tasks:
                # Determine best agent based on task type
                if "langchain" in task["title"].lower():
                    agent_type = "agent-developer"
                elif "api" in task["title"].lower():
                    agent_type = "integration-engineer"
                elif "refactor" in task["title"].lower():
                    agent_type = "refactoring-agent"
                elif "doc" in task["title"].lower():
                    agent_type = "doc-writer"
                else:
                    agent_type = "agent-developer"  # default
                
                if agent_type not in assignments:
                    assignments[agent_type] = []
                assignments[agent_type].append(task)
            
            # Create GitHub issues for tasks
            import requests
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            created_tasks = []
            for agent_type, agent_tasks in assignments.items():
                for task in agent_tasks:
                    # Create issue in appropriate repository
                    repo = task["repository"]
                    url = f"https://api.github.com/repos/{self.github_org}/{repo}/issues"
                    
                    # Add task metadata
                    body = f"""## Task Description
{task['description']}

## Agent Assignment
- Type: {agent_type}
- Sprint: {sprint_number}

## Acceptance Criteria
{task.get('criteria', '- [ ] Task completed')}

## Technical Notes
- Repository: {repo}
- Priority: {task.get('priority', 'medium')}
- Estimated effort: {task.get('effort', '1 day')}
"""
                    
                    data = {
                        "title": task["title"],
                        "body": body,
                        "milestone": sprint_data["milestones"][repo]["milestone"]["number"],
                        "labels": [agent_type, "sprint-task"] + task.get("labels", [])
                    }
                    
                    response = requests.post(url, json=data, headers=headers)
                    response.raise_for_status()
                    issue = response.json()
                    
                    # Add to project board
                    card_url = f"{sprint_data['milestones'][repo]['columns'][0]}/cards"
                    card_data = {
                        "content_id": issue["id"],
                        "content_type": "Issue"
                    }
                    response = requests.post(card_url, json=card_data, headers=headers)
                    response.raise_for_status()
                    
                    created_tasks.append({
                        "number": issue["number"],
                        "title": issue["title"],
                        "url": issue["html_url"],
                        "agent": agent_type,
                        "repository": repo
                    })
            
            # Update sprint data
            sprint_data["tasks"].extend(created_tasks)
            with open(sprint_file, "w") as f:
                json.dump(sprint_data, f, indent=2)
            
            return {
                "status": "success",
                "message": f"Tasks assigned for sprint {sprint_number}",
                "assignments": assignments,
                "tasks": created_tasks
            }
            
        except Exception as e:
            logger.error(f"Error assigning tasks: {e}")
            return {"error": str(e)}
    
    def track_progress(
        self,
        sprint_number: int
    ) -> Dict[str, Any]:
        """Track sprint progress.
        
        Args:
            sprint_number: Sprint number
        
        Returns:
            Sprint progress status
        """
        try:
            # Load sprint data
            sprint_dir = os.path.join(self.work_dir, "sprints")
            sprint_file = os.path.join(sprint_dir, f"sprint_{sprint_number}.json")
            
            if not os.path.exists(sprint_file):
                return {
                    "status": "error",
                    "error": f"Sprint {sprint_number} data not found"
                }
            
            with open(sprint_file, "r") as f:
                sprint_data = json.load(f)
            
            # Get progress for each repository
            import requests
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            progress = {}
            for repo in sprint_data["repositories"]:
                repo_name = repo["name"]
                milestone = sprint_data["milestones"][repo_name]["milestone"]
                
                # Get milestone progress
                url = f"https://api.github.com/repos/{self.github_org}/{repo_name}/milestones/{milestone['number']}"
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                milestone_data = response.json()
                
                # Get project board status
                board = sprint_data["milestones"][repo_name]["board"]
                columns = sprint_data["milestones"][repo_name]["columns"]
                
                column_stats = {}
                for i, column_url in enumerate(columns):
                    response = requests.get(column_url, headers=headers)
                    response.raise_for_status()
                    column = response.json()
                    
                    column_stats[column["name"]] = len(column["cards"])
                
                progress[repo_name] = {
                    "milestone": {
                        "open_issues": milestone_data["open_issues"],
                        "closed_issues": milestone_data["closed_issues"],
                        "progress": (
                            milestone_data["closed_issues"] /
                            (milestone_data["open_issues"] + milestone_data["closed_issues"]) * 100
                            if milestone_data["open_issues"] + milestone_data["closed_issues"] > 0
                            else 0
                        )
                    },
                    "board": column_stats
                }
            
            # Calculate overall progress
            total_issues = sum(
                p["milestone"]["open_issues"] + p["milestone"]["closed_issues"]
                for p in progress.values()
            )
            completed_issues = sum(
                p["milestone"]["closed_issues"]
                for p in progress.values()
            )
            
            overall_progress = {
                "total_issues": total_issues,
                "completed_issues": completed_issues,
                "progress_percentage": (
                    completed_issues / total_issues * 100
                    if total_issues > 0
                    else 0
                ),
                "repositories": progress
            }
            
            # Save progress
            progress_file = os.path.join(sprint_dir, f"progress_{sprint_number}.json")
            with open(progress_file, "w") as f:
                json.dump(overall_progress, f, indent=2)
            
            return {
                "status": "success",
                "message": f"Sprint {sprint_number} progress tracked",
                "progress": overall_progress
            }
            
        except Exception as e:
            logger.error(f"Error tracking progress: {e}")
            return {"error": str(e)}
    
    def manage_repositories(
        self,
        action: str,
        repositories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Manage multiple repositories.
        
        Args:
            action: Action to perform (add/update/sync)
            repositories: List of repositories to manage
        
        Returns:
            Status of the operation
        """
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            if action == "add":
                # Add new repositories to track
                added_repos = []
                for repo in repositories:
                    # Create repository if it doesn't exist
                    url = f"https://api.github.com/orgs/{self.github_org}/repos"
                    data = {
                        "name": repo["name"],
                        "description": repo.get("description", ""),
                        "private": repo.get("private", False),
                        "has_issues": True,
                        "has_projects": True,
                        "has_wiki": True
                    }
                    
                    response = requests.post(url, json=data, headers=headers)
                    if response.status_code == 422:  # Repository exists
                        response = requests.get(
                            f"https://api.github.com/repos/{self.github_org}/{repo['name']}",
                            headers=headers
                        )
                    response.raise_for_status()
                    repo_data = response.json()
                    
                    added_repos.append({
                        "name": repo_data["name"],
                        "url": repo_data["html_url"],
                        "clone_url": repo_data["clone_url"]
                    })
                
                return {
                    "status": "success",
                    "message": f"Added {len(added_repos)} repositories",
                    "repositories": added_repos
                }
                
            elif action == "update":
                # Update repository settings
                updated_repos = []
                for repo in repositories:
                    url = f"https://api.github.com/repos/{self.github_org}/{repo['name']}"
                    data = {
                        "name": repo["name"],
                        "description": repo.get("description"),
                        "private": repo.get("private"),
                        "has_issues": repo.get("has_issues"),
                        "has_projects": repo.get("has_projects"),
                        "has_wiki": repo.get("has_wiki")
                    }
                    
                    response = requests.patch(url, json=data, headers=headers)
                    response.raise_for_status()
                    repo_data = response.json()
                    
                    updated_repos.append({
                        "name": repo_data["name"],
                        "url": repo_data["html_url"]
                    })
                
                return {
                    "status": "success",
                    "message": f"Updated {len(updated_repos)} repositories",
                    "repositories": updated_repos
                }
                
            elif action == "sync":
                # Sync repositories (e.g., shared workflows, templates)
                synced_repos = []
                for repo in repositories:
                    # Get repository contents
                    url = f"https://api.github.com/repos/{self.github_org}/{repo['name']}/contents"
                    response = requests.get(url, headers=headers)
                    response.raise_for_status()
                    contents = response.json()
                    
                    # Check for workflow files
                    workflows = [c for c in contents if c["path"].startswith(".github/workflows")]
                    if not workflows:
                        # Create default workflow
                        workflow_url = f"https://api.github.com/repos/{self.github_org}/{repo['name']}/contents/.github/workflows/ci.yml"
                        workflow_content = """name: CI
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.8'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    - name: Run tests
      run: |
        pytest
"""
                        
                        data = {
                            "message": "Add default CI workflow",
                            "content": workflow_content.encode("utf-8").hex(),
                            "branch": "main"
                        }
                        
                        response = requests.put(workflow_url, json=data, headers=headers)
                        response.raise_for_status()
                    
                    synced_repos.append({
                        "name": repo["name"],
                        "synced_files": [
                            ".github/workflows/ci.yml"
                        ]
                    })
                
                return {
                    "status": "success",
                    "message": f"Synced {len(synced_repos)} repositories",
                    "repositories": synced_repos
                }
            
            return {
                "status": "error",
                "error": f"Unknown action: {action}"
            }
            
        except Exception as e:
            logger.error(f"Error managing repositories: {e}")
            return {"error": str(e)}
    
    def generate_reports(
        self,
        sprint_number: int,
        report_type: str = "sprint"
    ) -> Dict[str, Any]:
        """Generate sprint reports.
        
        Args:
            sprint_number: Sprint number
            report_type: Type of report (sprint/velocity/burndown)
        
        Returns:
            Generated report
        """
        try:
            # Load sprint data
            sprint_dir = os.path.join(self.work_dir, "sprints")
            sprint_file = os.path.join(sprint_dir, f"sprint_{sprint_number}.json")
            progress_file = os.path.join(sprint_dir, f"progress_{sprint_number}.json")
            
            if not os.path.exists(sprint_file) or not os.path.exists(progress_file):
                return {
                    "status": "error",
                    "error": f"Sprint {sprint_number} data not found"
                }
            
            with open(sprint_file, "r") as f:
                sprint_data = json.load(f)
            
            with open(progress_file, "r") as f:
                progress_data = json.load(f)
            
            if report_type == "sprint":
                # Generate sprint report
                report = f"""# Sprint {sprint_number} Report

## Overview
- Start Date: {sprint_data['start_date']}
- End Date: {sprint_data['end_date']}
- Total Issues: {progress_data['total_issues']}
- Completed Issues: {progress_data['completed_issues']}
- Progress: {progress_data['progress_percentage']:.1f}%

## Repository Status
"""
                
                for repo_name, repo_progress in progress_data["repositories"].items():
                    report += f"""
### {repo_name}
- Open Issues: {repo_progress['milestone']['open_issues']}
- Closed Issues: {repo_progress['milestone']['closed_issues']}
- Progress: {repo_progress['milestone']['progress']:.1f}%

Board Status:
"""
                    for column, count in repo_progress["board"].items():
                        report += f"- {column}: {count}\n"
                
                report += "\n## Tasks by Agent\n"
                agent_tasks = {}
                for task in sprint_data["tasks"]:
                    agent = task["agent"]
                    if agent not in agent_tasks:
                        agent_tasks[agent] = []
                    agent_tasks[agent].append(task)
                
                for agent, tasks in agent_tasks.items():
                    report += f"\n### {agent}\n"
                    for task in tasks:
                        report += f"- [{task['title']}]({task['url']})\n"
                
            elif report_type == "velocity":
                # Calculate velocity across sprints
                velocities = []
                for sprint_file in sorted(os.listdir(sprint_dir)):
                    if not sprint_file.startswith("sprint_"):
                        continue
                    
                    sprint_num = int(sprint_file.split("_")[1].split(".")[0])
                    progress_file = os.path.join(sprint_dir, f"progress_{sprint_num}.json")
                    
                    if os.path.exists(progress_file):
                        with open(progress_file, "r") as f:
                            progress = json.load(f)
                            velocities.append({
                                "sprint": sprint_num,
                                "completed": progress["completed_issues"]
                            })
                
                # Generate velocity report
                report = "# Velocity Report\n\n"
                
                if velocities:
                    avg_velocity = sum(v["completed"] for v in velocities) / len(velocities)
                    report += f"\nAverage Velocity: {avg_velocity:.1f} issues/sprint\n\n"
                    
                    report += "## Sprint History\n"
                    for velocity in velocities:
                        report += f"- Sprint {velocity['sprint']}: {velocity['completed']} issues completed\n"
                
            elif report_type == "burndown":
                # Generate burndown report
                report = f"""# Sprint {sprint_number} Burndown Report

## Overview
- Total Issues: {progress_data['total_issues']}
- Completed Issues: {progress_data['completed_issues']}
- Remaining Issues: {progress_data['total_issues'] - progress_data['completed_issues']}

## Repository Details
"""
                
                for repo_name, repo_progress in progress_data["repositories"].items():
                    report += f"""
### {repo_name}
- Open Issues: {repo_progress['milestone']['open_issues']}
- Closed Issues: {repo_progress['milestone']['closed_issues']}
- Progress: {repo_progress['milestone']['progress']:.1f}%
"""
            
            # Save report
            report_file = os.path.join(sprint_dir, f"{report_type}_report_{sprint_number}.md")
            with open(report_file, "w") as f:
                f.write(report)
            
            return {
                "status": "success",
                "message": f"Generated {report_type} report for sprint {sprint_number}",
                "report_file": report_file,
                "report": report
            }
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {"error": str(e)}
    
    def get_sprint_length(self) -> int:
        """Get the sprint length.
        
        Returns:
            Sprint length in days
        """
        return self.sprint_length
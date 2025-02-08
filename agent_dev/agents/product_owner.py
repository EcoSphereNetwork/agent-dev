"""Product Owner Agent for managing requirements and user stories."""
from typing import Dict, Any, List, Optional
import logging
import os
import json
from langchain.agents import Tool
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class ProductOwnerAgent(BaseAgent):
    """Agent for managing product requirements and user stories."""
    
    def __init__(
        self,
        github_token: str,
        github_org: str,
        github_repo: str,
        memory_path: Optional[str] = None,
        **kwargs
    ):
        """Initialize the product owner agent.
        
        Args:
            github_token: GitHub API token
            github_org: GitHub organization name
            github_repo: GitHub repository name
            memory_path: Optional path for ChromaDB memory
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(
            role="product-owner",
            memory_path=memory_path,
            **kwargs
        )
        self.github_token = github_token
        self.github_org = github_org
        self.github_repo = github_repo
        self.setup_po_tools()
    
    def setup_po_tools(self):
        """Set up product owner specific tools."""
        self.add_tools([
            Tool(
                name="create_story",
                func=self.create_story,
                description="Create a user story"
            ),
            Tool(
                name="prioritize_backlog",
                func=self.prioritize_backlog,
                description="Prioritize product backlog"
            ),
            Tool(
                name="define_sprint_goals",
                func=self.define_sprint_goals,
                description="Define sprint goals"
            ),
            Tool(
                name="analyze_feedback",
                func=self.analyze_feedback,
                description="Analyze user feedback"
            ),
            Tool(
                name="generate_roadmap",
                func=self.generate_roadmap,
                description="Generate product roadmap"
            )
        ])
    
    def create_story(
        self,
        title: str,
        description: str,
        acceptance_criteria: List[str],
        repository: str,
        labels: Optional[List[str]] = None,
        priority: str = "medium"
    ) -> Dict[str, Any]:
        """Create a user story.
        
        Args:
            title: Story title
            description: Story description
            acceptance_criteria: List of acceptance criteria
            repository: Target repository
            labels: Optional list of labels
            priority: Story priority (low/medium/high)
        
        Returns:
            Created story details
        """
        try:
            # Format story description
            story_body = f"""# User Story
{description}

## Acceptance Criteria
{chr(10).join(f'- [ ] {criterion}' for criterion in acceptance_criteria)}

## Technical Notes
- Repository: {repository}
- Priority: {priority}
- Labels: {', '.join(labels) if labels else 'None'}

## Definition of Done
- [ ] Code implemented
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] Code reviewed
- [ ] Merged to main branch
"""
            
            # Create GitHub issue
            import requests
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            url = f"https://api.github.com/repos/{self.github_org}/{repository}/issues"
            data = {
                "title": title,
                "body": story_body,
                "labels": ["user-story", f"priority-{priority}"] + (labels or [])
            }
            
            response = requests.post(url, json=data, headers=headers)
            response.raise_for_status()
            issue = response.json()
            
            # Store in memory
            if self.db:
                self.db.add_texts(
                    texts=[f"Story: {title}\n{story_body}"],
                    metadatas=[{
                        "type": "user_story",
                        "repository": repository,
                        "priority": priority,
                        "issue_number": issue["number"]
                    }]
                )
            
            return {
                "status": "success",
                "message": "User story created successfully",
                "story": {
                    "number": issue["number"],
                    "title": issue["title"],
                    "url": issue["html_url"],
                    "repository": repository,
                    "priority": priority
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating story: {e}")
            return {"error": str(e)}
    
    def prioritize_backlog(
        self,
        criteria: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Prioritize product backlog.
        
        Args:
            criteria: Optional weighting criteria (e.g., {"urgency": 0.4})
        
        Returns:
            Prioritized backlog
        """
        try:
            # Default criteria if none provided
            if not criteria:
                criteria = {
                    "urgency": 0.4,
                    "value": 0.3,
                    "effort": 0.2,
                    "risk": 0.1
                }
            
            # Get all backlog items
            import requests
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            url = f"https://api.github.com/repos/{self.github_org}/{self.github_repo}/issues"
            params = {
                "labels": "user-story",
                "state": "open",
                "sort": "created",
                "direction": "desc"
            }
            
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            issues = response.json()
            
            # Calculate priority scores
            scored_issues = []
            for issue in issues:
                # Extract metrics from issue body and labels
                urgency = 1.0 if "priority-high" in [l["name"] for l in issue["labels"]] else (
                    0.5 if "priority-medium" in [l["name"] for l in issue["labels"]] else 0.2
                )
                value = 1.0 if "high-value" in [l["name"] for l in issue["labels"]] else (
                    0.5 if "medium-value" in [l["name"] for l in issue["labels"]] else 0.2
                )
                effort = 0.2 if "size-small" in [l["name"] for l in issue["labels"]] else (
                    0.5 if "size-medium" in [l["name"] for l in issue["labels"]] else 1.0
                )
                risk = 1.0 if "high-risk" in [l["name"] for l in issue["labels"]] else (
                    0.5 if "medium-risk" in [l["name"] for l in issue["labels"]] else 0.2
                )
                
                # Calculate weighted score
                score = (
                    urgency * criteria["urgency"] +
                    value * criteria["value"] +
                    (1 - effort) * criteria["effort"] +  # Lower effort is better
                    (1 - risk) * criteria["risk"]  # Lower risk is better
                )
                
                scored_issues.append({
                    "number": issue["number"],
                    "title": issue["title"],
                    "url": issue["html_url"],
                    "score": score,
                    "metrics": {
                        "urgency": urgency,
                        "value": value,
                        "effort": effort,
                        "risk": risk
                    }
                })
            
            # Sort by score
            scored_issues.sort(key=lambda x: x["score"], reverse=True)
            
            # Update issue labels with new priorities
            for i, issue in enumerate(scored_issues):
                priority = "high" if i < len(scored_issues) // 3 else (
                    "medium" if i < len(scored_issues) * 2 // 3 else "low"
                )
                
                # Update issue
                url = f"https://api.github.com/repos/{self.github_org}/{self.github_repo}/issues/{issue['number']}"
                
                # Get current labels without priority labels
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                current_issue = response.json()
                labels = [
                    l["name"] for l in current_issue["labels"]
                    if not l["name"].startswith("priority-")
                ]
                
                # Add new priority label
                labels.append(f"priority-{priority}")
                
                data = {"labels": labels}
                response = requests.patch(url, json=data, headers=headers)
                response.raise_for_status()
                
                scored_issues[i]["priority"] = priority
            
            return {
                "status": "success",
                "message": "Backlog prioritized successfully",
                "criteria": criteria,
                "backlog": scored_issues
            }
            
        except Exception as e:
            logger.error(f"Error prioritizing backlog: {e}")
            return {"error": str(e)}
    
    def define_sprint_goals(
        self,
        sprint_number: int,
        repositories: List[str]
    ) -> Dict[str, Any]:
        """Define sprint goals.
        
        Args:
            sprint_number: Sprint number
            repositories: List of repositories
        
        Returns:
            Sprint goals
        """
        try:
            # Get high-priority stories for each repository
            import requests
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            goals = {}
            for repo in repositories:
                url = f"https://api.github.com/repos/{self.github_org}/{repo}/issues"
                params = {
                    "labels": "priority-high",
                    "state": "open",
                    "sort": "created",
                    "direction": "desc"
                }
                
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                issues = response.json()
                
                # Group issues by type
                issue_types = {}
                for issue in issues:
                    for label in issue["labels"]:
                        if label["name"].startswith("type-"):
                            type_name = label["name"][5:]  # Remove "type-" prefix
                            if type_name not in issue_types:
                                issue_types[type_name] = []
                            issue_types[type_name].append(issue)
                
                # Define goals based on issue types
                repo_goals = []
                for type_name, type_issues in issue_types.items():
                    if type_issues:
                        goal = {
                            "type": type_name,
                            "description": f"Complete high-priority {type_name} stories",
                            "stories": [
                                {
                                    "number": issue["number"],
                                    "title": issue["title"],
                                    "url": issue["html_url"]
                                }
                                for issue in type_issues[:3]  # Top 3 stories
                            ]
                        }
                        repo_goals.append(goal)
                
                goals[repo] = repo_goals
            
            # Create sprint goals document
            sprint_goals = f"""# Sprint {sprint_number} Goals

## Overview
This document outlines the goals for Sprint {sprint_number} across multiple repositories.

"""
            
            for repo, repo_goals in goals.items():
                sprint_goals += f"""
## {repo}
"""
                for goal in repo_goals:
                    sprint_goals += f"""
### {goal['type'].title()} Goal
{goal['description']}

Key Stories:
"""
                    for story in goal["stories"]:
                        sprint_goals += f"- [{story['title']}]({story['url']})\n"
            
            # Save goals
            sprint_dir = os.path.join(self.work_dir, "sprints")
            os.makedirs(sprint_dir, exist_ok=True)
            
            goals_file = os.path.join(sprint_dir, f"goals_sprint_{sprint_number}.md")
            with open(goals_file, "w") as f:
                f.write(sprint_goals)
            
            return {
                "status": "success",
                "message": f"Sprint {sprint_number} goals defined",
                "goals_file": goals_file,
                "goals": goals
            }
            
        except Exception as e:
            logger.error(f"Error defining sprint goals: {e}")
            return {"error": str(e)}
    
    def analyze_feedback(
        self,
        feedback: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze user feedback.
        
        Args:
            feedback: List of feedback items
        
        Returns:
            Feedback analysis
        """
        try:
            # Group feedback by type
            feedback_types = {}
            for item in feedback:
                type_name = item.get("type", "general")
                if type_name not in feedback_types:
                    feedback_types[type_name] = []
                feedback_types[type_name].append(item)
            
            # Analyze each type
            analysis = {}
            for type_name, items in feedback_types.items():
                # Calculate sentiment
                sentiment_scores = []
                for item in items:
                    # Use LLM to analyze sentiment
                    result = self.run(
                        f"Analyze the sentiment of this feedback (positive/negative/neutral):\n{item['text']}"
                    )
                    sentiment = result.get("result", "neutral")
                    sentiment_score = 1 if sentiment == "positive" else (
                        -1 if sentiment == "negative" else 0
                    )
                    sentiment_scores.append(sentiment_score)
                
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                
                # Extract common themes
                all_text = "\n".join(item["text"] for item in items)
                themes_result = self.run(
                    f"Extract common themes from this feedback:\n{all_text}"
                )
                themes = themes_result.get("result", "No themes identified").split("\n")
                
                analysis[type_name] = {
                    "count": len(items),
                    "sentiment": "positive" if avg_sentiment > 0.3 else (
                        "negative" if avg_sentiment < -0.3 else "neutral"
                    ),
                    "sentiment_score": avg_sentiment,
                    "themes": themes,
                    "items": items
                }
            
            # Generate recommendations
            recommendations = []
            for type_name, type_analysis in analysis.items():
                if type_analysis["sentiment"] == "negative":
                    recommendations.append({
                        "type": type_name,
                        "priority": "high",
                        "action": f"Address negative feedback about {type_name}",
                        "themes": type_analysis["themes"]
                    })
                elif type_analysis["count"] >= 3:
                    recommendations.append({
                        "type": type_name,
                        "priority": "medium",
                        "action": f"Review common themes in {type_name} feedback",
                        "themes": type_analysis["themes"]
                    })
            
            # Store analysis in memory
            if self.db:
                self.db.add_texts(
                    texts=[json.dumps(analysis, indent=2)],
                    metadatas=[{
                        "type": "feedback_analysis",
                        "date": datetime.datetime.now().isoformat(),
                        "feedback_count": len(feedback)
                    }]
                )
            
            return {
                "status": "success",
                "message": "Feedback analyzed successfully",
                "analysis": analysis,
                "recommendations": recommendations
            }
            
        except Exception as e:
            logger.error(f"Error analyzing feedback: {e}")
            return {"error": str(e)}
    
    def generate_roadmap(
        self,
        timeframe: str = "6months",
        repositories: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Generate product roadmap.
        
        Args:
            timeframe: Roadmap timeframe (3months/6months/12months)
            repositories: Optional list of repositories to include
        
        Returns:
            Generated roadmap
        """
        try:
            # Get repositories if not provided
            import requests
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            if not repositories:
                url = f"https://api.github.com/orgs/{self.github_org}/repos"
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                repos = response.json()
                repositories = [repo["name"] for repo in repos]
            
            # Get issues for each repository
            all_issues = []
            for repo in repositories:
                url = f"https://api.github.com/repos/{self.github_org}/{repo}/issues"
                params = {"state": "open", "sort": "created", "direction": "desc"}
                
                response = requests.get(url, headers=headers, params=params)
                response.raise_for_status()
                issues = response.json()
                
                for issue in issues:
                    issue["repository"] = repo
                    all_issues.append(issue)
            
            # Group issues by milestone/epic
            epics = {}
            for issue in all_issues:
                for label in issue["labels"]:
                    if label["name"].startswith("epic-"):
                        epic_name = label["name"][5:]  # Remove "epic-" prefix
                        if epic_name not in epics:
                            epics[epic_name] = []
                        epics[epic_name].append(issue)
            
            # Define roadmap phases based on timeframe
            if timeframe == "3months":
                phases = ["Month 1", "Month 2", "Month 3"]
            elif timeframe == "6months":
                phases = ["Q1 Month 1-2", "Q1 Month 3", "Q2 Month 1-2", "Q2 Month 3"]
            else:  # 12months
                phases = ["Q1", "Q2", "Q3", "Q4"]
            
            # Assign epics to phases based on priority and dependencies
            roadmap = {phase: [] for phase in phases}
            
            # Sort epics by priority
            sorted_epics = sorted(
                epics.items(),
                key=lambda x: sum(
                    1 for issue in x[1]
                    if any(l["name"] == "priority-high" for l in issue["labels"])
                ),
                reverse=True
            )
            
            # Distribute epics across phases
            epics_per_phase = len(sorted_epics) // len(phases) + 1
            for i, (epic_name, epic_issues) in enumerate(sorted_epics):
                phase_index = min(i // epics_per_phase, len(phases) - 1)
                roadmap[phases[phase_index]].append({
                    "name": epic_name,
                    "issues": [
                        {
                            "number": issue["number"],
                            "title": issue["title"],
                            "url": issue["html_url"],
                            "repository": issue["repository"]
                        }
                        for issue in epic_issues
                    ]
                })
            
            # Generate roadmap document
            roadmap_doc = f"""# Product Roadmap ({timeframe})

## Overview
This roadmap outlines the planned development across {len(repositories)} repositories over the next {timeframe}.

"""
            
            for phase, phase_epics in roadmap.items():
                roadmap_doc += f"""
## {phase}
"""
                for epic in phase_epics:
                    roadmap_doc += f"""
### {epic['name']}
"""
                    for issue in epic["issues"]:
                        roadmap_doc += f"- [{issue['title']}]({issue['url']}) ({issue['repository']})\n"
            
            # Save roadmap
            roadmap_file = os.path.join(self.work_dir, "roadmap.md")
            with open(roadmap_file, "w") as f:
                f.write(roadmap_doc)
            
            return {
                "status": "success",
                "message": "Roadmap generated successfully",
                "roadmap_file": roadmap_file,
                "roadmap": roadmap
            }
            
        except Exception as e:
            logger.error(f"Error generating roadmap: {e}")
            return {"error": str(e)}
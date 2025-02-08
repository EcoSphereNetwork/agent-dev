"""Code Reviewer Agent for reviewing code changes."""
from typing import Dict, Any, List, Optional
import logging
import os
import json
import ast
import difflib
from langchain.agents import Tool
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class CodeReviewerAgent(BaseAgent):
    """Agent for reviewing code changes."""
    
    def __init__(
        self,
        github_token: str,
        github_org: str,
        github_repo: str,
        review_checklist: Optional[List[str]] = None,
        **kwargs
    ):
        """Initialize the code reviewer agent.
        
        Args:
            github_token: GitHub API token
            github_org: GitHub organization name
            github_repo: GitHub repository name
            review_checklist: Optional review checklist items
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(role="code-reviewer", **kwargs)
        self.github_token = github_token
        self.github_org = github_org
        self.github_repo = github_repo
        self.review_checklist = review_checklist or [
            "code_style",
            "best_practices",
            "security",
            "performance",
            "tests",
            "documentation"
        ]
        self.setup_review_tools()
    
    def setup_review_tools(self):
        """Set up code review tools."""
        self.add_tools([
            Tool(
                name="analyze_changes",
                func=self.analyze_changes,
                description="Analyze code changes"
            ),
            Tool(
                name="review_pull_request",
                func=self.review_pull_request,
                description="Review pull request"
            ),
            Tool(
                name="suggest_improvements",
                func=self.suggest_improvements,
                description="Suggest code improvements"
            ),
            Tool(
                name="check_guidelines",
                func=self.check_guidelines,
                description="Check coding guidelines"
            ),
            Tool(
                name="generate_summary",
                func=self.generate_summary,
                description="Generate review summary"
            )
        ])
    
    def analyze_changes(
        self,
        old_code: str,
        new_code: str
    ) -> Dict[str, Any]:
        """Analyze code changes.
        
        Args:
            old_code: Original code
            new_code: Modified code
        
        Returns:
            Analysis results
        """
        try:
            # Parse code
            old_tree = ast.parse(old_code)
            new_tree = ast.parse(new_code)
            
            # Analyze changes
            changes = {
                "added": {
                    "classes": [],
                    "functions": [],
                    "imports": []
                },
                "modified": {
                    "classes": [],
                    "functions": [],
                    "imports": []
                },
                "removed": {
                    "classes": [],
                    "functions": [],
                    "imports": []
                }
            }
            
            # Get items from old code
            old_items = {
                "classes": {
                    node.name: node
                    for node in ast.walk(old_tree)
                    if isinstance(node, ast.ClassDef)
                },
                "functions": {
                    node.name: node
                    for node in ast.walk(old_tree)
                    if isinstance(node, ast.FunctionDef)
                },
                "imports": {
                    node.names[0].name: node
                    for node in ast.walk(old_tree)
                    if isinstance(node, (ast.Import, ast.ImportFrom))
                }
            }
            
            # Compare with new code
            for node in ast.walk(new_tree):
                if isinstance(node, ast.ClassDef):
                    if node.name not in old_items["classes"]:
                        changes["added"]["classes"].append(node.name)
                    else:
                        # Check if class was modified
                        old_class = old_items["classes"][node.name]
                        if ast.dump(node) != ast.dump(old_class):
                            changes["modified"]["classes"].append(node.name)
                        del old_items["classes"][node.name]
                
                elif isinstance(node, ast.FunctionDef):
                    if node.name not in old_items["functions"]:
                        changes["added"]["functions"].append(node.name)
                    else:
                        # Check if function was modified
                        old_func = old_items["functions"][node.name]
                        if ast.dump(node) != ast.dump(old_func):
                            changes["modified"]["functions"].append(node.name)
                        del old_items["functions"][node.name]
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_name = node.names[0].name
                    if import_name not in old_items["imports"]:
                        changes["added"]["imports"].append(import_name)
                    else:
                        # Check if import was modified
                        old_import = old_items["imports"][import_name]
                        if ast.dump(node) != ast.dump(old_import):
                            changes["modified"]["imports"].append(import_name)
                        del old_items["imports"][import_name]
            
            # Remaining items were removed
            changes["removed"]["classes"] = list(old_items["classes"].keys())
            changes["removed"]["functions"] = list(old_items["functions"].keys())
            changes["removed"]["imports"] = list(old_items["imports"].keys())
            
            # Calculate diff
            diff = list(difflib.unified_diff(
                old_code.splitlines(),
                new_code.splitlines(),
                lineterm=""
            ))
            
            # Analyze complexity changes
            old_complexity = self._calculate_complexity(old_tree)
            new_complexity = self._calculate_complexity(new_tree)
            
            analysis = {
                "changes": changes,
                "diff": diff,
                "complexity_delta": new_complexity - old_complexity,
                "lines_added": len([l for l in diff if l.startswith("+")]),
                "lines_removed": len([l for l in diff if l.startswith("-")])
            }
            
            return {
                "status": "success",
                "message": "Code changes analyzed",
                "analysis": analysis
            }
            
        except Exception as e:
            logger.error(f"Error analyzing changes: {e}")
            return {"error": str(e)}
    
    def review_pull_request(
        self,
        pr_number: int
    ) -> Dict[str, Any]:
        """Review pull request.
        
        Args:
            pr_number: Pull request number
        
        Returns:
            Review results
        """
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            # Get PR details
            url = f"https://api.github.com/repos/{self.github_org}/{self.github_repo}/pulls/{pr_number}"
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            pr = response.json()
            
            # Get PR files
            files_url = f"https://api.github.com/repos/{self.github_org}/{self.github_repo}/pulls/{pr_number}/files"
            response = requests.get(files_url, headers=headers)
            response.raise_for_status()
            files = response.json()
            
            review_comments = []
            
            for file in files:
                if not file["filename"].endswith(".py"):
                    continue
                
                # Get file content before and after changes
                if file["status"] != "added":
                    response = requests.get(file["contents_url"], headers=headers)
                    response.raise_for_status()
                    old_content = response.json()["content"]
                else:
                    old_content = ""
                
                response = requests.get(
                    f"https://raw.githubusercontent.com/{self.github_org}/{self.github_repo}/{pr['head']['ref']}/{file['filename']}",
                    headers=headers
                )
                new_content = response.text
                
                # Analyze changes
                analysis = self.analyze_changes(old_content, new_content)
                if "error" in analysis:
                    continue
                
                # Review changes
                for item in self.review_checklist:
                    if item == "code_style":
                        # Check code style
                        style_issues = self._check_code_style(new_content)
                        for issue in style_issues:
                            review_comments.append({
                                "path": file["filename"],
                                "line": issue["line"],
                                "body": f"Style issue: {issue['message']}"
                            })
                    
                    elif item == "best_practices":
                        # Check best practices
                        practice_issues = self._check_best_practices(new_content)
                        for issue in practice_issues:
                            review_comments.append({
                                "path": file["filename"],
                                "line": issue["line"],
                                "body": f"Best practice: {issue['message']}"
                            })
                    
                    elif item == "security":
                        # Check security
                        security_issues = self._check_security(new_content)
                        for issue in security_issues:
                            review_comments.append({
                                "path": file["filename"],
                                "line": issue["line"],
                                "body": f"Security issue: {issue['message']}"
                            })
                    
                    elif item == "performance":
                        # Check performance
                        performance_issues = self._check_performance(new_content)
                        for issue in performance_issues:
                            review_comments.append({
                                "path": file["filename"],
                                "line": issue["line"],
                                "body": f"Performance issue: {issue['message']}"
                            })
                    
                    elif item == "tests":
                        # Check tests
                        test_issues = self._check_tests(new_content)
                        for issue in test_issues:
                            review_comments.append({
                                "path": file["filename"],
                                "line": issue["line"],
                                "body": f"Test issue: {issue['message']}"
                            })
                    
                    elif item == "documentation":
                        # Check documentation
                        doc_issues = self._check_documentation(new_content)
                        for issue in doc_issues:
                            review_comments.append({
                                "path": file["filename"],
                                "line": issue["line"],
                                "body": f"Documentation issue: {issue['message']}"
                            })
            
            # Submit review
            review_url = f"https://api.github.com/repos/{self.github_org}/{self.github_repo}/pulls/{pr_number}/reviews"
            
            review_body = f"""## Code Review

I've reviewed the changes and here are my findings:

### Summary
- Files changed: {len(files)}
- Lines added: {sum(f.get('additions', 0) for f in files)}
- Lines removed: {sum(f.get('deletions', 0) for f in files)}

### Issues Found
{chr(10).join(f'- {c["body"]} ({c["path"]}, line {c["line"]})' for c in review_comments)}

### Recommendations
{self._generate_recommendations(review_comments)}
"""
            
            review_data = {
                "commit_id": pr["head"]["sha"],
                "body": review_body,
                "event": "COMMENT" if review_comments else "APPROVE",
                "comments": review_comments
            }
            
            response = requests.post(review_url, json=review_data, headers=headers)
            response.raise_for_status()
            
            return {
                "status": "success",
                "message": "Pull request reviewed",
                "review": review_data,
                "comments": review_comments
            }
            
        except Exception as e:
            logger.error(f"Error reviewing pull request: {e}")
            return {"error": str(e)}
    
    def suggest_improvements(
        self,
        code: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Suggest code improvements.
        
        Args:
            code: Code to improve
            context: Optional context about the code
        
        Returns:
            Improvement suggestions
        """
        try:
            # Parse code
            tree = ast.parse(code)
            
            suggestions = []
            
            # Use LLM to suggest improvements
            prompt = f"""Suggest improvements for this code:

{code}

{f'Context: {context}' if context else ''}

Focus on:
1. Code readability
2. Error handling
3. Performance optimization
4. Security best practices
5. Testing considerations

Return specific suggestions with code examples."""
            
            result = self.run(prompt)
            if "error" not in result:
                suggestions.extend(result["result"].split("\n"))
            
            # Add specific suggestions based on static analysis
            complexity = self._calculate_complexity(tree)
            if complexity > 10:
                suggestions.append(
                    f"Consider breaking down complex code (complexity: {complexity})"
                )
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Try) and not node.handlers:
                    suggestions.append(
                        "Add proper error handling to try blocks"
                    )
                elif isinstance(node, ast.FunctionDef) and len(node.body) > 50:
                    suggestions.append(
                        f"Consider splitting large function {node.name}"
                    )
            
            return {
                "status": "success",
                "message": "Improvement suggestions generated",
                "suggestions": suggestions
            }
            
        except Exception as e:
            logger.error(f"Error suggesting improvements: {e}")
            return {"error": str(e)}
    
    def check_guidelines(
        self,
        code: str,
        guidelines: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Check coding guidelines.
        
        Args:
            code: Code to check
            guidelines: Optional list of guidelines to check
        
        Returns:
            Guideline check results
        """
        try:
            # Default guidelines
            if not guidelines:
                guidelines = [
                    "pep8",
                    "docstrings",
                    "imports",
                    "naming",
                    "complexity"
                ]
            
            results = {}
            
            for guideline in guidelines:
                if guideline == "pep8":
                    # Check PEP 8 compliance
                    import pycodestyle
                    style_guide = pycodestyle.StyleGuide()
                    result = style_guide.input_file(code)
                    results["pep8"] = {
                        "passed": result.total_errors == 0,
                        "errors": result.messages
                    }
                
                elif guideline == "docstrings":
                    # Check docstring presence and format
                    tree = ast.parse(code)
                    missing_docs = []
                    
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                            if not ast.get_docstring(node):
                                missing_docs.append(node.name)
                    
                    results["docstrings"] = {
                        "passed": len(missing_docs) == 0,
                        "missing": missing_docs
                    }
                
                elif guideline == "imports":
                    # Check import organization
                    tree = ast.parse(code)
                    import_issues = []
                    
                    imports = [
                        node for node in ast.walk(tree)
                        if isinstance(node, (ast.Import, ast.ImportFrom))
                    ]
                    
                    # Check import order
                    current_group = None
                    for imp in imports:
                        if isinstance(imp, ast.Import):
                            module = imp.names[0].name.split(".")[0]
                        else:
                            module = imp.module.split(".")[0]
                        
                        if module in ["typing", "abc"]:
                            new_group = "typing"
                        elif module in ["os", "sys", "io"]:
                            new_group = "stdlib"
                        else:
                            new_group = "third_party"
                        
                        if current_group and new_group != current_group:
                            import_issues.append(
                                f"Import groups not separated: {module}"
                            )
                        
                        current_group = new_group
                    
                    results["imports"] = {
                        "passed": len(import_issues) == 0,
                        "issues": import_issues
                    }
                
                elif guideline == "naming":
                    # Check naming conventions
                    tree = ast.parse(code)
                    naming_issues = []
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            if not node.name[0].isupper():
                                naming_issues.append(
                                    f"Class {node.name} should use CapWords"
                                )
                        elif isinstance(node, ast.FunctionDef):
                            if not node.name.islower():
                                naming_issues.append(
                                    f"Function {node.name} should use lowercase"
                                )
                        elif isinstance(node, ast.Name):
                            if node.id.isupper() and not (
                                isinstance(node.ctx, ast.Store) and
                                isinstance(node.parent, ast.Assign)
                            ):
                                naming_issues.append(
                                    f"Variable {node.id} should not be UPPERCASE"
                                )
                    
                    results["naming"] = {
                        "passed": len(naming_issues) == 0,
                        "issues": naming_issues
                    }
                
                elif guideline == "complexity":
                    # Check code complexity
                    tree = ast.parse(code)
                    complexity_issues = []
                    
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            complexity = self._calculate_complexity(node)
                            if complexity > 10:
                                complexity_issues.append({
                                    "function": node.name,
                                    "complexity": complexity
                                })
                    
                    results["complexity"] = {
                        "passed": len(complexity_issues) == 0,
                        "issues": complexity_issues
                    }
            
            # Calculate overall compliance
            total_checks = len(guidelines)
            passed_checks = sum(
                1 for g in guidelines
                if results[g]["passed"]
            )
            
            return {
                "status": "success",
                "message": "Guidelines checked",
                "compliance": (passed_checks / total_checks * 100),
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error checking guidelines: {e}")
            return {"error": str(e)}
    
    def generate_summary(
        self,
        review_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate review summary.
        
        Args:
            review_results: Results from review
        
        Returns:
            Generated summary
        """
        try:
            # Group issues by type
            issues = {}
            for comment in review_results.get("comments", []):
                issue_type = comment["body"].split(":")[0].strip()
                if issue_type not in issues:
                    issues[issue_type] = []
                issues[issue_type].append(comment)
            
            # Generate summary
            summary = f"""# Code Review Summary

## Overview
- Total comments: {len(review_results.get('comments', []))}
- Review status: {review_results.get('review', {}).get('event', 'COMMENT')}

## Issues by Type
"""
            
            for issue_type, comments in issues.items():
                summary += f"""
### {issue_type}
{chr(10).join(f'- {c["body"]} ({c["path"]}, line {c["line"]})' for c in comments)}
"""
            
            # Add recommendations
            summary += f"""
## Recommendations
{self._generate_recommendations(review_results.get('comments', []))}
"""
            
            # Save summary
            summary_dir = os.path.join(self.work_dir, "review_summaries")
            os.makedirs(summary_dir, exist_ok=True)
            
            summary_file = os.path.join(
                summary_dir,
                f"review_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
            
            with open(summary_file, "w") as f:
                f.write(summary)
            
            return {
                "status": "success",
                "message": "Review summary generated",
                "summary": summary,
                "summary_file": summary_file
            }
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {"error": str(e)}
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try,
                                ast.ExceptHandler, ast.With, ast.Assert,
                                ast.Raise, ast.Return)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _check_code_style(self, code: str) -> List[Dict[str, Any]]:
        """Check code style."""
        issues = []
        
        # Use LLM to check style
        prompt = f"""Check code style issues in this code:

{code}

Focus on:
1. Naming conventions
2. Indentation
3. Line length
4. Whitespace
5. Comments

Return a list of issues with line numbers."""
        
        result = self.run(prompt)
        if "error" not in result:
            for line in result["result"].split("\n"):
                if ":" in line:
                    line_num, message = line.split(":", 1)
                    issues.append({
                        "line": int(line_num),
                        "message": message.strip()
                    })
        
        return issues
    
    def _check_best_practices(self, code: str) -> List[Dict[str, Any]]:
        """Check best practices."""
        issues = []
        
        # Use LLM to check best practices
        prompt = f"""Check best practice issues in this code:

{code}

Focus on:
1. Code organization
2. Function design
3. Class design
4. Error handling
5. Resource management

Return a list of issues with line numbers."""
        
        result = self.run(prompt)
        if "error" not in result:
            for line in result["result"].split("\n"):
                if ":" in line:
                    line_num, message = line.split(":", 1)
                    issues.append({
                        "line": int(line_num),
                        "message": message.strip()
                    })
        
        return issues
    
    def _check_security(self, code: str) -> List[Dict[str, Any]]:
        """Check security issues."""
        issues = []
        
        # Use LLM to check security
        prompt = f"""Check security issues in this code:

{code}

Focus on:
1. Input validation
2. Authentication
3. Authorization
4. Data handling
5. Dependencies

Return a list of issues with line numbers."""
        
        result = self.run(prompt)
        if "error" not in result:
            for line in result["result"].split("\n"):
                if ":" in line:
                    line_num, message = line.split(":", 1)
                    issues.append({
                        "line": int(line_num),
                        "message": message.strip()
                    })
        
        return issues
    
    def _check_performance(self, code: str) -> List[Dict[str, Any]]:
        """Check performance issues."""
        issues = []
        
        # Use LLM to check performance
        prompt = f"""Check performance issues in this code:

{code}

Focus on:
1. Algorithm efficiency
2. Memory usage
3. I/O operations
4. Resource management
5. Caching

Return a list of issues with line numbers."""
        
        result = self.run(prompt)
        if "error" not in result:
            for line in result["result"].split("\n"):
                if ":" in line:
                    line_num, message = line.split(":", 1)
                    issues.append({
                        "line": int(line_num),
                        "message": message.strip()
                    })
        
        return issues
    
    def _check_tests(self, code: str) -> List[Dict[str, Any]]:
        """Check test issues."""
        issues = []
        
        # Use LLM to check tests
        prompt = f"""Check test issues in this code:

{code}

Focus on:
1. Test coverage
2. Test design
3. Assertions
4. Edge cases
5. Test isolation

Return a list of issues with line numbers."""
        
        result = self.run(prompt)
        if "error" not in result:
            for line in result["result"].split("\n"):
                if ":" in line:
                    line_num, message = line.split(":", 1)
                    issues.append({
                        "line": int(line_num),
                        "message": message.strip()
                    })
        
        return issues
    
    def _check_documentation(self, code: str) -> List[Dict[str, Any]]:
        """Check documentation issues."""
        issues = []
        
        # Use LLM to check documentation
        prompt = f"""Check documentation issues in this code:

{code}

Focus on:
1. Docstrings
2. Comments
3. Type hints
4. Examples
5. API documentation

Return a list of issues with line numbers."""
        
        result = self.run(prompt)
        if "error" not in result:
            for line in result["result"].split("\n"):
                if ":" in line:
                    line_num, message = line.split(":", 1)
                    issues.append({
                        "line": int(line_num),
                        "message": message.strip()
                    })
        
        return issues
    
    def _generate_recommendations(
        self,
        comments: List[Dict[str, Any]]
    ) -> str:
        """Generate recommendations from review comments."""
        if not comments:
            return "No issues found. Code looks good! 👍"
        
        # Group comments by type
        grouped = {}
        for comment in comments:
            issue_type = comment["body"].split(":")[0].strip()
            if issue_type not in grouped:
                grouped[issue_type] = []
            grouped[issue_type].append(comment)
        
        # Generate recommendations
        recommendations = []
        
        for issue_type, type_comments in grouped.items():
            if len(type_comments) >= 3:
                recommendations.append(
                    f"Consider focusing on {issue_type.lower()} issues"
                )
        
        if not recommendations:
            recommendations.append(
                "Address individual issues as noted in the comments"
            )
        
        return "\n".join(f"- {r}" for r in recommendations)
    
    def get_review_checklist(self) -> List[str]:
        """Get the review checklist.
        
        Returns:
            Review checklist items
        """
        return self.review_checklist
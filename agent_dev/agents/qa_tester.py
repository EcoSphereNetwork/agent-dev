"""QA Tester Agent for quality assurance and testing."""
from typing import Dict, Any, List, Optional
import logging
import os
import json
import subprocess
import ast
from langchain.agents import Tool
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class QATesterAgent(BaseAgent):
    """Agent for quality assurance and testing."""
    
    def __init__(
        self,
        github_token: str,
        github_org: str,
        github_repo: str,
        test_frameworks: Optional[List[str]] = None,
        coverage_threshold: float = 80.0,
        **kwargs
    ):
        """Initialize the QA tester agent.
        
        Args:
            github_token: GitHub API token
            github_org: GitHub organization name
            github_repo: GitHub repository name
            test_frameworks: Optional list of test frameworks to use
            coverage_threshold: Minimum test coverage percentage
            **kwargs: Additional arguments for BaseAgent
        """
        super().__init__(role="qa-tester", **kwargs)
        self.github_token = github_token
        self.github_org = github_org
        self.github_repo = github_repo
        self.test_frameworks = test_frameworks or ["pytest", "unittest"]
        self.coverage_threshold = coverage_threshold
        self.setup_qa_tools()
    
    def setup_qa_tools(self):
        """Set up QA testing tools."""
        self.add_tools([
            Tool(
                name="analyze_tests",
                func=self.analyze_tests,
                description="Analyze test coverage and quality"
            ),
            Tool(
                name="generate_tests",
                func=self.generate_tests,
                description="Generate test cases"
            ),
            Tool(
                name="run_tests",
                func=self.run_tests,
                description="Run test suite"
            ),
            Tool(
                name="report_bugs",
                func=self.report_bugs,
                description="Report bugs as issues"
            ),
            Tool(
                name="verify_fixes",
                func=self.verify_fixes,
                description="Verify bug fixes"
            )
        ])
    
    def analyze_tests(
        self,
        module_path: str
    ) -> Dict[str, Any]:
        """Analyze test coverage and quality.
        
        Args:
            module_path: Path to Python module
        
        Returns:
            Analysis results
        """
        try:
            if not os.path.exists(module_path):
                return {
                    "status": "error",
                    "error": f"Module {module_path} not found"
                }
            
            # Run coverage
            import coverage
            cov = coverage.Coverage()
            cov.start()
            
            try:
                # Run tests
                if "pytest" in self.test_frameworks:
                    subprocess.run(
                        ["pytest", module_path],
                        capture_output=True,
                        text=True
                    )
                elif "unittest" in self.test_frameworks:
                    subprocess.run(
                        ["python", "-m", "unittest", "discover", module_path],
                        capture_output=True,
                        text=True
                    )
            finally:
                cov.stop()
                cov.save()
            
            # Get coverage data
            coverage_data = cov.report(show_missing=True)
            
            # Parse test file
            test_file = module_path.replace(".py", "_test.py")
            if not os.path.exists(test_file):
                test_file = module_path.replace(".py", "/test_" + os.path.basename(module_path))
            
            test_metrics = {
                "test_count": 0,
                "assertion_count": 0,
                "parameterized_tests": 0,
                "mock_usage": 0,
                "fixture_usage": 0
            }
            
            if os.path.exists(test_file):
                with open(test_file, "r") as f:
                    test_code = f.read()
                tree = ast.parse(test_code)
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and (
                        node.name.startswith("test_") or
                        hasattr(node, "decorator_list") and any(
                            d.id == "pytest" for d in node.decorator_list
                            if isinstance(d, ast.Name)
                        )
                    ):
                        test_metrics["test_count"] += 1
                        
                        # Count assertions
                        assertions = [
                            n for n in ast.walk(node)
                            if isinstance(n, ast.Assert) or (
                                isinstance(n, ast.Call) and
                                isinstance(n.func, ast.Attribute) and
                                n.func.attr.startswith("assert")
                            )
                        ]
                        test_metrics["assertion_count"] += len(assertions)
                    
                    # Check for parameterization
                    if isinstance(node, ast.FunctionDef) and any(
                        isinstance(d, ast.Call) and
                        isinstance(d.func, ast.Attribute) and
                        d.func.attr == "parametrize"
                        for d in node.decorator_list
                    ):
                        test_metrics["parameterized_tests"] += 1
                    
                    # Check for mocks
                    if isinstance(node, (ast.Call, ast.Name)) and any(
                        name in str(node) for name in ["Mock", "patch", "MagicMock"]
                    ):
                        test_metrics["mock_usage"] += 1
                    
                    # Check for fixtures
                    if isinstance(node, ast.FunctionDef) and any(
                        isinstance(d, ast.Name) and d.id == "fixture"
                        for d in node.decorator_list
                    ):
                        test_metrics["fixture_usage"] += 1
            
            # Calculate quality metrics
            quality_metrics = {
                "coverage": coverage_data,
                "meets_threshold": coverage_data >= self.coverage_threshold,
                "test_density": test_metrics["test_count"] / (
                    len(list(ast.walk(ast.parse(open(module_path).read()))))
                    if os.path.exists(module_path) else 0
                ),
                "assertion_density": test_metrics["assertion_count"] / test_metrics["test_count"]
                if test_metrics["test_count"] > 0 else 0
            }
            
            # Generate recommendations
            recommendations = []
            
            if not quality_metrics["meets_threshold"]:
                recommendations.append({
                    "type": "coverage",
                    "message": f"Increase test coverage to meet {self.coverage_threshold}% threshold"
                })
            
            if quality_metrics["test_density"] < 0.2:
                recommendations.append({
                    "type": "test_density",
                    "message": "Add more test cases to improve test density"
                })
            
            if quality_metrics["assertion_density"] < 2:
                recommendations.append({
                    "type": "assertion_density",
                    "message": "Add more assertions to existing test cases"
                })
            
            if test_metrics["parameterized_tests"] == 0:
                recommendations.append({
                    "type": "parameterization",
                    "message": "Consider adding parameterized tests for better coverage"
                })
            
            # Save analysis
            analysis = {
                "module": module_path,
                "test_metrics": test_metrics,
                "quality_metrics": quality_metrics,
                "recommendations": recommendations
            }
            
            analysis_dir = os.path.join(self.work_dir, "qa_analysis")
            os.makedirs(analysis_dir, exist_ok=True)
            
            analysis_file = os.path.join(
                analysis_dir,
                f"{os.path.basename(module_path)}.qa.json"
            )
            
            with open(analysis_file, "w") as f:
                json.dump(analysis, f, indent=2)
            
            return {
                "status": "success",
                "message": f"QA analysis completed for {module_path}",
                "analysis": analysis,
                "analysis_file": analysis_file
            }
            
        except Exception as e:
            logger.error(f"Error analyzing tests: {e}")
            return {"error": str(e)}
    
    def generate_tests(
        self,
        module_path: str,
        analysis_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate test cases.
        
        Args:
            module_path: Path to Python module
            analysis_file: Optional path to analysis file
        
        Returns:
            Generated tests
        """
        try:
            # Get or create analysis
            if analysis_file and os.path.exists(analysis_file):
                with open(analysis_file, "r") as f:
                    analysis = json.load(f)
            else:
                result = self.analyze_tests(module_path)
                if "error" in result:
                    return result
                analysis = result["analysis"]
            
            # Read module code
            with open(module_path, "r") as f:
                code = f.read()
            
            # Parse code to find testable items
            tree = ast.parse(code)
            testable_items = {
                "classes": [],
                "functions": [],
                "methods": []
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    testable_items["classes"].append({
                        "name": node.name,
                        "methods": [
                            n.name for n in node.body
                            if isinstance(n, ast.FunctionDef)
                        ]
                    })
                elif isinstance(node, ast.FunctionDef):
                    if not any(
                        node.name in cls["methods"]
                        for cls in testable_items["classes"]
                    ):
                        testable_items["functions"].append(node.name)
            
            # Generate test file path
            test_dir = os.path.dirname(module_path).replace("src", "tests")
            os.makedirs(test_dir, exist_ok=True)
            
            test_file = os.path.join(
                test_dir,
                f"test_{os.path.basename(module_path)}"
            )
            
            # Generate test code
            if "pytest" in self.test_frameworks:
                test_code = f"""\"\"\"Tests for {os.path.basename(module_path)}.\"\"\"
import pytest
from unittest.mock import Mock, patch
from {os.path.splitext(os.path.basename(module_path))[0]} import *

"""
                
                # Add fixtures
                test_code += """@pytest.fixture
def setup():
    \"\"\"Test fixture for setup.\"\"\"
    # Add setup code
    yield
    # Add teardown code

"""
                
                # Generate class tests
                for cls in testable_items["classes"]:
                    test_code += f"""class Test{cls['name']}:
    \"\"\"Tests for {cls['name']} class.\"\"\"
    
"""
                    
                    # Use LLM to generate test methods
                    prompt = f"""Generate pytest test methods for this class:

class {cls['name']}:
    {chr(10).join(f'def {method}(self):' for method in cls['methods'])}

Include:
1. Happy path tests
2. Edge cases
3. Error cases
4. Parameterized tests
5. Mock usage where appropriate

Return only the test code without explanations."""
                    
                    result = self.run(prompt)
                    if "error" not in result:
                        test_code += result["result"] + "\n\n"
                
                # Generate function tests
                for func in testable_items["functions"]:
                    # Use LLM to generate test function
                    prompt = f"""Generate a pytest test function for:

def {func}():
    pass

Include:
1. Happy path test
2. Edge cases
3. Error cases
4. Parameterized tests if applicable
5. Mock usage if needed

Return only the test code without explanations."""
                    
                    result = self.run(prompt)
                    if "error" not in result:
                        test_code += result["result"] + "\n\n"
                
            else:  # unittest
                test_code = f"""\"\"\"Tests for {os.path.basename(module_path)}.\"\"\"
import unittest
from unittest.mock import Mock, patch
from {os.path.splitext(os.path.basename(module_path))[0]} import *

"""
                
                # Generate class tests
                for cls in testable_items["classes"]:
                    test_code += f"""class Test{cls['name']}(unittest.TestCase):
    \"\"\"Tests for {cls['name']} class.\"\"\"
    
    def setUp(self):
        \"\"\"Set up test cases.\"\"\"
        # Add setup code
        pass
    
    def tearDown(self):
        \"\"\"Clean up after tests.\"\"\"
        # Add cleanup code
        pass
    
"""
                    
                    # Use LLM to generate test methods
                    prompt = f"""Generate unittest test methods for this class:

class {cls['name']}:
    {chr(10).join(f'def {method}(self):' for method in cls['methods'])}

Include:
1. Happy path tests
2. Edge cases
3. Error cases
4. Mock usage where appropriate

Return only the test code without explanations."""
                    
                    result = self.run(prompt)
                    if "error" not in result:
                        test_code += result["result"] + "\n\n"
                
                # Generate function tests
                if testable_items["functions"]:
                    test_code += """class TestFunctions(unittest.TestCase):
    \"\"\"Tests for standalone functions.\"\"\"
    
"""
                    
                    for func in testable_items["functions"]:
                        # Use LLM to generate test method
                        prompt = f"""Generate unittest test methods for:

def {func}():
    pass

Include:
1. Happy path test
2. Edge cases
3. Error cases
4. Mock usage if needed

Return only the test code without explanations."""
                        
                        result = self.run(prompt)
                        if "error" not in result:
                            test_code += result["result"] + "\n\n"
                
                test_code += """
if __name__ == '__main__':
    unittest.main()
"""
            
            # Save test code
            with open(test_file, "w") as f:
                f.write(test_code)
            
            # Commit changes
            self.commit_changes(
                message=f"""Add tests for {os.path.basename(module_path)}

Generated tests for:
{chr(10).join(f'- {cls["name"]}' for cls in testable_items["classes"])}
{chr(10).join(f'- {func}' for func in testable_items["functions"])}""",
                files=[test_file]
            )
            
            return {
                "status": "success",
                "message": f"Tests generated for {module_path}",
                "test_file": test_file,
                "testable_items": testable_items
            }
            
        except Exception as e:
            logger.error(f"Error generating tests: {e}")
            return {"error": str(e)}
    
    def run_tests(
        self,
        path: str,
        coverage: bool = True,
        parallel: bool = True
    ) -> Dict[str, Any]:
        """Run test suite.
        
        Args:
            path: Path to test directory or file
            coverage: Whether to collect coverage data
            parallel: Whether to run tests in parallel
        
        Returns:
            Test results
        """
        try:
            results_dir = os.path.join(self.work_dir, "test_results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Build test command
            if "pytest" in self.test_frameworks:
                cmd = ["pytest"]
                if coverage:
                    cmd.extend(["--cov", "--cov-report=html:coverage_html"])
                if parallel:
                    cmd.append("-n=auto")
                cmd.extend(["-v", path])
            else:  # unittest
                cmd = ["python", "-m", "unittest", "discover", "-v"]
                if coverage:
                    cmd = ["coverage", "run", "-m", "unittest", "discover", "-v"]
                cmd.append(path)
            
            # Run tests
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Parse results
            test_results = {
                "command": " ".join(cmd),
                "exit_code": result.returncode,
                "output": result.stdout,
                "errors": result.stderr if result.returncode != 0 else None,
                "coverage_report": None
            }
            
            # Generate coverage report if enabled
            if coverage:
                if "pytest" in self.test_frameworks:
                    test_results["coverage_report"] = "coverage_html/index.html"
                else:
                    subprocess.run(
                        ["coverage", "html"],
                        capture_output=True,
                        text=True
                    )
                    test_results["coverage_report"] = "htmlcov/index.html"
            
            # Save results
            results_file = os.path.join(
                results_dir,
                f"test_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            
            with open(results_file, "w") as f:
                json.dump(test_results, f, indent=2)
            
            return {
                "status": "success" if result.returncode == 0 else "failure",
                "message": "Tests completed",
                "results": test_results,
                "results_file": results_file
            }
            
        except Exception as e:
            logger.error(f"Error running tests: {e}")
            return {"error": str(e)}
    
    def report_bugs(
        self,
        bugs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Report bugs as GitHub issues.
        
        Args:
            bugs: List of bug reports
        
        Returns:
            Created issues
        """
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            created_issues = []
            for bug in bugs:
                # Create issue body
                body = f"""## Bug Report

### Description
{bug['description']}

### Steps to Reproduce
{chr(10).join(f'{i+1}. {step}' for i, step in enumerate(bug['steps']))}

### Expected Behavior
{bug['expected']}

### Actual Behavior
{bug['actual']}

### Test Case
```python
{bug.get('test_case', 'No test case provided')}
```

### Additional Information
- Priority: {bug.get('priority', 'medium')}
- Component: {bug.get('component', 'unknown')}
- Version: {bug.get('version', 'latest')}
"""
                
                # Create issue
                url = f"https://api.github.com/repos/{self.github_org}/{self.github_repo}/issues"
                data = {
                    "title": f"[BUG] {bug['title']}",
                    "body": body,
                    "labels": ["bug", f"priority-{bug.get('priority', 'medium')}"]
                }
                
                response = requests.post(url, json=data, headers=headers)
                response.raise_for_status()
                issue = response.json()
                
                created_issues.append({
                    "number": issue["number"],
                    "title": issue["title"],
                    "url": issue["html_url"]
                })
            
            return {
                "status": "success",
                "message": f"Created {len(created_issues)} bug reports",
                "issues": created_issues
            }
            
        except Exception as e:
            logger.error(f"Error reporting bugs: {e}")
            return {"error": str(e)}
    
    def verify_fixes(
        self,
        issues: List[int]
    ) -> Dict[str, Any]:
        """Verify bug fixes.
        
        Args:
            issues: List of issue numbers to verify
        
        Returns:
            Verification results
        """
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            results = []
            for issue_number in issues:
                # Get issue details
                url = f"https://api.github.com/repos/{self.github_org}/{self.github_repo}/issues/{issue_number}"
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                issue = response.json()
                
                # Extract test case from issue body
                import re
                test_match = re.search(
                    r"### Test Case\n```python\n(.*?)\n```",
                    issue["body"],
                    re.DOTALL
                )
                
                if not test_match:
                    results.append({
                        "issue": issue_number,
                        "title": issue["title"],
                        "status": "error",
                        "message": "No test case found in issue"
                    })
                    continue
                
                test_code = test_match.group(1)
                
                # Create temporary test file
                test_dir = os.path.join(self.work_dir, "verification_tests")
                os.makedirs(test_dir, exist_ok=True)
                
                test_file = os.path.join(test_dir, f"test_issue_{issue_number}.py")
                with open(test_file, "w") as f:
                    f.write(test_code)
                
                # Run test
                result = self.run_tests(test_file)
                
                verified = result["status"] == "success"
                
                # Update issue
                comment = "✅ Fix verified successfully" if verified else "❌ Fix verification failed"
                
                comment_url = f"https://api.github.com/repos/{self.github_org}/{self.github_repo}/issues/{issue_number}/comments"
                requests.post(
                    comment_url,
                    json={"body": comment},
                    headers=headers
                )
                
                # Update labels
                current_labels = [l["name"] for l in issue["labels"]]
                new_labels = [
                    l for l in current_labels
                    if l != "needs-verification"
                ]
                if verified:
                    new_labels.append("verified")
                
                requests.patch(
                    url,
                    json={"labels": new_labels},
                    headers=headers
                )
                
                results.append({
                    "issue": issue_number,
                    "title": issue["title"],
                    "verified": verified,
                    "test_results": result.get("results")
                })
            
            return {
                "status": "success",
                "message": f"Verified {len(results)} fixes",
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Error verifying fixes: {e}")
            return {"error": str(e)}
    
    def get_test_frameworks(self) -> List[str]:
        """Get the test frameworks.
        
        Returns:
            List of test frameworks
        """
        return self.test_frameworks
    
    def get_coverage_threshold(self) -> float:
        """Get the coverage threshold.
        
        Returns:
            Coverage threshold percentage
        """
        return self.coverage_threshold
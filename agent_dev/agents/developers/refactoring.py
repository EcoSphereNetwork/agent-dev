"""Refactoring Agent for improving code quality and architecture."""
from typing import Dict, Any, List, Optional
import logging
import os
import json
import ast
import importlib
import inspect
from langchain.agents import Tool
from .base_developer import BaseDeveloperAgent

logger = logging.getLogger(__name__)

class RefactoringAgent(BaseDeveloperAgent):
    """Agent specialized in code refactoring."""
    
    def __init__(
        self,
        code_metrics: Optional[Dict[str, float]] = None,
        **kwargs
    ):
        """Initialize the refactoring agent.
        
        Args:
            code_metrics: Optional code quality metrics thresholds
            **kwargs: Additional arguments for BaseDeveloperAgent
        """
        super().__init__(specialty="refactoring", **kwargs)
        self.code_metrics = code_metrics or {
            "max_complexity": 10,
            "max_line_length": 88,
            "max_function_length": 50,
            "min_test_coverage": 80
        }
        self.setup_refactoring_tools()
    
    def setup_refactoring_tools(self):
        """Set up refactoring specific tools."""
        self.add_tools([
            Tool(
                name="analyze_code",
                func=self.analyze_code,
                description="Analyze code quality"
            ),
            Tool(
                name="suggest_refactoring",
                func=self.suggest_refactoring,
                description="Suggest code improvements"
            ),
            Tool(
                name="apply_refactoring",
                func=self.apply_refactoring,
                description="Apply code improvements"
            ),
            Tool(
                name="verify_refactoring",
                func=self.verify_refactoring,
                description="Verify code improvements"
            ),
            Tool(
                name="generate_report",
                func=self.generate_report,
                description="Generate refactoring report"
            )
        ])
    
    def analyze_code(
        self,
        file_path: str
    ) -> Dict[str, Any]:
        """Analyze code quality.
        
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
            
            # Read file
            with open(file_path, "r") as f:
                code = f.read()
            
            # Parse AST
            tree = ast.parse(code)
            
            # Analyze code metrics
            metrics = {
                "complexity": self._calculate_complexity(tree),
                "line_length": max(len(line) for line in code.split("\n")),
                "function_lengths": {
                    node.name: len(node.body)
                    for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                },
                "class_count": len([
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.ClassDef)
                ]),
                "function_count": len([
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                ]),
                "import_count": len([
                    node for node in ast.walk(tree)
                    if isinstance(node, (ast.Import, ast.ImportFrom))
                ])
            }
            
            # Calculate test coverage if it's a test file
            if "test" in file_path:
                import coverage
                cov = coverage.Coverage()
                cov.start()
                
                try:
                    # Import and run tests
                    module_name = os.path.splitext(os.path.basename(file_path))[0]
                    spec = importlib.util.spec_from_file_location(module_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                finally:
                    cov.stop()
                    cov.save()
                
                metrics["test_coverage"] = cov.report()
            
            # Identify issues
            issues = []
            
            if metrics["complexity"] > self.code_metrics["max_complexity"]:
                issues.append({
                    "type": "complexity",
                    "message": f"Cyclomatic complexity {metrics['complexity']} exceeds threshold {self.code_metrics['max_complexity']}"
                })
            
            if metrics["line_length"] > self.code_metrics["max_line_length"]:
                issues.append({
                    "type": "line_length",
                    "message": f"Maximum line length {metrics['line_length']} exceeds threshold {self.code_metrics['max_line_length']}"
                })
            
            for func, length in metrics["function_lengths"].items():
                if length > self.code_metrics["max_function_length"]:
                    issues.append({
                        "type": "function_length",
                        "message": f"Function {func} length {length} exceeds threshold {self.code_metrics['max_function_length']}"
                    })
            
            if metrics.get("test_coverage", 100) < self.code_metrics["min_test_coverage"]:
                issues.append({
                    "type": "test_coverage",
                    "message": f"Test coverage {metrics['test_coverage']}% below threshold {self.code_metrics['min_test_coverage']}%"
                })
            
            # Save analysis
            analysis = {
                "file": file_path,
                "metrics": metrics,
                "issues": issues,
                "thresholds": self.code_metrics
            }
            
            analysis_dir = os.path.join(self.work_dir, "analysis")
            os.makedirs(analysis_dir, exist_ok=True)
            
            analysis_file = os.path.join(
                analysis_dir,
                f"{os.path.basename(file_path)}.analysis.json"
            )
            
            with open(analysis_file, "w") as f:
                json.dump(analysis, f, indent=2)
            
            return {
                "status": "success",
                "message": f"Code analysis completed for {file_path}",
                "analysis": analysis,
                "analysis_file": analysis_file
            }
            
        except Exception as e:
            logger.error(f"Error analyzing code: {e}")
            return {"error": str(e)}
    
    def suggest_refactoring(
        self,
        file_path: str,
        analysis_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Suggest code improvements.
        
        Args:
            file_path: Path to Python file
            analysis_file: Optional path to analysis file
        
        Returns:
            Refactoring suggestions
        """
        try:
            # Get or create analysis
            if analysis_file and os.path.exists(analysis_file):
                with open(analysis_file, "r") as f:
                    analysis = json.load(f)
            else:
                result = self.analyze_code(file_path)
                if "error" in result:
                    return result
                analysis = result["analysis"]
            
            # Read file
            with open(file_path, "r") as f:
                code = f.read()
            
            # Generate suggestions based on issues
            suggestions = []
            
            for issue in analysis["issues"]:
                if issue["type"] == "complexity":
                    # Use LLM to suggest complexity reduction
                    prompt = f"""Suggest ways to reduce complexity in this code:
{code}

Current complexity: {analysis['metrics']['complexity']}
Target complexity: {self.code_metrics['max_complexity']}

Focus on:
1. Breaking down complex functions
2. Simplifying conditional logic
3. Extracting helper methods
4. Using early returns

Return specific suggestions with code examples."""
                    
                    result = self.run(prompt)
                    if "error" not in result:
                        suggestions.append({
                            "type": "complexity",
                            "description": "Reduce code complexity",
                            "suggestions": result["result"].split("\n")
                        })
                
                elif issue["type"] == "line_length":
                    # Use LLM to suggest line length reduction
                    prompt = f"""Suggest ways to reduce line length in this code:
{code}

Current max length: {analysis['metrics']['line_length']}
Target length: {self.code_metrics['max_line_length']}

Focus on:
1. Breaking long expressions
2. Using line continuation
3. Extracting variables
4. Formatting string operations

Return specific suggestions with code examples."""
                    
                    result = self.run(prompt)
                    if "error" not in result:
                        suggestions.append({
                            "type": "line_length",
                            "description": "Reduce line length",
                            "suggestions": result["result"].split("\n")
                        })
                
                elif issue["type"] == "function_length":
                    # Use LLM to suggest function splitting
                    prompt = f"""Suggest ways to split long functions in this code:
{code}

Functions exceeding limit:
{chr(10).join(f'- {func}: {length} lines' for func, length in analysis['metrics']['function_lengths'].items() if length > self.code_metrics['max_function_length'])}

Target length: {self.code_metrics['max_function_length']}

Focus on:
1. Extracting helper functions
2. Grouping related operations
3. Using composition
4. Applying single responsibility principle

Return specific suggestions with code examples."""
                    
                    result = self.run(prompt)
                    if "error" not in result:
                        suggestions.append({
                            "type": "function_length",
                            "description": "Split long functions",
                            "suggestions": result["result"].split("\n")
                        })
                
                elif issue["type"] == "test_coverage":
                    # Use LLM to suggest test improvements
                    prompt = f"""Suggest ways to improve test coverage in this code:
{code}

Current coverage: {analysis['metrics'].get('test_coverage', 0)}%
Target coverage: {self.code_metrics['min_test_coverage']}%

Focus on:
1. Adding missing test cases
2. Testing edge cases
3. Adding error cases
4. Testing all code paths

Return specific suggestions with code examples."""
                    
                    result = self.run(prompt)
                    if "error" not in result:
                        suggestions.append({
                            "type": "test_coverage",
                            "description": "Improve test coverage",
                            "suggestions": result["result"].split("\n")
                        })
            
            # Save suggestions
            suggestions_dir = os.path.join(self.work_dir, "suggestions")
            os.makedirs(suggestions_dir, exist_ok=True)
            
            suggestions_file = os.path.join(
                suggestions_dir,
                f"{os.path.basename(file_path)}.suggestions.json"
            )
            
            with open(suggestions_file, "w") as f:
                json.dump(suggestions, f, indent=2)
            
            return {
                "status": "success",
                "message": f"Refactoring suggestions generated for {file_path}",
                "suggestions": suggestions,
                "suggestions_file": suggestions_file
            }
            
        except Exception as e:
            logger.error(f"Error suggesting refactoring: {e}")
            return {"error": str(e)}
    
    def apply_refactoring(
        self,
        file_path: str,
        suggestions_file: str
    ) -> Dict[str, Any]:
        """Apply code improvements.
        
        Args:
            file_path: Path to Python file
            suggestions_file: Path to suggestions file
        
        Returns:
            Status of the operation
        """
        try:
            if not os.path.exists(suggestions_file):
                return {
                    "status": "error",
                    "error": f"Suggestions file {suggestions_file} not found"
                }
            
            # Read suggestions
            with open(suggestions_file, "r") as f:
                suggestions = json.load(f)
            
            # Read original code
            with open(file_path, "r") as f:
                code = f.read()
            
            # Create refactoring branch
            branch_result = self.create_branch(
                f"refactor-{os.path.basename(file_path)}"
            )
            if "error" in branch_result:
                return branch_result
            
            # Apply suggestions
            improved_code = code
            applied_suggestions = []
            
            for suggestion in suggestions:
                # Use LLM to apply suggestion
                prompt = f"""Apply this refactoring suggestion to the code:

Code:
{improved_code}

Suggestion type: {suggestion['type']}
Description: {suggestion['description']}
Specific suggestions:
{chr(10).join(suggestion['suggestions'])}

Return only the refactored code without explanations."""
                
                result = self.run(prompt)
                if "error" not in result:
                    improved_code = result["result"]
                    applied_suggestions.append(suggestion)
            
            # Save improved code
            with open(file_path, "w") as f:
                f.write(improved_code)
            
            # Update tests if needed
            test_file = None
            if "test" not in file_path:
                test_dir = os.path.dirname(file_path).replace("src", "tests")
                test_file = os.path.join(
                    test_dir,
                    f"test_{os.path.basename(file_path)}"
                )
                
                if os.path.exists(test_file):
                    # Use LLM to update tests
                    with open(test_file, "r") as f:
                        test_code = f.read()
                    
                    prompt = f"""Update these tests for the refactored code:

Refactored code:
{improved_code}

Current tests:
{test_code}

Return only the updated test code without explanations."""
                    
                    result = self.run(prompt)
                    if "error" not in result:
                        with open(test_file, "w") as f:
                            f.write(result["result"])
            
            # Commit changes
            files = [file_path]
            if test_file:
                files.append(test_file)
            
            commit_result = self.commit_changes(
                message=f"""Refactor {os.path.basename(file_path)}

Applied suggestions:
{chr(10).join(f'- {s["description"]}' for s in applied_suggestions)}""",
                files=files
            )
            
            if "error" in commit_result:
                return commit_result
            
            # Create pull request
            pr_result = self.create_pull_request(
                title=f"Refactor {os.path.basename(file_path)}",
                body=f"""This PR applies the following refactoring suggestions:
{chr(10).join(f'- {s["description"]}' for s in applied_suggestions)}

Please review the changes and ensure all tests pass."""
            )
            
            return {
                "status": "success",
                "message": f"Refactoring applied to {file_path}",
                "applied_suggestions": applied_suggestions,
                "pull_request": pr_result.get("pr_url")
            }
            
        except Exception as e:
            logger.error(f"Error applying refactoring: {e}")
            return {"error": str(e)}
    
    def verify_refactoring(
        self,
        file_path: str,
        original_analysis_file: str
    ) -> Dict[str, Any]:
        """Verify code improvements.
        
        Args:
            file_path: Path to Python file
            original_analysis_file: Path to original analysis file
        
        Returns:
            Verification results
        """
        try:
            if not os.path.exists(original_analysis_file):
                return {
                    "status": "error",
                    "error": f"Original analysis file {original_analysis_file} not found"
                }
            
            # Get original analysis
            with open(original_analysis_file, "r") as f:
                original_analysis = json.load(f)
            
            # Analyze current code
            result = self.analyze_code(file_path)
            if "error" in result:
                return result
            current_analysis = result["analysis"]
            
            # Compare metrics
            improvements = {
                "complexity": {
                    "original": original_analysis["metrics"]["complexity"],
                    "current": current_analysis["metrics"]["complexity"],
                    "improved": current_analysis["metrics"]["complexity"] < original_analysis["metrics"]["complexity"]
                },
                "line_length": {
                    "original": original_analysis["metrics"]["line_length"],
                    "current": current_analysis["metrics"]["line_length"],
                    "improved": current_analysis["metrics"]["line_length"] < original_analysis["metrics"]["line_length"]
                },
                "function_lengths": {
                    "original": original_analysis["metrics"]["function_lengths"],
                    "current": current_analysis["metrics"]["function_lengths"],
                    "improved": all(
                        length <= self.code_metrics["max_function_length"]
                        for length in current_analysis["metrics"]["function_lengths"].values()
                    )
                }
            }
            
            if "test_coverage" in original_analysis["metrics"]:
                improvements["test_coverage"] = {
                    "original": original_analysis["metrics"]["test_coverage"],
                    "current": current_analysis["metrics"]["test_coverage"],
                    "improved": current_analysis["metrics"]["test_coverage"] > original_analysis["metrics"]["test_coverage"]
                }
            
            # Check if all issues are resolved
            remaining_issues = len(current_analysis["issues"])
            resolved_issues = len(original_analysis["issues"]) - remaining_issues
            
            verification = {
                "file": file_path,
                "improvements": improvements,
                "resolved_issues": resolved_issues,
                "remaining_issues": remaining_issues,
                "success": all(imp["improved"] for imp in improvements.values())
            }
            
            # Save verification
            verification_dir = os.path.join(self.work_dir, "verification")
            os.makedirs(verification_dir, exist_ok=True)
            
            verification_file = os.path.join(
                verification_dir,
                f"{os.path.basename(file_path)}.verification.json"
            )
            
            with open(verification_file, "w") as f:
                json.dump(verification, f, indent=2)
            
            return {
                "status": "success",
                "message": f"Refactoring verification completed for {file_path}",
                "verification": verification,
                "verification_file": verification_file
            }
            
        except Exception as e:
            logger.error(f"Error verifying refactoring: {e}")
            return {"error": str(e)}
    
    def generate_report(
        self,
        file_path: str,
        analysis_file: str,
        suggestions_file: str,
        verification_file: str
    ) -> Dict[str, Any]:
        """Generate refactoring report.
        
        Args:
            file_path: Path to Python file
            analysis_file: Path to analysis file
            suggestions_file: Path to suggestions file
            verification_file: Path to verification file
        
        Returns:
            Generated report
        """
        try:
            # Load all data
            with open(analysis_file, "r") as f:
                analysis = json.load(f)
            
            with open(suggestions_file, "r") as f:
                suggestions = json.load(f)
            
            with open(verification_file, "r") as f:
                verification = json.load(f)
            
            # Generate report
            report = f"""# Refactoring Report: {os.path.basename(file_path)}

## Original Analysis
- Complexity: {analysis['metrics']['complexity']}
- Maximum line length: {analysis['metrics']['line_length']}
- Function count: {analysis['metrics']['function_count']}
- Class count: {analysis['metrics']['class_count']}

### Issues Found
{chr(10).join(f'- {issue["message"]}' for issue in analysis['issues'])}

## Applied Improvements
"""
            
            for suggestion in suggestions:
                report += f"""
### {suggestion['description']}
{chr(10).join(f'- {s}' for s in suggestion['suggestions'])}
"""
            
            report += """
## Verification Results
"""
            
            for metric, data in verification["improvements"].items():
                report += f"""
### {metric.replace("_", " ").title()}
- Original: {data['original']}
- Current: {data['current']}
- Improved: {'Yes' if data['improved'] else 'No'}
"""
            
            report += f"""
### Issue Resolution
- Resolved issues: {verification['resolved_issues']}
- Remaining issues: {verification['remaining_issues']}
- Success: {'Yes' if verification['success'] else 'No'}
"""
            
            # Save report
            report_dir = os.path.join(self.work_dir, "reports")
            os.makedirs(report_dir, exist_ok=True)
            
            report_file = os.path.join(
                report_dir,
                f"{os.path.basename(file_path)}.report.md"
            )
            
            with open(report_file, "w") as f:
                f.write(report)
            
            return {
                "status": "success",
                "message": f"Refactoring report generated for {file_path}",
                "report_file": report_file,
                "report": report
            }
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
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
    
    def get_code_metrics(self) -> Dict[str, float]:
        """Get code quality metrics thresholds.
        
        Returns:
            Code metrics thresholds
        """
        return self.code_metrics
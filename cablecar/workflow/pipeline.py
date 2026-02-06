"""Analysis pipeline for chaining research steps."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
from cablecar.workflow.state import WorkflowState

@dataclass
class PipelineStep:
    name: str
    description: str
    function: Callable | None = None
    result: Any = None
    status: str = "pending"  # pending, running, completed, failed, skipped
    error: str = ""

class AnalysisPipeline:
    """Chain analysis steps with state management."""

    def __init__(self, name: str = ""):
        self.name = name
        self._steps: list[PipelineStep] = []
        self._state = WorkflowState(study_name=name)
        self._results: dict[str, Any] = {}

    @property
    def state(self) -> WorkflowState:
        return self._state

    def add_step(self, name: str, description: str, function: Callable | None = None) -> AnalysisPipeline:
        self._steps.append(PipelineStep(name=name, description=description, function=function))
        return self

    def run_step(self, step_name: str, **kwargs) -> Any:
        """Run a specific step by name."""
        for step in self._steps:
            if step.name == step_name:
                step.status = "running"
                self._state = self._state.with_update(current_step=step_name)
                try:
                    if step.function:
                        result = step.function(**kwargs)
                    else:
                        result = kwargs
                    step.result = result
                    step.status = "completed"
                    self._results[step_name] = result
                    self._state = self._state.add_analysis(step_name)
                    return result
                except Exception as e:
                    step.status = "failed"
                    step.error = str(e)
                    raise
        raise KeyError(f"Step '{step_name}' not found")

    def run_all(self, **kwargs) -> dict[str, Any]:
        """Run all steps in order."""
        for step in self._steps:
            if step.status != "completed":
                self.run_step(step.name, **kwargs)
        return self._results

    def get_result(self, step_name: str) -> Any:
        return self._results.get(step_name)

    def summary(self) -> dict:
        return {
            "name": self.name,
            "steps": [
                {"name": s.name, "status": s.status, "description": s.description}
                for s in self._steps
            ],
            "state": self._state.summary(),
        }

"""Jinja2-based scaffold generation for reproducible analysis scripts."""
from __future__ import annotations
from pathlib import Path
import jinja2
from cablecar.codegen.provenance import AnalysisProvenance


class CodeGenerator:
    """Generate site-portable scaffold scripts from Jinja2 templates.

    The scaffold handles deterministic boilerplate:
    - Imports (conditional on analysis types)
    - Data loading (CLIF table names, CSV/Parquet)
    - Cohort definition (inclusion/exclusion filters)
    - Stub functions for each analysis step
    - Output formatting and main block

    Analysis-specific code is left as TODO stubs for Claude to fill in,
    using the rich context from AnalysisProvenance.to_llm_context().
    """

    def __init__(self, template_dir: str | Path | None = None):
        if template_dir is None:
            template_dir = Path(__file__).parent / "templates"
        self._template_dir = Path(template_dir)
        self._env = jinja2.Environment(
            loader=jinja2.FileSystemLoader([
                str(self._template_dir / "python"),
                str(self._template_dir / "r"),
                str(self._template_dir),
            ]),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
        )

    def generate_scaffold(
        self,
        language: str,
        provenance: AnalysisProvenance,
    ) -> str:
        """Generate a site-portable scaffold script.

        Args:
            language: "python" or "r"
            provenance: Analysis provenance with study context

        Returns:
            Complete scaffold script with TODO stubs for analysis code.
        """
        if language not in ("python", "r"):
            raise ValueError(f"Unsupported language: {language}. Use 'python' or 'r'.")

        template_name = f"scaffold.{'py' if language == 'python' else 'R'}.j2"
        template = self._env.get_template(template_name)
        context = provenance.to_scaffold_context()
        return template.render(**context)

    def generate(
        self,
        language: str,
        analysis_type: str,
        provenance: AnalysisProvenance,
        **kwargs,
    ) -> str:
        """Backward-compatible entry point. Delegates to generate_scaffold().

        Kept for existing test compatibility. New code should use
        generate_scaffold() directly.
        """
        has_step = any(s.analysis_type == analysis_type for s in provenance.steps)
        if not has_step:
            provenance.add_step(
                name=analysis_type,
                description=f"{analysis_type} analysis",
                parameters=kwargs,
                analysis_type=analysis_type,
            )
        return self.generate_scaffold(language, provenance)

    def list_templates(self) -> list[str]:
        """List available templates."""
        templates = []
        for lang_dir in ["python", "r"]:
            lang_path = self._template_dir / lang_dir
            if lang_path.exists():
                templates.extend(
                    str(f.relative_to(self._template_dir))
                    for f in lang_path.glob("*.j2")
                )
        return templates

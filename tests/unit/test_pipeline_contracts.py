"""Contract tests binding params.yaml, dvc.yaml, the drivers and the reports.

These catch the class of failure that unit tests miss and that only shows up when
a grader runs `make all` on a clean clone: a stage that declares a params section
which no longer exists, a report that ``\\input``s a table no stage emits, or a
Makefile target naming a stage that was renamed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def dvc_config() -> dict:
    return yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text())


@pytest.fixture(scope="module")
def params() -> dict:
    return yaml.safe_load((REPO_ROOT / "params.yaml").read_text())


def test_every_declared_params_section_exists(dvc_config, params):
    for stage, spec in dvc_config["stages"].items():
        for section in spec.get("params", []):
            assert section in params, f"stage {stage!r} declares missing params section {section!r}"


def test_dag_covers_all_three_phases(dvc_config):
    assert {"ingest", "clean_features", "scale_reduce_tendency", "phase2", "phase3", "report"} <= set(
        dvc_config["stages"]
    )


def _out_paths(spec: dict) -> list[str]:
    paths = []
    for out in spec.get("outs", []):
        paths.append(next(iter(out)) if isinstance(out, dict) else out)
    return paths


def test_no_two_stages_own_the_same_output(dvc_config):
    """DVC rejects a path claimed by two stages; catch it before `dvc repro` does."""
    seen: dict[str, str] = {}
    for stage, spec in dvc_config["stages"].items():
        for path in _out_paths(spec):
            assert path not in seen, f"{path!r} is an output of both {seen[path]!r} and {stage!r}"
            seen[path] = stage


def test_git_tracked_deliverables_are_declared_uncached(dvc_config):
    """A cached out under reports/ makes `dvc repro` refuse to run the stage."""
    for stage, spec in dvc_config["stages"].items():
        for out in spec.get("outs", []):
            path = next(iter(out)) if isinstance(out, dict) else out
            if path.startswith("reports/") or path.startswith("results/") or path.startswith("models/"):
                assert isinstance(out, dict), f"{stage}: {path} must be declared with cache: false"
                assert out[path]["cache"] is False, f"{stage}: {path} must set cache: false"


# DVC stage names keep the course's phase vocabulary (the reports and the brief
# use it); the driver modules are named after what they actually do.
STAGE_DRIVERS = {"phase2": "run_portfolio", "phase3": "run_deep_clustering"}


def test_phase_stages_depend_on_their_driver_modules(dvc_config):
    for stage, module in STAGE_DRIVERS.items():
        deps = dvc_config["stages"][stage]["deps"]
        assert f"src/clustering_analysis/{module}.py" in deps


def test_stage_commands_invoke_their_driver_module(dvc_config):
    for stage, module in STAGE_DRIVERS.items():
        assert dvc_config["stages"][stage]["cmd"] == f"python -m clustering_analysis.{module}"


def test_phase3_depends_on_phase2_output(dvc_config):
    deps = dvc_config["stages"]["phase3"]["deps"]
    assert "results/phase2/summary.json" in deps
    assert "data/processed/phase2_clusters.npy" in deps


# --- reports vs pipeline outputs -------------------------------------------- #
def _inputs_of(report: Path) -> set[str]:
    return set(re.findall(r"\\input\{(tables/[^}]+)\}", report.read_text()))


def _includegraphics_of(report: Path) -> set[str]:
    return set(re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", report.read_text()))


@pytest.mark.parametrize("report_name", ["phase2_report.tex", "phase3_report.tex"])
def test_report_only_inputs_tables_a_driver_emits(report_name):
    """Every \\input'd table must be written unconditionally by its driver."""
    report = REPO_ROOT / "reports" / report_name
    phase = report_name.split("_")[0]
    driver = (REPO_ROOT / "src" / "clustering_analysis" / f"{STAGE_DRIVERS[phase]}.py").read_text()
    for table in _inputs_of(report):
        stem = Path(table).name
        assert stem in driver, f"{report_name} inputs {stem}, which {phase}.py never writes"


@pytest.mark.parametrize("report_name", ["phase2_report.tex", "phase3_report.tex"])
def test_report_figures_are_produced_by_their_driver(report_name):
    report = REPO_ROOT / "reports" / report_name
    phase = report_name.split("_")[0]
    driver = (REPO_ROOT / "src" / "clustering_analysis" / f"{STAGE_DRIVERS[phase]}.py").read_text()
    for figure in _includegraphics_of(report):
        assert figure in driver, f"{report_name} includes {figure}, which {phase}.py never writes"


def test_makefile_targets_reference_real_dvc_stages(dvc_config):
    makefile = (REPO_ROOT / "Makefile").read_text()
    for stage in re.findall(r"dvc repro (\w+)", makefile):
        assert stage in dvc_config["stages"], f"Makefile drives unknown stage {stage!r}"


def test_reports_makefile_builds_every_committed_report():
    makefile = (REPO_ROOT / "reports" / "Makefile").read_text()
    for report in ("analysis_plan", "phase1_report", "phase2_report", "phase3_report"):
        assert f"{report}.pdf" in makefile


# --- generated LaTeX must actually compile ---------------------------------- #
@pytest.mark.parametrize("phase", ["phase2", "phase3"])
def test_generated_tables_have_no_unescaped_percent(phase):
    """A bare %% comments out the rest of the line and orphans \\caption{."""
    for table in (REPO_ROOT / "reports" / "tables").glob(f"{phase}_*.tex"):
        for lineno, line in enumerate(table.read_text().splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("%"):
                continue  # a deliberate whole-line comment
            for i, char in enumerate(line):
                if char == "%" and (i == 0 or line[i - 1] != "\\"):
                    raise AssertionError(f"{table.name}:{lineno} has an unescaped %: {line!r}")


@pytest.mark.parametrize("phase", ["phase2", "phase3"])
def test_generated_tables_have_balanced_braces_and_environments(phase):
    for table in (REPO_ROOT / "reports" / "tables").glob(f"{phase}_*.tex"):
        text = table.read_text()
        assert text.count("{") == text.count("}"), f"{table.name}: unbalanced braces"
        assert text.count("\\begin{tabular}") == text.count("\\end{tabular}"), table.name
        assert text.count("\\begin{table}") == text.count("\\end{table}"), table.name


@pytest.mark.parametrize("phase", ["phase2", "phase3"])
def test_generated_tables_avoid_packages_the_reports_do_not_load(phase):
    """siunitx is not installed on BasicTeX; \\num{} would abort the build."""
    for table in (REPO_ROOT / "reports" / "tables").glob(f"{phase}_*.tex"):
        text = table.read_text()
        assert "\\num{" not in text, f"{table.name} uses \\num{{}} but siunitx is not loaded"
        for char in ("σ", "×", "±"):
            assert char not in text, f"{table.name} contains non-ASCII {char!r}"


# --- notebook / script pairing ---------------------------------------------- #
# The notebook is what the pipeline executes; the script is what a reviewer
# reads and re-runs. They drifted apart twice during development (an edit to one
# never reached the other), so this is enforced rather than trusted to a hook.
NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))


def _notebook_code(path: Path) -> list[str]:
    import json

    cells = json.loads(path.read_text())["cells"]
    return [
        "".join(c["source"]).strip()
        for c in cells
        if c["cell_type"] == "code" and "".join(c["source"]).strip()
    ]


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_every_notebook_has_a_paired_script(notebook):
    assert notebook.with_suffix(".py").exists(), (
        f"{notebook.name} has no paired .py script; reviewers cannot re-run it cleanly"
    )


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_and_script_contain_the_same_code(notebook):
    """Every code cell in the notebook must appear verbatim in the script."""
    script_text = notebook.with_suffix(".py").read_text()
    missing = [cell for cell in _notebook_code(notebook) if cell not in script_text]
    assert not missing, (
        f"{notebook.name} has {len(missing)} code cell(s) absent from its script, "
        f"starting with: {missing[0][:120]!r}"
    )


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_committed_in_executed_form(notebook):
    """The brief requires notebooks committed with their outputs visible."""
    import json

    cells = json.loads(notebook.read_text())["cells"]
    code = [c for c in cells if c["cell_type"] == "code" and "".join(c["source"]).strip()]
    executed = [c for c in code if c.get("execution_count") is not None]
    assert len(executed) == len(code), (
        f"{notebook.name}: {len(code) - len(executed)} code cell(s) were never executed"
    )


def test_jupytext_pairing_is_configured():
    """Without a formats setting, `jupytext --sync` silently does nothing."""
    import tomllib

    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert "ipynb" in config["tool"]["jupytext"]["formats"]

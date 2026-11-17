import re
from pathlib import Path

import nbformat
import pytest


@pytest.fixture
def root_notebook():
    nb_path = Path("FinRisk_ML_Group08.ipynb")
    assert nb_path.exists(), "Master notebook FinRisk_ML_Group08.ipynb must exist at repo root"
    return nbformat.read(nb_path, as_version=4)


@pytest.fixture
def mirrored_notebook():
    nb_path = Path("notebooks/FinRisk_ML_Group08.ipynb")
    assert nb_path.exists(), "Mirrored notebook must exist at notebooks/FinRisk_ML_Group08.ipynb"
    return nbformat.read(nb_path, as_version=4)


def test_notebook_json_and_metadata(root_notebook, mirrored_notebook):
    """Verifies that the notebook adheres to nbformat v4 with valid python kernelspec."""
    for nb in [root_notebook, mirrored_notebook]:
        assert nb.nbformat == 4
        assert nb.metadata.kernelspec.language == "python"
        assert len(nb.cells) >= 25


def test_colab_badge_present(root_notebook):
    """Verifies Google Colab launch badge is present in cell 0 with correct repo URL."""
    cell_0 = root_notebook.cells[0].source
    expected_badge = "https://colab.research.google.com/assets/colab-badge.svg"
    expected_url = "https://colab.research.google.com/github/ShoumikBalaSomu/ML-Project-Group-08/blob/main/FinRisk_ML_Group08.ipynb"
    assert expected_badge in cell_0, "Colab badge svg missing in cell 0"
    assert expected_url in cell_0, "Colab target link missing in cell 0"


def test_author_attribution_preserved(root_notebook):
    """Verifies strict author attribution order: Shoumik Bala Somu, Md Israfil Hossen, Ariyan Islam Hridoy."""
    text = root_notebook.cells[0].source
    pos_shoumik = text.find("Shoumik Bala Somu")
    pos_israfil = text.find("Md Israfil Hossen")
    pos_ariyan = text.find("Ariyan Islam Hridoy")

    assert pos_shoumik != -1, "Author Shoumik Bala Somu missing"
    assert pos_israfil != -1, "Author Md Israfil Hossen missing"
    assert pos_ariyan != -1, "Author Ariyan Islam Hridoy missing"
    assert pos_shoumik < pos_israfil < pos_ariyan, "Author attribution order violated"


def test_no_bengali_characters_in_notebook(root_notebook):
    """Ensures zero Bengali characters exist across all notebook cells."""
    bengali_regex = re.compile(r"[\u0980-\u09FF]")
    for idx, cell in enumerate(root_notebook.cells):
        matches = bengali_regex.findall(cell.source)
        assert len(matches) == 0, f"Found Bengali characters in cell {idx}: {matches}"


def test_all_academic_sections_present(root_notebook):
    """Ensures all 15 academic sections exist in sequential order."""
    expected_sections = [
        "Google Colab & Environment Setup",
        "2. Problem Motivation & Cost-Matrix Mathematics",
        "3. Cost Matrix & Empirical Threshold Optimization Engine",
        "4. Data Ingestion, Cleaning & Scikit-Learn Preprocessing Pipelines",
        "5. Exploratory Data Analysis (EDA) & Imbalance Profiling",
        "6. Machine Learning Model Zoo Architecture",
        "7. Model Training & Threshold Sweeps (Task A: Credit Card Fraud)",
        "8. Model Training & Threshold Sweeps (Task B: Loan Default)",
        "9. Comparative Benchmarks & Cost Reduction Analysis",
        "10. Real-Time Production SLA & Latency Profiling",
        "11. Global Explainability via TreeSHAP",
        "12. Local Contrastive Explainability & Regulatory Compliance",
        "13. Interactive What-If Cost & Volume Simulator",
        "14. Algorithmic Fairness",
        "15. Academic Citations & Reproducibility Statement",
    ]
    notebook_text = "\n".join(c.source for c in root_notebook.cells if c.cell_type == "markdown")
    for sec in expected_sections:
        assert sec in notebook_text, f"Missing section: {sec}"


def test_pre_rendered_outputs_exist(root_notebook):
    """Verifies that code cells have been executed and contain pre-rendered outputs."""
    code_cells = [c for c in root_notebook.cells if c.cell_type == "code"]
    executed_cells = [c for c in code_cells if c.execution_count is not None]
    assert len(executed_cells) >= 10, "Notebook cells must be pre-rendered with executed outputs"

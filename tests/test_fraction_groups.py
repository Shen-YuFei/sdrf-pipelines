"""Regression coverage for experimental-design fraction group numbering."""

from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from sdrf_pipelines.converters.diann.diann import DiaNN
from sdrf_pipelines.converters.openms.experimental_design import FractionGroupTracker
from sdrf_pipelines.converters.openms.openms import OpenMS


@pytest.mark.parametrize(
    "logical_groups",
    [
        [1, 9, 10, 2, 3, 11, 4, 12, 5, 13, 14, 6, 7, 15, 8, 16],
        [4, 5, 6, 10, 11, 12],
        [6, 5, 4, 12, 11, 10],
    ],
)
def test_distinct_fraction_groups_remain_distinct(logical_groups: list[int]) -> None:
    tracker = FractionGroupTracker()
    files = [f"run_{group}.raw" for group in logical_groups]
    assigned = [tracker.get_fraction_group(raw, group) for raw, group in zip(files, logical_groups)]

    assert sorted(assigned) == list(range(1, len(logical_groups) + 1))
    # Returning to an earlier group must not change its assignment.
    assert [tracker.get_fraction_group(raw, group) for raw, group in zip(files, logical_groups)] == assigned


def test_fractionated_multiplex_files_share_groups() -> None:
    tracker = FractionGroupTracker()
    assignments = [
        ("run1_fraction1.raw", 4),
        ("run1_fraction1.raw", 10),
        ("run2_fraction1.raw", 5),
        ("run2_fraction1.raw", 11),
        ("run1_fraction2.raw", 4),
        ("run1_fraction2.raw", 10),
        ("run2_fraction2.raw", 5),
        ("run2_fraction2.raw", 11),
    ]

    assert [tracker.get_fraction_group(*assignment) for assignment in assignments] == [1, 1, 2, 2, 1, 1, 2, 2]


@pytest.mark.parametrize("converter", ["openms", "openms_one_table", "diann"])
@pytest.mark.parametrize("replicates", [(1, 2, 3), (4, 5, 6)])
@pytest.mark.parametrize("order", ["interleaved", "reversed", "shuffled"])
def test_converters_preserve_fraction_group_membership(
    converter: str, replicates: tuple[int, ...], order: str, on_tmpdir: Path
) -> None:
    template_path = Path(__file__).parent / "data" / "diann" / "gpf_fractions.sdrf.tsv"
    template = pd.read_csv(template_path, sep="\t", dtype=str).iloc[0].to_dict()
    rows = []
    for replicate in replicates:
        for fraction in (1, 2):
            for sample in ("control", "treated"):
                row = dict(template)
                row.update(
                    {
                        "source name": sample,
                        "assay name": f"{sample}_{replicate}_{fraction}",
                        "comment[data file]": f"{sample}_{replicate}_{fraction}.raw",
                        "comment[technical replicate]": str(replicate),
                        "comment[fraction identifier]": str(fraction),
                        "factor value[treatment]": sample,
                    }
                )
                rows.append(row)
    sdrf = pd.DataFrame(rows)
    if order == "reversed":
        sdrf = sdrf.iloc[::-1]
    elif order == "shuffled":
        sdrf = sdrf.sample(frac=1, random_state=42)
    input_path = on_tmpdir / "interleaved.sdrf.tsv"
    sdrf.to_csv(input_path, sep="\t", index=False)

    if converter == "diann":
        DiaNN().diann_convert(str(input_path))
        design = pd.read_csv(on_tmpdir / "diann_design.tsv", sep="\t")
        group_column, file_column = "FractionGroup", "Filename"
    else:
        OpenMS().openms_convert(str(input_path), one_table=converter == "openms_one_table")
        file_table = (on_tmpdir / "experimental_design.tsv").read_text().split("\n\n")[0]
        design = pd.read_csv(StringIO(file_table), sep="\t")
        group_column, file_column = "Fraction_Group", "Spectra_Filepath"

    assert set(design[file_column]) == set(sdrf["comment[data file]"])
    assert len(design) == len(sdrf)
    assert not design.duplicated([group_column, "Fraction", "Label"]).any()
    groups = design[group_column].unique()
    assert sorted(groups) == list(range(1, 2 * len(replicates) + 1))

    merged = design.merge(sdrf, left_on=file_column, right_on="comment[data file]", validate="one_to_one")
    assert (merged["Fraction"] == merged["comment[fraction identifier]"].astype(int)).all()
    # Each sample/technical replicate keeps both fractions in one group;
    # each output group contains exactly that sample/technical replicate.
    assert (merged.groupby(["source name", "comment[technical replicate]"])[group_column].nunique() == 1).all()
    assert (merged.groupby(group_column)[["source name", "comment[technical replicate]"]].nunique() == 1).all().all()

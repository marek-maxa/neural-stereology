from pathlib import Path
from types import SimpleNamespace

import main as application
import pytest
from core import runner, tools


def test_main_connects_simulation_output_to_save_and_assessment(monkeypatch):
    events = []
    logger = SimpleNamespace(info=lambda *args: None)
    cells = [object()]
    segments = [[(1.0, "gap")]]

    monkeypatch.setattr(application, "load_config", lambda path: {"log_level": "INFO"})
    monkeypatch.setattr(application, "setup_logging", lambda **kwargs: logger)

    def run_simulation(config, strain, active_logger):
        events.append(("run", strain, active_logger))
        return cells, segments

    monkeypatch.setattr(application, "run_simulation", run_simulation)
    monkeypatch.setattr(
        application,
        "save_results",
        lambda result_cells, active_logger: events.append(
            ("save", result_cells, active_logger)
        ),
    )
    monkeypatch.setattr(
        application,
        "assess_results",
        lambda result_cells, result_segments, active_logger: events.append(
            ("assess", result_cells, result_segments, active_logger)
        ),
    )

    result = application.main(
        "config.json",
        twinning_strain="strain-model",
        assess_results_flag=True,
        get_results=True,
    )

    assert result == (cells, segments)
    assert [event[0] for event in events] == ["run", "save", "assess"]


def test_neper_preparation_runs_all_pipeline_stages(monkeypatch):
    events = []
    cells = [object()]
    selected_cells = [object()]
    selected_segments = [[(1.0, "gap")]]

    monkeypatch.setattr(tools, "run_feret_code", lambda: events.append("feret"))
    monkeypatch.setattr(
        tools,
        "read_feret_data",
        lambda path: events.append("read") or ([0.0], [1.0]),
    )
    monkeypatch.setattr(
        tools,
        "process_feret_data",
        lambda left, right, source_cells: (
            events.append("process")
            or (selected_cells, [[0.0, 0.0, 0.0]], [1.0], left, right)
        ),
    )
    monkeypatch.setattr(
        tools,
        "write_auxiliary_files",
        lambda *args: events.append("auxiliary"),
    )
    monkeypatch.setattr(
        tools,
        "create_first_tessellation",
        lambda *args: events.append("first-tessellation"),
    )
    monkeypatch.setattr(
        tools,
        "recompute_lamellae",
        lambda *args: events.append("segments") or selected_segments,
    )
    monkeypatch.setattr(
        tools,
        "write_additional_files_for_neper",
        lambda *args: events.append("multiscale-inputs"),
    )
    monkeypatch.setattr(
        tools,
        "create_second_tessellation",
        lambda *args: events.append("second-tessellation"),
    )

    result = tools.prepare_for_neper(cells)

    assert result == (selected_segments, selected_cells)
    assert events == [
        "feret",
        "read",
        "process",
        "auxiliary",
        "first-tessellation",
        "segments",
        "multiscale-inputs",
        "second-tessellation",
    ]


def test_orientation_reader_accepts_generic_whitespace(tmp_path):
    orientation_file = tmp_path / "orientation"
    orientation_file.write_text("1\t2  3\n\n4 5\t6\n", encoding="utf-8")

    assert runner.import_orientation(orientation_file) == [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]


def test_volume_function_reader_keeps_the_expected_partition_shape(tmp_path):
    volume_file = tmp_path / "volume-function"
    values = list(range(runner.FERET_PARTITION_NUMBER + 1))
    volume_file.write_text("  ".join(map(str, values)) + "\n", encoding="utf-8")

    matrix = runner.read_volume_function(volume_file, num_of_cells=1)

    assert matrix.shape == (1, runner.FERET_PARTITION_NUMBER)
    assert matrix[0, -1] == runner.FERET_PARTITION_NUMBER - 1


def test_segment_partition_preserves_the_clipped_interval():
    lamellae = [
        SimpleNamespace(center=0.3, width=0.1),
        SimpleNamespace(center=0.7, width=0.1),
    ]

    segments = tools.segment_lengths_in_subinterval(lamellae, 0.0, 1.0, 0.1, 0.9)

    assert all(width > 0 for width, _ in segments)
    assert sum(width for width, _ in segments) == pytest.approx(0.8)
    assert {label for _, label in segments} <= {"gap", "lamella"}


def test_neper_width_and_orientation_rows_stay_aligned(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    cells = [
        SimpleNamespace(
            orientation=[0.0, 0.0, 0.0],
            lamella_orientation=[1.0, 1.0, 1.0],
            runtime_metadata={},
        ),
        SimpleNamespace(
            orientation=[2.0, 2.0, 2.0],
            lamella_orientation=[3.0, 3.0, 3.0],
            runtime_metadata={},
        ),
    ]
    segments = [
        [(0.4, "gap"), (0.2, "lamella")],
        [(0.6, "gap")],
    ]

    tools.write_additional_files_for_neper(cells, segments)
    specifications = tools._load_multiscale_cell_specs(tmp_path / "data")

    assert [len(specification["widths"]) for specification in specifications] == [2, 1]
    assert [len(cell.runtime_metadata["neper_segments"]) for cell in cells] == [2, 1]


def test_neper_fallback_records_the_effective_structure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data").mkdir()
    cell = SimpleNamespace(
        orientation=[0.0, 0.0, 0.0],
        lamella_orientation=[1.0, 1.0, 1.0],
        runtime_metadata={},
    )
    tools.write_additional_files_for_neper(
        [cell],
        [[(0.4, "gap"), (0.6, "lamella")]],
    )
    attempts = []

    def run_neper(command):
        attempts.append(command)
        if len(attempts) == 1:
            raise tools.subprocess.CalledProcessError(
                -11,
                command,
                stderr="Flattening 2-scale tessellation",
            )

    monkeypatch.setattr(tools, "_run_neper_second_tessellation", run_neper)

    tools.create_second_tessellation([cell])

    assert len(attempts) == 2
    assert Path.cwd() == tmp_path
    assert cell.runtime_metadata["neper_simplification"]["lossy"] is True
    assert cell.runtime_metadata["neper_segments"] == [
        {"width": 0.6, "label": "lamella"}
    ]


def test_precision_assessment_compares_one_value_per_cell(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "2scale.stcell").write_text("2\n1\n3\n", encoding="utf-8")
    calls = []
    histogram_values = []

    monkeypatch.setattr(
        tools.subprocess,
        "run",
        lambda *args, **kwargs: calls.append(kwargs),
    )

    class Axes:
        def hist(self, values, **kwargs):
            histogram_values.extend(values)

        def set_xlabel(self, label):
            pass

        def set_ylabel(self, label):
            pass

        def set_title(self, title):
            pass

    class Figure:
        def savefig(self, path):
            calls.append({"saved": Path(path)})

    monkeypatch.setattr(tools.plt, "subplots", lambda: (Figure(), Axes()))
    monkeypatch.setattr(tools.plt, "close", lambda figure: None)

    cells = [
        SimpleNamespace(cid=1, volume_fraction=1 / 3, runtime_metadata={}),
        SimpleNamespace(cid=2, volume_fraction=0.0, runtime_metadata={}),
    ]
    segments = [
        [(1.0, "gap"), (1.0, "lamella")],
        [(1.0, "gap")],
    ]

    tools.check_precision(cells, segments)

    assert calls[0]["cwd"] == data_dir
    assert histogram_values == pytest.approx([0.0, 0.0])
    assert len(histogram_values) == len(cells)


def test_unknown_configuration_keys_are_rejected():
    with pytest.raises(RuntimeError, match="Unsupported config keys"):
        application.validate_simulation_config({"misspelled-option": True})

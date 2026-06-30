"""Sample test — verifies pytest collection from tests/."""

from spatialx_transform import main


def test_main_prints(capsys):
    """main() prints a greeting."""
    main()
    captured = capsys.readouterr()
    assert "spatialx-transform" in captured.out

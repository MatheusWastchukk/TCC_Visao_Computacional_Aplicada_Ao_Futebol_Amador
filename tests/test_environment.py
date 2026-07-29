from scripts.check_environment import REQUIRED_MODULES, check_module


def test_required_environment_module_list_contains_core_dependencies() -> None:
    assert REQUIRED_MODULES["cv2"] == "opencv-python"
    assert REQUIRED_MODULES["torch"] == "torch"
    assert REQUIRED_MODULES["ultralytics"] == "ultralytics"


def test_standard_library_import_check_passes() -> None:
    result = check_module("json", "json")

    assert result.available is True
    assert result.error is None

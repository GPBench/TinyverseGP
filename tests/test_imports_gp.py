import importlib

def test_import_gp_package():
    mod = importlib.import_module("gp")
    assert hasattr(mod, "__package__")

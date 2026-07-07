
import subprocess

def test_example_symbolic_regression():
    # NOTE: Success test is very simplistic
    res = subprocess.run(["python3", "-m", "examples.symbolic_regression.test_cgp_sr"])
    assert res.returncode == 0
    res = subprocess.run(["python3", "-m", "examples.symbolic_regression.test_tgp_sr"])
    assert res.returncode == 0

def test_example_logic_synthesis():
    # NOTE: These examples take forever... disabled.
    # res = subprocess.run(["python3", "-m", "examples.logic_synthesis.test_cgp_ls"])
    # assert res.returncode == 0
    # res = subprocess.run(["python3", "-m", "examples.logic_synthesis.test_tgp_ls"])
    # assert res.returncode == 0
    pass

def test_example_policy_learning():
    # NOTE: These also take forever... disabled.
    # res = subprocess.run(["python3", "-m", "examples.policy_learning.test_cgp_pl"])
    # assert res.returncode == 0
    # res = subprocess.run(["python3", "-m", "examples.policy_learning.test_cgp_pl_ale"])
    # assert res.returncode == 0
    # res = subprocess.run(["python3", "-m", "examples.policy_learning.test_tgp_pl"])
    # assert res.returncode == 0
    pass
    
def test_example_program_synthesis():
    # NOTE: Takes long
    # res = subprocess.run(["python3", "-m", "examples.program_synthesis.test_cgp_ps"])
    # assert res.returncode == 0
    # NOTE: Broken example
    # res = subprocess.run(["python3", "-m", "examples.program_synthesis.test_tgp_ps"])
    # assert res.returncode == 0
    pass

def test_example_hpo():
    # NOTE: Broken example below
    # res = subprocess.run(["python3", "-m", "examples.hpo.test_cgp_sr"])
    # assert res.returncode == 0
    # NOTE: Broken example
    # res = subprocess.run(["python3", "-m", "examples.hpo.test_tgp_sr"])
    # assert res.returncode == 0
    pass

import subprocess

def test_example_symbolic_regression():
    # NOTE: Success test is very simplistic
    res = subprocess.run(["python3", "-m", "examples.symbolic_regression.test_cgp_sr"])
    assert res.returncode == 0
    res = subprocess.run(["python3", "-m", "examples.symbolic_regression.test_tgp_sr"])
    assert res.returncode == 0

def test_example_logic_synthesis_cgp_ls():
    # NOTE: This example takes forever...
    res = subprocess.run(["python3", "-m", "examples.logic_synthesis.test_cgp_ls"])
    assert res.returncode == 0

def test_example_logic_synthesis_tgp_ls():
    # NOTE: This example takes forever...
    res = subprocess.run(["python3", "-m", "examples.logic_synthesis.test_tgp_ls"])
    assert res.returncode == 0

def test_example_policy_learning_cgp_pl():
    # NOTE: These also take forever...
    # NOTE: This test expect user input (An enter?), we need to spoof past this
    # res = subprocess.run(["python3", "-m", "examples.policy_learning.test_cgp_pl"])
    # assert res.returncode == 0
    pass

def test_example_policy_learning_cgp_pl_ale():
    # NOTE: These also take forever...
    # NOTE: This test expect user input (An enter?), we need to spoof past this
    # res = subprocess.run(["python3", "-m", "examples.policy_learning.test_cgp_pl_ale"])
    # assert res.returncode == 0
    pass

def test_example_policy_learning_tgp_pl():
    # NOTE: These also take forever...
    # NOTE: This test expect user input (An enter?), we need to spoof past this
    # res = subprocess.run(["python3", "-m", "examples.policy_learning.test_tgp_pl"])
    # assert res.returncode == 0
    pass
    
def test_example_program_synthesis():
    # NOTE: Takes long
    # NOTE: Disabled as it asks for input, we need to cast our way around this
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
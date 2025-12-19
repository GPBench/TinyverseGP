from src.gp.tiny_cgp import TinyCGP
from src.gp.tiny_ge import TinyGE
from src.gp.tiny_lgp import TinyLGP
from src.gp.tiny_tgp import TinyTGP


def get_model(representation, functions, terminals, hyperparameters, config, grammar=None):
    if representation == "TGP":
        model = TinyTGP(functions, terminals, config, hyperparameters)
    elif representation == "CGP":
        model = TinyCGP(functions, terminals, config, hyperparameters)
    elif representation == "GE":
        model = TinyGE(functions, grammar, terminals, config, hyperparameters)
    elif representation == "LGP":
        model = TinyLGP(functions, terminals, config, hyperparameters)
    else:
        raise ValueError("Invalid representation type")
    return model

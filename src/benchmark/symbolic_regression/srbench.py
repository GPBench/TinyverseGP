"""
This file contains the SRBench class which is used to define the configuration of the symbolic regression benchmarking problem.
"""

from src.gp.functions import ADD, SUB, MUL, DIV, EXP, LOG, SQR, CUBE
from src.gp.tinyverse import Const, Var
from src.gp.tiny_cgp import TinyCGP
#from src.gp.tiny_3ge import Tiny3GE, TreeGEHyperparameters, TreeGEConfig
from src.gp.tiny_ge import TinyGE
from src.gp.tiny_lgp import TinyLGP
from src.gp.tiny_tgp import TinyTGP, Node
import re
from src.gp.loss import mean_squared_error, linear_scaling_mse, linear_scaling_coeff
from src.gp.problem import BlackBox

from sklearn.base import RegressorMixin
import sympy as sp
import numpy as np

strfun = {
    "+": ADD,
    "-": SUB,
    "*": MUL,
    "/": DIV,
    "exp": EXP,
    "log": LOG,
    "square": SQR,
    "cube": CUBE,
}


class SRBench(RegressorMixin):
    def __init__(
        self,
        representation,
        config,
        hyperparameters,
        functions,
        terminals=[1, 0.5, np.pi, np.sqrt(2)],
        scaling_=False,
        grammar={"<expr>": ["ADD(<expr>, <expr>)", 
                          "SUB(<expr>, <expr>)", 
                          "MUL(<expr>, <expr>)", 
                          "DIV(<expr>, <expr>)", 
                          "EXP(<expr>)", 
                          "LOG(<expr>)", 
                          "SQR(<expr>)", 
                          "CUBE(<expr>)", 
                          "<const>", 
                          "<var>",
                                ],
                "<const>": ["1", "0.5", "3.14159", "1.41421"], 
                "<var>": []
        },
        loss=mean_squared_error,
        optimized=False
    ):
        self.representation = representation
        self.scaling = scaling_
        self.functions = functions
        self.grammar = grammar or self._make_default_grammar(self.functions, terminals)
        self.loss = linear_scaling_mse if self.scaling else mean_squared_error
        # self.functions = [strfun[f] for f in functions]
        self.locals = {f.name: f.custom for f in self.functions if f.custom is not None}
        self.terminals = [Const(c) for c in terminals]
        self.fitted_ = False
        self.config = config
        self.optimized = optimized 
        self.grammar = grammar
        self.hyperparameters = hyperparameters

    def _make_default_grammar(self, functions, terminals):
        # Ensure grammar uses uppercase function names matching Function objects
        return {
            "<expr>": [f"{f.name.upper()}(<expr>, <expr>)" for f in functions if f.arity == 2]
                    + [f"{f.name.upper()}(<expr>)" for f in functions if f.arity == 1]
                    + ["<const>", "<var>"],
            "<const>": [str(c) for c in [1, 0.5, "3.14159", "1.41421"]],
            "<var>": [],
        }

    def fit(self, X, y, checkpoint=None, **log_kwargs):
        problem = BlackBox(X, y, self.loss, 1e-16, True, **log_kwargs)
        self.terminals = [Var(i) for i in range(X.shape[1])] + self.terminals
        # Always set <var> in grammar to correct variable names before fitting
        self.grammar["<var>"] = [f"x{i}" for i in range(X.shape[1])]
        if self.representation == "TGP":
            #self.functions = [strfun[f] for f in self.functions]
            self.model = TinyTGP(
                self.functions, self.terminals, self.config, self.hyperparameters
            )
        elif self.representation == "CGP":
            # self.functions = [strfun[f] for f in self.functions]
            self.model = TinyCGP(
                self.functions, self.terminals, self.config, self.hyperparameters
            )
        elif self.representation == "LGP":
            # self.functions = [strfun[f] for f in self.functions]
            self.model = TinyLGP(
                self.functions, self.terminals, self.config, self.hyperparameters
            )
        elif self.representation == "GE" or self.representation == "3GE":
            # NOTE: T.S. Disabled unused code
            # newfunctions = {f.name.upper(): f.function for f in self.functions}
            arguments = [f"x{i}" for i in range(X.shape[1])]
            self.grammar["<var>"] = arguments
            if self.representation == "3GE":
                #self.model = Tiny3GE(self.functions, self.grammar, arguments, self.config, self.hyperparameters)
                pass
            elif self.representation == "GE":
                self.model = TinyGE(self.functions, self.grammar, arguments, self.config, self.hyperparameters)
        else:
            raise ValueError("Invalid representation type")
        if checkpoint is not None:
            self.model.resume(checkpoint, problem)
        self.program = self.model.evolve(problem)
        if self.representation == "TGP" and self.scaling:
            yhat = np.array([self.model.predict(self.program.genome, x)[0] for x in X])
            a, b = linear_scaling_coeff(yhat, y)
            self.program.genome[0] = Node(
                ADD,
                [
                    Node(MUL, [Node(Const(a), []), self.program.genome[0]]),
                    Node(Const(b), []),
                ],
            )
        self.fitted_ = True

    def predict(self, X):
        if not self.fitted_:
            raise ValueError("Model not fitted")
        yhat = np.array([self.model.predict(self.program.genome, x)[0] for x in X])
        # replace NaNs and Infs to calculate score
        yhat[np.isnan(yhat)] = 1e50
        yhat[np.isinf(yhat)] = 1e50
        return yhat

    def get_model(self, X=None):
        if not self.fitted_:
            raise ValueError("Model not fitted")
        expr = self.model.expression(self.program.genome)[0]

        if self.representation == "3GE":
            expr = self.model.expression(self.program.genome)
        if X is None:
            expr = re.sub(r"Var\((\d+)\)", r"x\1", expr)
            expr = re.sub(
                r"Const\(([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\)", r"\1", expr
            )
            expr = sp.sympify(expr, locals=self.locals)
        else:
            # replace all occurrences of 'Var(i)' in expr with the values in X[i]
            self.locals.update({x: sp.Symbol(x) for x in X})
            expr = re.sub(r"Var\((\d+)\)", lambda m: str(X[int(m.group(1))]), expr)
            expr = re.sub(
                r"Const\(([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)\)", r"\1", expr
            )
            expr = sp.sympify(expr, locals=self.locals)
        return expr

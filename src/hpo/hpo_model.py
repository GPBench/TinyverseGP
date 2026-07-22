# This class defines the problem interface for hyperparameter optimization of GP models using SMAC.
# Note: This model is tailored to exclusively run on symbolic regression problems.

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pmlb import fetch_data
import numpy as np
import sys

from src.hpo.hpo import SMACInterface
from src.gp.tiny_cgp import TinyCGP
from src.gp.tiny_3ge import Tiny3GE
from src.gp.tiny_ge import TinyGE
from src.gp.tiny_tgp import TinyTGP
from src.gp.tinyverse import GPModel
from src.gp.functions import ADD, SUB, MUL, DIV, EXP, LOG, SQR, CUBE
from src.gp.problem import BlackBox
from src.gp.loss import absolute_distance

seed_ = int(sys.argv[1])

class HPOModel():
    def __init__(
        self, 
        representation, 
        config, 
        hyperparameters, 
        dataset, 
        functions=[ADD, SUB, MUL, DIV, EXP, LOG, SQR, CUBE], 
        terminals=[1, 0.5, np.pi, np.sqrt(2)], 
        grammar=None, 
        loss=absolute_distance
    ):
        self.representation = representation
        self.config = config
        self.hyperparameters = hyperparameters
        self.functions = functions
        self.terminals = terminals
        self.grammar = grammar
        self.dataset = dataset  
        self.loss = loss


    def _make_default_grammar(self, functions, terminals):
        # Ensure grammar uses uppercase function names matching Function objects
        return {
            "<expr>": [f"{f.name.upper()}(<expr>, <expr>)" for f in functions if f.arity == 2]
                    + [f"{f.name.upper()}(<expr>)" for f in functions if f.arity == 1]
                    + ["<const>", "<var>"],
            "<const>": [str(c) for c in [1, 0.5, "3.14159", "1.41421"]],
            "<var>": [],
        }
    

    def create_model(self, hyperparameters, num_vars):

        
        if self.representation == "TGP":
            model = TinyTGP(
                self.functions, self.terminals, self.config, hyperparameters
            )
        elif self.representation == "CGP":
            model = TinyCGP(
                self.functions, self.terminals, self.config, hyperparameters
            )
        elif self.representation == "GE" or self.representation == "3GE":
            newfunctions = {f.name.upper(): f.function for f in self.functions}
            arguments = [f"x{i}" for i in range(num_vars)]
            self.grammar["<var>"] = arguments
            print(f"arguments: {arguments}")
            print(self.grammar)
            if self.representation == "3GE":
                model = Tiny3GE(self.functions, self.grammar, arguments, self.config, hyperparameters)
                # print("Im here")
                # newmodel = deepcopy(model)
            elif self.representation == "GE":
                model = TinyGE(self.functions, self.grammar, arguments, self.config, hyperparameters)
        else:
            raise ValueError("Invalid representation type")
        
        return model


    def evaluate_score(self, problem, model: GPModel, X, y):
        """Compute R² score given a GP model and dataset."""
        program = model.evolve(problem)
        yhat = np.array([model.predict(program.genome, x)[0] for x in X])
        return r2_score(y, yhat)
    
    def dataset_split(self, dataset, train_size):
        X, y = fetch_data(dataset, return_X_y=True)
        train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=train_size, shuffle=False)
        return train_X, test_X, train_y, test_y

    def run_smac(self, train_size=0.75, n_trials=50):
        results = {}
        for dataset in self.dataset:
            print(f"Dataset: {dataset}")
            print(f"==== seed: {seed_} ====")

            train_X, test_X, train_y, test_y = self.dataset_split(dataset, train_size)
            problem = BlackBox(train_X, train_y, self.loss, 1e-16, True)

            GP_model = self.create_model(self.hyperparameters, train_X.shape[1])
            # GP_model.fit(train_X, train_y)
            interface = SMACInterface()    

            # run SMAC
            opt_hyperparameters = interface.optimise(GP_model, problem, n_trials, seed_)
            results[dataset] = opt_hyperparameters
            
            train_base = self.evaluate_score(problem, GP_model, train_X, train_y)
            test_base = self.evaluate_score(problem, GP_model, test_X, test_y)
            print("="*50)
            print(f"Optimized hyperparameters: {opt_hyperparameters}")

            # run the optimized model
            opt_GP_model = self.create_model(opt_hyperparameters, train_X.shape[1])
            print("running optimized model...")
            # opt_GP_model.fit(test_X, test_y)
            train_opt = self.evaluate_score(problem, opt_GP_model, train_X, train_y)
            test_opt = self.evaluate_score(problem, opt_GP_model, test_X, test_y)
            print("="*50)
            optimized_GP_model_expression = opt_GP_model.expression(opt_GP_model.best_individual.genome)
            
            print("="*80)

            results[dataset] = {
                "seed" : seed_,
                "baseline" : {
                    "train_score": train_base,
                    "test_score": test_base
                },
                "optimized": {
                    "Optimized hyperparameters": opt_hyperparameters,
                    "train_score": train_opt,
                    "test_score": test_opt,
                    "solution": optimized_GP_model_expression,
                }
            }

            print("="*80)

            # -------------------------
            # Print structured output
            # -------------------------
            print(f"seed: {seed_}")
            print("\nBaseline Model:")
            print("  train-score:", train_base)
            print("  test-score:", test_base)
            print("-"*60)
            print("\nOptimized Model:")
            print("  Optimized Hyperparameters:", opt_hyperparameters)
            print("  train-score:", train_opt)
            print("  test-score :", test_opt)
            print("  Expression:", optimized_GP_model_expression)
            print("="*60)

        return results
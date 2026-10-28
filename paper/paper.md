---
title: 'TinyverseGP'
# TODO: add author list, affiliations, date, and tags (see https://joss.readthedocs.io/en/latest/paper.html)
# authors:
#   - name: ...
#     affiliation: 1
# affiliations:
#   - index: 1
#     name: ...
# date: ...
# bibliography: paper.bib
---

## Summary
Genetic Programming (Cramer, 1985, Dickmanns et al., 1987, Koza, 1989) (GP) is a general search meta-heuristic that searches for computer programs capable of solving a certain problem. This technique has been successfully applied to many problem domains, from machine learning to policy learning and program synthesis. Its main advantage lies on its interpretability and transparency, having achieved above human-level performance in many tasks while allowing a higher understanding of the solution. <!-- Reviewer note: I don't understand how the two parts of the sentence articulate, I am not sure it is grammatically correct or if I am missing something -->
To represent candidate programs as individuals within populations, GP traditionally uses parse trees as representation model which has been inspired by LISP S-expressions. An example of a parse tree is illustrated below.  


- Function set: $\mathcal{F} = \{+, -, *, \div\}$
- Terminal set: $\mathcal{T} = \{A, X, C, B\}$
- Example program: $\Psi = (A + X) * (C \div B)$

**Example of a parse tree as used in conventional (tree-based) GP.**


GP-driven search (Algorithm 1) relies on the transformation of a population of candidate programs iteratively, from generation to generation, into new populations consisting of programs with the expectation of an improved fitness score.

**Algorithm 1: Basic Genetic Programming Search Algorithm**

1. $\mathcal{P} \gets \texttt{Initialization}(\mathcal{P})$ (initial population)
2. $\mathcal{F_P} \gets \texttt{Evaluate}(\mathcal{P})$ (evaluate the fitness of the population)
3. **repeat**
   - $\mathcal{Q} \gets \texttt{Breeding}(\mathcal{P})$ (breed $\lambda$ offspring by mutation)
   - $\mathcal{F_Q} \gets \texttt{Evaluation}(\mathcal{Q})$ (evaluate the fitness of the offspring)
   - $\mathcal{P} \gets \texttt{Replace}(\mathcal{P}, \mathcal{F_P}, \mathcal{Q}, \mathcal{F_Q})$ (replace current population following certain criteria)
4. **until** $\mathcal{P}$ *meets termination criterion or a certain number of evaluations*
5. **return** $\mathcal{P}$

<!-- A parse tree is a composition of elements taken from the function set $\mathcal{F}$ and terminal set $\mathcal{T}$. The decoding of the parse tree in the example leads to the symbolic expression $\Psi$ which is the phenotype of $\mathcal{P}$ -->

A recent joint effort among researchers in the field of GP led
to the creation of TinyverseGP [@tinyversegp25], a Python library implementing the different GP representations <!-- Reviewer note: here we mention several representations, but it is never mentioned that researchers designed other possible representations than the parse tree --> along with their original evolutionary algorithm implementation. This framework also
supports different problem domains such as symbolic regression, logic synthesis, and policy learning and connects to well established problem sets. It allows to easily incorporate advanced features, new problem domains,
and new algorithms. 

TinyverseGP takes a step forward in benchmarking GP by providing:

- Minimalist implementation of different representations for GP.
- Modular design making it easy to integrate new representations.
- Support to different problem domains with a common interface, making it transparent to the GP implementation how to evaluate each problem.

## Statement of Need

Despite its success, the field of GP is still highly fragmented as there exists many variations of their inner components and solution representations, significantly altering the search space. Besides, as new variations and implementations are proposed, the number of hyperparameters grows, with very limited sensitivity analysis to the advised default values or reasoning on how their values affect the overall behaviour (e.g., for exploitation-exploration balance). Alternative representations to the predominant tree-based one have been proposed, such as linear sequence representations [@ieee94:perkis; @Openshaw94buildingnew], graph-based representation models [@Poli1996; @miller:1999:ACGP], or grammar-guided representations [@ryan:1998:geepal]. The interplay between these new representation models and the discovery of new applications led to a versatile landscape within the field of GP. However, as diverse contexts can easily entail fragmentation and encapsulation, the field of GP suffers from local over-concentration of respective representation models, linked with their use for particular problems, hindering progress towards a more unified and cross-domain knowledge in GP.

## State of the field

<!-- Reviewer note: I am a bit confused about the articulation of the two sections. I'd see the one above as the state of the field: alternative representations, multiplication of the hyperparameters, benchmarks hard to connect to the methods. Then in the statement of need, which I'd put second, I'd mention the need for a unified way to evaluate all these articulation, a unified interface to the datasets, and the need for proper HPO, profiling and HP sensitivity analysis tools. -->

Like in other subfields of randomised search heuristics, obsolete real-world problems have been proposed and declared as benchmarks. As these benchmark problems accumulated over time, they were bundled and proposed in the framework of benchmark suites for major problem domains such as program synthesis, symbolic regression, and logic synthesis. However, when being used for cross-domain evaluation of GP representations or search algorithms, the respective GP implementation has to be connected with each of these benchmark suites, entailing an implementation overhead.
Apart from the upcoming implementation of other representations and the more general program synthesis domain, a challenge still remains to use TinyverseGP as a benchmarking tool: hyperparameter optimisation and performance profiling. 
Many GP methods require a large number of hyperparameters meant to fine-tune the behaviour of the search to either adapt to certain needs (e.g., constraining the grammar) or to improve the performance to a specific problem instance (e.g., change the balance between exploration and exploitation). Some of these hyperparamters are unintuitive in the sense that the effect of changing their values is unpredictable to the overall performance of the method. Also, since different methods and representations require an overlapping, but different, set of hyperparamters, a common API to perform HPO is challenging.

## Software design
An explanation of the trade-offs you weighed, the design/architecture you chose, and why it matters for your research application. This should demonstrate meaningful design thinking beyond a superficial code structure description.

## Research impact statement
Evidence of realized impact (publications, external use, integrations) or credible near-term significance (benchmarks, reproducible materials, community-readiness signals). The evidence should be compelling and specific, not aspirational.

## AI usage disclosure
Transparent disclosure of any use of generative AI in the software creation, documentation, or paper authoring. If no AI tools were used, state this explicitly. If AI tools were used, describe how they were used and how the quality and correctness of AI-generated content was verified.

## Acknowledgements
This work was supported by an Alexander von Humboldt Professorship in AI held by Holger Hoos, the Czech Science Foundation project 25-15490S and Conselho Nacional de Desenvolvimento Cientifico e Tecnologico (CNPq) grant 301596/2022-0.
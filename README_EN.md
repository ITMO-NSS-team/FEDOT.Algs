# FEDOT.Algs
FEDOT is a framework for generative automatic machine learning that allows you to create complex composite models that provide an effective solution to various applied problems.
This repository contains implementations of various algorithms included in the FEDOT framework.

There are 4 main algorithms:

- Algorithm [E*](https://github.com/ITMO-NSS-team/FEDOT.Algs/wiki/E*-%D0%B0%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC%D1%8B), allowing user to create data-driven models from elementary operators (differential operators, functions);
- Algorithm  [GPComp](https://github.com/ITMO-NSS-team/FEDOT.Algs/wiki/%D0%9E%D0%BF%D0%B8%D1%81%D0%B0%D0%BD%D0%B8%D0%B5-%D0%B0%D0%BB%D0%B3%D0%BE%D1%80%D0%B8%D1%82%D0%BC%D0%B0-%D0%B4%D0%BB%D1%8F-%D1%81%D0%BE%D0%B7%D0%B4%D0%B0%D0%BD%D0%B8%D1%8F-%D1%86%D0%B5%D0%BF%D0%BE%D1%87%D0%B5%D0%BA-%D0%BC%D0%BE%D0%B4%D0%B5%D0%BB%D0%B5%D0%B9-%D0%9C%D0%9E), allowing you to optimize the chains of ML models according to a given criterion;  
- Algorithm  [BN-synthetic-generator](https://github.com/ITMO-NSS-team/bayesian-synthetic-generator/wiki), allowing to generate synthetic personal data based on the construction of a hierarchical block structure of a Bayesian network.
- Algorithm  [PS](https://github.com/ITMO-NSS-team/FEDOT.Algs/wiki/Patterns:descr), based on E* allowing to obtain data-driven algebraic expression;

Also, the repository contains 2  applications that illustrate the work with these algorithms:
- [PDE discovery](https://github.com/ITMO-NSS-team/FEDOT.Algs/blob/master/estar/examples/ESTAR_synth_wave.ipynb) using EPDE;
- [Credit scoring](https://github.com/ITMO-NSS-team/FEDOT.Algs/wiki/%D0%9F%D1%80%D0%B8%D0%BA%D0%BB%D0%B0%D0%B4%D0%BD%D0%BE%D0%B9-%D0%BF%D1%80%D0%B8%D0%BC%D0%B5%D1%80-(%D0%BA%D1%80%D0%B5%D0%B4%D0%B8%D1%82%D0%BD%D1%8B%D0%B9-%D1%81%D0%BA%D0%BE%D1%80%D0%B8%D0%BD%D0%B3)) using FEDOT.

For more information on the basic concepts and approaches of generative automatic machine learning, the principles of the framework, the description of algorithms, as well as the user-programmer's documentation for subject applications, see the section (rus) [Wiki](https://github.com/ITMO-NSS-team/FEDOT.Algs/wiki).

The main version of the documentation for the Fedot framework is available here: https://itmo-nss-team.github.io/FEDOT.Docs.

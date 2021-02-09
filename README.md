# ml_kitchen_sink

Requirements:
  SKlearn
  SK-opt
  pandas
  

Module for the hyperparameter optimization of sklearn algorithms in predicting NMR shifts

Requires pandas dataframes as inputs with the columns containing 'atomic_rep' and 'shift'

use in tandem with mol_translator to generate the required dataframes from nmr log files.


# ml_kitchen_sink

Requirements:
  SKlearn
  SK-opt
  pandas
  numpy

Module for the hyperparameter optimization of sklearn algorithms in predicting NMR shifts

Requires pandas dataframes as inputs with the columns containing 'atomic_rep' and 'shift'

use in tandem with mol_translator (https://github.com/Buttsgroup/mol_translator/tree/master/mol_translator) to generate the required dataframes from nmr log files.


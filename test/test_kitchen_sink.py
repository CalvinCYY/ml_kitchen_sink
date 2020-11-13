'''
Write test for kitchen sink
'''

from ml_kitchen_sink.cv import kitchen_sink as ks

def test_model_selection_cv():
    test_atoms = "test_dataset/test_df_atoms.pkl"
    test_pairs = "test_dataset/test_df_pairs.pkl"

    result_dict = ks.model_selection_cv(atoms_file=test_atoms)
    print(result_dict)

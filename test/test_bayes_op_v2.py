from ml_kitchen_sink.cv import bayes_opt_v2 as bayes_opt
from ml_kitchen_sink.cv.params_search import reg_param_search
from sklearn.model_selection import train_test_split

def test_model_selection_cv():
    test_atoms = "test/test_dataset/test_df_atoms.pkl"
    test_pairs = "test/test_dataset/test_df_pairs.pkl"

    result_dict = ks.model_selection_cv(test_atoms)
    print(result_dict)

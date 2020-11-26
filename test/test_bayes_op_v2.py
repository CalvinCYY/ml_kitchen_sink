from ml_kitchen_sink.cv import bayes_opt_v2 as bayes_opt

def on_step(optim_result):
    """
    View scores after each iteration
    while performing Bayesian
    Optimization in Skopt"""
    score = dtr_search.best_score_
    print("best score: %s" % score)
    if score >= -0.90:
        print('Interrupting!')
        return True

test_atoms = "/Users/bd20841/dataset/atoms_df_data4.pkl"
test_pairs = "test_dataset/test_df_pairs.pkl"

dtr_search, X_train, y_train, X_test, y_test= bayes_opt.model_selection_cv(test_atoms)
dtr_search.fit(X_train, y_train, callback=on_step)

print("val. score: %s" % dtr_search.best_score_)
print("test score: %s" % dtr_search.score(X_test, y_test))
print("best score: %s" % dtr_search.best_params_)

def print_scores(alg, cv_model):
    if alg == 'DTR':
        print('               criterion: {}'.format(cv_model.x[0]))
        print('                splitter: {}'.format(cv_model.x[1]))
        print('       min_samples_split: {}'.format(cv_model.x[2]))
        print('        min_samples_leaf: {}'.format(cv_model.x[3]))
        print('min_weight_fraction_leaf: {}'.format(cv_model.x[4]))
        print('            max_features: {}'.format(cv_model.x[5]))
        print('   min_impurity_decrease: {}'.format(cv_model.x[6]))

    elif alg == 'KRR':
        print('                   alpha: {}'.format(cv_model.x[0]))
        print('                  kernel: {}'.format(cv_model.x[1]))
        print('                   gamma: {}'.format(cv_model.x[2]))
        print('                  degree: {}'.format(cv_model.x[3]))
        print('                   coef0: {}'.format(cv_model.x[4]))

    elif alg == 'KNN':
        print('             n_neighbors: {}'.format(cv_model.x[0]))
        print('                 weights: {}'.format(cv_model.x[1]))
        print('               algorithm: {}'.format(cv_model.x[2]))
        print('               leaf_size: {}'.format(cv_model.x[3]))
        print('                       p: {}'.format(cv_model.x[4]))
        print('                  metric: {}'.format(cv_model.x[5]))

    elif alg == 'RFR':
        print('            n_estimators: {}'.format(cv_model.x[0]))
        print('               criterion: {}'.format(cv_model.x[1]))
        print('       min_samples_split: {}'.format(cv_model.x[2]))
        print('        min_samples_leaf: {}'.format(cv_model.x[3]))
        print('min_weight_fraction_leaf: {}'.format(cv_model.x[4]))
        print('            max_features: {}'.format(cv_model.x[5]))
        print('   min_impurity_decrease: {}'.format(cv_model.x[6]))
        print('               bootstrap: {}'.format(cv_model.x[7]))
        print('               oob_score: {}'.format(cv_model.x[8]))
        print('               ccp_alpha: {}'.format(cv_model.x[9]))

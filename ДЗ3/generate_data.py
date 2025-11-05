def generate_ab_data(num_days, frac_test):
    NUM_DAYS = num_days
    NUM_USERS = 2_032
    np.random.seed(1)

    user_ids = np.arange(1, NUM_USERS+1)
    test_size = int(NUM_USERS * frac_test)
    test_group = np.random.choice(user_ids, test_size, replace=False)
    test_group_flg = np.array([id in test_group for id in user_ids]).astype(int)
    user_p_app = sps.beta(a=1, b=2).rvs(NUM_USERS)
    user_p_app += (test_group_flg * 0.02)
    user_p_app = np.minimum(1, user_p_app)
    user_p_workouts = sps.beta(a=1, b=2).rvs(NUM_USERS)
    user_p_workouts += (test_group_flg * 0.05)
    user_p_workouts = np.minimum(1, user_p_workouts)
    user_p_buy = sps.beta(a=1, b=10).rvs(NUM_USERS)
    users_avg_train_minutes = sps.gamma(a=20, scale=5).rvs(NUM_USERS)

    user_app_by_day = sps.bernoulli(user_p_app).rvs((NUM_DAYS, NUM_USERS))
    user_workouts_by_day = user_app_by_day * sps.bernoulli(user_p_workouts).rvs((NUM_DAYS, NUM_USERS))
    users_train_minutes_by_day = user_workouts_by_day * np.minimum(200, np.maximum(20, 
                                sps.norm(loc=users_avg_train_minutes, scale=10).rvs((NUM_DAYS, NUM_USERS))))
    user_buy_by_day = user_workouts_by_day * sps.bernoulli(user_p_buy).rvs((NUM_DAYS, NUM_USERS))

    users = np.repeat([user_ids], NUM_DAYS, axis=0)
    test_groups = np.repeat([test_group_flg], NUM_DAYS, axis=0)
    dates = np.repeat(pd.date_range(start='2025-09-01', periods=NUM_DAYS, freq='D'), NUM_USERS)
    data = pd.DataFrame(data={
        'user_id': users.flatten(),
        'date': dates,
        'use_app_flg': user_app_by_day.flatten(),
        'workout_done_flg': user_workouts_by_day.flatten(),
        'workout_minutes': users_train_minutes_by_day.flatten(),
        'upgrade_analysis_flg': user_buy_by_day.flatten(),
        'test_group': test_groups.flatten()
    })

    return data
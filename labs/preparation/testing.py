from helpers.dslabs_functions import mvi_by_dropping, evaluate_approach

import pandas as pd


def render_mv(df):
    print(df.isna().sum())


def check(df):
    cols = df.columns
    print(df.shape)

    print(f"col\trow")
    print("-----")
    for i in range(101):
        cur = mvi_by_dropping(df, min_pct_per_variable=(i / 100), min_pct_per_record=0)
        print(i / 100)
        print(cur.shape)
        if cur.columns.to_list() != cols.to_list():
            print(cur.columns.symmetric_difference(cols))
        # print(mvi_by_dropping(df, min_pct_per_variable=0.9, min_pct_per_record=(i / 100)).shape)


if __name__ == "__main__":
    cov = pd.read_csv('../../datasets/prepared/class_pos_covid_2_cust.csv', na_values="")
    render_mv(cov)
    #check(cov)

    # fin = pd.read_csv('../../datasets/prepared/class_credit_score_2_1.csv', na_values="")
    # render_mv(fin)
    # check(fin)

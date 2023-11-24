import numpy as np
import pandas as pd

from helpers.dslabs_functions import get_variable_types


def calulate_mean_age(age_categories):
    mean_ages = []
    for value in age_categories:
        split = value.split()
        age1, age2 = int(split[1]), split[3]
        if age2.isdigit():
            mean_ages.append((age1 + int(age2)) / 2)
        else:
            mean_ages.append(85)  # 85 is an approximation of the mean age, chosen arbitrarily
    return mean_ages


def yes_no_mapping(answer):
    if pd.isna(answer):
        return np.nan
    elif answer.startswith('Yes'):
        return 1
    elif answer.startswith('No'):
        return 0
    else:
        return np.nan  # or a specific code for answers that do not start with Yes/No


def parse_location_file(file_path):
    location_mapping = {}

    with open(file_path, 'r') as file:
        next(file)  # Skip the header line
        for line in file:
            # Remove quotation marks and split the line
            parts = line.strip().replace('"', '').split(':')
            state = parts[0].strip()
            coords = parts[1].strip().strip('[]').split(',')
            longitude = float(coords[0].strip())
            latitude = float(coords[1].strip())

            location_mapping[state] = {'Latitude': latitude, 'Longitude': longitude}

    return location_mapping


def add_coordinates(df, mapping):
    # Extract latitude and longitude from the mapping
    df['Latitude'] = df['State'].map(lambda x: mapping[x]['Latitude'] if x in mapping else None)
    df['Longitude'] = df['State'].map(lambda x: mapping[x]['Longitude'] if x in mapping else None)
    return df


def encode_symbolic(df, encoding):
    df = pd.get_dummies(df, columns=['RaceEthnicityCategory'], dummy_na=False)
    transform_bools(df, "RaceEthnicityCategory")
    file_path = "../../datasets/State, LON, LAT.txt"
    state_encoding = parse_location_file(file_path)
    df = add_coordinates(df, state_encoding)
    df = df.drop(columns=["State"])
    df = df.replace(encoding, inplace=False)
    return df


def transform_bools(df, keyword):
    for col in df.columns:
        if col.startswith(keyword):
            df[col] = df[col].astype(int)
    return df


def encode_health():
    df = pd.read_csv('../../datasets/class_pos_covid.csv')

    # Binaries
    yes_no: dict[str, int] = {"no": 0, "No": 0, "yes": 1, "Yes": 1}
    sex_values: dict[str, int] = {"Female": 0, "Male": 1}
    variable_types = get_variable_types(df)
    encoding = {variable: yes_no for variable in variable_types["binary"][1:]}
    encoding["Sex"] = sex_values
    df = df.replace(encoding, inplace=False)

    # Symbolic
    general_health_encoding = {'Very good': 3, 'Excellent': 4, 'Fair': 1, 'Poor': 0, 'Good': 2, 'nan': np.nan}
    last_checkup_time_encoding = {'Within past year (anytime less than 12 months ago)': 0.5, 'nan': np.nan,
                                  'Within past 2 years (1 year but less than 2 years ago)': 1.5,
                                  'Within past 5 years (2 years but less than 5 years ago)': 3.5,
                                  '5 or more years ago': 7}
    removed_teeth_encoding = {"nan": np.nan, 'None of them': 0, '1 to 5': 2, '6 or more, but not all': 13, 'All': 32}
    had_diabetes_encoding = {'Yes': 2, 'No': 0, 'No, pre-diabetes or borderline diabetes': 1, "nan": np.nan,
                             'Yes, but only during pregnancy (female)': 1}
    smoker_status_encoding = {'Never smoked': 0, 'Current smoker - now smokes some days': 2,
                              'Former smoker': 1, "nan": np.nan, 'Current smoker - now smokes every day': 3}
    ECiggarette_usage_encoding = {'Not at all (right now)': 1,
                                  'Never used e-cigarettes in my entire life': 0, 'Use them every day': 3,
                                  'Use them some days': 2, "nan": np.nan}

    age_category_value_counts = df["AgeCategory"].value_counts()
    mean_ages = calulate_mean_age(age_category_value_counts.index)
    age_category_encoding = {age_category_value_counts.index[i]: mean_ages[i] for i in range(len(mean_ages))}
    age_category_encoding["nan"] = np.nan

    tetanus_encoding = {answer: yes_no_mapping(answer) for answer in df['TetanusLast10Tdap'].unique()}

    encoding = {}
    encoding["GeneralHealth"] = general_health_encoding
    encoding["LastCheckupTime"] = last_checkup_time_encoding
    encoding["RemovedTeeth"] = removed_teeth_encoding
    encoding["HadDiabetes"] = had_diabetes_encoding
    encoding["SmokerStatus"] = smoker_status_encoding
    encoding["ECigaretteUsage"] = ECiggarette_usage_encoding
    encoding["AgeCategory"] = age_category_encoding
    encoding["TetanusLast10Tdap"] = tetanus_encoding

    df = encode_symbolic(df, encoding)

    df.to_csv("../../datasets/prepared/class_pos_covid_encoded_1.csv")


if __name__ == "__main__":
    encode_health()

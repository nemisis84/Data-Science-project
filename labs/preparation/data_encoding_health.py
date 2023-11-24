
import pandas as pd
from helpers.dslabs_functions import get_variable_types, encode_cyclic_variables, dummify
import numpy as np


def calulate_mean_age(age_categories):
    mean_ages = []
    for value in age_categories:
        split = value.split()
        age1, age2 = int(split[1]), split[3]
        if age2.isdigit():
            mean_ages.append((age1+int(age2))/2)
        else:
            mean_ages.append(85)
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
    df['Latitude'] = df['State'].map(lambda x: mapping[x]['Latitude'] if x in mapping else np.nan)
    df['Longitude'] = df['State'].map(lambda x: mapping[x]['Longitude'] if x in mapping else np.nan)
    return df

def encode_symbolic(df, encoding):
    
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

    #Binaries
    yes_no: dict[str, int] = {"no": 0, "No": 0, "yes": 1, "Yes": 1}
    sex_values: dict[str, int] = {"Female": 0, "Male": 1}
    variable_types = get_variable_types(df)
    encoding = {variable: yes_no for variable in variable_types["binary"][1:]}
    encoding["Sex"] = sex_values
    df = df.replace(encoding, inplace=False)

    # Symbolic
    general_health_encoding = {'Very good': 3, 'Excellent': 4, 'Fair': 1, 'Poor': 0, 'Good': 2, 'nan': np.nan}
    last_checkup_time_encoding = {'Within past year (anytime less than 12 months ago)': 0, 'nan': np.nan,
                                  'Within past 2 years (1 year but less than 2 years ago)': 1,
                                  'Within past 5 years (2 years but less than 5 years ago)': 2,
                                  '5 or more years ago': 3}
    removed_teeth_encoding = {"nan": np.nan, 'None of them': 0, '1 to 5': 2, '6 or more, but not all': 13, 'All': 32}
    had_diabetes_encoding = {'Yes': 2, 'No': 0, 'No, pre-diabetes or borderline diabetes': 1, "nan": np.nan,
                             'Yes, but only during pregnancy (female)': 1}
    smoker_status_encoding = {'Never smoked': 0, 'Current smoker - now smokes some days': 2,
                              'Former smoker': 1, "nan": np.nan, 'Current smoker - now smokes every day': 3}
    ECiggarette_usage_encoding = {'Not at all (right now)': 1,
                                  'Never used e-cigarettes in my entire life': 0, 'Use them every day': 3,
                                  'Use them some days': 2, "nan": np.nan}
    age_category_encoding = {'Age 80 or older':13, 'Age 40 to 44':5, 'Age 75 to 79':12,
                            'Age 70 to 74':11, 'Age 55 to 59':8, 'Age 65 to 69':10, 'Age 60 to 64':9,
                            'Age 50 to 54':7, 'Age 45 to 49':6, 'Age 35 to 39':4, 'Age 30 to 34':3,
                            'Age 25 to 29':2, 'Age 18 to 24':0, "nan": np.nan}
    race_ethnicity_category_encoding = {'White only, Non-Hispanic':4, 'Black only, Non-Hispanic':2,
       'Multiracial, Non-Hispanic':3, "nan": np.nan, 'Hispanic':0,
       'Other race only, Non-Hispanic':5}

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
    encoding["RaceEthnicityCategory"] = race_ethnicity_category_encoding
    df = encode_symbolic(df, encoding)

    df.to_csv("../../datasets/prepared/class_pos_covid_encoded_1.csv")

if __name__ == "__main__":
    encode_health()

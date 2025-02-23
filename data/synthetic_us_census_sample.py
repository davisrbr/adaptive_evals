#!/usr/bin/env python
"""
synthetic_population_pipeline.py

This script generates a synthetic population using U.S. Census ACS 5-Year data.
It loads state-level data for several demographic and socioeconomic attributes and computes
target marginal proportions. The attributes include:

  • Gender (from B01001)
  • Ethnicity (from B03002)
  • Education (from B15003)
  • Language (from B16001)
  • Age (sampled continuously and rounded; later bucketed into 0–17, 18–64, 65+ for IPF checks)
  • Income (sampled continuously as an annual income value; later bucketed into Low/Medium/High)
  • SES (bucketed into Low, Middle, and High based on ACS poverty data; with an upper cap on High SES)
  • Occupation (from C24050)
  • UrbanRural (default placeholder)

For each state, individuals are initially sampled using ACS‐derived proportions. Then, an
Iterative Proportional Fitting (IPF) procedure is applied (updating one attribute per iteration)
to adjust record weights so that the weighted marginals match ACS targets.
For continuous attributes (Age and Income), the raw values are kept in the final CSV, but for
IPF and diagnostic checks the values are bucketed.

Outputs:
  - "synthetic_population.csv": Final synthetic population (with rounded Age and continuous Income).
  - "optimization.csv": Log of IPF iterations.

Dependencies:
  pip install requests pandas numpy scipy tqdm
"""

import requests
import pandas as pd
import numpy as np
from tqdm import tqdm
import click

# ---------------------------
# 1. Data Loading Functions
# ---------------------------
def load_census_data(api_key):
    base_url = 'https://api.census.gov/data/2019/acs/acs5'
    variables = ['NAME', 'B01001_001E', 'B01001_002E', 'B01001_026E']
    url = f"{base_url}?get={','.join(variables)}&for=state:*&key={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Error fetching Census data.")
    data = response.json()
    df = pd.DataFrame(data[1:], columns=data[0])
    for col in ['B01001_001E', 'B01001_002E', 'B01001_026E']:
        df[col] = pd.to_numeric(df[col])
    return df

def load_ethnicity_data(api_key):
    base_url = 'https://api.census.gov/data/2019/acs/acs5'
    variables = ['NAME', 'B03002_001E', 'B03002_002E', 'B03002_003E',
                 'B03002_004E', 'B03002_005E', 'B03002_006E',
                 'B03002_007E', 'B03002_008E', 'B03002_009E']
    url = f"{base_url}?get={','.join(variables)}&for=state:*&key={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Error fetching ethnicity data from Census API.")
    data = response.json()
    df_eth = pd.DataFrame(data[1:], columns=data[0])
    for col in ['B03002_001E','B03002_002E','B03002_003E','B03002_004E',
                'B03002_005E','B03002_006E','B03002_007E','B03002_008E','B03002_009E']:
        df_eth[col] = pd.to_numeric(df_eth[col])
    return df_eth

def load_education_data(api_key):
    base_url = 'https://api.census.gov/data/2019/acs/acs5'
    vars_edu = ['NAME'] + [f"B15003_{str(i).zfill(3)}E" for i in range(1, 26)]
    url = f"{base_url}?get={','.join(vars_edu)}&for=state:*&key={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Error fetching education data from Census API.")
    data = response.json()
    edu_df = pd.DataFrame(data[1:], columns=data[0])
    for col in vars_edu[1:]:
        edu_df[col] = pd.to_numeric(edu_df[col])
    cols_less_hs = [f"B15003_{str(i).zfill(3)}E" for i in range(2, 17)]
    edu_df['LessThanHS'] = edu_df[cols_less_hs].sum(axis=1)
    edu_df['HighSchool'] = edu_df['B15003_017E']
    edu_df['SomeCollege'] = edu_df['B15003_018E'] + edu_df['B15003_019E']
    cols_bach = [f"B15003_{str(i).zfill(3)}E" for i in range(20, 26)]
    edu_df['BachelorsPlus'] = edu_df[cols_bach].sum(axis=1)
    edu_df['TotalEdu'] = edu_df['B15003_001E']
    edu_df['p_LessThanHS'] = edu_df['LessThanHS'] / edu_df['TotalEdu']
    edu_df['p_HighSchool'] = edu_df['HighSchool'] / edu_df['TotalEdu']
    edu_df['p_SomeCollege'] = edu_df['SomeCollege'] / edu_df['TotalEdu']
    edu_df['p_BachelorsPlus'] = edu_df['BachelorsPlus'] / edu_df['TotalEdu']
    edu_df = edu_df[['NAME', 'state', 'p_LessThanHS', 'p_HighSchool', 'p_SomeCollege', 'p_BachelorsPlus']]
    return edu_df

def load_language_data(api_key):
    base_url = 'https://api.census.gov/data/2019/acs/acs5'
    variables = ['NAME', 'B16001_001E', 'B16001_002E', 'B16001_003E']
    url = f"{base_url}?get={','.join(variables)}&for=state:*&key={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Error fetching language data from Census API.")
    data = response.json()
    lang_df = pd.DataFrame(data[1:], columns=data[0])
    for col in ['B16001_001E','B16001_002E','B16001_003E']:
        lang_df[col] = pd.to_numeric(lang_df[col])
    lang_df['p_English'] = lang_df['B16001_002E'] / lang_df['B16001_001E']
    lang_df['p_Spanish'] = lang_df['B16001_003E'] / lang_df['B16001_001E']
    lang_df['p_Other'] = 1 - (lang_df['p_English'] + lang_df['p_Spanish'])
    lang_df = lang_df[['NAME', 'state', 'p_English', 'p_Spanish', 'p_Other']]
    return lang_df

def load_age_data(api_key):
    base_url = 'https://api.census.gov/data/2019/acs/acs5'
    variables = ['NAME', 'B01001_001E',
                 'B01001_003E','B01001_004E','B01001_005E','B01001_006E',
                 'B01001_020E','B01001_021E','B01001_022E','B01001_023E','B01001_024E','B01001_025E',
                 'B01001_027E','B01001_028E','B01001_029E','B01001_030E',
                 'B01001_044E','B01001_045E','B01001_046E','B01001_047E','B01001_048E','B01001_049E']
    url = f"{base_url}?get={','.join(variables)}&for=state:*&key={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Error fetching age data from Census API.")
    data = response.json()
    age_df = pd.DataFrame(data[1:], columns=data[0])
    for col in variables[1:]:
        age_df[col] = pd.to_numeric(age_df[col])
    age_df['Age_0_17'] = (age_df['B01001_003E'] + age_df['B01001_004E'] +
                          age_df['B01001_005E'] + age_df['B01001_006E'] +
                          age_df['B01001_027E'] + age_df['B01001_028E'] +
                          age_df['B01001_029E'] + age_df['B01001_030E'])
    age_df['Age_65_plus'] = (age_df['B01001_020E'] + age_df['B01001_021E'] +
                             age_df['B01001_022E'] + age_df['B01001_023E'] +
                             age_df['B01001_024E'] + age_df['B01001_025E'] +
                             age_df['B01001_044E'] + age_df['B01001_045E'] +
                             age_df['B01001_046E'] + age_df['B01001_047E'] +
                             age_df['B01001_048E'] + age_df['B01001_049E'])
    age_df['Age_18_64'] = age_df['B01001_001E'] - (age_df['Age_0_17'] + age_df['Age_65_plus'])
    age_df['p_Age_0_17'] = age_df['Age_0_17'] / age_df['B01001_001E']
    age_df['p_Age_18_64'] = age_df['Age_18_64'] / age_df['B01001_001E']
    age_df['p_Age_65_plus'] = age_df['Age_65_plus'] / age_df['B01001_001E']
    age_df = age_df[['NAME', 'state', 'p_Age_0_17', 'p_Age_18_64', 'p_Age_65_plus']]
    return age_df

def load_income_data(api_key):
    base_url = 'https://api.census.gov/data/2019/acs/acs5'
    variables = ['NAME', 'B19001_001E', 'B19001_002E', 'B19001_003E', 'B19001_004E',
                 'B19001_005E', 'B19001_006E', 'B19001_007E', 'B19001_008E', 'B19001_009E']
    url = f"{base_url}?get={','.join(variables)}&for=state:*&key={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Error fetching income data from Census API.")
    data = response.json()
    inc_df = pd.DataFrame(data[1:], columns=data[0])
    for col in variables[1:]:
        inc_df[col] = pd.to_numeric(inc_df[col])
    inc_df['LowIncome'] = inc_df['B19001_002E'] + inc_df['B19001_003E'] + inc_df['B19001_004E']
    inc_df['MediumIncome'] = inc_df['B19001_005E'] + inc_df['B19001_006E'] + inc_df['B19001_007E']
    inc_df['HighIncome'] = inc_df['B19001_008E'] + inc_df['B19001_009E']
    inc_df['TotalIncome'] = inc_df['B19001_001E']
    inc_df['p_LowIncome'] = inc_df['LowIncome'] / inc_df['TotalIncome']
    inc_df['p_MediumIncome'] = inc_df['MediumIncome'] / inc_df['TotalIncome']
    inc_df['p_HighIncome'] = inc_df['HighIncome'] / inc_df['TotalIncome']
    inc_df = inc_df[['NAME', 'state', 'p_LowIncome', 'p_MediumIncome', 'p_HighIncome']]
    return inc_df

def load_ses_data(api_key):
    base_url = 'https://api.census.gov/data/2019/acs/acs5'
    variables = ['NAME', 'B17001_001E', 'B17001_002E']
    url = f"{base_url}?get={','.join(variables)}&for=state:*&key={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Error fetching SES (poverty) data from Census API.")
    data = response.json()
    ses_df = pd.DataFrame(data[1:], columns=data[0])
    for col in ['B17001_001E','B17001_002E']:
        ses_df[col] = pd.to_numeric(ses_df[col])
    ses_df['p_LowSES'] = ses_df['B17001_002E'] / ses_df['B17001_001E']
    computed = (1 - ses_df['p_LowSES']) / 2
    ses_df['p_HighSES'] = computed.clip(upper=0.30)
    ses_df['p_MiddleSES'] = 1 - ses_df['p_LowSES'] - ses_df['p_HighSES']
    ses_df = ses_df[['NAME', 'state', 'p_LowSES', 'p_MiddleSES', 'p_HighSES']]
    return ses_df

def load_occupation_data(api_key):
    base_url = 'https://api.census.gov/data/2019/acs/acs5'
    variables = ['NAME', 'C24050_001E', 'C24050_002E', 'C24050_003E', 
                 'C24050_004E', 'C24050_005E', 'C24050_006E', 'C24050_007E']
    url = f"{base_url}?get={','.join(variables)}&for=state:*&key={api_key}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("Error fetching occupation data from Census API.")
    data = response.json()
    occ_df = pd.DataFrame(data[1:], columns=data[0])
    for col in variables[1:]:
        occ_df[col] = pd.to_numeric(occ_df[col])
    occ_df['p_Management'] = occ_df['C24050_002E'] / occ_df['C24050_001E']
    occ_df['p_Professional'] = occ_df['C24050_003E'] / occ_df['C24050_001E']
    occ_df['p_Service'] = occ_df['C24050_004E'] / occ_df['C24050_001E']
    occ_df['p_Sales'] = occ_df['C24050_005E'] / occ_df['C24050_001E']
    occ_df['p_NaturalResources'] = occ_df['C24050_006E'] / occ_df['C24050_001E']
    occ_df['p_Production'] = occ_df['C24050_007E'] / occ_df['C24050_001E']
    occ_df = occ_df[['NAME', 'state', 'p_Management', 'p_Professional', 'p_Service', 
                     'p_Sales', 'p_NaturalResources', 'p_Production']]
    return occ_df

# ---------------------------
# 2. Merging and Processing Functions
# ---------------------------
def compute_conditional_gender_distribution(df):
    df['p_male'] = df['B01001_002E'] / df['B01001_001E']
    df['p_female'] = df['B01001_026E'] / df['B01001_001E']
    return df

def merge_ethnicity_data(census_df, eth_df):
    merged = pd.merge(census_df, eth_df, on=["NAME", "state"])
    # Compute Hispanic or Latino as Total minus Not Hispanic/Latino
    merged['eth_Hispanic'] = merged['B03002_001E'] - merged['B03002_002E']
    merged['eth_White']    = merged['B03002_003E']
    merged['eth_Black']    = merged['B03002_004E']
    merged['eth_AIAN']     = merged['B03002_005E']
    merged['eth_Asian']    = merged['B03002_006E']
    merged['eth_NHPI']     = merged['B03002_007E']
    merged['eth_Other']    = merged['B03002_008E'] + merged['B03002_009E']
    merged['eth_total'] = merged['B03002_001E']
    for col in ['eth_Hispanic','eth_White','eth_Black','eth_AIAN','eth_Asian','eth_NHPI','eth_Other']:
        merged[f"p_{col}"] = merged[col] / merged['eth_total']
    return merged

def merge_education_data(merged_df, edu_df):
    return pd.merge(merged_df, edu_df, on=["NAME", "state"])

def merge_language_data(merged_df, lang_df):
    return pd.merge(merged_df, lang_df, on=["NAME", "state"])

# ---------------------------
# Bucket functions for continuous attributes
# ---------------------------
def bucket_age(age):
    if age < 18:
        return "0-17"
    elif age < 65:
        return "18-64"
    else:
        return "65+"

def bucket_income(income):
    if income < 30000:
        return "Low Income"
    elif income < 75000:
        return "Medium Income"
    else:
        return "High Income"

def bucket_ses(ses):
    if ses in ["Low SES", "Middle SES", "High SES"]:
        return ses
    if ses < 0.33:
        return "Low SES"
    elif ses < 0.66:
        return "Middle SES"
    else:
        return "High SES"

# ---------------------------
# 3. Synthetic Population Sampling Functions
# ---------------------------
def sample_state_personas(state_row, n):
    state_name = state_row["NAME"]
    records = []
    for _ in range(n):
        # Gender
        gender = np.random.choice(['Male', 'Female'], p=[state_row['p_male'], state_row['p_female']])
        # Ethnicity
        eth_categories = [
            "Hispanic or Latino", "White (Non-Hispanic)", "Black (Non-Hispanic)",
            "American Indian/Alaska Native", "Asian", "Native Hawaiian/Pacific Islander", "Other/Mixed"
        ]
        eth_probs = np.array([
            state_row['p_eth_Hispanic'],
            state_row['p_eth_White'],
            state_row['p_eth_Black'],
            state_row['p_eth_AIAN'],
            state_row['p_eth_Asian'],
            state_row['p_eth_NHPI'],
            state_row['p_eth_Other']
        ])
        ethnicity = np.random.choice(eth_categories, p=eth_probs/eth_probs.sum())
        # Education
        edu_categories = ["Less than High School", "High School Graduate",
                          "Some College/Associate's", "Bachelor's or Higher"]
        edu_probs = np.array([
            state_row['p_LessThanHS'],
            state_row['p_HighSchool'],
            state_row['p_SomeCollege'],
            state_row['p_BachelorsPlus']
        ])
        education = np.random.choice(edu_categories, p=edu_probs/edu_probs.sum())
        # Language
        lang_categories = ["English Only", "Spanish", "Other"]
        lang_probs = np.array([state_row['p_English'], state_row['p_Spanish'], state_row['p_Other']])
        language = np.random.choice(lang_categories, p=lang_probs/lang_probs.sum())
        # Age: sample continuous then round
        r_age = np.random.rand()
        if r_age < state_row['p_Age_0_17']:
            age_cont = np.random.uniform(0, 18)
        elif r_age < state_row['p_Age_0_17'] + state_row['p_Age_18_64']:
            age_cont = np.random.uniform(18, 65)
        else:
            age_cont = np.random.uniform(65, 100)
        age = int(round(age_cont))
        # Income: sample continuous income from a bucket range
        income_categories = ["Low Income", "Medium Income", "High Income"]
        income_probs = np.array([state_row['p_LowIncome'], state_row['p_MediumIncome'], state_row['p_HighIncome']])
        income_bucket = np.random.choice(income_categories, p=income_probs/income_probs.sum())
        if income_bucket == "Low Income":
            income = np.random.uniform(0, 30000)
        elif income_bucket == "Medium Income":
            income = np.random.uniform(30000, 75000)
        else:
            income = np.random.uniform(75000, 200000)
        # SES: sample discretely based on ACS target proportions for SES.
        p_low = state_row['p_LowSES']
        p_middle = state_row['p_MiddleSES']
        p_high = state_row['p_HighSES']
        ses = np.random.choice(["Low SES", "Middle SES", "High SES"], p=[p_low, p_middle, p_high])
        # Occupation
        occ_categories = ["Management", "Professional", "Service", "Sales", "NaturalResources", "Production"]
        occ_probs = np.array([
            state_row['p_Management'],
            state_row['p_Professional'],
            state_row['p_Service'],
            state_row['p_Sales'],
            state_row['p_NaturalResources'],
            state_row['p_Production']
        ])
        occupation = np.random.choice(occ_categories, p=occ_probs/occ_probs.sum())
        # UrbanRural (placeholder)
        urbanrural = "Urban"
        
        record = {
            'State': state_name,
            'Gender': gender,
            'Ethnicity': ethnicity,
            'Education': education,
            'Language': language,
            'Age': age,
            'Income': income,
            'SES': ses,
            'Occupation': occupation,
            'UrbanRural': urbanrural
        }
        records.append(record)
    return pd.DataFrame(records)

def sample_synthetic_population(merged_df, N):
    print(f"Merged_df has {len(merged_df)} rows. Sample states: {merged_df['NAME'].head().tolist()}")
    state_population = merged_df['B01001_001E']
    state_probs = state_population / state_population.sum()
    sampled_states = np.random.choice(merged_df['NAME'], size=N, p=state_probs)
    print(f"Sampled {len(sampled_states)} states from merged_df.")
    synthetic_data = []
    state_lookup = merged_df.set_index('NAME', drop=False)
    for state in sampled_states:
        try:
            row = state_lookup.loc[state]
        except KeyError:
            print(f"State {state} not found in state_lookup!")
            continue
        record_df = sample_state_personas(row, 1)
        synthetic_data.append(record_df)
    if synthetic_data:
        return pd.concat(synthetic_data, ignore_index=True)
    else:
        return pd.DataFrame()

# ---------------------------
# 4. Vectorized IPF-Based Calibration (Per-State)
# ---------------------------
def calibrate_state_ipf(state_synth, state_target, state_name, attr_map, tol=0.001, max_iterations=100, damping=0.2):
    """
    Calibrates synthetic records for a single state using a vectorized IPF update.
    A damping factor is applied to avoid overshooting.
    """
    state_synth = state_synth.copy()
    n = len(state_synth)
    w = np.ones(n)
    optimization_log = []
    state_name = state_target["NAME"]
    eps = 1e-12
    attributes = list(attr_map.keys())

    # Bucketing functions for continuous attributes
    def ipf_bucket_age(age_array):
        buckets = np.empty(age_array.shape, dtype=object)
        buckets[age_array < 18] = "0-17"
        buckets[(age_array >= 18) & (age_array < 65)] = "18-64"
        buckets[age_array >= 65] = "65+"
        return buckets

    def ipf_bucket_income(income_array):
        buckets = np.empty(income_array.shape, dtype=object)
        buckets[income_array < 30000] = "Low Income"
        buckets[(income_array >= 30000) & (income_array < 75000)] = "Medium Income"
        buckets[income_array >= 75000] = "High Income"
        return buckets

    for iteration in tqdm(range(max_iterations), desc=state_name):
        max_diffs = []
        for attr in attributes:
            if attr == "Age":
                values = ipf_bucket_age(state_synth["Age"].values)
            elif attr == "Income":
                values = ipf_bucket_income(state_synth["Income"].values)
            else:
                values = state_synth[attr].values

            valid = [(cat, colname) for cat, colname in attr_map[attr].items() if colname is not None]
            if valid:
                categories_valid = np.array([cat for cat, _ in valid])
                target_arr = np.array([state_target[colname] for _, colname in valid], dtype=float)
                mask = np.array([values == cat for cat in categories_valid])
                total_w = max(w.sum(), eps)
                group_sum = np.dot(mask, w)
                current_prop = group_sum / total_w

                lower_bound = 0.1 if attr == "Ethnicity" else 0.5
                factors = np.divide(target_arr, current_prop, out=np.ones_like(target_arr), where=(current_prop > eps))
                factors = np.clip(factors, lower_bound, 2.0)
                for j in range(len(categories_valid)):
                    w[mask[j]] *= (1 + damping * (factors[j] - 1))
                diff_attr = np.max(np.abs(current_prop - target_arr))
                max_diffs.append(diff_attr)
                optimization_log.append({
                    "State": state_name,
                    "Iteration": iteration,
                    "Attribute": attr,
                    "MaxDiff": diff_attr
                })
        overall_max_diff = max(max_diffs) if max_diffs else 0.0
        optimization_log.append({
            "State": state_name,
            "Iteration": iteration,
            "Attribute": "Overall",
            "MaxDiff": overall_max_diff
        })
        if overall_max_diff < tol:
            break

    state_synth["weight"] = w
    final_weights = w / w.sum() if w.sum() > 0 else np.ones(n) / n
    indices = np.arange(n)
    resample_idx = np.random.choice(indices, size=n, replace=True, p=final_weights)
    calibrated_state = state_synth.iloc[resample_idx].drop(columns=["weight"])
    return calibrated_state, optimization_log

def enforce_conditional_attributes_ipf(synth_df, merged_df, attr_map, tol=0.001, max_iterations=100, damping=0.2):
    calibrated_records = []
    optimization_logs = []
    for state in merged_df['NAME'].unique():
        state_target = merged_df[merged_df['NAME'] == state].iloc[0]
        state_synth = synth_df[synth_df['State'] == state]
        if len(state_synth) == 0:
            continue
        calibrated, opt_log = calibrate_state_ipf(state_synth, state_target, state, attr_map, tol, max_iterations, damping)
        calibrated_records.append(calibrated)
        optimization_logs.extend(opt_log)
    if calibrated_records:
        return pd.concat(calibrated_records, ignore_index=True), optimization_logs
    else:
        return pd.DataFrame(), optimization_logs

def test_conditional_attributes(synth_df, merged_df, attr_map, tol=0.001, min_samples=1):
    for state in merged_df['NAME'].unique():
        subset = synth_df[synth_df['State'] == state]
        print(f"Testing state: {state}, count: {len(subset)}")
        if len(subset) < min_samples:
            continue
        subset = subset.copy()
        subset["AgeBucket"] = subset["Age"].apply(bucket_age)
        subset["IncomeBucket"] = subset["Income"].apply(bucket_income)
        for attr, cat_map in attr_map.items():
            if attr == "Age":
                values = subset["AgeBucket"]
            elif attr == "Income":
                values = subset["IncomeBucket"]
            else:
                values = subset[attr]
            state_target = merged_df[merged_df['NAME'] == state].iloc[0]
            for category, target_col in cat_map.items():
                if target_col is None:
                    continue
                observed = (values == category).mean()
                expected = state_target[target_col]
                diff = abs(observed - expected)
                print(f"  {attr} - {category}: Synthetic={observed:.3f}, Expected={expected:.3f}, Diff={diff:.3f}")
        print("")

# ---------------------------
# 5. Overall Verification Function
# ---------------------------
def verify_population(synth_df, census_df):
    print("=== Synthetic Population Verification ===\n")
    print("Gender Distribution (Synthetic):")
    print(synth_df['Gender'].value_counts(normalize=True))
    print("\nEthnicity Distribution (Synthetic):")
    print(synth_df['Ethnicity'].value_counts(normalize=True))
    print("\nEducation Distribution (Synthetic):")
    print(synth_df['Education'].value_counts(normalize=True))
    print("\nLanguage Distribution (Synthetic):")
    print(synth_df['Language'].value_counts(normalize=True))
    print("\n=========================================")

# ---------------------------
# 6. Main Execution
# ---------------------------

@click.command()
@click.option('--n', default=10000, help='Number of individuals in the synthetic population.')
def main(n: int):
    api_key = "PUT_API_KEY_HERE"
    
    print("Loading basic Census data...")
    census_df = load_census_data(api_key)
    census_df = compute_conditional_gender_distribution(census_df)
    
    print("Loading Ethnicity data...")
    eth_df = load_ethnicity_data(api_key)
    merged_df = merge_ethnicity_data(census_df, eth_df)
    
    print("Loading Education data...")
    edu_df = load_education_data(api_key)
    merged_df = merge_education_data(merged_df, edu_df)
    
    print("Loading Language data...")
    lang_df = load_language_data(api_key)
    merged_df = merge_language_data(merged_df, lang_df)
    
    print("Loading Age data...")
    age_df = load_age_data(api_key)
    merged_df = pd.merge(merged_df, age_df, on=["NAME", "state"])
    
    print("Loading Income data...")
    inc_df = load_income_data(api_key)
    merged_df = pd.merge(merged_df, inc_df, on=["NAME", "state"])
    
    print("Loading SES data...")
    ses_df = load_ses_data(api_key)
    merged_df = pd.merge(merged_df, ses_df, on=["NAME", "state"])
    
    print("Loading Occupation data...")
    occ_df = load_occupation_data(api_key)
    merged_df = pd.merge(merged_df, occ_df, on=["NAME", "state"])
    
    print(f"Generating synthetic population of {n} individuals...")
    synthetic_population = sample_synthetic_population(merged_df, n)
    
    print("\nSample of Synthetic Population:")
    print(synthetic_population.head())
    
    verify_population(synthetic_population, census_df)
    
    # Define the IPF attribute map (DISABILITY removed)
    attr_map = {
        "Gender": {
            "Male": "p_male",
            "Female": "p_female"
        },
        "Ethnicity": {
            "Hispanic or Latino": "p_eth_Hispanic",
            "White (Non-Hispanic)": "p_eth_White",
            "Black (Non-Hispanic)": "p_eth_Black",
            "American Indian/Alaska Native": "p_eth_AIAN",
            "Asian": "p_eth_Asian",
            "Native Hawaiian/Pacific Islander": "p_eth_NHPI",
            "Other/Mixed": "p_eth_Other"
        },
        "Education": {
            "Less than High School": "p_LessThanHS",
            "High School Graduate": "p_HighSchool",
            "Some College/Associate's": "p_SomeCollege",
            "Bachelor's or Higher": "p_BachelorsPlus"
        },
        "Language": {
            "English Only": "p_English",
            "Spanish": "p_Spanish",
            "Other": "p_Other"
        },
        "Age": {
            "0-17": "p_Age_0_17",
            "18-64": "p_Age_18_64",
            "65+": "p_Age_65_plus"
        },
        "Income": {
            "Low Income": "p_LowIncome",
            "Medium Income": "p_MediumIncome",
            "High Income": "p_HighIncome"
        },
        "SES": {
            "Low SES": "p_LowSES",
            "Middle SES": "p_MiddleSES",
            "High SES": "p_HighSES"
        },
        "Occupation": {
            "Management": "p_Management",
            "Professional": "p_Professional",
            "Service": "p_Service",
            "Sales": "p_Sales",
            "NaturalResources": "p_NaturalResources",
            "Production": "p_Production"
        },
        "UrbanRural": {
            "Urban": None,
            "Rural": None
        }
    }
    
    print("\n--- Conditional Attribute Distribution Before IPF Calibration ---")
    test_conditional_attributes(synthetic_population, merged_df, attr_map, tol=0.001, min_samples=1)
    
    synthetic_population, opt_logs = enforce_conditional_attributes_ipf(synthetic_population, merged_df, attr_map, tol=0.001, max_iterations=10, damping=0.2)
    
    print("\n--- Conditional Attribute Distribution After IPF Calibration ---")
    test_conditional_attributes(synthetic_population, merged_df, attr_map, tol=0.001, min_samples=1)
    
    synthetic_population.to_csv("synthetic_population.csv", index=False)
    print("\nSynthetic population written to synthetic_population.csv")
    
    opt_df = pd.DataFrame(opt_logs)
    opt_df["IsFinal"] = opt_df.groupby("State")["Iteration"].transform(max) == opt_df["Iteration"]
    opt_df.to_csv("optimization.csv", index=False)
    print("Optimization log written to optimization.csv")

if __name__ == "__main__":
    main()

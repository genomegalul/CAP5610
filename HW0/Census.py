import pandas as pd

# Setting file paths.
train_path = "./CensusData/census-income.data"
test_path  = "./CensusData/census-income.test"

# Reading the training data.
train = pd.read_csv(
    train_path,
    names=[f"col{i}" for i in range(42)],
    header=None,
    skipinitialspace=True,
    na_values=["?"],
    engine="python"
)
# Stripping the trailing period.
train["col41"] = train["col41"].str.replace(".", "", regex=False).str.strip()

# Reading the testing data.
test = pd.read_csv(
    test_path,
    names=[f"col{i}" for i in range(42)],
    header=None,
    skipinitialspace=True,
    na_values=["?"],
    engine="python"
)
# Stripping the trailing period.
test["col41"] = test["col41"].str.replace(".", "", regex=False).str.strip()

# (a) Number of people in training with income > 50K.
count_train_over_50k = train[ train["col41"] == "50000+" ].shape[0]

# (b) Number of people in testing with income > 50K.
count_test_over_50k  = test[  test["col41"] == "50000+" ].shape[0]

# (c) Number of people in testing who are "Asian or Pacific Islander".
count_test_asian_pac = test[ test["col10"] == "Asian or Pacific Islander" ].shape[0]

# (d) Average age in training among those with income > 50K.
avg_age_train_over_50k = train.loc[ train["col41"] == "50000+", "col0" ].mean()

# (e) Average age in testing among those with income > 50K.
avg_age_test_over_50k  = test.loc[  test["col41"] == "50000+", "col0" ].mean()

print("Results:")
print(f"(a) Number of people in training with income > 50K: {count_train_over_50k}")
print(f"(b) Number of people in testing with income > 50K: {count_test_over_50k}")
print(f"(c) Number of people in testing who are Asian / Pacific Islander: {count_test_asian_pac}")
print(f"(d) Average age in training among those with income > 50K: {avg_age_train_over_50k}")
print(f"(e) Average age in testing among those with income > 50K: {avg_age_test_over_50k}")

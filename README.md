# predict-animal-outcomes

## quickstart

Requires Python 3.13.3.

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
python prepare_data.py
pytest tests/
pytest tests/ --inspect -s   # show failing rows
python -c "import pyarrow.parquet as pq; print(pq.read_schema('data/dataset.parquet'))"  # inspect parquet schema and metadata
```

## todo:  
1. make a clean dataset that handles the null values on Outcome Type  
2. drop Outcome Subtype - dont care about it and too big of a hint for the model.  
3. ask: i didnt create a manifest.json for when we spawn the parquet and just save a creation time, is this ok or trash.  

## test output

```
tests/test_distributions.py::test_animal_type_values_in_expected_set PASSED [  6%]
tests/test_distributions.py::test_animal_type_dogs_and_cats_dominate PASSED [ 13%]
tests/test_distributions.py::test_outcome_type_values_in_expected_set PASSED [ 20%]
tests/test_distributions.py::test_outcome_type_adoption_rate_in_range PASSED [ 26%]
tests/test_missing_values.py::test_animal_id_not_null PASSED             [ 33%]
tests/test_missing_values.py::test_animal_type_not_null PASSED           [ 40%]
tests/test_missing_values.py::test_outcome_type_not_null FAILED          [ 46%]
tests/test_missing_values.py::test_datetime_not_null PASSED              [ 53%]
tests/test_missing_values.py::test_date_of_birth_mostly_not_null PASSED  [ 60%]
tests/test_missing_values.py::test_name_mostly_not_null PASSED           [ 66%]
tests/test_missing_values.py::test_monthyear_mostly_not_null PASSED      [ 73%]
tests/test_missing_values.py::test_sex_upon_outcome_mostly_not_null PASSED [ 80%]
tests/test_missing_values.py::test_age_upon_outcome_mostly_not_null PASSED [ 86%]
tests/test_missing_values.py::test_breed_mostly_not_null PASSED          [ 93%]
tests/test_missing_values.py::test_color_mostly_not_null PASSED          [100%]

FAILED tests/test_missing_values.py::test_outcome_type_not_null
1 failed, 14 passed
```
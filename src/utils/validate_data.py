import great_expectations as ge

def validate_telco_data(df):
    """
    Validate Telco churn dataset using Great Expectations PandasDataset API.
    Returns: (success: bool, failed_expectations: list)
    """
    ge_df = ge.dataset.PandasDataset(df)

    # Expectations
    ge_df.expect_column_to_exist("customerID")
    ge_df.expect_column_values_to_not_be_null("customerID")
    ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
    ge_df.expect_column_values_to_be_between("tenure", min_value=0)
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0)

    results = ge_df.validate()

    failed_expectations = [
        r["expectation_config"]["expectation_type"]
        for r in results["results"]
        if not r["success"]
    ]

    return results["success"], failed_expectations

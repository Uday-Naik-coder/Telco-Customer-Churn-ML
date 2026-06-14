"""
DATA VALIDATION - Production-Ready Data Quality Checks
======================================================

This module implements comprehensive data validation using Great Expectations,
which is ESSENTIAL for production machine learning systems. Data quality issues
are one of the primary causes of model failures in production.

Why Data Validation is Critical:
1. Prevents training on corrupted/inconsistent data
2. Catches data pipeline failures early  
3. Ensures model assumptions are met
4. Provides audit trail for compliance
5. Reduces silent model degradation

Great Expectations Benefits:
- Declarative data quality tests
- Automated validation reporting
- Integration with MLflow for tracking
- Standardized expectation patterns
- Rich validation result metadata

Production Usage:
- Run validation BEFORE every model training
- Log validation results to MLflow for monitoring
- Fail training pipeline if critical validations fail
- Use for ongoing data monitoring in production

IMPORTANT: Add new expectations as data understanding improves
"""

import great_expectations as ge
from typing import Tuple, List


def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco Customer Churn dataset using Great Expectations.
    
    This function implements critical data quality checks that must pass before model training.
    It validates data integrity, business logic constraints, and statistical properties
    that the ML model expects.
    
    Validation Categories:
    1. Schema Validation: Required columns exist and have correct types
    2. Business Logic: Values fall within expected business ranges
    3. Data Integrity: No critical missing values or impossible combinations
    4. Statistical Properties: Distributions within expected ranges
    
    Args:
        df: Raw pandas DataFrame with telco customer data
        
    Returns:
        Tuple containing:
        - success (bool): True if all validations pass, False if any fail
        - failed_expectations (List[str]): List of failed expectation types for debugging
        
    Example:
        >>> is_valid, failures = validate_telco_data(df)
        >>> if not is_valid:
        ...     print(f"Data validation failed: {failures}")
    """
    print("🔍 Starting data validation with Great Expectations...")
    
    # Convert pandas DataFrame to Great Expectations Dataset
    ge_df = ge.dataset.PandasDataset(df)
    
    # === SCHEMA VALIDATION - ESSENTIAL COLUMNS ===
    print("   📋 Validating schema and required columns...")
    
    # Customer identifier must exist (required for business operations)  
    ge_df.expect_column_to_exist("customerID")
    ge_df.expect_column_values_to_not_be_null("customerID")
    
    # Core demographic features
    ge_df.expect_column_to_exist("gender") 
    ge_df.expect_column_to_exist("Partner")
    ge_df.expect_column_to_exist("Dependents")
    
    # Service features (critical for churn analysis)
    ge_df.expect_column_to_exist("PhoneService")
    ge_df.expect_column_to_exist("InternetService")
    ge_df.expect_column_to_exist("Contract")
    
    # Financial features (key churn predictors)
    ge_df.expect_column_to_exist("tenure")
    ge_df.expect_column_to_exist("MonthlyCharges")
    ge_df.expect_column_to_exist("TotalCharges")
    
    # === BUSINESS LOGIC VALIDATION ===
    print("   💼 Validating business logic constraints...")
    
    # Gender must be one of expected values (data integrity)
    ge_df.expect_column_values_to_be_in_set("gender", ["Male", "Female"])
    
    # Yes/No fields must have valid values
    ge_df.expect_column_values_to_be_in_set("Partner", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("Dependents", ["Yes", "No"])
    ge_df.expect_column_values_to_be_in_set("PhoneService", ["Yes", "No"])
    
    # Contract types must be valid (business constraint)
    ge_df.expect_column_values_to_be_in_set(
        "Contract", 
        ["Month-to-month", "One year", "Two year"]
    )
    
    # Internet service types (business constraint)
    ge_df.expect_column_values_to_be_in_set(
        "InternetService",
        ["DSL", "Fiber optic", "No"]
    )
    
    # === NUMERIC RANGE VALIDATION ===
    print("   📊 Validating numeric ranges and business constraints...")
    
    # Tenure must be non-negative (business logic - can't have negative tenure)
    ge_df.expect_column_values_to_be_between("tenure", min_value=0)
    
    # Monthly charges must be positive (business logic - no free service)
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0)
    
    # Total charges should be non-negative (business logic)
    ge_df.expect_column_values_to_be_between("TotalCharges", min_value=0)
    
    # === STATISTICAL VALIDATION ===
    print("   📈 Validating statistical properties...")
    
    # Tenure should be reasonable (max ~10 years = 120 months for telecom)
    ge_df.expect_column_values_to_be_between("tenure", min_value=0, max_value=120)
    
    # Monthly charges should be within reasonable business range
    ge_df.expect_column_values_to_be_between("MonthlyCharges", min_value=0, max_value=200)
    
    # No missing values in critical numeric features  
    ge_df.expect_column_values_to_not_be_null("tenure")
    ge_df.expect_column_values_to_not_be_null("MonthlyCharges")
    
    # === DATA CONSISTENCY CHECKS ===
    print("   🔗 Validating data consistency...")
    
    # Total charges should generally be >= Monthly charges (except for very new customers)
    # This is a business logic check to catch data entry errors
    ge_df.expect_column_pair_values_A_to_be_greater_than_B(
        column_A="TotalCharges",
        column_B="MonthlyCharges",
        or_equal=True,
        mostly=0.95  # Allow 5% exceptions for edge cases
    )
    
    # === RUN VALIDATION SUITE ===
    print("   ⚙️  Running complete validation suite...")
    results = ge_df.validate()
    
    # === PROCESS RESULTS ===
    # Extract failed expectations for detailed error reporting
    failed_expectations = []
    for r in results["results"]:
        if not r["success"]:
            expectation_type = r["expectation_config"]["expectation_type"]
            failed_expectations.append(expectation_type)
    
    # Print validation summary
    total_checks = len(results["results"])
    passed_checks = sum(1 for r in results["results"] if r["success"])
    failed_checks = total_checks - passed_checks
    
    if results["success"]:
        print(f"✅ Data validation PASSED: {passed_checks}/{total_checks} checks successful")
    else:
        print(f"❌ Data validation FAILED: {failed_checks}/{total_checks} checks failed")
        print(f"   Failed expectations: {failed_expectations}")
    
    return results["success"], failed_expectations

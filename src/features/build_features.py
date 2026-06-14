"""
FEATURE ENGINEERING - Training Pipeline Feature Transformations
================================================================

This module implements the core feature engineering logic for the Telco Customer Churn model.
The transformations applied here are CRITICAL and must be exactly replicated in the serving
pipeline to prevent train/serve skew.

Key Transformations:
1. Binary Encoding: Convert Yes/No, Male/Female to 0/1
2. One-Hot Encoding: Convert multi-category features with drop_first=True
3. Boolean Conversion: Convert boolean columns to integers
4. Type Handling: Ensure consistent data types for XGBoost

CRITICAL PATTERN: Feature Engineering Consistency
- Binary mappings must be deterministic and documented
- One-hot encoding parameters (drop_first=True) must match serving
- Feature column ordering must be preserved for model compatibility
- All transformations must handle missing/unknown values gracefully

Production Considerations:
- New categories in serving data will be handled by fill_value=0 in serving pipeline
- Feature engineering logic is replicated in src/serving/inference.py
- Any changes here require corresponding updates in serving pipeline
"""

import pandas as pd


def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Apply deterministic binary encoding to 2-category features.
    
    This function implements the core binary encoding logic that converts
    categorical features with exactly 2 values into 0/1 integers. The mappings
    are deterministic and must be consistent between training and serving.
    
    Known Binary Mappings:
    - Yes/No → No=0, Yes=1 (business logic: Yes is positive)
    - Male/Female → Female=0, Male=1 (alphabetical consistency)
    
    For unknown binary features, uses stable ordering based on unique values.
    
    Args:
        s: pandas Series with exactly 2 unique values
        
    Returns:
        pandas Series with 0/1 integer encoding, or original series if not binary
        
    IMPORTANT: These mappings must match exactly in serving pipeline (BINARY_MAP)
    """
    # Get unique values and remove NaN
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    # === DETERMINISTIC BINARY MAPPINGS ===
    # CRITICAL: These exact mappings are hardcoded in serving pipeline
    
    # Yes/No mapping (most common pattern in telecom data)
    if valset == {"Yes", "No"}:
        return s.map({"No": 0, "Yes": 1}).astype("Int64")
        
    # Gender mapping (demographic feature)
    if valset == {"Male", "Female"}:
        return s.map({"Female": 0, "Male": 1}).astype("Int64")

    # === GENERIC BINARY MAPPING ===
    # For any other 2-category feature, use stable alphabetical ordering
    if len(vals) == 2:
        # Sort values to ensure consistent mapping across runs
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")

    # === NON-BINARY FEATURES ===
    # Return unchanged - will be handled by one-hot encoding
    return s


def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Apply complete feature engineering pipeline for training data.
    
    This is the main feature engineering function that transforms raw customer data
    into ML-ready features. The transformations must be exactly replicated in the
    serving pipeline to ensure prediction accuracy.
    
    Feature Engineering Pipeline:
    1. Identify categorical vs numeric columns
    2. Split categorical into binary (2 values) vs multi-category (>2 values) 
    3. Apply binary encoding to 2-value features
    4. Apply one-hot encoding to multi-value features
    5. Convert boolean columns to integers
    6. Clean up data types for XGBoost compatibility
    
    Args:
        df: DataFrame with raw customer data including target column
        target_col: Name of target column to exclude from feature engineering
        
    Returns:
        DataFrame with engineered features ready for ML training
        
    Key Parameters:
    - drop_first=True: Prevents multicollinearity in one-hot encoding
    - astype(int): Ensures XGBoost compatibility (no nullable integers)
    
    CRITICAL: Any changes here must be reflected in serving pipeline inference.py
    """
    df = df.copy()
    print(f"🔧 Starting feature engineering on {df.shape[1]} columns...")

    # === STEP 1: Identify Feature Types ===
    # Find categorical columns (object dtype) excluding the target variable
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    
    print(f"   📊 Found {len(obj_cols)} categorical and {len(numeric_cols)} numeric columns")

    # === STEP 2: Split Categorical by Cardinality ===
    # Binary features (exactly 2 unique values) get binary encoding
    # Multi-category features (>2 unique values) get one-hot encoding
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]
    
    print(f"   🔢 Binary features: {len(binary_cols)} | Multi-category features: {len(multi_cols)}")
    if binary_cols:
        print(f"      Binary: {binary_cols}")
    if multi_cols:
        print(f"      Multi-category: {multi_cols}")

    # === STEP 3: Apply Binary Encoding ===
    # Convert 2-category features to 0/1 using deterministic mappings
    for c in binary_cols:
        original_dtype = df[c].dtype
        df[c] = _map_binary_series(df[c].astype(str))
        print(f"      ✅ {c}: {original_dtype} → binary (0/1)")

    # === STEP 4: Convert Boolean Columns ===
    # XGBoost requires integer inputs, not boolean
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f"   🔄 Converted {len(bool_cols)} boolean columns to int: {bool_cols}")

    # === STEP 5: One-Hot Encoding for Multi-Category Features ===
    # CRITICAL: drop_first=True prevents multicollinearity
    if multi_cols:
        print(f"   🌟 Applying one-hot encoding to {len(multi_cols)} multi-category columns...")
        original_shape = df.shape
        
        # Apply one-hot encoding with drop_first=True (same as serving)
        df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
        
        new_features = df.shape[1] - original_shape[1] + len(multi_cols)
        print(f"      ✅ Created {new_features} new features from {len(multi_cols)} categorical columns")

    # === STEP 6: Data Type Cleanup ===
    # Convert nullable integers (Int64) to standard integers for XGBoost
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            # Fill any NaN values with 0 and convert to int
            df[c] = df[c].fillna(0).astype(int)

    print(f"✅ Feature engineering complete: {df.shape[1]} final features")
    return df


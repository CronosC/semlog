import unittest
import pandas as pd
import re
from semanticlog.data.cleaning import *

class TestApplyFunctionToValue(unittest.TestCase):
    def test_apply_function_to_value_with_dict(self):
        sample_dict = {'A': 1, 'B': {'C': 'hello', 'D': 2}}
        result = apply_function_to_value(sample_dict, lambda x: x.upper() if isinstance(x, str) else x)
        self.assertEqual(result, {'A': 1, 'B': {'C': 'HELLO', 'D': 2}})

    def test_apply_function_to_value_with_dataframe(self):
        sample_df = pd.DataFrame({'col1': ['a', 'b'], 'col2': [1, 2]})
        result = apply_function_to_value(sample_df, lambda x: str(x).upper())
        expected_df = pd.DataFrame({'COL1': ['A', 'B'], 'COL2': ['1', '2']})
        pd.testing.assert_frame_equal(result, expected_df)

    def test_apply_function_to_value_with_str(self):
        result = apply_function_to_value('hello', lambda x: x.upper())
        self.assertEqual(result, 'HELLO')


class TestCleanFunctions(unittest.TestCase):

    # Tests for clean_string
    def test_clean_string_casefold_true(self):
        # Special characters replaced and string lowercased.
        self.assertEqual(clean_string("Hello World!", casefold=True), "hello_world_")

    def test_clean_string_casefold_false(self):
        # When casefold is False, the case is preserved.
        self.assertEqual(clean_string("Hello World!", casefold=False), "Hello_World_")

    # Tests for clean_value with a string that is not a timestamp.
    def test_clean_value_string_non_timestamp(self):
        self.assertEqual(clean_value("Hello World!"), "hello_world_")

    # Test clean_value with a string that represents a timestamp.
    def test_clean_value_string_timestamp(self):
        ts = "2023-03-15T12:45:00Z"
        # The value should be returned as is if it parses as a timestamp.
        self.assertEqual(clean_value(ts), ts)

    # Test clean_value for dict input.
    def test_clean_value_dict(self):
        input_dict = {"Hello World!": "Test Value!"}
        expected = {"hello_world_": "test_value_"}
        self.assertEqual(clean_value(input_dict), expected)

    # Test clean_value for a pandas DataFrame.
    def test_clean_value_dataframe(self):
        df = pd.DataFrame({
            "Name!": ["Alice?", "Bob$"],
            "Age*": [25, 30]
        })
        cleaned = clean_value(df)
        # Check that the column names have been cleaned.
        self.assertIn("name_", cleaned.columns)
        self.assertIn("age_", cleaned.columns)
        # Check that cell values in string columns are cleaned.
        self.assertEqual(cleaned["name_"].iloc[0], "alice_")
        # Numeric values should remain unchanged.
        self.assertEqual(cleaned["age_"].iloc[0], 25)

    # Test clean_value with non-string/dict/DataFrame types (e.g., int, float, bool).
    def test_clean_value_numeric_and_bool(self):
        self.assertEqual(clean_value(42), 42)
        self.assertEqual(clean_value(3.14), 3.14)
        self.assertEqual(clean_value(True), True)

    # Test clean_dataframe directly.
    def test_clean_dataframe(self):
        df = pd.DataFrame({
            "Column One!": ["Hello World!", "Goodbye?"],
            "Column Two$": [1, 2]
        })
        cleaned = clean_dataframe(df)
        # Verify column names have been cleaned.
        self.assertIn("column_one_", cleaned.columns)
        self.assertIn("column_two_", cleaned.columns)
        # Verify that cell values have been cleaned.
        self.assertEqual(cleaned["column_one_"].iloc[0], "hello_world_")
        self.assertEqual(cleaned["column_two_"].iloc[0], 1)

    # Tests for clean_column_values.
    def test_clean_column_values_success(self):
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["x", "y", "z"]
        })

        def double(x):
            return x * 2

        result = clean_column_values(df.copy(), {"A": double})
        self.assertEqual(list(result["A"]), [2, 4, 6])
        # Other columns remain unchanged.
        self.assertEqual(list(result["B"]), ["x", "y", "z"])

    def test_clean_column_values_missing_column(self):
        df = pd.DataFrame({
            "A": [1, 2, 3],
        })
        with self.assertRaises(ValueError):
            clean_column_values(df.copy(), {"B": lambda x: x})

if __name__ == '__main__':
    unittest.main()
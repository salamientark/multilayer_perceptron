"""
Unit tests for the preprocessing helpers.

Covers feature selection, class handling, NaN handling, standardization and
the dataset split.
"""

import unittest
import numpy as np
import pandas as pd

from ft_mlp.preprocessing import (select_columns, get_numerical_features,
                                  get_class_list, convert_classes_to_nbr,
                                  remove_nan, replace_nan, remove_missing,
                                  classify, standardize_array,
                                  get_standardization_stats, standardize_df,
                                  split_dataset)


def sample_df():
    return pd.DataFrame({
        'a': [1.0, 2.0, 3.0, 4.0],
        'b': [10.0, 20.0, 30.0, 40.0],
        'label': ['M', 'B', 'M', 'B'],
        })


class TestSelectColumns(unittest.TestCase):

    def test_select_columns_keeps_only_requested(self):
        result = select_columns(sample_df(), ['a', 'b'])
        self.assertEqual(list(result.columns), ['a', 'b'])
        self.assertEqual(len(result), 4)

    def test_select_columns_preserves_order(self):
        result = select_columns(sample_df(), ['b', 'a'])
        self.assertEqual(list(result.columns), ['b', 'a'])


class TestNumericalFeatures(unittest.TestCase):

    def test_detects_numerical_columns(self):
        self.assertEqual(get_numerical_features(sample_df()), ['a', 'b'])

    def test_exclude_is_honoured(self):
        self.assertEqual(get_numerical_features(sample_df(), exclude=['a']),
                         ['b'])

    def test_skips_leading_nan_to_find_the_type(self):
        df = pd.DataFrame({'a': [np.nan, 2.0], 'label': ['M', 'B']})
        self.assertEqual(get_numerical_features(df), ['a'])


class TestClassList(unittest.TestCase):

    def test_returns_classes_in_order_of_first_appearance(self):
        self.assertEqual(get_class_list(sample_df(), 'label'), ['M', 'B'])

    def test_ignores_nan(self):
        df = pd.DataFrame({'label': ['M', np.nan, 'B', 'M']})
        self.assertEqual(get_class_list(df, 'label'), ['M', 'B'])

    def test_missing_column_raises_a_clear_error(self):
        with self.assertRaises(Exception) as context:
            get_class_list(sample_df(), 'nope')
        self.assertIn("not found in the dataframe", str(context.exception))

    def test_empty_column_gives_empty_list(self):
        df = pd.DataFrame({'label': []})
        self.assertEqual(get_class_list(df, 'label'), [])


class TestConvertClasses(unittest.TestCase):

    def test_converts_matching_class_to_one(self):
        result = convert_classes_to_nbr('M', pd.Series(['M', 'B', 'M']))
        self.assertEqual(result.tolist(), [1, 0, 1])

    def test_non_matching_class_gives_all_zeros(self):
        result = convert_classes_to_nbr('X', pd.Series(['M', 'B']))
        self.assertEqual(result.tolist(), [0, 0])


class TestNanHandling(unittest.TestCase):

    def test_remove_nan_filters_missing_values(self):
        result = remove_nan(np.array([1.0, np.nan, 3.0]))
        np.testing.assert_array_equal(result, [1.0, 3.0])

    def test_remove_nan_on_clean_data_is_identity(self):
        result = remove_nan(np.array([1.0, 2.0]))
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_replace_nan_fills_with_the_column_mean(self):
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        result = replace_nan(df, columns=['a'])
        self.assertAlmostEqual(result['a'][1], 2.0)

    def test_replace_nan_accepts_a_custom_function(self):
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0]})
        result = replace_nan(df, columns=['a'], func=lambda x: 99.0)
        self.assertAlmostEqual(result['a'][1], 99.0)

    def test_replace_nan_does_not_mutate_the_input(self):
        df = pd.DataFrame({'a': [1.0, np.nan]})
        replace_nan(df, columns=['a'])
        self.assertTrue(pd.isna(df['a'][1]))

    def test_remove_missing_drops_rows_with_nan(self):
        df = pd.DataFrame({'a': [1.0, np.nan, 3.0], 'b': [1.0, 2.0, 3.0]})
        self.assertEqual(len(remove_missing(df)), 2)

    def test_remove_missing_exclude_ignores_a_column(self):
        df = pd.DataFrame({'a': [1.0, np.nan], 'b': [1.0, 2.0]})
        self.assertEqual(len(remove_missing(df, exclude=['a'])), 2)


class TestClassify(unittest.TestCase):

    def test_groups_rows_by_target(self):
        result = classify(sample_df(), 'label', ['a', 'b'])
        self.assertEqual(sorted(result.keys()), ['B', 'M'])
        self.assertEqual(len(result['M']), 2)
        self.assertEqual(list(result['M'].columns), ['a', 'b'])


class TestStandardize(unittest.TestCase):

    def test_standardize_array_gives_zero_mean_unit_std(self):
        values = np.array([1.0, 2.0, 3.0, 4.0])
        result = standardize_array(values)
        self.assertAlmostEqual(float(np.mean(result)), 0.0)
        self.assertAlmostEqual(float(np.std(result)), 1.0)

    def test_standardize_array_uses_given_mean_and_std(self):
        values = np.array([2.0, 4.0])
        result = standardize_array(values, mean=2.0, std=2.0)
        np.testing.assert_allclose(result, [0.0, 1.0])

    def test_standardize_array_with_zero_std_returns_zeros(self):
        result = standardize_array(np.array([5.0, 5.0]), mean=5.0, std=0.0)
        np.testing.assert_array_equal(result, [0.0, 0.0])

    def test_get_standardization_stats_matches_numpy(self):
        df = sample_df()
        stats = get_standardization_stats(df, ['a', 'b'])
        self.assertAlmostEqual(stats['a']['mean'], float(np.mean(df['a'])))
        self.assertAlmostEqual(stats['a']['std'], float(np.std(df['a'])))

    def test_get_standardization_stats_defaults_to_numerical_columns(self):
        stats = get_standardization_stats(sample_df())
        self.assertEqual(sorted(stats.keys()), ['a', 'b'])

    def test_standardize_df_without_stats_fits_the_frame(self):
        result = standardize_df(sample_df(), ['a'])
        self.assertAlmostEqual(float(np.mean(result['a'])), 0.0)

    def test_standardize_df_applies_given_stats(self):
        stats = {'a': {'mean': 1.0, 'std': 1.0}}
        result = standardize_df(sample_df(), ['a'], stats)
        np.testing.assert_allclose(result['a'].to_numpy(),
                                   [0.0, 1.0, 2.0, 3.0])

    def test_standardize_df_infers_columns_from_stats(self):
        stats = {'a': {'mean': 0.0, 'std': 1.0}}
        result = standardize_df(sample_df(), [], stats)
        np.testing.assert_allclose(result['a'].to_numpy(), [1.0, 2.0, 3.0, 4.0])
        # 'b' was not in the stats, so it is untouched
        np.testing.assert_allclose(result['b'].to_numpy(),
                                   [10.0, 20.0, 30.0, 40.0])

    def test_standardize_df_missing_stats_for_a_feature_raises(self):
        stats = {'a': {'mean': 0.0, 'std': 1.0}}
        with self.assertRaises(Exception) as context:
            standardize_df(sample_df(), ['a', 'b'], stats)
        self.assertIn("No standardization statistics", str(context.exception))

    def test_standardize_df_constant_column_becomes_zero(self):
        df = pd.DataFrame({'a': [7.0, 7.0, 7.0]})
        result = standardize_df(df, ['a'])
        np.testing.assert_array_equal(result['a'].to_numpy(), [0.0, 0.0, 0.0])

    def test_standardize_df_does_not_mutate_the_input(self):
        df = sample_df()
        standardize_df(df, ['a'])
        np.testing.assert_array_equal(df['a'].to_numpy(),
                                      [1.0, 2.0, 3.0, 4.0])


class TestSplitDataset(unittest.TestCase):

    def test_split_ratio_and_total(self):
        df = pd.DataFrame({'a': range(100)})
        train, test = split_dataset(df, ratio=0.8, seed=1)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(test), 20)
        self.assertEqual(len(train) + len(test), 100)

    def test_split_is_reproducible_for_a_seed(self):
        df = pd.DataFrame({'a': range(50)})
        first, _ = split_dataset(df, ratio=0.8, seed=7)
        second, _ = split_dataset(df, ratio=0.8, seed=7)
        np.testing.assert_array_equal(first['a'].to_numpy(),
                                      second['a'].to_numpy())

    def test_split_differs_between_seeds(self):
        df = pd.DataFrame({'a': range(50)})
        first, _ = split_dataset(df, ratio=0.8, seed=1)
        second, _ = split_dataset(df, ratio=0.8, seed=2)
        self.assertFalse(np.array_equal(first['a'].to_numpy(),
                                        second['a'].to_numpy()))

    def test_split_partitions_without_overlap(self):
        df = pd.DataFrame({'a': range(50)})
        train, test = split_dataset(df, ratio=0.8, seed=3)
        self.assertEqual(set(train.index) & set(test.index), set())
        self.assertEqual(set(train['a']) | set(test['a']), set(range(50)))


if __name__ == '__main__':
    unittest.main(verbosity=2)

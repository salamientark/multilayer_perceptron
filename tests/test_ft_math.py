"""
Unit tests for the hand-rolled math helpers.

Each function is checked against its numpy/pandas equivalent, which is the
reference these re-implementations are meant to reproduce.
"""

import unittest
import numpy as np
import pandas as pd

from ft_mlp.ft_math import (ft_isnbr, ft_mean, ft_variance, ft_std, ft_min,
                            ft_max, ft_q1, ft_q2, ft_q3, ft_skew, ft_kurtosis,
                            ft_argmax)


SAMPLE = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])


class TestIsNumber(unittest.TestCase):

    def test_accepts_int_and_float(self):
        self.assertTrue(ft_isnbr(1))
        self.assertTrue(ft_isnbr(1.5))
        self.assertTrue(ft_isnbr(-3))
        self.assertTrue(ft_isnbr(0))

    def test_rejects_non_numbers(self):
        self.assertFalse(ft_isnbr("1"))
        self.assertFalse(ft_isnbr(None))
        self.assertFalse(ft_isnbr([1]))

    def test_rejects_nan(self):
        self.assertFalse(ft_isnbr(float('nan')))
        self.assertFalse(ft_isnbr(np.nan))


class TestMeanVarianceStd(unittest.TestCase):

    def test_mean_matches_numpy(self):
        self.assertAlmostEqual(ft_mean(SAMPLE), float(np.mean(SAMPLE)))

    def test_mean_single_element(self):
        self.assertAlmostEqual(ft_mean(np.array([42.0])), 42.0)

    def test_mean_with_explicit_count(self):
        # count limits how many elements are averaged
        self.assertAlmostEqual(ft_mean(SAMPLE, count=2), 3.0)

    def test_variance_matches_numpy_population_variance(self):
        self.assertAlmostEqual(ft_variance(SAMPLE), float(np.var(SAMPLE)))

    def test_variance_with_precomputed_mean(self):
        mean = float(np.mean(SAMPLE))
        self.assertAlmostEqual(ft_variance(SAMPLE, mean=mean),
                               float(np.var(SAMPLE)))

    def test_variance_of_constant_is_zero(self):
        self.assertAlmostEqual(ft_variance(np.array([3.0, 3.0, 3.0])), 0.0)

    def test_std_matches_numpy_population_std(self):
        self.assertAlmostEqual(ft_std(SAMPLE), float(np.std(SAMPLE)))

    def test_std_with_precomputed_variance(self):
        self.assertAlmostEqual(ft_std(SAMPLE, var=4.0), 2.0)

    def test_std_of_constant_is_zero(self):
        self.assertAlmostEqual(ft_std(np.array([7.0, 7.0])), 0.0)


class TestMinMax(unittest.TestCase):

    def test_min_of_1d_matches_numpy(self):
        self.assertEqual(ft_min(SAMPLE), float(np.min(SAMPLE)))

    def test_max_of_1d_matches_numpy(self):
        self.assertEqual(ft_max(SAMPLE), float(np.max(SAMPLE)))

    def test_min_max_with_negatives(self):
        values = np.array([-5.0, 3.0, -1.0])
        self.assertEqual(ft_min(values), -5.0)
        self.assertEqual(ft_max(values), 3.0)

    def test_max_of_2d_returns_row_maximum_as_column(self):
        """softmax relies on this shape for per-row max subtraction"""
        values = np.array([[1.0, 5.0, 3.0], [9.0, 2.0, 4.0]])
        result = ft_max(values)
        self.assertEqual(result.shape, (2, 1))
        np.testing.assert_array_equal(result, [[5.0], [9.0]])


class TestQuartiles(unittest.TestCase):

    def test_q2_is_the_median_index(self):
        self.assertEqual(ft_q2(SAMPLE), sorted(SAMPLE)[len(SAMPLE) // 2])

    def test_q1_q2_q3_are_ordered(self):
        self.assertLessEqual(ft_q1(SAMPLE), ft_q2(SAMPLE))
        self.assertLessEqual(ft_q2(SAMPLE), ft_q3(SAMPLE))

    def test_quartiles_on_known_values(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        self.assertEqual(ft_q1(values), 3.0)   # index 8//4 = 2
        self.assertEqual(ft_q2(values), 5.0)   # index 8//2 = 4
        self.assertEqual(ft_q3(values), 7.0)   # index (8//4)*3 = 6

    def test_quartiles_accept_explicit_count(self):
        self.assertEqual(ft_q2(SAMPLE, count=len(SAMPLE)), ft_q2(SAMPLE))


class TestSkewKurtosis(unittest.TestCase):

    def test_skew_matches_pandas_population_skew(self):
        series = pd.Series(SAMPLE)
        expected = float(((series - series.mean()) ** 3).mean()
                         / series.std(ddof=0) ** 3)
        self.assertAlmostEqual(ft_skew(SAMPLE), expected, places=9)

    def test_skew_of_symmetric_data_is_zero(self):
        self.assertAlmostEqual(ft_skew(np.array([-2.0, -1.0, 0.0, 1.0, 2.0])),
                               0.0, places=9)

    def test_skew_of_constant_is_zero(self):
        self.assertEqual(ft_skew(np.array([5.0, 5.0, 5.0])), 0)

    def test_kurtosis_matches_excess_kurtosis(self):
        series = pd.Series(SAMPLE)
        expected = float(((series - series.mean()) ** 4).mean()
                         / series.std(ddof=0) ** 4 - 3)
        self.assertAlmostEqual(ft_kurtosis(SAMPLE), expected, places=9)

    def test_kurtosis_of_constant_is_zero(self):
        self.assertEqual(ft_kurtosis(np.array([5.0, 5.0])), 0)


class TestArgmax(unittest.TestCase):

    def test_argmax_matches_numpy(self):
        values = np.array([[0.1, 0.9], [0.7, 0.3], [0.2, 0.8]])
        np.testing.assert_array_equal(ft_argmax(values),
                                      np.argmax(values, axis=1))

    def test_argmax_returns_first_index_on_ties(self):
        values = np.array([[0.5, 0.5]])
        self.assertEqual(ft_argmax(values)[0], 0)
        self.assertEqual(ft_argmax(values)[0], np.argmax(values, axis=1)[0])

    def test_argmax_wider_than_two_columns(self):
        values = np.array([[1.0, 3.0, 2.0], [5.0, 0.0, 1.0]])
        np.testing.assert_array_equal(ft_argmax(values),
                                      np.argmax(values, axis=1))

    def test_argmax_returns_int_array(self):
        result = ft_argmax(np.array([[0.1, 0.9]]))
        self.assertEqual(result.dtype, int)
        self.assertEqual(result.shape, (1,))


if __name__ == '__main__':
    unittest.main(verbosity=2)

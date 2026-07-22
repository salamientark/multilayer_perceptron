"""
Unit tests for the dataset analysis helpers and the analyse_data program.

The plotting calls are mocked: the tests check the statistics, not the
rendering.
"""

import argparse
import contextlib
import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from ft_mlp.analysis import (ft_describe, ft_shape, correlation_coefficient,
                             correlation_matrix, print_describe_result)
import ft_mlp.analyse_data as analyse_module


def numeric_df():
    rng = np.random.default_rng(4)
    return pd.DataFrame({
        'a': rng.normal(size=40),
        'b': rng.normal(size=40),
        'c': rng.normal(size=40),
        })


class TestShape(unittest.TestCase):

    def test_shape_matches_pandas(self):
        df = numeric_df()
        self.assertEqual(ft_shape(df), df.shape)

    def test_shape_of_empty_frame(self):
        self.assertEqual(ft_shape(pd.DataFrame()), (0, 0))


class TestCorrelationCoefficient(unittest.TestCase):

    def test_perfect_positive_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(correlation_coefficient(x, x), 1.0)

    def test_perfect_negative_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(correlation_coefficient(x, -x), -1.0)

    def test_matches_numpy_corrcoef(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=50)
        y = rng.normal(size=50)
        self.assertAlmostEqual(correlation_coefficient(x, y),
                               float(np.corrcoef(x, y)[0, 1]), places=9)

    def test_constant_input_gives_zero_not_nan(self):
        x = np.array([5.0, 5.0, 5.0])
        y = np.array([1.0, 2.0, 3.0])
        self.assertEqual(correlation_coefficient(x, y), 0)

    def test_explicit_count_is_used(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(correlation_coefficient(x, x, count=len(x)),
                               1.0)


class TestCorrelationMatrix(unittest.TestCase):

    def test_matrix_matches_numpy(self):
        df = numeric_df()
        result = correlation_matrix(df)
        expected = np.corrcoef(df.to_numpy(), rowvar=False)
        np.testing.assert_allclose(result, expected, atol=1e-9)

    def test_diagonal_is_one(self):
        result = correlation_matrix(numeric_df())
        np.testing.assert_allclose(np.diag(result), np.ones(3))

    def test_matrix_is_symmetric(self):
        result = correlation_matrix(numeric_df())
        np.testing.assert_allclose(result, result.T)

    def test_matrix_shape_matches_column_count(self):
        result = correlation_matrix(numeric_df())
        self.assertEqual(result.shape, (3, 3))


class TestDescribe(unittest.TestCase):

    def _describe_output(self, df, exclude=[]):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            ft_describe(df, exclude=exclude)
        return output.getvalue()

    def test_describe_lists_every_numerical_feature(self):
        text = self._describe_output(numeric_df())
        for feature in ('a', 'b', 'c'):
            self.assertIn(feature, text)

    def test_describe_reports_the_standard_statistics(self):
        text = self._describe_output(numeric_df())
        for label in ('Count', 'Mean', 'Std', 'Min', 'Max'):
            self.assertIn(label, text)

    def test_describe_excludes_requested_columns(self):
        text = self._describe_output(numeric_df(), exclude=['a'])
        self.assertNotIn(' a ', text)
        self.assertIn('b', text)

    def test_describe_count_matches_row_count(self):
        text = self._describe_output(pd.DataFrame({'a': [1.0, 2.0, 3.0]}))
        self.assertIn('3', text)

    def test_describe_ignores_nan_in_the_count(self):
        text = self._describe_output(pd.DataFrame({'a': [1.0, np.nan, 3.0]}))
        self.assertIn('2', text)

    def test_describe_skips_non_numerical_columns(self):
        df = pd.DataFrame({'a': [1.0, 2.0], 'label': ['M', 'B']})
        text = self._describe_output(df)
        self.assertIn('a', text)
        self.assertNotIn('label', text)

    def test_print_describe_result_renders_a_row(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            print_describe_result(['f'], [3], [1.0], [0.5], [0.0], [0.5],
                                  [1.0], [1.5], [2.0], [0.25], [0.0], [0.0])
        self.assertIn('f', output.getvalue())


class TestAnalyseDataProgram(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, 'data.csv')
        rng = np.random.default_rng(6)
        rows = 30
        features = len(analyse_module.data_columns_names) - 2
        df = pd.DataFrame(rng.normal(size=(rows, features)))
        df.insert(0, 'diagnosis', ['M', 'B'] * (rows // 2))
        df.insert(0, 'id', range(rows))
        df.to_csv(self.path, index=False, header=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse_args_requires_a_dataset(self):
        original = sys.argv
        try:
            sys.argv = ['analyse_data.py']
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    analyse_module.parse_args()
        finally:
            sys.argv = original

    def test_parse_args_accepts_a_dataset(self):
        original = sys.argv
        try:
            sys.argv = ['analyse_data.py', 'data.csv']
            args = analyse_module.parse_args()
        finally:
            sys.argv = original
        self.assertEqual(args.dataset, 'data.csv')

    def test_help_is_not_treated_as_a_filename(self):
        """--help used to reach pd.read_csv as a path"""
        original = sys.argv
        try:
            sys.argv = ['analyse_data.py', '--help']
            with contextlib.redirect_stdout(io.StringIO()) as out:
                with self.assertRaises(SystemExit) as context:
                    analyse_module.parse_args()
        finally:
            sys.argv = original
        self.assertEqual(context.exception.code, 0)
        self.assertIn('usage:', out.getvalue())

    @mock.patch('ft_mlp.analyse_data.heatmap')
    @mock.patch('ft_mlp.analyse_data.pairplot')
    def test_main_describes_the_dataset(self, pairplot, heatmap):
        args = argparse.Namespace(dataset=self.path)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            analyse_module.main(args)
        self.assertIn('Count', out.getvalue())
        self.assertEqual(pairplot.call_count, 3)
        self.assertEqual(heatmap.call_count, 1)

    def test_cli_reports_a_missing_file_and_exits_non_zero(self):
        original = sys.argv
        try:
            sys.argv = ['analyse_data.py', 'does_not_exist.csv']
            with contextlib.redirect_stdout(io.StringIO()) as out, \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                with self.assertRaises(SystemExit) as context:
                    analyse_module.cli()
        finally:
            sys.argv = original
        self.assertEqual(context.exception.code, 1)
        self.assertIn('Error', err.getvalue())
        self.assertNotIn('Error', out.getvalue())

    def test_main_propagates_instead_of_swallowing(self):
        """main() must not absorb failures; cli() owns reporting + exit."""
        args = argparse.Namespace(dataset='does_not_exist.csv')
        with self.assertRaises(FileNotFoundError):
            analyse_module.main(args)

    @mock.patch('ft_mlp.analyse_data.sns.pairplot')
    @mock.patch('ft_mlp.analyse_data.plt')
    def test_pairplot_is_invoked_with_features(self, plt, sns_pp):
        df = pd.read_csv(self.path, header=None)
        df.columns = analyse_module.data_columns_names
        analyse_module.pairplot(df, ['radius_mean', 'texture_mean'],
                                'diagnosis')
        self.assertTrue(sns_pp.called)

    @mock.patch('ft_mlp.analyse_data.sns.heatmap')
    @mock.patch('ft_mlp.analyse_data.plt')
    def test_heatmap_renders_without_error(self, plt, sns_heatmap):
        """heatmap labels its axes with the 30 features + diagnosis, so it
        expects a frame of exactly that shape."""
        rng = np.random.default_rng(9)
        columns = analyse_module.data_columns_names[2:] + ['diagnosis']
        df = pd.DataFrame(rng.normal(size=(20, len(columns))),
                          columns=columns)
        analyse_module.heatmap(df)
        self.assertTrue(sns_heatmap.called)
        self.assertTrue(plt.show.called)


if __name__ == '__main__':
    unittest.main(verbosity=2)

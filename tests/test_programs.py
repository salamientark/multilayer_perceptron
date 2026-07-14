"""
End-to-end tests for the three programs.

They run split_dataset -> train -> predict in a temporary directory, through
the same main()/cli() entry points an evaluator uses.
"""

import contextlib
import io
import os
import re
import sys
import unittest
from unittest import mock

import numpy as np
import pandas as pd

import ft_mlp.split_dataset as split_module
import ft_mlp.train as train_module
import ft_mlp.predict as predict_module


def strip_ansi(text):
    """Remove the colour escape codes the programs print"""
    return re.sub(r'\x1b\[[0-9;]*m', '', text)


def write_raw_dataset(path, rows=120, seed=5):
    """Write a headerless dataset shaped like data.csv (id, diagnosis, 30f)"""
    rng = np.random.default_rng(seed)
    half = rows // 2
    features = len(split_module.data_columns_names) - 2
    benign = rng.normal(loc=0.0, scale=1.0, size=(half, features))
    malignant = rng.normal(loc=2.5, scale=1.0, size=(rows - half, features))
    data = np.vstack([benign, malignant])
    df = pd.DataFrame(data)
    df.insert(0, 'diagnosis', ['B'] * half + ['M'] * (rows - half))
    df.insert(0, 'id', [800000 + i for i in range(rows)])
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_csv(path, index=False, header=False)
    return df


@contextlib.contextmanager
def argv(*args):
    original = sys.argv
    sys.argv = list(args)
    try:
        yield
    finally:
        sys.argv = original


class TestProgramsEndToEnd(unittest.TestCase):
    """Run the documented pipeline start to finish"""

    def setUp(self):
        self.tmp = __import__('tempfile').TemporaryDirectory()
        self.dir = self.tmp.name
        self.cwd = os.getcwd()
        self.raw = os.path.join(self.dir, 'data.csv')
        write_raw_dataset(self.raw)
        os.chdir(self.dir)          # train/predict write into the cwd

    def tearDown(self):
        os.chdir(self.cwd)
        self.tmp.cleanup()

    def _split(self):
        with argv('split_dataset.py', self.raw), \
                contextlib.redirect_stdout(io.StringIO()):
            split_module.cli()
        return (os.path.join(self.dir, 'data_training.csv'),
                os.path.join(self.dir, 'data_validation.csv'))

    def _train(self, *shape):
        train_csv = os.path.join(self.dir, 'data_training.csv')
        shape = shape or ('8', '8')
        with argv('train.py', train_csv, '--shape', *shape,
                  '-e', '15', '-a', '0.1', '-b', '16', '-s', '42'), \
                mock.patch('matplotlib.pyplot.show'), \
                mock.patch('matplotlib.pyplot.get_current_fig_manager'), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            train_module.cli()
        return out.getvalue()

    def _predict(self):
        with argv('predict.py', '-m', 'trained_model.json',
                  '-w', 'weights.npz', '-d', 'data_validation.csv'), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            predict_module.cli()
        return out.getvalue()

    def test_split_writes_both_files_and_keeps_every_row(self):
        train_csv, val_csv = self._split()
        self.assertTrue(os.path.exists(train_csv))
        self.assertTrue(os.path.exists(val_csv))
        total = len(pd.read_csv(train_csv)) + len(pd.read_csv(val_csv))
        self.assertEqual(total, 120)

    def test_train_writes_model_and_weights(self):
        self._split()
        self._train()
        self.assertTrue(os.path.exists(os.path.join(self.dir,
                                                    'trained_model.json')))
        self.assertTrue(os.path.exists(os.path.join(self.dir, 'weights.npz')))

    def test_train_prints_metrics_for_every_epoch(self):
        self._split()
        output = self._train()
        self.assertEqual(output.count('val_loss:'), 15)
        self.assertIn('Epoch 01/15', output)
        self.assertIn('Epoch 15/15', output)

    def test_full_pipeline_predicts_better_than_chance(self):
        self._split()
        self._train()
        output = self._predict()
        self.assertIn('Valid predictions', output)
        self.assertTrue(os.path.exists(os.path.join(self.dir,
                                                    'prediction.csv')))
        plain = strip_ansi(output)
        valid = int(re.search(r"Valid predictions: (\d+)", plain).group(1))
        self.assertGreater(valid, 12)      # 24 validation rows, >50%

    def test_prediction_file_has_one_row_per_sample(self):
        _, val_csv = self._split()
        self._train()
        self._predict()
        predictions = pd.read_csv(os.path.join(self.dir, 'prediction.csv'))
        self.assertEqual(len(predictions), len(pd.read_csv(val_csv)))
        self.assertEqual(list(predictions.columns), ['Index', 'Prediction'])
        self.assertTrue(set(predictions['Prediction']) <= {'B', 'M'})

    def test_train_runs_with_defaults_and_two_hidden_layers(self):
        """The subject requires a default of at least two hidden layers"""
        self._split()
        train_csv = os.path.join(self.dir, 'data_training.csv')
        with argv('train.py', train_csv, '-e', '3'), \
                mock.patch('matplotlib.pyplot.show'), \
                mock.patch('matplotlib.pyplot.get_current_fig_manager'), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            train_module.cli()
        self.assertIn('Epoch 3/3', out.getvalue())
        import json
        with open(os.path.join(self.dir, 'trained_model.json')) as f:
            saved = json.load(f)
        self.assertGreaterEqual(len(saved['layers']), 2)

    def test_train_outfile_options_are_honoured(self):
        """--outfile used to be silently ignored in favour of hardcoded names"""
        self._split()
        train_csv = os.path.join(self.dir, 'data_training.csv')
        with argv('train.py', train_csv, '-e', '3',
                  '-of', 'custom_weights.npz', '-mo', 'custom_model.json'), \
                mock.patch('matplotlib.pyplot.show'), \
                mock.patch('matplotlib.pyplot.get_current_fig_manager'), \
                contextlib.redirect_stdout(io.StringIO()):
            train_module.cli()
        self.assertTrue(os.path.exists(os.path.join(self.dir,
                                                    'custom_weights.npz')))
        self.assertTrue(os.path.exists(os.path.join(self.dir,
                                                    'custom_model.json')))

    def test_train_with_layer_and_neurons(self):
        self._split()
        self._train('6', '6', '6')
        import json
        with open(os.path.join(self.dir, 'trained_model.json')) as f:
            saved = json.load(f)
        self.assertEqual([layer['shape'] for layer in saved['layers']],
                         [6, 6, 6])


class TestProgramErrorHandling(unittest.TestCase):
    """Bad input must produce 'Error: ...' and no traceback"""

    def test_train_missing_dataset_reports_an_error(self):
        with argv('train.py', 'does_not_exist.csv'), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            train_module.cli()
        self.assertIn('Error', out.getvalue())

    def test_train_rejects_bad_learning_rate(self):
        with argv('train.py', 'x.csv', '-a', '5'), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            train_module.cli()
        self.assertIn('Learning rate must be in the range', out.getvalue())

    def test_train_rejects_layer_without_neurons(self):
        with argv('train.py', 'x.csv', '--layer', '3'), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            train_module.cli()
        self.assertIn('MUST be used together', out.getvalue())

    def test_predict_missing_model_reports_an_error(self):
        with argv('predict.py', '-m', 'nope.json', '-w', 'nope.npz',
                  '-d', 'nope.csv'), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            predict_module.cli()
        self.assertIn('Error', out.getvalue())

    def test_split_rejects_bad_ratio(self):
        with argv('split_dataset.py', 'x.csv', '-r', '1.5'), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                split_module.parse_args()

    def test_split_rejects_identical_outfiles(self):
        with argv('split_dataset.py', 'x.csv', '-o', 'a.csv', 'a.csv'), \
                contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                split_module.parse_args()

    def test_split_accepts_comma_separated_outfiles(self):
        """The dataset must come first: --outfile uses nargs='+', which
        would otherwise swallow the positional argument."""
        with argv('split_dataset.py', 'x.csv', '-o', 'a.csv,b.csv'):
            args = split_module.parse_args()
        self.assertEqual(args.outfile, ['a.csv', 'b.csv'])

    def test_split_accepts_space_separated_outfiles(self):
        with argv('split_dataset.py', 'x.csv', '-o', 'a.csv', 'b.csv'):
            args = split_module.parse_args()
        self.assertEqual(args.outfile, ['a.csv', 'b.csv'])

    def test_split_outfile_before_dataset_is_rejected_not_silent(self):
        """nargs='+' swallows the positional; argparse must complain rather
        than silently treat the dataset as an output filename."""
        with argv('split_dataset.py', '-o', 'a.csv,b.csv', 'x.csv'), \
                contextlib.redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(SystemExit):
                split_module.parse_args()
        self.assertIn('required: dataset_path', err.getvalue())

    def test_split_missing_file_reports_an_error(self):
        args = __import__('argparse').Namespace(
                dataset_path='nope.csv', outfile=['a.csv', 'b.csv'],
                seed=1, train_ratio=0.8)
        with contextlib.redirect_stdout(io.StringIO()) as out:
            split_module.main(args)
        self.assertIn('Error', out.getvalue())


class TestOneDecoded(unittest.TestCase):

    def test_one_decoded_maps_columns_to_class_names(self):
        onecoded = np.array([[0.9, 0.1], [0.2, 0.8]])
        self.assertEqual(predict_module.one_decoded(onecoded, ['B', 'M']),
                         ['B', 'M'])

    def test_one_decoded_respects_class_order(self):
        onecoded = np.array([[0.9, 0.1]])
        self.assertEqual(predict_module.one_decoded(onecoded, ['M', 'B']),
                         ['M'])


if __name__ == '__main__':
    unittest.main(verbosity=2)

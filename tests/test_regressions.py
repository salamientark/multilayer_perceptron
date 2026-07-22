"""
Regression tests for defects found during the pre-defense audit.

Each test here pins a bug that shipped and produced plausible-looking but
wrong numbers. They run the real pipeline on a small synthetic dataset.
"""

import argparse
import contextlib
import io
import json
import os
import re
import tempfile
import unittest

import numpy as np
import pandas as pd

from ft_mlp.create_model import create_model
from ft_mlp.preprocessing import one_encode, get_standardization_stats
from ft_mlp.train import init_model, train, check_model, validate_args
from ft_mlp.model_utils import save_model
from ft_mlp.create_model import load_model_from_json
from ft_mlp.load_predict_model import load_predict_model_data
import ft_mlp.split_dataset as split_dataset_module


FEATURES = ['f0', 'f1', 'f2', 'f3']
TARGET = 'diagnosis'


def make_dataset(path, rows=60, seed=3):
    """Write a small separable two-class dataset with a header"""
    rng = np.random.default_rng(seed)
    half = rows // 2
    benign = rng.normal(loc=0.0, scale=1.0, size=(half, len(FEATURES)))
    malignant = rng.normal(loc=3.0, scale=1.0, size=(rows - half,
                                                     len(FEATURES)))
    data = np.vstack([benign, malignant])
    df = pd.DataFrame(data, columns=FEATURES)
    df[TARGET] = ['B'] * half + ['M'] * (rows - half)
    # interleave so both classes appear in both splits
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_csv(path, index=False)
    return df


def make_args(dataset, epoch=5, alpha=0.1, batch=8, seed=42):
    """Build a train.py-style argparse namespace"""
    return argparse.Namespace(
            conf=None, shape=[4, 4], layer=None, neurons=None,
            features=FEATURES, loss='categoricalCrossentropy', epoch=epoch,
            alpha=alpha, batch=batch, seed=seed, train_ratio=0.8,
            outfile='weights.csv', dataset=dataset)


def run_training(dataset, epoch, seed=42, batch=8):
    """Run the real training pipeline, returning the model and stdout"""
    args = make_args(dataset, epoch=epoch, seed=seed, batch=batch)
    model = create_model(args, TARGET, FEATURES)
    check_model(model)
    init_model(model)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        train(model)
    return model, output.getvalue()


def first_epoch_loss(output):
    """Extract the loss and val_loss printed for epoch 1"""
    match = re.search(r"Epoch 0*1/\d+ - loss: ([\d.]+) - val_loss: ([\d.]+)",
                      output)
    if match is None:
        raise AssertionError(f"No epoch 1 line in output:\n{output}")
    return float(match.group(1)), float(match.group(2))


class TestPrintedLossIsEpochInvariant(unittest.TestCase):
    """B2: the printed loss was divided by the epoch count, not the sample
    count, so the same training state reported a different loss purely
    because --epoch changed."""

    def test_epoch_one_loss_does_not_depend_on_total_epochs(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = os.path.join(tmp, 'data.csv')
            make_dataset(dataset)
            _, short = run_training(dataset, epoch=2)
            _, long = run_training(dataset, epoch=8)

        short_loss, short_val = first_epoch_loss(short)
        long_loss, long_val = first_epoch_loss(long)
        self.assertAlmostEqual(short_loss, long_loss, places=10)
        self.assertAlmostEqual(short_val, long_val, places=10)

    def test_initial_loss_is_near_ln2_for_two_balanced_classes(self):
        """A fresh two-class model should start near -log(1/2)"""
        with tempfile.TemporaryDirectory() as tmp:
            dataset = os.path.join(tmp, 'data.csv')
            make_dataset(dataset)
            _, output = run_training(dataset, epoch=1)
        loss, _ = first_epoch_loss(output)
        self.assertLess(abs(loss - np.log(2)), 0.35)

    def test_loss_arrays_have_one_value_per_epoch(self):
        """Train and validation losses must use the same denominator"""
        with tempfile.TemporaryDirectory() as tmp:
            dataset = os.path.join(tmp, 'data.csv')
            make_dataset(dataset)
            model, _ = run_training(dataset, epoch=4)
        self.assertEqual(model['train_loss'].shape, (4,))
        self.assertEqual(model['test_loss'].shape, (4,))


class TestReproducibility(unittest.TestCase):
    """Gap 5: --seed was not passed to the mini-batch shuffle"""

    def test_same_seed_gives_identical_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = os.path.join(tmp, 'data.csv')
            make_dataset(dataset)
            _, first = run_training(dataset, epoch=5, seed=42, batch=8)
            _, second = run_training(dataset, epoch=5, seed=42, batch=8)
            _, other = run_training(dataset, epoch=5, seed=43, batch=8)
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)


class TestClassOrderPersistence(unittest.TestCase):
    """S1: the class order was re-derived from the prediction file, so
    output column 0 could mean B at training time and M at predict time."""

    def test_one_encode_respects_an_explicit_class_order(self):
        df = pd.DataFrame({TARGET: ['M', 'B']})
        # order of first appearance would be ['M', 'B']
        encoded = one_encode(df, TARGET, ['B', 'M'])
        np.testing.assert_array_equal(encoded, [[0, 1], [1, 0]])

    def test_one_encode_is_independent_of_row_order(self):
        first = pd.DataFrame({TARGET: ['M', 'B', 'B']})
        second = pd.DataFrame({TARGET: ['B', 'M', 'B']})
        classes = ['B', 'M']
        self.assertEqual(one_encode(first, TARGET, classes)[0].tolist(),
                         one_encode(second, TARGET, classes)[1].tolist())

    def test_one_encode_rejects_unknown_class(self):
        df = pd.DataFrame({TARGET: ['B', 'X']})
        with self.assertRaises(Exception) as context:
            one_encode(df, TARGET, ['B', 'M'])
        self.assertIn('Unknown class', str(context.exception))

    def test_predict_encodes_truth_with_the_model_class_order(self):
        """The prediction file's row order must not define the columns

        This is the actual shipped defect: predict re-derived the order from
        whatever CSV it was handed, so a file starting with an M inverted
        every prediction.
        """
        stats = {f: {'mean': 0.0, 'std': 1.0} for f in FEATURES}
        model = {'classes': ['B', 'M'], 'standardization': stats}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'predict_me.csv')
            # first row is M, so first-appearance order would be ['M', 'B']
            df = pd.DataFrame(np.zeros((2, len(FEATURES))), columns=FEATURES)
            df[TARGET] = ['M', 'B']
            df.to_csv(path, index=False)
            load_predict_model_data(model, path, FEATURES, TARGET)

        # column 0 means 'B' because the model says so
        np.testing.assert_array_equal(model['truth'], [[0, 1], [1, 0]])

    def test_predict_truth_is_independent_of_prediction_file_row_order(self):
        stats = {f: {'mean': 0.0, 'std': 1.0} for f in FEATURES}
        encodings = {}
        for name, labels in (('m_first', ['M', 'B']), ('b_first', ['B', 'M'])):
            model = {'classes': ['B', 'M'], 'standardization': stats}
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, f'{name}.csv')
                df = pd.DataFrame(np.zeros((2, len(FEATURES))),
                                  columns=FEATURES)
                df[TARGET] = labels
                df.to_csv(path, index=False)
                load_predict_model_data(model, path, FEATURES, TARGET)
            # map each label to its encoding
            encodings[name] = {label: row.tolist() for label, row
                               in zip(labels, model['truth'])}
        self.assertEqual(encodings['m_first']['M'], encodings['b_first']['M'])
        self.assertEqual(encodings['m_first']['B'], encodings['b_first']['B'])

    def test_predict_without_saved_classes_is_an_error(self):
        """Rather than silently guessing the order from the file"""
        stats = {f: {'mean': 0.0, 'std': 1.0} for f in FEATURES}
        model = {'classes': None, 'standardization': stats}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'predict_me.csv')
            df = pd.DataFrame(np.zeros((2, len(FEATURES))), columns=FEATURES)
            df[TARGET] = ['M', 'B']
            df.to_csv(path, index=False)
            with self.assertRaises(Exception) as context:
                load_predict_model_data(model, path, FEATURES, TARGET)
        self.assertIn('class list', str(context.exception))

    def test_classes_are_saved_to_the_model_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = os.path.join(tmp, 'data.csv')
            make_dataset(dataset)
            model, _ = run_training(dataset, epoch=2)
            model_path = os.path.join(tmp, 'model.json')
            with contextlib.redirect_stdout(io.StringIO()):
                save_model(model_path, model)
            loaded = load_model_from_json(model_path)
        self.assertEqual(loaded['classes'], model['classes'])
        self.assertEqual(sorted(loaded['classes']), ['B', 'M'])


class TestStandardizationStats(unittest.TestCase):
    """Gap 2: stats were fitted on the full dataframe (leak) and recomputed
    from the prediction file (drift)."""

    def test_stats_are_fitted_on_the_training_split_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = os.path.join(tmp, 'data.csv')
            make_dataset(dataset)
            model, _ = run_training(dataset, epoch=1)
            full = pd.read_csv(dataset)

        train_stats = get_standardization_stats(model['data_train'], FEATURES)
        full_stats = get_standardization_stats(full, FEATURES)
        for feature in FEATURES:
            # the model's stats must reproduce the training split
            model_mean = model['standardization'][feature]['mean']
            self.assertNotAlmostEqual(model_mean,
                                      full_stats[feature]['mean'], places=9,
                                      msg="stats look fitted on full set")
            # data_train is already standardized, so its mean is ~0
            self.assertAlmostEqual(train_stats[feature]['mean'], 0.0, places=6)

    def test_standardized_training_data_has_zero_mean_unit_std(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = os.path.join(tmp, 'data.csv')
            make_dataset(dataset)
            model, _ = run_training(dataset, epoch=1)
        data = model['input']['train_data']
        np.testing.assert_allclose(data.mean(axis=0), np.zeros(len(FEATURES)),
                                   atol=1e-9)
        np.testing.assert_allclose(data.std(axis=0), np.ones(len(FEATURES)),
                                   atol=1e-6)

    def test_predict_standardizes_with_the_saved_stats(self):
        """predict must apply the training mean/std, not refit the file

        Refitting means the same sample is scaled differently depending on
        which other rows happen to share its file.
        """
        stats = {f: {'mean': 10.0, 'std': 2.0} for f in FEATURES}
        model = {'classes': ['B', 'M'], 'standardization': stats}
        raw = np.array([[12.0, 14.0, 8.0, 10.0],
                        [20.0, 10.0, 6.0, 16.0]])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'predict_me.csv')
            df = pd.DataFrame(raw, columns=FEATURES)
            df[TARGET] = ['B', 'M']
            df.to_csv(path, index=False)
            load_predict_model_data(model, path, FEATURES, TARGET)

        np.testing.assert_allclose(model['data'], (raw - 10.0) / 2.0)

    def test_predict_scaling_is_independent_of_the_other_rows(self):
        """The same sample must scale identically in any prediction file"""
        stats = {f: {'mean': 10.0, 'std': 2.0} for f in FEATURES}
        sample = [12.0, 14.0, 8.0, 10.0]
        scaled = {}
        for name, rows in (('alone', [sample]),
                           ('with_others', [sample, [50.0, 60.0, 70.0, 80.0],
                                            [1.0, 2.0, 3.0, 4.0]])):
            model = {'classes': ['B', 'M'], 'standardization': stats}
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, f'{name}.csv')
                df = pd.DataFrame(rows, columns=FEATURES)
                df[TARGET] = ['B'] * len(rows)
                df.to_csv(path, index=False)
                load_predict_model_data(model, path, FEATURES, TARGET)
            scaled[name] = model['data'][0]
        np.testing.assert_allclose(scaled['alone'], scaled['with_others'])

    def test_predict_without_saved_stats_is_an_error(self):
        model = {'classes': ['B', 'M'], 'standardization': None}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'predict_me.csv')
            df = pd.DataFrame(np.ones((2, len(FEATURES))), columns=FEATURES)
            df[TARGET] = ['B', 'M']
            df.to_csv(path, index=False)
            with self.assertRaises(Exception) as context:
                load_predict_model_data(model, path, FEATURES, TARGET)
        self.assertIn('standardization', str(context.exception))

    def test_stats_are_saved_to_the_model_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = os.path.join(tmp, 'data.csv')
            make_dataset(dataset)
            model, _ = run_training(dataset, epoch=2)
            model_path = os.path.join(tmp, 'model.json')
            with contextlib.redirect_stdout(io.StringIO()):
                save_model(model_path, model)
            loaded = load_model_from_json(model_path)
        self.assertEqual(sorted(loaded['standardization'].keys()),
                         sorted(FEATURES))
        for feature in FEATURES:
            self.assertAlmostEqual(loaded['standardization'][feature]['mean'],
                                   model['standardization'][feature]['mean'])


class TestJsonConfigPath(unittest.TestCase):
    """create_model passed the --conf FILENAME straight to json.load(),
    which needs an open file, so --conf never worked at all."""

    def _write_conf(self, tmp, conf):
        path = os.path.join(tmp, 'conf.json')
        with open(path, 'w') as f:
            json.dump(conf, f)
        return path

    def _base_conf(self):
        return {
            'model': 'multilayer perceptron',
            'epoch': 3, 'alpha': 0.1, 'batch': 8, 'seed': 42,
            'loss': 'categoricalCrossentropy',
            'features': FEATURES, 'target': TARGET,
            'layers': [
                {'shape': 4, 'activation': 'sigmoid',
                 'weights_initializer': 'heUniform'},
                {'shape': 4, 'activation': 'sigmoid',
                 'weights_initializer': 'heUniform'},
                ],
            'output': {'shape': 2, 'activation': 'softmax',
                       'weights_initializer': 'heUniform'},
            }

    def _args(self, conf_path, dataset):
        return argparse.Namespace(
                conf=conf_path, shape=None, layer=None, neurons=None,
                features=None, loss=None, epoch=None, alpha=None, batch=None,
                seed=None, train_ratio=0.8, outfile='weights.npz',
                dataset=dataset)

    def test_conf_file_is_loaded_from_its_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = os.path.join(tmp, 'data.csv')
            make_dataset(dataset)
            conf = self._write_conf(tmp, self._base_conf())
            model = create_model(self._args(conf, dataset), TARGET)
        self.assertEqual(model['epoch'], 3)
        self.assertEqual(model['seed'], 42)
        self.assertEqual([layer['shape'] for layer in model['layers']], [4, 4])

    def test_conf_values_are_not_overridden_by_cli_defaults(self):
        """validate_args must not fill defaults when --conf is used"""
        with tempfile.TemporaryDirectory() as tmp:
            dataset = os.path.join(tmp, 'data.csv')
            make_dataset(dataset)
            spec = self._base_conf()
            spec['epoch'] = 7
            spec['alpha'] = 0.42
            conf = self._write_conf(tmp, spec)
            args = self._args(conf, dataset)
            validate_args(args)
            model = create_model(args, TARGET)
        self.assertEqual(model['epoch'], 7)
        self.assertAlmostEqual(model['alpha'], 0.42)

    def test_conf_trains_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = os.path.join(tmp, 'data.csv')
            make_dataset(dataset)
            conf = self._write_conf(tmp, self._base_conf())
            args = self._args(conf, dataset)
            model = create_model(args, TARGET)
            check_model(model)
            init_model(model)
            with contextlib.redirect_stdout(io.StringIO()) as out:
                train(model)
        self.assertIn('Epoch 3/3', out.getvalue())

    def test_unknown_function_name_gives_a_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = os.path.join(tmp, 'data.csv')
            make_dataset(dataset)
            spec = self._base_conf()
            spec['loss'] = 'mse'
            conf = self._write_conf(tmp, spec)
            with self.assertRaises(Exception) as context:
                create_model(self._args(conf, dataset), TARGET)
        self.assertIn("Unknown function 'mse'", str(context.exception))

    def test_activation_without_derivative_gives_a_clear_error(self):
        """DERIVATIVE_MAP has only sigmoid: this used to be a raw KeyError"""
        with tempfile.TemporaryDirectory() as tmp:
            dataset = os.path.join(tmp, 'data.csv')
            make_dataset(dataset)
            spec = self._base_conf()
            spec['layers'][0]['activation'] = 'softmax'
            conf = self._write_conf(tmp, spec)
            with self.assertRaises(Exception) as context:
                create_model(self._args(conf, dataset), TARGET)
        self.assertIn('No derivative is implemented', str(context.exception))


class TestSplitterKeepsEveryRow(unittest.TestCase):
    """S2: pandas consumed the first sample of the headerless data.csv as a
    header row, destroying it."""

    def _write_headerless(self, path, rows=50):
        rng = np.random.default_rng(1)
        columns = split_dataset_module.data_columns_names
        data = rng.normal(size=(rows, len(columns) - 2))
        df = pd.DataFrame(data)
        df.insert(0, 'diagnosis', ['M', 'B'] * (rows // 2))
        df.insert(0, 'id', [842302 + i for i in range(rows)])
        df.to_csv(path, index=False, header=False)
        return df

    def test_no_sample_is_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = os.path.join(tmp, 'data.csv')
            train_path = os.path.join(tmp, 'train.csv')
            test_path = os.path.join(tmp, 'test.csv')
            self._write_headerless(dataset, rows=50)
            args = argparse.Namespace(dataset_path=dataset,
                                      outfile=[train_path, test_path],
                                      seed=1, train_ratio=0.8)
            with contextlib.redirect_stdout(io.StringIO()):
                split_dataset_module.main(args)
            train_rows = len(pd.read_csv(train_path))
            test_rows = len(pd.read_csv(test_path))
        self.assertEqual(train_rows + test_rows, 50)

    def test_first_sample_survives_the_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset = os.path.join(tmp, 'data.csv')
            train_path = os.path.join(tmp, 'train.csv')
            test_path = os.path.join(tmp, 'test.csv')
            self._write_headerless(dataset, rows=50)
            args = argparse.Namespace(dataset_path=dataset,
                                      outfile=[train_path, test_path],
                                      seed=1, train_ratio=0.8)
            with contextlib.redirect_stdout(io.StringIO()):
                split_dataset_module.main(args)
            ids = (pd.read_csv(train_path)['id'].tolist()
                   + pd.read_csv(test_path)['id'].tolist())
        self.assertIn(842302, ids)


class TestSplitterDefaults(unittest.TestCase):
    """S3: the --outfile default was a string, and argparse does not apply
    nargs to a default, so the validator counted its characters."""

    def test_outfile_default_is_a_pair_of_filenames(self):
        import sys
        original = sys.argv
        try:
            sys.argv = ['split_dataset.py', 'data.csv']
            args = split_dataset_module.parse_args()
        finally:
            sys.argv = original
        self.assertEqual(args.outfile,
                         ['data_training.csv', 'data_validation.csv'])


class TestHeaderlessDatasetReading(unittest.TestCase):
    """S4: the raw data.csv is headerless, but train/predict read their csv
    with an inferred header, so feeding the raw file straight in died with a
    twelve-line pandas KeyError instead of just working."""

    def test_headerless_file_gets_the_schema_column_names(self):
        from ft_mlp.dataset_io import read_dataset
        from ft_mlp.dataset_schema import DATA_COLUMNS_NAMES
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'raw.csv')
            rows = [[i, 'M'] + [0.5] * 30 for i in range(3)]
            pd.DataFrame(rows).to_csv(path, header=False, index=False)
            df = read_dataset(path)
        self.assertEqual(list(df.columns), DATA_COLUMNS_NAMES)
        # No sample is consumed as a header row.
        self.assertEqual(len(df), 3)

    def test_file_with_header_is_read_as_is(self):
        from ft_mlp.dataset_io import read_dataset
        from ft_mlp.dataset_schema import DATA_COLUMNS_NAMES
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'split.csv')
            rows = [[i, 'B'] + [0.5] * 30 for i in range(3)]
            pd.DataFrame(rows, columns=DATA_COLUMNS_NAMES).to_csv(
                path, index=False)
            df = read_dataset(path)
        self.assertEqual(list(df.columns), DATA_COLUMNS_NAMES)
        self.assertEqual(len(df), 3)

    def test_wrong_column_count_raises_a_readable_error(self):
        from ft_mlp.dataset_io import read_dataset
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'bad.csv')
            pd.DataFrame([[1, 2, 3]]).to_csv(path, header=False, index=False)
            with self.assertRaises(Exception) as ctx:
                read_dataset(path)
        self.assertIn('expected 32', str(ctx.exception))


class TestExplicitValidationSet(unittest.TestCase):
    """S5: ft-split-dataset wrote a validation file that ft-train never read,
    because train re-split its input internally. --validation consumes it."""

    def _args(self, **overrides):
        base = dict(conf=None, shape=[2], layer=None, neurons=None,
                    epoch=1, alpha=0.1, batch=2, seed=1, train_ratio=None,
                    validation=None, loss=None, features=None)
        base.update(overrides)
        return argparse.Namespace(**base)

    def test_validation_and_train_ratio_are_mutually_exclusive(self):
        args = self._args(validation='valid.csv', train_ratio=0.8)
        with self.assertRaises(Exception) as ctx:
            validate_args(args)
        self.assertIn('mutually exclusive', str(ctx.exception))

    def test_train_ratio_still_defaults_when_no_validation_given(self):
        args = self._args()
        validate_args(args)
        self.assertIsNotNone(args.train_ratio)
        self.assertTrue(0 < args.train_ratio < 1)

    def test_explicit_validation_set_is_used_whole(self):
        from ft_mlp.create_model import (fill_model_datasets,
                                         init_model_template)
        with tempfile.TemporaryDirectory() as tmp:
            train_path = os.path.join(tmp, 'train.csv')
            valid_path = os.path.join(tmp, 'valid.csv')
            columns = FEATURES + [TARGET]
            rng = np.random.default_rng(0)
            for path, count in ((train_path, 20), (valid_path, 7)):
                frame = pd.DataFrame(rng.normal(size=(count, 4)),
                                     columns=FEATURES)
                frame[TARGET] = ['M', 'B'] * (count // 2) + ['M'] * (count % 2)
                frame[columns].to_csv(path, index=False)
            model = fill_model_datasets(init_model_template(),
                                        train_path, 0.8, 1,
                                        TARGET, FEATURES,
                                        validation=valid_path)
        # No internal re-split: every row of each file is used as given.
        self.assertEqual(len(model['input']['train_data']), 20)
        self.assertEqual(len(model['input']['test_data']), 7)


if __name__ == '__main__':
    unittest.main(verbosity=2)

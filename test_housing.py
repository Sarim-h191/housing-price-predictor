import unittest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from housing import FEATURES, run


class HousingTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(7)
        train = pd.DataFrame(rng.normal(size=(60, 5)), columns=FEATURES)
        train['SalePrice'] = 200000 + train[FEATURES[0]] * 10000
        fit_ids, valid_ids = train_test_split(train.index, test_size=0.2, random_state=42)
        train.loc[valid_ids, FEATURES[1]] = 1000000
        train.loc[fit_ids[:5], FEATURES[2]] = np.nan
        test = train.iloc[:8][FEATURES].copy()
        test['Id'] = range(100, 108)
        test.loc[test.index[0], FEATURES[1]] = np.nan
        return train, test, fit_ids

    def test_split_medians_refit_and_prediction_consistency(self):
        train, test, fit_ids = self.fixture()
        metrics, submission, final, candidates, *_ = run(train, test)
        for pipeline in candidates.values():
            np.testing.assert_allclose(pipeline[0].statistics_, train.loc[fit_ids, FEATURES].median().values)
        np.testing.assert_allclose(final[0].statistics_, train[FEATURES].median().values)
        np.testing.assert_allclose(submission.SalePrice, final.predict(test[FEATURES]))
        self.assertEqual(metrics['final_training_rows'], 60)
        self.assertEqual(metrics['fit_rows'], 48)
        self.assertEqual(metrics['selected_model'], min(metrics['validation_rmse'], key=metrics['validation_rmse'].get))
        self.assertEqual(list(submission.Id), list(test.Id))
        self.assertTrue(np.isfinite(submission.SalePrice).all())

    def test_reproducible(self):
        train, test, _ = self.fixture()
        first = run(train, test)
        second = run(train, test)
        self.assertEqual(first[0], second[0])
        pd.testing.assert_frame_equal(first[1], second[1])

    def test_invalid_data(self):
        train, test, _ = self.fixture()
        with self.assertRaises(ValueError):
            run(train.drop(columns=['SalePrice']), test)
        train.loc[0, 'SalePrice'] = np.nan
        with self.assertRaises(ValueError):
            run(train, test)


if __name__ == '__main__':
    unittest.main()

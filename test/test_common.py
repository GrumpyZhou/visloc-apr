import unittest
import numpy as np

from utils.common.setup import load_weights_to_gpu, config_to_string, cal_quat_angle_error, make_deterministic, lprint


class SetupTestCase(unittest.TestCase):
    def test_types(self):
        self.assertIsNone(load_weights_to_gpu(),
                          'load_weights_to_gpu from utils.common.setup is not None without args')

        self.assertIsNone(lprint(''),
                          'lprint from utils.common.setup return not None')

        self.assertIsInstance(cal_quat_angle_error(np.array([[0, 1], [2, 3], [4, 5]]),
                                                   np.array([[0, 1], [2, 3], [4, 5]])), np.ndarray,
                              'cal_quat_angle_error return not np.ndarray')
        # print(cal_quat_angle_error(np.array([[0, 1], [2, 3], [4, 5]]), np.array([[0, 1], [2, 3], [4, 5]])))


    def test_result(self):
        error_result = cal_quat_angle_error(np.array([[0, 1], [2, 3], [4, 5]]), np.array([[0, 1], [2, 3], [4, 5]]))
        pred_result = np.array([[0.], [0.], [0.]])

        np.testing.assert_almost_equal(error_result, pred_result)

        error_result = cal_quat_angle_error(np.array([[0.1, 0.2], [0.2, 0.3], [0.4, 0.5]]),
                                            np.array([[0.2, 0.3], [1000, 1000], [1000, 1000]]))
        pred_result = np.array([[14.2500327], [22.61986495], [12.68038349]])

        np.testing.assert_almost_equal(error_result, pred_result)

    def test_exceptions(self):
        with self.assertRaises(ValueError):
            cal_quat_angle_error(np.array([[0, 1], [2, 3]]), np.array([[0, 1], [2, 3], [4, 5]]))

        with self.assertRaises(ValueError):
            make_deterministic(-1)

        with self.assertRaises(TypeError):
            make_deterministic('Hello')

        with self.assertRaises(TypeError):
            make_deterministic(4.2)

        with self.assertRaises(AttributeError):
            load_weights_to_gpu(5)

        with (self.assertRaises(TypeError)):
            config_to_string(42)

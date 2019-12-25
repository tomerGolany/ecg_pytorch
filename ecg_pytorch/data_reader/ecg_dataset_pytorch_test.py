import logging
logging.basicConfig(level=logging.INFO)
import unittest
from ecg_pytorch.data_reader import ecg_dataset_pytorch, dataset_configs, heartbeat_types


class TestEcgHearBeatsDatasetPytorch(unittest.TestCase):
    def test_one_vs_all(self):
        configs = dataset_configs.DatasetConfigs('train', 'N', one_vs_all=True, lstm_setting=False,
                                                 over_sample_minority_class=False,
                                                 under_sample_majority_class=False, only_take_heartbeat_of_type=None)
        ds = ecg_dataset_pytorch.EcgHearBeatsDatasetPytorch(configs, transform=ecg_dataset_pytorch.ToTensor)

        for sample in ds:
            heartbeat = sample['cardiac_cycle']
            label_str = sample['beat_type']
            label_one_hot = sample['label']
            self.assertEqual(len(label_one_hot), 2)
            self.assertIn(label_one_hot[0], [0, 1])
            self.assertIn(label_one_hot[1], [0, 1])

            self.assertIn(label_str, heartbeat_types.AAMIHeartBeatTypes.__members__)
            if label_str == heartbeat_types.AAMIHeartBeatTypes.N.name:
                self.assertEqual(label_one_hot, [1, 0])
            else:
                self.assertEqual(label_one_hot, [0, 1])


if __name__ == '__main__':

    unittest.main()

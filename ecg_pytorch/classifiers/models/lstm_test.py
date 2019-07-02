from ecg_pytorch.classifiers.models import lstm
from ecg_pytorch.data_reader.ecg_dataset_lstm import ToTensor, EcgHearBeatsDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

def test_lstm():
    composed = transforms.Compose([ToTensor()])
    transformed_dataset = EcgHearBeatsDataset(transform=composed)
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)

    lstmN = lstm.ECGLSTM(5, 512,  5, 2)

    for i, data in enumerate(dataloader):
        ecg_batch = data['cardiac_cycle'].permute(1, 0, 2).float()
        first_beat = ecg_batch[:, 0, :]
        print("First beat shape: {}".format(first_beat.shape))
        print("First beat label {}".format(data['beat_type'][0]))
        print("First beat label one hot {}".format(data['label'][0]))
        first_beat = first_beat.numpy().flatten()
        plt.plot(first_beat)
        plt.show()
        plt.figure()
        plt.plot(data['orig_beat'][0].numpy())
        plt.show()

        preds = lstmN(ecg_batch)
        print("Module output shape = {}".format(preds.shape))
        print("Preds: {}".format(preds))
        break


if __name__ == "__main__":
    test_lstm()
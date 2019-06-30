from ecg_pytorch.classifiers.models import lstm
from ecg_pytorch.data_reader.ecg_dataset_lstm import ToTensor, EcgHearBeatsDataset
from torchvision import transforms
from torch.utils.data import DataLoader


def test_lstm():
    composed = transforms.Compose([ToTensor()])
    transformed_dataset = EcgHearBeatsDataset(transform=composed)
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)

    lstmN = lstm.ECGLSTM(5, 512,  5, 2)

    for i, data in enumerate(dataloader):
        ecg_batch = data['cardiac_cycle'].view(43, -1, 5).float()
        preds = lstmN(ecg_batch)
        print("Module output shape = {}".format(preds.shape))


if __name__ == "__main__":
    test_lstm()
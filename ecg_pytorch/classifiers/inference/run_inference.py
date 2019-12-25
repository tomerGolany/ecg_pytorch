import torch
import cv2
from matplotlib import pyplot as plt
from ecg_pytorch.data_reader import patient
from ecg_pytorch.classifiers.models import fully_connected
from ecg_pytorch.classifiers.models import cnn
import logging
import numpy as np
import tqdm
from ecg_pytorch.classifiers.inference import checkpoint_paths
from ecg_pytorch.classifiers.models import deep_residual_conv
import pandas as pd

CLASS_TO_IND = {'N': 0, 'S': 1, 'V': 2, 'F': 3, 'Q': 4}
IND_TO_CLASS = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}

N_COLOR = (255, 128, 0)
S_COLOR = (51, 255, 510)
V_COLOR = (0, 0, 255)
F_COLOR = (255, 51, 255)
Q_COLOR = (255, 0, 0)
CLASS_TO_COLOR = {'N': N_COLOR, 'S': S_COLOR, 'V': V_COLOR, 'F': F_COLOR, 'Q': Q_COLOR}

train_set = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 201, 203, 205, 207, 208, 209, 215, 220,
                 223, 230]  # DS1
train_set = [str(x) for x in train_set]
test_set = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212, 213, 214, 219, 221, 222, 228, 231, 232,
            233, 234]  # DS2
test_set = [str(x) for x in test_set]


def eval_model(model, model_chk, patient_number):

    p_data = patient.beats_from_patient(patient_number)

    checkpoint = torch.load(model_chk, map_location='cpu')
    model.load_state_dict(checkpoint['net'])
    model.eval()
    fourcc = cv2.VideoWriter_fourcc('P', 'N', 'G', ' ')
    out = cv2.VideoWriter('test/patient_{}_inference_fc.avi'.format(patient_number), fourcc,  4, (640, 480))

    softmax = torch.nn.Softmax()
    with torch.no_grad():

        for i, beat_dict in enumerate(tqdm.tqdm(p_data)):
            beat = torch.Tensor(beat_dict['cardiac_cycle'])

            prediction = model(beat)
            prediction = softmax(prediction)
            pred_class_ind = torch.argmax(prediction).item()
            pred_class = IND_TO_CLASS[pred_class_ind]
            prediction = prediction.numpy()

            true_class = beat_dict['beat_type']
            logging.info("predicted class: {}\t\t\t gt class: {}".format(pred_class, true_class))
            logging.info("Scores:\t N: {}\t\t\t S: {}\t\t\t V: {}\t\t\t F: {}\t\t\t Q: {}".format(prediction[0], prediction[1],
                                                                                  prediction[2], prediction[3],
                                                                                  prediction[4]))

            plt.figure()
            plt.plot(beat_dict['cardiac_cycle'])
            plt.xlabel('Sample # (360 HZ)')
            plt.ylabel('Voltage')
            plt.savefig('temp.png')
            img = cv2.imread('temp.png', cv2.IMREAD_UNCHANGED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            pt_print = "Patient: {}".format(patient_number)
            true_class_print = "Ground truth beat: {}".format(true_class)
            prdicted_beat_print = "Prediction: {}".format(pred_class)
            scores_print = "Scores: N: {:.2f}  S: {:.2f}    V: {:.2f}   F: {:.2f}   Q: {:.2f}".format(prediction[0], prediction[1],
                                                                                  prediction[2], prediction[3],
                                                                                  prediction[4])
            beat_num_print = "Beat #{}".format(i)

            truce_class_color = CLASS_TO_COLOR[true_class]
            pred_class_color = CLASS_TO_COLOR[pred_class]

            cv2.putText(img, pt_print, (300, 30), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, true_class_print, (10, 30), font, 0.5, truce_class_color, 1, cv2.LINE_AA)
            cv2.putText(img, prdicted_beat_print, (500, 30), font, 0.5, pred_class_color, 1, cv2.LINE_AA)
            cv2.putText(img, scores_print, (84, 70), font, 0.32, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(img, beat_num_print, (10, 450), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            out.write(img)
            plt.close()
            plt.clf()
        out.release()


class ECGInferenceOneVsAll(object):
    def __init__(self, beat_type, model, model_chk, patient_number):
        """Initialize a new ECGInferenceOneVsAll object.

        :param beat_type: The the of beat which is classified.
        :param model: The network object without the loaded weights.
        :param model_chk: path to checkpoint file with weight values.
        :param patient_number: string - number of patient from the mit-bih dataset.
        """
        self.beat_type = beat_type
        self.model = model
        self.model_chk = model_chk
        self.patient_number = patient_number

    def predict(self):
        """Returns probability predictions from an ECG signal of a patient."""
        logging.info("Inference model {} Vs. all".format(self.beat_type))
        p = patient.Patient(self.patient_number)

        heartbeats = p.heartbeats

        checkpoint = torch.load(self.model_chk, map_location='cpu')
        self.model.load_state_dict(checkpoint['net'])
        self.model.eval()
        softmax = torch.nn.Softmax()
        predictions = []
        ground_truths_one_hot = []
        ground_truths_class_str = []
        classes_ind = []
        predicted_classes_str = []
        cardiac_cycles = []
        with torch.no_grad():
            for i, beat_dict in enumerate(tqdm.tqdm(heartbeats)):
                # logging.info("beat # {}".format(beat_dict['beat_ind']))
                beat = torch.Tensor(beat_dict['cardiac_cycle'])
                cardiac_cycles.append(beat.numpy())
                model_output = self.model(beat)
                output_probability = softmax(model_output)
                predicted_class_ind = torch.argmax(output_probability).item()
                if predicted_class_ind == 1:
                    predicted_class_str = 'Other'
                else:
                    predicted_class_str = self.beat_type

                output_probability = output_probability.numpy()
                # logging.info("prediction: {}\t ground truth: {}".format(prediction, beat_dict['aami_label_str']))
                predictions.append([output_probability[0][0], output_probability[0][1]])
                classes_ind.append(predicted_class_ind)
                predicted_classes_str.append(predicted_class_str)

                if beat_dict['aami_label_str'] == self.beat_type:
                    ground_truths_one_hot.append([1, 0])
                    ground_truths_class_str.append("{}".format(self.beat_type))
                else:
                    ground_truths_one_hot.append([0, 1])
                    ground_truths_class_str.append("Other")

        return ground_truths_one_hot, predictions, classes_ind, predicted_classes_str, cardiac_cycles, \
               ground_truths_class_str

    def inference_summary_df(self):
        ground_truths, predictions, classes_ind, predicted_classes_str, cardiac_cycles, ground_truths_class_str = \
            self.predict()
        df = pd.DataFrame(list(zip(ground_truths, predictions, classes_ind, predicted_classes_str,
                                   ground_truths_class_str)),
                          columns=['Ground Truth', 'Predictions', 'predicted class index', 'predicted class str',
                                   'ground truth class str'])
        return df


def inference_one_vs_all(beat_type, model, model_chk, patient_number):
    """Returns probability predictions from an ECG signal of a patient.

    :param beat_type: The the of beat which is classified.
    :param model: The network object without the loaded weights.
    :param model_chk: path to checkpoint file with weight values.
    :param patient_number: string - number of patient from the mit-bih dataset.
    :return:
    """
    logging.info("Inference model {} Vs. all".format(beat_type))
    p = patient.Patient(patient_number)

    heartbeats = p.heartbeats

    checkpoint = torch.load(model_chk, map_location='cpu')
    model.load_state_dict(checkpoint['net'])
    model.eval()
    softmax = torch.nn.Softmax()
    predictions = []
    ground_truths = []
    with torch.no_grad():
        for i, beat_dict in enumerate(tqdm.tqdm(heartbeats)):
            # logging.info("beat # {}".format(beat_dict['beat_ind']))
            beat = torch.Tensor(beat_dict['cardiac_cycle'])

            prediction = model(beat)
            prediction = softmax(prediction)
            prediction = prediction.numpy()
            # logging.info("prediction: {}\t ground truth: {}".format(prediction, beat_dict['aami_label_str']))
            predictions.append([prediction[0][0], prediction[0][1]])

            if beat_dict['aami_label_str'] == beat_type:
                ground_truths.append([1, 0])
            else:
                ground_truths.append([0, 1])

    return predictions, ground_truths


def predictions_ground_truths_data_frame(beat_type, model, model_chk, patient_number):
    """Returns pandas dataframe of probability predictions from an ECG signal of a patient.

    :param beat_type: The the of beat which is classified.
    :param model: The network object without the loaded weights.
    :param model_chk: path to checkpoint file with weight values.
    :param patient_number: string - number of patient from the mit-bih dataset.
    :return:
    """
    predictions, ground_truths = inference_one_vs_all(beat_type, model, model_chk, patient_number)
    df = pd.DataFrame(list(zip(predictions, ground_truths)), columns=['Predictions', 'Ground Truth'])
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # ff = fully_connected.FF()
    # model_chk = checkpoint_paths.FF_CHECKPOINT_PATH
    # eval_model(ff, model_chk, patient_number='117')
    model_chk = '/Users/tomer.golany/PycharmProjects/ecg_pytorch/ecg_pytorch/classifiers/tensorboard/s_resnet_raw_v2/vgan_1000/checkpoint_epoch_iters_2685'
    net = deep_residual_conv.Net(2)
    p = '100'
    # preds, gts = inference_one_vs_all('S', net, model_chk, p)
    # preds = np.array(preds)
    # gts = np.array(gts)
    # print(preds.shape)
    # print(gts.shape)

    # pred_gts_df = predictions_ground_truths_data_frame('S', net, model_chk, p)
    # print(pred_gts_df)
    ecg_inference = ECGInferenceOneVsAll('S', net, model_chk, p)

    print(ecg_inference.inference_summary_df())
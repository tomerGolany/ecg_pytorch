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


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    ff = fully_connected.FF()
    model_chk = checkpoint_paths.FF_CHECKPOINT_PATH
    eval_model(ff, model_chk, patient_number='117')
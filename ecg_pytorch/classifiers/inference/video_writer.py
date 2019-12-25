"""Module that writes ecg signal videos for visualization."""
import cv2
import matplotlib.pyplot as plt
from ecg_pytorch.classifiers.inference import run_inference
from ecg_pytorch.classifiers.models import deep_residual_conv
import tqdm


N_COLOR = (255, 128, 0)
S_COLOR = (51, 255, 510)
V_COLOR = (0, 0, 255)
F_COLOR = (255, 51, 255)
Q_COLOR = (255, 0, 0)
CLASS_TO_COLOR = {'N': N_COLOR, 'S': S_COLOR, 'V': V_COLOR, 'F': F_COLOR, 'Q': Q_COLOR, 'Other': Q_COLOR}


class ECGVideo(object):
    def __init__(self, beat_type, model, model_chk, patient_number):
        self.fourcc = cv2.VideoWriter_fourcc('P', 'N', 'G', ' ')
        self.ecg_inference_obj = run_inference.ECGInferenceOneVsAll(beat_type, model, model_chk, patient_number)
        self.patient_number = patient_number
        self.beat_type = beat_type

        # out = cv2.VideoWriter('test/patient_{}_inference_fc.avi'.format(patient_number), fourcc, 4, (640, 480))

    def write(self):
        out = cv2.VideoWriter('test/patient_{}_inference.avi'.format(self.patient_number), self.fourcc, 4,
                              (640, 480))
        ground_truths, predictions, classes_ind, predicted_classes_str, cardiac_cycles, gts_class_str = \
            self.ecg_inference_obj.predict()
        i = 0
        for heartbeat, gt_one_hot, pred_one_hot, pred_class_str, true_class in tqdm.tqdm(zip(cardiac_cycles,
                                                                                                ground_truths,
                                                                                                predictions,
                                                                                                predicted_classes_str,
                                                                                                gts_class_str)):
            i += 1
            plt.figure()
            plt.plot(heartbeat)
            plt.xlabel('Sample # (360 HZ)')
            plt.ylabel('Voltage')
            plt.savefig('temp.png')
            img = cv2.imread('temp.png', cv2.IMREAD_UNCHANGED)
            font = cv2.FONT_HERSHEY_SIMPLEX
            pt_print = "Patient: {}".format(self.patient_number)
            true_class_print = "Ground truth beat: {}".format(true_class)
            prdicted_beat_print = "Prediction: {}".format(pred_class_str)
            scores_print = "Scores: {}: {:.2f}  Others: {:.2f}".format(self.beat_type, pred_one_hot[0], pred_one_hot[1])
            beat_num_print = "Beat #{}".format(i)

            truce_class_color = CLASS_TO_COLOR[true_class]
            pred_class_color = CLASS_TO_COLOR[pred_class_str]

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
    model_chk = '/Users/tomer.golany/PycharmProjects/ecg_pytorch/ecg_pytorch/classifiers/tensorboard/s_resnet_raw_v2/vgan_1000/checkpoint_epoch_iters_2685'
    net = deep_residual_conv.Net(2)
    p = '118'

    ecg_video = ECGVideo('S', net, model_chk, p)
    ecg_video.write()



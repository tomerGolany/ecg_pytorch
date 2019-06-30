import torch.nn as nn


class ECGLSTM(nn.Module):
    def __init__(self, length_of_each_word, number_of_hidden_neurons,  num_of_classes, num_of_layers):
        super(ECGLSTM, self).__init__()

        self.len_of_word = length_of_each_word
        self.num_hidden_neurons = number_of_hidden_neurons
        self.output_size = num_of_classes
        self.num_of_layesr = num_of_layers
        self.lstm = nn.LSTM(length_of_each_word, number_of_hidden_neurons, num_of_layers)
        self.output_layer = nn.Linear(number_of_hidden_neurons, num_of_classes)

    def forward(self, sentence):
        # sentence shape should be [len_of_sentence, batch_size, len_of_each_word]
        # lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
        # "out" will give you access to all hidden states in the sequence
        # "hidden" will allow you to continue the sequence and backpropagate,
        # by passing it as an argument  to the lstm at a later time
        # Add the extra 2nd dimension
        # print("input shape: {}".format(sentence.shape))
        out, hidden = self.lstm(sentence)  # If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        last_output = out[-1]
        # print("out shape: {}".format(out.size()))
        # out dim should be - [len_of_sentence, batch_size, hidden_size]
        # reshaped_last_output = out[-1].view(-1, self.num_hidden_neurons)
        # print("output from last unit: {}".format(reshaped_last_output.shape))
        y_pred = self.output_layer(last_output)
        # y_pred should be shape [batch_size, num_of_classes]
        # print("y pred shape: {}".format(y_pred.shape))
        return y_pred

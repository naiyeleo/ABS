# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import sys
sys.path.append('./trojai/')

import torch

import trojai.modelgen.architecture_factory

ALL_ARCHITECTURE_KEYS = ['LstmLinear', 'GruLinear', 'Linear']


class LinearModel(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float):
        super().__init__()

        self.linear = torch.nn.Linear(input_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        # get rid of implicit sequence length
        # for GRU and LSTM input needs to be [batch size, sequence length, embedding length]
        # sequence length is 1
        # however the linear model need the input to be [batch size, embedding length]
        data = data[:, 0, :]
        # input data is after the embedding
        hidden = self.dropout(data)

        # hidden = [batch size, hid dim]
        output = self.linear(hidden)
        # output = [batch size, out dim]

        return output


class GruLinearModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float, bidirectional: bool, n_layers: int):
        super().__init__()

        self.rnn = torch.nn.GRU(input_size,
                          hidden_size,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.linear = torch.nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        # input data is after the embedding

        # data = [batch size, sent len, emb dim]
        self.rnn.flatten_parameters()
        _, hidden = self.rnn(data)

        # hidden = [n layers * n directions, batch size, emb dim]
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]
        output = self.linear(hidden)
        # output = [batch size, out dim]

        return output


class LstmLinearModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float, bidirectional: bool, n_layers: int):
        super().__init__()

        self.rnn = torch.nn.LSTM(input_size,
                          hidden_size,
                          num_layers=n_layers,
                          bidirectional=bidirectional,
                          batch_first=True,
                          dropout=0 if n_layers < 2 else dropout)

        self.linear = torch.nn.Linear(hidden_size * 2 if bidirectional else hidden_size, output_size)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, data):
        # input data is after the embedding

        # data = [batch size, sent len, emb dim]
        self.rnn.flatten_parameters()
        packed_output, (hidden, cell) = self.rnn(data)

        # hidden = [n layers * n directions, batch size, emb dim]
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1, :, :])

        # hidden = [batch size, hid dim]
        output = self.linear(hidden)
        # output = [batch size, out dim]

        return output


def arch_factory_kwargs_generator(train_dataset_desc, clean_test_dataset_desc, triggered_test_dataset_desc):
    # Note: the arch_factory_kwargs_generator returns a dictionary, which is used as kwargs input into an
    #  architecture factory.  Here, we allow the input-dimension and the pad-idx to be set when the model gets
    #  instantiated.  This is useful because these indices and the vocabulary size are not known until the
    #  vocabulary is built.
    # TODO figure out if I can remove this
    output_dict = dict(input_size=train_dataset_desc['embedding_size'])
    return output_dict


# https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/2%20-%20Upgraded%20Sentiment%20Analysis.ipynb

# class EmbeddingLSTMFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
#     def new_architecture(self, input_dim=25000, embedding_dim=100, hidden_dim=256, output_dim=1,
#                          n_layers=2, bidirectional=True, dropout=0.5, pad_idx=-999):
#         return trojai.modelgen.architectures.text_architectures.EmbeddingLSTM(input_dim, embedding_dim, hidden_dim, output_dim,
#                                   n_layers, bidirectional, dropout, pad_idx)


class LinearFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, input_size: int, hidden_size: int, output_size: int, dropout: float, bidirectional: bool, n_layers: int):
        model = LinearModel(input_size, output_size, dropout)
        return model


class GruLinearFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, input_size: int, hidden_size: int, output_size: int, dropout: float, bidirectional: bool, n_layers: int):
        model = GruLinearModel(input_size, hidden_size, output_size, dropout, bidirectional, n_layers)
        return model


class LstmLinearFactory(trojai.modelgen.architecture_factory.ArchitectureFactory):
    def new_architecture(self, input_size: int, hidden_size: int, output_size: int, dropout: float, bidirectional: bool, n_layers: int):
        model = LstmLinearModel(input_size, hidden_size, output_size, dropout, bidirectional, n_layers)
        return model


def get_factory(model_name: str):
    model = None

    if model_name == 'LstmLinear':
        model = LstmLinearFactory()
    elif model_name == 'GruLinear':
        model = GruLinearFactory()
    elif model_name == 'Linear':
        model = LinearFactory()
    else:
        raise RuntimeError('Invalid Model Architecture Name: {}'.format(model_name))

    return model

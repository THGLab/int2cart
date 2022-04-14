import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modelling.layers import ScaleShift, Standardize, Dense, shifted_softplus, GaussianSmearing, VariableLengthBatchNorm
from modelling.models.mlp import MLP


class VAEAtomWise(nn.Module):
    """
    The variational autoencoder model using LSTM units with
    bias and batch size at the begining of input tensor.

    Parameters
    ----------
    n_atoms: int
        number of atoms in the IDP state

    input_size: int
        number of input features to each LSTM.

    hidden_size: int
        number of hidden features of LSTM.

    lstm_layers: int
        number of lstm layers

    atom_embedding: int
        number of atom embeddings

    distrib_size: int
        number of means and standard deviations of generative models.

    distrib_atomwise: bool
        If True, find a distribution for each atom separately.

    dropout: float
        the dropout for lstm layers

    normalizer: tuple
        a tuple of two values: mean and SD of features

    device: torch.device
        either CPU or GPU
    """

    def __init__(self,
                 n_atoms,
                 input_size,
                 hidden_size,
                 lstm_layers,
                 atom_embedding,
                 distrib_size,
                 dropout= 0.0,
                 normalizer=(0.0,1.0),
                 device=None):
        super(VAEAtomWise, self).__init__()

        self.normalize = Standardize(mean=torch.tensor(normalizer[0], dtype=torch.float32, device=device),
                                     stddev=torch.tensor(normalizer[1], dtype=torch.float32, device=device)
                                    )

        self.atom_embedding = atom_embedding
        if atom_embedding >0:
            self.embedding = nn.Embedding(3, atom_embedding)

        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        self.downsampling = nn.ModuleList([
            Dense(hidden_size+atom_embedding, hidden_size, activation=nn.ReLU()),
            Dense(hidden_size, hidden_size, activation=nn.ReLU())
            ])
        self.z_mean = Dense(hidden_size, distrib_size, activation=None)
        self.z_std = Dense(hidden_size, distrib_size, activation=None)
        self.upsampling = nn.ModuleList([
            Dense(distrib_size, hidden_size, activation=nn.ReLU()),
            Dense(hidden_size, hidden_size, activation=nn.ReLU())
        ])

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        self.output = Dense(hidden_size, input_size, activation=None)

        self.inverse_normalize = ScaleShift(mean=torch.tensor(normalizer[0], dtype=torch.float32, device=device),
                                            stddev=torch.tensor(normalizer[1], dtype=torch.float32, device=device)
                                            )

    def forward(self, data):
        """

        Parameters
        ----------
        data: autoeisd.data.BatchDataset
            instance of autoeisd.data.BatchDataset

        Returns
        -------

        """
        structures = data.structures

        x = self.normalize(structures)

        # encoder
        x, _ = self.encoder(x)

        if self.atom_embedding > 0:
            atom_embedding = self.embedding(data.atomic_numbers)
            x = torch.cat([x,atom_embedding], dim=-1)

        for downsample in self.downsampling:
            x = downsample(x)   # B, A, H+E  (batch, n_atoms, hidden size + atom embedding)

        # Todo: separate sampling
        # sampling
        z_mean = self.z_mean(x)
        z_std = self.z_std(x)
        x = torch.normal(z_mean, z_std)

        # Todo: separate decoder
        # decoder
        for upsample in self.upsampling:
            x = upsample(x)

        x, _ = self.decoder(x)

        x= self.output(x)

        # inverse transform
        x = self.inverse_normalize(x)

        return x, z_mean, z_std


class VAECentral(nn.Module):
    """
    The variational autoencoder model using LSTM units with
    bias and batch size at the begining of input tensor.

    Parameters
    ----------
    n_atoms: int
        number of atoms in the IDP state

    input_size: int
        number of input features to each LSTM.

    hidden_size: int
        number of hidden features of LSTM.

    lstm_layers: int
        number of lstm layers

    atom_embedding: int
        number of atom embeddings

    distrib_size: int
        number of means and standard deviations of generative models.

    dropout: float
        the dropout for lstm layers

    normalizer: tuple
        a tuple of two values: mean and SD of features

    device: torch.device
        either CPU or GPU
    """

    def __init__(self,
                 n_atoms,
                 input_size,
                 hidden_size,
                 lstm_layers,
                 atom_embedding,
                 distrib_size,
                 dropout= 0.0,
                 normalizer=(0.0,1.0),
                 device=None):
        super(VAECentral, self).__init__()

        self.n_atoms = n_atoms
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.atom_embedding = atom_embedding

        self.normalize = Standardize(mean=torch.tensor(normalizer[0], dtype=torch.float32, device=device),
                                     stddev=torch.tensor(normalizer[1], dtype=torch.float32, device=device)
                                    )

        # atom embedding
        if atom_embedding>0:
            self.embedding = nn.Embedding(3, atom_embedding)

        self.encoder = nn.LSTM(
            input_size=input_size+atom_embedding,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        self.downsampling = nn.ModuleList([
            Dense(n_atoms*hidden_size, n_atoms, activation=nn.ReLU()),
            Dense(n_atoms, n_atoms, activation=nn.ReLU())
            ])
        self.z_mean = Dense(n_atoms, distrib_size, activation=None)
        self.z_std = Dense(n_atoms, distrib_size, activation=None)
        self.upsampling = nn.ModuleList([
            Dense(distrib_size, n_atoms, activation=nn.ReLU()),
            Dense(n_atoms, n_atoms*hidden_size, activation=nn.ReLU())
        ])

        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        self.output = Dense(hidden_size, input_size, activation=None)

        self.inverse_normalize = ScaleShift(mean=torch.tensor(normalizer[0], dtype=torch.float32, device=device),
                                            stddev=torch.tensor(normalizer[1], dtype=torch.float32, device=device)
                                            )

    def forward(self, data):
        """

        Parameters
        ----------
        data: autoeisd.data.BatchDataset
            instance of autoeisd.data.BatchDataset

        Returns
        -------

        """
        structures = data.structures
        x = self.normalize(structures)

        if self.atom_embedding>0:
            atom_embedding = self.embedding(data.atomic_numbers)
            x = torch.cat([x, atom_embedding], dim=-1)

        # encoder
        x, _ = self.encoder(x)

        # flatten
        x = torch.flatten(x, start_dim=1, end_dim=-1)

        for downsample in self.downsampling:
            x = downsample(x)   # B, A*H  (batch, n_atoms*hidden size)

        # sampling
        z_mean = self.z_mean(x)
        z_log_var = self.z_std(x)
        epsilon = torch.normal(mean = torch.zeros_like(z_mean),
                               std = torch.ones_like(z_log_var))

        x = z_mean + torch.exp(z_log_var) * epsilon

        # Todo: separate decoder
        # decoder
        for upsample in self.upsampling:
            x = upsample(x)

        x = x.view(x.size()[0], self.n_atoms, self.hidden_size)
        x, _ = self.decoder(x)

        x = self.output(x)

        # inverse transform
        x = self.inverse_normalize(x)

        return x, z_mean, z_log_var


class EncoderLSTMCentral(nn.Module):
    """

    """
    def __init__(self,
                n_atoms,
                input_size,
                hidden_size,
                lstm_layers,
                atom_embedding,
                distrib_size,
                bidirectional=False,
                smearing=None,
                soft_target = False,
                dropout = 0.0,
                normalizer = (0.0, 1.0),
                device = None):
        super(EncoderLSTMCentral, self).__init__()

        self.n_atoms = n_atoms
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.atom_embedding = atom_embedding
        self.soft_target = soft_target

        self.normalize = Standardize(mean=torch.tensor(normalizer[0], dtype=torch.float32, device=device),
                                     stddev=torch.tensor(normalizer[1], dtype=torch.float32, device=device)
                                    )

        # gaussian smearing
        self.multi_smear = False
        self.smearing = smearing   # None, or a single gaussian smearing
        if isinstance(smearing, tuple):
            self.multi_smear = True
            self.smearing = nn.ModuleList(list(smearing))


        # atom embedding
        if atom_embedding>0:
            self.embedding = nn.Embedding(3, atom_embedding)

        self.lstm = nn.LSTM(
            input_size=input_size+atom_embedding,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        coeff=1
        if bidirectional:
            coeff=2
        self.downsampling = nn.ModuleList([
            Dense(n_atoms*hidden_size*coeff, n_atoms*3, activation=nn.ReLU()),
            Dense(n_atoms*3, n_atoms, activation=nn.ReLU())
            ])
        self.z_mean = Dense(n_atoms, distrib_size, activation=None)
        self.z_std = Dense(n_atoms, distrib_size, activation=None)

    def forward(self, data):
        """

        Parameters
        ----------
        data: autoeisd.data.BatchDataset
            instance of autoeisd.data.BatchDataset

        Returns
        -------

        """
        x = data.structures                         # B,A / B,A,D

        if self.smearing is not None:
            if not self.multi_smear:
                x = self.smearing(x)                    # B,A,nG
                if self.soft_target:
                    data.out_struct = x   # soft target
            else:
                y = torch.zeros_like(x)[:,:,None].repeat(1,1,self.input_size)# B,A,nG
                for i, smear in enumerate(self.smearing):
                    y[:,i::3,:] = smear(x[:,i::3])      # B,A/3,nG --> B,A,nG
                x = y                                   # B,A,nG
                if self.soft_target:
                    data.out_struct = x   # soft target

        x = self.normalize(x)              # B,A, nG

        if self.atom_embedding > 0:
            atom_embedding = self.embedding(data.atomic_numbers)
            x = torch.cat([x, atom_embedding], dim=-1)   # B,A, nG+nEmbed

        # encoder
        x, _ = self.lstm(x)                         # B,A,nH

        # flatten
        x = torch.flatten(x, start_dim=1, end_dim=-1)   # B,A*nH

        for downsample in self.downsampling:
            x = downsample(x)  # B, A*nH --> B, A*3  --> B,A

        # sampling
        z_mean = self.z_mean(x)      # B,dist
        z_log_var = self.z_std(x)    # B,dist

        return z_mean, z_log_var


class DecoderLSTMCentral(nn.Module):
    """

    """
    def __init__(self,
                 n_atoms,
                 input_size,
                 hidden_size,
                 lstm_layers,
                 atom_embedding,
                 distrib_size,
                 bidirectional=False,
                 activation_output=None,
                 dropout=0.0,
                 normalizer=(0.0, 1.0),
                 device=None):
        super(DecoderLSTMCentral, self).__init__()

        self.n_atoms = n_atoms
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.atom_embedding = atom_embedding

        self.upsampling = nn.ModuleList([
            Dense(distrib_size, n_atoms*3, activation=nn.ReLU()),
            Dense(n_atoms*3, n_atoms*hidden_size, activation=nn.ReLU())
        ])

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            bias=True,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )

        coeff = 1
        if bidirectional:
            coeff = 2
        if activation_output is None:
            self.output = Dense(hidden_size*coeff, input_size, activation=None)
            self.inverse_normalize = ScaleShift(mean=torch.tensor(normalizer[0], dtype=torch.float32, device=device),
                                                stddev=torch.tensor(normalizer[1], dtype=torch.float32, device=device)
                                                )
        elif activation_output == 'softmax':
            self.output = nn.ModuleList([
                Dense(hidden_size*coeff, input_size, activation=None),
                nn.Softmax(dim=-1)
            ])
        self.activation_output = activation_output



    def forward(self, x):
        """

        Parameters
        ----------
        x: torch.tensor
            tensor of (batch_size, distribution dim)

        Returns
        -------

        """
        for upsample in self.upsampling:
            x = upsample(x)      # B,A --> B,A*3 --> B,A*nH

        x = x.view(x.size()[0], self.n_atoms, self.hidden_size)
        x, _ = self.lstm(x)

        if self.activation_output is None:
            x = self.output(x)
            # inverse transform
            x = self.inverse_normalize(x)


        elif self.activation_output=='softmax':
            for layer in self.output:
                x = layer(x)      # B,A,nH --> B,A,nG  --> B,A,nG

        return x


class VAEDenseEncoder(nn.Module):

    def __init__(self,
                 filter_in,
                 n_filter_layers=1,
                 filter_out=32,
                 encoder_in=32*174,
                 encoder_neurons=[256,128],
                 encoder_out=64,
                 latent_dimension=50
                 ):
        """
        Fully Connected layers to encode molecule to latent space

        Parameters
        ----------
        n_filter_layers: int
        filter_out: int
        neurons: list
        latent_dimension; int
        """
        super(VAEDenseEncoder, self).__init__()
        self.latent_dimension = latent_dimension

        # filter and transform torsion ohe
        self.filter = MLP(filter_in, filter_out,
                          n_layers=n_filter_layers,
                          activation=nn.ReLU())

        # Reduce dimension up to second last layer of Encoder
        self.encode_nn = MLP(encoder_in,
                             encoder_out,
                             n_hidden=encoder_neurons,
                             activation=shifted_softplus,
                             out_activation=shifted_softplus) #nn.ReLU()

        # Latent space mean
        self.encode_mu = nn.Linear(encoder_out, latent_dimension)

        # Latent space variance
        self.encode_log_var = nn.Linear(encoder_out, latent_dimension)

    @staticmethod
    def reparameterize(z_mu, z_log_var):
        """
        This trick is explained well here:
            https://stats.stackexchange.com/a/16338
        """
        epsilon = torch.randn_like(z_log_var)
        z = z_mu + torch.exp(0.5 * z_log_var) * epsilon
        return z

        # std = torch.exp(0.5 * z_log_var)
        # eps = torch.randn_like(std)
        # return eps.mul(std).add_(z_mu)

    def forward(self, x):
        """
        Pass throught the Encoder
        """
        # filter
        x = self.filter(x)

        # flatten and concatenate all features
        x = x.flatten(start_dim=1)

        # Get results of encoder network
        h1 = self.encode_nn(x)

        # latent space
        z_mu = self.encode_mu(h1)
        z_log_var = self.encode_log_var(h1)

        # Reparameterize
        z = self.reparameterize(z_mu, z_log_var)
        return z, z_mu, z_log_var


class VAEGRUDecoder(nn.Module):

    def __init__(self, latent_dimension, gru_stack_size,
                 gru_neurons_num, gru_dropout, out_dimension):
        """
        Through Decoder
        """
        super(VAEGRUDecoder, self).__init__()
        self.latent_dimension = latent_dimension
        self.gru_stack_size = gru_stack_size
        self.gru_neurons_num = gru_neurons_num

        # Simple Decoder
        self.decode_RNN = nn.GRU(
            input_size=latent_dimension,
            hidden_size=gru_neurons_num,
            num_layers=gru_stack_size,
            batch_first=False,
            dropout=gru_dropout)

        self.decode_FC = nn.Sequential(
            nn.Linear(gru_neurons_num, out_dimension),
        )

    def init_hidden(self, batch_size=1):
        weight = next(self.parameters())
        return weight.new_zeros(self.gru_stack_size, batch_size,
                                self.gru_neurons_num)

    def forward(self, z, hidden):
        """
        A forward pass throught the entire model.
        """

        # Decode
        l1, hidden = self.decode_RNN(z, hidden)
        decoded = self.decode_FC(l1)  # fully connected layer

        return decoded, hidden


class VAEGRUEncoderAtomwise(nn.Module):

    def __init__(self,
                 filter_in,
                 n_filter_layers,
                 filter_out,
                 gru_stack_size,
                 gru_neurons_num,
                 latent_dimension=50
                 ):
        """
        Fully Connected layers to encode molecule to latent space

        Parameters
        ----------
        n_filter_layers: int
        filter_out: int
        neurons: list
        latent_dimension; int
        """
        super(VAEGRUEncoderAtomwise, self).__init__()
        self.latent_dimension = latent_dimension

        # filter and transform torsion ohe
        self.filter = MLP(filter_in, filter_out,
                          n_layers=n_filter_layers,
                          activation=nn.ReLU())

        # Reduce dimension up to second last layer of Encoder
        self.encode_nn = nn.GRU(input_size=filter_out,
                                hidden_size=gru_neurons_num,
                                num_layers=gru_stack_size,
                                batch_first=True)

        # Latent space mean
        self.encode_mu = Dense(gru_neurons_num, latent_dimension,bias=False,activation=None)

        # Latent space variance
        self.encode_log_var = Dense(gru_neurons_num, latent_dimension,bias=False,activation=None)

    @staticmethod
    def reparameterize(z_mu, z_log_var):
        """
        This trick is explained well here:
            https://stats.stackexchange.com/a/16338
        """
        epsilon = torch.randn_like(z_log_var)
        z = z_mu + torch.exp(0.5 * z_log_var) * epsilon
        return z

        # std = torch.exp(0.5 * z_log_var)
        # eps = torch.randn_like(std)
        # return eps.mul(std).add_(z_mu)

    def forward(self, x):
        """
        Pass throught the Encoder
        """
        # filter
        x = self.filter(x)
        # flatten and concatenate all features
        # x = x.flatten(start_dim=1)

        # Get results of encoder network
        h1, _ = self.encode_nn(x)

        # latent space
        z_mu = self.encode_mu(h1)
        z_log_var = self.encode_log_var(h1)

        # Reparameterize
        z = self.reparameterize(z_mu, z_log_var)
        return z, z_mu, z_log_var


class RecurrentModel(nn.Module):

    def __init__(self,
                 recurrent,
                 smearing_parameters,
                 n_filter_layers,
                 filter_size,
                 res_embedding_size,
                 rec_stack_size,
                 rec_neurons_num,
                 rec_dropout,
                 use_layernorm=False
                 ):
        """
        GRU Recurrent units for the language model.
        Model: recurrent model with torsion angle, type and

        Parameters
        ----------
        recurrent: str
            'gru', 'lstm'
        smearing_parameters: {start, stop, n_gaussians, ...}
        n_filter_layers: int
        filter_size: int
        res_embedding_size: int, residue embedding size
        neurons: list
        latent_dimension; int
        """
        super(RecurrentModel, self).__init__()
        self.rec_stack_size = rec_stack_size
        self.rec_neurons_num = rec_neurons_num
        # filter and transform torsion ohe
        self.angle_smearing = GaussianSmearing(**smearing_parameters)
        self.residue_embedding = nn.Embedding(20, res_embedding_size)
        self.filters = nn.ModuleList([MLP(smearing_parameters['n_gaussians'], filter_size,
                          n_layers=n_filter_layers,
                          activation=nn.ReLU()) for _ in range(3)])

        # phi,psi,omega each has filter_out, and then also embedded residue type
        self.mixing_filter = MLP(filter_size*3+res_embedding_size, filter_size,
                          n_layers=n_filter_layers,
                          activation=nn.ReLU())

        # Reduce dimension up to second last layer of Encoder
        if recurrent == 'gru':
            self.recurrent = nn.GRU(input_size=filter_size,   
                                    hidden_size=rec_neurons_num,
                                    num_layers=rec_stack_size,
                                    batch_first=False,
                                    dropout=rec_dropout,
                                    bidirectional=True)
        elif recurrent == 'lstm':
            self.recurrent = nn.LSTM(input_size=filter_size,
                                     hidden_size=rec_neurons_num,
                                     num_layers=rec_stack_size,
                                     batch_first=False,
                                    dropout=rec_dropout,
                                    bidirectional=True)
        if use_layernorm:
            # self.norm = nn.LayerNorm(rec_neurons_num * 2)
            self.norm = VariableLengthBatchNorm(rec_neurons_num * 2)
        else:
            self.norm = None

        # # linear mapping
        # self.linear = nn.Linear(rec_neurons_num, out_dimension)
        # # self.linear = Dense(rec_neurons_num, out_dimension,bias=False,activation=None)

    def init_hidden(self, batch_size=1, reference_tensor=None):
        # weight = next(self.parameters())
        # hidden =  weight.new_zeros(self.rec_stack_size * 2, batch_size,
        #                         self.rec_neurons_num)
        hidden = reference_tensor.new_zeros((self.rec_stack_size * 2, batch_size,
                                self.rec_neurons_num))
        if type(self.recurrent) is nn.LSTM:
            cell = self.init_cell(batch_size, reference_tensor)
            hidden = (hidden, cell)
        return hidden

    def init_cell(self, batch_size=1, reference_tensor=None):
        """
        only used when rnn type is lstm.

        Parameters
        ----------
        batch_size: int

        Returns
        -------

        """
        # weight = next(self.parameters())
        return reference_tensor.new_zeros(self.rec_stack_size * 2, batch_size,
                                self.rec_neurons_num)

    def forward(self, inputs):
        """
        Pass throught the language model.
        

        Parameters
        ----------
        inputs: dictionary with keys {"phi", "psi", "omega", "res_type"}
            "phi", "psi", "omega": float torch.tensor (l_seq, batch,)
            "res_type": long torch.tensor (l_seq, batch)
            "lengths": list of long (batch)
        """
        batch_size = len(inputs["phi"])
        phi_smeared = self.angle_smearing(inputs["phi"] * 180/np.pi)
        psi_smeared = self.angle_smearing(inputs["psi"] * 180/np.pi)
        omega_smeared = self.angle_smearing(inputs["omega"] * 180/np.pi)


        # filter
        phi_filtered = self.filters[0](phi_smeared)
        psi_filtered = self.filters[1](psi_smeared)
        omega_filtered = self.filters[2](omega_smeared)

        res_type_embedded = self.residue_embedding(inputs["res_type"])


        # concatenate all features
        x = torch.cat([phi_filtered, psi_filtered, omega_filtered, res_type_embedded], dim=2)
        # mixing filter
        x = self.mixing_filter(x)

        # initialize hidden state
        hidden = self.init_hidden(batch_size, reference_tensor=x)

        # do axis manipulation and handle different length sequences
        x = torch.moveaxis(x, 0, 1)
        packed = pack_padded_sequence(x, inputs['lengths'], enforce_sorted=False)

        # Get results of encoder network and recover axis order
        packed_x, hidden = self.recurrent(packed, hidden)

        x, len_unpacked = pad_packed_sequence(packed_x)
        x = torch.moveaxis(x, 0, 1)

        # apply layernorm when needed
        if self.norm is not None:
            x = self.norm(x, inputs['lengths'])

        # return latent representation
        return x

    def generate(self, tor_angle, tor_type, res_type, hidden):
        """
        Pass throught the language model.
        Both input and output are provided as (seq, batch, feature).

        Parameters
        ----------
        tor_angle: torch.tensor
            (seq, batch, tensor)
        """
        # filter
        x = self.filter(tor_angle)

        # concatenate all features
        x = torch.cat([x,res_type,tor_type],dim=2)

        # Get results of encoder network
        x, hidden = self.recurrent(x, hidden)

        # last layer
        x = self.linear(x)

        # softmax
        x = nn.functional.softmax(x, dim=-1)

        # sample from the network as a multinomial distribution
        # top_i = torch.multinomial(x.view[-1], 1)[0].data.cpu().numpy()

        return x, hidden


class RecurrentModelFast(nn.Module):

    def __init__(self,
                 recurrent,
                 filter_in,
                 n_filter_layers,
                 filter_out,
                 rec_stack_size,
                 rec_neurons_num,
                 rec_dropout,
                 out_dimension
                 ):
        """
        GRU Recurrent units for the language model.
        Model: recurrent model with torsion angle, type and

        Parameters
        ----------
        recurrent: str
            The core recurrent model, one of these two: 'gru', 'lstm'
        n_filter_layers: int
        filter_out: int
        neurons: list
        latent_dimension; int
        """
        super(RecurrentModelFast, self).__init__()
        self.rec_stack_size = rec_stack_size
        self.rec_neurons_num = rec_neurons_num
        # filter and transform torsion ohe
        self.filter = MLP(filter_in, filter_out,
                          n_layers=n_filter_layers,
                          activation=nn.ReLU())

        # Reduce dimension up to second last layer of Encoder
        if recurrent == 'gru':
            self.recurrent = nn.GRU(input_size=filter_out+59+3,
                                    hidden_size=rec_neurons_num,
                                    num_layers=rec_stack_size,
                                    batch_first=True,
                                    dropout=rec_dropout)
        elif recurrent == 'lstm':
            self.recurrent = nn.LSTM(input_size=filter_out+59+3,
                                     hidden_size=rec_neurons_num,
                                     num_layers=rec_stack_size,
                                     batch_first=True,
                                    dropout=rec_dropout)

        # linear mapping
        self.linear = nn.Linear(rec_neurons_num, out_dimension)
        # self.linear = Dense(rec_neurons_num, out_dimension,bias=False,activation=None)

    def init_hidden(self, batch_size=1):
        """
        initialize hidden state of the recurrent network
        Parameters
        ----------
        batch_size

        Returns
        -------

        """
        weight = next(self.parameters())
        return weight.new_zeros(self.rec_stack_size, batch_size,
                                self.rec_neurons_num)

    def init_cell(self, batch_size=1):
        """
        only used when rnn type is lstm.

        Parameters
        ----------
        batch_size: int

        Returns
        -------

        """
        weight = next(self.parameters())
        return weight.new_zeros(self.rec_stack_size, batch_size,
                                self.rec_neurons_num)

    def forward(self, tor_angle, tor_type, res_type):
        """
        Pass throught the language model.
        Both input and output are provided as (batch, seq, feature).

        Parameters
        ----------
        tor_angle: torch.tensor
            (batch, seq, feature)
        """
        # filter
        x = self.filter(tor_angle)

        # concatenate all features
        x = torch.cat([x,res_type,tor_type],dim=2)

        # Get results of encoder network
        x, _ = self.recurrent(x)

        # last layer
        x = self.linear(x)

        return x

    def generate(self, tor_angle, tor_type, res_type, hidden):
        """
        Pass throught the language model.
        Both input and output are provided as (seq, batch, feature).

        Parameters
        ----------
        tor_angle: torch.tensor
            (batch, seq, tensor)

        Returns
        -------
        torch.tensor: probabilities
        torch.tensor: hidden state for the next atom

        """
        # filter
        x = self.filter(tor_angle)

        # concatenate all features
        x = torch.cat([x, res_type, tor_type], dim=2)

        # Get results of encoder network
        x, hidden = self.recurrent(x, hidden)

        # last layer
        x = self.linear(x)

        # softmax
        x = nn.functional.softmax(x, dim=-1)

        # sample from the network as a multinomial distribution
        # top_i = torch.multinomial(x.view[-1], 1)[0].data.cpu().numpy()

        return x, hidden

    def generate_batch(self, n_conformer, data):
        """
        Pass throught the language model.
        Both input and output are provided as (seq, batch, feature).

        Parameters
        ----------
        n_conformer: int
            number of conformers to generate

        data: instance of BatchDataset
            should have following keys: structures, tor_type, res_type

        Returns
        -------
        list: list of n_conformer lists of torsion angles

        """

        # initialize hidden state
        hidden = self.init_hidden(1)
        cell = self.init_cell(1)
        hidden = (hidden, cell)

        # meta info
        seq_length = data.structures.shape[1]

        conformers = []
        for _ in range(n_conformer):
            conformer = []

            tor_angle = data.structures[0, 0, :].unsqueeze(0).unsqueeze(0)
            tor_type = data.tor_type[0, 0, :].unsqueeze(0).unsqueeze(0)
            res_type = data.res_type[0, 0, :].unsqueeze(0).unsqueeze(0)

            for i in range(seq_length):
                # filter
                x = self.filter(tor_angle)

                # concatenate all features
                x = torch.cat([x, res_type, tor_type], dim=2)

                # Get results of encoder network
                x, hidden = self.recurrent(x, hidden)

                # last layer
                x = self.linear(x)

                # softmax
                x = nn.functional.softmax(x, dim=-1)

                # sample from the network as a multinomial distribution
                # top_i = torch.multinomial(x.view[-1], 1)[0].data.cpu().numpy()

        return x, hidden

class RecurrentModelResidue(nn.Module):

    def __init__(self,
                 recurrent,
                 filter_in,
                 n_filter_layers,
                 filter_out,
                 rec_stack_size,
                 rec_neurons_num,
                 rec_dropout,
                 out_dimension
                 ):
        """
        GRU Recurrent units for the language model.
        Model: recurrent model with torsion angle, type and

        Parameters
        ----------
        recurrent: str
            'gru', 'lstm'
        n_filter_layers: int
        filter_out: int
        neurons: list
        latent_dimension; int
        """
        super(RecurrentModelResidue, self).__init__()
        self.rec_stack_size = rec_stack_size
        self.rec_neurons_num = rec_neurons_num

        # filter and transform torsion angles
        self.filter = MLP(filter_in, filter_out,
                          n_layers=n_filter_layers,
                          activation=nn.ReLU())

        # Reduce dimension up to second last layer of Encoder
        if recurrent == 'gru':
            self.recurrent = nn.GRU(input_size=3*filter_out+59,
                                    hidden_size=rec_neurons_num,
                                    num_layers=rec_stack_size,
                                    batch_first=True,
                                    dropout=rec_dropout)
        elif recurrent == 'lstm':
            self.recurrent = nn.LSTM(input_size=3*filter_out+59,
                                     hidden_size=rec_neurons_num,
                                     num_layers=rec_stack_size,
                                     batch_first=True,
                                    dropout=rec_dropout)

        # linear mapping
        self.linear_phi = nn.Linear(rec_neurons_num, out_dimension)
        self.linear_psi = nn.Linear(rec_neurons_num, out_dimension)
        self.linear_omega = nn.Linear(rec_neurons_num, out_dimension)
        self.linear_res_type = nn.Linear(rec_neurons_num, out_dimension)
        # self.linear = Dense(rec_neurons_num, out_dimension,bias=False,activation=None)

    def init_hidden(self, batch_size=1):
        """
        initialize hidden state of the recurrent network
        Parameters
        ----------
        batch_size

        Returns
        -------

        """
        weight = next(self.parameters())
        return weight.new_zeros(self.rec_stack_size, batch_size,
                                self.rec_neurons_num)

    def init_cell(self, batch_size=1):
        """
        only used when rnn type is lstm.

        Parameters
        ----------
        batch_size: int

        Returns
        -------

        """
        weight = next(self.parameters())
        return weight.new_zeros(self.rec_stack_size, batch_size,
                                self.rec_neurons_num)

    def forward(self, tor_angle, res_type):
        """
        Pass through the language model.

        Parameters
        ----------
        tor_angle: torch.tensor
            (batch, seq*3, ohe_size)

        res_type: torch.tensor
            (batch, seq, seq)
        """
        # filter
        x = self.filter(tor_angle)
        b, seq3, f = x.shape

        # concatenate 3 torsion angles
        x = x.view(b,seq3//3,f*3)     # batch,seq,3*filter_out
        # x = torch.flatten(x, start_dim=2, end_dim=3)    # batch,seq,3*filter_out

        # concatenate all features
        x = torch.cat([x,res_type],dim=2)

        # Get results of encoder network
        x, _ = self.recurrent(x)

        # last layer
        phi = self.linear_phi(x)
        psi = self.linear_psi(x)
        omega = self.linear_omega(x)
        torsions = torch.cat([phi, psi, omega], dim=-1)   # batch, seq, n_bins*3
        torsions = torsions.view(b, seq3, phi.shape[-1])  # batch, seq*3, n_bins
        res_type = self.linear_res_type(x)

        output = {'torsions':torsions, 'res_type':res_type}

        return output

    def generate(self, tor_angle, res_type, hidden):
        """
        Pass throught the language model.

        Parameters
        ----------
        tor_angle: torch.tensor
            (1, seq*3, n_bins)

        res_type: torch.tensor
            (1, seq*3, n_bins)

        Returns
        -------
        torch.tensor: probabilities
        torch.tensor: hidden state for the next atom

        """
        # filter
        x = self.filter(tor_angle)

        # concatenate 3 torsion angles
        x = torch.flatten(x, start_dim=2, end_dim=3)    # batch,seq,3*filter_out

        # concatenate all features
        x = torch.cat([x, res_type], dim=2)

        # Get results of encoder network
        x, hidden = self.recurrent(x, hidden)

        # last layer
        phi = self.linear_phi(x)
        psi = self.linear_psi(x)
        omega = self.linear_omega(x)
        res_type = self.linear_res_type(x)

        # softmax
        phi = nn.functional.softmax(phi, dim=-1)
        psi = nn.functional.softmax(psi, dim=-1)
        omega = nn.functional.softmax(omega, dim=-1)
        res_type = nn.functional.softmax(res_type, dim=-1)

        output = {'phi': phi, 'psi': psi, 'omega': omega, 'res_type': res_type}

        # sample from the network as a multinomial distribution
        # top_i = torch.multinomial(x.view[-1], 1)[0].data.cpu().numpy()

        return output, hidden

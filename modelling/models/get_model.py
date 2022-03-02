from modelling.models import RecurrentModel, MLP, MultiHeadModel, Sequential
from modelling.layers import ScaleShift
from modelling.utils.default_scalers import *

def get_model(settings):
    smearing = {'start': -180,
                'stop': 180,
                'n_gaussians': settings['model']['n_gaussians'],
                'margin': settings['model']['gaussian_margin'],
                'width_factor': settings['model']['gaussian_factor'],
                'normalize': settings['model']['gaussian_normalize']}

    latent_dim = settings['model']['rec_neurons_num'] 
    hidden_dim = latent_dim // 2

    rnn_encoder = RecurrentModel(recurrent=settings['model']['recurrent'],
                            smearing_parameters=smearing,
                            n_filter_layers=settings['model']['n_filter_layers'],
                            filter_size=settings['model']['filter_size'],
                            res_embedding_size=settings['model']['filter_size'],
                            rec_stack_size=settings['model']['rec_stack_size'],
                            rec_neurons_num=settings['model']['rec_neurons_num'] // 2,  # bidirectional
                            rec_dropout=settings['model']['rec_dropout'])

    # prepare output heads
    if settings['bins']['bond_length_bin']:
        n_ca_blens_predictor = MLP(latent_dim, settings['bins']['n-ca_bin'][-1], n_hidden=hidden_dim, use_batchnorm=settings['model']['use_batchnorm'])
        ca_c_blens_predictor = MLP(latent_dim, settings['bins']['ca-c_bin'][-1], n_hidden=hidden_dim, use_batchnorm=settings['model']['use_batchnorm'])
        c_n_blens_predictor = MLP(latent_dim, settings['bins']['c-n_bin'][-1], n_hidden=hidden_dim, use_batchnorm=settings['model']['use_batchnorm'])
    else:
        n_ca_blens_predictor = Sequential(MLP(latent_dim, 1, n_hidden=hidden_dim, use_batchnorm=settings['model']['use_batchnorm']), 
            ScaleShift(default_means["N_CA_bond_length"], default_stds["N_CA_bond_length"], False))
        ca_c_blens_predictor = Sequential(MLP(latent_dim, 1, n_hidden=hidden_dim, use_batchnorm=settings['model']['use_batchnorm']),
            ScaleShift(default_means["CA_C_bond_length"], default_stds["CA_C_bond_length"], False))
        c_n_blens_predictor = Sequential(MLP(latent_dim, 1, n_hidden=hidden_dim, use_batchnorm=settings['model']['use_batchnorm']),
            ScaleShift(default_means["C_N_bond_length"], default_stds["C_N_bond_length"], False))

    if settings['bins']['backbone_angle_bin']:
        n_ca_c_angle_predictor = MLP(latent_dim, settings['bins']['angle_bin_count'], n_hidden=hidden_dim, use_batchnorm=settings['model']['use_batchnorm'])
        ca_c_n_angle_predictor = MLP(latent_dim, settings['bins']['angle_bin_count'], n_hidden=hidden_dim, use_batchnorm=settings['model']['use_batchnorm'])
        c_n_ca_angle_predictor = MLP(latent_dim, settings['bins']['angle_bin_count'], n_hidden=hidden_dim, use_batchnorm=settings['model']['use_batchnorm'])
    else:
        n_ca_c_angle_predictor = Sequential(MLP(latent_dim, 1, n_hidden=hidden_dim, use_batchnorm=settings['model']['use_batchnorm']),
            ScaleShift(default_means["N_CA_C_bond_angle"], default_stds["N_CA_C_bond_angle"], True))
        ca_c_n_angle_predictor = Sequential(MLP(latent_dim, 1, n_hidden=hidden_dim, use_batchnorm=settings['model']['use_batchnorm']), 
            ScaleShift(default_means["CA_C_N_bond_angle"], default_stds["CA_C_N_bond_angle"], True))
        c_n_ca_angle_predictor = Sequential(MLP(latent_dim, 1, n_hidden=hidden_dim, use_batchnorm=settings['model']['use_batchnorm']), 
            ScaleShift(default_means["C_N_CA_bond_angle"], default_stds["C_N_CA_bond_angle"], True))

    sidechain_torsion_predictors = [MLP(latent_dim, settings['bins']['angle_bin_count'], n_hidden=hidden_dim, use_batchnorm=settings['model']['use_batchnorm']) for _ in range(6)]
    all_predictor_heads = [n_ca_c_angle_predictor, ca_c_n_angle_predictor, c_n_ca_angle_predictor] +\
        sidechain_torsion_predictors + [n_ca_blens_predictor, ca_c_blens_predictor, c_n_blens_predictor]
    
    model = MultiHeadModel(rnn_encoder, all_predictor_heads)
    return model
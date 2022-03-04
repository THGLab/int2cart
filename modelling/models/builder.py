from inspect import getmodule
from sidechainnet import StructureBuilder
from modelling.models import RecurrentModel, MLP, MultiHeadModel
from modelling.models import get_model
from torch import nn
import torch
from modelling.utils.geometry import calc_dist_mat

AA_MAP = {
    'A': 0,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'V': 17,
    'W': 18,
    'Y': 19
}
AA_MAP_INV = {v:k for k,v in AA_MAP.items()}

class BackboneBuilder(nn.Module):
    def __init__(self, settings):
        super().__init__()
        
        self.predictor = get_model(settings)

    def load_predictor_weights(self, model_state):
        self.predictor.load_state_dict(model_state)

    def replace_predictor(self, predictor):
        self.predictor = predictor

    def forward(self, inputs):
        predictions = self.predictor(inputs)
        all_angles = torch.cat([inputs["phi"].unsqueeze(-1), 
        inputs["psi"].unsqueeze(-1), 
        inputs["omega"].unsqueeze(-1)] + predictions[:3], dim=-1)
        zeros = torch.zeros_like(all_angles)
        all_angles = torch.cat([all_angles, zeros], dim=-1)
        all_blens = torch.cat(predictions[-3:], dim=-1)

        batch_size = len(inputs["lengths"])
        max_len = max(inputs['lengths'])
        result = torch.zeros((batch_size, max_len*4, 3), device=inputs["phi"].device)
        dist_mats = []
        for i in range(len(inputs['lengths'])):
            sequence = "".join([AA_MAP_INV[n] for n in inputs["res_type"][i][:inputs['lengths'][i]].cpu().numpy()])
            backbone_coord_indices = torch.cat([torch.arange(i*14,i*14+4) for i in range(len(sequence))])
            angles = all_angles[i]
            bond_lengths = all_blens[i]
            builder = StructureBuilder(sequence, ang=angles, bbb_len=bond_lengths, device=angles.device)
            coords = builder.build(with_sidechain=False)
            backbone_coords = coords[backbone_coord_indices]
            result[i, :len(sequence)*4] = backbone_coords

            dist_mat = calc_dist_mat(backbone_coords)
            dist_mats.append(dist_mat)
        return {"predictions": predictions, "coords": result, "dist_mats": dist_mats}


if __name__ == "__main__":
    test_run_dir = '/mnt/perlmutter_scratch/nn_modelling/local/gru/training_12/'
    import yaml

    settings = yaml.safe_load(open(test_run_dir + "run_scripts/gru.yml", "r"))
    settings["model"]["use_batchnorm"] = False

    builder = BackboneBuilder(settings)
    model_state = torch.load(test_run_dir + "models/best_model_state.tar")["model_state_dict"]
    builder.load_predictor_weights(model_state)
    builder.to("cuda:0")
    # builder.replace_predictor(torch.load(test_run_dir + "models/best_model.pt").to("cuda:0"))

    batch = torch.load("/mnt/perlmutter_scratch/nn_modelling/local/debug/training_3/debug/loss_explosion.pkl")["second_last_batch"]

    batch_inputs = {"phi": batch.angs[:, :, 0].to("cuda:0"),
                    "psi": batch.angs[:, :, 1].to("cuda:0"),
                    "omega": batch.angs[:, :, 2].to("cuda:0"),
                    "res_type": torch.argmax(batch.seqs, axis=-1).to("cuda:0"),
                    "lengths": batch.lengths}

    coords = builder(batch_inputs)
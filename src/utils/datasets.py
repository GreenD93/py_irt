import torch

import pyro
import pyro.poutine as poutine
import pyro.distributions as dist

import os

OUT_DIR = 'data'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def make_1pl_simulation_data(num_person, num_item, ability_dim):

    ability_prior = dist.Normal(
        torch.zeros(num_person, ability_dim, device=device),
        torch.ones(num_person, ability_dim, device=device)
    )

    item_feat_prior = dist.Normal(
        torch.zeros((num_item, 1), device=device),
        torch.ones((num_item, 1), device=device)
    )

    ability = pyro.sample("ability", ability_prior)
    item_feat = pyro.sample("item_feat", item_feat_prior)

    diff = item_feat

    logit = (torch.sum(ability, dim=1, keepdim=True) - diff.T).unsqueeze(2)
    response_mu = torch.sigmoid(1.7*logit)

    response_dist = dist.Bernoulli(response_mu)
    response = pyro.sample('response', response_dist)

    dataset = {
        'response': response,
        'ability': ability,
        'item_feat': item_feat
    }

    torch.save(dataset, os.path.join(OUT_DIR, '10000_100_simulation.path'))

if __name__ == "__main__":
    make_1pl_simulation_data(num_person=10000, num_item=100, ability_dim=1)

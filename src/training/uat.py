"""
Skeleton Code for Universal Adversarial Trigger (UAT) in PyTorch
"""
import torch
import random
from utils.data_processing import promptify_json
import json
import pathlib

def main(args, dataset):

    # each element in dataset is of form: (x_prefix_prompt, x_suffix_prompt, y_adv)
    # y_suffix_prompt is '' for right now

    for epoch in range(args.N_EPOCHS):
        x_best_trigger = (None, None, torch.inf)
        for example in dataset:
            x_pre, x_suf, y_adv = example
            candidate_losses = []
            modified_triggers = []
            for x_trig, sep_ind in T: # maybe an index of separation in the trigger
                x_trig_1, x_trig_2 = x_trig[:sep_ind], x_trig[sep_ind:]
                L = loss_func(args.model(x_pre, x_trig_1, y_adv, x_trig_2, x_suf), y_adv)
                candidate_losses.append(L)
                modified_triggers.append((x_trig, sep_ind))
                L.backward()
                for x_i in torch.concat(x_trig): # for each token in the trigger: vectorizable
                    L_alt = []
                    for v in V: # vectorizable
                        L_alt.append(
                            L + (v - V[x_i])*x_i.grad
                        )
                    S_xi = torch.topk(L_alt, K, largest=False)
                    args.S[x_trig] = S_xi # probably will be a tensor of shape (len(x_trig), K, len(V))
                for b in range(args.B):
                    I = torch.randperm(len(x_trig))[:C]
                    candidate_tokens = S[x_trig][I, random.choice(args.K, args.C)]
                    modified_trigger = x_trig.clone()
                    modified_trigger[I] = candidate_tokens
                    L_I = loss_func(args.model(x_pre, modified_trigger[:sep_ind], y_adv, modified_trigger[sep_ind:], x_suf), y_adv)
                    candidate_losses.append(L_I)
                    modified_triggers.append((modified_trigger, sep_ind))
            top_Q = torch.topk(candidate_losses, args.Q, largest=False)
            T = [modified_triggers[i] for i in top_Q.indices]
        val_loss = []
        for batch in val_loader:
            x_pre, x_suf, y_adv = batch
            for x_trig, sep_ind in T:
                val_loss += loss_func(args.model(x_pre, x_trig[:sep_ind], y_adv, x_trig[sep_ind:], x_suf), y_adv)
            # min index
            best_trig_candidate = torch.argmin(val_loss)
            if val_loss[best_trig_candidate] < x_best_trigger[2]:
                x_best_trigger = (T[best_trig_candidate], val_loss[best_trig_candidate])  

if __name__ == '__main__':
    args = {
        # need to calibrate the sizes of all these things
        'N_EPOCHS': 10,
        'B': 10, # candidate expansion size
        'Q': 10, # candidate queue size
        'C': 10, # number of coordinates for replacement
        'K': 10, # number of alternatives per token
        'T': [('adndflkj', 3), ('askdlgjhiu', 5), ('ldkfgkdfhrr', 2)], # a set of random initial triggers
        'model': None, # pre-trained Mistral model
        'V': None, # embedding table
        'S': {}, # { trigger: tensor } top K alternatives for each token for each trigger
    }
    
    dataset = []
    json_files = list(pathlib.Path('data').rglob('*.json'))[:3]
    for file in json_files:
        json_data = json.load(open(file, 'r'))
        json_data['goal_object'] = 'Help me navigate to the contact page'
        prompt = promptify_json(json_data)
        dataset.append((prompt, '', 'XXX BAD THING HERE XXX'))

    main(args, dataset)

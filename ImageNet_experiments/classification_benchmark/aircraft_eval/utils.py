import torch
import os
import torchvision.models as models
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


def load_pretrained_model(model, pretrained):
    #loading from mocov2 pretrained models
    if os.path.isfile(pretrained):
        print("=> loading pretrained from checkpoint {}".format(pretrained))
        checkpoint = torch.load(pretrained, map_location="cpu")
        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        #print(state_dict.keys())
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            elif k.startswith('encoder_q') and not k.startswith('encoder_q.fc'):
                state_dict[k[len("encoder_q."):]] = state_dict[k]
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=False)
        #assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        print(set(msg.missing_keys))
        print("=> loaded pre-trained model '{}'".format(pretrained))
    #loading from ImageNet pretrained models
    elif pretrained in model_names:
        print("=> loading pretrained from ImageNet pretrained {}".format(pretrained))
        checkpoint = models.__dict__[pretrained](pretrained=True)
        state_dict = checkpoint.state_dict()
        state_dict.pop('fc.weight')
        state_dict.pop('fc.bias')
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        print("=> loaded pretrained from ImageNet pretrained {}".format(pretrained))
    else:
        print("=> NOT load pretrained")
    return model
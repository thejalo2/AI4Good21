import torch


CUDA_THINGS = (torch.Tensor, torch.nn.utils.rnn.PackedSequence)


# push things to a device
def things2dev(obj, dev):
    if isinstance(obj, CUDA_THINGS):
        return obj.to(dev)
    if isinstance(obj, list):
        return [things2dev(x, dev) for x in obj]
    if isinstance(obj, tuple):
        return tuple(things2dev(list(obj), dev))
    if isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = things2dev(v, dev)
    return obj


def save_model(model, save_path):
    sd = model.state_dict()
    sd = things2dev(sd, 'cpu')
    torch.save(sd, save_path)
    print('Saved model to: %s' % save_path)


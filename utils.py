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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
from models.protonet_model.mlp import MLPProto


def get_model(P, modelstr):

    if modelstr == 'mlp':
        if 'protonet' in P.mode:
            if P.dataset == 'income':
                model = MLPProto(105, 1024, 1024)
    else:
        raise NotImplementedError()

    return model

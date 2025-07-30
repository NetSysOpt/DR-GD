def method_default_args(prob_type):
    defaults = {}
    defaults['simpleVar'] = 100
    defaults['simpleIneq'] = 50
    defaults['simpleEq'] = 50
    defaults['simpleEx'] = 120
    defaults['earlyStop'] = 20

    if 'simple' in prob_type or 'port' in prob_type:
        defaults['epochs'] = 1000
        defaults['batchSize'] = 2
        defaults['lr'] = 3e-4                                                                                                                                                                                              
        defaults['hiddenSize'] = 128
        defaults['embSize'] = 128
        defaults['numLayers'] = 4
        defaults['lambda1'] = 1
        defaults['etaBase'] = 0.1
    elif 'qplib' in prob_type:
        defaults['epochs'] = 1000
        defaults['batchSize'] = 2
        defaults['lr'] = 1e-5
        defaults['hiddenSize'] = 128
        defaults['embSize'] = 128
        defaults['numLayers'] = 4
        defaults['lambda1'] = 1e2 
        defaults['etaBase'] = 0.05
    else:
        raise NotImplementedError

    return defaults


def l2ws_default_args(prob_type):
    defaults = {}
    defaults['simpleVar'] = 100
    defaults['simpleIneq'] = 50
    defaults['simpleEq'] = 50
    defaults['simpleEx'] = 120
    defaults['earlyStop'] = 50

    if 'simple' in prob_type or 'port' in prob_type:
        defaults['epochs'] = 1000
        defaults['batchSize'] = 2
        defaults['lr'] = 1e-5
        defaults['hiddenSize'] = [500, 500, 500]

        defaults['embSize'] = None
        defaults['numLayers'] = None
        defaults['lambda1'] = None
        defaults['etaBase'] = None
        defaults['supervised'] = False
        defaults['train_unrolls'] = 0
    else:
        raise NotImplementedError

    return defaults
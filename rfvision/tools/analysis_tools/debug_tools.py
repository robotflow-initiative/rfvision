def count_paras(model):
    total_para = 0
    for i in model.state_dict().values():
        total_para += i.numel()
    return total_para
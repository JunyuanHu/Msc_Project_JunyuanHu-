import os
import sys

print('Python %s on %s' % (sys.version, sys.platform))
sys.path.extend(['../'])
import pickle
from train_val_model import *
from parser_args import *
from log import TimerBlock, IteratorTimer
from data_choose import data_choose, init_seed
from model_choose import model_choose
from loss_choose import loss_choose

with TimerBlock("Good Luck") as block:
    # params
    args = parser_args(block)
    init_seed(1)

    data_loader_train, data_loader_val = data_choose(args, block)
    global_step, start_epoch, model, optimizer_dict = model_choose(args, block)
    loss_function = loss_choose(args, block)

    model.cuda()
    model.eval()

    print('Validate')
    loss, acc, score_dict, all_pre_true, wrong_path_pre_true = val_classifier(data_loader_val, model,
                                                                                              loss_function, 0, args,
                                                                                              None)
    save_score = os.path.join(args.model_saved_name, 'score.pkl')
    with open(save_score, 'wb') as f:
        pickle.dump(score_dict, f)
    with open(args.model_saved_name + '/all_pre_true.txt', 'w') as f:
        f.writelines(all_pre_true)
    with open(args.model_saved_name + '/wrong_path_pre_true.txt', 'w') as f:
        f.writelines(wrong_path_pre_true)
    print('Final: {}'.format(float(acc)))

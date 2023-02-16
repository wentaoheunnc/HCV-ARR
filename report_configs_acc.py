import os

def init_configs_acc():
    configs_acc = {"center_single": [0, 0],
                  "distribute_four": [0, 0],
                  "distribute_nine": [0, 0],
                  "in_center_single_out_center_single": [0, 0],
                  "in_distribute_four_out_center_single": [0, 0],
                  "left_center_single_right_center_single": [0, 0],
                  "up_center_single_down_center_single": [0, 0],
                  }
    return configs_acc


def update_configs_acc(configs_acc, model_output, target, data_file):
    acc_one = model_output.data.max(1)[1] == target
    for i in range(model_output.shape[0]):
        config = data_file[i].split('\\' if os.name == 'nt' else '/')[0]
        configs_acc[config][0] += acc_one[i].item()
        configs_acc[config][1] += 1

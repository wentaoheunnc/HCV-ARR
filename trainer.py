import os
import pickle

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import criteria
from report_configs_acc import init_configs_acc, update_configs_acc

torch.backends.cudnn.benchmark = True

from checkpoint import load_state_dicts


def renormalize(images):
    return (images / 255 - 0.5) * 2


class Trainer:
    def __init__(self, args):
        self.args = args
        self.args.cuda = torch.cuda.is_available()
        torch.cuda.set_device(self.args.device)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if self.args.cuda:
            torch.cuda.manual_seed(self.args.seed)

        test_path = os.path.join('.', 'results/test/')
        self.save_path = os.path.join(test_path, 'save')
        self.log_path = os.path.join(test_path, 'log')

        if os.path.isdir(test_path):
            print(f'Removing existing save directory at {test_path}')
            import shutil
            shutil.rmtree(test_path)

        print(f'Creating new save directory at {test_path}')
        os.makedirs(self.save_path)
        os.makedirs(self.log_path)

        import yaml
        cfg_file = os.path.join(self.save_path, 'cfg.yml')
        with open(cfg_file, 'w') as f:
            yaml.dump(args.__dict__, f, default_flow_style=False)

        print('Loading datasets')
        from data.data_utils import get_data

        self.testloader = get_data(self.args.path, self.args.dataset, self.args.img_size,
                                   dataset_type="test", subset=self.args.subset,
                                   batch_size=self.args.batch_size, drop_last=True, num_workers=self.args.num_workers,
                                   ratio=self.args.ratio, shuffle=False)

        print('Building model')
        params = args.__dict__

        assert args.model_name == 'hcvarr'
        from networks.hcvarr import HCVARR
        self.model = HCVARR(dropout=args.dropout, levels=args.levels, do_contrast=args.contrast)


        self.model = torch.nn.DataParallel(self.model)
        load_state_dicts(self.model).load('./models/model.pth')

        if self.args.cuda:
            self.model.cuda()

        self.optimizer = optim.Adam([param for param in self.model.parameters() if param.requires_grad],
                                    self.args.lr, betas=(self.args.beta1, self.args.beta2), eps=self.args.epsilon,
                                    weight_decay=self.args.wd)

        self.criterion = lambda x, y: criteria.contrast_loss(x, y)


    def test(self, epoch):
        self.model.eval()

        counter = 0
        loss_avg = 0.0
        loss_meta_avg = 0.0
        acc_avg = 0.0

        configs_acc = init_configs_acc()

        for batch_data in tqdm(self.testloader, f'Test epoch {epoch}'):
            counter += 1

            image, target, data_file = batch_data
            image = renormalize(image)

            image = image.cuda()
            target = target.cuda()

            with torch.no_grad():

                model_output = self.model(image)

            loss = self.criterion(model_output, target)
            loss_avg += loss.item()
            acc = criteria.calculate_acc(model_output, target)
            acc_avg += acc.item()

            if configs_acc is not None:
                update_configs_acc(configs_acc, model_output, target, data_file)

        if counter > 0:
            print("Epoch {}, Test  Avg Loss: {:.6f}, Test  Avg Acc: {:.4f}".format(
                epoch, loss_avg / float(counter), acc_avg / float(counter)))

        if configs_acc is not None:
            for key in configs_acc.keys():
                if configs_acc[key] is not None:
                    if configs_acc[key][1] > 0:
                        configs_acc[key] = float(configs_acc[key][0]) / configs_acc[key][1] * 100
                    else:
                        configs_acc[key] = None

        return loss_avg / float(counter), acc_avg / float(counter), configs_acc


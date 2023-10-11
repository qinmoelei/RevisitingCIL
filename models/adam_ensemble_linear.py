import logging
import numpy as np
import torch
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import (
    IncrementalNet,
    SimpleCosineIncrementalNet,
    MultiBranchwithfcCosineIncrementalNet,
    FourBranchwithfcCosineIncrementalNet,
    SimpleVitNet_linear,
    SimpleVitNet,
)
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from types import SimpleNamespace
from convs.vision_transformer_adapter_LoRA import (
    vit_base_patch16_224_in21k_adapter_lora,
)
from convs.vpt import build_promptmodel
import timm
from utils.loss import (
    FocalLoss,
    LabelSmoothingLoss,
    kl_divergence_loss,
    huber_loss,
    hinge_loss,
)
from convs.vision_transformer_ssf import *
# tune the model at first session with adapter, and then conduct simplecil.
num_workers = 8


# 现在的这个结果用adapter_lora+ssf+vpt
# 现在调整一个adapater的参数 看效果如何
class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = (
            args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        )
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        self.args_args = SimpleNamespace(**self.args)
        _network_1 = SimpleVitNet_linear(args, True)
        _network_1.convnet = vit_base_patch16_224_in21k_adapter_lora(
            tuning_config=self.args_args
        )
        _network_1.convnet.out_dim = 768
        self._network_list = []
        self._network_list.append(_network_1)

        # _network_2 = SimpleVitNet(args, True)
        # _network_2.convnet = timm.create_model(
        #             "vit_base_patch16_224_in21k_ssf", pretrained=True, num_classes=0
        #         ).eval()
        # _network_2.convnet.out_dim = 768
        # self._network_list.append(_network_2)

        _network_3 = SimpleVitNet_linear(args, True)
        _network_3.convnet = build_promptmodel(
            modelname="vit_base_patch16_224_in21k",
            Prompt_Token_num=args.get("prompt_token_num", 10),
            VPT_type=args.get("vpt_type", "Shallow"),
        )
        prompt_state_dict = _network_3.convnet.obtain_prompt()
        _network_3.convnet.load_prompt(prompt_state_dict)
        _network_3.convnet.out_dim = 768
        self._network_list.append(_network_3)
        self._network = self._network_list[0]

    def after_task(self):
        self._known_classes = self._total_classes

    def replace_fc(self, trainloader, model, args):
        # replace fc.weight with the embedding average of train data
        model = model.eval()
        embedding_list = []
        label_list = []
        # data_list=[]
        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                (_, data, label) = batch
                data = data.cuda()
                label = label.cuda()
                embedding = model(data)["features"]
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(self.train_dataset.labels)
        proto_list = []
        for class_index in class_list:
            # print('Replacing...',class_index)
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto
        return model

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        if self._cur_task != 0:
            self._network.update_fc(self._total_classes)
        else:
            for network in self._network_list:
                network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        if len(self._multiple_gpus) > 1:
            print("Multiple GPUs")
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader, self.train_loader_for_protonet)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader, train_loader_for_protonet):
        if self._cur_task != 0:
            self._network.to(self._device)

        if self._cur_task == 0:
            for i in range(len(self._network_list)):
                network=self._network_list[i]
                network.to(self._device)
                # show total parameters and trainable parameters
                total_params = sum(p.numel() for p in network.parameters())
                print(f"{total_params:,} total parameters.")
                total_trainable_params = sum(
                    p.numel() for p in network.parameters() if p.requires_grad
                )
                print(f"{total_trainable_params:,} training parameters.")
                if total_params != total_trainable_params:
                    for name, param in network.named_parameters():
                        if param.requires_grad:
                            print(name, param.numel())
                if self.args["optimizer"][i] == "sgd":
                    optimizer = optim.SGD(
                        network.parameters(),
                        momentum=0.9,
                        lr=self.init_lr[i],
                        weight_decay=self.weight_decay[i],
                    )
                elif self.args["optimizer"][i] == "adam":
                    optimizer = optim.AdamW(
                        network.parameters(),
                        lr=self.init_lr[i],
                        weight_decay=self.weight_decay[i],
                    )
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.args["tuned_epoch"][i], eta_min=self.min_lr[i]
                )
                self._init_train(
                    network, train_loader, test_loader, optimizer, scheduler,i
                )
            self.construct_dual_branch_network()
        else:
            pass
        self.replace_fc(train_loader_for_protonet, self._network, None)

    def construct_dual_branch_network(self):
        network = FourBranchwithfcCosineIncrementalNet(self.args, True)
        network.construct_dual_branch_network(
            self._network_list, ["SimpleVitNet_linear", "SimpleVitNet_linear", "SimpleVitNet_linear"]
        )
        self._network = network.to(self._device)

    def _init_train(self, network, train_loader, test_loader, optimizer, scheduler,num):
        prog_bar = tqdm(range(self.args["tuned_epoch"][num]))
        for _, epoch in enumerate(prog_bar):
            network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                network.train()
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = network(inputs)["logits"]
                if self.args.get("use_KL_loss", False):
                    loss = kl_divergence_loss(logits, targets)
                elif self.args.get("use_huber_loss", False):
                    loss = huber_loss(logits, targets)
                elif self.args.get("use_hinge_loss", False):
                    loss = hinge_loss(logits, targets)
                else:
                    loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            test_acc = self._compute_accuracy(network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args["tuned_epoch"][num],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

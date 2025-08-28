import os
import sys
import yaml
import argparse
import os.path as osp
import logging.config
from utils import os_utils
from utils import log_utils
from utils import path_utils
args = None

class Config:
    def __init__(self):
        parser = argparse.ArgumentParser(description="Knowledge Evolution Training Approach")
        # Pre-training method selection
        parser.add_argument(
            "--pretrain_method",
            type=str,
            default=None,
            choices=["moco", "dino", "simsiam", None],
            help="Pre-training method to use: moco, simclr, simsiam, or None for no pre-training"
        )
        # Pre-trained weights paths
        parser.add_argument(
            "--big_model_weights_path",
            type=str,
            default="/content/imagenet_resnet50.pth",
            help="Path to ImageNet pre-trained weights for big model"
        )
        parser.add_argument(
            "--moco_weights_path",
            type=str,
            default="/content/moco_v1_200ep_pretrain.pth.tar",
            help="Path to MoCo pre-trained weights"
        )
        parser.add_argument(
            "--dino_weights_path",
            type=str,
            default=" /content/dino_resnet18_cifar10.pth",
            help="Path to Dino pre-trained weights"
        )
        parser.add_argument(
            "--simsiam_weights_path",
            type=str,
            default="/content/simsiam-cifar10-252e1tvw-ep=999.pth",
            help="Path to SimSiam pre-trained weights"
        )
        parser.add_argument(
            "--use_dino",
            action="store_true",
            default=False,
            help="Use Dino self-supervised pre-training (sets pretrain_method to dino if True)"
        )
        parser.add_argument(
            "--use_simsiam",
            action="store_true",
            default=False,
            help="Use SimSiam self-supervised pre-training (sets pretrain_method to byol if True)"
        )
        # KDloss
        parser.add_argument(
            "--target_samples_per_class_ratio",
            type=float,
            default=0.5,  # Default value (50% of samples per class)
            help="Ratio of target samples per class to subsample the dataset (0.0 to 1.0, e.g., 0.1 for 10%)"
        )
        parser.add_argument(
            '--kd_temperature', 
            type=float, 
            default=4.0, 
            help='Temperature for KD loss')
        parser.add_argument(
            '--kd_alpha', 
            type=float, 
            default=0.7, 
            help='Weight for KD loss')

        parser.add_argument(
            '--kd_beta', 
            type=float, 
            default=0.3, 
            help='Weight for KD loss')    
        ## Teacher model 
        parser.add_argument(
            '--big_arch', 
            default='Split_densenet121',
            type=str, 
          help='teacher model architecture')
        parser.add_argument(
          '--fine_tune_teacher', 
          action='store_true', 
          help='Fine-tune teacher model before KD'
          )
        parser.add_argument(
            "--mask_schedule",
            type=str,
            default='None',
            choices=["start_gen", "mid_gen", "every_epoch", "every_n_epochs", 'None'],
            help="When to compute the SNIP mask: start of generation, middle of generation, every epoch, or every n epochs"
        )
        parser.add_argument(
            "--mask_n_epochs",
            type=int,
            default=10,
            help="If mask_schedule is every_n_epochs, compute mask every n epochs"
        )
        parser.add_argument(
            "--epochs_fine_tune",
            default=20,
            type=int,
            help="Number of epochs to fine-tune the teacher model"
        )
        # Arguments for SNIP, SNIPit, and SNAPit
        parser.add_argument(
            "--prune_criterion",
            type=str,
            default="SNIPit",
            choices=["SNIP", "SNIPit", "SNAPit", "CNIPit"],
            help="Pruning criterion to use: SNIP (single-shot), SNIPit (iterative unstructured), SNAPit (iterative structured)"
        )
        parser.add_argument(
            "--snip",
            action="store_true",
            help="Use SNIP pruning (sets prune_criterion to SNIP)"
        )
        parser.add_argument(
            "--snipit",
            action="store_true",
            help="Use SNIPit pruning (sets prune_criterion to SNIPit)"
        )
        parser.add_argument(
            "--snapit",
            action="store_true",
            help="Use SNAPit pruning (sets prune_criterion to SNAPit)"
        )
        parser.add_argument(
            "--cnipit",
            action="store_true",
            help="Use CNIPit pruning (sets prune_criterion to CNIPit)"
        )
        parser.add_argument(
            "--pruning_steps",
            type=int,
            default=5,
            help="Number of iterative pruning steps for SNIPit and SNAPit"
        )
        parser.add_argument(
            "--structured",
            action="store_true",
            default=False,
            help="Use structured pruning (channels) for SNAPit"
        )
        parser.add_argument(
            "--pruning_start",
            type=float,
            default=0.5,
            help="Starting sparsity for SNAPit pruning schedule (0.0 to 1.0)"
        )
        # General Config
        parser.add_argument(
            "--grasp",
            action="store_true",
            default=False,
            help="Enable GRASP pruning method"
        )
        parser.add_argument(
            "--data", help="path to dataset base directory", default="/home/datasets"
        )
        parser.add_argument("--optimizer", help="Which optimizer to use", default="sgd")
        parser.add_argument(
            "--set", type=str, default="CUB200",
            choices=['Flower102Pytorch', 'Flower102', 'CUB200', 'Aircrafts', 'Dog120', 'MIT67',
                     'CIFAR10', 'CIFAR10val', 'CIFAR100', 'CIFAR100val', 'tinyImagenet_full', 'tinyImagenet_val',
                     'CUB200_val', 'Dog120_val', 'MIT67_val', 'imagenet']
        )
        parser.add_argument(
            "-a", "--arch", metavar="ARCH", default="Split_ResNet18", help="model architecture",
            choices=['Split_ResNet18', 'Split_ResNet18Norm', 'Split_ResNet34', 'Split_ResNet50', 'Split_ResNet50Norm',
                     'Split_ResNet101', 'Split_googlenet',
                     'Split_densenet121', 'Split_densenet161', 'Split_densenet169', 'Split_densenet201', 'vgg11', 'vgg11_bn', 'resnet18', 'Split_Xception']
        )
        parser.add_argument(
            "--config_file", help="Config file to use (see configs dir)", default=None
        )
        parser.add_argument(
            "--log-dir", help="Where to save the runs. If None use ./runs", default=None
        )
        parser.add_argument(
            '--evolve_mode', default='rand', choices=['rand', 'zero'],
            help='How to initialize the reset-hypothesis.'
        )
        parser.add_argument(
            "-t",
            "--num_threads",
            default=0,
            type=int,
            metavar="N",
            help="number of data loading workers (default: 10)",
        )
        parser.add_argument(
            "--epochs",
            default=100,
            type=int,
            metavar="N",
            help="number of total epochs to run",
        )
        parser.add_argument(
            "--eval_intermediate_tst",
            default=10,
            type=int,
            help="eval tst every N epochs instead of evaluating at the end of each generation",
        )
        parser.add_argument(
            "--start-epoch",
            default=None,
            type=int,
            metavar="N",
            help="manual epoch number (useful on restarts)",
        )
        parser.add_argument(
            "-lb",
            "--linear_batch_size",
            default=32,
            type=int,
            metavar="N",
            help="mini-batch size",
        )
        parser.add_argument(
            "-b",
            "--batch_size",
            default=128,
            type=int,
            metavar="N",
            help="mini-batch size",
        )
        parser.add_argument(
            "--lr",
            "--learning-rate",
            default=0.001,
            type=float,
            metavar="LR",
            help="initial learning rate",
            dest="lr",
        )
        parser.add_argument(
            "--warmup_length", default=5, type=int, help="Number of warmup iterations"
        )
        parser.add_argument(
            "--momentum", default=0.9, type=float, metavar="M", help="momentum"
        )
        parser.add_argument(
            "--wd",
            "--weight_decay",
            default=1e-4,
            type=float,
            metavar="W",
            help="weight decay (default: 1e-4)",
            dest="weight_decay",
        )
        parser.add_argument(
            "-p",
            "--print-freq",
            default=20,
            type=int,
            metavar="N",
            help="print frequency",
        )
        parser.add_argument('--samples_per_class', default=1, type=int,
                            help='Number of samples per class inside a mini-batch.')
        parser.add_argument('--alpha', default=8, type=float,
                            help='Scaling Parameter setting')
        parser.add_argument('--warm', default=5, type=int,
                            help='Warmup training epochs')
        parser.add_argument(
            "--resume",
            default="",
            type=str,
            metavar="PATH",
            help="path to latest checkpoint (default: none)",
        )
        parser.add_argument(
            "--reset_layer_name",
            default=None,
            type=str,
            help="layer to start resetting (LLF)",
        )
        parser.add_argument(
            "--pretrained",
            dest="pretrained",
            default=None,
            type=str,
            help="use pre-trained model",
        )
        parser.add_argument(
            "--init_path",
            default='/data/output-ai/',
            type=str,
            help="use pre-trained model",
        )
        parser.add_argument(
            "--seed", default=42, type=int, help="need to set fix_seed = True to take effect"
        )
        parser.add_argument(
            "--fix_seed",default=True, action="store_true", help="Fix random seed"
        )
        parser.add_argument(
            "--gpu",
            default='0',
            type=int,
            help="Which GPUs to use?",
        )
        parser.add_argument(
            "--test_interval", default=10, type=int, help="Eval on tst/val split every ? epochs"
        )
        # Learning Rate Policy Specific
        parser.add_argument(
            "--lr_policy", default="long_cosine_lr", help="Policy for the learning rate."
        )
        parser.add_argument(
            "--multistep-lr-adjust", default=1, type=int, help="Interval to drop lr"
        )
        parser.add_argument("--multistep-lr-gamma", default=0.97, type=int, help="Multistep multiplier")
        parser.add_argument(
            "--name", default=None, type=str, help="Experiment name to append to filepath"
        )
        parser.add_argument(
            "--log_file", default='train_log.txt', type=str, help="Experiment name to append to filepath"
        )
        parser.add_argument(
            "--save_every", default=-1, type=int, help="Save every ___ epochs"
        )
        parser.add_argument(
            "--num_generations", default=10, type=int, help="Number of training generations"
        )
        parser.add_argument('--lr-decay-step', default=10, type=int, help='Learning decay step setting')
        parser.add_argument('--lr-decay-gamma', default=0.5, type=float, help='Learning decay gamma setting')
        parser.add_argument('--gamma', default=0, type=float, help='fishermatrix update')
        parser.add_argument(
            "--split_rate",
            default=0.5,
            help="What is the split-rate for the split-network weights?",
            type=float,
        )
        parser.add_argument(
            "--sparsity",
            default=0.5,
            help="What is the split-rate for the split-network weights?",
            type=float,
        )
        parser.add_argument(
            "--bias_split_rate",
            default=0.5,
            help="What is the bias split-rate for the split-network weights?",
            type=float,
        )
        parser.add_argument(
            "--slim_factor",
            default=1.0,
            help="This variable is used to extract a slim network from a dense network. "
                 "It is initialized using the split_rate of the trained dense network.",
            type=float,
        )
        parser.add_argument(
            "--split_mode",
            default="wels",
            choices=['kels', 'wels'],
            help="how to split the binary mask",
        )
        parser.add_argument(
            "--conv_type", type=str, default='SplitConv', help="SplitConv | DenseConv"
        )
        parser.add_argument(
            "--linear_type", type=str, default='SplitLinear', help="SplitLinear | DenseLinear"
        )
        parser.add_argument("--mode", default="fan_in", help="Weight initialization mode")
        parser.add_argument(
            "--nonlinearity", default="relu", help="Nonlinearity used by initialization"
        )
        parser.add_argument("--bn_type", default='SplitBatchNorm', help="BatchNorm type",
                            choices=['NormalBatchNorm', 'NonAffineBatchNorm', 'SplitBatchNorm'])
        parser.add_argument(
            "--init", default="kaiming_normal", help="Weight initialization modifications"
        )
        parser.add_argument(
            "--no-bn-decay", action="store_true", default=False, help="No batchnorm decay"
        )
        parser.add_argument(
            "--scale-fan", action="store_true", default=False, help="scale fan"
        )
        parser.add_argument("--save_model", action="store_true", default=True, help="save model checkpoints")
        parser.add_argument("--cs_kd", action="store_true", default=False, help="Enable Cls_KD")
        parser.add_argument("--reset_hypothesis", action="store_true", default=False, help="Reset mask across generations")
        parser.add_argument("--reverse_reset", action="store_true", default=False, help="reset layers BEFORE reset layer name")
        parser.add_argument("--reverse_freeze", action="store_true", default=False, help="freeze reset layers")
        parser.add_argument("--freeze_non_reset", action="store_true", default=False, help="freeze non-reset layers")
        parser.add_argument(
            "--use-noisy-logit",
            action="store_true",
            default=False,
            help="Add noise to logits during training"
        )
        parser.add_argument("--no_wandb", action="store_true", default=False, help="no wandb")
        parser.add_argument("--group_vars", type=str, nargs='+', default="", help="variables used for grouping in wandb")
        parser.add_argument("--no_rescale_weights", type=bool, default=True)
        parser.add_argument("--no_normalize_LW", type=bool, default=True)
        parser.add_argument(
            "--supervised", action="store_true", default=False, help="supervised"
        )
        parser.add_argument(
            "--last_layer", action="store_true", default=True, help="supervised"
        )
        parser.add_argument(
            "--clf",
            type=str,
            default="sgd",
            choices=["sgd", "knn", "lbfgs"],
            help="classifier for test.py",
        )
        parser.add_argument(
            "--method",
            type=str,
            default="-",
            choices=["llf"],
            help="classifier for test.py",
        )
        parser.add_argument(
            "--label_smoothing",
            type=float,
            help="Label smoothing to use, default 0.0",
            default=0.1,
        )
        parser.add_argument(
            "--slow_lr_multiplier",
            type=float,
            help="0 means freeze important weights, otherwise slow learning rate",
            default=0.0,
        )
        parser.add_argument(
            "--trainer", type=str, default="default"
        )
        parser.add_argument(
            "--reset_bn", action="store_true", default=False, help="reset bn each generation"
        )
        parser.add_argument(
            "--no_reset", action="store_true", default=False, help="do not reset weight each generation"
        )
        parser.add_argument(
            "--reset_important_weights", action="store_true", default=False, help="do not reset weight each generation"
        )
        parser.add_argument(
            "--weight_pruning", action="store_true", default=False, help="do not reset weight each generation"
        )
        parser.add_argument(
            "--grow_sparcity_gen", action="store_true", default=False, help="increase sparcity as generation progress"
        )
        parser.add_argument(
            "--freeze_fisher", action="store_true", default=False, help="increase sparcity as generation progress"
        )
        parser.add_argument(
            "--deficit_epo", type=int, default=100,
            help='deficit Number of epochs'
        )
        parser.add_argument(
            "--use_deficit", action="store_true", default=False, help="deficit training"
        )
        parser.add_argument(
            "--use_pretrain", action="store_true", default=False, help="deficit training"
        )
        parser.add_argument(
            "--weights",
            default="freeze",
            type=str,
            choices=("finetune", "freeze"),
            help="finetune or freeze resnet weights",
        )
        parser.add_argument(
            "--eval_linear", action="store_true", default=False, help="diff aug"
        )
        parser.add_argument(
            "--use_train_val", action="store_true", default=False, help="diff aug"
        )
        parser.add_argument(
            "--use_val", action="store_true", default=False, help="diff aug"
        )

        self.parser = parser

    def parse(self, args):
        self.cfg = self.parser.parse_args(args)

        # Map pruning criteria based on flags
        if self.cfg.snip:
            self.cfg.prune_criterion = "SNIP"
        elif self.cfg.snipit:
            self.cfg.prune_criterion = "SNIPit"
        elif self.cfg.snapit:
            self.cfg.prune_criterion = "SNAPit"

        if self.cfg.set == 'Flower102' or self.cfg.set == 'Flower102Pytorch':
            self.cfg.num_cls = 102
            self.cfg.eval_tst = True
        elif self.cfg.set == 'CUB200':
            self.cfg.num_cls = 200
            self.cfg.eval_tst = True
        elif self.cfg.set == 'imagenet':
            self.cfg.num_cls = 100
            self.cfg.eval_tst = False
        elif self.cfg.set in ['tinyImagenet_full', 'tinyImagenet_val']:
            self.cfg.num_cls = 200
            self.cfg.eval_tst = False
        elif self.cfg.set == 'Dog120':
            self.cfg.num_cls = 120
            self.cfg.eval_tst = False
        elif self.cfg.set == 'MIT67':
            self.cfg.num_cls = 67
            self.cfg.eval_tst = False
        elif self.cfg.set == 'Aircrafts':
            self.cfg.num_cls = 100
            self.cfg.eval_tst = True
        elif self.cfg.set == 'CIFAR10' or self.cfg.set == 'CIFAR10val':
            self.cfg.num_cls = 10
            self.cfg.eval_tst = True
        elif self.cfg.set == 'CIFAR100' or self.cfg.set == 'CIFAR100val':
            self.cfg.num_cls = 100
            self.cfg.eval_tst = True
        else:
            raise NotImplementedError('Invalid dataset {}'.format(self.cfg.set))

        if self.cfg.cs_kd:
            self.cfg.samples_per_class = 3
            self.cfg.label_smoothing = 0

        self.cfg.bias_split_rate = self.cfg.split_rate

        self.cfg.group_name = self.cfg.name
        self.cfg.name = 'SPLT_CLS_{}_{}_cskd{}_smth{}_k{}_G{}_e{}_ev{}_hReset{}_sm{}_{}_seed{}/'.format(
            self.cfg.set, self.cfg.arch, self.cfg.cs_kd, self.cfg.label_smoothing, self.cfg.split_rate,
            self.cfg.num_generations, self.cfg.epochs, self.cfg.evolve_mode, self.cfg.reset_hypothesis,
            self.cfg.split_mode, self.cfg.name, self.cfg.seed
        )

        self.cfg.exp_dir = osp.join(path_utils.get_checkpoint_dir(), self.cfg.name)

        os_utils.touch_dir(self.cfg.exp_dir)
        log_file = os.path.join(self.cfg.exp_dir, self.cfg.log_file)
        logging.config.dictConfig(log_utils.get_logging_dict(log_file))
        self.cfg.logger = logging.getLogger('KE')

        return self.cfg

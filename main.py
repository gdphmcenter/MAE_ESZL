#’TODO 6‘Search requires data replacement

import torch
from torch.utils.data import DataLoader

import argparse
from tqdm import tqdm
from trainer import Trainer
from datautils import ZSLDataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='datasets_CWRU_0123')
    parser.add_argument('--n_epochs', type=int, default=150) #
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--latent_dim', type=int, default=64) #
    parser.add_argument('--lr', type=float, default=1e-4)#1e-4
    parser.add_argument('--gzsl', action='store_true', default=False)#
    parser.add_argument('--da', action='store_true', default=True)
    parser.add_argument('--ca', action='store_true', default=True)
    parser.add_argument('--support', action='store_true', default=True)
    #cen add
    parser.add_argument('--resume', action='store_true', default=False)#
    ###model for mae_model=====
    # mae_Model parameters
    # parser.add_argument('--model',  type=str,default='vit_large_patch16',help='Name of model to train')#large
    parser.add_argument('--global_pool', action='store_true', default=False)
    parser.add_argument('--maepath', default=r"checkpointlin6_model-399.pth", help='finetune from checkpoint')
    # parser.add_argument('--maepath_N', default=r"./checkpointlin1-0.pth", help='finetune from checkpoint')
    # parser.add_argument('--mask_ratio', default=0.35, type=float, help='number of the classification types')
    parser.add_argument('--dist_eval', action='store_true', default=False, help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--nb_classes', default=40, type=int, help='number of the classification types')
    parser.add_argument('--eval', type=bool, default=False)
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT', help='Drop path rate (default: 0.1)')
    parser.add_argument('--remin_head', type=bool, default=False)
    parser.add_argument('--finetune', type=bool, default=False)
    return parser.parse_args()

def main():
    # setup parameters for trainer

    args = parse_args()
    if args.dataset == 'awa2' or args.dataset == 'awa1':
        x_dim = 2048
        attr_dim = 85
        n_train = 40
        n_test = 10
    elif args.dataset == 'cub':
        x_dim = 2048
        attr_dim = 312
        n_train = 150
        n_test = 50
    elif args.dataset == 'sun':
        x_dim = 2048
        attr_dim = 102
        n_train = 645
        n_test = 72
    ###

    elif args.dataset == 'datasets_CWRU_0123':
        x_dim = 60416#768#TODO
        attr_dim = 10#256#512#10#300#13#13#10
        n_train = 37#TODO
        n_test = 3
    elif args.dataset == 'datasets_box':
        x_dim = 1024
        attr_dim = 13
        n_train = 25
        n_test = 3
    else:
        raise NotImplementedError

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    params = {
        'batch_size': args.batch_size,
        'shuffle': True,
        'num_workers': 0,
    }

    train_dataset = ZSLDataset(args.dataset, n_train, n_test, train=True, gzsl=args.gzsl)
    train_generator = DataLoader(train_dataset, **params,drop_last=True)

    layer_sizes = {
        'x_enc': 1560,
        'x_dec': 1660,
        'c_enc': 1450,
        'c_dec': 660
    }

    kwargs = {
        'gzsl': args.gzsl,
        'use_da': args.da,
        'use_ca': args.ca,
        'use_support': args.support,
    }

    train_agent = Trainer(
        args,device, args.dataset, x_dim, attr_dim, args.latent_dim,
        n_train, n_test, args.lr, layer_sizes, **kwargs
    )

    # load previous models, if any
    ##cen add
    if args.resume == True:
        vae_start_ep = train_agent.load_models()  # 1.Load the trained parameters of the model
    else:
        vae_start_ep = 0

    print('Training the VAE')
    for ep in range(vae_start_ep + 1, args.n_epochs + 1):
        # train the VAE
        vae_loss = 0.0
        da_loss, ca_loss = 0.0, 0.0
        train_generator = tqdm(train_generator)
        print("Train MVAE for the {} th epoch".format(ep))
        for idx, (img_features, attr, label_idx) in enumerate(train_generator):
            losses = train_agent.fit_VAE(args, img_features, attr, label_idx, ep)#2.train model MVAE
            # print(losses[0])
            vae_loss  += losses[0]
            da_loss   += losses[1]
            ca_loss   += losses[2]

        n_batches = idx + 1
        print("[MVAE Training] Losses for epoch: [%3d] : " \
                "%.4f(V), %.4f(D), %.4f(C)" \
                %(ep, vae_loss/n_batches, da_loss/n_batches, ca_loss/n_batches))

        # save VAE after each epoch
        train_agent.save_VAE(ep)
    ###注释end====
    seen_dataset = None
    if args.gzsl:
        seen_dataset = train_dataset.gzsl_dataset

    syn_dataset = train_agent.create_syn_dataset(
            train_dataset.test_classmap, train_dataset.attributes, seen_dataset)##1,5,9#40*13#None(GZSL        #ouput # Z[i], test_cls(ori_label, idx
    #train_dataset.test_classmap
    final_dataset = ZSLDataset(args.dataset, n_train, n_test,
            train=True, gzsl=args.gzsl, synthetic=True, syn_dataset=syn_dataset)#ouput syn_dataset img_feature, label_attr, label_idx
    final_train_generator = DataLoader(final_dataset, **params)

    # compute accuracy on test dataset
    test_dataset = ZSLDataset(args.dataset, n_train, n_test, False, args.gzsl)  #
    test_generator = DataLoader(test_dataset, **params,drop_last=True)

    best_acc = 0.0
    for ep in range(1, 1 + 2*args.n_epochs):#TODO 6 1+2*args.n_epochs
        # train final classifier
        total_loss = 0
        final_train_generator = tqdm(final_train_generator)
        print("Train the classifier with the {} th epoch".format(ep))
        for idx, (features, _, label_idx) in enumerate(final_train_generator):#Z[i], attr, idx
            loss = train_agent.fit_final_classifier(features, label_idx)
            total_loss += loss
        total_loss = total_loss / (idx + 1)
        print('[Final Classifier Training] Loss for epoch: [%3d]: %.3f' % (ep, total_loss))

        ## find accuracy on test data
        if args.gzsl:
            acc_s, acc_u = train_agent.compute_accuracy(test_generator)
            acc = 2 * acc_s * acc_u / (acc_s + acc_u)
            # print(acc, acc_s, acc_u)
        else:
            print("test model")
            acc = train_agent.compute_accuracy(test_generator)

        if acc >= best_acc:
            best_acc = acc
            if args.gzsl:
                best_acc_s = acc_s
                best_acc_u = acc_u
        if args.gzsl:
            print('epoch[%3d] Accuracy: %.3f ==== Seen: [%.3f] -- Unseen[%.3f]' % (ep, acc, acc_s, acc_u))
        else:
            print('epoch[%3d] Accuracy: %.3f' % (ep, best_acc))
    if args.gzsl:
        print('Best Accuracy: %.3f ==== Seen: [%.3f] -- Unseen[%.3f]' %(best_acc, best_acc_s, best_acc_u))
    else:
        print('Best Accuracy: %.3f' % best_acc)

if __name__ == '__main__':
    main()

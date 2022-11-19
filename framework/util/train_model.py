import torch
import cole as cl
import argparse
import os
import pickle
import json
from kornia.augmentation import RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomGrayscale
import torch.nn as nn
from torchvision import transforms

from torchvision.transforms import transforms
from augmentations import RandAugment

cl.set_data_path('./data')




input_size_match = {
    'mnist':[1,28,28],
    'cifar100': [3, 32, 32],
'cifar': [3, 32, 32],
    'cifar10': [3, 32, 32],
    'core50': [3, 128, 128],
'clrs25': [3, 128,128],#[3, 256, 256],
    'min': [3, 84, 84],
    'openloris': [3, 50, 50]
}
transforms_match = {
    'core50': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'clrs25': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'cifar100': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'cifar': transforms.Compose([
        transforms.ToTensor(),
        ]),
    'cifar10': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'mini_imagenet': transforms.Compose([
        transforms.ToTensor()]),
    'openloris': transforms.Compose([
            transforms.ToTensor()])
}
transform_train = transforms.Compose([
RandAugment(6,15),
    # transforms.RandomCrop(32, padding=4),
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
])
#transform_train.transforms.insert(0, RandAugment(2,14))

# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(_CIFAR_MEAN, _CIFAR_STD),
# ])


def maybe_cuda(what, use_cuda=True, **kw):
    """
    Moves `what` to CUDA and returns it, if `use_cuda` and it's available.
        Args:
            what (object): any object to move to eventually gpu
            use_cuda (bool): if we want to use gpu or cpu.
        Returns
            object: the same object but eventually moved to gpu.
    """

    if use_cuda is not False and torch.cuda.is_available():
        what = what.cuda()
    return what
def randaug(args,concat_batch_x,mem_num):



    n, c, w, h = concat_batch_x.shape

    mem_images = [transforms.ToPILImage()(concat_batch_x[i]) for i in range(mem_num)]
    incoming_images = [transforms.ToPILImage()(concat_batch_x[i]) for i in range(mem_num, n)]
    if (args.scraug == "Mem" and mem_num > 0):
        aug_mem = [transform_train(image).reshape([1, c, w, h]) for image in mem_images]
        aug_mem = maybe_cuda(torch.cat(aug_mem, dim=0))
    else:
        aug_mem = concat_batch_x[:mem_num, :, :, :]
    if (args.scraug == "Incoming"):
        aug_incoming = [transform_train(image).reshape([1, c, w, h]) for image in incoming_images]
        aug_incoming = maybe_cuda(torch.cat(aug_incoming, dim=0))
    else:
        aug_incoming = concat_batch_x[mem_num:, :, :, :]
    # if(mem_num>0):
    #     aug_concat_batch_x = aug_mem + aug_incoming
    # else:
    #
    #     aug_concat_batch_x =  aug_incoming

    if (mem_num > 0):
        aug_concat_batch_x = maybe_cuda(torch.cat((aug_mem, aug_incoming), dim=0))
    else:
        aug_concat_batch_x = maybe_cuda(aug_incoming)

    return aug_concat_batch_x


def train(args, data, model):

    c,w,h = input_size_match[args.data]

    scr_transform = nn.Sequential(
        RandomResizedCrop(size=(w,h),
                          scale=(0.2, 1.)),
        RandomHorizontalFlip(),
        ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
        RandomGrayscale(p=0.2)

    )

    train_loader = cl.CLDataLoader(data.train, bs=args.bs, shuffle=True)
    # Test on validation set if available (not so for miniIN), else on test set.
    try:
        val_loader = cl.CLDataLoader(data.validation, bs=args.bs, shuffle=False)
    except TypeError:
        val_loader = cl.CLDataLoader(data.test, bs=args.bs, shuffle=False)

    device = torch.device("cpu" if args.no_cuda else "cuda")
    buffer = cl.Buffer(args.buf_size)  # Size will be overwritten if buffer is loaded

    if args.init is not None:
        model.load_state_dict(torch.load(f"../models/{args.data}/{args.init}.pt"))
        if args.buffer:
            if args.buf_name is None:
                buffer_file_name = f"../models/{args.data}/{args.init}_buffer.pkl"
            else:
                buffer_file_name = f"../models/{args.data}/{args.buf_name}.pkl"

            with open(buffer_file_name, 'rb') as f:
                buffer = pickle.load(f)


    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
    loss_func = cl.loss_wrapper(args.loss)

    if args.save:
        if not os.path.exists(f"../models/{args.data}"):
            os.makedirs(f"../models/{args.data}")

    if args.save_path:
        os.mkdir(f"../models/{args.data}/{args.save}", 0o750)
        torch.save(model.state_dict(), f"../models/{args.data}/{args.save}/model_0_0.pt")

    for task, tr_loader in enumerate(train_loader):
        print(f' --- Started training task {task} ---')
        for epoch in range(args.epochs):
            print(f' --- Started epoch {epoch} ---')
            model.train()

            for i, (data, target) in enumerate(tr_loader):
                data, target = data.to(device), target.to(device)


                for k in range(args.memIter):
                    if args.buffer and buffer.__len__() >0:
                        #buffer.sample((data, target))
                        #print(buffer.__len__(),data.size(0))

                        buf_data, buf_target = buffer.retrieve((data, target), args.bs)

                        if(buf_target!= None):
                            mem_num=buf_target.size(0)
                        else:
                            mem_num=0
                        if buf_data is not None:
                            buf_data, buf_target = buf_data.to(device), buf_target.to(device)
                            if (args.scraug == "Mem"):

                                buf_data=scr_transform(buf_data)
                            elif(args.scraug == "Incoming"):
                                data = scr_transform(data)
                            con_data = torch.cat((data, buf_data))
                            con_target = torch.cat((target, buf_target))
                        else:
                            con_data = data
                            con_target = target

                        # ## aug
                        if(args.scraug == "Both"):
                            if(args.aug_type == "scr"):
                                aug_data=scr_transform(con_data)
                            else:
                                aug_data=randaug(args,con_data,mem_num)
                                #print("raug")
                        else:
                            aug_data = con_data
                        #print("!!!",data.shape)
                        #assert False

                        cl.step(model, opt, aug_data, con_target, loss_func)
                    else:
                        cl.step(model, opt, data, target, loss_func)
                if args.buffer:
                    buffer.sample((data, target))


                if i != 0 and i % args.test == 0:
                    if args.save_path:
                        torch.save(model.state_dict(), f"../models/{args.data}/{args.save}/model_{task}_{i}.pt")
                    acc, loss = cl.test(model, val_loader, avg=True, device=device)
                    print(f"\t Acc task {task}, step {i} / {len(tr_loader)}: {acc:.2f}% (Loss: {loss:3f})")

    acc, loss = cl.test(model, val_loader, avg=True, device=device)
    print(f"Final average acc: {acc:.2f}%, (Loss: {loss:3f})")

    if args.save is not None:
        with open(f'../models/{args.data}/{args.save}.json', 'w') as g:
            json.dump(vars(args), g, indent=2)

        torch.save(model.state_dict(), f'../models/{args.data}/{args.save}.pt')
        if args.save_path:
            torch.save(model.state_dict(), f"../models/{args.data}/{args.save}/model_{task}_{i}.pt")

        if args.buffer:
            with open(f"../models/{args.data}/{args.save}_buffer.pkl", 'wb+') as f:
                pickle.dump(buffer, f)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--memIter',default=1,type=int)
    parser.add_argument('--scraug',default="None",choices=["None","Both","Mem","Incoming"])
    parser.add_argument('--aug_type',default="scr",choices=["scr","raug"])
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate for SGD')
    parser.add_argument('--mom', type=float, default=0,
                        help='momentum for SGD')
    parser.add_argument('--bs', type=int, default=10,
                        help='Batch size')
    parser.add_argument('--test', type=int, default=200,
                        help='how many steps before test')
    parser.add_argument('--tasks', type=str, default=None,
                        help='Which tasks should be trained. Used as "123" for task 1, 2 and 3')
    parser.add_argument('--joint', action='store_true')
    parser.add_argument('--buffer', action='store_true',
                        help="Use replay buffer")
    parser.add_argument('--buf_name', type=str, default=None,
                        help="Use if buffer to load has different name than the model")
    parser.add_argument('--buf_size', type=int, default=50,
                        help="Size of buffer to be used. Default is 50")
    parser.add_argument('--save', type=str, default=None,
                        help="file name for saving model")
    parser.add_argument('--init', type=str, default=None,
                        help="initial_weights")
    parser.add_argument('--data', type=str, default='mnist',
                        choices=['mnist', 'cifar',"cifar100", 'min'], help="Dataset to train")
    parser.add_argument('--epochs', type=int, default=1, help="Nb of epochs")
    parser.add_argument('--det', action='store_true', help="Run in deterministic mode")
    parser.add_argument('--loss', choices=['CE', 'hinge'], default="CE", help="Loss function for optimizer")
    parser.add_argument('--save_path', action='store_true',
                        help='Store intermediate models at [test] intervals will crash if path (i.e. folder) already'
                             ' exists to prevent accidental overwrites')
    parser.add_argument('--comment', type=str, help="Comment, will be written to json file if model is stored")

    args = parser.parse_args()

    if args.tasks is not None:
        args.tasks = [int(i) for i in args.tasks]

    if not args.no_cuda:
        if torch.cuda.is_available():
            print(f"[INFO] using cuda")
        else:
            args.no_cuda = True

    if args.det:
        torch.manual_seed(1997)

    if args.data == 'mnist':
        data = cl.get_split_mnist(args.tasks, args.joint)
        model = cl.MLP(hid_nodes=400, down_sample=1)
    elif args.data == "cifar":
        data = cl.get_split_cifar10(args.tasks, args.joint)
        model = cl.get_resnet18()
    elif args.data == "cifar100":
        data = cl.get_split_cifar100(args.tasks, args.joint)
        model = cl.get_resnet18(nb_classes=100,)
    elif args.data == "min":
        data = cl.get_split_mini_imagenet(args.tasks, )
        model = cl.get_resnet18(nb_classes=100, input_size=(3, 84, 84))
    else:
        raise ValueError(f"Data {args.data} not known.")

    # print(data.train)
    # assert False

    train(args, data, model)


if __name__ == '__main__':
    main()

import time,copy,os,pickle, argparse
import numpy as np
import torch
from torchvision import models
import matplotlib.pyplot as plt

# cuda
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-s','--seed',
                           type=int,
                           default=42,
                           help='random seed')
    
    argparser.add_argument('-l','--length',
                           type = int,
                           default=1000,
                           help='num of iterations')
    
    argparser.add_argument('-m','--model',
                           type = str,
                           default='vgg11',
                           help='[resnet50, resnet101, resnet152, vgg11, vgg16, vgg19]')

    argparser.add_argument('-i','--save',
                           type = bool,
                           default=True,
                           help='if save results')

    argparser.add_argument('-a','--add',
                           type = str,
                           default='parallel_1_',
                           help='NA')

    args = argparser.parse_args()

    #### model ##################
    torch.manual_seed(args.seed)

    if args.model == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    elif args.model == 'resnet101':
        weights = models.ResNet101_Weights.DEFAULT
        model = models.resnet101(weights=weights)
    elif args.model == 'resnet152':
        weights = models.ResNet152_Weights.DEFAULT
        model = models.resnet152(weights=weights)
    elif args.model == 'vgg11':
        weights = models.VGG11_Weights.DEFAULT
        model = models.vgg11(weights=weights)
    elif args.model == 'vgg16':
        weights = models.VGG16_Weights.DEFAULT
        model = models.vgg16(weights=weights)
    elif args.model == 'vgg19':
        weights = models.VGG19_Weights.DEFAULT
        model = models.vgg19(weights=weights)
    else:
        raise ValueError('no implemented models')
    
    model.to(device)
    model.eval()

    #### test data ##############
    data = torch.randn(1, 3, 224, 224).to(device) 

    # model
    total_time = 0
    time_record = []

    with torch.no_grad(): 
        for _ in range(args.length):
            start_time = time.perf_counter()
            outputs = model(data)
            outputs.detach().cpu()
            end_time = time.perf_counter()
            time_record.append(end_time-start_time)

    if len(time_record)>100: time_record = time_record[100:]

    mean = int(np.mean(time_record)*1000*100) # 10us
    std = int(np.std(time_record)*1000*100) # 10u

    print(f'average time: {mean/100} ms, time std: {std/100} ms')

    save_name = str(args.model)+'_'+args.add+str(args.length)+'_mean_'+str(mean/100)+'_std_'+str(std/100)

    if args.save:
        pickle.dump(time_record, open('./profile_results/figures/result_'+save_name+'.pkl', 'wb'))

        plt.plot(time_record, label='Inference Times') #;plt.show()
        plt.legend()
        plt.savefig('./profile_results/figures/result_'+save_name+'.png')
        # plt.show()
        plt.close()
    else:
        pass

    ######### see plot_parallel.py for result plotting ##############

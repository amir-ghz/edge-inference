import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import torchvision
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torchmetrics
import random
import setting

batch_size = 256
num_workers = 4
world_size = 4

my_acc = 0



def inject_fault_weight_level(device_level_failure, failed_device_index, ten, num_device, packet_size, packet_loss_percentage):
    """
    If device_level_failure is True: node failure is mimicked,
    else, the paclet loss will be simulated ... 
    """
    if device_level_failure:
         
        num_weights_per_device = torch.numel(ten)//num_device 
        d0, d1, d2, d3 = ten.shape

        index_stride = d1//num_device

        ten[:, (failed_device_index*index_stride):((failed_device_index+1)*index_stride), :, :] = 0
        

    else:     

        """
        The below logic can take as input any tensor and partition it into equal sized smaller tensors, then 
        it can set a arndom portaion of the selected partions to zero to mimick packet loss ...
        """

        num_weights_per_device = torch.numel(ten)//num_device
        num_weights_dropped = round(packet_loss_percentage * ((num_weights_per_device*4)//(packet_size)))

        # print('num_activations_dropped: ', num_weights_dropped)

        d0, d1, d2, d3 = ten.shape

        num_kernels_per_device = d1//num_device

        for device_index in range(0, num_device):

            start_index = device_index*num_kernels_per_device
            end_index = start_index + num_kernels_per_device

            for i in range(0, num_weights_dropped):
                    
                    rand_d0, rand_d1, rand_d2, rand_d3 = random.randint(0, d0-1), random.randint(start_index, end_index-1), random.randint(0, d2-1), random.randint(0, d3-1) 
                    ten[rand_d0, rand_d1, rand_d2, rand_d3] = 0

    return ten


def zero_hook(_, input, output):

        # Call the fault injection function to manipulate intermediate tensor

        output = inject_fault_weight_level(device_level_failure=setting.device_level_failure, 
                                           failed_device_index=setting.failed_device_index,
                                           ten=output, 
                                           num_device=setting.num_device, 
                                           packet_size=setting.packet_size, 
                                           packet_loss_percentage=setting.packet_loss_percentage)#torch.zeros(list(output.size())) #.to("cuda:0")

        return output


def test():
    

    print("Begin test for: ", "# devices: ", setting.num_device, "\t PLR: ", setting.packet_loss_percentage, "\t Packet size: ", setting.packet_size)

    # initialize model
    metric = torchmetrics.Accuracy()

    cuda_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(cuda_device)

    imagenet_1k_dir = "../deit/IMG"
    val_dir = os.path.join(imagenet_1k_dir, 'val')

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_set = datasets.ImageFolder(
        val_dir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))


    val_loader = torch.utils.data.DataLoader(val_set,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             sampler=None,
                                             pin_memory=True)
    
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.metric = metric
    model = model.to(cuda_device)

    model.eval()

    # for name, params in model.named_parameters():
    #      print(name, params.size())

    # print(model)

    # for name, param in model.named_modules():
    #      if isinstance(param, nn.Linear):
    #         print(param)
    #         print(name)

    criterion = nn.CrossEntropyLoss().cuda(cuda_device)

    my_acc = 0

    with torch.no_grad():
        loop = tqdm(val_loader)
        for i, (images, target) in enumerate(loop):
            if cuda_device is not None:
                images = images.to(cuda_device)
            if torch.cuda.is_available():
                target = target.to(cuda_device)
            

            """
            register a pytorch forward hook on the final layer (which is avgpooling). This will send the outpt feature maps
            to the fault injection function.
            """

            handle = model.avgpool.register_forward_hook(zero_hook)
            # model.module.layer2[0].conv1.register_forward_hook(zero_hook)
            # model.module.layer3[0].conv1.register_forward_hook(zero_hook)
            # model.module.layer4[0].conv1.register_forward_hook(zero_hook)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            acc = metric(output, target)

            # _, predicted = torch.max(output.data, 1)
            # total += target.size(0)
            # correct += (predicted == target).sum().item()
            
            loop.set_description(f"Current Acc: [{acc*100}]")
        

        acc = metric.compute()
        print(f"Accuracy on all data: {acc}")
        my_acc = acc.item()
        # Reseting internal state such that metric ready for new data
    
        metric.reset()

        return my_acc




     

if __name__ == "__main__":

    setting.init()

    # n = number of devices
    # p = packet_size
    # i = packet_loss_percentage 

    results = []
    for n in range(2, 11):
        setting.num_device = n
        for p in [64, 128, 256, 512, 1024]:
            setting.packet_size = p
            for i in range(1, 11):
                setting.packet_loss_percentage = i/100
                acc = test()
                results.append([setting.num_device, setting.packet_size, setting.packet_loss_percentage, acc])
                print(results)


    torch.save(results, 'results.pt')





import torch
from torch import nn,optim
from torch.nn import functional as F
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models,transforms
import copy
from utils import ContentLoss,StyleLoss,Normalize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.ion()
def image_loader(image_name):
    imsize = 512 if torch.cuda.is_available() else 128
    loader = transforms.Compose([transforms.Resize(imsize),transforms.ToTensor()])
    image = Image.open(image_name)
    image = loader(image).unsqueeze(0)
    return image.to(device)


def get_model(content_img,style_img):
    vgg = models.vgg19(pretrained=True).features.to(device)
    mean = torch.tensor([0.485,0.456,0.406]).to(device)
    std = torch.tensor([0.229,0.224,0.225]).to(device)
    normalize = Normalize(mean,std).to(device)
    model = nn.Sequential(normalize)
    i = 0
    content_losses = []
    style_losses = []
    content_layers = ['conv_4']
    style_layers = ['conv_1','conv_2','conv_3','conv_4','conv_5']
    for layer in vgg.children():
        if isinstance(layer,nn.Conv2d):
            i+=1
            name = 'conv_{}'.format(i)
        elif isinstance(layer,nn.ReLU):
            layer = nn.ReLU()
            name = 'relu_{}'.format(i)
        elif isinstance(layer,nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer,nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        model.add_module(name,layer)
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module('content_loss_{}'.format(i),content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module('style_loss_{}'.format(i),style_loss)
            style_losses.append(style_loss)
            if name=='conv_5':
                break
    return model,style_losses,content_losses

def run_style_transfer(content_imt,style_img,num_steps=300, style_weight=1000000, content_weight=1):
    print('Building the style transfer model..')
    model, style_losses, content_losses = get_model(content_img,style_img)#给model中存入label
    input_img = content_img.clone()
    optimizer = optim.LBFGS([input_img.requires_grad_()])#只有图像更新参数

    print('Optimizing..')
    run = [0]
    input_img.data.clamp_(0, 1)
    while run[0] <= num_steps:
        def closure():
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            style_score *= style_weight
            content_score *= content_weight
            loss = style_score + content_score
            loss.backward()
            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run[0]))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()
            return style_score + content_score
        optimizer.step(closure)
    input_img.data.clamp_(0, 1)
    return input_img
def imshow(img):
    img = img.squeeze(img).to('cpu')
    img = transforms.ToPILImage(img)
    img.show()
if __name__ == '__main__':
    content_img = image_loader('./images/hoovertowernight.jpg')
    style_img = image_loader('./images/candy.jpg')
    img = run_style_transfer(content_img,style_img)
    imshow(img)
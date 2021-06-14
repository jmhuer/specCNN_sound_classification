import cv2
from Model import load_weights, image_transforms, tensor_transform, FCN, load_data
from Model import newest_model, classes
from PIL import Image
import numpy as np
import time
import torch
import torchvision
import csv


def create_dataset(model, data):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    cnt = 0
    DATASET = 'Data/images/train/'

    for img, label, label_id in data['train']:
        label = label.to(device)
        img = img.to(device)
        logit = model(img)

        path = DATASET + str(int(label_id[0])) + "_" + str(cnt) + ".jpg"
        datalist = []
        datalist.append([path, int(label_id)])
        with open(DATASET + "labels.csv", "a+") as f:
            writer = csv.writer(f)
            writer.writerows(datalist)

        mid_layer_sample = model.mid_layer[0].detach().cpu()


        torchvision.utils.save_image(mid_layer_sample, path)
        cnt += 1
        print(path)

    # for img, label, sound_name in data['val']:
    #     label = label.to(device)
    #     img = img.to(device)
    #     logit = model(img)
    #
    #     path = DATASET +  str(int(sound_name[0])) + "_" + str(cnt) + ".jpg"
    #     datalist = []
    #     datalist.append([path,  str(int(sound_name[0]))])
    #     with open(DATASET + "labels.csv", "a+") as f:
    #         writer = csv.writer(f)
    #         writer.writerows(datalist)
    #
    #     torchvision.utils.save_image(model.mid_layer.detach().cpu()[0], path)
    #     cnt += 1
    #     print(path)


if __name__ == '__main__':
    import argparse
    '''
    We default to most recent model created
    '''
    default_model_path = newest_model("Model/saved_models")

    parser = argparse.ArgumentParser()
    # Put custom arguments here
    parser.add_argument('-m', '--model_path', type=str, default=default_model_path)
    args = parser.parse_args()

    print('Using model path: {}'.format(args.model_path))
    model = FCN(use_skip=False)
    model = load_weights(model, args.model_path)

    data = load_data('Data/dataset', transforms=tensor_transform, num_workers=1, batch_size=1)

    create_dataset(model , data)

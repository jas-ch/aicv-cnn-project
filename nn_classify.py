import collections
import torch, torchvision
import cv2
import matplotlib.pyplot as plt

import os, sys, time
import glob
import argparse
from PIL import Image
import json

import first_nn_torch as first
import second_nn_torch as second
import third_nn_torch as third
import fourth_nn_torch as fourth

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

IMAGE_PATH = "../transfer-learning-project-jas-ch/asl/asl_alphabet_test/asl_alphabet_test"
#IMAGE_PATH = "asl_short/asl_alphabet_test/asl_alphabet_test"

torch.manual_seed(37)

model1 = first.Model([3, 192, 192])
model2 = second.Model([3, 96, 96])
model3 = third.Model([3, 192, 192])
model4 = fourth.Model([3, 192, 192])

models = [model1, model2, model3, model4]

for model in models:
    model.print = False

input_shape = [[3, 192, 192], [3, 96, 96], [3, 192, 192], [3, 192, 192]]

model1.to(device)
model2.to(device)
model3.to(device)
model4.to(device)

# optimizer = torch.optim.Adam(param_groups)
loss_fn = torch.nn.CrossEntropyLoss()

lr = 0.001

def find_latest_checkpoint(num:int) -> None:
    """Find the latest epoch checkpoint from 1-5"""
    all_checkpoints = glob.glob(f'nn{num}_checkpoint_epoch*.tar')
    all_checkpoints.sort(key=os.path.getmtime, reverse=True)

    for checkpoint in all_checkpoints:
        fh = torch.load(checkpoint, weights_only=True, map_location=device)
        if 'model_state_dict' not in fh:
            print(f"incomplete checkpoint {checkpoint}.\n")
        else:
            print(f"found latest checkpoint {checkpoint}")
            return fh, checkpoint
    return None, None

checkpoints = []
cp_filenames = []
for i in range(4):
    check, filename = find_latest_checkpoint(i) 
    checkpoints.append(check)
    cp_filenames.append(filename)

    if checkpoints[i]:
        models[i].load_state_dict(checkpoints[i]['model_state_dict'])
    #   optimizer.load_state_dict(checkpoint['optim_state_dict'])
        print(f"{cp_filenames[i]} model {i} weights loaded.")

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
           'del', 'nothing', 'space']
def classify(model, num, input_dir, loss_fn, classes, device, output_path, output_file):
    model.eval()

    loss = 0.0

    images = glob.glob(os.path.join(input_dir+"/*.jpg"))
    if output_path != '':
        output_file = os.path.join(output_path+'/'+output_file)
    with open(output_file, 'w') as f:
        f.write(f"results for running model on data in {input_dir}\n")
        for image in images:
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, input_shape[num][1:])
            img = img / 255.0
            img = img.transpose(2, 0, 1) # move channels to first
            input_tensor = torch.tensor(
                img,
                dtype = torch.float32,
                device = device,
            )
            input_tensor = input_tensor.unsqueeze(0)

            with torch.no_grad():
                outputs = model(input_tensor)
                prob = torch.softmax(outputs, dim=1)
                top_prob, top_class_idx = torch.max(prob, 1)

            label = os.path.basename(image).split('_')[0]
            predict = classes[top_class_idx.item()]
            loss = loss if (label == predict) else loss+1

            result = {
                'filename': image, 
                'predicted_class': predict,
                'confidence': top_prob.item()
            }

            f.write(json.dumps(result))
            f.write('\n')

        print(f"prediction accuracy = {1 - loss/len(images)}.")
        f.write(f"prediction accuracy = {1 - loss/len(images)}.")

def latest_epoch_losses(model, epochs):
    files_list = glob.glob(f"saves/{model}_printout_*.txt")
    latest_file = max(files_list, key=os.path.getctime)
    with open(latest_file, 'r') as f:
        lines = f.readlines()
        if lines:
            losses = []
            for i in range(epochs):
                losses.append(float(lines[-epochs+i]))
            return losses
        else:
            return ""
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_image_path', help='path to the input images folder', default=IMAGE_PATH)
    parser.add_argument('-p', '--output_path', help='path for output', default='saves')
    args = parser.parse_args()

    if args.input_image_path:
        for i in range(4):
            classify(models[i], i, args.input_image_path, loss_fn, classes, device, args.output_path, f"nn{i+1}_test_output.txt")
    else:
        print(f"missing input image path...")

    epochs = 5
    losses1 = latest_epoch_losses("nn1", epochs)
    losses2 = latest_epoch_losses("nn2", epochs)
    losses3 = latest_epoch_losses("nn3", epochs)
    losses4 = latest_epoch_losses("nn4", epochs)

    plt.figure()
    plt.plot(losses1, label="First", color="r")
    plt.plot(losses2, label="Second", color="y")
    plt.plot(losses3, label="Third", color="g")
    plt.plot(losses4, label="Fourth", color="b")

    plt.title("Loss Comparisons")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("saves/val_loss.jpg")
    plt.show()


if __name__ == '__main__':
    main()

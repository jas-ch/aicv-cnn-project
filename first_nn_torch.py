import pathlib
import time
import atexit

import torch
import cv2
import torchvision.transforms.v2 as transforms

INPUT_SHAPE = [3, 192, 192]
IMAGES_PATH = pathlib.Path("../transfer-learning-project-jas-ch/asl/asl_alphabet_train/asl_alphabet_train")
#IMAGES_PATH = pathlib.Path("asl_short/asl_alphabet_train/asl_alphabet_train") 
BATCH_SIZE = 32
TIME_STAMP = time.strftime("%Y_%m_%d_%H_%M")

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = DEV
WRITE = "f"

def clean_up(model, epoch) -> None:
    print("Saved final model")
    torch.save(model, "saves/nn1_model_" + str(epoch) + "_" + TIME_STAMP)


class Dataset(torch.utils.data.Dataset):
    """represent the dataset as an object"""
    def __init__(self, image_path: pathlib.Path):
        """provide a path where the images are, 1 folder for each category
        this assumes that train and validation are mingled."""
        print("Loading images...")
        self.class_names = []
        self.images = []
        for path in image_path.iterdir():
            if path.is_dir():
                self.class_names.append(path.name)
                for img in path.iterdir():
                    img = cv2.imread(img)
                    if img is not None:
                        label = torch.tensor(
                            self.class_names.index(path.name),
                            dtype = torch.long,
                            device= DEV,
                        )
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, INPUT_SHAPE[1:])
                        img = img / 255.0
                        img = img.transpose(2, 0, 1) # move channels to first
                        img = torch.tensor(
                            img,
                            dtype = torch.float32,
                            device = DEV,
                        )
                        self.images.append((img, label))
            print("images loaded :)")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> (torch.Tensor, torch.Tensor):
        return self.images[idx]

class Model(torch.nn.Module):
    """represent my CNN as an object"""
    def __init__(self, input_shape: (int, int, int)):
        """input_shape is expected to be channels first"""
        super().__init__()
        print(f"{'input shape':>30}", input_shape)
        self.relu = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels = 3,
            out_channels = 48,
            kernel_size = 9,
            stride = 3,
        )
        self.zp1 = torch.nn.ZeroPad2d((2, 1, 2, 1))
        self.conv2 = torch.nn.Conv2d(
            in_channels = 48,
            out_channels = 128,
            kernel_size = 5,
            stride = 2,
        )
        self.maxpool1 = torch.nn.MaxPool2d(
            kernel_size = 3,
            stride = 2,
        )
        self.zp2 = torch.nn.ZeroPad2d((1, 1, 1, 1))
        self.conv3 = torch.nn.Conv2d(
            in_channels = 128,
            out_channels = 192,
            kernel_size = 3,
            stride = 1,
        )
        self.maxpool2 = torch.nn.MaxPool2d(
            kernel_size = 3,
            stride = 2,
        )
        self.zp3 = torch.nn.ZeroPad2d((1,1,1,1))
        self.conv4 = torch.nn.Conv2d(
            in_channels = 192,
            out_channels = 192,
            kernel_size = 3,
            stride = 1,
        )
        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(
            in_features = 9408,  # 7 * 7 * 192 = 9408,
            out_features = 2048,  # classes?
        )
        self.linear2 = torch.nn.Linear(
            in_features = 2048,
            out_features = 256,
        )
        self.linear3 = torch.nn.Linear(
            in_features = 256,
            out_features = 29,      # classification = # of classes, regression = 1
        )

        self.print = True


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """run a forward pass of the CNN"""
        if self.print:
            print(f"{'input shape':>30}", x.shape)
        # y = self.zp1(x)
        # print(f"{'after 1st zero-pad':>30}", y.shape)
        # y = self.relu(self.conv1(y))
        y = self.relu(self.conv1(x))
        if self.print:
            print(f"{'after 1st convolution:':>30}", y.shape)
        y = self.zp1(y)
        if self.print:
            print(f"{'after 1st zero-pad:':>30}", y.shape)
        y = self.relu(self.conv2(y))
        if self.print:
            print(f"{'after 2nd convolution:':>30}", y.shape)
        y = self.maxpool1(y)
        if self.print:
            print(f"{'after 1st maxpool:':>30}", y.shape)
        y = self.zp2(y)
        if self.print:
            print(f"{'after 2nd zero-pad:':>30}", y.shape)
        y = self.relu(self.conv3(y))
        if self.print:
            print(f"{'after 3rd convolution:':>30}", y.shape)
        y = self.maxpool2(y)
        if self.print:
            print(f"{'after 2nd maxpool:':>30}", y.shape)
        y = self.zp3(y)
        if self.print:
            print(f"{'after 3rd zero-pad:':>30}", y.shape)
        y = self.relu(self.conv4(y))
        if self.print:
            print(f"{'after 4th convolution:':>30}", y.shape)
        y = self.flatten(y)
        if self.print:
            print(f"{'after flatten:':>30}", y.shape)
        y = self.linear1(y)
        if self.print:
            print(f"{'after 1st linear:':>30}", y.shape)
        y = self.linear2(y)
        if self.print:
            print(f"{'after 2nd linear:':>30}", y.shape)
        y = self.linear3(y)
        if self.print:
            print(f"{'after 3rd linear:':>30}", y.shape)
        return y

def get_dataloaders(dataset: Dataset, train_prop: float, batch_size: int
        ) -> (torch.utils.data.DataLoader, torch.utils.data.DataLoader):
    """split the dataset and prepare for training"""
    generator = torch.Generator().manual_seed(37)
    train_set, validation_set = torch.utils.data.random_split(
            dataset,
            lengths = [train_prop, 1-train_prop],
            generator = generator,
    )
    print(f"Number of images in training set before augmentation: ", end="")
    print(f"{len(train_set)}")
    aug_set = transforms.RandomHorizontalFlip(1.0)(train_set)
    train_set = torch.utils.data.ConcatDataset([train_set, aug_set])
    print(f"Number of images in training set after augmentation: ", end="")
    print(f"{len(train_set)}")


    train = torch.utils.data.DataLoader(train_set, batch_size = batch_size)
    validation = torch.utils.data.DataLoader(validation_set, 
            batch_size = batch_size)
    return train, validation

def validate(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
            for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

    model.train()
    return val_loss/len(val_loader), val_correct/val_total

def save_code() -> None:
    with open(__file__, 'r') as f:
        this_code = f.read()
    with open("saves/nn1_code_save_" + TIME_STAMP + ".py", 'w') as f:
        print(this_code, file=f)

def main():
    epochs = 5

    if WRITE == "f":
        pass
    elif WRITE == "wq":
        save_code()
        quit()
    else:
        save_code()
        
    dataset = Dataset(IMAGES_PATH)
    print(f"Found {len(dataset)} images")
    train, val = get_dataloaders(dataset, 0.8, BATCH_SIZE)
    model = Model(INPUT_SHAPE)
    x = next(iter(train))[0]
    model.to(DEV)
    model(x)
    model.print = False

    atexit.register(clean_up, model, epochs)

    output_file = open("saves/nn1_printout_" + TIME_STAMP + ".txt", "w")

    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    epoch_losses = []
    for i in range(epochs):
        batch_losses = []
        batch_accuracies = []
        
        print(f"---- EPOCH {i+1} ----")
        model.train()

        for batch_idx, (image_batch, label_batch) in enumerate(train):
            image_batch, label_batch = image_batch.to(device, non_blocking=True), label_batch.to(device, non_blocking=True)
            preds = model(image_batch)
            loss = loss_fn(preds, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            cur_loss = sum(batch_losses)/len(batch_losses)
            this_batch_acc = int(sum(preds.argmax(1)==label_batch)) / len(label_batch)
            batch_accuracies.append(this_batch_acc)
            cur_acc = sum(batch_accuracies)/len(batch_accuracies)

            print("Train:", end="\t\t")
            print(f"Batch: {len(batch_losses)}", end="\t")
            print(f"Loss: {cur_loss:.4f}", end="\t")
            print(f"Accuracy: {cur_acc:.4f}", end="\r")

        # epoch statistics
        epoch_loss = sum(batch_losses) / len(batch_losses)
        epoch_acc = sum(batch_accuracies) / len(batch_accuracies)

        # validation phase
        val_loss, val_acc = validate(model, val, loss_fn, device)
        epoch_losses.append(val_loss)
        epoch_summary = f"""\nEpoch {i+1} Summary:\n
        Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}\n
        Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}\n"""
        print(epoch_summary)
        output_file.write(epoch_summary)

        # save
        PATH = f'saves/nn1_checkpoint_epoch{i+1}.tar'
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'train_loss': epoch_loss,
            'train_acc': epoch_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }, PATH)

        print(f"Checkpoint saved to {PATH}\n")
        
    for loss in epoch_losses:
        output_file.write("\n"+str(loss))
        print("epoch loss saved")
    output_file.close()

if __name__ == "__main__":
    main()
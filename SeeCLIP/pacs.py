import os.path as osp

import torch
import torch.nn.utils as utils
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import os
import glob 
import random
from trainer.odgclip import *

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import torchvision.transforms as transforms

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device='cpu')

repeat_transform = transforms.Compose([
    transforms.ToTensor(),
])

class DataTrain(Dataset):
  def __init__(self,train_image_paths,train_domain,train_labels):
    self.image_path=train_image_paths
    self.domain=train_domain
    self.labels=train_labels

  def __len__(self):
    return len(self.labels)

  def __getitem__(self,idx):
    image = preprocess(Image.open(self.image_path[idx]))
    domain=self.domain[idx] 
    domain=torch.from_numpy(np.array(domain)) 
    label=self.labels[idx] 
    label=torch.from_numpy(np.array(label)) 
 
    label_one_hot=F.one_hot(label,num_classes)
  
    return image, domain, label, label_one_hot 


#################-------DATASET------#######################

domains = ['art_painting', 'cartoon', 'photo', 'sketch']

'''
############### The source dataset 1 ##################
'''

image_path_dom1=[]
label_class_dom1=[]
label_dom1=[]
class_names1=[]
path_dom1='./data/pacs/'+domains[0]
domain_name1 = path_dom1.split('/')[-1]
dirs_dom1=os.listdir(path_dom1)
class_names = dirs_dom1
num_classes = len(class_names)
class_names.sort()
dirs_dom1.sort()
c=0
index=0
index_dom1 = [0, 1, 2, 3, 4, 5]
for i in dirs_dom1:
    if index in index_dom1:
        class_names1.append(i)
        impaths = path_dom1 + '/' + i
        paths = glob.glob(impaths+'/**.jpg')
        random.shuffle(paths)
        image_path_dom1.extend(paths)
        label_class_dom1.extend([c for _ in range(len(paths))])
    c = c + 1
    index = index + 1
label_dom1=[0 for _ in range(len(image_path_dom1))] 


'''
############### The source dataset 2 ##################
'''

image_path_dom2=[]
label_class_dom2=[]
label_dom2=[]
class_names2=[]
path_dom2='./data/pacs/'+domains[1]
domain_name2 = path_dom2.split('/')[-1]
dirs_dom2=os.listdir(path_dom2)
dirs_dom2.sort()
c=0
index=0
index_dom2 = [0, 1, 2, 3, 4, 5]
for i in dirs_dom2:
  if index in index_dom2:
    class_names2.append(i)
    impaths=path_dom2+'/' +i
    paths=glob.glob(impaths+'*/**.jpg')
    random.shuffle(paths)
    image_path_dom2.extend(paths)
    label_class_dom2.extend([c for _ in range(len(paths))])
  c=c+1
  index=index+1  
label_dom2=[1 for _ in range(len(image_path_dom2))]  


'''
############### The source dataset 3 ##################
'''

image_path_dom3=[]
label_class_dom3=[]
label_dom3=[]
class_names3=[]
path_dom3='./data/pacs/'+domains[3]
domain_name3 = path_dom3.split('/')[-1]
dirs_dom3=os.listdir(path_dom3)
dirs_dom3.sort()
c=0
index=0
index_dom3 = [0, 1, 2, 3, 4, 5]
for i in dirs_dom3:
  if index in index_dom3:
    class_names3.append(i)
    impaths=path_dom3+'/' +i
    paths=glob.glob(impaths+'*/**.png')
    random.shuffle(paths)
    image_path_dom3.extend(paths)
    label_class_dom3.extend([c for _ in range(len(paths))])
  c=c+1
  index=index+1
label_dom3=[2 for _ in range(len(image_path_dom3))]

# Known Classes
index_dom = list(set(index_dom1 + index_dom2 + index_dom3))
known_class_names = [class_names[idx] for idx in index_dom]
known_classes = ",".join(known_class_names)
'''
############### The combining the source dataset ##################
'''   
  
image_path_final=[]
image_path_final.extend(image_path_dom1)
image_path_final.extend(image_path_dom2)
image_path_final.extend(image_path_dom3)
label_class_final=[]
label_class_final.extend(label_class_dom1)
label_class_final.extend(label_class_dom2)
label_class_final.extend(label_class_dom3)
label_dom_final=[]
label_dom_final.extend(label_dom1)
label_dom_final.extend(label_dom2)
label_dom_final.extend(label_dom3)
domain_names=[]
domain_names.append(domain_name1)
domain_names.append(domain_name2)
domain_names.append(domain_name3)
print("domain_names",domain_names)

    
'''
##### Creating dataloader ######
'''
batchsize = 16
train_prev_ds=DataTrain(image_path_final,label_dom_final,label_class_final)
print(f'length of train_prev_ds: {len(train_prev_ds)}')
train_dl=DataLoader(train_prev_ds,batch_size=batchsize, num_workers=2, shuffle=True)
img_prev, domain_prev, label_prev, label_prev_one_hot = next(iter(train_dl))

domain_prev = domain_prev.to(device)

class_names.sort()
train_prev_classnames = class_names[:6]


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
    

def domain_text_loss(diff_textfeatures, domain):
    losses = []
    for i in range(len(domain) - 1):
        if domain[i] != domain[i + 1]:
           loss = F.mse_loss(diff_textfeatures[i], diff_textfeatures[i + 1])
           losses.append(loss)

    mse_loss = torch.mean(torch.stack(losses))

    return mse_loss


class ImageFilter(nn.Module):
    def __init__(self, brightness_threshold=0.01):
        super(ImageFilter, self).__init__()
        self.brightness_threshold = brightness_threshold

    def calculate_brightness(self, images):
        grayscale_images = torch.mean(images, dim=1, keepdim=True)  # Convert to grayscale
        return grayscale_images.mean((2, 3))  # Calculate the average pixel value for each image

    def forward(self, image_tensor):
        batch_size = image_tensor.size(0)
        brightness_values = self.calculate_brightness(image_tensor)

        fraction_to_select = 1.0
        
        num_images_to_select = int(batch_size * fraction_to_select)
        indices_with_brightness_condition = [i for i, value in enumerate(brightness_values) if value >= self.brightness_threshold]
        if len(indices_with_brightness_condition) < num_images_to_select:
           selected_indices = indices_with_brightness_condition
           num_black_images_to_select = num_images_to_select - len(indices_with_brightness_condition)
           all_indices = list(range(batch_size))
           black_indices = [i for i in all_indices if i not in indices_with_brightness_condition]
           random_black_indices = random.sample(black_indices, num_black_images_to_select)
           selected_indices += random_black_indices
           return selected_indices
        else:
           selected_indices = random.sample(indices_with_brightness_condition, num_images_to_select)
           return selected_indices

image_filter = ImageFilter(brightness_threshold=0.01)

def train_epoch(model, domainnames, train_loader, optimizer, lr_scheduler, step, 
                alpha=1.0, beta=0.5, gamma=0.01, margin=0.5):
    """
    SeeCLIP training function
    Args:
        model: SeeCLIP model instance
        domainnames: List of domain names
        train_loader: Training data loader
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        step: Scheduler step type ("batch" or "epoch")
        alpha: Weight for repulsion loss
        beta: Weight for cohesive loss  
        gamma: Weight for regularization loss
        margin: Margin value for repulsion loss
    """
    model.train()
    
    # Loss and accuracy meters
    total_loss_meter = AvgMeter()
    alignment_loss_meter = AvgMeter()
    repulsion_loss_meter = AvgMeter()
    cohesive_loss_meter = AvgMeter()
    reg_loss_meter = AvgMeter()
    accuracy_meter = AvgMeter()
    
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    
    for img_prev, domain_prev, label_prev, label_one_hot_prev in tqdm_object:
        # Move data to device
        img_prev = img_prev.to(device)
        domain_prev = domain_prev.to(device) 
        label_prev = label_prev.to(device)
        label_one_hot_prev = label_one_hot_prev.to(device)
        batch_size = img_prev.shape[0]
        
        # Generate pseudo-unknown samples
        unknown_images_list = []
        unknown_domains_list = []
        
        for domain_idx, domain_name in enumerate(domainnames):
            # Use SeeCLIP's built-in diffusion generator
            with torch.no_grad():
                # First pass through model to obtain semantic tokens
                temp_results = model(img_prev[:min(4, batch_size)], generate_unknowns=False)
                semantic_tokens = temp_results['semantic_tokens']
                
                # Generate pseudo-unknown samples
                domain_clean_name = domain_name.replace("_", " ")
                pseudo_unknowns = model.diffusion_generator.generate_pseudo_unknowns(
                    semantic_tokens, 
                    domain_clean_name, 
                    model.classnames,
                    batch_size=max(1, int(batch_size * 0.1))
                )
                
                unknown_images_list.append(pseudo_unknowns)
                
                # Create corresponding domain labels
                unknown_domain_labels = torch.full(
                    (pseudo_unknowns.shape[0],), 
                    domain_idx
                ).to(device)
                unknown_domains_list.append(unknown_domain_labels)
        
        # Merge all generated unknown samples
        if unknown_images_list:
            generated_unknown_images = torch.cat(unknown_images_list, dim=0)
            unknown_domains = torch.cat(unknown_domains_list, dim=0)
            
            # Create unknown class labels
            unknown_label_rank = len(model.classnames)  # Label index for unknown class
            unknown_labels = torch.full(
                (generated_unknown_images.shape[0],), 
                unknown_label_rank
            ).to(device)
            
            # Image quality filtering (if needed)
            if 'image_filter' in globals():
                random_indices = image_filter(generated_unknown_images)
                selected_images = generated_unknown_images[random_indices]
                selected_labels = unknown_labels[random_indices] 
                selected_domains = unknown_domains[random_indices]
            else:
                selected_images = generated_unknown_images
                selected_labels = unknown_labels
                selected_domains = unknown_domains
            
            # Combine known and unknown samples
            img_combined = torch.cat((img_prev, selected_images), dim=0)
            label_combined = torch.cat((label_prev, selected_labels), dim=0)
            domain_combined = torch.cat((domain_prev, selected_domains), dim=0)
        else:
            # If no unknown samples generated, use only original data
            img_combined = img_prev
            label_combined = label_prev
            domain_combined = domain_prev
        
        # Forward pass
        results = model(img_combined, label_combined)
        
        # Extract results
        all_logits = results['logits']
        known_logits = results['known_logits']
        losses = results['losses']
        
        # Calculate total loss
        alignment_loss = losses.get('alignment', torch.tensor(0.0).to(device))
        repulsion_loss = losses.get('repulsion', torch.tensor(0.0).to(device))
        cohesive_loss = losses.get('cohesive', torch.tensor(0.0).to(device))
        reg_loss = losses.get('regularization', torch.tensor(0.0).to(device))
        
        # Domain text loss (if needed to maintain original functionality)
        domain_text_loss_val = torch.tensor(0.0).to(device)
        if 'domain_text_loss' in globals() and 'diff_projfeatures' in results:
            domain_text_loss_val = domain_text_loss(
                results['diff_projfeatures'], 
                domain_combined
            )
        
        # Total loss calculation (Equation 11)
        total_loss = (alignment_loss + 
                     alpha * repulsion_loss + 
                     beta * cohesive_loss + 
                     gamma * reg_loss + 
                     0.01 * domain_text_loss_val)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        
        # Gradient clipping
        if 'clip_model' in globals():
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Learning rate scheduling
        if step == "batch":
            lr_scheduler.step(total_loss_meter.avg)
        
        # Calculate accuracy
        count = img_combined.size(0)
        
        # For predictions including unknown classes, special handling for accuracy calculation
        if len(model.classnames) < all_logits.shape[-1]:  # Contains unknown class
            # Calculate known class accuracy and overall accuracy separately
            known_mask = label_combined < len(model.classnames)
            if known_mask.sum() > 0:
                known_acc = compute_accuracy(
                    known_logits[known_mask], 
                    label_combined[known_mask]
                )[0].item()
            else:
                known_acc = 0.0
            
            # Overall accuracy
            overall_acc = compute_accuracy(all_logits, label_combined)[0].item()
            accuracy = overall_acc
        else:
            accuracy = compute_accuracy(all_logits, label_combined)[0].item()
        
        # Update meters
        total_loss_meter.update(total_loss.item(), count)
        alignment_loss_meter.update(alignment_loss.item(), count)
        repulsion_loss_meter.update(repulsion_loss.item(), count)
        cohesive_loss_meter.update(cohesive_loss.item(), count)
        reg_loss_meter.update(reg_loss.item(), count)
        accuracy_meter.update(accuracy, count)
        
        # Update progress bar display
        tqdm_object.set_postfix(
            total_loss=total_loss_meter.avg,
            align_loss=alignment_loss_meter.avg,
            repul_loss=repulsion_loss_meter.avg,
            cohes_loss=cohesive_loss_meter.avg,
            reg_loss=reg_loss_meter.avg,
            accuracy=accuracy_meter.avg,
            lr=get_lr(optimizer) if 'get_lr' in globals() else optimizer.param_groups[0]['lr']
        )
    
    # Return training results
    return {
        'total_loss': total_loss_meter,
        'alignment_loss': alignment_loss_meter,
        'repulsion_loss': repulsion_loss_meter,
        'cohesive_loss': cohesive_loss_meter,
        'regularization_loss': reg_loss_meter,
        'accuracy': accuracy_meter.avg
    }


def train_epoch_alternating(model, domainnames, train_loader, optimizer, lr_scheduler, 
                           step, alpha=1.0, beta=0.5, gamma=0.01):
    """
    Alternating training version: separate alignment phase and repulsion phase training
    
    Args:
        model: SeeCLIP model instance
        domainnames: List of domain names
        train_loader: Training data loader
        optimizer: Optimizer
        lr_scheduler: Learning rate scheduler
        step: Scheduler step type
        alpha: Weight for repulsion loss
        beta: Weight for cohesive loss
        gamma: Weight for regularization loss
    """
    model.train()
    
    alignment_loss_meter = AvgMeter()
    repulsion_loss_meter = AvgMeter()
    accuracy_meter = AvgMeter()
    
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    
    for batch_idx, (img_prev, domain_prev, label_prev, label_one_hot_prev) in enumerate(tqdm_object):
        img_prev = img_prev.to(device)
        domain_prev = domain_prev.to(device)
        label_prev = label_prev.to(device)
        batch_size = img_prev.shape[0]
        
        # Alternate between alignment and repulsion phases
        if batch_idx % 2 == 0:
            # Alignment phase: optimize known class alignment
            results = model(img_prev, label_prev, generate_unknowns=False)
            
            alignment_loss = results['losses'].get('alignment', torch.tensor(0.0).to(device))
            reg_loss = results['losses'].get('regularization', torch.tensor(0.0).to(device))
            
            loss = alignment_loss + gamma * reg_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            alignment_loss_meter.update(alignment_loss.item(), batch_size)
            
        else:
            # Repulsion phase: optimize unknown class boundaries
            with torch.no_grad():
                temp_results = model(img_prev[:min(4, batch_size)], generate_unknowns=False)
                semantic_tokens = temp_results['semantic_tokens']
            
            # Generate pseudo-unknown samples
            domain_name = domainnames[0].replace("_", " ") if domainnames else "photo"
            pseudo_unknowns = model.diffusion_generator.generate_pseudo_unknowns(
                semantic_tokens, domain_name, model.classnames,
                batch_size=max(1, int(batch_size * 0.1))
            )
            
            # Create unknown labels
            unknown_label_rank = len(model.classnames)
            unknown_labels = torch.full((pseudo_unknowns.shape[0],), unknown_label_rank).to(device)
            unknown_domains = torch.zeros(pseudo_unknowns.shape[0], dtype=torch.long).to(device)
            
            # Combine data
            img_combined = torch.cat((img_prev, pseudo_unknowns), dim=0)
            label_combined = torch.cat((label_prev, unknown_labels), dim=0)
            domain_combined = torch.cat((domain_prev, unknown_domains), dim=0)
            
            results = model(img_combined, label_combined, generate_unknowns=False)
            
            repulsion_loss = results['losses'].get('repulsion', torch.tensor(0.0).to(device))
            cohesive_loss = results['losses'].get('cohesive', torch.tensor(0.0).to(device))
            reg_loss = results['losses'].get('regularization', torch.tensor(0.0).to(device))
            
            loss = alpha * repulsion_loss + beta * cohesive_loss + gamma * reg_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            repulsion_loss_meter.update(repulsion_loss.item(), img_combined.size(0))
        
        # Calculate accuracy
        if batch_idx % 4 == 0:  # Calculate accuracy every 4 batches to save time
            with torch.no_grad():
                eval_results = model(img_prev, label_prev, generate_unknowns=False)
                accuracy = compute_accuracy(eval_results['known_logits'], label_prev)[0].item()
                accuracy_meter.update(accuracy, batch_size)
        
        # Learning rate scheduling
        if step == "batch":
            current_loss = alignment_loss_meter.avg if batch_idx % 2 == 0 else repulsion_loss_meter.avg
            lr_scheduler.step(current_loss)
        
        # Update progress bar
        tqdm_object.set_postfix(
            align_loss=alignment_loss_meter.avg,
            repul_loss=repulsion_loss_meter.avg,
            accuracy=accuracy_meter.avg,
            phase="Alignment" if batch_idx % 2 == 0 else "Repulsion"
        )
    
    return {
        'alignment_loss': alignment_loss_meter,
        'repulsion_loss': repulsion_loss_meter, 
        'accuracy': accuracy_meter.avg
    }
  
unknown_image_generator = GenerateUnknownImages().to(device)

train_classnames = train_prev_classnames + ['unknown']
print(f'length of train_classnames : {len(train_classnames)}')

train_model = CustomCLIP(train_classnames, domain_names, clip_model).to(device)

params = [
            {"params": train_model.domainclass_pl.parameters()},
            {"params": train_model.domain_pl.parameters()},
            {"params": train_model.conv_layer.parameters()},
            {"params": train_model.upsample_net.parameters()}
        ]

optimizer = torch.optim.AdamW(params,  weight_decay=0.00001)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=1, factor=0.8
        )
scaler = GradScaler() 

'''
Test dataset
'''
test_image_path_dom=[]
test_label_class_dom=[]
test_label_dom=[]
test_domain_names=[]
test_path_dom='./data/pacs/'+domains[2]
test_domain_name = test_path_dom.split('/')[-1]
test_dirs_dom=os.listdir(test_path_dom)
test_class_names = test_dirs_dom
test_num_classes = len(test_class_names)
test_dirs_dom.sort()
c=0
index=0
text_index = [0,1,2,3,4,5,6]
for i in test_dirs_dom:
  if index in text_index:
    impaths=test_path_dom+'/' +i
    paths=glob.glob(impaths+'*/**.jpg')
    test_image_path_dom.extend(paths)
    test_label_class_dom.extend([c for _ in range(len(paths))])
  c=c+1
  index=index+1  
test_label_dom=[3 for _ in range(len(test_image_path_dom))]
  
test_image_path_final=[]
test_image_path_final.extend(test_image_path_dom)

test_label_class_final=[]
test_label_class_final_modified = [label if label <= 5 else 6 for label in test_label_class_dom]
test_label_class_final.extend(test_label_class_final_modified)

test_label_dom_final=[]
test_label_dom_final.extend(test_label_dom)


test_domain_names.append(test_domain_name)
test_domain_names.append(test_domain_name)
test_domain_names.append(test_domain_name)

'''
############### Making the test dataloader ##################
''' 

test_ds=DataTrain(test_image_path_final,test_label_dom_final,test_label_class_final)
print(len(test_ds))
test_dl=DataLoader(test_ds,batch_size=32, num_workers=4, shuffle=True)
test_img, test_domain, test_label, test_label_one_hot = next(iter(test_dl))

num_epochs = 20
step = "epoch"
best_acc = 0
best_closed_set_acc = 0
best_open_set_acc = 0
best_avg_acc = 0
accuracy_file_path = "./results/pacs/photo.txt"  
accuracy_dir = os.path.dirname(accuracy_file_path)
if not os.path.exists(accuracy_dir):
    os.makedirs(accuracy_dir)
accuracy_file = open(accuracy_file_path, "w")
torch.autograd.set_detect_anomaly(True)

for epoch in range(num_epochs):
    print(f"Epoch: {epoch + 1}")
    train_model.train()
    train_loss, train_acc = train_epoch(train_model, unknown_image_generator, domain_names, train_dl, optimizer, lr_scheduler, step)
    print(f"epoch {epoch+1} : training accuracy: {train_acc}")

    TRAIN_MODEL_PATH = Path("./train_models/pacs/photo")
    TRAIN_MODEL_PATH.mkdir(parents=True, exist_ok=True)
    TRAIN_MODEL_NAME = f"photo_{epoch+1}.pth"
    TRAIN_MODEL_SAVE_PATH = TRAIN_MODEL_PATH / TRAIN_MODEL_NAME
    print(f"Saving train_model to: {TRAIN_MODEL_SAVE_PATH}")
    torch.save(obj=train_model.state_dict(), f=TRAIN_MODEL_SAVE_PATH)

    MODEL_PATH = "./train_models/pacs/photo"
    MODEL_NAME = f"photo_{epoch+1}.pth"
    MODEL_FILE = os.path.join(MODEL_PATH, MODEL_NAME)
    
    test_model = CustomCLIP(train_classnames, test_domain_names, clip_model).to(device)
    test_model.load_state_dict(torch.load(MODEL_FILE))

    with torch.no_grad():
        test_probs_all = torch.empty(0).to(device)
        test_labels_all = torch.empty(0).to(device)
        test_class_all = torch.empty(0).to(device)
        test_tqdm_object = tqdm(test_dl, total=len(test_dl))

        total_correct_a = 0
        total_samples_a = 0
        total_correct_b = 0
        total_samples_b = 0
        
        for test_img, test_domain, test_label, test_label_one_hot in test_tqdm_object:
            test_img = test_img.to(device)
            test_domain =test_domain.to(device)
            test_label = test_label.to(device)
            test_label_one_hot = test_label_one_hot.to(device)
            
            test_output, _ = test_model(test_img.to(device), test_label)

            predictions = torch.argmax(test_output, dim=1)
            class_a_mask = (test_label <= 5) 
            class_b_mask = (test_label > 5)

            correct_predictions_a = (predictions[class_a_mask] == test_label[class_a_mask]).sum().item()
            correct_predictions_b = (predictions[class_b_mask] == test_label[class_b_mask]).sum().item()
            
            total_correct_a += correct_predictions_a
            total_samples_a += class_a_mask.sum().item()
            
            total_correct_b += correct_predictions_b
            total_samples_b += class_b_mask.sum().item()
        
        closed_set_accuracy = total_correct_a / total_samples_a if total_samples_a > 0 else 0.0
        closed_set_acc = closed_set_accuracy*100
        open_set_accuracy = total_correct_b / total_samples_b if total_samples_b > 0 else 0.0
        open_set_acc = open_set_accuracy*100

        print(f"epoch {epoch+1} : open set prediction accuracy: {open_set_acc}")
        print(f"epoch {epoch+1} : closed set prediction accuracy: {closed_set_acc}")

        average_acc = (2*closed_set_acc*open_set_acc)/(closed_set_acc + open_set_acc)
        print(f"epoch {epoch+1} : harmonic score: {average_acc}")

        accuracy_file.write(f"Epoch {epoch+1} - Open Set Accuracy: {open_set_acc}%\n")
        accuracy_file.write(f"Epoch {epoch+1} - Closed Set Accuracy: {closed_set_acc}%\n")
        accuracy_file.write(f"Epoch {epoch+1} - Harmonic Score: {average_acc}%\n")
        accuracy_file.write("\n") 
        accuracy_file.flush()
        
        if average_acc > best_avg_acc:
            best_closed_set_acc = closed_set_acc
            best_open_set_acc = open_set_acc
            best_avg_acc = average_acc
            TEST_MODEL_PATH = Path("./test_models/pacs")
            TEST_MODEL_PATH.mkdir(parents=True, exist_ok=True)
            TEST_MODEL_NAME = "photo.pth"
            TEST_MODEL_SAVE_PATH = TEST_MODEL_PATH / TEST_MODEL_NAME
            print(f"Saving test_model with best harmonic score to: {TEST_MODEL_SAVE_PATH}")
            torch.save(obj=test_model.state_dict(), f=TEST_MODEL_SAVE_PATH) 
            
            print(f"Best open set prediction accuracy till now: {best_open_set_acc}")
            print(f"Best closed set prediction accuracy till now: {best_closed_set_acc}")
            print(f"Best harmonic score till now: {best_avg_acc}")
            accuracy_file.write(f"Epoch {epoch+1} - Best Open Set Accuracy till now : {best_open_set_acc}%\n")
            accuracy_file.write(f"Epoch {epoch+1} - Best Closed Set Accuracy till now: {best_closed_set_acc}%\n")
            accuracy_file.write(f"Epoch {epoch+1} - Best Harmonic Score now: {best_avg_acc}%\n")
            accuracy_file.write("\n") 
            accuracy_file.flush()
        else:
            print(f"Best open set prediction accuracy till now: {best_open_set_acc}")
            print(f"Best closed set prediction accuracy till now: {best_closed_set_acc}")
            print(f"Best harmonic score till now: {best_avg_acc}")
            accuracy_file.write(f"Epoch {epoch+1} - Best Open Set Accuracy till now : {best_open_set_acc}%\n")
            accuracy_file.write(f"Epoch {epoch+1} - Best Closed Set Accuracy till now: {best_closed_set_acc}%\n")
            accuracy_file.write(f"Epoch {epoch+1} - Best Harmonic Score now: {best_avg_acc}%\n")
            accuracy_file.write("\n") 
            accuracy_file.flush()

print(f"Best open set prediction accuracy till now: {best_open_set_acc}")
print(f"Best closed set prediction accuracy till now: {best_closed_set_acc}")
print(f"Best harmonic score till now: {best_avg_acc}")

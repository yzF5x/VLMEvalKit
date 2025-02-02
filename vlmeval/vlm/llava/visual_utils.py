
from io import BytesIO
from PIL import Image
import requests
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch.nn.functional as F

def aggregate_llm_attention(attn):
    '''Extract average attention vector'''
    avged = []
    for layer in attn:
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = torch.concat((
            # We zero the first entry because it's what's called
            # null attention (https://aclanthology.org/W19-4808.pdf)
            torch.tensor([0.]),
            # usually there's only one item in attns_per_head but
            # on the first generation, there's a row for each token
            # in the prompt as well, so take [-1]
            attns_per_head[-1][1:].cpu(),
            # attns_per_head[-1].cpu(),
            # add zero for the final generated token, which never
            # gets any attention
            torch.tensor([0.]),
        ))
        avged.append(vec / vec.sum())
    return torch.stack(avged).mean(dim=0)


def aggregate_visual_attention(attn, select_layer=-2, all_prev_layers=True):
    '''Assuming LLaVA-style `select_layer` which is -2 by default'''
    # print("attn : " , len(attn))
    if all_prev_layers:
        avged = []
        for i, layer in enumerate(attn):
            if i > len(attn) + select_layer:
                break
            layer_attns = layer.squeeze(0)
            attns_per_head = layer_attns.mean(dim=0)
            vec = attns_per_head[1:, 1:].cpu() # the first token is <CLS>
            avged.append(vec / vec.sum(-1, keepdim=True))
        return torch.stack(avged).mean(dim=0)
    else:
        layer = attn[select_layer]
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        vec = attns_per_head[1:, 1:].cpu()
        return vec / vec.sum(-1, keepdim=True)


def heterogenous_stack(vecs):
    '''Pad vectors with zeros then stack'''
    max_length = max(v.shape[0] for v in vecs)
    return torch.stack([
        torch.concat((v, torch.zeros(max_length - v.shape[0])))
        for v in vecs
    ])


def load_image(image_path_or_url):
    if image_path_or_url.startswith('http://') or image_path_or_url.startswith('https://'):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_path_or_url).convert('RGB')
    return image


def show_mask_on_image(img, mask):
    print("image : " , img.shape  , "MASK : " , mask.shape )
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_HSV)
    hm = np.float32(heatmap) / 255
    print(img.shape,hm.shape)
    cam = hm + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam), heatmap


def text_token_attention_vis(model , tokenizer , output_ids , images_list, input_ids, prompt):
    category = images_list[0].split("detection_orig/")[-1].split("/")[0]
    id = images_list[0].split("/")[-1].replace(".png","")
    id = category + "-" + id
    print(category, id)
    images = [Image.open(s).convert("RGB") for s in images_list]
    image = images[0].resize((336,336))

    aggregated_prompt_attention = []
    for i, layer in enumerate(output_ids["attentions"][0]):
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        cur = attns_per_head[:-1].cpu().clone()
        cur[1:, 0] = 0.
        cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
        aggregated_prompt_attention.append(cur)
    aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)

    # # llm_attn_matrix will be of torch.Size([N, N])
    # # where N is the total number of input (both image and text ones) + output tokens
    llm_attn_matrix = heterogenous_stack(
        [torch.tensor([1])]
        + list(aggregated_prompt_attention) 
        + list(map(aggregate_llm_attention, output_ids["attentions"]))
    )

    input_token_len = model.get_vision_tower().num_patches + len(input_ids[0]) - 1 # -1 for the <image> token
    vision_token_start = len(tokenizer(prompt.split("<image>")[0], return_tensors='pt')["input_ids"][0])
    vision_token_end = vision_token_start + model.get_vision_tower().num_patches
    output_token_len = len(output_ids["sequences"][0])
    output_token_start = input_token_len
    output_token_end = input_token_len + output_token_len
    # # visualize the llm attention matrix
    # # ===> adjust the gamma factor to enhance the visualization
    # #      higer gamma brings out more low attention values
    gamma_factor = 1
    enhanced_attn_m = np.power(llm_attn_matrix.numpy(), 1 / gamma_factor)

    fig, ax = plt.subplots(figsize=(10, 20), dpi=150)
    ax.imshow(enhanced_attn_m, vmin=enhanced_attn_m.min(), vmax=enhanced_attn_m.max(), interpolation="nearest")

    overall_attn_weights_over_vis_tokens = []
    for i, (row, token) in enumerate(
        zip(llm_attn_matrix[input_token_len:], output_ids["sequences"][0].tolist())
    ):
        overall_attn_weights_over_vis_tokens.append(
            row[vision_token_start:vision_token_end].sum().item())

    fig, ax = plt.subplots(figsize=(55, 5))
    ax.plot(overall_attn_weights_over_vis_tokens)
    ax.set_xticks(range(len(overall_attn_weights_over_vis_tokens)))
    ax.set_xticklabels(
        [tokenizer.decode(token, add_special_tokens=False).strip() for token in output_ids["sequences"][0].tolist()],
        rotation=90
    )
    ax.set_title("attn weights of each text token")
    plt.savefig(f'./outputs/vis/llava_v1.5_7b/{id}_text_attn.png', format='png', dpi=300, bbox_inches='tight')
    
def text_visual_token_attention_vis(model , tokenizer , output_ids ,images_list,input_ids,prompt):
    category = images_list[0].split("detection_orig/")[-1].split("/")[0]
    id = images_list[0].split("/")[-1].replace(".png","")
    id = category + "-" + id
    print(category, id)
    images = [Image.open(s).convert("RGB") for s in images_list]
    # image = images[0]
    image = images[0].resize((336,336))
    image_size = image.size
    aggregated_prompt_attention = []
    for i, layer in enumerate(output_ids["attentions"][0]):
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        cur = attns_per_head[:-1].cpu().clone()
        cur[1:, 0] = 0.
        cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
        aggregated_prompt_attention.append(cur)
    aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)

    # # llm_attn_matrix will be of torch.Size([N, N])
    # # where N is the total number of input (both image and text ones) + output tokens
    llm_attn_matrix = heterogenous_stack(
        [torch.tensor([1])]
        + list(aggregated_prompt_attention) 
        + list(map(aggregate_llm_attention, output_ids["attentions"]))
    )

    input_token_len = model.get_vision_tower().num_patches + len(input_ids[0]) - 1 # -1 for the <image> token
    vision_token_start = len(tokenizer(prompt.split("<image>")[0], return_tensors='pt')["input_ids"][0])
    vision_token_end = vision_token_start + model.get_vision_tower().num_patches
    output_token_len = len(output_ids["sequences"][0])
    output_token_start = input_token_len
    output_token_end = input_token_len + output_token_len
    
    vis_attn_matrix = aggregate_visual_attention(
        model.get_vision_tower().image_attentions,
        select_layer=model.get_vision_tower().select_layer,
        all_prev_layers=True  #False只返回特定layer的attention
    )
    grid_size = model.get_vision_tower().num_patches_per_side
    num_image_per_row = 8
    image_ratio = image_size[0] / image_size[1]
    num_rows = output_token_len // num_image_per_row + (1 if output_token_len % num_image_per_row != 0 else 0)
    fig, axes = plt.subplots(
        num_rows, num_image_per_row, 
        figsize=(10, (10 / num_image_per_row) * image_ratio * num_rows), 
        dpi=150
    )
    plt.subplots_adjust(wspace=0.05, hspace=0.2)

    vis_overlayed_with_attn = True

    output_token_inds = list(range(output_token_start, output_token_end))
    for i, ax in enumerate(axes.flatten()):
        if i >= output_token_len:
            ax.axis("off")
            continue

        target_token_ind = output_token_inds[i]
        attn_weights_over_vis_tokens = llm_attn_matrix[target_token_ind][vision_token_start:vision_token_end]
        attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_weights_over_vis_tokens.sum()

        attn_over_image = []
        for weight, vis_attn in zip(attn_weights_over_vis_tokens, vis_attn_matrix):
            vis_attn = vis_attn.reshape(grid_size, grid_size)
            attn_over_image.append(vis_attn * weight)
        attn_over_image = torch.stack(attn_over_image).sum(dim=0)
        attn_over_image = attn_over_image / attn_over_image.max()
        attn_over_image = attn_over_image.to(torch.float32)
        attn_over_image = F.interpolate(
            attn_over_image.unsqueeze(0).unsqueeze(0), 
            size=image.size, 
            mode='nearest', 
        ).squeeze()
        attn_over_image = attn_over_image.numpy()
        # attn_over_image = np.transpose(attn_over_image)
        # print("image : " , image.size  , "attn_over_image : " , attn_over_image.shape )
        np_img = np.array(image)
        np_img = np_img[:, :, ::-1]
        img_with_attn, heatmap = show_mask_on_image(np_img, attn_over_image)
        ax.imshow(heatmap if not vis_overlayed_with_attn else img_with_attn)
        ax.set_title(
            tokenizer.decode(output_ids["sequences"][0][i], add_special_tokens=False).strip(),
            fontsize=7,
            pad=1
        )
        ax.axis("off")
    plt.savefig(f'./outputs/vis/llava_v1.5_7b/{id}_text_token_vis_attention_heatmap.png', format='png', dpi=300, bbox_inches='tight')
    
def visual_attention_vis(model , tokenizer , output_ids ,images_list,input_ids,prompt):
    category = images_list[0].split("detection_orig/")[-1].split("/")[0]
    id = images_list[0].split("/")[-1].replace(".png","")
    id = category + "-" + id
    print(category, id)
    images = [Image.open(s).convert("RGB") for s in images_list]
    image = images[0].resize((336,336))
    vis_attn_matrix = aggregate_visual_attention(
        model.get_vision_tower().image_attentions,
        select_layer=model.get_vision_tower().select_layer,
        all_prev_layers=True  #False只返回特定layer的attention
    )
    aggregated_prompt_attention = []
    for i, layer in enumerate(output_ids["attentions"][0]):
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        cur = attns_per_head[:-1].cpu().clone()
        cur[1:, 0] = 0.
        cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
        aggregated_prompt_attention.append(cur)
    aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)

    # # llm_attn_matrix will be of torch.Size([N, N])
    # # where N is the total number of input (both image and text ones) + output tokens
    llm_attn_matrix = heterogenous_stack(
        [torch.tensor([1])]
        + list(aggregated_prompt_attention) 
        + list(map(aggregate_llm_attention, output_ids["attentions"]))
    )
    # # ---
    
    # # identify length or index of tokens
    input_token_len = model.get_vision_tower().num_patches + len(input_ids[0]) - 1 # -1 for the <image> token
    vision_token_start = len(tokenizer(prompt.split("<image>")[0], return_tensors='pt')["input_ids"][0])
    vision_token_end = vision_token_start + model.get_vision_tower().num_patches
    output_token_len = len(output_ids["sequences"][0])
    output_token_start = input_token_len
    output_token_end = input_token_len + output_token_len
    
    output_token_inds = list(range(output_token_start, output_token_end))
    aggregated_attn_weights = torch.zeros_like(vis_attn_matrix[0])  
    for target_token_ind in range(output_token_start, output_token_end):
        attn_weights_over_vis_tokens = llm_attn_matrix[target_token_ind][vision_token_start:vision_token_end]
        attn_weights_over_vis_tokens = attn_weights_over_vis_tokens / attn_weights_over_vis_tokens.sum() 
        aggregated_attn_weights += attn_weights_over_vis_tokens 

    aggregated_attn_weights = aggregated_attn_weights / aggregated_attn_weights.sum()
    grid_size = model.get_vision_tower().num_patches_per_side

    attn_over_image = []
    for weight, vis_attn in zip(aggregated_attn_weights, vis_attn_matrix):
        # vis_attn : [576]  reshape : [24,24]
        vis_attn = vis_attn.reshape(grid_size, grid_size) 
        attn_over_image.append(vis_attn * weight) 
    attn_over_image = torch.stack(attn_over_image).sum(dim=0)  
    attn_over_image = attn_over_image / attn_over_image.max()  
    attn_over_image = attn_over_image.to(torch.float32)
    attn_over_image = F.interpolate(
        attn_over_image.unsqueeze(0).unsqueeze(0), 
        size=image.size, 
        mode='nearest'  # 或 'bicubic'
    ).squeeze()
    print("global:image : " , image.size  , "attn_over_image : " , attn_over_image.shape )
    attn_over_image_np = attn_over_image.numpy()
    # attn_over_image = np.transpose(attn_over_image)
    print("global :image : " , image.size  , "attn_over_image : " , attn_over_image.shape )

    np_img = np.array(image)
    np_img = np_img[:, :, ::-1]  
    
    img_with_attn, heatmap = show_mask_on_image(np_img, attn_over_image_np) 
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xticks([])  
    ax.set_yticks([])  
    plt.imshow(heatmap, cmap='jet')
    plt.savefig(f"./outputs/vis/llava_v1.5_7b/{id}_global_attn_heatmap.png" , format='png')
    plt.imshow(img_with_attn)
    plt.savefig(f'./outputs/vis/llava_v1.5_7b/{id}_global_attn_heatmap_with_img.png', format='png', dpi=300)
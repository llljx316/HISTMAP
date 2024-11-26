
# from sam2.build_sam import build_sam2
# from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
# import torch
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# from PIL import Image


# #open photo
# image = Image.open('../dataset/Beinecke Library/3431755.jpg')
# image = np.array(image.convert("RGB"))



# #load checkpoint
# sam2_checkpoint = "../checkpoints/sam2_hiera_large.pt"
# model_cfg = "sam2_hiera_l.yaml"

# sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)

# mask_generator = SAM2AutomaticMaskGenerator(sam2)

# masks = mask_generator.generate(image)

# print(len(masks))
# print(masks[0].keys())

# plt.figure(figsize=(20, 20))
# plt.imshow(image)
# show_anns(masks)
# plt.axis('off')
# plt.show() 


import os
# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import pickle
import cv2
import json


#show segment result

def show_anns(anns, borders=True, save_path=None):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask 
        if borders:
            contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            # Try to smooth contours
            contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
            cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

    ax.imshow(img)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)

def filtered_anns_to_file(anns, filepath):
    filtered_anns = [
        {
            'predicted_iou': ann['predicted_iou'],
            'stability_score': ann['stability_score']
        }
        for ann in anns
    ]
    with open(filepath, 'wb') as pickle_file:
        pickle.dump(filtered_anns, pickle_file)

def filtered_anns_to_json(anns, filepath):
    filtered_anns = [
        {
            'predicted_iou': ann['predicted_iou'],
            'stability_score': ann['stability_score']
        }
        for ann in anns
    ]
    with open(filepath, 'wb') as f:
        json.dump(filtered_anns, f)

def apply_mask_and_crop(img, mask, crop_size=(100, 100), save_path=None):
    """
    读取图像，应用掩码，裁剪为指定大小的图像。

    参数:
    - image_path: 图像的路径
    - mask: 掩码数组，大小应与图像的宽高相同，布尔数组或0/1数组
    - crop_size: 要裁剪的输出图像大小 (宽, 高)
    
    返回:
    - 裁剪并应用掩码的图像
    """
    # 创建一个黑色的空白图像
    masked_img = np.zeros_like(img)

    # 应用掩码，将掩码区域的图像像素保留下来
    masked_img[mask == 1] = img[mask == 1]

    # 找到掩码区域的最小外接矩形坐标
    x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))

    # 裁剪图像，根据掩码区域裁剪
    cropped_img = masked_img[y:y+h, x:x+w]

    if save_path is not None:
        cv2.imwrite(save_path, cropped_img)
        print(f"Image saved to {save_path}")
    
    return cropped_img

    # # 调整裁剪后的图像到指定大小
    # cropped_resized_img = cv2.resize(cropped_img, crop_size)

    # # 返回裁剪并调整大小后的图像
    # return cropped_resized_img




def generate_input_point(img):
    # 询问用户希望输入多少个点
    num_points = int(input("请输入您想要选择的点的数量："))

    plt.imshow(img, cmap='gray')
    pts = plt.ginput(n=num_points)  # 用户选择指定数量的点
    print(f"Selected points: {pts}")
    plt.show()
    return pts, [1]*len(pts)



def pointMaskGenerator(sam2, image):

    mask_generator = SAM2AutomaticMaskGenerator(sam2)
    input_point, input_label = generate_input_point(image)
    masks = mask_generator.generate(image)

    print(len(masks))
    print(masks[0].keys())

    save_path = './segres/'+os.path.splitext(os.path.basename(image_path))[0]+'/'
    #no exist then create
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    filtered_anns_to_file(masks, save_path+'anns.pickle')
    # filtered_anns_to_json(masks, save_path+'anns.json')
    for i,mask in enumerate(masks):
        apply_mask_and_crop(image, mask['segmentation'], save_path=save_path+f'{i}.png')
    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks, save_path=save_path+'final.png')
    plt.axis('off')
    # plt.show() 

if __name__ == '__main__':
    #Environment Set-up

    # select the device for computation
    if torch.cuda.is_available():
        device = torch.device("cuda:1")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    if device.type == "cuda":
        # use bfloat16 for the entire notebook
        torch.autocast("cuda").__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(
            "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
            "give numerically different outputs and sometimes degraded performance on MPS. "
            "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
        )

    np.random.seed(3)


    #show example
    image_path = 'images/3431780.jpg'
    image = Image.open(image_path)
    image = np.array(image.convert("RGB"))

    #automatic generation
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

    sam2_checkpoint = "segment-anything-2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"

    sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    pointMaskGenerator(sam2, image)
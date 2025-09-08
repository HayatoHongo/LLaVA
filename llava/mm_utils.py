from PIL import Image
from io import BytesIO
import base64
import torch
import math
import ast

from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX


def select_best_resolution(original_size, possible_resolutions):

    print("current file path", "llava/mm_utils.py")
    print("def select_best_resolution(original_size, possible_resolutions)")
    print("original_size\n", original_size)
    print("possible_resolutions\n", possible_resolutions)
    """
    Selects the best resolution from a list of possible resolutions based on the original size.

    Args:
        original_size (tuple): The original size of the image in the format (width, height).
        possible_resolutions (list): A list of possible resolutions in the format [(width1, height1), (width2, height2), ...].

    Returns:
        tuple: The best fit resolution in the format (width, height).
    """
    original_width, original_height = original_size
    best_fit = None
    max_effective_resolution = 0
    min_wasted_resolution = float('inf')

    for width, height in possible_resolutions:
        scale = min(width / original_width, height / original_height)
        downscaled_width, downscaled_height = int(original_width * scale), int(original_height * scale)
        effective_resolution = min(downscaled_width * downscaled_height, original_width * original_height)
        wasted_resolution = (width * height) - effective_resolution

        if effective_resolution > max_effective_resolution or (effective_resolution == max_effective_resolution and wasted_resolution < min_wasted_resolution):
            max_effective_resolution = effective_resolution
            min_wasted_resolution = wasted_resolution
            best_fit = (width, height)

    print("best_fit (return)\n", best_fit)
    return best_fit


def resize_and_pad_image(image, target_resolution):

    print("current file path", "llava/mm_utils.py")
    print("def resize_and_pad_image(image, target_resolution)")
    print("image\n", image)
    print("target_resolution\n", target_resolution)
    """
    Resize and pad an image to a target resolution while maintaining aspect ratio.

    Args:
        image (PIL.Image.Image): The input image.
        target_resolution (tuple): The target resolution (width, height) of the image.

    Returns:
        PIL.Image.Image: The resized and padded image.
    """
    original_width, original_height = image.size
    target_width, target_height = target_resolution

    scale_w = target_width / original_width
    scale_h = target_height / original_height

    if scale_w < scale_h:
        new_width = target_width
        new_height = min(math.ceil(original_height * scale_w), target_height)
    else:
        new_height = target_height
        new_width = min(math.ceil(original_width * scale_h), target_width)

    # Resize the image
    resized_image = image.resize((new_width, new_height))

    new_image = Image.new('RGB', (target_width, target_height), (0, 0, 0))
    paste_x = (target_width - new_width) // 2
    paste_y = (target_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    print("new_image (return)\n", new_image)
    return new_image


def divide_to_patches(image, patch_size):

    print("current file path", "llava/mm_utils.py")
    print("def divide_to_patches(image, patch_size)")
    print("image\n", image)
    print("patch_size\n", patch_size)
    """
    Divides an image into patches of a specified size.

    Args:
        image (PIL.Image.Image): The input image.
        patch_size (int): The size of each patch.

    Returns:
        list: A list of PIL.Image.Image objects representing the patches.
    """
    patches = []
    width, height = image.size
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            box = (j, i, j + patch_size, i + patch_size)
            patch = image.crop(box)
            patches.append(patch)

    print("patches (return)\n", patches)
    return patches


def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):

    print("current file path", "llava/mm_utils.py")
    print("def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size)")
    print("image_size\n", image_size)
    print("grid_pinpoints\n", grid_pinpoints)
    print("patch_size\n", patch_size)
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (tuple): The size of the input image in the format (width, height).
        grid_pinpoints (str): A string representation of a list of possible resolutions.
        patch_size (int): The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    width, height = select_best_resolution(image_size, possible_resolutions)
    result = (width // patch_size, height // patch_size)
    print("result (return)\n", result)
    return result


def process_anyres_image(image, processor, grid_pinpoints):

    print("current file path", "llava/mm_utils.py")
    print("def process_anyres_image(image, processor, grid_pinpoints)")
    print("image\n", image)
    print("processor\n", processor)
    print("grid_pinpoints\n", grid_pinpoints)
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size['height'])

    image_original_resize = image.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                     for image_patch in image_patches]
    result = torch.stack(image_patches, dim=0)
    print("result (return)\n", result)
    return result


def load_image_from_base64(image):
    print("current file path", "llava/mm_utils.py")
    print("def load_image_from_base64(image)")
    print("image\n", image)
    result = Image.open(BytesIO(base64.b64decode(image)))
    print("result (return)\n", result)
    return result


def expand2square(pil_img, background_color):

    print("current file path", "llava/mm_utils.py")
    print("def expand2square(pil_img, background_color)")
    print("pil_img\n", pil_img)
    print("background_color\n", background_color)
    width, height = pil_img.size
    if width == height:
        print("pil_img (return)\n", pil_img)
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        print("result (return)\n", result)
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        print("result (return)\n", result)
        return result


def process_images(images, image_processor, model_cfg):

    print("current file path", "llava/mm_utils.py")
    print("def process_images(images, image_processor, model_cfg)")
    print("images\n", images)
    print("image_processor\n", image_processor)
    print("model_cfg\n", model_cfg)
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    elif image_aspect_ratio == "anyres":
        for image in images:
            image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    print("new_images (return)\n", new_images)
    return new_images


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):

    print("current file path", "llava/mm_utils.py")
    print("def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None)")
    print("prompt\n", prompt)
    print("tokenizer\n", tokenizer)
    print("image_token_index\n", image_token_index)
    print("return_tensors\n", return_tensors)
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    print("input_ids (return)\n", input_ids)
    return input_ids


def get_model_name_from_path(model_path):
    print("current file path", "llava/mm_utils.py")
    print("def get_model_name_from_path(model_path)")
    print("model_path\n", model_path)
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        result = model_paths[-2] + "_" + model_paths[-1]
    else:
        result = model_paths[-1]
    print("result (return)\n", result)
    return result

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):

        print("current file path", "llava/mm_utils.py")
        print("def __init__(self, keywords, tokenizer, input_ids)")
        print("keywords\n", keywords)
        print("tokenizer\n", tokenizer)
        print("input_ids\n", input_ids)
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:

        print("current file path", "llava/mm_utils.py")
        print("def call_for_batch(self, output_ids, scores, **kwargs)")
        print("output_ids\n", output_ids)
        print("scores\n", scores)
        print("kwargs\n", kwargs)
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                print("return True")
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                print("return True")
                return True
        print("return False")
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:

        print("current file path", "llava/mm_utils.py")
        print("def __call__(self, output_ids, scores, **kwargs)")
        print("output_ids\n", output_ids)
        print("scores\n", scores)
        print("kwargs\n", kwargs)
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        print("outputs (return)\n", outputs)
        return all(outputs)

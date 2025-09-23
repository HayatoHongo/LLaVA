def prepare_inputs_labels_for_multimodal(
    self, input_ids, position_ids, attention_mask, past_key_values, labels,
    images, image_sizes=None
):
    print("current file path", "llava/model/llava_arch.py")
    """
    llava/llava/model/language_model/llava_llama.py
    """
    print("def LlavaMetaForCausalLM(ABC).prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes=None)")  # not found
    print("input_ids\n", input_ids)
    """
    tensor([[    1,  -200,   278, 25616, 26624,   297,   902, 19430, 11105, 29879,
             10508,  1596, 23425,   278,  3700,   322,  6567,   310,   263,  6114,
               411,  2654, 11315,    13]])
    """

    print("position_ids\n", position_ids)  # None
    print("attention_mask\n", attention_mask)
    """
    tensor([[True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True]])
    """

    print("past_key_values\n", past_key_values)  # None
    print("labels\n", labels)
    """
    tensor([[ -100,  -100,   278, 25616, 26624,   297,   902, 19430, 11105, 29879,
             10508,  1596, 23425,   278,  3700,   322,  6567,   310,   263,  6114,
               411,  2654, 11315,    13]])
    """

    print("images\n", images)
    """
    tensor([[[[ 0.0325,  0.0325,  0.0325,  ..., -0.7120, -0.3616, -0.1280],
              [ 0.0325,  0.0325,  0.0325,  ..., -0.3908, -0.1718, -0.0259],
              [ 0.0325,  0.0325,  0.0325,  ..., -0.0113,  0.0471,  0.0909],
              ...,
              [-1.0331, -1.0331, -1.0331,  ..., -1.0623, -1.0623, -1.0623],
              [-1.0477, -1.0331, -1.0331,  ..., -1.0623, -1.0623, -1.0623],
              [-1.0477, -1.0331, -1.0331,  ..., -1.0623, -1.0623, -1.0623]],
    
             [[ 0.3190,  0.3190,  0.3190,  ..., -0.3864, -0.0112,  0.2139],
              [ 0.3190,  0.3190,  0.3190,  ..., -0.0712,  0.1539,  0.3190],
              [ 0.3190,  0.3190,  0.3190,  ...,  0.2890,  0.3640,  0.4390],
              ...,
              [-1.0167, -1.0167, -1.0167,  ..., -1.0017, -1.0017, -1.0017],
              [-1.0317, -1.0167, -1.0167,  ..., -1.0017, -1.0017, -1.0017],
              [-1.0317, -1.0167, -1.0167,  ..., -1.0017, -1.0017, -1.0017]],
    
             [[ 0.9656,  0.9656,  0.9656,  ...,  0.0982,  0.4537,  0.6670],
              [ 0.9656,  0.9656,  0.9656,  ...,  0.3968,  0.6101,  0.7523],
              [ 0.9656,  0.9656,  0.9656,  ...,  0.7523,  0.8092,  0.8377],
              ...,
              [-0.3711, -0.3853, -0.3995,  ..., -0.4279, -0.4279, -0.4279],
              [-0.3711, -0.3711, -0.3853,  ..., -0.4279, -0.4279, -0.4279],
              [-0.3853, -0.3711, -0.3711,  ..., -0.4279, -0.4279, -0.4279]]]])
    """

    print("image_sizes\n", image_sizes)  # None
    vision_tower = self.get_vision_tower()
    print("vision_tower\n", vision_tower)
    """
    CLIPVisionTower(
      (vision_tower): CLIPVisionModel(
        (vision_model): CLIPVisionTransformer(
          (embeddings): CLIPVisionEmbeddings(
            (patch_embedding): Conv2d(3, 1024, kernel_size=(14, 14), stride=(14, 14), bias=False)
            (position_embedding): Embedding(577, 1024)
          )
          (pre_layrnorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
          (encoder): CLIPEncoder(
            (layers): ModuleList(
              (0-23): 24 x CLIPEncoderLayer(
                (self_attn): CLIPAttention(
                  (k_proj): Linear(in_features=1024, out_features=1024, bias=True)
                  (v_proj): Linear(in_features=1024, out_features=1024, bias=True)
                  (q_proj): Linear(in_features=1024, out_features=1024, bias=True)
                  (out_proj): Linear(in_features=1024, out_features=1024, bias=True)
                )
                (layer_norm1): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
                (mlp): CLIPMLP(
                  (activation_fn): QuickGELUActivation()
                  (fc1): Linear(in_features=1024, out_features=4096, bias=True)
                  (fc2): Linear(in_features=4096, out_features=1024, bias=True)
                )
                (layer_norm2): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
              )
            )
          )
          (post_layernorm): LayerNorm((1024,), eps=1e-05, elementwise_affine=True)
        )
      )
    )
    """

    print(f"【COND】 vision_tower_is_None={vision_tower is None} images_is_None={images is None} input_ids_shape_1_eq_1={input_ids.shape[1] == 1}")
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        pass

    print("【COND】type(images)\n", type(images))  # <class 'torch.Tensor'>
    print("【COND】images.ndim\n", images.ndim)  # 4
    if type(images) is list or images.ndim == 5:
        pass
    else:
        # 【ENTER】
        print("【ENTER】else of if type(images) is list or images.ndim == 5:")
        image_features = self.encode_images(images)
        print("image_features after encode_images shape \n", image_features.shape)  # torch.Size([1, 576, 2048])
        print("image_features after encode_images\n", image_features)
        """
        tensor([[[-0.1943,  0.1157, -0.0747,  ...,  0.0027, -0.1691, -0.3439],
                 [ 0.0437,  0.1717, -0.0998,  ...,  0.0930, -0.1386, -0.0731],
                 [-0.0505,  0.1592, -0.0982,  ...,  0.0866, -0.1123, -0.2177],
                 ...,
                 [-0.0182,  0.0850, -0.0556,  ...,  0.0622, -0.1969,  0.0129],
                 [-0.0651,  0.0586, -0.1218,  ..., -0.0614, -0.1158, -0.0104],
                 [ 0.0863,  0.0081, -0.1651,  ..., -0.2040, -0.0455,  0.0618]]],
               grad_fn=<ViewBackward0>)
        """
        print("【EXIT】else of if type(images) is list or images.ndim == 5:")

    # TODO: image start / end is not implemented here to support pretraining.
    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
        print("【ENTER】if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):")  # not found
        raise NotImplementedError

    # Let's just add dummy tensors if they do not exist,
    # it is a headache to deal with None all the time.
    # But it is not ideal, and if you have a better idea,
    # please open an issue / submit a PR, thanks.

    print("labels before\n", labels)
    """
    tensor([[ -100,  -100,   278, 25616, 26624,   297,   902, 19430, 11105, 29879,
             10508,  1596, 23425,   278,  3700,   322,  6567,   310,   263,  6114,
               411,  2654, 11315,    13]])
    """

    print("position_ids before\n", position_ids)  # None

    print("attention_mask before\n", attention_mask)
    """
    tensor([[True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True]])
    """

    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        pass
    else:
        # 【ENTER】
        print("【ENTER】else of if attention_mask is None:")
        attention_mask = attention_mask.bool()
 
        print("attention_mask（after）shape \n", attention_mask.shape)  # torch.Size([1, 24])
        print("attention_mask (after)\n", attention_mask)
        """
        tensor([[True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True]])
        """
        print("【EXIT】else of if attention_mask is None:")
    if position_ids is None:
        print("【ENTER】if position_ids is None:")
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        print("position_ids (after) shape \n", position_ids.shape)  # torch.Size([24])
        print("position_ids (after)\n", position_ids)
        """
        tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
                18, 19, 20, 21, 22, 23])
        """
        print("【EXIT】if position_ids is None:")
    print(f"【COND】 labels_is_None={labels is None}")
    if labels is None:
        pass

    # remove the padding using attention_mask -- FIXME
    _input_ids = input_ids
    input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
    print("input_ids after removing padding\n", input_ids)
    """
    [tensor([    1,  -200,   278, 25616, 26624,   297,   902, 19430, 11105, 29879,
            10508,  1596, 23425,   278,  3700,   322,  6567,   310,   263,  6114,
              411,  2654, 11315,    13])]
    """

    print("labels after removing padding\n", labels)
    """
    [tensor([ -100,  -100,   278, 25616, 26624,   297,   902, 19430, 11105, 29879,
            10508,  1596, 23425,   278,  3700,   322,  6567,   310,   263,  6114,
              411,  2654, 11315,    13])]
    """


    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        print("cur_input_ids shape\n", cur_input_ids.shape)   # torch.Size([24])
        print("cur_input_ids\n", cur_input_ids)
        """
        tensor([    1,  -200,   278, 25616, 26624,   297,   902, 19430, 11105, 29879,
                10508,  1596, 23425,   278,  3700,   322,  6567,   310,   263,  6114,
                  411,  2654, 11315,    13])
        """
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        print("【COND】num_images:", num_images)  # tensor(1)
        if num_images == 0:
            print("【ENTER】if num_images == 0:")
            cur_image_features = image_features[cur_image_idx]
            cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
            cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
            new_input_embeds.append(cur_input_embeds)
            new_labels.append(labels[batch_idx])
            cur_image_idx += 1
            print("【EXIT】if num_images == 0:")
            continue

        image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
        print("image_token_indices\n", image_token_indices)  # [-1, 1, 24]
        print("len image_token_indices", len(image_token_indices))   # 3
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        print("cur_labels\n", cur_labels)
        """
        tensor([ -100,  -100,   278, 25616, 26624,   297,   902, 19430, 11105, 29879,
                10508,  1596, 23425,   278,  3700,   322,  6567,   310,   263,  6114,
                  411,  2654, 11315,    13])
        """
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
        print("cur_input_ids_noim (after)\n", cur_input_ids_noim)
        """
        [tensor([1]), tensor([  278, 25616, 26624,   297,   902, 19430, 11105, 29879, 10508,  1596,
                23425,   278,  3700,   322,  6567,   310,   263,  6114,   411,  2654,
                11315,    13])]
        """
        print("cur_labels_noim (after) \n", cur_labels_noim)
        """
        [tensor([-100]), tensor([  278, 25616, 26624,   297,   902, 19430, 11105, 29879, 10508,  1596,
                23425,   278,  3700,   322,  6567,   310,   263,  6114,   411,  2654,
                11315,    13])]
        """
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        print("split_sizes\n", split_sizes)  # [1, 22]
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        print("cur_input_embeds shape\n", cur_input_embeds.shape)  # torch.Size([23, 2048])
        print("cur_input_embeds\n", cur_input_embeds)
        """
        tensor([[-1.0910e-03,  1.9302e-03, -1.6632e-03,  ...,  1.9932e-04,
                 -6.5231e-04, -4.9973e-04],
                [ 7.0801e-03,  1.0452e-03,  6.0425e-03,  ...,  3.9673e-03,
                  1.2817e-03, -1.1215e-03],
                [-2.2949e-02, -2.6226e-05,  6.8359e-03,  ..., -2.4658e-02,
                 -9.4604e-03,  1.5869e-02],
                ...,
                [ 2.1240e-02, -2.2705e-02, -1.4221e-02,  ..., -2.8229e-03,
                 -8.3618e-03, -9.4604e-03],
                [ 3.7079e-03, -3.6011e-03,  9.0332e-03,  ..., -1.3672e-02,
                 -2.5177e-03, -8.0566e-03],
                [-6.3705e-04, -1.0605e-03, -1.1841e-02,  ...,  2.1935e-04,
                 -7.3242e-04,  2.7924e-03]], requires_grad=True)
        """
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        print("cur_input_embeds_no_im\n", cur_input_embeds_no_im)
        """
        (tensor([[-0.0011,  0.0019, -0.0017,  ...,  0.0002, -0.0007, -0.0005]],
               grad_fn=<SplitWithSizesBackward0>), tensor([[ 7.0801e-03,  1.0452e-03,  6.0425e-03,  ...,  3.9673e-03,
                  1.2817e-03, -1.1215e-03],
                [-2.2949e-02, -2.6226e-05,  6.8359e-03,  ..., -2.4658e-02,
                 -9.4604e-03,  1.5869e-02],
                [-2.3499e-03,  1.4893e-02, -2.0447e-03,  ..., -8.6060e-03,
                  2.3193e-03,  3.0670e-03],
                ...,
                [ 2.1240e-02, -2.2705e-02, -1.4221e-02,  ..., -2.8229e-03,
                 -8.3618e-03, -9.4604e-03],
                [ 3.7079e-03, -3.6011e-03,  9.0332e-03,  ..., -1.3672e-02,
                 -2.5177e-03, -8.0566e-03],
                [-6.3705e-04, -1.0605e-03, -1.1841e-02,  ...,  2.1935e-04,
                 -7.3242e-04,  2.7924e-03]], grad_fn=<SplitWithSizesBackward0>))
        """
        cur_new_input_embeds = []
        cur_new_labels = []

        for i in range(num_images + 1):
            cur_new_input_embeds.append(cur_input_embeds_no_im[i])
            cur_new_labels.append(cur_labels_noim[i])
            print(f"【COND】 i={i} num_images={num_images}")
            if i < num_images:
                print("【ENTER】if i < num_images:")
                cur_image_features = image_features[cur_image_idx]
                cur_image_idx += 1
                cur_new_input_embeds.append(cur_image_features)
                cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                print("【EXIT】if i < num_images:")

        print("cur_new_input_embeds (before cat) shape\n", [x.shape for x in cur_new_input_embeds])
        """
        [torch.Size([1, 2048]), torch.Size([576, 2048]), torch.Size([22, 2048])]
        """
        print("cur_new_input_embeds (before cat)\n", cur_new_input_embeds)
        """
        [tensor([[-0.0011,  0.0019, -0.0017,  ...,  0.0002, -0.0007, -0.0005]],
               grad_fn=<SplitWithSizesBackward0>), tensor([[-0.1943,  0.1157, -0.0747,  ...,  0.0027, -0.1691, -0.3439],
                [ 0.0437,  0.1717, -0.0998,  ...,  0.0930, -0.1386, -0.0731],
                [-0.0505,  0.1592, -0.0982,  ...,  0.0866, -0.1123, -0.2177],
                ...,
                [-0.0182,  0.0850, -0.0556,  ...,  0.0622, -0.1969,  0.0129],
                [-0.0651,  0.0586, -0.1218,  ..., -0.0614, -0.1158, -0.0104],
                [ 0.0863,  0.0081, -0.1651,  ..., -0.2040, -0.0455,  0.0618]],
               grad_fn=<SelectBackward0>), tensor([[ 7.0801e-03,  1.0452e-03,  6.0425e-03,  ...,  3.9673e-03,
                  1.2817e-03, -1.1215e-03],
                [-2.2949e-02, -2.6226e-05,  6.8359e-03,  ..., -2.4658e-02,
                 -9.4604e-03,  1.5869e-02],
                [-2.3499e-03,  1.4893e-02, -2.0447e-03,  ..., -8.6060e-03,
                  2.3193e-03,  3.0670e-03],
                ...,
                [ 2.1240e-02, -2.2705e-02, -1.4221e-02,  ..., -2.8229e-03,
                 -8.3618e-03, -9.4604e-03],
                [ 3.7079e-03, -3.6011e-03,  9.0332e-03,  ..., -1.3672e-02,
                 -2.5177e-03, -8.0566e-03],
                [-6.3705e-04, -1.0605e-03, -1.1841e-02,  ...,  2.1935e-04,
                 -7.3242e-04,  2.7924e-03]], grad_fn=<SplitWithSizesBackward0>)]
        """

        print("cur_new_labels (before cat) shape\n", [x.shape for x in cur_new_labels])
        """
        [torch.Size([1]), torch.Size([576]), torch.Size([22])]
        """
        print("cur_new_labels (before cat)\n", cur_new_labels)
        """
        [tensor([-100]), tensor([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]), tensor([  278, 25616, 26624,   297,   902, 19430, 11105, 29879, 10508,  1596,
                23425,   278,  3700,   322,  6567,   310,   263,  6114,   411,  2654,
                11315,    13])]
        """
        cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        print("cur_new_input_embeds (after cat) shape\n", cur_new_input_embeds.shape)  # torch.Size([599, 2048])
        print("cur_new_input_embeds (after cat)\n", cur_new_input_embeds)
        """
        tensor([[-1.0910e-03,  1.9302e-03, -1.6632e-03,  ...,  1.9932e-04,
                 -6.5231e-04, -4.9973e-04],
                [-1.9428e-01,  1.1569e-01, -7.4740e-02,  ...,  2.6653e-03,
                 -1.6907e-01, -3.4387e-01],
                [ 4.3680e-02,  1.7172e-01, -9.9813e-02,  ...,  9.3004e-02,
                 -1.3859e-01, -7.3106e-02],
                ...,
                [ 2.1240e-02, -2.2705e-02, -1.4221e-02,  ..., -2.8229e-03,
                 -8.3618e-03, -9.4604e-03],
                [ 3.7079e-03, -3.6011e-03,  9.0332e-03,  ..., -1.3672e-02,
                 -2.5177e-03, -8.0566e-03],
                [-6.3705e-04, -1.0605e-03, -1.1841e-02,  ...,  2.1935e-04,
                 -7.3242e-04,  2.7924e-03]], grad_fn=<CatBackward0>)
        """

        print("cur_new_labels (after cat) shape\n", cur_new_labels.shape)  # torch.Size([599])
        print("cur_new_labels (after cat)\n", cur_new_labels)
        """
        tensor([ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,   278, 25616, 26624,
                  297,   902, 19430, 11105, 29879, 10508,  1596, 23425,   278,  3700,
                  322,  6567,   310,   263,  6114,   411,  2654, 11315,    13])
        """

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)
        print("new_input_embeds (so far) shape\n", [x.shape for x in new_input_embeds])  # [torch.Size([599, 2048])]
        print("new_input_embeds (so far)\n", new_input_embeds)
        """
        [tensor([[-1.0910e-03,  1.9302e-03, -1.6632e-03,  ...,  1.9932e-04,
                 -6.5231e-04, -4.9973e-04],
                [-1.9428e-01,  1.1569e-01, -7.4740e-02,  ...,  2.6653e-03,
                 -1.6907e-01, -3.4387e-01],
                [ 4.3680e-02,  1.7172e-01, -9.9813e-02,  ...,  9.3004e-02,
                 -1.3859e-01, -7.3106e-02],
                ...,
                [ 2.1240e-02, -2.2705e-02, -1.4221e-02,  ..., -2.8229e-03,
                 -8.3618e-03, -9.4604e-03],
                [ 3.7079e-03, -3.6011e-03,  9.0332e-03,  ..., -1.3672e-02,
                 -2.5177e-03, -8.0566e-03],
                [-6.3705e-04, -1.0605e-03, -1.1841e-02,  ...,  2.1935e-04,
                 -7.3242e-04,  2.7924e-03]], grad_fn=<CatBackward0>)]
        """

        print("new_labels (so far) shape\n", [x.shape for x in new_labels])  # [torch.Size([599])]
        print("new_labels (so far)\n", new_labels)
        """
        [tensor([ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                 -100,  -100,  -100,  -100,  -100,  -100,  -100,   278, 25616, 26624,
                  297,   902, 19430, 11105, 29879, 10508,  1596, 23425,   278,  3700,
                  322,  6567,   310,   263,  6114,   411,  2654, 11315,    13])]
        """

    # Truncate sequences to max length as image embeddings can make the sequence longer
    tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
    print(f"【COND】 tokenizer_model_max_length_is_not_None={tokenizer_model_max_length is not None}")
    if tokenizer_model_max_length is not None:
        print("【ENTER】if tokenizer_model_max_length is not None:")
        new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
        new_labels = [x[:tokenizer_model_max_length] for x in new_labels]
        print("【EXIT】if tokenizer_model_max_length is not None:")

    # Combine them
    max_len = max(x.shape[0] for x in new_input_embeds)
    print("max_len\n", max_len)  # 599
    batch_size = len(new_input_embeds)
    print("batch_size\n", batch_size)  # 1

    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    print("new_labels_padded (before) shape\n", new_labels_padded.shape)  # torch.Size([1, 599])
    print("new_labels_padded (before)\n", new_labels_padded)
    """
    tensor([[-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100,
             -100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100]])
    """
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    print("attention_mask (before) shape\n", attention_mask.shape)  # torch.Size([1, 599])
    print("attention_mask (before)\n", attention_mask)
    """
    tensor([[False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False, False,
             False, False, False, False, False, False, False, False, False]])
    """
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
    print("position_ids (before) shape\n", position_ids.shape)  # torch.Size([1, 599])
    print("position_ids (before)\n", position_ids)
    """
    tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    """

    for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
        cur_len = cur_new_embed.shape[0]
        print(f"【COND】 padding_side={getattr(self.config, 'tokenizer_padding_side', 'right')} cur_len={cur_len} max_len={max_len}")
        if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
            pass
        else:
            print("【ENTER】else (padding_side != 'left'):")
            #【ENTER】
            new_input_embeds_padded.append(torch.cat((
                cur_new_embed,
                torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
            ), dim=0))
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            print("new_input_embeds_padded (so far) shape\n", [x.shape for x in new_input_embeds_padded])  # [torch.Size([599, 2048])]
            print("new_input_embeds_padded (so far)\n", new_input_embeds_padded)
            """
            [tensor([[-1.0910e-03,  1.9302e-03, -1.6632e-03,  ...,  1.9932e-04,
                     -6.5231e-04, -4.9973e-04],
                    [-1.9428e-01,  1.1569e-01, -7.4740e-02,  ...,  2.6653e-03,
                     -1.6907e-01, -3.4387e-01],
                    [ 4.3680e-02,  1.7172e-01, -9.9813e-02,  ...,  9.3004e-02,
                     -1.3859e-01, -7.3106e-02],
                    ...,
                    [ 2.1240e-02, -2.2705e-02, -1.4221e-02,  ..., -2.8229e-03,
                     -8.3618e-03, -9.4604e-03],
                    [ 3.7079e-03, -3.6011e-03,  9.0332e-03,  ..., -1.3672e-02,
                     -2.5177e-03, -8.0566e-03],
                    [-6.3705e-04, -1.0605e-03, -1.1841e-02,  ...,  2.1935e-04,
                     -7.3242e-04,  2.7924e-03]], grad_fn=<CatBackward0>)]
            """

            print("new_labels_padded (so far) shape\n", new_labels_padded.shape)  # torch.Size([1, 599])
            print("new_labels_padded (so far)\n", new_labels_padded)
            """
            tensor([[ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                      -100,  -100,  -100,  -100,  -100,  -100,  -100,   278, 25616, 26624,
                       297,   902, 19430, 11105, 29879, 10508,  1596, 23425,   278,  3700,
                       322,  6567,   310,   263,  6114,   411,  2654, 11315,    13]])
            """

            print("attention_mask (so far) shape\n", attention_mask.shape)  # torch.Size([1, 599])
            print("attention_mask (so far)\n", attention_mask)
            """
            tensor([[True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True, True,
                     True, True, True, True, True, True, True, True, True, True, True]])
            """

            print("position_ids (so far) shape\n", position_ids.shape)  # torch.Size([1, 599])
            print("position_ids (so far)\n", position_ids)
            """
            tensor([[  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
                      14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,
                      28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,  41,
                      42,  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,
                      56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,
                      70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,  83,
                      84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,  97,
                      98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                     112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125,
                     126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139,
                     140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153,
                     154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167,
                     168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181,
                     182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195,
                     196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
                     210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
                     224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237,
                     238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251,
                     252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265,
                     266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279,
                     280, 281, 282, 283, 284, 285, 286, 287, 288, 289, 290, 291, 292, 293,
                     294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307,
                     308, 309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321,
                     322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335,
                     336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349,
                     350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363,
                     364, 365, 366, 367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377,
                     378, 379, 380, 381, 382, 383, 384, 385, 386, 387, 388, 389, 390, 391,
                     392, 393, 394, 395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405,
                     406, 407, 408, 409, 410, 411, 412, 413, 414, 415, 416, 417, 418, 419,
                     420, 421, 422, 423, 424, 425, 426, 427, 428, 429, 430, 431, 432, 433,
                     434, 435, 436, 437, 438, 439, 440, 441, 442, 443, 444, 445, 446, 447,
                     448, 449, 450, 451, 452, 453, 454, 455, 456, 457, 458, 459, 460, 461,
                     462, 463, 464, 465, 466, 467, 468, 469, 470, 471, 472, 473, 474, 475,
                     476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 486, 487, 488, 489,
                     490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503,
                     504, 505, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517,
                     518, 519, 520, 521, 522, 523, 524, 525, 526, 527, 528, 529, 530, 531,
                     532, 533, 534, 535, 536, 537, 538, 539, 540, 541, 542, 543, 544, 545,
                     546, 547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559,
                     560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573,
                     574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587,
                     588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598]])
            """
            print("【EXIT】else (padding_side != 'left'):")

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
    print("new_input_embeds (after) shape\n", new_input_embeds.shape)  # torch.Size([1, 599, 2048])
    print("new_input_embeds (after)\n", new_input_embeds)
    """
    tensor([[[-1.0910e-03,  1.9302e-03, -1.6632e-03,  ...,  1.9932e-04,
              -6.5231e-04, -4.9973e-04],
             [-1.9428e-01,  1.1569e-01, -7.4740e-02,  ...,  2.6653e-03,
              -1.6907e-01, -3.4387e-01],
             [ 4.3680e-02,  1.7172e-01, -9.9813e-02,  ...,  9.3004e-02,
              -1.3859e-01, -7.3106e-02],
             ...,
             [ 2.1240e-02, -2.2705e-02, -1.4221e-02,  ..., -2.8229e-03,
              -8.3618e-03, -9.4604e-03],
             [ 3.7079e-03, -3.6011e-03,  9.0332e-03,  ..., -1.3672e-02,
              -2.5177e-03, -8.0566e-03],
             [-6.3705e-04, -1.0605e-03, -1.1841e-02,  ...,  2.1935e-04,
              -7.3242e-04,  2.7924e-03]]], grad_fn=<StackBackward0>)
    """

    print(f"【COND】 _labels_is_None={_labels is None}") 
    if _labels is None:
        #【SKIP】
        print("【ENTER】if _labels is None:")
        new_labels = None
        print("【EXIT】if _labels is None:")
    else:
        # 【ENTER】
        print("【ENTER】else of if _labels is None:")
        new_labels = new_labels_padded
        print("new_labels (after)\n", new_labels)
        """
        tensor([[ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
                  -100,  -100,  -100,  -100,  -100,  -100,  -100,   278, 25616, 26624,
                   297,   902, 19430, 11105, 29879, 10508,  1596, 23425,   278,  3700,
                   322,  6567,   310,   263,  6114,   411,  2654, 11315,    13]])
        """
        print("【EXIT】else of if _labels is None:")

    print(f"【COND】 _attention_mask_is_None={_attention_mask is None}") 
    if _attention_mask is None:
        # 【SKIP】
        print("【ENTER】if _attention_mask is None:")
        attention_mask = None
        print("【EXIT】if _attention_mask is None:")
    else:
        # 【ENTER】
        print("【ENTER】else of if _attention_mask is None:")
        attention_mask = attention_mask.to(dtype=_attention_mask.dtype)
        print("attention_mask (after)\n", attention_mask)
        """
        tensor([[True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True, True,
                 True, True, True, True, True, True, True, True, True, True, True]])
        """

    print(f"【COND】 _position_ids_is_None={_position_ids is None}")
    if _position_ids is None:
        print("【ENTER】if _position_ids is None:")
        position_ids = None
        print("【EXIT】if _position_ids is None:")

    print("position_ids (return)\n", position_ids)  # None
    print("attention_mask (return)\n", attention_mask)
    """
    tensor([[True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True, True,
             True, True, True, True, True, True, True, True, True, True, True]])
    """
    print("past_key_values (return)\n", past_key_values)  # None
    print("new_input_embeds (return)\n", new_input_embeds)
    """
    tensor([[[-1.0910e-03,  1.9302e-03, -1.6632e-03,  ...,  1.9932e-04,
              -6.5231e-04, -4.9973e-04],
             [-1.9428e-01,  1.1569e-01, -7.4740e-02,  ...,  2.6653e-03,
              -1.6907e-01, -3.4387e-01],
             [ 4.3680e-02,  1.7172e-01, -9.9813e-02,  ...,  9.3004e-02,
              -1.3859e-01, -7.3106e-02],
             ...,
             [ 2.1240e-02, -2.2705e-02, -1.4221e-02,  ..., -2.8229e-03,
              -8.3618e-03, -9.4604e-03],
             [ 3.7079e-03, -3.6011e-03,  9.0332e-03,  ..., -1.3672e-02,
              -2.5177e-03, -8.0566e-03],
             [-6.3705e-04, -1.0605e-03, -1.1841e-02,  ...,  2.1935e-04,
              -7.3242e-04,  2.7924e-03]]], grad_fn=<StackBackward0>)
    """
    print("new_labels (return)\n", new_labels)
    """
    tensor([[ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100,   278, 25616, 26624,
               297,   902, 19430, 11105, 29879, 10508,  1596, 23425,   278,  3700,
               322,  6567,   310,   263,  6114,   411,  2654, 11315,    13]])
    """
    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
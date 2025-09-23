def prepare_inputs_labels_for_multimodal(
    self, input_ids, position_ids, attention_mask, past_key_values, labels,
    images, image_sizes=None
):
    print("current file path", "llava/model/llava_arch.py")
    print("def LlavaMetaForCausalLM(ABC).prepare_inputs_labels_for_multimodal(self, input_ids, position_ids, attention_mask, past_key_values, labels, images, image_sizes=None)")
    print("input_ids\n", input_ids)

    print("position_ids\n", position_ids)
    print("attention_mask\n", attention_mask)

    print("past_key_values\n", past_key_values)
    print("labels\n", labels)

    print("images\n", images)

    print("image_sizes\n", image_sizes)
    vision_tower = self.get_vision_tower()
    print("vision_tower\n", vision_tower)

    print(f"【COND】 vision_tower_is_None={vision_tower is None} images_is_None={images is None} input_ids_shape_1_eq_1={input_ids.shape[1] == 1}")
    if vision_tower is None or images is None or input_ids.shape[1] == 1:
        pass

    print("【COND】type(images)\n", type(images))
    print("【COND】images.ndim\n", images.ndim)
    if type(images) is list or images.ndim == 5:
        pass
    else:
        # 【ENTER】
        print("【ENTER】else of if type(images) is list or images.ndim == 5:")
        image_features = self.encode_images(images)
        print("image_features after encode_images shape \n", image_features.shape)
        print("image_features after encode_images\n", image_features)
        print("【EXIT】else of if type(images) is list or images.ndim == 5:")

    # TODO: image start / end is not implemented here to support pretraining.
    if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
        print("【ENTER】if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):")
        raise NotImplementedError

    # Let's just add dummy tensors if they do not exist,
    # it is a headache to deal with None all the time.
    # But it is not ideal, and if you have a better idea,
    # please open an issue / submit a PR, thanks.

    print("labels before\n", labels)

    print("position_ids before\n", position_ids)

    print("attention_mask before\n", attention_mask)

    _labels = labels
    _position_ids = position_ids
    _attention_mask = attention_mask
    if attention_mask is None:
        pass
    else:
        # 【ENTER】
        print("【ENTER】else of if attention_mask is None:")
        attention_mask = attention_mask.bool()
 
        print("attention_mask（after）shape \n", attention_mask.shape)
        print("attention_mask (after)\n", attention_mask)
        print("【EXIT】else of if attention_mask is None:")
    if position_ids is None:
        print("【ENTER】if position_ids is None:")
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        print("position_ids (after) shape \n", position_ids.shape)
        print("position_ids (after)\n", position_ids)
        print("【EXIT】if position_ids is None:")
    print(f"【COND】 labels_is_None={labels is None}")
    if labels is None:
        pass

    # remove the padding using attention_mask -- FIXME
    _input_ids = input_ids
    input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
    labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
    print("input_ids after removing padding\n", input_ids)

    print("labels after removing padding\n", labels)


    new_input_embeds = []
    new_labels = []
    cur_image_idx = 0
    for batch_idx, cur_input_ids in enumerate(input_ids):
        print("cur_input_ids shape\n", cur_input_ids.shape) 
        print("cur_input_ids\n", cur_input_ids)
        num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
        print("【COND】num_images:", num_images)
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
        print("image_token_indices\n", image_token_indices)
        print("len image_token_indices", len(image_token_indices)) 
        cur_input_ids_noim = []
        cur_labels = labels[batch_idx]
        print("cur_labels\n", cur_labels)
        cur_labels_noim = []
        for i in range(len(image_token_indices) - 1):
            cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
        print("cur_input_ids_noim (after)\n", cur_input_ids_noim)
        print("cur_labels_noim (after) \n", cur_labels_noim)
        split_sizes = [x.shape[0] for x in cur_labels_noim]
        print("split_sizes\n", split_sizes)
        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
        print("cur_input_embeds shape\n", cur_input_embeds.shape)
        print("cur_input_embeds\n", cur_input_embeds)
        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
        print("cur_input_embeds_no_im\n", cur_input_embeds_no_im)
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
        print("cur_new_input_embeds (before cat)\n", cur_new_input_embeds)

        print("cur_new_labels (before cat) shape\n", [x.shape for x in cur_new_labels])
        print("cur_new_labels (before cat)\n", cur_new_labels)
        cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

        cur_new_input_embeds = torch.cat(cur_new_input_embeds)
        cur_new_labels = torch.cat(cur_new_labels)

        print("cur_new_input_embeds (after cat) shape\n", cur_new_input_embeds.shape)
        print("cur_new_input_embeds (after cat)\n", cur_new_input_embeds)

        print("cur_new_labels (after cat) shape\n", cur_new_labels.shape)
        print("cur_new_labels (after cat)\n", cur_new_labels)

        new_input_embeds.append(cur_new_input_embeds)
        new_labels.append(cur_new_labels)
        print("new_input_embeds (so far) shape\n", [x.shape for x in new_input_embeds])
        print("new_input_embeds (so far)\n", new_input_embeds)

        print("new_labels (so far) shape\n", [x.shape for x in new_labels])
        print("new_labels (so far)\n", new_labels)

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
    print("max_len\n", max_len)
    batch_size = len(new_input_embeds)
    print("batch_size\n", batch_size)

    new_input_embeds_padded = []
    new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
    print("new_labels_padded (before) shape\n", new_labels_padded.shape)
    print("new_labels_padded (before)\n", new_labels_padded)
    attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
    print("attention_mask (before) shape\n", attention_mask.shape)
    print("attention_mask (before)\n", attention_mask)
    position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)
    print("position_ids (before) shape\n", position_ids.shape)
    print("position_ids (before)\n", position_ids)

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
            print("new_input_embeds_padded (so far) shape\n", [x.shape for x in new_input_embeds_padded])
            print("new_input_embeds_padded (so far)\n", new_input_embeds_padded)

            print("new_labels_padded (so far) shape\n", new_labels_padded.shape)
            print("new_labels_padded (so far)\n", new_labels_padded)

            print("attention_mask (so far) shape\n", attention_mask.shape)
            print("attention_mask (so far)\n", attention_mask)

            print("position_ids (so far) shape\n", position_ids.shape)
            print("position_ids (so far)\n", position_ids)
            print("【EXIT】else (padding_side != 'left'):")

    new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
    print("new_input_embeds (after) shape\n", new_input_embeds.shape)
    print("new_input_embeds (after)\n", new_input_embeds)

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

    print(f"【COND】 _position_ids_is_None={_position_ids is None}")
    if _position_ids is None:
        print("【ENTER】if _position_ids is None:")
        position_ids = None
        print("【EXIT】if _position_ids is None:")

    print("position_ids (return)\n", position_ids)
    print("attention_mask (return)\n", attention_mask)
    print("past_key_values (return)\n", past_key_values)
    print("new_input_embeds (return)\n", new_input_embeds)
    print("new_labels (return)\n", new_labels)
    return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
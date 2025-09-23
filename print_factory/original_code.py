def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    images: Optional[torch.FloatTensor] = None,
    image_sizes: Optional[List[List[int]]] = None,
    return_dict: Optional[bool] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:

    print("current file path", "llava/llava/model/language_model/llava_llama.py")
    print("def LlavaLlamaForCausalLM.forward(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, images, image_sizes, return_dict)")
    print("input_ids\n", input_ids)
    if hasattr(input_ids, 'shape'):
        print("input_ids.shape\n", input_ids.shape) 
    print("attention_mask\n", attention_mask)

    print("position_ids\n", position_ids) 
    print("past_key_values\n", past_key_values) 
    print("inputs_embeds\n", inputs_embeds) 
    if hasattr(inputs_embeds, 'shape'):
        print("inputs_embeds.shape\n", inputs_embeds.shape)
    print("labels\n", labels)

    print("use_cache\n", use_cache)
    print("output_attentions\n", output_attentions)
    print("output_hidden_states\n", output_hidden_states)
    print("images\n", images)

    if hasattr(images, 'shape'):
        print("images.shape\n", images.shape)
    print("image_sizes\n", image_sizes)
    print("return_dict\n", return_dict)

    print(f"【COND】 inputs_embeds_is_None={inputs_embeds is None}")
    if inputs_embeds is None:
        # 【ENTER】
        print("【ENTER】if inputs_embeds is None:")
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
            image_sizes
        )
        print("【EXIT】if inputs_embeds is None:")

    print("input_ids (after prepare_inputs_labels_for_multimodal)\n", input_ids)

    print("position_ids shape (after prepare_inputs_labels_for_multimodal) \n", position_ids.shape)
    print("position_ids (after prepare_inputs_labels_for_multimodal)\n", position_ids)

    print("attention_mask shape (after prepare_inputs_labels_for_multimodal)\n", attention_mask.shape)
    print("attention_mask (after prepare_inputs_labels_for_multimodal)\n", attention_mask)

    print("past_key_values shape (after prepare_inputs_labels_for_multimodal)\n", None if past_key_values is None else len(past_key_values))
    print("past_key_values (after prepare_inputs_labels_for_multimodal)\n", past_key_values)

    print("inputs_embeds shape (after prepare_inputs_labels_for_multimodal)\n", None if inputs_embeds is None else inputs_embeds.shape)
    print("inputs_embeds (after prepare_inputs_labels_for_multimodal)\n", inputs_embeds)

    print("labels shape (after prepare_inputs_labels_for_multimodal)\n", labels.shape)
    print("labels (after prepare_inputs_labels_for_multimodal)\n", labels)

    #  LlamaForCausalLM.forward
    # Trainer > def train > def inner_training_loop > def training_step > model(**inputs) > model.forward
    result = super().forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        labels=labels,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict
    )
    print("Return of def LlavaLlamaForCausalLM.forward(self, input_ids, attention_mask, position_ids, past_key_values, inputs_embeds, labels, use_cache, output_attentions, output_hidden_states, images, image_sizes, return_dict)")
    print("result of LlavaLlamaForCausalLM.forward (return)\n", result)
    print("logits tensor shape  LlavaLlamaForCausalLM.forward\n", result.logits.shape)
    print("logits tensor (first 2 tokens)  LlavaLlamaForCausalLM.forward\n", result.logits[0, :10, :])
    print("loss (return)  LlavaLlamaForCausalLM.forward \n", result.loss)

    return result
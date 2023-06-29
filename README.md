# LongChat
The official repository for LongChat and LongEval, which supports training and evaluating long-context LLM based chatbots. 

## Environment setup
```bash
conda create -n longeval python=3.10
conda activate longeval
git clone https://github.com/DachengLi1/LongChat/
cd LongChat/
pip install -e .
```
For users who want to test very long sequence length, please also install [FlashAttention](https://github.com/HazyResearch/flash-attention).

## longchat
To train a LongChat model yourself, replace <path-to-llama> to the llama checkpoint director, and run:
```bash
python -m torch.distributed.run --nproc_per_node=8 \
         longchat/train/fine_tune/train_condense_16K.py \
        --model_name_or_path <path-to-llama> \
        --data_path data/dummy_conversation.json  \
        --bf16 \
        --output_dir outputs \
        --num_train_epochs 3    \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 4  \
        --gradient_accumulation_steps 1 \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 1000  \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0.  \
        --warmup_ratio 0.03  \
        --lr_scheduler_type "cosine" \
        --logging_steps 1  \
        --fsdp "full_shard auto_wrap" \
        --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
        --tf32 True  \
        --model_max_length 16384  \
        --gradient_checkpointing True  \
        --lazy_preprocess True
```
This script assumes 8xA100 GPUs and use the dummy data in the repository for example usage only. Please adapt to your use case. 
We provided models trained on conversation data in HuggingFace Hub: [LongChat-13b-16k](https://huggingface.co/lmsys/longchat-13b-16k) and [LongChat-7b-16k](https://huggingface.co/lmsys/longchat-7b-16k).

## longeval
We provided a simple [notebook](https://github.com/DachengLi1/LongChat/blob/longeval/longeval/topics_lines_demo.ipynb) to demonstrate following steps.

To evaluate the LongChat model on the coarsed-grained topics benchmark:
```bash
cd longeval
python3 eval.py --model-name-or-path  lmsys/longchat-13b-16k --task topics --longchat_flash_attn
```

More generally, to evaluate new models, simply choose a ```<task>``` from ["topics", "lines"], replace ```<your-model>``` with your model path:
```bash
python3 eval.py --model-name-or-path <your-model> --task <task>
```
Some models require memory efficient flash attention to evaluate super long test. We include the commands we used in the release [blog](https://lmsys.org/blog/2023-06-29-longchat):
```bash
python3 eval.py --model-name-or-path  lmsys/longchat-7b-16k --task <task> --longchat_flash_attn
python3 eval.py --model-name-or-path  lmsys/longchat-13b-16k --task <task> --longchat_flash_attn
python3 eval.py --model-name-or-path  mosaicml/mpt-7b-storywriter --task <task>
python3 eval.py --model-name-or-path  mosaicml/mpt-30b-chat --task <task> --num_gpus 8 --max_gpu_memory 10
python3 eval.py --model-name-or-path  THUDM/chatglm2-6b --task <task>
```
The output will be stored under evaluation/task/predictions/your-model. The line recall experiment directly outputs an accuracy. The topics recall experiments outputs natural languages that are hard to parsed. you can manually inspect the model output and calculate an accuracy or use chatgpt-3.5-turbo to automatically calculate it. In the latter case, [set](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety) OPENAI_API_KEY and run:
```bash
python auto_topic_eval.py  --test_file <generated_output>
```
Replace <generated_output> with the generated topic prediction, e.g. evaluation/topics/predictions/longchat_13b_16k/5_response.txt.

### Citation
If you find this repo to be useful, plese cite:
```
@misc{longchat2023,
    title = {LongChat: How Long Can Context Length of Open-Source LLMs truly Promise?},
    url = {https://lmsys.org/blog/2023-06-29-longchat},
    author = {Dacheng Li*, Rulin Shao*, Anze Xie, Ying Sheng, Lianmin Zheng, Ion Stoica, Xuezhe Ma, and Hao Zhang},
    month = {June},
    year = {2023}
}
```





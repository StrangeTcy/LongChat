# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from longchat.train.monkey_patch.llama_condense_monkey_patch import replace_llama_with_condense

replace_llama_with_condense(ratio=8)

from longchat.train.monkey_patch.llama_xformer_monkey_patch import replace_llama_attn_with_xformer

replace_llama_attn_with_xformer()

from longchat.train.fine_tune.train import train

if __name__ == "__main__":
    train()

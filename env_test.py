from transformers import CLIPTokenizer, CLIPTextModel

local_dir = "/home/tobiasvavroch/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/32bd64288804d66eefd0ccbe215aa642df71cc41"

tokenizer = CLIPTokenizer.from_pretrained(local_dir, local_files_only=True, use_fast=False)
transformer = CLIPTextModel.from_pretrained(local_dir, local_files_only=True)

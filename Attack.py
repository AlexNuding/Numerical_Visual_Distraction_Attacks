import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import torch
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset
from PIL import Image
from transformers import (
	LlavaNextProcessor,
	LlavaNextForConditionalGeneration,
	BitsAndBytesConfig,
	AutoProcessor
)
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import gc
import re

def generate(processor, model, query, image, max_new_tokens=200):
	# Wrap query + image into chat format
	conversation = [
		{"role": "user", "content": [
			{"type": "text", "text": query},
			{"type": "image"}  # Image is passed separately
		]}
	]
	prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

	# Process inputs
	inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

	with torch.inference_mode():
		out = model.generate(
			**inputs,
			max_new_tokens=max_new_tokens,
			do_sample=True,
			temperature=0.2,
			eos_token_id=processor.tokenizer.eos_token_id,
		)

	response = processor.decode(out[0], skip_special_tokens=True).strip()
	
	# get only output of model
	output = re.sub(r"\[INST\].*?\[/INST\]", "", response, flags=re.DOTALL)

	# cleanup GPU memory
	del inputs, out
	torch.cuda.empty_cache()
	gc.collect()

	return output

def generate_qwen(processor, model, query, image, max_new_tokens=200):
	params = SamplingParams(
        temperature=0.2,
        max_tokens=max_new_tokens,
        top_k=-1,
        stop_token_ids=[],
    )

	messages = [
			{
				"role": "user",
				"content": [
					{"type": "image", "image": image},
					{"type": "text", "text": query},
				],
			}
		]
	prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

	image_inputs, _, mm_kwargs = process_vision_info(
			messages,
			image_patch_size=processor.image_processor.patch_size,
			return_video_kwargs=True,
			return_video_metadata=True,
		)
	
	mm_data = {}
	if image_inputs is not None:
		mm_data['image'] = image_inputs

	inputs = {
        'prompt': prompt,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': mm_kwargs
    }
	outputs = model.generate(inputs, sampling_params=params)

	return outputs[0].outputs[0].text.strip()


def evaluate(args):
	model_id = args.model_path

	if model_id == "llava-hf/llava-v1.6-mistral-7b-hf":
		processor = LlavaNextProcessor.from_pretrained(model_id, use_fast = True)

		model = LlavaNextForConditionalGeneration.from_pretrained(
			model_id,
			force_download=True,
			device_map="auto",
			low_cpu_mem_usage=True,
			torch_dtype=torch.bfloat16,		
		).eval()
	elif model_id == "Qwen/Qwen3-VL-8B-Instruct-FP8":
		processor = AutoProcessor.from_pretrained(model_id)

		model = LLM(
			model=model_id,
			trust_remote_code=True,
			gpu_memory_utilization=0.70,
			enforce_eager=False,
			tensor_parallel_size=torch.cuda.device_count(),
			seed=0,
			max_model_len=32768,
		)

	if args.dataset_type == "HADES":
		if args.local:
			hades = load_from_disk(args.dataset_path)["test"]
		else:
			hades = load_dataset(args.dataset_path)["test"]

		if args.step == 'last':
			test_split = [item for item in hades if item['step'] == 5]
		else:
			test_split = hades

	elif args.dataset_type == "SAFETYBENCH":
		if args.local:
			test_split = load_from_disk(args.dataset_path)
		else:
			test_split = load_dataset(args.dataset_path)
	

	if args.partial > 1:
		if args.part < 1 or args.part > args.partial:
			raise ValueError(f"args.part must be between 1 and {args.partial}, got {args.part}")
		
		total_len = len(test_split)
		part_size = total_len // args.partial
		start_idx = (args.part - 1) * part_size
		end_idx = total_len if args.part == args.partial else args.part * part_size

		test_split = test_split[start_idx:end_idx]

		print(f"Selected {len(test_split)} items (part {args.part}/{args.partial})")


	print(f"started run with {len(test_split)} items")
	for item in tqdm(test_split):
		image = item['image'].convert('RGB')

		if args.dataset_type == "HADES":	
			inst = item['instruction']

			# rewrite instruction per hades
			if item["category"] == "behavior":
				mask = "conduct this behavior on"
			else:
				mask = f"this {item['category']}"
			new_inst = f"The bottom of the image shows a {item['category']}. " + inst.replace(item["keywords"], mask)

		elif args.dataset_type == "SAFETYBENCH":
			new_inst = item['question']

		if args.basic: #only for hades
			if model_id == "llava-hf/llava-v1.6-mistral-7b-hf":
				response = generate(processor, model, new_inst, image, max_new_tokens=args.max_new_tokens)
			elif model_id == "Qwen/Qwen3-VL-8B-Instruct-FP8":
				response = generate_qwen(processor, model, new_inst, image, max_new_tokens=args.max_new_tokens)

			if args.ensure_accurate_OCR and item['keywords'] not in response:
				for _ in range(args.max_attempts):
					if model_id == "llava-hf/llava-v1.6-mistral-7b-hf":
						response = generate(processor, model, new_inst, image, max_new_tokens=args.max_new_tokens)
					elif model_id == "Qwen/Qwen3-VL-8B-Instruct-FP8":
						response = generate_qwen(processor, model, new_inst, image, max_new_tokens=args.max_new_tokens)

					if item['keywords'] in response:
						break

			result = {
				'id': item['id'],
				"scenario": item['scenario'],
				"keywords": item['keywords'],
				"step": item['step'],
				"category": item['category'],
				"instruction": item['instruction'],
				"response": [response]
			}	
		else:
			responses = []
			for _ in range(args.max_attempts):
				if model_id == "llava-hf/llava-v1.6-mistral-7b-hf":
					response = generate(processor, model, new_inst, image, max_new_tokens=args.max_new_tokens)
				elif model_id == "Qwen/Qwen3-VL-8B-Instruct-FP8":
					response = generate_qwen(processor, model, new_inst, image, max_new_tokens=args.max_new_tokens)

				responses.append(response)
			
			if args.dataset_type == "HADES":
				result = {
					'id': item['id'],
					"scenario": item['scenario'],
					"keywords": item['keywords'],
					"step": item['step'],
					"category": item['category'],
					"instruction": item['instruction'],
					"response": responses
				}
			elif args.dataset_type == "SAFETYBENCH":
				result = {
					'id': item['id'],
					"instruction": item['question'],
					"response": responses
				}

		with open(args.output_path, "a", encoding="utf-8") as f:
			f.write(json.dumps(result, ensure_ascii=False) + "\n")

		# free data
		del image
		gc.collect()


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_path", type=str, default="Monosail/HADES")
	parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct-FP8")
	parser.add_argument("--output_path", type=str, default="results/test.jsonl")
	parser.add_argument("--ensure_accurate_OCR", type=bool, default=True)
	parser.add_argument("--max_attempts", type=int, default=2)
	parser.add_argument("--step", type=str, default="last")  # "last" or "all"
	parser.add_argument("--max_new_tokens", type=int, default=500)
	parser.add_argument("--local", action="store_true", default=False, help="Use local dataset instead of HuggingFace")
	parser.add_argument("--partial", type=int, default=150)
	parser.add_argument("--part", type=int, default=1)
	parser.add_argument("--basic", action="store_true", default=False)
	parser.add_argument("--dataset_type", type=str, default="HADES")
	args = parser.parse_args()

	evaluate(args)

import os
import json
from openai import OpenAI
from tqdm.auto import tqdm
import json
import argparse
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
import gc
import base64
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

def img_convert(image):
	buffer = BytesIO()
	image.save(buffer, format="JPEG")
	b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
	mime = "image/jpeg"
	return f"data:{mime};base64,{b64}"

def create_split(data, splits):
	total_num = len(data)

	size = total_num // splits
	split_list = [data[i * size:(i + 1) * size] for i in range(splits - 1)]
	split_list.append(data[(splits - 1) * size:])  
	return split_list

def load_data(file):
	data = []
	with open(file, "r", encoding="utf-8") as f:
		for line in f:
			if not line.strip():
				continue

			item_data = json.loads(line)
			data.append(item_data)

	return data


def attack(tmp_save_file, data_list, client, thread_nr, dataset_type):
	error = 0
	total_calls = len(data_list)
	description = f"Performing atttacks, Thread {thread_nr}"
	
	with tqdm(total=total_calls, desc=description, unit="resp") as pbar:
		for item in data_list:
			image = img_convert(item['image'].convert('RGB'))

			if dataset_type == "HADES":	
				inst = item['instruction']

				# Rewrite instruction slightly
				if item["category"] == "behavior":
					mask = "conduct this behavior on"
				else:
					mask = f"this {item['category']}"
				new_inst = f"The bottom of the image shows a {item['category']}. " + inst.replace(item["keywords"], mask)

			elif dataset_type == "SAFETYBENCH":
				new_inst = item['question']

			responses = []

			for i in range(1):
				try:
					response = client.chat.completions.create(
						model="gpt-5",
						messages=[
							{
								"role": "user", 
								"content": [
									{"type": "text", "text": new_inst},
									{"type": "image_url", "image_url": {"url": image}}					
								]
							}
						],
						n=5
					)

					for choice in response.choices:
						responses.append(choice.message.content)

				# if error occours, mark as harmfull to be safe
				except Exception as e:
					print(f"error with output {item["id"]}")
					print (e)
					responses.append("Refused to answer completely.")
					error += 1

				finally:
					pbar.update(1)
					pbar.set_postfix(errors=error)


			if dataset_type == "HADES":
				result = {
					'id': item['id'],
					"scenario": item['scenario'],
					"keywords": item['keywords'],
					"step": item['step'],
					"category": item['category'],
					"instruction": item['instruction'],
					"response": responses
				}
			elif dataset_type == "SAFETYBENCH":
				result = {
					'id': item['id'],
					"instruction": item['question'],
					"response": responses
				}

			with open(tmp_save_file, "a", encoding="utf-8") as f:
				f.write(json.dumps(result, ensure_ascii=False) + "\n")

			# free image reference too
			del image
			gc.collect()

def main(args):
	if args.dataset_type == "HADES":
		if args.local:
			hades = load_from_disk(args.dataset_path)["test"]
		else:
			hades = load_dataset(args.dataset_path)["test"]

		data_list = [item for item in hades if item['step'] == 5]

	elif args.dataset_type == "SAFETYBENCH":
		if args.local:
			safety = load_from_disk(args.dataset_path)
		else:
			safety = load_dataset(args.dataset_path)

		data_list = [item for item in safety]

	client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

	datalists = create_split(data_list, args.workers)
	
	tmpfiles = []
	for i in range(args.workers):
		name = f"tmp_{i}.jsonl"
		tmpfiles.append(args.tmpfile_location+name)

	futures = []
	with ThreadPoolExecutor(max_workers=args.workers) as executor:
		for i in range(args.workers):
			futures.append(executor.submit(attack, tmpfiles[i], datalists[i], client, i, args.dataset_type))

		for f in futures:
			f.result()

	combined = []
	for file in tmpfiles:
		data = load_data(file)
		combined += data

	for line in combined:
		with open(args.output_path, 'a') as f:
			json.dump(line, f)
			f.write('\n')

	for file in tmpfiles:
		os.remove(file)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_path", type=str, default="./Data/SafetyBench_Main_V2/")
	parser.add_argument("--output_path", type=str, default="results/Model_gpt-5/safetybench/main_v2.jsonl")
	parser.add_argument("--tmpfile_location", type=str, default="results/Model_gpt-5/safetybench/")
	parser.add_argument("--workers", type=int, default=6)
	parser.add_argument("--local", action="store_true", default=False, help="Use local dataset instead of HuggingFace")
	parser.add_argument("--dataset_type", type=str, default="SAFETYBENCH")
	args = parser.parse_args()

	main(args)
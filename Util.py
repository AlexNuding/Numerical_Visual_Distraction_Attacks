import json
import json
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from datasets import load_dataset, DatasetDict, Dataset, load_from_disk
from difflib import SequenceMatcher
from itertools import combinations

def load_data(file = "./results/Model_gpt-5/hades/base.jsonl", number = -1):
	data = []
	with open(file, "r", encoding="utf-8") as f:
		current_line = 0
		for line in f:
			if not line.strip():
				continue
			if current_line >= number and number > 0:
				break

			item_data = json.loads(line)
			data.append(item_data)
			current_line += 1

	return data

def display_result(file_path: str, dataset_path: str, result_nr: int = 1, ver = 1):
	hades = load_from_disk(dataset_path)["test"]
	test_split = [item for item in hades if item['step'] == 5]

	if result_nr > len(test_split):
		print(f"Result {result_nr} not found in dataset split.")
		return
	
	image_item = test_split[result_nr - 1]
	image = image_item['image'].convert('RGB')
	id = image_item['id']
	result_item = None


	with open(file_path, "r", encoding="utf-8") as file:
		item_data = None

		for line in file:
			if not line.strip():
				continue

			item_data = json.loads(line)
			if item_data.get('id') == id:
				result_item = item_data
				break

	if not result_item:
		print(f"Result {result_nr} not found in {file_path}.")
		return
	else:
		instr = result_item.get('instruction', '')
		resp = result_item.get('response', '')[ver -1]
		judge = result_item.get('judgment', '')[ver -1]
		tox = result_item.get('toxicity', '')[ver -1]

	fig, ax_img = plt.subplots(figsize=(14, 7))
	ax_img.imshow(image)
	ax_img.axis("off")
	ax_img.set_aspect("equal", adjustable="box")

	divider = make_axes_locatable(ax_img)
	txt_width_in = 4.5
	ax_txt = divider.append_axes(
		"right",
		size=axes_size.Fixed(txt_width_in),
		pad=0.4
	)

	info_text = (
		f"Instruction:\n{instr}\n\n"
		f"Response:\n{resp}\n\n"
		f"Judgment: {judge}\n\n"
		f"Toxicity: {tox}"
	)
	ax_txt.axis("off")
	ax_txt.text(
		0.0, 1.0, info_text,
		transform=ax_txt.transAxes,
		va="top", ha="left",
		wrap=True, fontsize=10
	)
	plt.show()

def compare_automatic_manual(automatic_file: str, manual_file: str, nr_results: int = 250):
	manual_data = {}
	automatic_data = {}
	manual_Harmful = 0
	automatic_Harmful = 0

	#load manual data
	with open(manual_file, "r", encoding="utf-8") as file:
		current_line = 0
		for line in file:
			if not line.strip():
				continue
			if current_line >= nr_results:
				break

			item_data = json.loads(line)
			manual_data[item_data['id']] = item_data
			if item_data['judgment'] == 'HARMFUL':
				manual_Harmful += 1
			current_line += 1

	print(f"Manual HARMFUL: {manual_Harmful} out of {nr_results} ({(manual_Harmful/nr_results)*100:.2f}%)")

	# load automatic data
	with open(automatic_file, "r", encoding="utf-8") as file:
		current_line = 0
		for line in file:
			if not line.strip():
				continue
			if current_line >= nr_results:
				break

			item_data = json.loads(line)
			automatic_data[item_data['id']] = item_data
			if item_data['judgment'] == 'HARMFUL':
				automatic_Harmful += 1
			current_line += 1

	print(f"Automatic HARMFUL: {automatic_Harmful} out of {nr_results} ({(automatic_Harmful/nr_results)*100:.2f}%)")

	# compare
	same_judgements = 0
	for id, manual_item in manual_data.items():
		automatic_item = automatic_data.get(id)
		if not automatic_item:
			print(f"ID {id} not found in automatic data.")
			continue

		if manual_item['judgment'] == automatic_item['judgment']:
			same_judgements += 1

	print(f"Same judgements: {same_judgements} out of {nr_results} ({(same_judgements/nr_results)*100:.2f}%)")

def view_modified_image(dataset_path: str, result_nr: int =1, original = False, safetybench = False, Save = False):
	if original:
		hades = load_dataset("Monosail/HADES")["test"]
		test_split = [item for item in hades if item['step'] == 5]
	elif safetybench:
		test_split = load_from_disk(dataset_path)
	else:
		hades = load_from_disk(dataset_path)["test"]
		test_split = [item for item in hades if item['step'] == 5]

	if result_nr > len(test_split):
		print(f"Result {result_nr} not found in dataset split.")
		return
	
	image_item = test_split[result_nr - 1]
	image = image_item['image'].convert('RGB')
	id = image_item['id']

	if Save:
		location = "./"
		if safetybench:
			location += "SafetyBench_"
		else:
			location += "HADES_"

		location += id
		location += ".png"
		image.save(location)


	fig, ax_img = plt.subplots(figsize=(14, 7))
	ax_img.imshow(image)
	ax_img.axis("off")
	plt.show()

def find_change(base: str, modified: str, nr_results: int = 250):
	base_harmless = []
	modified_harmful = []

	#load base data
	with open(base, "r", encoding="utf-8") as file:
		current_line = 0
		for line in file:
			if not line.strip():
				continue
			if current_line >= nr_results:
				break

			item_data = json.loads(line)
			id = item_data['id']
			if item_data['judgment'] == 'HARMLESS':
				base_harmless.append(id)
			current_line += 1

	# load modified data
	with open(modified, "r", encoding="utf-8") as file:
		current_line = 0
		for line in file:
			if not line.strip():
				continue
			if current_line >= nr_results:
				break

			item_data = json.loads(line)
			id = item_data['id']
			if item_data['judgment'] == 'HARMFUL':
				modified_harmful.append(id)
			current_line += 1

	changed_elements = set(base_harmless) & set(modified_harmful)
	changed_list = list(changed_elements)

	print(f"Elementrs with Change: {changed_list}")

def compare_reruns(file: str):
	data = load_data(file, 750)

	dup_items_sum = 0
	any_dup = 0
	sim_sum = 0.0
	n = 0

	for dp in data:
		s = dp["response"]  # list of 5 strings
		n += 1

		dup_items = len(s) - len(set(s))
		dup_items_sum += dup_items
		if dup_items:
			any_dup += 1

		sim_sum += sum(SequenceMatcher(None, a, b).ratio() for a, b in combinations(s, 2)) / 10

	avg_dup = dup_items_sum / n if n else 0.0
	pct_any = (any_dup / n * 100) if n else 0.0
	avg_sim = sim_sum / n if n else 0.0

	print(f"Datapoints: {n}")
	print(f"Avg duplicate items (out of 5): {avg_dup:.3f}")
	print(f"Datapoints w/ any duplicate: {pct_any:.1f}%")
	print(f"Avg pairwise similarity (0..1): {avg_sim:.4f}")

	return avg_dup, pct_any, avg_sim

def find_best_showcase_results(base_file = "", image_file = ""):
	base_data = load_data(base_file)
	image_data = load_data(image_file)

	base_all_harmless = []
	base_4_harmless = []

	for item in base_data:
		harmfull = 0

		for rating in item["judgment"]:
			if rating == "HARMFUL":
				harmfull += 1

		if harmfull == 0:
			base_all_harmless.append(item["id"])
		elif harmfull == 1:
			base_4_harmless.append(item["id"])


	perfect_result = []
	secondary_result = []

	for item in image_data:
		harmfull = 0

		for rating in item["judgment"]:
			if rating == "HARMFUL":
				harmfull += 1

		if harmfull == 5:
			if item["id"] in base_all_harmless:
				perfect_result.append(item)
			elif item["id"] in base_4_harmless:
				secondary_result.append(item)

	if len(perfect_result) > 0:
		print("found Perfect matches:")
		for item in perfect_result:
			print(f"Result ID {item["id"]}")

	print("\n\n")
	if len(secondary_result) > 0:
		print("found secondary matches:")
		for item in secondary_result:
			print(f"Result ID {item["id"]}")

	if len(perfect_result) == 0 and len(secondary_result) == 0:
		print("found no good matches")

def create_combined_safetybench_dataset():
	combined_dataset = []
	categories = ["EconomicHarm", "Fraud", "HateSpeech", "Illegal_Activitiy", "Malware_Generation", "Physical_Harm", "Political_Lobbying", "Privacy_Violence", "Financial_Advice", "Gov_Decision", "Health_Consultation", "Legal_Opinion", "Sex"]

	for cat in categories:
		data = load_dataset("PKU-Alignment/MM-SafetyBench", cat)["SD"]

		for item in data:
			new_id = cat + "_" + item["id"]
			new_sample = dict(item)
			new_sample["id"] = new_id

			combined_dataset.append(new_sample)

	print(len(combined_dataset))
	new_dataset = Dataset.from_list(combined_dataset)
	new_dataset.save_to_disk("./Data/SafetyBench")

def remove_math_instruction(base_dataset, new_dataset):
	dataset = load_from_disk(base_dataset)["test"]
	split = [item for item in dataset if item['step'] == 5]

	new_splits = {}
	modified_samples = []

	for sample in split:
		instr = sample["instruction"]
		new_sample = dict(sample)

		new_instruction = ""

		new_sample["instruction"] = new_instruction
		modified_samples.append(new_sample)

	new_splits["test"] = Dataset.from_list(modified_samples)
	DatasetDict(new_splits).save_to_disk(new_dataset)


if __name__ == "__main__":
	#display_result("./results/Model_gpt-5/hades/judgements_base_toxicity.jsonl", "./Data/Numbers_in_Image_Main_V2M",  30, 3)
	view_modified_image("./Data/Numbers_in_Image_Main_V2M", 357, Save=True)

# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
import sys
import time
import json
import random
import argparse

import torch
import numpy as np
from PIL import Image
from tabulate import tabulate
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

from eval_following import call_azure_gpt4v
from eval_keep_detail import call_azure_gpt4v as call_azure_gpt4v2
from eval_quality import call_azure_gpt4v as call_azure_gpt4v3

from accelerate import PartialState



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)

def extract_score(string):
    # Use regular expression to match the floating number after 'score':
    match = re.search(r"'score': (\d+\.\d+)", string)
    return float(match.group(1))

def save_imgs(img, edited_img, high_res_edited_img, prompt, save_dir, name_str='', scores=''):
    high_res_edited_img.save(save_dir + "/{}_{}_result.jpg".format(prompt.replace(" ", "_"), name_str + '_' + scores))
    combined = Image.fromarray(np.hstack((np.array(img), np.array(edited_img))))
    combined.save(save_dir + "/{}_{}.jpg".format(prompt.replace(" ", "_"), name_str + '_' + scores))
    print(save_dir + "/{}_{}.jpg".format(prompt.replace(" ", "_"), name_str + '_' + scores))

def read_eval_imgs_prompt():
    res = []
    data = load_json("eval/eval_instruct.json")
    for k,v in data.items():
        img_path = "eval/imgs/" + k
        for p in v:
            res.append((img_path, p.split(": ")[-1], p.split(": ")[0].split('. ')[-1]))
    return res


def get_argparse():
    parser = argparse.ArgumentParser(description="eval on the evaluate dataset")

    parser.add_argument("--model_path", type=str, default="timbrooks/instruct-pix2pix")
    parser.add_argument("--work_dir", type=str, default="work_dirs/results/sd15")
    parser.add_argument("--ckpt", type=str, default="last")
    parser.add_argument("--azure_endpoint", type=str, default="your_azure_endpoint")
    parser.add_argument("--api_key", type=str, default="your_api_key")
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--text_guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    set_seed(42)
    args = get_argparse()
    model_path = args.model_path

    # data = read_eval_imgs_prompt()[:140][::2]  # for quick validation
    data = read_eval_imgs_prompt()  # for full evaluation
    resolution = 512

    model_id = model_path
    distributed_state = PartialState()

    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(distributed_state.device)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    generator = torch.Generator(device="cuda").manual_seed(42)

    class_type_results = {}
    overall_results = {
        'fyn_list': [],
        'fscore_list': [],
        'kyn_list': [],
        'kscore_list': [],
        'qyn_list': [],
        'qscore_list': []
    }

    save_dir = f"{args.work_dir}/{args.model_path.replace('/', '_')}/ckpt_{args.ckpt}"
    save_dir = os.path.join(save_dir, f'text-{args.text_guidance_scale}_image-{args.image_guidance_scale}_step-{args.num_inference_steps}')
    os.makedirs(save_dir, exist_ok=True)

    with distributed_state.split_between_processes(list(range(len(data)))) as idxs:
        for idx in idxs:
            img_path, prompt, class_type = data[idx]
            stime = time.time()

            img_idx = img_path.split('/')[-1].split('.')[0]

            img = Image.open(img_path).convert('RGB')
            with torch.no_grad():
                edited_img = pipe(
                    prompt,
                    image=img,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.text_guidance_scale,
                    image_guidance_scale=args.image_guidance_scale,
                    generator=generator,
                ).images[0]

            high_res_edited_img = edited_img
            size = Image.open(img_path).convert('RGB').size
            img = img.resize((int(resolution * size[0] / size[1]), resolution))
            edited_img = edited_img.resize((int(resolution * size[0] / size[1]), resolution))
            print("Generate Time: {}".format(time.time() - stime))

            # Retry mechanism
            max_retries = 100
            retry_delay = 3
            for attempt in range(max_retries):
                try:
                    stime = time.time()
                    follow_score_str = call_azure_gpt4v([img, edited_img], prompt, args.azure_endpoint, args.api_key)
                    print("Calculate Following Time: {}".format(time.time() - stime))

                    stime = time.time()
                    keep_score_str = call_azure_gpt4v2([img, edited_img], prompt, args.azure_endpoint, args.api_key)
                    print("Calculate Keeping Time: {}".format(time.time() - stime))

                    stime = time.time()
                    quality_score_str = call_azure_gpt4v3(edited_img, args.azure_endpoint, args.api_key)
                    print("Calculate quality Time: {}".format(time.time() - stime))

                    if any('qpm limit' in lst for lst in [follow_score_str, keep_score_str, quality_score_str]):
                        continue
                    else:
                        break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1}/{max_retries} failed with error: {e}. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                    else:
                        print(f"Attempt {attempt + 1}/{max_retries} failed with error: {e}. Moving to the next image.")
                        continue

            if isinstance(follow_score_str, str) and isinstance(keep_score_str, str) and isinstance(quality_score_str, str):
                try:
                    fyn = 1 if "yes" in follow_score_str else 0
                    fscore = extract_score(follow_score_str)
                    kyn = 1 if "yes" in keep_score_str else 0
                    kscore = extract_score(keep_score_str)
                    qyn = 1 if "yes" in quality_score_str else 0
                    qscore = extract_score(quality_score_str)

                    if class_type not in class_type_results:
                        class_type_results[class_type] = {
                            'fyn_list': [],
                            'fscore_list': [],
                            'kyn_list': [],
                            'kscore_list': [],
                            'qyn_list': [],
                            'qscore_list': [],
                        }

                    class_type_results[class_type]['fyn_list'].append(fyn)
                    class_type_results[class_type]['fscore_list'].append(fscore)
                    class_type_results[class_type]['kyn_list'].append(kyn)
                    class_type_results[class_type]['kscore_list'].append(kscore)
                    class_type_results[class_type]['qyn_list'].append(qyn)
                    class_type_results[class_type]['qscore_list'].append(qscore)

                    overall_results['fyn_list'].append(fyn)
                    overall_results['fscore_list'].append(fscore)
                    overall_results['kyn_list'].append(kyn)
                    overall_results['kscore_list'].append(kscore)
                    overall_results['qyn_list'].append(qyn)
                    overall_results['qscore_list'].append(qscore)
                except:
                    print(quality_score_str, qscore)
                    continue

            prompt = img_idx + '_' + prompt
            scores = 'follow-' + str(fscore) + '_keep-' + str(kscore) + '_quality-' + str(qscore)
            save_imgs(img, edited_img, high_res_edited_img, prompt, save_dir, scores=scores)

    # Save the temporary results of each process
    temp_file = f"{save_dir}/temp_results_{distributed_state.process_index}.json"
    with open(temp_file, "w") as f:
        json.dump(class_type_results, f)

    # Wait for all processes to complete
    distributed_state.wait_for_everyone()

    if distributed_state.is_main_process:
        # Summarize and calculate the results of each class_type
        final_class_type_results = {}
        final_overall_results = {
            'fyn_list': [],
            'fscore_list': [],
            'kyn_list': [],
            'kscore_list': [],
            'qyn_list': [],
            'qscore_list': []
        }

        # Read all temporary files and merge data
        for i in range(distributed_state.num_processes):
            temp_file = f"{save_dir}/temp_results_{i}.json"
            with open(temp_file, "r") as f:
                partial_class_type_results = json.load(f)
                for class_type, scores in partial_class_type_results.items():
                    if class_type not in final_class_type_results:
                        final_class_type_results[class_type] = {
                            'fyn_list': [],
                            'fscore_list': [],
                            'kyn_list': [],
                            'kscore_list': [],
                            'qyn_list': [],
                            'qscore_list': []
                        }
                    final_class_type_results[class_type]['fyn_list'].extend(scores['fyn_list'])
                    final_class_type_results[class_type]['fscore_list'].extend(scores['fscore_list'])
                    final_class_type_results[class_type]['kyn_list'].extend(scores['kyn_list'])
                    final_class_type_results[class_type]['kscore_list'].extend(scores['kscore_list'])
                    final_class_type_results[class_type]['qyn_list'].extend(scores['qyn_list'])
                    final_class_type_results[class_type]['qscore_list'].extend(scores['qscore_list'])
                    final_overall_results['fyn_list'].extend(scores['fyn_list'])
                    final_overall_results['fscore_list'].extend(scores['fscore_list'])
                    final_overall_results['kyn_list'].extend(scores['kyn_list'])
                    final_overall_results['kscore_list'].extend(scores['kscore_list'])
                    final_overall_results['qyn_list'].extend(scores['qyn_list'])
                    final_overall_results['qscore_list'].extend(scores['qscore_list'])
            os.remove(temp_file)

        # Calculate the final results
        final_results = {}
        for class_type, scores in final_class_type_results.items():
            num_of_images = len(scores['fscore_list'])
            final_results[class_type] = {
                "num of eval images": num_of_images,
                "follow_yn": sum(scores['fyn_list']) / num_of_images,
                "follow_score": sum(scores['fscore_list']) / num_of_images,
                "keep_yn": sum(scores['kyn_list']) / num_of_images,
                "keep_score": sum(scores['kscore_list']) / num_of_images,
                "quality_yn": sum(scores['qyn_list']) / num_of_images,
                "quality_yn_score": sum(scores['qscore_list']) / num_of_images
            }

        num_of_images = len(final_overall_results['fscore_list'])
        overall_summary = {
            "num of eval images": num_of_images,
            "follow_yn": sum(final_overall_results['fyn_list']) / num_of_images,
            "follow_score": sum(final_overall_results['fscore_list']) / num_of_images,
            "keep_yn": sum(final_overall_results['kyn_list']) / num_of_images,
            "keep_score": sum(final_overall_results['kscore_list']) / num_of_images,
            "quality_yn": sum(final_overall_results['qyn_list']) / num_of_images,
            "quality_yn_score": sum(final_overall_results['qscore_list']) / num_of_images
        }

        # Print and save the results of different class_types
        for class_type, metrics in final_results.items():
            print(f"Results for class_type {class_type}:")
            table_data = [[metric, value] for metric, value in metrics.items()]
            table_str = tabulate(table_data, headers=["Metric", "Value"], tablefmt="grid")
            print(table_str)
            with open(f"{save_dir}/result_{class_type}.txt", "w") as f:
                f.write(table_str + "\n")

        # Print and save the overall results
        print("Overall Results:")
        overall_table_data = [[metric, value] for metric, value in overall_summary.items()]
        overall_table_str = tabulate(overall_table_data, headers=["Metric", "Value"], tablefmt="grid")
        print(overall_table_str)
        with open(f"{save_dir}/result_summary.txt", "w") as f:
            f.write(overall_table_str + "\n")

        print(f"Saved to {save_dir}/result_summary.txt")
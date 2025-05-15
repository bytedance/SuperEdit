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

import io
import json
import openai
import base64

from PIL import Image
from time import time
# import logid


headers={
        "security-switch": json.dumps({"pre": "open", "post-stream": "open", "post-not-stream": "close"}),
        "scene": "action_ai_research",
}

def read_base64_img(img):
    buf = io.BytesIO()
    if isinstance(img, str):
        img = Image.open(img)
    img.save(buf, format='JPEG')
    byte_im = buf.getvalue()
    orig_base64_image = base64.b64encode(byte_im).decode('utf-8')
    return orig_base64_image

def call_azure_gpt4v(image_pair, prompt, azure_endpoint, api_key):
    orig_img, edited_img = image_pair
    orig_base64_image = read_base64_img(orig_img)
    edited_base64_image = read_base64_img(edited_img)

    client = openai.AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_version="2023-07-01-preview",
        api_key=api_key
    )

    # logid_gx = logid.generate()
    # headers["x-tt-logid"] = logid_gx
    start = time()
    try:
        response = client.chat.completions.create(
        extra_headers=headers,
        model="gpt-4o-2024-05-13",
        messages=[
            {
                "role": "system",
                "content":
                    "You are an advanced AI designed to assess the consistency of image edits in areas unrelated to the given instructions."
                    "Your task is to examine an edited image and determine if the areas unrelated to the editing instructions remain consistent with the original image. Here's how you can perform the evaluation:" 
                    "------"
                    "## INSTRUCTIONS: "
                    "- Focus on the areas of the edited image that are unrelated to the given instructions. These areas should remain consistent with the original image and should not be affected by the editing process.\n" 
                    "- The edited image must be consistent with the original image in the areas unrelated to the editing instructions.\n"
                    "- Consider alternative interpretations or creative approaches that still meet the consistency criteria as valid.\n"
                    "- Assess the consistency of the non-edited areas in comparison to the original image."
            },
            {
            "role": "user",
            "content": [
                {"type": "text", "text":
                    "Please evaluate the following image edit based on the provided instructions and the original image:\n\n"
                    "The first image is the original image and the second image is the edited image\n"
                    f"Editing Instructions: {prompt}\n"
                    "Based on your evaluation, answer the following two questions:\n"
                    "1. Does the area of the edited image that is unrelated to the editing instructions remain consistent with the original image? Please respond with 'yes' or 'no'.\n"
                    "2. Provide your evaluation solely as an image consistency score where the image consistency score is a float value between 0 and 5, with 5 indicating the highest level of consistency with the original image.\n"
                    "Please generate the response in the form of a Python dictionary string with keys 'consistent' and 'score'. The value of 'consistent' should be a string ('yes' or 'no') indicating whether the area of the edited image that is unrelated to the editing instructions remains consistent with the original image, and the value of 'score' should be a float indicating the image consistency score.\n"
                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string.\n"
                    "For example, your response should look like this: \"{'consistent': 'yes', 'score': 4.5}\"."
                },
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{orig_base64_image}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{edited_base64_image}"}}
            ],
            }
        ],
        max_tokens=50,
        timeout=100,
        temperature=0.0
        )
    except Exception as e:
        print(e)
        return None
    score = response.choices[0].message.content
    # print(time() - start, logid_gx)
    return score


if __name__ == "__main__":
    original_img = "path_to_original_image"
    edited_img = "path_to_edited_image"
    image_pair = [original_img, edited_img]
    prompt = "editing_prompt"
    score = call_azure_gpt4v(image_pair, prompt)
    print(score)

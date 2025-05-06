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
    base64_image = base64.b64encode(byte_im).decode('utf-8')
    return base64_image

def call_azure_gpt4v(edited_img, azure_endpoint, api_key):
    edited_base64_image = read_base64_img(edited_img)

    client = openai.AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_version="2023-07-01-preview",
        api_key=api_key
    )

    try:
        response = client.chat.completions.create(
        extra_headers=headers,
        model="gpt-4o-2024-05-13",
        messages=[
            {
                "role": "system",
                "content":
                    "You are a sophisticated AI model trained to evaluate the quality of images.\n"
                    "Your task is to examine an image and evaluate its quality based on various aspects such as clarity, composition, lighting, subject matter, and whether the edits appear natural. Here's how you can perform the evaluation:\n"
                    "------"
                    "## INSTRUCTIONS:\n"
                    "- Pay close attention to the clarity of the image. The image should be sharp and the details should be clear.\n"
                    "- Look for any generated artifacts in the image. There should be no artificial patterns or distortions caused by the image generation process.\n"
                    "- Assess the composition of the image. The arrangement of elements should be balanced and visually appealing.\n"
                    "- Evaluate the lighting in the image. The lighting should be appropriate for the scene and enhance the subject matter.\n"
                    "- Consider the subject matter of the image. The subject should be well-defined and contribute to the overall quality of the image.\n"
                    "- Check if the edits made to the image appear natural. The modifications should blend seamlessly with the original elements, without any obvious signs of tampering or inconsistency."
            },
            {
            "role": "user",
            "content": [
                {"type": "text", "text":
                    "Please evaluate the quality of the following image based on its clarity, the presence of any generated artifacts, composition, lighting, subject matter, and whether the edits appear natural.\n"
                    "Based on your evaluation, answer the following two questions:\n"
                    "1. Is the image clear, free of generated artifacts, well-composed, properly lit, with a well-defined subject, and do the edits appear natural? Please respond with 'yes' or 'no'.\n"
                    "2. Provide your evaluation as an image quality score where the image quality score is a float value between 0 and 5, with 5 indicating the highest level of quality.\n"
                    "Please generate the response in the form of a Python dictionary string with keys 'clear' and 'score'. The value of 'clear' should be a string ('yes' or 'no') indicating whether the image meets all the criteria, and the value of 'score' should be a float indicating the image quality score.\n"
                    "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string.\n"
                    "For example, your response should look like this: \"{'clear': 'yes', 'score': 4.5}\"."
                },
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
    return score


if __name__ == "__main__":
    original_img = "path_to_original_image"
    edited_img = "path_to_edited_image"
    score = call_azure_gpt4v(edited_img)
    print(score)

"""
Copyright (c) 2024, UChicago Argonne, LLC. All rights reserved.

Copyright 2024. UChicago Argonne, LLC. This software was produced
under U.S. Government contract DE-AC02-06CH11357 for Argonne National
Laboratory (ANL), which is operated by UChicago Argonne, LLC for the
U.S. Department of Energy. The U.S. Government has rights to use,
reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR
UChicago Argonne, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is
modified to produce derivative works, such modified software should
be clearly marked, so as not to confuse it with the version available
from ANL.

Additionally, redistribution and use in source and binary forms, with
or without modification, are permitted provided that the following
conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in
      the documentation and/or other materials provided with the
      distribution.

    * Neither the name of UChicago Argonne, LLC, Argonne National
      Laboratory, ANL, the U.S. Government, nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY UChicago Argonne, LLC AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL UChicago
Argonne, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

### Initial Author <2024>: Xiangyu Yin

import os
import time
import base64
import requests
from typing import Optional, Annotated, List

from nodeology.client import R2R_Client
from nodeology.state import State
from nodeology.node import Node, as_node
from nodeology.log import log_print_color

MARKER_API_URL = "https://www.datalab.to/api/v1/marker"
DATALAB_API_KEY = os.environ.get("DATALAB_API_KEY")


class KnowledgeState(State):
    paper_text: str
    paper_images: List[str]
    paper_summary: str
    pdf_info: tuple[str, str]
    log: str
    log_summary: str
    attributes: str
    insights: str
    effects: str


class RAGState(State):
    query: str
    context: List[dict]
    rag_response: str


@as_node(sink=["paper_text", "images"])
def pdf2md(file_name, file_path, output_dir, api_key=DATALAB_API_KEY):
    # Check if the file has already been processed
    markdown_file = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}.md")
    images_folder = os.path.join(output_dir, f"{os.path.splitext(file_name)[0]}_images")

    if os.path.exists(markdown_file) and os.path.exists(images_folder):
        # Read the markdown content
        with open(markdown_file, "r", encoding="utf-8") as f:
            markdown_content = f.read()

        # Get the list of image paths
        image_paths = [
            os.path.join(images_folder, img)
            for img in os.listdir(images_folder)
            if img.lower().endswith((".png", ".jpg", ".jpeg", ".gif"))
        ]

        # Return the markdown content and image paths
        return markdown_content, image_paths

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if api_key is not None:
        file_content = open(file_path, "rb")
        # Prepare the form data
        form_data = {
            "file": (file_name, file_content, "application/pdf"),
            "langs": (None, "English"),
            "force_ocr": (None, False),
            "paginate": (None, False),
        }

        # Set up the headers
        headers = {"X-Api-Key": api_key}

        # Make the initial request
        response = requests.post(MARKER_API_URL, files=form_data, headers=headers)
        data = response.json()

        if data["success"]:
            # Get the URL to check the request status
            check_url = data["request_check_url"]

            # Poll for results
            max_polls = 300
            for i in range(max_polls):
                time.sleep(2)
                response = requests.get(check_url, headers=headers)
                data = response.json()

                if data["status"] == "complete":
                    break

            # Process the results
            if data["success"]:
                # Create a folder for images
                os.makedirs(images_folder, exist_ok=True)

                # Save markdown content
                with open(markdown_file, "w", encoding="utf-8") as f:
                    f.write(data["markdown"])
                log_print_color(f"Markdown content saved to: {markdown_file}", "white")

                # Save images and collect image paths
                image_paths = []
                for filename, image_data in data["images"].items():
                    image_path = os.path.join(images_folder, filename)
                    with open(image_path, "wb") as f:
                        f.write(base64.b64decode(image_data))
                    image_paths.append(image_path)
                log_print_color(f"Images saved to: {images_folder}", "white")

                # Return markdown content and list of image paths
                return data["markdown"], image_paths
    else:
        from marker.converters.pdf import PdfConverter
        from marker.models import create_model_dict
        from marker.output import text_from_rendered

        converter = PdfConverter(
            artifact_dict=create_model_dict(),
        )
        rendered = converter(file_path)
        full_text, _, images = text_from_rendered(rendered)
        # Create a folder for images
        os.makedirs(images_folder, exist_ok=True)

        # Save markdown content
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(full_text)
        log_print_color(f"Markdown content saved to: {markdown_file}", "white")

        # Save images and collect image paths
        image_paths = []
        for filename, image_data in images.items():
            image_path = os.path.join(images_folder, filename)
            image_data.save(image_path)
            image_paths.append(image_path)
        log_print_color(f"Images saved to: {images_folder}", "white")

        # Return markdown content and list of image paths
        return full_text, image_paths

    return None, None


content_summarizer = Node(
    node_type="content_summarizer",
    prompt_template="""# INPUT TEXT:
{paper_text}

# Instructions:
{instructions}
The following sections should keep original text if present: {keep_original}
Only return the markdown text, do not add texts before or after.""",
    sink=["paper_summary"],
)


def content_summarizer_pre_process(state, client, **kwargs):
    sections = kwargs.get(
        "sections", ["Overview", "Key Points", "Details", "Conclusions"]
    )
    keep_original = kwargs.get("keep_original", ["Overview"])

    default_instructions = f"Summarize the text and output a markdown file with sections: {', '.join(f'# {s}' for s in sections)}."
    kwargs["instructions"] = kwargs.get("custom_instructions", default_instructions)
    kwargs["keep_original"] = ", ".join(keep_original)

    return state


content_summarizer.pre_process = content_summarizer_pre_process

# Attributes Extractor Template
attributes_extractor = Node(
    node_type="attributes_extractor",
    prompt_template="""# INPUT TEXT:
{paper_text}

# Instructions:
{extraction_prompt}
Only return the Markdown, do not add texts before or after.""",
    sink=["attributes"],
)


def attributes_extractor_pre_process(state, client, **kwargs):
    columns = kwargs.get("columns", ["attribute", "description", "value", "unit"])
    attribute_type = kwargs.get("attribute_type", "settings")

    default_prompt = f"Extract all {attribute_type} from the text.\nReturn a markdown table with columns including {', '.join(columns)}."
    kwargs["extraction_prompt"] = kwargs.get("extraction_prompt", default_prompt)

    return state


def attributes_extractor_post_process(state, client, **kwargs):
    sink = kwargs.get("sink")
    attribute_type = kwargs.get("attribute_type", "settings")

    if sink:
        state[sink] = state["attributes"]
    elif attribute_type in state:
        state[attribute_type] = state["attributes"]

    return state


attributes_extractor.pre_process = attributes_extractor_pre_process
attributes_extractor.post_process = attributes_extractor_post_process

# Effect Analyzer Template
effect_analyzer = Node(
    node_type="effect_analyzer",
    prompt_template="""# INPUT TEXT:
{paper_text}

# Instructions:
{analysis_prompt}
Only return the markdown text, do not add texts before or after.""",
    sink=["effects"],
)


def effect_analyzer_pre_process(state, client, **kwargs):
    analysis_type = kwargs.get("analysis_type", "params")
    default_prompt = f"""Analyze the {analysis_type} effects in the text. Identify:
(1) how changes affect outcomes
(2) the generality of these effects
Return a markdown table with columns including name, change, amount, effect, generality."""

    kwargs["analysis_prompt"] = kwargs.get("custom_prompt", default_prompt)

    return state


def effect_analyzer_post_process(state, client, **kwargs):
    sink = kwargs.get("sink")
    analysis_type = kwargs.get("analysis_type", "params")

    if sink:
        state[sink] = state["effects"]
    elif analysis_type + "_effects" in state:
        state[analysis_type + "_effects"] = state["effects"]

    return state


effect_analyzer.pre_process = effect_analyzer_pre_process
effect_analyzer.post_process = effect_analyzer_post_process


# Questions Generator Template
questions_generator = Node(
    node_type="questions_generator",
    prompt_template="""# ATTRIBUTES DESCRIPTIONS:
{attributes_desc}

# EXAMPLE QUESTIONS:
{example_questions}

# Instructions:
Generate a list of {question_type} questions based on the ATTRIBUTES DESCRIPTIONS.
The questions should be asked in a conversational manner one by one.
Be careful about units and conditional dependencies between attributes.
Output MUST be a list of questions ONLY.""",
    sink=["questions"],
)


def questions_generator_pre_process(state, client, **kwargs):
    default_examples = [
        "What is the value of X?",
        "How many units of Y do you want?",
        "Do you want to enable feature Z? If yes, please provide the required settings.",
    ]

    question_type = kwargs.get("question_type", "params")
    source = kwargs.get("source", f"{question_type}_desc")

    kwargs["attributes_desc"] = state[source]
    kwargs["example_questions"] = "\n".join(
        f"- {q}" for q in kwargs.get("example_questions", default_examples)
    )
    kwargs["question_type"] = question_type

    return state


def questions_generator_post_process(state, client, **kwargs):
    sink = kwargs.get("sink")
    question_type = kwargs.get("question_type", "params")

    if sink:
        state[sink] = state["questions"]
    elif question_type + "_questions" in state:
        state[question_type + "_questions"] = state["questions"]

    return state


questions_generator.pre_process = questions_generator_pre_process
questions_generator.post_process = questions_generator_post_process

# Log Summarizer Template
log_summarizer = Node(
    node_type="log_summarizer",
    prompt_template="""# LOG TEXT:
{log}

# Instructions:
{summary_prompt}
Only return the markdown table, do not add texts before or after.""",
    sink=["log_summary"],
)


def log_summarizer_pre_process(state, client, **kwargs):
    columns = kwargs.get(
        "columns", ["timestamp", "event", "details", "status", "next_steps"]
    )
    default_prompt = f"Summarize the log text and output a markdown table with columns: {', '.join(columns)}."

    kwargs["summary_prompt"] = kwargs.get("custom_prompt", default_prompt)
    kwargs["log"] = state[kwargs.get("source", "log")]

    return state


log_summarizer.pre_process = log_summarizer_pre_process

# Insights Extractor Template
insights_extractor = Node(
    node_type="insights_extractor",
    prompt_template="""# INPUT DATA:
{log_summary}

# Instructions:
{analysis_prompt}""",
    sink=["insights"],
)


def insights_extractor_pre_process(state, client, **kwargs):
    default_aspects = [
        "Key trends and patterns",
        "Notable correlations",
        "Anomalies or outliers",
        "Actionable recommendations",
    ]

    aspects = kwargs.get("analysis_aspects", default_aspects)
    default_prompt = f"""Analyze the data and extract insights about:
{chr(10).join(f'- {aspect}' for aspect in aspects)}
Answers should be concise but include specific values where relevant."""

    kwargs["analysis_prompt"] = kwargs.get("custom_prompt", default_prompt)

    return state


insights_extractor.pre_process = insights_extractor_pre_process


@as_node(sink=["context"])
def context_retriever(
    query: Annotated[str, "Search query"],
    r2r_client: Annotated[R2R_Client, "R2R client instance"],
    use_hybrid: Annotated[bool, "Whether to use hybrid search"] = True,
    search_filters: Annotated[Optional[dict], "Additional search filters"] = None,
) -> list:
    vector_search_settings = {"use_hybrid_search": use_hybrid, **(search_filters or {})}

    results = r2r_client.search(
        query,
        vector_search_settings=vector_search_settings,
    )[
        "results"
    ]["vector_search_results"]

    return results


@as_node(sink=["rag_response", "context"])
def context_augmented_generator(
    query: Annotated[str | list, "Query text or conversation history"],
    r2r_client: Annotated[R2R_Client, "R2R client instance"],
    prompt_override: Annotated[Optional[str], "Optional prompt override"] = None,
) -> tuple:

    if isinstance(query, list):
        response = r2r_client(query)
        search_results = []
    elif isinstance(query, str):
        results = r2r_client.client.rag(
            query,
            rag_generation_config={"model": r2r_client.llm_name},
            vector_search_settings={
                "use_hybrid_search": r2r_client.search_strategy == "hybrid",
                **(
                    {"search_strategy": r2r_client.rag_strategy}
                    if r2r_client.rag_strategy != "vanilla"
                    else {}
                ),
            },
            task_prompt_override=prompt_override,
        )["results"]
        response = results["completion"]["choices"][0]["message"]["content"]
        search_results = results["search_results"]["vector_search_results"]

    return response, search_results

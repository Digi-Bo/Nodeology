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

### This is a demo of simplified ptychography parameter optimization workflow at automation level 2 from PEAR.
### "PEAR: A Robust and Flexible Automation Framework for Ptychography Enabled by Multiple Large Language Model Agents"
### https://arxiv.org/abs/2410.09034

import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nodeology.prebuilt import (
    survey,
    formatter,
    recommender,
    updater,
    code_rewriter,
    execute_code,
    commentator,
    HilpState,
    ParamsOptState,
    CodingState,
    RecommendationState,
    DiagnosisState,
)
from nodeology.workflow import Workflow
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver


class PtychoParamsOpt(Workflow):
    def __init__(
        self,
        data_path,
        script_path,
        knowledge_path,
        executable_path,
        capture_output_func=lambda x: x.strip().split("\n")[-1],
        result2image_convert_func=lambda x: x,
        **kwargs,
    ) -> None:
        """Initialize PtychoParamsOpt workflow

        Args:
            data_path: Path to data directory
            script_path: Path to reconstruction script
            knowledge_path: Path to knowledge base directory
            executable_path: Path to executable
            capture_output_func: Function to capture output
            result2image_convert_func: Function to convert results to images
            memory_saver: Memory saver instance
            **kwargs: Additional arguments passed to Workflow
        """
        # Store instance variables before super().__init__
        self.data_path = data_path
        self.script_path = script_path
        self.knowledge_path = knowledge_path
        self.executable_path = executable_path
        self.capture_output_func = capture_output_func
        self.result2image_convert_func = result2image_convert_func

        # Call parent constructor with required arguments
        super().__init__(
            name="ptycho_params_opt",
            state_defs=[
                HilpState,
                ParamsOptState,
                CodingState,
                RecommendationState,
                DiagnosisState,
            ],
            llm_name="gpt-4o",
            vlm_name="gpt-4o",
            **kwargs,
        )

        # Load knowledge base and initialize workflow
        # self._read_knowledge()
        # self.initialize({
        #     "data_path": self.data_path,
        #     "params_desc": self.params_desc,
        #     "params_questions": self.params_questions,
        #     "recommender_knowledge": self.recommender_knowledge,
        #     "code_example": self.example_recon_script,
        #     "example_diagnosis": self.example_recon_qualities
        # })

    def _read_knowledge(self):
        """Read necessary knowledge base files

        The example knowledge_path directory structure:
        knowledge_path/
        ├── example_script.py          # Example reconstruction script
        ├── params_description.md      # Parameter descriptions
        ├── params_questions.md        # Questions for parameter collection
        ├── params_knowledge.md        # Knowledge for AI recommendations
        └── example_recons/           # Example reconstructions
            ├── example1.png          # Example reconstruction image1
            ├── example1.md           # Quality description for example1
            ├── example2.png          # Example reconstruction image2
            └── example2.md           # Quality description for example2
        """
        # Read example reconstruction script
        with open(f"{self.knowledge_path}/example_script.py", "r") as f:
            self.example_recon_script = f.read()

        # Read parameter descriptions
        with open(f"{self.knowledge_path}/params_description.md", "r") as f:
            self.params_desc = f.read()

        # Read parameter questions for AI
        with open(f"{self.knowledge_path}/params_questions.md", "r") as f:
            self.params_questions = f.read()

        # Read recommendation knowledge base
        with open(f"{self.knowledge_path}/params_knowledge.md", "r") as f:
            self.recommender_knowledge = f.read()

        # Read example reconstructions and their qualities
        self.example_recon_path = f"{self.knowledge_path}/example_recons"
        self.example_recon_qualities = {}
        if os.path.exists(self.example_recon_path):
            for recon_name in os.listdir(self.example_recon_path):
                if recon_name.endswith((".png", ".jpg")):
                    recon_path = f"{self.example_recon_path}/{recon_name}"
                    quality_path = f"{self.example_recon_path}/{os.path.splitext(recon_name)[0]}.md"
                    if os.path.exists(quality_path):
                        with open(quality_path, "r") as f:
                            self.example_recon_qualities[recon_path] = f.read()

    def create_workflow(self):
        """Create fully automated workflow (automation_level=2)"""
        # Add nodes with simplified syntax
        self.add_node("params_collector", survey)
        self.add_node("params_formatter", formatter, source="conversation_summary")
        self.add_node("params_recommender", recommender, source="conversation_summary")
        self.add_node("params_updater", updater, source="recommendation")
        self.add_node("params_confirmer", updater, source="human_input")
        self.add_node(
            "script_generator", code_rewriter, source={"context": "params_desc"}
        )
        self.add_node(
            "script_runner",
            execute_code,
            executable_path=self.executable_path,
            capture_output_func=self.capture_output_func,
        )
        self.add_node(
            "quality_commentator",
            commentator,
            result2image_convert_func=self.result2image_convert_func,
        )
        self.add_node("updates_recommender", recommender, source="diagnosis")

        # Add edges with simplified syntax
        self.add_conditional_flow(
            "params_collector",
            "end_conversation",
            then="params_formatter",
            otherwise="params_collector",
        )
        self.add_flow("params_formatter", "params_recommender")
        self.add_flow("params_recommender", "params_updater")
        self.add_flow("params_updater", "params_confirmer")
        self.add_conditional_flow(
            "params_confirmer",
            "end_conversation",
            then="script_generator",
            otherwise="params_confirmer",
        )
        self.add_flow("script_generator", "script_runner")
        self.add_flow("script_runner", "quality_commentator")
        self.add_flow("quality_commentator", "updates_recommender")
        self.add_flow("updates_recommender", "params_updater")

        # Set entry point
        self.set_entry("params_collector")

        # Compile workflow
        self.compile(interrupt_before=["params_collector", "params_confirmer"])


if __name__ == "__main__":
    workflow = PtychoParamsOpt(
        data_path="data",
        script_path="script",
        knowledge_path="knowledge",
        executable_path="executable",
    )
    workflow.to_yaml("ptycho_params_opt.yaml")
    workflow.graph.get_graph().draw_mermaid_png(
        output_file_path="ptycho_params_opt.png"
    )

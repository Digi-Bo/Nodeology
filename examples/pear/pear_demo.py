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

### This is a demo of simplified PEAR workflow at automation level 2 for parameter optimization of Ptychography.
### "PEAR: A Robust and Flexible Automation Framework for Ptychography Enabled by Multiple Large Language Model Agents"
### https://arxiv.org/abs/2410.09034

import os
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


class PtychoOpt(Workflow):
    def __init__(self, data_path, script_path, knowledge_path, executable_path,
                 capture_output_func=lambda x: x.strip().split("\n")[-1],
                 result2image_convert_func=lambda x: x,
                 memory_saver=MemorySaver(),
                 **kwargs) -> None:
        """Initialize PtychoOpt workflow
        
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
        self.memory_saver = memory_saver

        # Call parent constructor with required arguments
        super().__init__(
            name="ptycho_opt",
            state_defs=[HilpState, ParamsOptState, CodingState, RecommendationState, DiagnosisState],
            llm_name="gpt-4o-mini",
            exit_commands=["stop pear", "quit pear", "terminate pear"],
            **kwargs
        )
        
        # Load knowledge base after initialization
        self._read_knowledge()
        self._initialize_pear()

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
    
    def _initialize_pear(self):
        """Initialize PEAR workflow"""
        self.initialize({
            "data_path": self.data_path,
            "params_desc": self.params_desc,
            "params_questions": self.params_questions,
            "recommender_knowledge": self.recommender_knowledge,
            "code_example": self.example_recon_script,
            "example_diagnosis": self.example_recon_qualities
        })

    def create_workflow(self):
        """Create fully automated workflow (automation_level=2)"""
        self.workflow = StateGraph(self.state_schema)

        # Add nodes for each step of the workflow
        self.workflow.add_node(
            "params_collector",
            lambda state: survey(state, self.llm_client)
        )
        self.workflow.add_node(
            "params_collector_input",
            lambda state: state
        )
        self.workflow.add_node(
            "params_formatter",
            lambda state: formatter(state, self.llm_client, source="conversation_summary")
        )
        self.workflow.add_node(
            "params_recommender",
            lambda state: recommender(state, self.llm_client, source="conversation_summary")
        )
        self.workflow.add_node(
            "params_updater",
            lambda state: updater(state, self.llm_client, source="recommendation")
        )
        self.workflow.add_node(
            "params_confirmer",
            lambda state: updater(state, self.llm_client, source="human_input")
        )
        self.workflow.add_node(
            "params_confirm_input",
            lambda state: state
        )
        self.workflow.add_node(
            "script_generator",
            lambda state: code_rewriter(state, self.llm_client, source={"context": "params_desc"})
        )
        self.workflow.add_node(
            "script_runner",
            lambda state: execute_code(
                state,
                self.llm_client,
                executable_path=self.executable_path,
                capture_output_func=self.capture_output_func
            )
        )
        self.workflow.add_node(
            "quality_commentator",
            lambda state: commentator(
                state,
                self.llm_client,
                result2image_convert_func=self.result2image_convert_func
            )
        )
        self.workflow.add_node(
            "updates_recommender",
            lambda state: recommender(state, self.llm_client, source="diagnosis")
        )

        # Add edges between nodes
        self.workflow.add_conditional_edges(
            "params_collector",
            lambda state: "then" if state["end_conversation"] else "continue",
            {
                "then": "params_formatter",
                "continue": "params_collector_input",
            }
        )
        self.workflow.add_edge("params_collector_input", "params_collector")
        self.workflow.add_edge("params_formatter", "params_recommender")
        self.workflow.add_edge("params_recommender", "params_updater")
        self.workflow.add_edge("params_updater", "params_confirmer")
        self.workflow.add_edge("params_confirmer_input", "params_confirmer")
        self.workflow.add_conditional_edges(
            "params_confirmer",
            lambda state: "then" if state["end_conversation"] else "continue",
            {
                "then": "script_generator",
                "continue": "params_confirmer_input",
            }
        )
        self.workflow.add_edge("script_generator", "script_runner")
        self.workflow.add_edge("script_runner", "quality_commentator")
        self.workflow.add_edge("quality_commentator", "updates_recommender")
        self.workflow.add_edge("updates_recommender", "params_updater")
        
        # Set entry point
        self.workflow.set_entry_point("params_collector")
        
        # Compile workflow with memory saver and interrupt points
        interrupt_before_list = ["params_collector_input", "params_confirmer_input"]
        self.graph = self.workflow.compile(
            checkpointer=self.memory_saver, 
            interrupt_before=interrupt_before_list
        )
 
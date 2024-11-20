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

from nodeology.prebuilt.coding import *
from nodeology.prebuilt.diagnosis import *
from nodeology.prebuilt.hilp import *
from nodeology.prebuilt.knowledge import *
from nodeology.prebuilt.params import *
from nodeology.prebuilt.reasoning import *

prebuilt_nodes = {
    # From reasoning.py
    "planner": planner,
    # From coding.py
    "code_executor": execute_code,
    "code_rewriter": code_rewriter,
    "error_corrector": error_corrector,
    "code_tweaker": code_tweaker,
    "code_explainer": code_explainer,
    "code_designer": code_designer,
    # From knowledge.py
    "pdf2md_converter": pdf2md,
    "content_summarizer": content_summarizer,
    "attributes_extractor": attributes_extractor,
    "effect_analyzer": effect_analyzer,
    "questions_generator": questions_generator,
    "log_summarizer": log_summarizer,
    "insights_extractor": insights_extractor,
    "context_searcher": context_retriever,
    "rag_generator": context_augmented_generator,
    # From params.py
    "formatter": formatter,
    "recommender": recommender,
    "updater": updater,
    # From hilp.py
    "conversation_summarizer": conversation_summarizer,
    "survey": survey,
    # From diagnosis.py
    "commentator": commentator,
}

prebuilt_states = {
    "HilpState": HilpState,
    "CodingState": CodingState,
    "ParamsOptState": ParamsOptState,
    "RecommendationState": RecommendationState,
    "KnowledgeState": KnowledgeState,
    "RAGState": RAGState,
    "PlanningState": PlanningState,
    "DiagnosisState": DiagnosisState,
}

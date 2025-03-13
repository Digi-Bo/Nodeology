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

### Initial Author <2025>: Xiangyu Yin

import os, json, importlib, threading, traceback, contextvars
import logging
import chainlit as cl
from chainlit.cli import run_chainlit
from nodeology.state import StateEncoder, convert_serialized_objects


logger = logging.getLogger(__name__)


def run_chainlit_for_workflow(workflow, initial_state=None):
    """
    Called by workflow.run(ui=True). This function:
      1. stores workflow and initial_state in user session data
      2. starts the chainlit server
      3. returns the final state when the workflow completes
    """
    os.environ["NODEOLOGY_WORKFLOW_CLASS"] = (
        workflow.__class__.__module__ + "." + workflow.__class__.__name__
    )

    logger.info(f"Starting UI for workflow: {workflow.__class__.__name__}")

    # Save the initialization arguments to ensure they're passed when recreating the workflow
    if hasattr(workflow, "_init_kwargs"):
        logger.info(
            f"Found initialization kwargs: {list(workflow._init_kwargs.keys())}"
        )

        # We need to handle non-serializable objects in the kwargs
        serializable_kwargs = {}
        for key, value in workflow._init_kwargs.items():
            # Handle special cases
            if key == "state_defs":
                # For state_defs, we need to handle it specially
                if value is None:
                    # If None, we'll use the workflow's state_schema
                    serializable_kwargs[key] = None
                    logger.info(
                        f"Serializing {key} as None (will use workflow's state_schema)"
                    )
                elif isinstance(value, type) and hasattr(value, "__annotations__"):
                    # If it's a TypedDict or similar class with annotations, we'll use its name
                    # The workflow class will handle recreating it
                    serializable_kwargs["_state_defs_class"] = (
                        f"{value.__module__}.{value.__name__}"
                    )
                    logger.info(
                        f"Serializing state_defs as class reference: {serializable_kwargs['_state_defs_class']}"
                    )
                elif isinstance(value, list):
                    # If it's a list of state definitions, we'll try to serialize it
                    # This is complex and might not work for all cases
                    try:
                        # Convert any TypedDict classes to their module.name string
                        serialized_list = []
                        for item in value:
                            if isinstance(item, type) and hasattr(
                                item, "__annotations__"
                            ):
                                serialized_list.append(
                                    f"{item.__module__}.{item.__name__}"
                                )
                            elif isinstance(item, tuple) and len(item) == 2:
                                # Handle (name, type) tuples
                                name, type_hint = item
                                if isinstance(type_hint, type):
                                    serialized_list.append(
                                        [
                                            name,
                                            f"{type_hint.__module__}.{type_hint.__name__}",
                                        ]
                                    )
                                else:
                                    serialized_list.append([name, str(type_hint)])
                            elif isinstance(item, dict) and len(item) == 1:
                                # Handle {"name": type} dictionaries
                                name, type_hint = next(iter(item.items()))
                                if isinstance(type_hint, type):
                                    serialized_list.append(
                                        {
                                            name: f"{type_hint.__module__}.{type_hint.__name__}"
                                        }
                                    )
                                else:
                                    serialized_list.append({name: str(type_hint)})
                            else:
                                # Skip items we can't serialize
                                logger.info(
                                    f"Skipping non-serializable state_def item: {item}"
                                )

                        if serialized_list:
                            serializable_kwargs["_state_defs_list"] = serialized_list
                            logger.info(
                                f"Serializing state_defs as list: {serialized_list}"
                            )
                        else:
                            logger.info("Could not serialize any state_defs items")
                    except Exception as e:
                        logger.error(f"Error serializing state_defs list: {str(e)}")
                else:
                    logger.info(
                        f"Cannot serialize state_defs of type {type(value).__name__}"
                    )
            elif key == "checkpointer":
                # For checkpointer, just store "memory" if it's a string or an object
                if isinstance(value, str):
                    serializable_kwargs[key] = value
                else:
                    serializable_kwargs[key] = "memory"
                logger.info(f"Serializing checkpointer as: {serializable_kwargs[key]}")
            elif isinstance(value, (str, int, float, bool, type(None))):
                serializable_kwargs[key] = value
                logger.info(
                    f"Serializing {key} as primitive type: {type(value).__name__}"
                )
            elif isinstance(value, list) and all(
                isinstance(item, (str, int, float, bool, type(None))) for item in value
            ):
                serializable_kwargs[key] = value
                logger.info(f"Serializing {key} as list of primitives")
            elif isinstance(value, dict) and all(
                isinstance(k, str)
                and isinstance(v, (str, int, float, bool, type(None)))
                for k, v in value.items()
            ):
                serializable_kwargs[key] = value
                logger.info(f"Serializing {key} as dict of primitives")
            else:
                logger.info(
                    f"Skipping non-serializable {key} of type {type(value).__name__}"
                )
            # Skip other complex objects that can't be easily serialized

        # For client objects, just store their names
        if (
            "llm_name" in workflow._init_kwargs
            and hasattr(workflow, "llm_client")
            and hasattr(workflow.llm_client, "model_name")
        ):
            serializable_kwargs["llm_name"] = workflow.llm_client.model_name
            logger.info(
                f"Using llm_client.model_name: {workflow.llm_client.model_name}"
            )

        if (
            "vlm_name" in workflow._init_kwargs
            and hasattr(workflow, "vlm_client")
            and hasattr(workflow.vlm_client, "model_name")
        ):
            serializable_kwargs["vlm_name"] = workflow.vlm_client.model_name
            logger.info(
                f"Using vlm_client.model_name: {workflow.vlm_client.model_name}"
            )

        # Store the workflow's state_schema class name if available
        if hasattr(workflow, "state_schema") and hasattr(
            workflow.state_schema, "__name__"
        ):
            serializable_kwargs["_state_schema_class"] = (
                f"{workflow.state_schema.__module__}.{workflow.state_schema.__name__}"
            )
            logger.info(
                f"Storing state_schema class: {serializable_kwargs['_state_schema_class']}"
            )

        os.environ["NODEOLOGY_WORKFLOW_ARGS"] = json.dumps(
            serializable_kwargs, cls=StateEncoder
        )
        logger.info(f"Serialized kwargs: {list(serializable_kwargs.keys())}")
    else:
        logger.info("No initialization kwargs found on workflow")

    # Serialize any initial state if needed
    if initial_state:
        # Use StateEncoder to handle NumPy arrays
        os.environ["NODEOLOGY_INITIAL_STATE"] = json.dumps(
            initial_state, cls=StateEncoder
        )
        logger.info("Serialized initial state")

    # Create a shared variable to store the final state
    os.environ["NODEOLOGY_FINAL_STATE"] = "{}"

    # This file is nodeology/chainlit_interface.py, get its path:
    this_file = os.path.abspath(__file__)
    # Start with some standard arguments
    logger.info("Starting Chainlit server")
    run_chainlit(target=this_file)

    # Return the final state from the last session
    final_state = {}
    if (
        "NODEOLOGY_FINAL_STATE" in os.environ
        and os.environ["NODEOLOGY_FINAL_STATE"] != "{}"
    ):
        try:
            final_state_json = os.environ["NODEOLOGY_FINAL_STATE"]
            final_state_dict = json.loads(final_state_json)
            logger.info(
                f"Retrieved final state with keys: {list(final_state_dict.keys())}"
            )

            # Convert any serialized NumPy arrays back to arrays
            final_state = convert_serialized_objects(final_state_dict)
            logger.info("Converted any serialized objects in final state")

        except Exception as e:
            logger.error(f"Error parsing final state: {str(e)}")

    return final_state


@cl.on_chat_start
async def on_chat_start():
    """
    Called once a new user session is started in the chainlit UI.
    We will instantiate a new workflow for this session.
    """
    try:
        # Get the workflow class from environment variable
        workflow_class_path = os.environ.get("NODEOLOGY_WORKFLOW_CLASS")
        if not workflow_class_path:
            await cl.Message(content="No workflow class specified.").send()
            return

        logger.info(f"Creating workflow from class: {workflow_class_path}")

        # Import the workflow class dynamically
        module_path, class_name = workflow_class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        WorkflowClass = getattr(module, class_name)
        logger.info(f"Successfully imported workflow class: {class_name}")

        # Get the saved initialization arguments
        workflow_args = {}
        state_defs_processed = False

        if "NODEOLOGY_WORKFLOW_ARGS" in os.environ:
            try:
                serialized_args = json.loads(os.environ["NODEOLOGY_WORKFLOW_ARGS"])
                logger.info(f"Loaded serialized args: {list(serialized_args.keys())}")

                # Handle special parameters

                # 1. Handle state_defs
                if "_state_defs_class" in serialized_args:
                    # We have a class reference for state_defs
                    state_defs_class_path = serialized_args.pop("_state_defs_class")
                    try:
                        module_path, class_name = state_defs_class_path.rsplit(".", 1)
                        module = importlib.import_module(module_path)
                        state_defs_class = getattr(module, class_name)
                        workflow_args["state_defs"] = state_defs_class
                        logger.info(
                            f"Imported state_defs class: {state_defs_class_path}"
                        )
                        state_defs_processed = True
                    except Exception as e:
                        logger.error(f"Error importing state_defs class: {str(e)}")
                elif "_state_defs_list" in serialized_args:
                    # We have a list of state definitions
                    state_defs_list = serialized_args.pop("_state_defs_list")
                    try:
                        # Process each item in the list
                        processed_list = []
                        for item in state_defs_list:
                            if isinstance(item, str):
                                # It's a class reference
                                try:
                                    module_path, class_name = item.rsplit(".", 1)
                                    module = importlib.import_module(module_path)
                                    class_obj = getattr(module, class_name)
                                    processed_list.append(class_obj)
                                except Exception as e:
                                    logger.error(
                                        f"Error importing state def class {item}: {str(e)}"
                                    )
                            elif isinstance(item, list) and len(item) == 2:
                                # It's a [name, type] tuple
                                name, type_str = item
                                if "." in type_str:
                                    # It's a class reference
                                    try:
                                        module_path, class_name = type_str.rsplit(
                                            ".", 1
                                        )
                                        module = importlib.import_module(module_path)
                                        type_obj = getattr(module, class_name)
                                        processed_list.append((name, type_obj))
                                    except Exception as e:
                                        logger.error(
                                            f"Error importing type {type_str}: {str(e)}"
                                        )
                                        # Fall back to string representation
                                        processed_list.append((name, type_str))
                                else:
                                    # It's a primitive type string
                                    processed_list.append((name, type_str))
                            elif isinstance(item, dict) and len(item) == 1:
                                # It's a {name: type} dict
                                name, type_str = next(iter(item.items()))
                                if "." in type_str:
                                    # It's a class reference
                                    try:
                                        module_path, class_name = type_str.rsplit(
                                            ".", 1
                                        )
                                        module = importlib.import_module(module_path)
                                        type_obj = getattr(module, class_name)
                                        processed_list.append({name: type_obj})
                                    except Exception as e:
                                        logger.error(
                                            f"Error importing type {type_str}: {str(e)}"
                                        )
                                        # Fall back to string representation
                                        processed_list.append({name: type_str})
                                else:
                                    # It's a primitive type string
                                    processed_list.append({name: type_str})

                        if processed_list:
                            workflow_args["state_defs"] = processed_list
                            logger.info(
                                f"Processed state_defs list with {len(processed_list)} items"
                            )
                            state_defs_processed = True
                        else:
                            logger.info("No state_defs items could be processed")
                    except Exception as e:
                        logger.error(f"Error processing state_defs list: {str(e)}")
                elif (
                    "state_defs" in serialized_args
                    and serialized_args["state_defs"] is None
                ):
                    # Explicit None value
                    workflow_args["state_defs"] = None
                    serialized_args.pop("state_defs")
                    logger.info("Using None for state_defs")
                    state_defs_processed = True

                # 2. Handle state_schema if needed
                if "_state_schema_class" in serialized_args:
                    # We have a class reference for state_schema
                    state_schema_class_path = serialized_args.pop("_state_schema_class")
                    logger.info(
                        f"Found state_schema class: {state_schema_class_path} (will be handled by workflow)"
                    )

                    # If we couldn't process state_defs, try to use the state_schema class as a fallback
                    if not state_defs_processed:
                        try:
                            module_path, class_name = state_schema_class_path.rsplit(
                                ".", 1
                            )
                            module = importlib.import_module(module_path)
                            state_schema_class = getattr(module, class_name)
                            workflow_args["state_defs"] = state_schema_class
                            logger.info(
                                f"Using state_schema class as fallback for state_defs: {state_schema_class_path}"
                            )
                            state_defs_processed = True
                        except Exception as e:
                            logger.error(
                                f"Error importing state_schema class as fallback: {str(e)}"
                            )

                # Add remaining arguments
                for key, value in serialized_args.items():
                    workflow_args[key] = value

                # Convert any serialized NumPy arrays back to arrays
                workflow_args = convert_serialized_objects(workflow_args)
                logger.info(f"Final workflow args: {list(workflow_args.keys())}")
            except Exception as e:
                logger.error(f"Error parsing workflow arguments: {str(e)}")
                traceback.print_exc()
                # Continue with empty args if there's an error

        # If we couldn't process state_defs, check if the workflow class has a state_schema attribute
        if not state_defs_processed and hasattr(WorkflowClass, "state_schema"):
            logger.info(f"Using workflow class's state_schema attribute as fallback")
            # We don't need to set state_defs explicitly, the workflow will use its state_schema

        # Create a new instance of the workflow with the saved arguments
        logger.info(
            f"Creating workflow instance with args: {list(workflow_args.keys())}"
        )
        workflow = WorkflowClass(**workflow_args)
        logger.info(f"Successfully created workflow instance: {workflow.name}")

        # Check if VLM client is available
        if hasattr(workflow, "vlm_client") and workflow.vlm_client is not None:
            logger.info(f"VLM client is available")
        else:
            logger.info("VLM client is not available")

        # Get initial state if available
        initial_state = None
        initial_state_json = os.environ.get("NODEOLOGY_INITIAL_STATE")
        if initial_state_json:
            logger.info("Found initial state in environment")
            # Parse the JSON and convert any serialized NumPy arrays back to arrays
            initial_state_dict = json.loads(initial_state_json)
            logger.info(
                f"Loaded initial state with keys: {list(initial_state_dict.keys())}"
            )

            # Convert any serialized NumPy arrays back to arrays
            initial_state = convert_serialized_objects(initial_state_dict)
            logger.info("Converted any serialized objects in initial state")

        # Initialize the workflow
        if initial_state:
            logger.info(
                f"Initializing workflow with initial state: {list(initial_state.keys())}"
            )
            workflow.initialize(initial_state)
        else:
            logger.info("Initializing workflow with default state")
            workflow.initialize()
        logger.info("Workflow initialized successfully")

        # Store in user session
        cl.user_session.set("workflow", workflow)
        logger.info("Stored workflow in user session")

        # Capture the current Chainlit context
        parent_ctx = contextvars.copy_context()
        logger.info("Captured Chainlit context")

        # Create a function to save the final state when workflow completes
        def save_final_state(state):
            try:
                # Use StateEncoder to handle NumPy arrays and other complex objects
                serialized_state = json.dumps(state, cls=StateEncoder)
                os.environ["NODEOLOGY_FINAL_STATE"] = serialized_state
                logger.info(f"Saved final state with keys: {list(state.keys())}")
            except Exception as e:
                logger.error(f"Error saving final state: {str(e)}")
                traceback.print_exc()

        # Start the workflow in a background thread from within the Chainlit context
        def run_workflow_with_polling():
            try:
                logger.info("Starting workflow execution in background thread")
                # Run the workflow inside the captured context
                final_state = parent_ctx.run(
                    lambda: workflow._run(initial_state, ui=True)
                )
                logger.info("Workflow execution completed")

                # Save the final state
                if final_state:
                    save_final_state(final_state)
                    logger.info("Final state saved")
            except Exception as e:
                logger.error(f"Error in workflow execution: {str(e)}")
                traceback.print_exc()

        # Start the workflow thread
        workflow_thread = threading.Thread(
            target=run_workflow_with_polling, daemon=True
        )
        cl.user_session.set("workflow_thread", workflow_thread)
        logger.info("Created workflow thread")
        workflow_thread.start()
        logger.info("Started workflow thread")

        await cl.Message(
            content=f"Welcome to the {workflow.__class__.__name__} via Nodeology!"
        ).send()
        logger.info("Sent welcome message")
    except Exception as e:
        logger.error(f"Error in on_chat_start: {str(e)}")
        traceback.print_exc()
        await cl.Message(content=f"Error initializing workflow: {str(e)}").send()

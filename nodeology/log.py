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

import os, logging, logging.config
import sys

logger = logging.getLogger(__name__)


def setup_logging(log_dir, log_name, debug_mode=False, base_dir=None):
    """Configure the logging system with console and/or file handlers.

    Args:
        log_dir (str): Directory where log files will be stored
        log_name (str): Name of the log file (without extension)
        debug_mode (bool): If True, only console logging with debug level is enabled
        base_dir (str, optional): Base directory to prepend to log_dir
    """
    # Get root logger
    root_logger = logging.getLogger()

    # Remove any existing handlers first
    for handler in root_logger.handlers[:]:
        handler.close()  # Properly close handlers
        root_logger.removeHandler(handler)

    # Create handlers
    console_handler = logging.StreamHandler(sys.stdout)

    # Set the root logger level to DEBUG to capture all messages
    root_logger.setLevel(logging.DEBUG)

    # Use base_dir if provided, otherwise use log_dir directly
    full_log_dir = os.path.join(base_dir, log_dir) if base_dir else log_dir

    if not os.path.exists(full_log_dir):
        os.makedirs(full_log_dir)

    log_file_path = f"{full_log_dir}/{log_name}.log"

    if os.path.isfile(log_file_path):
        log_print_color(
            f"WARNING: {log_file_path} already exists and will be overwritten.",
            "red",
        )

    # Create file handler for both modes
    file_handler = logging.FileHandler(log_file_path, "w")

    if debug_mode:
        # Debug mode configuration
        # Console shows DEBUG and above
        console_handler.setLevel(logging.DEBUG)
        console_format = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        console_handler.setFormatter(console_format)

        # File handler captures everything
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )
        file_handler.setFormatter(file_format)
    else:
        # Production mode configuration
        # Console only shows PRINTLOG and WARNING+ messages
        console_handler.setLevel(logging.PRINTLOG)
        console_format = logging.Formatter("%(message)s")
        console_handler.setFormatter(console_format)

        # File handler captures everything (DEBUG and above)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s - %(message)s", datefmt="%Y%m%d-%H:%M:%S"
        )
        file_handler.setFormatter(file_format)

    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Configure third-party loggers to be less verbose
    # This prevents them from cluttering the console
    for logger_name in logging.root.manager.loggerDict:
        if logger_name != __name__ and not logger_name.startswith("nodeology"):
            third_party_logger = logging.getLogger(logger_name)
            if debug_mode:
                # In debug mode, third-party loggers show WARNING and above
                third_party_logger.setLevel(logging.WARNING)
            else:
                # In production mode, third-party loggers show ERROR and above
                third_party_logger.setLevel(logging.ERROR)

    # Store handlers in logger for later cleanup
    root_logger.handlers_to_close = root_logger.handlers[:]


def cleanup_logging():
    """Properly clean up logging handlers to prevent resource leaks."""
    root_logger = logging.getLogger()

    # Close and remove any existing handlers
    if hasattr(root_logger, "handlers_to_close"):
        for handler in root_logger.handlers_to_close:
            try:
                handler.close()
            except:
                pass  # Ignore errors during cleanup
            if handler in root_logger.handlers:
                root_logger.removeHandler(handler)
        root_logger.handlers_to_close = []


# https://stackoverflow.com/questions/2183233/how-to-add-a-custom-loglevel-to-pythons-logging-facility/35804945#35804945
def add_logging_level(levelName, levelNum, methodName=None):
    """Add a new logging level to the logging module.

    Args:
        levelName (str): Name of the new level (e.g., 'TRACE')
        levelNum (int): Numeric value for the level
        methodName (str, optional): Method name to add. Defaults to levelName.lower()

    Raises:
        AttributeError: If levelName or methodName already exists
    """
    if not methodName:
        methodName = levelName.lower()

    if hasattr(logging, levelName):
        raise AttributeError("{} already defined in logging module".format(levelName))
    if hasattr(logging, methodName):
        raise AttributeError("{} already defined in logging module".format(methodName))
    if hasattr(logging.getLoggerClass(), methodName):
        raise AttributeError("{} already defined in logger class".format(methodName))

    # This method was inspired by the answers to Stack Overflow post
    # http://stackoverflow.com/q/2183233/2988730, especially
    # http://stackoverflow.com/a/13638084/2988730
    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)


def log_print_color(text, color="", print_to_console=True):
    """Print colored text to console and log it to file.

    Args:
        text (str): Text to print and log
        color (str): Color name ('green', 'red', 'blue', 'yellow', or '' for white)
        print_to_console (bool): If True, print the text to console
    """
    # Define color codes as constants at the top of the function
    COLOR_CODES = {
        "green": "\033[92m",
        "red": "\033[91m",
        "blue": "\033[94m",
        "yellow": "\033[93m",
        "": "\033[97m",  # default white
    }

    # Get color code from dictionary, defaulting to white
    ansi_code = COLOR_CODES.get(color, COLOR_CODES[""])

    if print_to_console:
        print(ansi_code + text + "\033[0m")
    logger.logonly(text)

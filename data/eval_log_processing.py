from inspect_ai.log import EvalLog, read_eval_log, resolve_sample_attachments
from typing import Literal
import logging

logger = logging.getLogger(__name__)


async def read_eval_log_async(
    log_file: str ,
    header_only: bool = False,
    resolve_attachments: bool = False,
    format: Literal["eval", "json", "auto"] = "auto",
) -> EvalLog:
    """Read an evaluation log.

    Args:
       log_file (str | FileInfo): Log file to read.
       header_only (bool): Read only the header (i.e. exclude
         the "samples" and "logging" fields). Defaults to False.
       resolve_attachments (bool): Resolve attachments (e.g. images)
          to their full content.
       format (Literal["eval", "json", "auto"]): Read from format
          (defaults to 'auto' based on `log_file` extension)

    Returns:
       EvalLog object read from file.
    """
    from inspect_ai.log._recorders import recorder_type_for_format, recorder_type_for_location
    # resolve to file path
    log_file = log_file if isinstance(log_file, str) else log_file.name
    logger.debug(f"Reading eval log from {log_file}")

    # get recorder type
    if format == "auto":
        recorder_type = recorder_type_for_location(log_file)
    else:
        recorder_type = recorder_type_for_format(format)
    log = await recorder_type.read_log(log_file, header_only)

    # resolve attachement if requested
    if resolve_attachments and log.samples:
        log.samples = [resolve_sample_attachments(sample) for sample in log.samples]

    # provide sample ids if they aren't there
    if log.eval.dataset.sample_ids is None and log.samples is not None:
        sample_ids: dict[str | int, None] = {}
        for sample in log.samples:
            if sample.id not in sample_ids:
                sample_ids[sample.id] = None
        log.eval.dataset.sample_ids = list(sample_ids.keys())

    logger.debug(f"Completed reading eval log from {log_file}")

    return log


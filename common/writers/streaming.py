from __future__ import annotations


def apply_stream_options(
    writer,
    checkpoint=None,
    output_mode=None,
    query_name=None,
    trigger=None,
):
    """
    Apply canonical Structured Streaming configuration.

    Different sinks may have different requirements, so all
    configuration parameters are optional.
    """

    if output_mode:
        writer = writer.outputMode(output_mode)

    if checkpoint:
        writer = writer.option(
            "checkpointLocation",
            checkpoint,
        )

    if query_name:
        writer = writer.queryName(
            query_name,
        )

    if trigger:
        if trigger.get("processingTime"):
            writer = writer.trigger(
                processingTime=trigger["processingTime"],
            )
        elif trigger.get("availableNow"):
            writer = writer.trigger(
                availableNow=True,
            )
        elif trigger.get("once"):
            writer = writer.trigger(
                once=True,
            )

    return writer

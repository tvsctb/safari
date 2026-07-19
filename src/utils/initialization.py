import fnmatch

import torch


def transplant_initialization(
    model,
    donor,
    patterns,
    embedding_rows=None,
):
    """Copy selected donor parameters, optionally slicing embedding rows."""
    donor_parameters = dict(donor.named_parameters())
    copied = []
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns):
                continue
            if name == "embedding.weight" and embedding_rows not in (None, "all"):
                if embedding_rows == "vocabulary":
                    row_slice = slice(0, model.vocab_size)
                elif embedding_rows == "boundary":
                    row_slice = slice(model.vocab_size, None)
                else:
                    raise ValueError(
                        "initialization_transplant_embedding_rows must be "
                        "one of: all, vocabulary, boundary"
                    )
                parameter[row_slice].copy_(donor_parameters[name][row_slice])
                copied.append(f"{name}[{embedding_rows}]")
            else:
                parameter.copy_(donor_parameters[name])
                copied.append(name)
    if not copied:
        raise ValueError(
            "initialization_transplant_patterns did not match any parameters"
        )
    return copied

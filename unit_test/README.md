# Unit Tests for BERT, GPT Implementations

This directory contains unit tests designed to validate the correct behavior of the BERT, GPT, and T5 implementations, focusing on the most common issues related to tensor manipulations. Specifically, the tests aim to verify that the implemented classes handle tensors with correct:

- **Shape:** Ensuring that the output tensors have the expected dimensions.
- **Dtype (Data Type):** Confirming that the data types of the tensors are as expected.
- **Device:** Checking that the tensors are allocated on the correct device (CPU or GPU).

## Test Files

- **`bert_test.py`:** This file contains the unit tests for the BERT package. It includes tests that validate the shape, dtype, and device of the tensors output by each BERT component.
- **`gpt_test.py`:** This file contains the unit tests for the GPT package. Similar to `bert_test.py`, it verifies the shape, dtype, and device of the tensors output by each GPT component.

## Test Coverage

The following table provides an overview of the aspects tested for each model:

| Model | Shape | Dtype | Device |
|-------|-------|-------|--------|
| BERT  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| GPT   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |

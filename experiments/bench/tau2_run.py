"""tau2 CLI with the NL-assertion judge pointed at LLM_* (.env) instead of tau2's hard-coded gpt-4.1.

112 of the 114 retail tasks are graded on DB + NL_ASSERTION; without an OpenAI key every one of
their evaluations failed as an infrastructure error (34/114 in one trial, 151/456 in baseline A).
The judge is the same for every arm.   .bench/tau2/bin/python experiments/bench/tau2_run.py run ...
"""
import os
import sys

import tau2.evaluator.evaluator_nl_assertions as nl

nl.DEFAULT_LLM_NL_ASSERTIONS = 'openai/' + os.environ['LLM_MODEL']
nl.DEFAULT_LLM_NL_ASSERTIONS_ARGS = {'temperature': 0.0, 'api_base': os.environ['LLM_BASE_URL'],
                                     'api_key': os.environ['LLM_API_KEY']}

from tau2.cli import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main())

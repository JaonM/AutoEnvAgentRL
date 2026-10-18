"""The teacher evaluation must preserve API reasoning without altering task messages."""
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'experiments/bench'))
from run_deepseek_control import with_reasoning


class DeepSeekAdapterTests(unittest.TestCase):
    def test_reasoning_survives_tool_continuation_without_changing_other_fields(self):
        messages=[SimpleNamespace(raw_data=None),SimpleNamespace(raw_data={
            'choices':[{'message':{'reasoning_content':'synthetic test reasoning'}}]}),SimpleNamespace(raw_data=None)]
        wire=[{'role':'user','content':'test'},
              {'role':'assistant','content':None,'tool_calls':[{'id':'a','function':{'name':'lookup','arguments':'{}'}}]},
              {'role':'tool','tool_call_id':'a','content':'ok'}]
        actual=with_reasoning(messages,lambda _:copy.deepcopy(wire))
        self.assertEqual(actual[1].pop('reasoning_content'),'synthetic test reasoning')
        self.assertEqual(actual,wire)

    def test_absent_reasoning_is_not_fabricated_or_copied_into_user_messages(self):
        raw={'choices':[{'message':{'reasoning_content':'synthetic test reasoning'}}]}
        messages=[SimpleNamespace(raw_data=raw),SimpleNamespace(raw_data=None)]
        wire=[{'role':'user','content':'test'},{'role':'assistant','content':'reply','tool_calls':None}]
        self.assertEqual(with_reasoning(messages,lambda _:copy.deepcopy(wire)),wire)


if __name__=='__main__':unittest.main()

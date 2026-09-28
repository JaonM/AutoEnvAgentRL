import unittest

from env_factory.tasks.task_similarity import task_partition_isolation


class TaskPartitionIsolationTest(unittest.TestCase):
    def test_numeric_and_punctuation_variants_cannot_cross_partitions(self):
        report = task_partition_isolation({
            "development": [{"task": "查询客户 A 的订单 100，并更新配送地址。"}],
            "holdout:1": [{"task": "查询客户A的订单200并更新配送地址!"}],
            "holdout:2": [{"task": "分析仓库库存并生成补货建议"}],
        })
        self.assertFalse(report["isolated"])
        self.assertEqual(report["cross_partition_family_count"], 1)
        overlap = report["cross_partition_families"][0]
        self.assertEqual(overlap["partitions"], ["development", "holdout:1"])
        self.assertNotIn("查询", str(report))

    def test_distinct_task_families_are_isolated(self):
        report = task_partition_isolation({
            "development": [{"task": "核对发票税额并记录差异"}],
            "holdout:1": [{"task": "规划会议室预订并通知参会者"}],
            "holdout:2": [{"task": "查询库存后创建补货申请"}],
        })
        self.assertTrue(report["isolated"])
        self.assertEqual(report["family_count"], 3)

    def test_empty_corpus_is_not_positive_evidence(self):
        report = task_partition_isolation({"holdout:1": []})
        self.assertFalse(report["isolated"])


if __name__ == "__main__":
    unittest.main()

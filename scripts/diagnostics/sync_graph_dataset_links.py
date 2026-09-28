"""Verify reviewed raw sources and synchronize Scene -> Dataset graph relations."""

from env_factory.graph.dataset_planner import reviewed_links, sync_reviewed_links
from env_factory.graph.graph_builder import Neo4jGraphStore


if __name__ == "__main__":
    with Neo4jGraphStore() as store:
        count = sync_reviewed_links(store)
        persisted = {link for platform in ("kaggle", "data_gov_hk")
                     for _, link in store.supported_dataset_links(platform)}
        if not set(reviewed_links()).issubset(persisted):
            raise RuntimeError("Neo4j did not persist every reviewed Scene -> Dataset relation")
    print(f"Synchronized and verified {count} reviewed Scene -> Dataset links")

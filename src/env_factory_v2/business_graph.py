"""Scoped Neo4j business evidence; existing scene graph is read-only."""
from __future__ import annotations

import hashlib
import json

from .expressions import canonical
from .contract import require


def install(driver, namespace, documents, *, database='neo4j'):
    """Append a frozen synthetic corpus, never overwrite other graph namespaces."""
    records = []
    for document in documents:
        identity = namespace + ':' + document['document_id']
        business = document['business']
        records.append({'id': identity, 'namespace': namespace, 'document_id': document['document_id'],
            'title': document['title'], 'policy': business['policy'],
            'sha256': hashlib.sha256(canonical(business).encode()).hexdigest(),
            'tables': [{'id': identity + ':' + table['id'], 'name': table['id'],
                'columns': list(table.get('generator', {}).get('columns', {})) or
                    list(table.get('rows', [{}])[0]), 'provenance': business['provenance']}
                for table in business['tables']], 'provenance': business['provenance']})
    with driver.session(database=database) as session:
        existing = session.run('MATCH (d:EFV2BusinessDocument {namespace:$ns}) RETURN d.document_id AS id,d.sha256 AS sha', ns=namespace)
        previous = {r['id']: r['sha'] for r in existing}
        require(not previous or previous == {r['document_id']: r['sha256'] for r in records}, 'graph namespace already has different evidence')
        session.execute_write(_write, records)
    return records


def _write(tx, records):
    tx.run('''UNWIND $records AS row
        MERGE (d:EFV2BusinessDocument {id:row.id})
        SET d.namespace=row.namespace,d.document_id=row.document_id,d.title=row.title,
            d.policy=row.policy,d.sha256=row.sha256,d.provenance=row.provenance
        MERGE (r:EFV2BusinessRule {id:row.id + ':policy'})
        SET r.text=row.policy,r.provenance=row.provenance
        MERGE (d)-[:GOVERNED_BY]->(r)
        WITH d,row UNWIND row.tables AS tab
        MERGE (t:EFV2BusinessObject {id:tab.id})
        SET t.name=tab.name,t.columns=tab.columns,t.provenance=tab.provenance
        MERGE (d)-[:USES_OBJECT]->(t)''', records=records).consume()


def retrieve(driver, namespace, *, database='neo4j'):
    with driver.session(database=database) as session:
        rows = [dict(r) for r in session.run('''MATCH (d:EFV2BusinessDocument {namespace:$ns})-[:GOVERNED_BY]->(rule)
            OPTIONAL MATCH (d)-[:USES_OBJECT]->(obj)
            RETURN d.document_id AS document_id,d.title AS title,d.sha256 AS sha256,
              rule.text AS policy_summary,collect({name:obj.name,columns:obj.columns}) AS objects
            ORDER BY document_id''', ns=namespace)]
    require(len(rows) == 20, 'business graph corpus must contain all 20 documents')
    return {'backend': 'neo4j', 'namespace': namespace, 'documents': rows,
            'provenance': 'synthetic business dossiers imported as scoped graph objects/rules; not production data'}

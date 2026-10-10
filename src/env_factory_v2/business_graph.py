"""Scoped Neo4j business evidence; existing scene graph is read-only."""
from __future__ import annotations

import copy
import hashlib
import json

from .expressions import canonical
from .contract import require


def validate_relations(business):
    """Validate declared, publicly sourced links against complete fixture rows."""
    from .fixtures import materialize
    from .contract import fields
    tables = copy.deepcopy(business['tables'])
    for table in tables:
        if 'generator' in table and 'schema' not in table:
            columns = table['generator']['columns']
            table['schema'] = {'required': list(columns), 'properties': {key: {} for key in columns}}
    materialize(tables)
    index = {table['id']: table for table in tables}
    result, identities = [], set()
    for relation in business.get('relations', []):
        fields(relation, ['id', 'source_table', 'source_column', 'target_table',
                          'target_column', 'policy_text', 'nullable'])
        require(isinstance(relation['id'], str) and relation['id'] and relation['id'] not in identities,
                'unique business relation identity required')
        identities.add(relation['id'])
        require(isinstance(relation['policy_text'], str) and relation['policy_text'].strip()
                and relation['policy_text'] in business['policy'], 'relation public policy provenance required')
        require(type(relation['nullable']) is bool, 'relation nullable must be boolean')
        require(relation['source_table'] in index and relation['target_table'] in index, 'unknown relation table')
        source, target = index[relation['source_table']], index[relation['target_table']]
        def columns(table):
            return set(table.get('schema', {}).get('properties', {})) or set(table.get('rows', [{}])[0])
        require(relation['source_column'] in columns(source) and relation['target_column'] in columns(target),
                'unknown relation column')
        keys = [canonical(row[relation['target_column']]) for row in target['rows']]
        require(len(keys) == len(set(keys)), 'relation target must identify unique records')
        for row in source['rows']:
            value = row[relation['source_column']]
            require((value is None and relation['nullable']) or (value is not None and canonical(value) in keys),
                    'business relation points to missing record')
        result.append(copy.deepcopy(relation))
    return result


def install(driver, namespace, documents, *, database='neo4j'):
    """Append a frozen synthetic corpus, never overwrite other graph namespaces."""
    records = []
    for document in documents:
        identity = namespace + ':' + document['document_id']
        business = document['business']
        records.append({'id': identity, 'namespace': namespace, 'document_id': document['document_id'],
            'title': document['title'], 'policy': business['policy'],
            'sha256': hashlib.sha256(canonical(business).encode()).hexdigest(),
            'relations': validate_relations(business),
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
    tx.run('''UNWIND $records AS row UNWIND row.relations AS rel
        MATCH (source:EFV2BusinessObject {id:row.id + ':' + rel.source_table})
        MATCH (target:EFV2BusinessObject {id:row.id + ':' + rel.target_table})
        MERGE (source)-[r:BUSINESS_REFERENCE {id:row.id + ':' + rel.id}]->(target)
        SET r.name=rel.id,r.source_column=rel.source_column,r.target_column=rel.target_column,
            r.policy_text=rel.policy_text,r.nullable=rel.nullable''', records=records).consume()


def retrieve(driver, namespace, *, database='neo4j'):
    with driver.session(database=database) as session:
        rows = [dict(r) for r in session.run('''MATCH (d:EFV2BusinessDocument {namespace:$ns})-[:GOVERNED_BY]->(rule)
            OPTIONAL MATCH (d)-[:USES_OBJECT]->(obj)
            RETURN d.document_id AS document_id,d.title AS title,d.sha256 AS sha256,
              rule.text AS policy_summary,collect({name:obj.name,columns:obj.columns}) AS objects
            ORDER BY document_id''', ns=namespace)]
        relations = [dict(r) for r in session.run('''MATCH (d:EFV2BusinessDocument {namespace:$ns})-[:USES_OBJECT]->(source)
            MATCH (source)-[rel:BUSINESS_REFERENCE]->(target)
            RETURN d.document_id AS document_id,rel.name AS id,source.name AS source_table,
                rel.source_column AS source_column,target.name AS target_table,
                rel.target_column AS target_column,rel.policy_text AS policy_text,rel.nullable AS nullable
            ORDER BY document_id,id''', ns=namespace)]
        for document in rows:
            document['relations'] = [{key: value for key, value in relation.items() if key != 'document_id'}
                                     for relation in relations if relation['document_id'] == document['document_id']]
    require(len(rows) == 20, 'business graph corpus must contain all 20 documents')
    return {'backend': 'neo4j', 'namespace': namespace, 'documents': rows,
            'provenance': 'synthetic business dossiers imported as scoped graph objects/rules; not production data'}

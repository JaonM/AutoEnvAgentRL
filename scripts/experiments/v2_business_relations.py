"""Operator-declared relationships for new synthetic dossiers, never inferred."""
import copy
from env_factory_v2.requirements.business_graph import validate_relations

# Explicit business-family declarations. No column-name inference or old asset edits.
RELATIONS = {
    'shipment': [('orders', 'shipment_id', 'shipments', False)],
    'warranty': [('devices', 'model_id', 'models', False)],
    'employees': [('employees', 'department_id', 'departments', False)],
    'cancel': [('reservations', 'slot_id', 'slots', False)],
    'chain': [('applications', 'entry_id', 'nodes', False), ('nodes', 'next_id', 'nodes', True),
              ('nodes', 'owner_id', 'people', False)],
    'expenses': [('expenses', 'project_id', 'projects', False)],
    'inventory': [('inventory', 'warehouse_id', 'warehouses', False)],
    'replace': [('assignments', 'device_id', 'equipment', False), ('maintenance', 'device_id', 'equipment', False)],
    'transfer': [('transfers', 'source_id', 'stocks', False), ('transfers', 'target_id', 'stocks', False)],
    'enroll': [('activities', 'registration_id', 'registrations', False), ('registrations', 'activity_id', 'activities', False)],
    'correction': [('bookings', 'venue_id', 'venues', True)],
    'sequential': [('features', 'account_id', 'accounts', False), ('features', 'log_id', 'logs', False),
                   ('logs', 'feature_id', 'features', False)],
}


def with_business_relations(entries):
    result = copy.deepcopy(entries)
    for case in result:
        business = case['business']
        relations = []
        for source, column, target, nullable in RELATIONS.get(case['acceptance']['kind'], []):
            meaning = f'业务关联：{source}.{column} 指向 {target}.id' + ('，允许未关联的空值。' if nullable else '，必须关联已有记录。')
            business['policy'] += '\n' + meaning
            relations.append({'id': source + '_' + column, 'source_table': source,
                'source_column': column, 'target_table': target, 'target_column': 'id',
                'nullable': nullable, 'policy_text': meaning})
        business['relations'] = relations
        validate_relations(business)
    return result

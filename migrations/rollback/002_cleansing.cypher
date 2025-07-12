MATCH ()-[r]-() DELETE r;

// Remove all nodes
MATCH (n) DELETE n;

// Drop all indexes (comprehensive list)
DROP INDEX entity_context_fulltext_idx IF EXISTS;
DROP INDEX entity_aliases_fulltext_idx IF EXISTS;
DROP INDEX entity_name_fulltext_idx IF EXISTS;
DROP INDEX gpe_fulltext_idx IF EXISTS;
DROP INDEX law_fulltext_idx IF EXISTS;
DROP INDEX temporal_duration_relationship_idx IF EXISTS;
DROP INDEX temporal_start_relationship_idx IF EXISTS;
DROP INDEX legal_implements_relationship_idx IF EXISTS;
DROP INDEX legal_amends_relationship_idx IF EXISTS;
DROP INDEX admin_contains_relationship_idx IF EXISTS;
DROP INDEX entity_topic_relationship_idx IF EXISTS;
DROP INDEX chunk_entity_relationship_idx IF EXISTS;
DROP INDEX chunk_doc_relationship_idx IF EXISTS;
DROP INDEX language_code_idx IF EXISTS;
DROP INDEX facility_normalized_name_idx IF EXISTS;
DROP INDEX product_normalized_name_idx IF EXISTS;
DROP INDEX political_org_normalized_name_idx IF EXISTS;
DROP INDEX organization_normalized_name_idx IF EXISTS;
DROP INDEX person_normalized_name_idx IF EXISTS;
DROP INDEX money_value_currency_idx IF EXISTS;
DROP INDEX percentage_value_idx IF EXISTS;
DROP INDEX quantity_value_unit_idx IF EXISTS;
DROP INDEX time_value_confidence_idx IF EXISTS;
DROP INDEX date_value_confidence_idx IF EXISTS;
DROP INDEX law_type_confidence_idx IF EXISTS;
DROP INDEX law_type_identifier_idx IF EXISTS;
DROP INDEX gpe_admin_confidence_idx IF EXISTS;
DROP INDEX gpe_admin_hierarchy_idx IF EXISTS;
DROP INDEX entity_normalized_name_idx IF EXISTS;
DROP INDEX entity_type_frequency_idx IF EXISTS;
DROP INDEX entity_type_confidence_idx IF EXISTS;
DROP INDEX time_value_idx IF EXISTS;
DROP INDEX date_value_idx IF EXISTS;
DROP INDEX law_type_idx IF EXISTS;
DROP INDEX law_identifier_idx IF EXISTS;
DROP INDEX gpe_admin_name_idx IF EXISTS;
DROP INDEX gpe_admin_level_idx IF EXISTS;
DROP INDEX entity_confidence_idx IF EXISTS;
DROP INDEX entity_type_name_idx IF EXISTS;
DROP INDEX entity_type_idx IF EXISTS;
DROP INDEX entity_name_idx IF EXISTS;
DROP INDEX chunk_created_at_idx IF EXISTS;
DROP INDEX chunk_content_hash_idx IF EXISTS;
DROP INDEX chunk_doc_id_idx IF EXISTS;
DROP INDEX document_file_type_idx IF EXISTS;
DROP INDEX document_created_at_idx IF EXISTS;
DROP INDEX document_file_path_idx IF EXISTS;
DROP INDEX rag_response_timestamp_idx IF EXISTS;
DROP INDEX rag_query_timestamp_idx IF EXISTS;
DROP INDEX rag_session_timestamp_idx IF EXISTS;

// Drop all constraints (comprehensive list)
DROP CONSTRAINT rag_response_id IF EXISTS;
DROP CONSTRAINT rag_query_id IF EXISTS;
DROP CONSTRAINT rag_session_id IF EXISTS;
DROP CONSTRAINT language_code IF EXISTS;
DROP CONSTRAINT artwork_name IF EXISTS;
DROP CONSTRAINT religion_name IF EXISTS;
DROP CONSTRAINT product_name IF EXISTS;
DROP CONSTRAINT time_value IF EXISTS;
DROP CONSTRAINT date_value IF EXISTS;
DROP CONSTRAINT event_name IF EXISTS;
DROP CONSTRAINT law_identifier IF EXISTS;
DROP CONSTRAINT facility_name IF EXISTS;
DROP CONSTRAINT location_name IF EXISTS;
DROP CONSTRAINT gpe_name_admin IF EXISTS;
DROP CONSTRAINT political_org_name IF EXISTS;
DROP CONSTRAINT organization_name IF EXISTS;
DROP CONSTRAINT person_name IF EXISTS;
DROP CONSTRAINT entity_name_type IF EXISTS;
DROP CONSTRAINT topic_name IF EXISTS;
DROP CONSTRAINT chunk_id IF EXISTS;
DROP CONSTRAINT document_id IF EXISTS;
DROP CONSTRAINT migration_id IF EXISTS;

// Remove all EntityType nodes created in this migration
MATCH (n)
WHERE n.code IN [
    'PER', 'ORG', 'NOR', 'GPE', 'LOC', 'FAC', 'LAW', 'EVT', 
    'DAT', 'TIM', 'CRD', 'ORD', 'QTY', 'PRC', 'MON', 'PRD', 
    'REG', 'WOA', 'LAN', 'O'
]
DELETE n;

RETURN {
    status: 'completed',
    message: 'Neo4j database completely cleaned - all data, indexes, and constraints removed',
    timestamp: datetime()
} AS result;

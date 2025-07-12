// ============================================================================
// LocalRAG Neo4j Foundation Migration Rollback (Complete)
// Rollback for: 006_foundation_rebase_indexes_only
// Purpose: Complete rollback of all indexes and configuration nodes
// ============================================================================

// ============================================================================
// 1. DROP FULLTEXT INDEXES - Search Capabilities
// ============================================================================

DROP INDEX topic_fulltext IF EXISTS;
DROP INDEX document_fulltext IF EXISTS;
DROP INDEX indonesian_evt_fulltext IF EXISTS;
DROP INDEX indonesian_law_fulltext IF EXISTS;
DROP INDEX indonesian_fac_fulltext IF EXISTS;
DROP INDEX indonesian_loc_fulltext IF EXISTS;
DROP INDEX indonesian_gpe_fulltext IF EXISTS;
DROP INDEX indonesian_nor_fulltext IF EXISTS;
DROP INDEX indonesian_org_fulltext IF EXISTS;
DROP INDEX indonesian_per_fulltext IF EXISTS;
DROP INDEX indonesian_prd_fulltext IF EXISTS;
DROP INDEX indonesian_woa_fulltext IF EXISTS;
DROP INDEX indonesian_chunk_fulltext IF EXISTS;

// ============================================================================
// 2. DROP ENTITY LANGUAGE INDEXES - Cross-cutting Concerns  
// ============================================================================

DROP INDEX o_language_idx IF EXISTS;
DROP INDEX lan_language_idx IF EXISTS;
DROP INDEX woa_language_idx IF EXISTS;
DROP INDEX reg_language_idx IF EXISTS;
DROP INDEX prd_language_idx IF EXISTS;
DROP INDEX mon_language_idx IF EXISTS;
DROP INDEX prc_language_idx IF EXISTS;
DROP INDEX qty_language_idx IF EXISTS;
DROP INDEX ord_language_idx IF EXISTS;
DROP INDEX crd_language_idx IF EXISTS;
DROP INDEX tim_language_idx IF EXISTS;
DROP INDEX dat_language_idx IF EXISTS;
DROP INDEX evt_language_idx IF EXISTS;
DROP INDEX law_language_idx IF EXISTS;
DROP INDEX fac_language_idx IF EXISTS;
DROP INDEX loc_language_idx IF EXISTS;
DROP INDEX gpe_language_idx IF EXISTS;
DROP INDEX nor_language_idx IF EXISTS;
DROP INDEX org_language_idx IF EXISTS;
DROP INDEX per_language_idx IF EXISTS;

// ============================================================================
// 3. DROP INDONESIAN NER ENTITY INDEXES (20 Types) - Comprehensive Cleanup
// ============================================================================

// Other entities (O)
DROP INDEX o_confidence_idx IF EXISTS;
DROP INDEX o_normalized_name_idx IF EXISTS;
DROP INDEX o_name_idx IF EXISTS;

// Language entities (LAN)
DROP INDEX lan_confidence_idx IF EXISTS;
DROP INDEX lan_normalized_name_idx IF EXISTS;
DROP INDEX lan_name_idx IF EXISTS;

// Work of art entities (WOA)
DROP INDEX woa_confidence_idx IF EXISTS;
DROP INDEX woa_normalized_name_idx IF EXISTS;
DROP INDEX woa_name_idx IF EXISTS;

// Religion entities (REG)
DROP INDEX reg_confidence_idx IF EXISTS;
DROP INDEX reg_normalized_name_idx IF EXISTS;
DROP INDEX reg_name_idx IF EXISTS;

// Product entities (PRD)
DROP INDEX prd_confidence_idx IF EXISTS;
DROP INDEX prd_normalized_name_idx IF EXISTS;
DROP INDEX prd_name_idx IF EXISTS;

// Money entities (MON)
DROP INDEX mon_confidence_idx IF EXISTS;
DROP INDEX mon_normalized_name_idx IF EXISTS;
DROP INDEX mon_name_idx IF EXISTS;

// Percent entities (PRC)
DROP INDEX prc_confidence_idx IF EXISTS;
DROP INDEX prc_normalized_name_idx IF EXISTS;
DROP INDEX prc_name_idx IF EXISTS;

// Quantity entities (QTY)
DROP INDEX qty_confidence_idx IF EXISTS;
DROP INDEX qty_normalized_name_idx IF EXISTS;
DROP INDEX qty_name_idx IF EXISTS;

// Ordinal entities (ORD)
DROP INDEX ord_confidence_idx IF EXISTS;
DROP INDEX ord_normalized_name_idx IF EXISTS;
DROP INDEX ord_name_idx IF EXISTS;

// Cardinal entities (CRD)
DROP INDEX crd_confidence_idx IF EXISTS;
DROP INDEX crd_normalized_name_idx IF EXISTS;
DROP INDEX crd_name_idx IF EXISTS;

// Time entities (TIM)
DROP INDEX tim_confidence_idx IF EXISTS;
DROP INDEX tim_normalized_name_idx IF EXISTS;
DROP INDEX tim_name_idx IF EXISTS;

// Date entities (DAT)
DROP INDEX dat_confidence_idx IF EXISTS;
DROP INDEX dat_normalized_name_idx IF EXISTS;
DROP INDEX dat_name_idx IF EXISTS;

// Event entities (EVT)
DROP INDEX evt_confidence_idx IF EXISTS;
DROP INDEX evt_normalized_name_idx IF EXISTS;
DROP INDEX evt_name_idx IF EXISTS;

// Law entities (LAW)
DROP INDEX law_confidence_idx IF EXISTS;
DROP INDEX law_normalized_name_idx IF EXISTS;
DROP INDEX law_name_idx IF EXISTS;

// Facility entities (FAC)
DROP INDEX fac_confidence_idx IF EXISTS;
DROP INDEX fac_normalized_name_idx IF EXISTS;
DROP INDEX fac_name_idx IF EXISTS;

// Location entities (LOC)
DROP INDEX loc_confidence_idx IF EXISTS;
DROP INDEX loc_normalized_name_idx IF EXISTS;
DROP INDEX loc_name_idx IF EXISTS;

// Geopolitical entities (GPE)
DROP INDEX gpe_confidence_idx IF EXISTS;
DROP INDEX gpe_normalized_name_idx IF EXISTS;
DROP INDEX gpe_name_idx IF EXISTS;

// Political organization entities (NOR)
DROP INDEX nor_confidence_idx IF EXISTS;
DROP INDEX nor_normalized_name_idx IF EXISTS;
DROP INDEX nor_name_idx IF EXISTS;

// Organization entities (ORG)
DROP INDEX org_confidence_idx IF EXISTS;
DROP INDEX org_normalized_name_idx IF EXISTS;
DROP INDEX org_name_idx IF EXISTS;

// Person entities (PER)
DROP INDEX per_confidence_idx IF EXISTS;
DROP INDEX per_normalized_name_idx IF EXISTS;
DROP INDEX per_name_idx IF EXISTS;

// ============================================================================
// 4. DROP TOPIC INDEXES - Topic Management
// ============================================================================

DROP INDEX topic_created_at_idx IF EXISTS;
DROP INDEX topic_frequency_idx IF EXISTS;
DROP INDEX topic_name_idx IF EXISTS;

// ============================================================================
// 5. DROP CHUNK INDEXES - Chunk Operations and Relationships
// ============================================================================

DROP INDEX chunk_structure_type_idx IF EXISTS;
DROP INDEX chunk_index_idx IF EXISTS;
DROP INDEX chunk_name_topic_idx IF EXISTS;
DROP INDEX chunk_semantic_class_idx IF EXISTS;
DROP INDEX chunk_created_at_idx IF EXISTS;
DROP INDEX chunk_content_hash_idx IF EXISTS;
DROP INDEX chunk_doc_id_idx IF EXISTS;

// ============================================================================
// 6. DROP DOCUMENT INDEXES - Core Document Operations
// ============================================================================

DROP INDEX document_hash_idx IF EXISTS;
DROP INDEX document_status_idx IF EXISTS;
DROP INDEX document_file_type_idx IF EXISTS;
DROP INDEX document_created_at_idx IF EXISTS;
DROP INDEX document_file_path_idx IF EXISTS;

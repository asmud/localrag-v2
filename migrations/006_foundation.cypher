// ============================================================================
// 1. DOCUMENT INDEXES - Core Document Operations
// ============================================================================

CREATE INDEX document_file_path_idx IF NOT EXISTS FOR (d:Document) ON (d.file_path);
CREATE INDEX document_created_at_idx IF NOT EXISTS FOR (d:Document) ON (d.created_at);
CREATE INDEX document_file_type_idx IF NOT EXISTS FOR (d:Document) ON (d.file_type);
CREATE INDEX document_status_idx IF NOT EXISTS FOR (d:Document) ON (d.status);
CREATE INDEX document_hash_idx IF NOT EXISTS FOR (d:Document) ON (d.content_hash);

// ============================================================================
// 2. CHUNK INDEXES - Chunk Operations and Relationships  
// ============================================================================

CREATE INDEX chunk_doc_id_idx IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id);
CREATE INDEX chunk_content_hash_idx IF NOT EXISTS FOR (c:Chunk) ON (c.content_hash);
CREATE INDEX chunk_created_at_idx IF NOT EXISTS FOR (c:Chunk) ON (c.created_at);
CREATE INDEX chunk_semantic_class_idx IF NOT EXISTS FOR (c:Chunk) ON (c.semantic_class);
CREATE INDEX chunk_name_topic_idx IF NOT EXISTS FOR (c:Chunk) ON (c.name, c.topic);
CREATE INDEX chunk_index_idx IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_index);
CREATE INDEX chunk_structure_type_idx IF NOT EXISTS FOR (c:Chunk) ON (c.structure_type);

// ============================================================================
// 3. TOPIC INDEXES - Topic Management
// ============================================================================

CREATE INDEX topic_name_idx IF NOT EXISTS FOR (t:Topic) ON (t.name);
CREATE INDEX topic_frequency_idx IF NOT EXISTS FOR (t:Topic) ON (t.frequency);
CREATE INDEX topic_created_at_idx IF NOT EXISTS FOR (t:Topic) ON (t.created_at);

// ============================================================================
// 4. INDONESIAN NER ENTITY INDEXES (20 Types) - Performance Critical
// ============================================================================

// Person entities (PER)
CREATE INDEX per_name_idx IF NOT EXISTS FOR (n:PER) ON (n.name);
CREATE INDEX per_normalized_name_idx IF NOT EXISTS FOR (n:PER) ON (n.normalized_name);
CREATE INDEX per_confidence_idx IF NOT EXISTS FOR (n:PER) ON (n.confidence);

// Organization entities (ORG)
CREATE INDEX org_name_idx IF NOT EXISTS FOR (n:ORG) ON (n.name);
CREATE INDEX org_normalized_name_idx IF NOT EXISTS FOR (n:ORG) ON (n.normalized_name);
CREATE INDEX org_confidence_idx IF NOT EXISTS FOR (n:ORG) ON (n.confidence);

// Political organization entities (NOR)
CREATE INDEX nor_name_idx IF NOT EXISTS FOR (n:NOR) ON (n.name);
CREATE INDEX nor_normalized_name_idx IF NOT EXISTS FOR (n:NOR) ON (n.normalized_name);
CREATE INDEX nor_confidence_idx IF NOT EXISTS FOR (n:NOR) ON (n.confidence);

// Geopolitical entities (GPE)
CREATE INDEX gpe_name_idx IF NOT EXISTS FOR (n:GPE) ON (n.name);
CREATE INDEX gpe_normalized_name_idx IF NOT EXISTS FOR (n:GPE) ON (n.normalized_name);
CREATE INDEX gpe_confidence_idx IF NOT EXISTS FOR (n:GPE) ON (n.confidence);

// Location entities (LOC)
CREATE INDEX loc_name_idx IF NOT EXISTS FOR (n:LOC) ON (n.name);
CREATE INDEX loc_normalized_name_idx IF NOT EXISTS FOR (n:LOC) ON (n.normalized_name);
CREATE INDEX loc_confidence_idx IF NOT EXISTS FOR (n:LOC) ON (n.confidence);

// Facility entities (FAC)
CREATE INDEX fac_name_idx IF NOT EXISTS FOR (n:FAC) ON (n.name);
CREATE INDEX fac_normalized_name_idx IF NOT EXISTS FOR (n:FAC) ON (n.normalized_name);
CREATE INDEX fac_confidence_idx IF NOT EXISTS FOR (n:FAC) ON (n.confidence);

// Law entities (LAW)
CREATE INDEX law_name_idx IF NOT EXISTS FOR (n:LAW) ON (n.name);
CREATE INDEX law_normalized_name_idx IF NOT EXISTS FOR (n:LAW) ON (n.normalized_name);
CREATE INDEX law_confidence_idx IF NOT EXISTS FOR (n:LAW) ON (n.confidence);

// Event entities (EVT)
CREATE INDEX evt_name_idx IF NOT EXISTS FOR (n:EVT) ON (n.name);
CREATE INDEX evt_normalized_name_idx IF NOT EXISTS FOR (n:EVT) ON (n.normalized_name);
CREATE INDEX evt_confidence_idx IF NOT EXISTS FOR (n:EVT) ON (n.confidence);

// Date entities (DAT)
CREATE INDEX dat_name_idx IF NOT EXISTS FOR (n:DAT) ON (n.name);
CREATE INDEX dat_normalized_name_idx IF NOT EXISTS FOR (n:DAT) ON (n.normalized_name);
CREATE INDEX dat_confidence_idx IF NOT EXISTS FOR (n:DAT) ON (n.confidence);

// Time entities (TIM)
CREATE INDEX tim_name_idx IF NOT EXISTS FOR (n:TIM) ON (n.name);
CREATE INDEX tim_normalized_name_idx IF NOT EXISTS FOR (n:TIM) ON (n.normalized_name);
CREATE INDEX tim_confidence_idx IF NOT EXISTS FOR (n:TIM) ON (n.confidence);

// Cardinal entities (CRD)
CREATE INDEX crd_name_idx IF NOT EXISTS FOR (n:CRD) ON (n.name);
CREATE INDEX crd_normalized_name_idx IF NOT EXISTS FOR (n:CRD) ON (n.normalized_name);
CREATE INDEX crd_confidence_idx IF NOT EXISTS FOR (n:CRD) ON (n.confidence);

// Ordinal entities (ORD)
CREATE INDEX ord_name_idx IF NOT EXISTS FOR (n:ORD) ON (n.name);
CREATE INDEX ord_normalized_name_idx IF NOT EXISTS FOR (n:ORD) ON (n.normalized_name);
CREATE INDEX ord_confidence_idx IF NOT EXISTS FOR (n:ORD) ON (n.confidence);

// Quantity entities (QTY)
CREATE INDEX qty_name_idx IF NOT EXISTS FOR (n:QTY) ON (n.name);
CREATE INDEX qty_normalized_name_idx IF NOT EXISTS FOR (n:QTY) ON (n.normalized_name);
CREATE INDEX qty_confidence_idx IF NOT EXISTS FOR (n:QTY) ON (n.confidence);

// Percent entities (PRC)
CREATE INDEX prc_name_idx IF NOT EXISTS FOR (n:PRC) ON (n.name);
CREATE INDEX prc_normalized_name_idx IF NOT EXISTS FOR (n:PRC) ON (n.normalized_name);
CREATE INDEX prc_confidence_idx IF NOT EXISTS FOR (n:PRC) ON (n.confidence);

// Money entities (MON)
CREATE INDEX mon_name_idx IF NOT EXISTS FOR (n:MON) ON (n.name);
CREATE INDEX mon_normalized_name_idx IF NOT EXISTS FOR (n:MON) ON (n.normalized_name);
CREATE INDEX mon_confidence_idx IF NOT EXISTS FOR (n:MON) ON (n.confidence);

// Product entities (PRD)
CREATE INDEX prd_name_idx IF NOT EXISTS FOR (n:PRD) ON (n.name);
CREATE INDEX prd_normalized_name_idx IF NOT EXISTS FOR (n:PRD) ON (n.normalized_name);
CREATE INDEX prd_confidence_idx IF NOT EXISTS FOR (n:PRD) ON (n.confidence);

// Religion entities (REG)
CREATE INDEX reg_name_idx IF NOT EXISTS FOR (n:REG) ON (n.name);
CREATE INDEX reg_normalized_name_idx IF NOT EXISTS FOR (n:REG) ON (n.normalized_name);
CREATE INDEX reg_confidence_idx IF NOT EXISTS FOR (n:REG) ON (n.confidence);

// Work of art entities (WOA)
CREATE INDEX woa_name_idx IF NOT EXISTS FOR (n:WOA) ON (n.name);
CREATE INDEX woa_normalized_name_idx IF NOT EXISTS FOR (n:WOA) ON (n.normalized_name);
CREATE INDEX woa_confidence_idx IF NOT EXISTS FOR (n:WOA) ON (n.confidence);

// Language entities (LAN)
CREATE INDEX lan_name_idx IF NOT EXISTS FOR (n:LAN) ON (n.name);
CREATE INDEX lan_normalized_name_idx IF NOT EXISTS FOR (n:LAN) ON (n.normalized_name);
CREATE INDEX lan_confidence_idx IF NOT EXISTS FOR (n:LAN) ON (n.confidence);

// Other entities (O)
CREATE INDEX o_name_idx IF NOT EXISTS FOR (n:O) ON (n.name);
CREATE INDEX o_normalized_name_idx IF NOT EXISTS FOR (n:O) ON (n.normalized_name);
CREATE INDEX o_confidence_idx IF NOT EXISTS FOR (n:O) ON (n.confidence);

// ============================================================================
// 5. ENTITY LANGUAGE AND TYPE INDEXES - Cross-cutting Concerns
// ============================================================================

// Language-based indexes for all entity types
CREATE INDEX per_language_idx IF NOT EXISTS FOR (n:PER) ON (n.language);
CREATE INDEX org_language_idx IF NOT EXISTS FOR (n:ORG) ON (n.language);
CREATE INDEX nor_language_idx IF NOT EXISTS FOR (n:NOR) ON (n.language);
CREATE INDEX gpe_language_idx IF NOT EXISTS FOR (n:GPE) ON (n.language);
CREATE INDEX loc_language_idx IF NOT EXISTS FOR (n:LOC) ON (n.language);
CREATE INDEX fac_language_idx IF NOT EXISTS FOR (n:FAC) ON (n.language);
CREATE INDEX law_language_idx IF NOT EXISTS FOR (n:LAW) ON (n.language);
CREATE INDEX evt_language_idx IF NOT EXISTS FOR (n:EVT) ON (n.language);
CREATE INDEX dat_language_idx IF NOT EXISTS FOR (n:DAT) ON (n.language);
CREATE INDEX tim_language_idx IF NOT EXISTS FOR (n:TIM) ON (n.language);
CREATE INDEX crd_language_idx IF NOT EXISTS FOR (n:CRD) ON (n.language);
CREATE INDEX ord_language_idx IF NOT EXISTS FOR (n:ORD) ON (n.language);
CREATE INDEX qty_language_idx IF NOT EXISTS FOR (n:QTY) ON (n.language);
CREATE INDEX prc_language_idx IF NOT EXISTS FOR (n:PRC) ON (n.language);
CREATE INDEX mon_language_idx IF NOT EXISTS FOR (n:MON) ON (n.language);
CREATE INDEX prd_language_idx IF NOT EXISTS FOR (n:PRD) ON (n.language);
CREATE INDEX reg_language_idx IF NOT EXISTS FOR (n:REG) ON (n.language);
CREATE INDEX woa_language_idx IF NOT EXISTS FOR (n:WOA) ON (n.language);
CREATE INDEX lan_language_idx IF NOT EXISTS FOR (n:LAN) ON (n.language);
CREATE INDEX o_language_idx IF NOT EXISTS FOR (n:O) ON (n.language);

// ============================================================================
// 6. FULLTEXT INDEXES - Advanced Search Capabilities
// ============================================================================

// Chunk fulltext search
CREATE FULLTEXT INDEX indonesian_chunk_fulltext IF NOT EXISTS FOR (n:Chunk) ON EACH [n.name, n.summary, n.content];

// Indonesian NER entity fulltext indexes (8 main types for search)
CREATE FULLTEXT INDEX indonesian_per_fulltext IF NOT EXISTS FOR (n:PER) ON EACH [n.name, n.normalized_name];
CREATE FULLTEXT INDEX indonesian_org_fulltext IF NOT EXISTS FOR (n:ORG) ON EACH [n.name, n.normalized_name];
CREATE FULLTEXT INDEX indonesian_nor_fulltext IF NOT EXISTS FOR (n:NOR) ON EACH [n.name, n.normalized_name];
CREATE FULLTEXT INDEX indonesian_gpe_fulltext IF NOT EXISTS FOR (n:GPE) ON EACH [n.name, n.normalized_name];
CREATE FULLTEXT INDEX indonesian_loc_fulltext IF NOT EXISTS FOR (n:LOC) ON EACH [n.name, n.normalized_name];
CREATE FULLTEXT INDEX indonesian_fac_fulltext IF NOT EXISTS FOR (n:FAC) ON EACH [n.name, n.normalized_name];
CREATE FULLTEXT INDEX indonesian_law_fulltext IF NOT EXISTS FOR (n:LAW) ON EACH [n.name, n.normalized_name];
CREATE FULLTEXT INDEX indonesian_evt_fulltext IF NOT EXISTS FOR (n:EVT) ON EACH [n.name, n.normalized_name];
CREATE FULLTEXT INDEX indonesian_prd_fulltext IF NOT EXISTS FOR (n:PRD) ON EACH [n.name, n.normalized_name];
CREATE FULLTEXT INDEX indonesian_woa_fulltext IF NOT EXISTS FOR (n:WOA) ON EACH [n.name, n.normalized_name];

// Document fulltext search
CREATE FULLTEXT INDEX document_fulltext IF NOT EXISTS FOR (n:Document) ON EACH [n.file_name, n.title];

// Topic fulltext search
CREATE FULLTEXT INDEX topic_fulltext IF NOT EXISTS FOR (n:Topic) ON EACH [n.name];

// Document-Chunk relationship indexes
//CREATE INDEX chunk_doc_relationship_idx IF NOT EXISTS 
//FOR ()-[r:BELONGS_TO]-() ON (r.doc_id);

CREATE INDEX chunk_entity_relationship_idx IF NOT EXISTS 
FOR ()-[r:MENTIONS]-() ON (r.confidence);

// Entity-Topic relationship indexes
CREATE INDEX entity_topic_relationship_idx IF NOT EXISTS 
FOR ()-[r:ABOUT]-() ON (r.relevance);
